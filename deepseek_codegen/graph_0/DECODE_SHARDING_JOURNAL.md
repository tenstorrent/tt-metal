# Decode resharding / collective-fusion experiment (branch: mvasiljevic/deepseek-decode-sharding)

Base = mvasiljevic/deepseek-tm-optimization @ dc2b203 (EST 177.8 ms, 5.96x). Goal: cut the
latency-bound TP collectives (~27% of decode e2e) by fusing/restructuring, keeping I/O + PCC.

## Idea: fuse the per-matmul cross-device reductions
The qkv-down matmul's partial output was reduced 3x SEPARATELY (kv_a all_gather+sum, q_a
reduce_scatter[+all_gather for q_norm in L1], indexer all_gather+sum). Fuse into ONE
all_gather+sum of the full [32,2304], slice locally; q_a re-sharded via local mesh_partition
(L0, feeds TP q_b) or fed full to q_norm (L1). Numerically identical, argmax-gated.

| step | change | attn_0 | attn_1 | EST e2e | PCC | notes |
|------|--------|--------|--------|---------|-----|-------|
| base | (tm-opt HEAD) | 1.089 | 1.124 | 177.8 | argmax 100% | attn CCL 6/layer |
| s1 | fuse matmul_0 reductions (attn L0) | 0.989 | 1.124 | ~172 | ==base | -2 CCL/layer; mesh_partition ~free (1us) |
| s2 | fuse matmul_15 reductions (attn L1; q_a 2-CCL bonus) | 0.989 | **0.963** | **168.8** | ==base | -3 CCL L1; attn/layer 1.107->0.976; **6.28x** |

KEY: collective COUNT reduction works (latency-bound CCL). attn_1 dropped more (-161us) because its
q_a was a 2-CCL all_reduce (reduce_scatter+all_gather) into the q_norm — the fused full reduction
feeds q_norm directly. No #46208 (these are the qkv-down DRAM reductions). mesh_partition (replicated
-> sharded) is a free local op.
Next: same fusion pattern in dense MLP (mlp_0) + MoE shared-FFN/down reductions.

## MoE collective fusion — BLOCKED by moe_compute hang (2026-06-11)
| step | change | result |
|------|--------|--------|
| s3 | MoE shared-FFN matmul_31 reduce_scatter+all_gather -> all_gather+sum (1 CCL) | **HANGS** moe_compute combine, 2x (both at moe_start). PCC clean otherwise. Reverted. |

The same fusion that's a clean win in ATTENTION reliably hangs the moe_compute
selective_reduce_combine when applied in the MoE phase — any L1-layout shift there trips
the #46208-class combine sensitivity (glx_reset_auto recovered each time). So MoE-phase
collective restructuring is blocked until the moe_compute combine L1 reservation is fixed
(upstream). The dense MLP (mlp_0, layer 0, no moe_compute) would be #46208-safe but is only
x3 layers (~low value).

## MoE ROUTER collective fusion — WORKS (combine-safe, pre-dispatch) (2026-06-12)
| step | change | result |
|------|--------|--------|
| s4 | router logits all-reduce: reduce_scatter_9(dim3,ax1)+all_gather_19(dim1,ax1) -> 1 all_gather(dim0,ax1)+local HiFi4 sum | **KEEP**. moe phase 1804.8->1764.8us (-40us). PCC vs HEAD: logits cos64 0.99999860, argmax 100%, non-MoE bit-identical. Commit b3e0f41. |

KEY: unlike s3 (shared-FFN matmul_31, which is CONCURRENT with the combine and hung it), the router
all-reduce sits BEFORE all_to_all_dispatch / moe_compute combine, so fusing it is combine-SAFE. CCL
proof (moe phase): ReduceScatter 4->3 ops (291->229us, -1 CCL = reduce_scatter_9 gone); AllGather 4->4
(177->183us, fused gather moves a bit more); Reduce 1->2 (+7.5us, the new local sum). Net CCL ~-48us.
EST e2e 168.8 -> 166.4 ms (6.28x -> 6.36x); -40us x 58 MoE layers.

HANG NOTE (profiling, not the model): the FIRST tracy-instrumented profile wedged at moe_start (~21min,
combine deadlock — the documented DRAM-pressure #46208 sensitivity, amplified by profiler L1 buffers). But
the EAGER run (main_pcc.py) completed the full graph incl. MoE TWICE with correct logits, and the profile
RETRY (after glx_reset_auto) cleared the MoE phase in ~7s. So the hang was a flaky combine deadlock, NOT a
deterministic property of the change. Discipline: kill -KILL + tt-smi -glx_reset_auto + rerun once cleared it.

ENV NOTE (2026-06-12): runs must execute INSIDE the container `tt-xla-ird-mvasiljevic` (host
/home/ubuntu/mvasiljevic bind-mounts to container /home/mvasiljevic; the venv's uv-python lives at
/home/mvasiljevic/.local and only resolves in-container). Drive via `docker exec tt-xla-ird-mvasiljevic`.

## TILE-ALIGNMENT AUDIT + CONVERGENCE (2026-06-12) — safe positive-EV levers exhausted
Audited the rfuse perf report (perf_reports/rfuse_2026_06_12_05_45_41) for tile (32x32) misalignment.

MATMULS: all tile-aligned. Every contraction/output dim is a multiple of 32 (32,128,512,896,2048,2304,
3072,4608,7168,129280). They show "SLOW" purely because decode M=32 (1 tile-row -> low arithmetic
intensity, DRAM-bound) — INHERENT to decode, not an alignment defect. Math fidelity already tuned
(HiFi2 BF16xBFP8 bulk; HiFi4 FP32 only on router gate + lm_head where top-k / final logits need it).

REAL ALIGNMENT WASTE = heads-per-device = 16 (128 heads / 8-way TP). 16 lands in a TILED dim in a few
attention tensors (reshape_42 [32,1,16,192], slice_86 [32,1,16,128], _sdpa_q [1,32,16,576]) -> padded
16->32 (2x tiles, half padding) + the head-split reshape forces a physical retile (the 52us 2-core
ReshapeView + a 32us one per attn layer). BUT this is STRUCTURAL, not cheaply removable: the per-head
absorbed matmul (matmul_6, b={16}) REQUIRES heads as the batch dim, reached via permute_29 [2,0,1,3];
after that permute the data is [16,32,128] (16=batch, last-2 aligned, NO pad). Any alternative access
(e.g. reshape to aligned [512,192] then back to [32,16,128]) just RE-PADS at a different step — the
padded intermediate is intrinsic to going from token-major [32,3072] to head-batch [16,32,128].
And permute_29/38 is the exact op whose L1-layout shift DETERMINISTICALLY trips the #46208 combine hang
(the a7 revert). So the head relayout is zero/negative-EV AND high-risk. NOT pursued.

OTHER LEVERS RULED OUT: (a) dense-MLP all-reduce fusion (reduce_scatter_3/4 + all_gather_10/11) — same
pattern as the router but on [128,4608] (18x wider): the fused all_gather(dim0) would move ~(N-1)=7x the
data vs the decomposed ~1.75x -> bandwidth-bound NET LOSS (router won only because [32,256] is tiny /
latency-bound), and dense is only x3 layers (~0.4ms ceiling). (b) MoE expert/combine collectives (59%
bulk) — blocked by combine L1 fragility. (c) lm_head ArgMax/matmul (445us+1291us) — runs x1, <1% e2e.
(d) RoPE cos/sin repeats (42us/layer) — per-batch broadcast is required (correct as-is).

CONVERGED: every clean collective fusion is landed (qkv x2 attn, router); remaining attention TMs are
structurally necessary MLA format conversions (head split, RoPE interleave, SDPA concat) — prior round
already eliminated the foldable ones. STOP per skill criterion (no patch moves the number w/o breaking
PCC or hitting the blocked combine).

## BRANCH RESULT: attention + router collective fusion = 177.8 -> 166.4 ms (6.36x), PCC bit-identical
Net deliverable: (1) fuse the qkv-down 3-way reduction into 1 all_gather+sum (both attn layers),
attn/layer 1.107 -> 0.976 (-131us); (2) fuse the MoE router all-reduce into 1 all_gather+sum (-40us/MoE
layer). The MoE expert/combine collectives (the 59% bulk) remain blocked by the combine L1 fragility.
