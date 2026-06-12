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

## STAT-GATHER TILE REPACK — BIG WIN (2026-06-12, credit: user "CCL sends full tiles" insight)
| step | change | result |
|------|--------|--------|
| s5 | RMS-norm partial-stat all_gathers: gather [1,32,1] dim=0 -> [8,32,1] (8 padded tiles) REPLACED by gather [32,1] dim=1 -> [32,8] (1 tile) + sum([1]). All 5 sites (all_gather_2/8/12/18/31). | **KEEP**. Commit fb2ac9d. PCC vs HEAD: logits cos64 0.99979782, argmax 100%, non-MoE bit-identical. |

ROOT CAUSE: CCL transmits FULL 32x32 TILES. The TP RMS-norm reduces a per-token partial stat
([32,1], one scalar/token) across the 8 hidden-shard devices. Gathering [1,32,1] over dim=0 makes the
output [8,32,1] = 8 SEPARATE tiles (each [32,1] padded to a full 32-wide tile = 1024 elems for 32 real)
-> the gather moved 8 padded tiles. Gathering [32,1] over dim=1 packs the 8 partials as columns [32,8]
= 1 tile. Numerically identical sum (fp reduction-order only -> cos 0.9998, argmax 100%).

MEASURED (clean, op-level; AllGather device time per phase, summary.csv): stat gather 41-47us -> ~2us,
AllGather total dropped -44 to -47us in EVERY phase (attn0 167->120, dense 205->160, attn1 126->82,
moe 199->154, lmhead 307->263). ReduceScatter UNCHANGED (control). The phase-TOTALS were noise-dominated
(+110us/phase global slowdown hit untouched lmhead/prologue too) so the keep rests on the deterministic
AllGather component delta, not the noisy totals.

EST e2e: -45us per stat gather x ~123 gathers (61 layers x 2 norms [input + post-attn] + lm_head) =
~-5.5ms. 166.4 -> ~160.9 ms (6.36x -> ~6.58x). This is the bandwidth-recovery the earlier
"convergence" missed: the stat gathers WERE sending mostly tile-padding, and dim=1 packing recovers it.

WHY the gathers can't be eliminated entirely (user Q): hidden is TP-sharded (axis1, 8x896) and matmuls
are row-parallel (contract the 896 shard -> reduce), so activations stay hidden-sharded and norms need
the full-hidden stat -> an axis1 reduction. Gathering the scalar stat (not the [.,7168] activation) is
already bandwidth-minimal; the dim=1 repack removes the remaining tile-padding waste. Eliminating the
gather entirely needs a fundamental re-shard (replicate activations / column-parallel) that relocates
cost to activation gathers/reshards — high risk on generated code, not pursued.

## ROPE COS/SIN CSE — cross-layer redundancy (2026-06-12, "bigger patterns")
| step | change | result |
|------|--------|--------|
| s6 | all 6 RoPE sites (both layers) repeat the SAME _rope_cos/sin_pos_4d to 2 shapes ([1,1,32,1] x4, [1,1,512,1] x2). Compute the 4 distinct broadcast tensors ONCE, alias every site, free at rope5 (before moe_start). | **KEEP**. Commit 9c666aa. Repeat 14 ops/85us -> 6 ops/40us; **attn1 Repeat 41us -> 0** (layer 1 reuses layer 0's tensors). PCC bit-identical (cos64 unchanged, argmax 100%). |

PATTERN: the cos/sin tables depend only on the decode POSITION (same for all layers in one
step), so the per-site repeats are loop-invariant. The codegen emitted a fresh repeat at each of
the 6 sites. CSE -> 4 shared. rotary_embedding_llama only READS cos/sin (deallocated after) so
sharing is safe. EST: each of 61 attn layers recomputes ~6 rope repeats (~41us); shared-once ->
layers 1-60 reuse -> ~-41us x 60 = ~-2.4ms. ~161 -> ~158.5 ms (~6.68x). Combine-safe (freed
before moe_start). Verified via the deterministic Repeat op-count/time delta, not the noisy totals.

## TILE-ALIGNMENT AUDIT (2026-06-12) — matmuls aligned; head-dim padding structural
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

Remaining attention TMs are structurally necessary MLA format conversions (head split, RoPE interleave,
SDPA concat) — prior round eliminated the foldable ones. (This "audit" first concluded CONVERGED, but the
user's "CCL sends full tiles" prompt then surfaced the stat-gather tile-padding win above — s5.)

## BRANCH RESULT: collective fusion + tile repack + rope CSE = 177.8 -> ~158.5 ms (~6.68x), PCC argmax-100%
Net deliverable: (1) fuse the qkv-down 3-way reduction into 1 all_gather+sum (both attn layers),
attn/layer 1.107 -> 0.976 (-131us); (2) fuse the MoE router all-reduce into 1 all_gather+sum (-40us/MoE
layer); (3) repack the 5 RMS-norm stat gathers dim=0->dim=1 (8 padded tiles -> 1 tile), -45us per stat
gather x ~123 gathers (~-5.5ms); (4) CSE the loop-invariant RoPE cos/sin repeats (~-2.4ms). The MoE
expert/combine collectives (the 59% bulk) remain blocked by the combine L1 fragility.
THEME: the biggest wins all came from recognizing PATTERNS repeated across the graph (per-matmul
reductions, full-tile CCL padding, loop-invariant repeats) rather than single-op knobs.

## INDEXER K_NORM BF16 — drop fp32 round-trip (2026-06-12, E_idxnorm)
| step | change | result |
|------|--------|--------|
| E_idxnorm | indexer k_norm (both layers): slice(bf16)->typecast(fp32)->layer_norm->typecast(bf16) -> feed bf16 directly, layer_norm out bf16. Removes 4 typecasts/decode-step. | **KEEP**. Commit 55d3fe8. PCC vs ropecse golden: full-graph live-outs **BIT-IDENTICAL** (argmax 100%, logits to_layout_267 exact). |

RATIONALE: layer_norm accumulates fp32 internally regardless of input dtype; the indexer is a top-k
SELECTION signal (picks which KV to attend) so bf16 input is tolerable. Bit-identical proves the
selection is unchanged by bf16 -> downstream attention identical.

MEASURED (idxnorm_2026_06_12_09_49_16 vs ropecse, 2-layer graph): Typecast 23->19 ops, 57.3->48.2us
(-4 ops, -9.15us exactly as designed). attn0 6->4 typecasts (-4.6us), attn1 3->1 (-4.4us); attn0 phase
-8.6us, attn1 -16.5us. Full/lmhead totals were noise-dominated (+27.6/+63.8us on UNTOUCHED ops, ArgMax
alone +75us run-to-run) -> keep rests on the deterministic typecast op-delta + bit-identical PCC, per
the established noisy-totals methodology. EST e2e: ~-4.5us x 61 attn layers ~= -0.27ms (small, safe).

NOTE: profiler windowing needs the full path to tt-perf-report (python_env/bin); run_profile.sh's bare
`tt-perf-report` isn't on the docker login-shell PATH. Re-window the existing CSV if windows fail.

## MoE COMBINE FRAGILITY FIXED → shared-FFN CCL fusion UNBLOCKED (2026-06-12, BIG WIN)
| step | change | result |
|------|--------|--------|
| mux | Vendor PR #46544 (e4bc86b) dynamic mux-buffer sizing into selective_reduce_combine_program_factory; supersedes local num_buffers=13 hack | commit 17e504e. Rebuilt ttnncpp. get_fabric_mux_config recursively decrements buffers (from 15) until the mux L1 region fits below the lowest occupied compute L1 tensor. |
| E_moe_ccl_fuse | shared-FFN w1 (matmul_31) + w3 (matmul_32) all-reduce: reduce_scatter(dim3,ax1)+all_gather(dim1,ax1) -> all_gather(dim0,ax1)+local bf16 sum (FastReduceNC). Both layers. | **KEEP**. Commit 2c84eea. No hang (was deterministic #46208/s3 hang). |

KEY: the s3 fusion that DETERMINISTICALLY hung the moe_compute combine (06-11, even with the
vendored #45764 writer-semaphore fix + static num_buffers=13) now runs CLEAN once #46544's dynamic
mux sizing gives the combine adaptive L1 headroom under the fused layout's L1 shift. So the combine
"L1 fragility" that blocked ALL MoE-phase collective restructuring is RESOLVED by #46544.

MEASURED (muxs3b_2026_06_12_10_50_04 vs ropecse, 2-layer = 1 MoE layer):
  ReduceScatter 228.6us/3 -> 84.4us/1   (-144us, -2 ops: both shared-FFN RS gone)
  AllGather     139.9us/3 -> 177.4us/3  (+37.5us: 2 dim0 gathers replace 2 small dim1 gathers)
  FastReduceNC   13.2us/1 -> 22.6us/3   (+9.4us: 2 new bf16 dim0 local sums on the fast path)
  MoE phase total 1889.7 -> 1802.6      = -87.1us / MoE layer  (full 2L -99us)
EST e2e: -87.1us x 58 MoE layers ~= -5.0ms  (~158.5 -> ~153.5 ms).

CORRECTNESS: argmax 100% bit-identical to ropecse; HEAD-rel logits cos64 0.999996; ABSOLUTE golden
PCC = 0.8989 (UNCHANGED from the ropecse 0.8989 bf4 floor -> the fp reduction-order change is
numerically free on top of the bf4 quantization). Verified vs ../golden_logits.pt.

MYTH BUSTED: the earlier "fusing a wide all-reduce is bandwidth net-loss" projection (used to rule out
dense-MLP + shared-FFN fusion) is WRONG when measured: the HiFi4-fp32 reduce_scatter (compute+comm)
costs ~80us, far more than the extra dim0-gather bytes (~19us) + a cheap FastReduceNC local sum (~5us).
REOPENS: (a) the 1 remaining MoE-phase reduce_scatter (84us); (b) dense-MLP reduce_scatter_3/4 +
all_gather_10/11 (x3 layers); (c) broader MoE-phase L1-sharding (skill's last untapped lever) - all now
combine-safe under #46544.

## DENSE-MLP all-reduce fusion — FAILED (hangs, 2026-06-12, E_dense_ccl_fuse)
| step | change | result |
|------|--------|--------|
| E_dense_ccl_fuse | dense MLP w1/w3 (matmul_12/13, [1,1,128,4608]): reduce_scatter(dim3,ax1)+all_gather(dim1,ax1) -> all_gather(dim0,ax1)+local sum, same fusion as the MoE shared-FFN | **FAILED — HANGS reproducibly** (2 runs, both wedged the device with no output; glx_reset_auto recovered each). Reverted (uncommitted). |

The shared-FFN fusion ([32,2048]) works + wins, but the IDENTICAL fusion on the WIDER dense tensor
([128,4608], 9x elements) hangs deterministically. The all_gather(dim0) produces [8,1,128,4608]
(~9.4MB bf16) -> likely a CCL buffer/deadlock at this size (NOT the moe_compute combine -- dense layer
has no moe_compute; the hang is the dim0 gather itself or its DRAM/L1 footprint). So the fusion pattern
does NOT generalize to wide tensors. Bound: narrow reductions (shared-FFN [32,2048]) fuse safely;
wide ones (dense [128,4608], lm_head [32,129280]) do not. Dense was only x3 layers (~0.1ms ceiling)
so low loss. Decomposed reduce_scatter+all_gather (bandwidth-optimal all-reduce) stays for dense.

## ROPE x-perm -> transpose — NO-OP (2026-06-12, E_rope_transpose, reverted)
Converted the rank-4 rope x-perm [0,1,3,2] -> ttnn.transpose(2,3) (both layers). PCC clean
(argmax 100%, golden 0.8989). But MEASURED no-op: total Permute+Transpose TM time flat (attn0 0.0us,
attn1 -0.5us) -- the rope x-perm is a tiny [32,1,2,32] tensor, transpose==permute for it (the
"~10x faster" only applies to LARGE transposing permutes). Reverted per skill (number didn't drop).
Also found: ttnn.transpose has NO rank-5 path (only unsqueezes rank<4) -> the 5D rope out-perm
[0,1,2,4,3] CRASHES if converted. The real attention TM lever is the head-relayout permute_29/38
([2,0,1,3] 3-cycle, the #46208 trigger -- now combine-safe under #46544 but needs a restructure,
not a simple transpose). NOTE: device hit a transient ethernet-core-26-25 init fault mid-session
(after repeated resets); glx_reset_auto recovered it. Crashes during that window were device, not code.

## INDEXER ROPE -> rotary_embedding_hf — ATTENTION TM WIN (2026-06-12, E_idxrope_hf)
| step | change | result |
|------|--------|--------|
| E_idxrope_hf | indexer-K RoPE (both layers): rotary_embedding_llama (interleaved-pair, needs reshape->permute->reshape before+after) -> rotary_embedding_hf (native rotate_half on half-concat, fed [1,1,32,64] directly). Added HF concat-doubled cos/sin const tables + one-time seq=32 pipeline. | **KEEP**. Commit (E_idxrope_hf). PCC vs ropecse golden: argmax 100% bit-identical, logits cos64 0.999996, absolute golden 0.8989 (floor held). |

ROOT CAUSE: rotary_embedding_llama's kernel does tile-local ADJACENT-PAIR rotation (cos/sin
repeat_interleave-doubled, 32x32 trans_mat) -> the model's HALF-CONCAT rope layout had to be permuted
to interleaved-pair (and back). Those reshapes ([32,1,64]->[32,1,2,32] etc) CROSS TILE BOUNDARIES ->
expensive physical retiles. rotary_embedding_hf does native rotate_half (cat((-x2,x1)), midpoint on the
32-tile boundary) -> takes half-concat directly, no permute, no tile-crossing reshape.

MEASURED (idxrope_2026_06_12_12_22_02 vs ropecse): ReshapeView attn0 194->143us (-52us,-2ops),
attn1 253->199us (-54us,-2ops); Permute -1 op/layer; UntilizeWithUnpadding flat (savings are real).
attn1 phase -59.6us (clean layer); attn0 -9.3us (hosts the one-time HF cos/sin setup). full 2L -176us.
EST e2e: ~-53us deterministic ReshapeView x 61 attn layers ~= -3.0ms (minus ~one-time HF setup).

GENERALIZES: convention-matched ops avoid format-bridging TM. rotary_embedding_hf (rotate_half) fits
half-concat models; rotary_embedding_llama (adjacent-pair) needs the interleave permute. The main Q/K
ropes (rope1/2/4/5) already feed interleaved-pair (no permute) so were left on rotary_embedding_llama.

## ATTENTION TM — FLOOR ASSESSMENT after E_idxrope_hf (2026-06-12)
After the indexer-rope HF swap, audited the remaining attn1 TM/CCL (idxrope profile). Conclusion:
remaining attention cost is STRUCTURAL with the current op set — no clean lever left:
- ReshapeView 199us: dominated by the head-split retiles (16 heads in a tiled dim -> 16->32 pad;
  token-major [32,..] -> head-batch [16,32,128] forces a physical retile). Intrinsic to MLA's
  per-head absorbed matmul (tile-alignment audit, prior). Rank-conversion reshapes are cheap views.
- permute_29/38 (head-relayout [2,0,1,3], 33us/layer ~= 2ms x61): the ONLY sizable remaining lever,
  but structural — the per-head absorbed matmul REQUIRES heads as batch-dim-0; [2,0,1,3] is a 3-cycle
  (= 2 transposes, MORE ops, not fewer). Now combine-safe under #46544 but needs an upstream
  layout change (produce head-major earlier) which just relocates the retile. Not pursued (high
  risk / structural, the a7 lesson).
- reduce_scatter_8 (80us): GENUINE reduce_scatter — output stays hidden-sharded, feeds the residual
  add_22 directly (no following all_gather). NOT the fusable all-reduce pattern. AllBroadcast (78us)
  is op-internal (no explicit call in _main). AllGather (66us) is the qkv/stat gather, already minimal.
- Main Q/K ropes (rope1/2/4/5): already interleaved-pair (no permute) -> left on rotary_embedding_llama.
RULED OUT this round: nlp_create_qkv_heads_decode (emits [batch,heads,1,hd], not absorbed [16,32,128]),
a3 SDPA-L1 (compute-bound no-op), rope x-perm->transpose (no-op, tiny tensor). Attention is at its
practical floor; further gains need a different MLA head-layout strategy (upstream/codegen), not op swaps.

## MoE FLOOR ASSESSMENT (2026-06-12, after shared-FFN fusion)
Investigated the MoE phase for further levers (the "L1-sharding" / knob-tuning candidates). Conclusion:
MoE is at its practical floor with the current op set + bf4 precision.
- **MoECompute 690us (38% of MoE, biggest single op)**: tried MOE_NUM_LINKS=2 (env, now stable post
  #45764/#46544) -> MoECompute 690.5->695.8us (FLAT, noise); MoE phase 1802.6->1784.2 (within noise).
  So the combine is NOT link-bound -> MoECompute is dominated by the **bf4 expert-weight DRAM reads**
  (9 experts x 7168x2048x3 @ bf4 per token) = fundamental decode cost, already at the lowest precision.
  Other knobs (bh_ring, ohsd) are combine-side too -> won't help. Not pursued further.
- **AllGather 192us / AllBroadcast 147us / AllToAll 81us**: AllBroadcast is INTERNAL to composite_all_gather
  + composite_all_to_all (composite_common.cpp:344/431) -- i.e. how the dispatch gather + stat gathers are
  implemented. Structural to MoE expert parallelism; CCL count already minimized (router/shared-FFN fusions).
- **ReduceScatter 84us / tail**: already optimized (BF16 weighted-k-sum -> FastReduceNC, reduce_scatter
  fuses all_reduce+mesh_partition). Collar Tilize/Untilize = dispatch ROW_MAJOR conversions (structural).
NET: after the shared-FFN CCL fusion (-5ms) the MoE has no remaining op-level lever. Further MoE gains need
architecture/codegen changes (precision below bf4, fewer active experts, or a different dispatch/combine kernel).
