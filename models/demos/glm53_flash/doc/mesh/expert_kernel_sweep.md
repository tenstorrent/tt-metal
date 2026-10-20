# Routed-expert kernel efficiency at the 8x4 per-chip shapes (2026-10-20)

Task: on the 8x4 Galaxy the hybrid placement gives every chip W13 `[1, 72, 4096, 512]` and W2 `[1, 72, 256, 4096]`
(bfp8, 240 MB per layer) and `tt/moe.py` streamed them with `ttnn.sparse_matmul` at ~26 % of the 512 GB/s DRAM
peak: the 8x4 profile (`_logs/mesh/report_perf8x4_l4_b{4,32}`) shows the gate|up call at 733 us and the down call
at 130 us per 32-token block, plus 92 us of output zero-fill and 55 us of expert-sum per block, x8 blocks at
32 users/row. This note is the single-chip sweep behind the fix: `tests/test_expert_kernel_perf.py` (device times
from the Tracy device profiler, 4 calls per point, every point checked against torch or the baseline output),
raw tables in `expert_kernel_sweep_tables.md`, logs `_logs/optexperts_*.log`, report CSVs
`_logs/optexperts_report_*.csv`. Chip 2, device grid 12x10 (120 cores). All bytes are bfp8 tile bytes
(1088 B per 32x32 tile); "% peak" is of 512 GB/s.

## 1. How the kernel behaves (why 26 %)

`sparse_matmul` only has the 1D `mcast_in0` factory: the cores are output-width blocks (`cores = Nt / per_core_N`,
a rectangle that must divide Nt), the activation block is read by ONE sender core and multicast, each core streams
its own `[in0_block_w x per_core_N]` tile block of the weight per K block and barriers, and every K block costs a
memory latency plus a semaphore round trip (`reader_bmm_tile_layout_in{0,1}_sender_*`). Consequences measured on
the per-chip shapes:

| finding | evidence (32 active of 72, M = 32) |
|---|---|
| N = 512 is capped at 16 cores (Nt = 16) and the default 4-tile K blocks make it latency-bound: 32 experts x 32 K blocks x ~1.06 us | 1092 us at 16 cores, 1083 us at 8 cores (same time, half the cores) |
| bigger K blocks remove the per-block round trips | in0_block_w 4 / 16 / 32 / 64 / 128: 1092 / 501 / 408 / **369** / 375 us (13 -> 38 % of peak) |
| grid orientation matters at the same core count | 16 cores as (4,4) 375 us, (8,2) 377 us, (2,8) 463 us |
| beyond 64 tiles the 16-core shape is bound by the per-core read rate (~14 GB/s per core with 1 KB tile reads) and the sender's 256 KB in0 read + multicast per expert | x in L1 352 us, x as bfp8 318 us, both 315 us (not exact: measured for the model only) |
| with >= 32 cores the kernel tops out at 54 % (1 tile per core) to 66 % (4 tiles per core) | N = 1024: 518 us (32 cores, ibw 16); N = 2048: 1034 us (64 cores, ibw 16); N = 4096: 1694 us on 32 cores x 4 tiles vs 1948 us on 64 cores x 2 |
| the per-slot validity scan costs ~1 us per inactive expert; the indexed mode also compacts the output | N = 512, 8 active: 128 us -> nnz exact 86 us -> indices 78 us; 32 active: 369 -> 331 -> 302 us (neither usable in a trace: both need the active count on the host) |
| W2 (K = 256, Nt = 128) is per-expert-overhead bound: ~4.4 us per active expert + ~30 us fixed | 8 / 32 / 72 active: 67 / 144 / 319 us at the best config (32 cores x 4 tiles, ibw 8, subblock 4) vs 96 / 170 / 349 us today |
| every sparse_matmul is preceded by a `zeros_like` of its expanded output (`create_output_tensors`), unavoidable without an exact `nnz` | 12.6 us for the [72,32,512] gate|up slab, 93 us for the [72,32,4096] down slab, 735 us for [72,256,4096] |
| one call over M gathered tokens is NOT an option for this kernel: the sender re-reads and re-multicasts the whole `[M, 4096]` activation for every active expert (2 MB at M = 256) | N = 512, 72 active: M = 32 / 64 / 128 / 256 -> 761 / 1365 / 2464 / 4720 us (41 -> 6.6 % of peak), indices mode 4811 us, x in L1 4871 us |
| the 2D mcast config and more cores than Nt are not available | the device op accepts only the 1D config ("Only mcast_in0 is supported for sparse matmul"); Nt = 128 has no rectangle on 12x10 beyond 64 cores |

## 2. Best sparse configuration per 8x4 shape (M = 32, 32 active; `sparse_matmul_config` now derives these)

| shape | today (`moe.py` before) | best measured | rule |
|---|---|---|---|
| W13 N = 512, K = 4096 (hybrid) | 16 cores (2,8), ibw 4: 1092 us, 13 % | 16 cores (4,4), ibw 64, sw 1: **369 us, 38 %** (3.0x) | <= 16 cores: 64-tile K blocks |
| W13 N = 1024, K = 4096 (EP8xTP4) | 32 cores, ibw 4: 1102 us, 25 % | 32 cores, ibw 16: **518 us, 54 %** | >= 32 cores: 16-tile blocks |
| W13 N = 2048 (2x4 block) | 64 cores, ibw 4: 1342 us, 41 % | 32 cores, ibw 16, sw 2: 1027 us / 64 cores ibw 16: **1034 us, 54 %** | |
| W13 N = 4096 (EP32, single chip) | 64 cores, ibw 4: 1948 us, 57 % | 32 cores x 4 tiles, ibw 4: **1694 us, 66 %** | Nt >= 128: 4 tiles per core |
| W2 I = 256 (hybrid) | 64 cores, ibw 4: 171 us, 41 % | 32 cores x 4, ibw 8, sw 4: **144 us, 48 %** | whole K when <= 64 tiles per core |
| W2 I = 512 (EP8xTP4) | 64 cores: 296 us | 32 cores, ibw 16, sw 2: **254 us, 55 %** | |
| W2 I = 1024 (2x4 block) | 64 cores: 530 us | 32 cores, ibw 32, sw 4: **472 us, 59 %** | |
| W2 I = 2048 (EP32, single chip) | 64 cores: 1009 us | 32 cores, ibw 4, sw 1: **882 us, 63 %** | |

`GLM53_SPARSE_CONFIG=legacy` restores the old rule for A/B runs.

## 3. Tokens per chip and the placement alternatives (sparse kernel, best config per point, us per layer per chip)

Active experts from a Monte-Carlo of uniform top-8 routing (max over the chips, the step waits for the slowest
one); M = all gathered tokens in one call; per layer = W13 + W2 + their output fills; the collectives
(50-300 us per layer measured on the mesh) are the same for every placement here. The expert-grouped hybrid
(`g` experts' slices packed side by side so N = 512 g) is the hybrid placement with the EP32 kernel shapes.

| tokens per chip | hybrid EP4xTP8 (72 x [4096,512]) | hybrid g=2 (36 x [4096,1024]) | hybrid g=4 (18 x [4096,2048]) | hybrid g=8 (9 x [4096,4096]) | EP8xTP4 (36 x [4096,1024]) | EP32 (9 x [4096,4096]) |
|---|---|---|---|---|---|---|
| 8 (1 user/row) | 18 active: **434** | 16: 443 | 13: 644 | 9: 792 | 11: **329** | 5: 453 |
| 32 (4 users/row) | 47: 850 | 32: 816 | 18: 884 | 9: 797 | 26: **672** | 8: 708 |
| 64 (8 users/row) | 63: 1795 | 36: 1201 | 18: 956 | 9: **871** | 33: 1106 | 9: 877 |
| 256 (32 users/row) | 72: 6629 (one call) / ~9.5 ms (8 blocks) | 36: 3627 | 18: 2127 | 9: **1511** | 36: 3634 | 9: 1511 |

Reading: every placement holds the same 240 MB per chip, so at 256 tokens the only question is how many times
the kernel re-reads the activation per expert (hybrid: 72 experts x 2 MB) or the weights (8 blocks); the full-width
shapes (EP32 / g=8, 9 experts x [4096, 4096]) are the best the SPARSE kernel can do at 64-256 tokens (1.5 ms) and
within 10 % of the best at 32; EP8xTP4 is the best small-batch shape. None reaches the 0.67 ms budget.

## 4. The dense layout: one matmul per projection over all 72 local experts

`TtRoutedExperts(dense=True)`: the chip's slices side by side in ONE W13 `[4096, 72 x 512]` (all gates, then
all ups) and ONE W2 `[72 x 256, 4096]`; the expert sum is the W2 K reduction, the routing weights are applied to
`h [M, 72 x 256]` through a `[M, 72] x [72, 72 x 256]` 0/1 expansion matmul. Every local expert is read exactly
once per step for any M, there is no expert slab, no zero-fill, no mask and no reduce op, at the price of reading
the inactive experts too (at >= 32 gathered tokens 47-72 of the 72 are active anyway). Measured (run 4,
`dense_matmul_config` carries the best configs):

| M | W13 [M,4096] x [4096,36864] (158 MB) | W2 [M,18432] x [18432,4096] (80 MB) | both | sparse kernel at the same point (section 3, hybrid) |
|---|---|---|---|---|
| 32 | 421 us, 74 % (96 cores, ibw 4) | 240 us, 65 % (32 cores x 4, ibw 8) | **662 us** | 850 us (+55 us expert sum) |
| 64 | 440 us, 71 % | 240 us, 65 % | **680 us** | 1795 us |
| 128 | 494 us, 63 % | 259 us, 60 % | **753 us** | - |
| 256 | 619 us, 51 % (120 cores, ibw 8) / 631 us (96, ibw 4) | 335 us, 47 % (64 cores, ibw 16) | **954 us** | 6629 us one call / ~9.5 ms as 8 blocks; EP32 1511 us |

The dense W13 at 74 % is the highest DRAM efficiency of anything measured (interleaved 1 KB tile reads on 96-120
cores); the transposed tall-skinny form (W13^T as in0 split over 72-96 cores, x^T multicast) reached only 42 %
(744 us at M = 32, 906 us at M = 256) and the batched bmm for W2^T is rejected for these shapes.

## 5. Recommendation

1. **Decode on the 8x4 with >= 8 users/row: the dense layout, hybrid placement unchanged** (`GLM53_DENSE_EXPERTS=1`
   / `TtMoE(dense_experts=True)`): 1.48 ms per layer per chip for the whole expert block at 256 gathered tokens
   in one pass (matmuls 0.95 ms), against 9.7 ms for today's 8 sparse blocks on the same real tokens (section 6)
   and ~8.5 ms of expert ops in the 8x4 profile. The collectives stay the row gather / reduce-scatter / column
   all-reduce of the hybrid. The remaining gap to the 0.67 ms budget is the dense W13 at 51 % (M = 256) and the
   ~0.5 ms of elementwise ops on the `[256, 36864]` gate|up.
2. **<= 4 users/row (32 gathered tokens or fewer): the sparse path with the new config** (`sparse_matmul_config`,
   the default): 705 us per block on real tokens against 1358 us today (3x on the N = 512 gate|up call itself,
   1.2x on the down call, section 2); the dense pass is within 6 % there (750 us) and wins from ~64 tokens on.
3. **Do not move the placement for the kernel's sake**: with the dense layout the per-chip matmul shapes are the
   same for every placement (same 240 MB), so EP32 / EP8xTP4 would only change the collectives and the
   small-batch imbalance (the design's section 5.4 argument), not the expert time.
4. Kernel-level follow-ups (tt-metal, not done here): in0 reuse across the batch in the sparse factory
   (`in0_reuse_in_dfb` exists in the sender kernel but the receivers lack the replay; it would remove the per-expert
   activation re-read + multicast that caps N = 512 at 38 %), an accumulate-over-batch mode for the down projection
   (today's expanded slab + zero-fill + reduce), and a runtime active count for the indexed mode.

## 6. Same-chip A/B on real layer-4 weights

`tests/test_moe_block.py`: chip (0,0)'s slice (column 0's 72 experts x row 0's 256-wide intermediate), real tokens
of the T4097 dump, the real router's weights, each path captured as a trace and replayed 20x (dispatch-free device
time of the whole composite: fills, slices, activation, routing-weight multiply, expert sum).

| M (tokens) | active of 72 (real routing) | path | us per call | us per 32-token block | pcc vs torch | pcc vs the current block |
|---|---|---|---|---|---|---|
| 32 (4 users/row) | 28 | sparse, legacy config (today) | 1357.8 | 1357.8 | 0.999916 | 1 |
| 32 | 28 | sparse, new config | **705.3** | 705.3 | 0.999916 | 1.000000 |
| 32 | 28 | dense | 749.9 | 749.9 (one pass) | 0.999916 | 0.999983 |
| 256 (32 users/row) | 31 | sparse, legacy config (today) | 9721.5 | 1215.2 x 8 | 0.999904 | 1 |
| 256 | 31 | sparse, new config | 5216.7 | 652.1 x 8 | 0.999904 | 1.000000 |
| 256 | 31 | dense | **1476.3** | 1476.3 (one pass) | 0.999901 | 0.999958 |

(`_logs/optexperts_moe_block.log`.) Real routing is much more concentrated than the uniform Monte-Carlo of
section 3 (28 of 72 local experts active for 32 real tokens, 31 for 256, against 47 / 72 uniform), which is why
the sparse path at 4 users/row (705 us) edges the dense one (750 us: its two matmuls are 662 us, the rest is the
activation composite on [32, 36864]); at 32 users/row the dense pass is 3.5x the tuned sparse blocks and 6.6x
today's. The dense composite at M = 256 spends ~520 us outside the matmuls (slices, clamps, silu, the two
multiplies on [256, 36864] / [256, 18432]): the next target.

`tests/test_perf_layer.py` layer 4 traced decode B=1 on the single-chip layout (288 experts x [4096, 4096]
gate|up, [2048, 4096] down, 8 active): legacy config 8.23 ms/step -> new config **7.96 ms/step** (20 traced steps,
`_logs/optexperts_perf_layer4_{legacy,tuned}.log`; the single-chip shapes only move from 64 to 32 cores x 4 tiles).

`tests/test_moe.py` (single chip, sparse path, new config): router sets 32/32 and 255/256 identical to HF, MoE PCC 0.999930 (T = 32)
/ 0.999946 (T = 256) (`_logs/optexperts_test_moe.log`).

## 7. Minimum op count of the expert block (2026-10-20, after the merge into `arg/glm53_mesh`)

Rule applied: every data-movement op merged into a single op or removed before any compute tuning. Op counts
are device operations of `TtRoutedExperts.forward` (from the router's dense matrix to the TP partial) from a
graph capture with the fast runtime mode off, so the ops hidden inside `sparse_matmul` (its output zero-fill)
and inside `ttnn.sum` (a pad fill for the non-tile-aligned 72 experts) count too; times are the traced replay of
the whole block on real layer-4 tokens and routing (`tests/test_moe_block.py`, `_logs/optexperts_moe_block_{before,after}.log`,
outputs compared against the saved pre-rewrite outputs `_logs/optexperts_block_ref.pt`).

| path | device ops before -> after (per 32-token block) | what changed | M=32 us before -> after | M=256 us before -> after | PCC vs the pre-rewrite block (M=32 / 256) |
|---|---|---|---|---|---|
| dense (`_block_dense`) | 11 -> **6** | gate and up as two weight tensors (two matmuls, no slices); `clamped_silu_glu` for clamp+clamp+silu+mul; the expert selector and the per-column expansion folded into one 0/1 constant `sel_expand` [E, E_loc x I_loc] (one matmul from the router's bf16 matrix, no typecast); the routing-weight multiply; the W2 matmul | 747.1 -> **721.4** | 1474.7 -> **1230.5** (one pass) | 0.999986 / 0.999964 |
| sparse (`_block`) | 18 (+2 input slices per block at M=256) -> **13** | mask: `sum` + `to_layout` on the router's bf16 matrix (no typecast); `clamped_silu_glu`; the [1,1,M,E] -> [1,E,M,1] routing weights as one `permute` (was typecast + transpose + reshape) | 699.9 -> **683.1** (legacy config 1355.8 -> 1338.8) | 5193.0 -> 5061.9 (8 blocks) | 1.000000 / 1.000000 |

Remaining ops of the dense path: `matmul` (routing weights x `sel_expand`), `matmul` (gate), `matmul` (up),
`clamped_silu_glu`, `multiply` (routing weights; it is what zeroes the non-routed experts, so it cannot fold into
the GLU without a per-element scale input), `matmul` (W2, expert sum = K reduction). Remaining ops of the sparse
path: pad-fill + `sum` + untilize (the ROW_MAJOR bf16 activity mask the sparse kernel requires: no single op
produces it), `sparse_matmul` + its zero-fill, two slices (the kernel's one `[E, M, 2I]` output; `clamped_silu_glu`
needs the halves as tensors), `clamped_silu_glu`, `permute`, `multiply`, `sparse_matmul` + its zero-fill,
`sum` over the experts (the accumulate-over-experts mode is a kernel change). On a multi-column mesh both paths
have one more matmul (the expert selector) unless the router emits the local columns directly.
`test_moe.py` (single chip, sparse path, bf16 router matrix): router sets 32/32 and 255/256 identical to HF, MoE PCC 0.999930 (T = 32) / 0.999946 (T = 256), unchanged (`_logs/optexperts_test_moe_minops.log`).

## 8. Sparse matmul kernel changes (2026-10-20, own tt-metal build on the worktree) -- EVIDENCE ONLY, NOT ADOPTED

Superseded by the user rule of 2026-10-20 14:50 (`_agents/COMMON.md`: model-local generic ops only, no edits under
`ttnn/cpp`): the factory / kernel changes below were measured and then dropped from the branch (nothing under
`ttnn/cpp` is committed); the numbers stay as the evidence behind the custom op of section 9 (contiguous per-bank
column streams at ~70 % of DRAM peak, the expert sum as a K reduction, no slabs / fills).

Budget: at 32 users (4/row) the design gives the experts 0.41 ms per layer per chip; the sparse block stood at
683 us (gate|up 369 us at 38 % of DRAM peak on 16 cores, down 144 us + 94 us slab zero-fill + 55 us expert sum).
Three changes to `sparse_matmul` (factory `sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp`, the shared
reader/writer/compute kernels behind `#ifdef`s, so the dense matmul factories are untouched; the regression suite
`test_sparse_matmul.py` + `_indexed.py` passes 88/88):

| change | what it does | how it is selected |
|---|---|---|
| batch groups | a grid of `G x (Nt / per_core_N)` cores runs `G` identical 1D-mcast pipelines side by side (own sender, multicast rectangle and receivers per group); group `g` streams the sparse slots `g, g+G, ...` (the validity scan stays per slot) | a program-config grid that is a multiple of the output-width blocks and tiles into rectangles (`nnz=None`, no `indices`); `moe.py::sparse_matmul_config(groups=)` / `group_grid` |
| in0 reuse | the in0 buffer holds the whole broadcast activation (`per_core_M x Kt` tiles); the sender reads + multicasts it on the first active expert, later experts replay it with buffer credits on the sender and every receiver (no DRAM re-read, no multicast) | automatic when `[1,1,M,K]` is broadcast, `batchA == 1` and it fits 640 KB per core |
| accumulate mode (`accumulate_batch=True`) | both-sparse (down projection): the valid experts are one K stream, every block reloads the fp32 running sum from the partials buffer and packs it back, the block is packed to the output once after the last expert and written once; output `[1, G, M, N]` (one sum per group), no per-expert slab, no reduce op; a block with no active expert keeps its zeros (`packer_l1_acc` off in this mode) | `ttnn.sparse_matmul(..., accumulate_batch=True)` |
| no zero-fill (`zero_fill_output=False`) | skips the `zeros_like` of the expanded gate\|up output (12.6 us; its inactive slots are never read: the accumulate-mode down skips them) | `ttnn.sparse_matmul(..., zero_fill_output=False)` |
| transposed weights (`in1_transposed=True`) | the weight stored `[1, E, N, K]` with transposed tiles (the matmul's `transpose_b` strides + `IN1_TRANSPOSE_TILE`), so a core's output-width column is a contiguous run of tiles, sequential per DRAM bank | measured below; not adopted unless it wins |

Two bugs found by the regression suite on the way: the receiver/compute loop count must be `batchA x slots`
(an in0-batched call deadlocked), and in0 reuse must require `batchA == 1`.

### Measured (chip 2, device time per call, M = 32, Tracy, `_logs/optexperts_report_kernel2*`)

| call | before (plain kernel, best config) | in0 reuse, G=1 | G=4 (64 cores) | G=6 (96 cores) |
|---|---|---|---|---|
| gate\|up N=512, 28 active (real routing) | 369 x 28/32 = ~323 us (38 %) | 261 us (47 %) | 222 us (55 %) | **204 us (60 %)**, ibw 16 |
| gate\|up N=512, 72 active | 761 us (41 %) | 666 us (47 %) | 554 us (57 %) | **497 us (63 %)** |

| down I=256 | slab kernel + fill | accumulate G=1 (32 cores) | accumulate G=2 | accumulate G=3 (96 cores) |
|---|---|---|---|---|
| 28 active | 129 + 94 us (47 %) | 104 + 2 us (59 %) | 99 + 4 us (61 %) | **92.5 + 4.4 us (66 %)** |
| 72 active | 322 + 94 us (49 %) | 233 + 2 us (67 %) | 242 + 4 us | **217 + 5 us (72 %)** |

Accumulate-mode output vs the slab summed in fp32: PCC 0.999995-0.999998 (one bf16 rounding of the fp32 sum
instead of a bf16 sum of bf16 slabs). Larger K blocks no longer help once every core of the chip streams: the
1-tile-wide column streams (one DRAM bank per core, 2 KB strides) plateau at ~300-330 GB/s; that is what the
transposed layout targets.

### The sparse block with the kernel features (`tests/test_moe_block.py`, real layer-4 tokens, traced replay)

| M | active | path | us/call | us per 32-token block | device ops | PCC vs torch | PCC vs the pre-change block |
|---|---|---|---|---|---|---|---|
| 32 | 28/72 | sparse, new config + kernel features (G=6 gate\|up, accumulate down G=1) | **393.5** | 393.5 | 11 | 0.999917 | 0.999996 |
| 32 | 28/72 | sparse, legacy config + kernel features | 448.8 | 448.8 | 11 | 0.999917 | 0.999996 |
| 256 | 31/72 | sparse, new config + kernel features | 2729.7 | 341.2 x 8 | 105 | 0.999905 | 0.999988 |
(dense unchanged: 720.7 / 1230.8 us.) Remaining ops: pad-fill + sum + untilize (mask), sparse_matmul, 2 slices,
clamped_silu_glu, permute, multiply, sparse_matmul (accumulate) + its 2 us output fill. With G=3 on the down
(`GLM53_SPARSE_W2_GROUPS=3`, now the default) the block is ~11 us shorter. `test_moe.py` (single chip):
32/32, 255/256, PCC 0.999931 / 0.999946. `GLM53_SPARSE_KERNEL=0` disables the features (plain kernel A/B).
Transposed weights (`in1_transposed`: a core's column = contiguous tiles, sequential per DRAM bank) on top of the
groups: gate|up N=512 at 28 active **177 us = 69 %** (G=6, 96 cores; 204 us row-major), at 72 active 425 us = 74 %
(`_logs/optexperts_report_kernel3_transposed.txt`), PCC 0.99997 vs torch: the per-bank contiguous stream is what
the custom op below is built on.

## 9. Custom op `custom_ops/expert_stream` (2026-10-20): the expert block as DRAM-bank-streaming generic ops

Per the user rule, the expert block is rebuilt as two model-local `ttnn.generic_op` programs after
`models/demos/gpt_oss/tt/fused_decode/stream.py` (README: `custom_ops/expert_stream/README.md`): weights stored
width-sharded over the 8 DRAM banks, bank-major and column-major (an expert's columns for a core = one contiguous
range, 8.7 KB NOC packets, one column in flight ahead of compute), the routing matrix scanned in every kernel (the
active set is data-dependent at runtime: trace-safe, no host counts), 96 cores (12 per bank, round-robin over the
active experts) for gate|up with `clamped_silu_glu` in DST and the routing weight as an exact diagonal-tile matmul,
16 cores (2 per bank, 8 output columns each) for the down projection with the expert sum accumulated in fp32 DST
and written once. Three device ops per 32-token block (the block's tokens copied to L1, gate|up, down; plus the
selector matmul on a multi-column mesh), no slab, no zero-fill, no reduce, no activation ops.

`tests/custom_ops/test_expert_stream.py` (random weights at the chip shapes, torch golden on bfp8-rounded weights,
`_logs/optexperts_stream_v3.log`):

| tokens | routed experts per token | active of 72 | act PCC | out PCC |
|---|---|---|---|---|
| 8 | 2 | 16 | 0.999980 | 0.999973 |
| 32 | 2 | 45 | 0.999982 | 0.999980 |
| 32 | 8 | 71 | 0.999982 | 0.999978 |
| 256 (8 blocks) | 2 | 40-45 per block | >= 0.999980 | >= 0.999973 |

### Variant sweep (`test_expert_stream_trace`, random weights, 43 of 72 active, 20 traced replays, `_logs/optexperts_stream_trace*_*.log`)

| gate\|up groups (cores) | down readers (cores) | x | act | us per block | eager gate\|up / down (us) |
|---|---|---|---|---|---|
| 12 (96) | 2 (16) | DRAM | DRAM | 516 | 380 / 158 |
| 12 (96) | 2 (16) | L1 | DRAM | 485 | 351 / 159 |
| 6 (48) | 2 (16) | L1 | DRAM | 609 | 459 / 159 |
| 12 (96) | 4 (32) | L1 | DRAM | 510 | 354 / 185 |
| **12 (96)** | **2 (16)** | **L1** | **L1** | **476** | 346 / 155 |
| 12 (96) | 4 (32) | L1 | L1 | 612 | 568 / 161 |
| 15 (120) | 2 (16) | L1 | L1 | 486 | 489 / 160 |
| 15 (120) | 4 (32) | L1 | L1 | 495 | 382 / 447 |

The eager per-call numbers are single host-dispatched samples (noisy); the traced block is the measure. 96 gate|up
cores are the knee (48 cores 1.28x slower, 120 cores no gain: the 8 banks are the limit), 2 down readers per bank
beat 4 (4 readers halve the contiguous segment per core to 4.3 KB packets), x and act in L1 are worth ~8 % together
(the `moe.py` defaults: `GLM53_STREAM_X_L1` / `GLM53_STREAM_ACT_L1` = 1). With 69 of 72 active (8 routed experts
per token): 707 us per block (gate|up 523 us, down 214 us).

Bandwidth: an expert's chip slice is 3.34 MB (gate|up 2.23 MB + down 1.11 MB, bfp8); 43 active = 144 MB in 476 us
= 302 GB/s (59 % of the 512 GB/s DRAM peak), 69 active = 231 MB in 707 us = 326 GB/s (64 %).

### The expert block (`tests/test_moe_block.py`, real layer-4 weights and routing, traced, `_logs/optexperts_moe_block_stream{,_l1}.log`)

| M | active | path | us per call | us per 32-token block | device ops | PCC vs torch | PCC vs sparse-legacy |
|---|---|---|---|---|---|---|---|
| 32 | 28 | sparse legacy | 1342 | 1342 | 13 | 0.999916 | 1 |
| 32 | 28 | sparse tuned | 683 | 683 | 13 | 0.999916 | 1 |
| 32 | 28 | dense | 720 | 720 | 6 | 0.999917 | 0.999996 |
| 32 | 28 | stream, x and act in DRAM | 372 | 372 | 2 | 0.999926 | 0.999972 |
| 32 | 28 | **stream (default: x and act in L1)** | **341** | **341** | 3 | 0.999926 | 0.999972 |
| 256 | 31 | sparse legacy | 9629 | 1204 x 8 | 121 | 0.999904 | 1 |
| 256 | 31 | sparse tuned | 5036 | 630 x 8 | 121 | 0.999904 | 1 |
| 256 | 31 | dense | 1256 | 1256 x 1 | 6 | 0.999905 | 0.999988 |
| 256 | 31 | stream, DRAM | 2561 | 320 x 8 | 33 | 0.999911 | 0.999952 |
| 256 | 31 | stream (default) | 2410 | 301 x 8 | 41 | 0.999911 | 0.999952 |

Targets: M = 32 real routing <= 400 us: **met** (341 us: 2.0x the tuned sparse block, 2.1x dense, 3.9x the block
before this item; 28 active = 93.6 MB in 341 us = 274 GB/s, 54 % of peak -- the rest is the fixed per-block cost:
program launch x 2, the x copy and load, the routing scans, the DST fill / pack / reconfig per expert, ~110 us).
M = 256 <= 1.0 ms: **not met** -- the block op streams the weights once per 32-token block, so 8 blocks re-stream
the 31 active experts 8x (2.41 ms vs 1.26 ms for the dense layout, which reads each expert once); the dense layout
stays the path at >= 8 users/row until the dispatch form below exists.

### Wider per-chip slices (2026-10-20, after the coordinator's (2,4) mesh run)

The op is generalized to `ppb = I_local / 32 / banks` column pairs per bank (`gate_up_stream_layout`: bank b holds
the pairs `b ppb + q`; the compute loops the pairs per expert with the expert's `D` tile shared; the down segments
of `I/32` K tiles are read in packets that divide the segment and the readers per bank are chosen so that an
expert's block stays <= 128 tiles: 2 readers x 8 columns at I = 256, 4 x 4 at 1024, 8 x 2 at 2048). The 8x4 case
(ppb = 1, one 8.7 KB packet per segment) is unchanged: 476.5 us traced block, same PCCs. `test_expert_stream_widths`
(`_logs/optexperts_p2a_2.log`): I = 1024 (the (2,4) block, 43 active) act 0.999981 / out 0.999974; I = 2048 (one
chip, E = 24) 0.999983 / 0.999976.

### Phase 2 (design, not built): the per-expert dispatch form for M = 256

To read every active expert once at 256 gathered tokens the op has to process, per expert, the rows routed to it
(T x 8 / 288 rows per expert on average, ~7 at 256 tokens; a 32-row DST pass per expert, several passes for an
expert with more rows):
1. on-device dispatch: the writer scans the [T, E] routing matrix once per expert and gathers the routed rows of x
   (<= 32 per pass, T/32 routing tiles scanned) into one x_e tile row in L1 with 64 B row-segment reads (128 per
   row, ~5 us per expert) -- no host-side counts, trace-safe as today;
2. gate|up as now on x_e (K = 4096, one [32, 32] tile per core), `clamped_silu_glu`, the routing weight as
   `diag(w[rows_e, e])` -- act_e [32, 256] per expert, L1 resident;
3. down: `act_e @ W2[e]` [32, 4096] per expert, scattered back as fp32 row partials into an L1-interleaved
   [T, 4096] fp32 accumulator owned by output column: 16 cores x 16 tile columns x T rows is 512 KB fp32 per core
   at T = 256 (too big), so 64 cores x 4 tile columns (128 KB per core; 4-column weight segments = the 4-reader
   packet penalty above, ~+10 %), or bf16 partials per block with one fp32 sum at the end;
4. a last pass packs the accumulator to bf16 [T, 4096].
Expected: weights once (31 active = 104 MB, ~350 us at 300 GB/s) + the gathers (~150 us) + the compute of 256 x ~2
routed rows per chip at HiFi2 (~100 us): ~0.6-0.8 ms per layer per chip, against 1.26 ms dense / 2.41 ms blocks.

### Knobs

`GLM53_STREAM_EXPERTS=1` (the stream block in `TtRoutedExperts` / `TtMoE`, cache tag `_stream`),
`GLM53_STREAM_GROUPS` (12), `GLM53_STREAM_DOWN_READERS` (2), `GLM53_STREAM_X_L1` / `GLM53_STREAM_ACT_L1` (1).
The mesh variants of the unit test ((1,4) / (2,4): per-chip expert slices, replicated tokens) are written but
gated on `GLM53_STREAM_MESH_TESTS=1`: multi-chip meshes were not opened in this pass (board note), so they are not
yet run.

## 10. Phase 2 built: the per-expert dispatch ops (2026-10-20, `custom_ops/expert_stream`, `ExpertDispatch`)

Two more generic ops under the same directory, for `T` gathered tokens (32 < T <= 256) with every active expert's
weights read once (`doc`: `custom_ops/expert_stream/README.md`):

* inputs: the block inputs plus `x_rm = to_layout(x, ROW_MAJOR)` (L1) and `counts = sum_t (routing != 0)`
  (`ttnn.ne` + `ttnn.sum`, exact BF16 integers up to 256): three small ttnn ops; every kernel derives the same pass
  enumeration from the counts (expert ascending, 32 routed rows per pass, `P <= T/4 + E` passes) and the same LPT
  expert -> core-group assignment (`kernels/dispatch_common.hpp`), so the active set stays data-dependent on device;
* `ExpertGateUpDispatch` (96 cores): per pass the writer gathers the pass's token rows of `x_rm` (one 8 KB NOC read
  per row) into a row-major block, the compute tilizes it (`tilize_block`, 128 tiles) and runs the gate|up / SwiGLU /
  `D @ h` of the block op into `act [1, P, 32, I]`; bank 0 writes the pass table `[P, 64]` uint16
  (`[n_rows, e, t_0 .. t_31]`). The (gate, up) pair of an expert stays resident over its passes (two pair slots of
  2 x 139 KB; a pair must never straddle the CB end: the first version with three single-column slots and a 2-column
  wait read past the CB on every second expert);
* `ExpertDownDispatch` (64 cores: 8 per bank x 2 output columns): phase 1 `h_k = act[k] @ W2[e][:, cols]` per pass
  (the expert's 17 KB block resident over its passes, pass counts through a metadata page read by the unpack thread
  and forwarded to the math / pack threads), packed BF16 into a resident buffer (`P x 2` tiles = 544 KB); phase 2
  per token row tile: the writer copies every (pass, row) contribution of the tile's tokens (<= 8 per token) into
  gather slot tiles, the compute sums the slots in fp32 DST as identity-tile matmuls (HiFi4, exact) and packs the
  two output tiles.

`tests/custom_ops/test_expert_dispatch.py` (random weights, Zipf-skewed routing, 2 picks per token, `popular` =
experts routed by every token -> multi-pass; `_logs/optexperts_p2g_1.log`):

| T | I | active of 72 | passes | act PCC | out PCC | pass table |
|---|---|---|---|---|---|---|
| 64 | 256 | 31 | 32 | 0.999982 | 0.999978 | exact |
| 256 | 256 | 58 | 66 | 0.999981 | 0.999997 | exact |
| 256 | 256 (+2 popular) | 58 | 80 | 0.999984 | 0.999991 | exact |
| 64 | 1024 (+1 popular) | 29 | 30 | 0.999983 | 0.999982 | exact |

Traced (prepare ops + both generic ops, 20 replays): T = 256, I = 256, 58 active / 72 passes: 1118 us per call
(eager gate|up 554 us, down 595 us); T = 64, I = 1024 (the (2,4) chip width, 35 active): 2073 us (gate|up 1369 us:
35 x 8.9 MB of full-width... 4x wider slices = 312 MB of weights, the platform's cost, not the op's).

Real routing (`tests/test_moe_block.py`, layer-4 weights, M = 256 = 31 of 72 active, `GLM53_STREAM_EXPERTS=1`
default policy: block form at T <= 32, dispatch above; `_logs/optexperts_moe_block_dispatch3.log`):

| M | path | us per layer per chip | device ops | PCC vs torch | PCC vs sparse-legacy |
|---|---|---|---|---|---|
| 32 | stream block (unchanged) | 343 | 3 | 0.999926 | 0.999972 |
| 256 | stream, 8 blocks (section 9) | 2410 | 41 | 0.999911 | 0.999952 |
| 256 | dense one pass | 1256 | 6 | 0.999905 | 0.999988 |
| 256 | **stream dispatch** | **779** | 6 | 0.999908 | 0.999954 |

Target M = 256 <= 0.8 ms per layer per chip: **met, 779 us** (1.6x the dense one-pass layout, 3.1x the 8-block
stream form, 6.5x the tuned sparse blocks); runs vary 779-822 us between kernel variants of the down op (see below). Where the time goes (unit test at 58
active, eager per op): gate|up ~560 us (58 x 2.23 MB = 129 MB, ~75 % of it DRAM time at 300 GB/s; the rest the
per-pass gather + tilize), down ~600 us (64 MB of W2 at 17 KB blocks from 8 cores per bank, 74 MB of activation
tile reads over the NOC, ~80 us of gather copies). The down op is the remaining lever: an expert-parallel
split (each of a bank's 8 cores streams whole 70 KB expert blocks for a subset of the experts, as the block op's
readers do at 310 GB/s, then reduces its [T, 16-column] partial into the bank owner's L1) would halve its DRAM
inefficiency and cut the activation traffic 8x; estimated down ~150-200 us, layer ~550-600 us at real routing.
Transaction-id pipelining of the activation reads (`noc_async_read_set_trid` + plain reads) did not synchronize
on the writer's NOC (stale slots); the reads are batched three passes per barrier instead.

## 11. EP32 expert path (2026-10-20, orchestration v2: `TtEPExperts`, `GLM53_MOE_LAYOUT=ep`)

Direction change (doc/mesh/orchestration_v2.md, DeepSeek-V3 Galaxy reference): every chip owns
`E / D` FULL-width experts (9 on the 8x4, 36 on a (2,4) block), the rows' tokens reach their column's experts
through `all_to_all_dispatch` along the mesh rows, the expert compute is the dispatch op in **EP mode**, and
`all_to_all_combine` (local reduce) brings the weighted sums back; the column all-reduce stays.

EP mode of the dispatch ops (`custom_ops/expert_stream`, compile-time `EP`): the routing arrives as the top-k
lists (index tiles `[T, 32]` BF16 = global expert ids, weight tiles, both exact BF16) plus the chip's expert base
(one uint16 page per chip, sharded over the mesh); every kernel counts the chip's experts' rows from the index list
(T x K checks: no `ne` / `sum` helper ops), the writer gathers the rows and builds `D` from the weights directly.
The 288-expert id space is untouched (ids outside `[base, base + E_local)` are other chips' experts, as after a
dispatch). Unit test `test_expert_dispatch_ep` (random weights, Zipf lists over 288 experts, I = 2048 = 8 pairs
per bank, `_logs/optexperts_ep*_*.log`):

| T | experts owned | routed rows | passes | act PCC | out PCC | pass table |
|---|---|---|---|---|---|---|
| 64 | [9, 18) | 18 | 5 | 0.999981 | 0.999978 | exact |
| 256 | [27, 36) | 31 | 8 | 0.999982 | 0.999978 | exact |
| 64 | [36, 72) | 81 | 15 | 0.999983 | 0.999979 | exact |
| 256 | [72, 108) | 244 | 35 | 0.999986 | 0.999991 | exact |
| 64 (bfp4 gate\|up) | [9, 18) | 18 | 5 | 0.999983 | 0.999978 (vs the bfp4-rounded golden) | - |

(The down op's L1 budget shrinks its activation / weight slots to 2 + 2 at the (T = 256, E = 36) corner: the
resident per-pass h buffer is 400 KB there.)

Single-chip path (`tests/test_moe_ep.py`, D = 1: dispatch / combine are the identity; real layer-4 weights and
routing, the chip's share = the first `E_loc` experts at full width, the tokens' top-8 lists over all 288 experts;
the hybrid stream path on the same experts as the reference; `_logs/optexperts_ep6_1.log`, traced, 20 replays):

| experts owned | T | active | routed rows | hybrid stream us | EP us | EP device ops | EP vs torch | hybrid vs torch | EP vs hybrid |
|---|---|---|---|---|---|---|---|---|---|
| 9 (the 8x4 share) | 32 | 2 | 10 | 306.5 | 297.2 | 7 | 0.999922 | 0.999919 | 0.999994 |
| 9 | 256 | 3 | 81 | 489.0 | 589.6 | 7 | 0.999918 | 0.999918 | 1.000000 |
| 36 (a (2,4) share) | 32 | 11 | 37 | 1225.8 | 1377.5 | 7 | 0.999925 | 0.999923 | 0.999992 |
| 36 | 256 | 12 | 329 | 2079.0 | 2069.4 | 7 | 0.999925 | 0.999925 | 1.000000 |

Exact vs the hybrid block (>= 0.99999; bit-identical at T = 256 where both run the dispatch op). The EP path is the
same two generic ops plus the index-tile prep (typecast + concat x 2, the row-major copy): 7 device ops, no counts
ops. Per-chip expert time is weight-bound at full width: 9 experts x 26.7 MB = 240 MB when all are active -- at
T = 256 only 3 of this chip's 9 were active (81 rows): 590 us; 12 of 36 active: 2.07 ms (320 MB). The 8x4 EP
layer time at 32 users per row is then ~0.6 ms + the dispatch / combine pair + the column all-reduce.

Dispatched-row imbalance of the real routing (layer 4, rows per device, chip d owns experts `[d E/D, (d+1) E/D)`):

| tokens | D | rows per device max / mean / min | max / mean | active experts |
|---|---|---|---|---|
| 32 | 32 (the 8x4) | 17 / 8.0 / 1 | 2.12 | 92 of 288 |
| 256 | 32 | 155 / 64.0 / 2 | 2.42 | 112 of 288 |
| 32 | 8 (a (2,4) block) | 49 / 32.0 / 15 | 1.53 | 92 of 288 |
| 256 | 8 | 342 / 256.0 / 85 | 1.34 | 112 of 288 |

The busiest chip of the 8x4 sees 2.4x the mean rows at 256 tokens (the step time follows the max); the streaming
kernels pad nothing, so the imbalance costs only the busiest chip's extra passes (155 rows = ~9 passes over its
active experts).

Gate|up bfp4 knob (`GLM53_EXPERT_W13_DTYPE=bfp4`, 576 B tiles through the same kernels, `w13_dtype` of the ops):
unit test `test_expert_dispatch_ep_bfp4` PCC 0.999983 / 0.999978 vs the bfp4-rounded golden (the kernels are
format-agnostic; the block-level PCC vs the bfp8 reference on real weights is the next run of
`tests/test_moe_ep.py` with the knob set -- not run before the board reset).

Mesh path (`TtEPExperts.forward` on D > 1: `all_to_all_dispatch` of the row-major tokens and of the weight lists
along axis 0, the EP op on the dispatched rows, `all_to_all_combine(local_reduce=True)`): written, UNTESTED until
a (2,4) block is available; the (2,4) numbers (per-layer time at 4 / 32 tokens per device, dispatch / combine
times) are the coordinator's follow-up.

Down-op note for the EP shapes: every down core reads every pass's full [32, I] activation block (I = 2048:
128 KB per pass per core, 64 cores); at the 8x4 shape (<= ~20 passes) that is ~2.6 MB per core, on a (2,4)
chip (36 experts, ~36 passes) ~4.6 MB: the expert-parallel down split (section 10) would cut it 8x.

## 12. Pair-parallel work units in the gate|up dispatch (2026-10-20, host-only during the board reset; UNMEASURED)

Why: at full width a chip's experts are few (9 on the 8x4, 2-3 active at 32-256 tokens) and the LPT over
*experts* left one expert per core: a core streamed its expert's 2.2 MB (8 column pairs) alone at a 1/12 bank
share, ~310 us per expert whatever the token count (`test_moe_ep.py`: 297 us at T = 32 with 2 active, 590 us at
T = 256 with 3 active, of which ~350 us gate|up).

Change (`for_each_unit` in `dispatch_common.hpp`, the three gate|up dispatch kernels): the work unit is
(expert, column pair); the bank's `ppb` pairs of an active expert spread over the bank's 12 cores (LPT by pass
count: with one pair per bank it is exactly the old per-expert assignment, so the 8x4 hybrid path is
unchanged); every unit's pair is streamed once and stays resident over the expert's passes (no per-pass
re-streaming for `ppb > 1` any more); the unit's core gathers / tilizes the pass rows itself (8x the gather +
tilize work per expert at `ppb = 8`, spread over 8 cores: ~16 us per pass per unit); the pass table is written by
the (bank 0, pair 0) unit. No unit list in memory: each kernel replays the deterministic assignment and acts on its
own units.

Expected at the 8x4 EP shape: 3 active experts = 24 units per bank = 2 per core = 557 KB per core at ~5.3 GB/s:
~105 us + 2 units x passes x 16 us -> ~140 us gate|up (was ~350); 2 active at T = 32 -> ~75 us (was ~300);
9 active -> bank-bound ~310 us as before. The (2,4) share (36 experts) is bank-bound either way.
To measure after the reset: `pytest tests/custom_ops/test_expert_dispatch.py -k ep` (correctness, 4 shapes +
bfp4) then `tests/test_moe_ep.py` (the table of section 11 again).
