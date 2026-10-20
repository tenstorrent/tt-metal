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
