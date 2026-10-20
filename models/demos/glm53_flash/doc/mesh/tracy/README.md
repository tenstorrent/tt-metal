# GLM-5.3-Flash Phase 2: traced decode step on the (2,4) chip block (tt-perf-report, 2026-10-20)

Design of record section 7 ("tt-perf-report; traced decode"). Layers 4 (KDA+MoE) and 3 (DSA+MoE) on the
grid rows 4-5 block of the Blackhole Galaxy (`TT_VISIBLE_DEVICES=27,26,31,30,23,22,19,18`, `MESH_DEVICE=P150x8`,
`p150_x8_mesh_graph_descriptor.textproto`, FABRIC_2D RELAXED_INIT, max-payload router config, `l1_small_size`
32768; `tt/mesh.py::mesh_device_params`), TP4 over the columns, one DP user group per row, hybrid EP4 x TP2
experts (72 of the 288 experts x 1024-wide halves of the 2048 intermediate per chip), at B = 4 and 32 users per
row, position 1024, expert weights bfp8. tt-metal main `8cfcaf754d3` + `arg/glm53_meshperf` (the commit adding
this directory).

## Method

- `tests/test_perf_mesh.py`: `FunctionalDecoder.from_state_dict` on the mesh, `max_batch = B`, decode streams =
  `rows * B` distinct real token rows of the T4097 dump (so the router selects realistic expert sets), row-sharded
  (`host_streams_to_device(..., mesh=)`), positions `[1024] * rows * B` (`make_decode_positions`). The mesh has no
  prefill path, so a DSA layer's latent pages are seeded with random rows (the single-chip `GLM53_PERF_SEED_ONLY`
  method: the decode profile does not depend on the cache content, and no SDPA prefill kernel is compiled, which is
  also what sidesteps the Tracy zone-hash collision of `compute_streaming.hpp` on this commit, see
  `doc/functional_decoder/README.md`). Trace captured with `capture_decode_trace`, one warm replay,
  `ReadDeviceProfiler` drain, signpost `PERF_DECODE`, ONE replay (`GLM53_PERF_DECODE_STEPS=1`, so rows == ops),
  `synchronize_device`, `PERF_DECODE_END`, drain; then one eager decode step inside `PERF_EAGER .. PERF_EAGER_END`
  with nested per-module signposts (`ATTN`, `NORM`, `ROUTER`, `EXPERTS`, `SHARED`, `MOE`, and the collectives
  patched on `GLMMesh`): the eager step issues exactly the ops of the captured graph in the same order (checked:
  equal op counts in every run), so every op of the traced window gets the module that issued it (innermost open
  label; ops outside every sublayer are the mHC hyper-connections). The wall time comes from profiler-free runs
  (`GLM53_PERF_WALL_STEPS=20` non-blocking replays + `synchronize_device`); under the device profiler the wall loop
  is skipped: a layer-3 B=32 step leaves ~5k markers on some cores, 3 un-drained replays overflow the 12000-entry
  per-core buffers and 10 crashed the Tracy capture (no CSV).
- Collection (one layer x B point per run):
  `GLM53_PERF_LAYER=<3|4> GLM53_PERF_B=<4|32> GLM53_PERF_DECODE_STEPS=1 python -m tracy -r -p -v -m pytest
  models/demos/glm53_flash/tests/test_perf_mesh.py -s --timeout 3000`; the mesh profiler writes ONE ops CSV with a
  `DEVICE ID` column (local chip id; row-major mesh order `[1, 0, 2, 3, 5, 4, 6, 7]` on this block, i.e. chip (0,0) is
  id 1, logged by the test). `tests/perf_mesh_report.py <csv> --out <dir> --rows 2 --cols 4 --ids 1,0,2,3,5,4,6,7`
  splits it per device, runs `tt-perf-report --start-signpost PERF_DECODE --end-signpost PERF_DECODE_END` on each
  (`dev<id>_r<r>c<c>_decode_perf_report.txt`, the full per-op tables; the per-op CSVs are git-excluded) and writes
  `summary.md` (step time per chip, shares and top ops of chips (0,0) and (1,0), the CCL ops in order).
- Logs: `~/ashai/glm53/_logs/meshperf_plain_l{4,3}_b{4,32}.log` (no profiler, 20-step wall),
  `meshperf_tracy_l4_b4.log`, `meshperf_tracy_l4_b32.log`, `meshperf_tracy_l3_b4.log`, `meshperf_tracy_l3_b32_2.log`
  (Tracy runs; ops CSVs under `$W/generated/profiler/reports/<run>/`, run ids in the summary titles;
  `meshperf_tracy_l3_b32.log` is the crashed-capture attempt).

## Results (device time = `Device Time` sum of chip (0,0)'s traced window; the 8 chips agree within 15 us)

| layer | users/row (total) | device ms/step (0,0) | all chips ms | wall ms/step, no profiler (20 replays) | wall under Tracy | single-layer tok/s per user | single-layer tok/s aggregate (8 chips, 2 rows) | x45 layers tok/s per user / aggregate | ops/step |
|---|---|---|---|---|---|---|---|---|---|
| 4 KDA+MoE | 4 (8) | 7.985 | 7.974-7.985 | 8.02 | 8.29 (10 replays) | 125 | 1002 | 2.8 / 22 | 498 |
| 4 KDA+MoE | 32 (64) | 12.448 | 12.435-12.453 | 12.43 | 12.63 (10 replays) | 80 | 5141 | 1.8 / 114 | 479 |
| 3 DSA+MoE | 4 (8) | 7.785 | 7.771-7.785 | 7.75 | 8.01 (10 replays) | 128 | 1028 | 2.9 / 23 | 462 |
| 3 DSA+MoE | 32 (64) | 9.940 | 9.929-9.940 | 9.93 | 10.24 (3 replays) | 101 | 6439 | 2.2 / 143 | 443 |

tok/s = 1000 / device ms per step (per user) and x total users (aggregate); "x45 layers" divides by 45 as the
single-chip table does (every layer costed like this one; layers 0-2 are dense and cheaper, embedding / head not
included). Wall = host `perf_counter` around non-blocking replays + `synchronize_device`: without the profiler
it is within 0.04 ms of the device-time sum at every point (8.02 / 12.43 / 7.75 / 9.93 vs 7.985 / 12.448 / 7.785 /
9.940: the traced step is dispatch-free, and the Tracy sum carries the per-kernel marker cost); under Tracy the
replays run 0.2-0.3 ms slower.

## Shares of the step, chip (0,0) (device us; full tables in `<case>/summary.md`)

| group | L4 B=4 | L4 B=32 | L3 B=4 | L3 B=32 |
|---|---|---|---|---|
| step | 7985 | 12448 | 7785 | 9940 |
| CCL kernels (ReduceScatter / AllGather / AllBroadcast) | 135 (1.7 %) | 238 (1.9 %) | 230 (3.0 %) | 333 (3.3 %) |
| CCL composites (kernels + their pad/tilize/untilize/concat at 8 rows) | 214 (2.7 %) | 238 (1.9 %) | 309 (4.0 %) | 333 (3.3 %) |
| `sparse_matmul` (the 2 expert matmuls per 32-token block) | 899 (11.3 %) | 2353 (18.9 %) | 802 (10.3 %) | 2106 (21.2 %) |
| experts local total (selector matmul, activation, pad, sparse_matmul, reduce; no CCL) | 1424 (17.8 %) | 3215 (25.8 %) | 1326 (17.0 %) | 2965 (29.8 %) |
| router (exact fp32 composite top-8) | 3391 (42.5 %) | 3380 (27.2 %) | 3396 (43.6 %) | 3382 (34.0 %) |
| shared expert | 89 (1.1 %) | 90 (0.7 %) | 89 (1.1 %) | 88 (0.9 %) |
| MoE sublayer total | 5080 (63.6 %) | 6862 (55.1 %) | 5084 (65.3 %) | 6732 (67.7 %) |
| attention composite incl. its all-reduce (L4: KDA recurrence; L3: flash-MLA + pooled-key index) | 928 (11.6 %) | 3269 (26.3 %) | 738 (9.5 %) | 908 (9.1 %) |
| RMSNorm x2 | 93 (1.2 %) | 93 (0.7 %) | 93 (1.2 %) | 95 (1.0 %) |
| mHC hyper-connections x2 (everything outside the sublayers) | 1885 (23.6 %) | 2224 (17.9 %) | 1869 (24.0 %) | 2205 (22.2 %) |

Row 1 chip (1,0) matches row 0 within 10 us per group in every case (the rows run the same graph on different
users). The chips of a row are in lockstep (step spread < 15 us); where one chip shows more CCL kernel time (column 3:
0.37-0.72 ms vs 0.12-0.25 ms; chip (0,0) of layer 3: 122 / 215 us in the MoE all-reduce's ReduceScatter vs 23 us on
its neighbours) it is the ring skew absorbed as wait inside that kernel, not extra work.

### Top ops, chip (0,0) (repeats of the same op collapsed; "cores" = the op's grid)

Layer 4, B=4/row (7985 us):

| op | module | cores | count | us each | us total | % |
|---|---|---|---|---|---|---|
| AccumulationDeviceOperation (`ttnn.cumsum` first-match tie-break, one per top-8 round) | MoE/router | 1 | 8 | 330 | 2642 | 33.1 |
| SparseMatmulDeviceOperation 72 experts x 32 x 4096 x 2048 (gate\|up) | MoE/experts | 64 | 1 | 670 | 670 | 8.4 |
| TernaryDeviceOperation (stream mixing) | mHC | 120 | 38 | 16 | 622 | 7.8 |
| BinaryNgDeviceOperation (masks, score updates) | MoE/router | 120 | 68 | 4 | 285 | 3.6 |
| MatmulDeviceOperation 32 x 16384 x 32 (mHC mixing weights) | mHC | 1 | 2 | 131 | 262 | 3.3 |
| BinaryNgDeviceOperation (gated recurrence) | KDA | 120 | 20 | 13 | 256 | 3.2 |
| BinaryNgDeviceOperation | mHC | 120 | 18 | 13 | 235 | 2.9 |
| ReduceDeviceOperation | mHC | 1 | 2 | 115 | 229 | 2.9 |
| SparseMatmulDeviceOperation 72 x 32 x 1024 x 4096 (down) | MoE/experts | 64 | 1 | 229 | 229 | 2.9 |
| UnaryDeviceOperation (expert activation / masks) | MoE/experts | 120 | 5 | 46 | 228 | 2.9 |
| TilizeWithValPaddingDeviceOperation | mHC | 1 | 42 | 3 | 133 | 1.7 |
| FillPadDeviceOperation | MoE/router | 9 | 24 | 5 | 118 | 1.5 |
| MhcSplitSinkhornDeviceOperation | mHC | 1 | 2 | 52 | 104 | 1.3 |
| MatmulDeviceOperation 32 x 4096 x 128 (KDA gate projections) | KDA | 4 | 2 | 51 | 102 | 1.3 |

Layer 4, B=32/row (12448 us):

| op | module | cores | count | us each | us total | % |
|---|---|---|---|---|---|---|
| AccumulationDeviceOperation (router cumsum) | MoE/router | 1 | 8 | 330 | 2642 | 21.2 |
| SparseMatmulDeviceOperation 72 x 32 x 4096 x 2048 (gate\|up, 2 token blocks) | MoE/experts | 64 | 2 | 888 | 1776 | 14.3 |
| BinaryNgDeviceOperation (gated recurrence on [32 users x 16 local heads, 128, 128] fp32) | KDA | 120 | 20 | 67 | 1332 | 10.7 |
| TernaryDeviceOperation | mHC | 120 | 38 | 16 | 622 | 5.0 |
| SparseMatmulDeviceOperation 72 x 32 x 1024 x 4096 (down, 2 blocks) | MoE/experts | 64 | 2 | 288 | 577 | 4.6 |
| ReshapeViewDeviceOperation | KDA | 120 | 13 | 42 | 549 | 4.4 |
| UnaryDeviceOperation | MoE/experts | 120 | 10 | 46 | 459 | 3.7 |
| UntilizeWithUnpaddingDeviceOperation | mHC | 1 | 42 | 9 | 398 | 3.2 |
| TilizeWithValPaddingDeviceOperation | KDA | 96 | 3 | 98 | 294 | 2.4 |
| BinaryNgDeviceOperation | MoE/router | 120 | 68 | 4 | 286 | 2.3 |
| MatmulDeviceOperation 32 x 16384 x 32 | mHC | 1 | 2 | 132 | 263 | 2.1 |
| PermuteDeviceOperation | MoE/router | 120 | 16 | 15 | 244 | 2.0 |
| BinaryNgDeviceOperation | mHC | 120 | 18 | 13 | 237 | 1.9 |
| ReduceDeviceOperation | mHC | 1 | 2 | 115 | 229 | 1.8 |

Layer 3, B=4/row (7785 us):

| op | module | cores | count | us each | us total | % |
|---|---|---|---|---|---|---|
| AccumulationDeviceOperation (router cumsum) | MoE/router | 1 | 8 | 330 | 2644 | 34.0 |
| TernaryDeviceOperation | mHC | 120 | 38 | 16 | 610 | 7.8 |
| SparseMatmulDeviceOperation 72 x 32 x 4096 x 2048 (gate\|up) | MoE/experts | 64 | 1 | 591 | 591 | 7.6 |
| BinaryNgDeviceOperation | MoE/router | 120 | 68 | 4 | 284 | 3.6 |
| MatmulDeviceOperation 32 x 16384 x 32 | mHC | 1 | 2 | 131 | 262 | 3.4 |
| BinaryNgDeviceOperation | mHC | 120 | 18 | 13 | 231 | 3.0 |
| ReduceDeviceOperation | mHC | 1 | 2 | 115 | 229 | 2.9 |
| UnaryDeviceOperation | MoE/experts | 120 | 5 | 45 | 224 | 2.9 |
| SparseMatmulDeviceOperation 72 x 32 x 1024 x 4096 (down) | MoE/experts | 64 | 1 | 211 | 211 | 2.7 |
| TilizeWithValPaddingDeviceOperation | mHC | 1 | 42 | 3 | 133 | 1.7 |
| ReduceScatterDeviceOperation (MoE all-reduce; ring skew lands here) | MoE/CCL | 10 | 1 | 122 | 122 | 1.6 |
| FillPadDeviceOperation | MoE/router | 9 | 24 | 5 | 119 | 1.5 |
| MhcSplitSinkhornDeviceOperation | mHC | 1 | 2 | 52 | 104 | 1.3 |
| MatmulDeviceOperation 32 x 4096 x 128 (indexer projections) | DSA | 4 | 2 | 51 | 102 | 1.3 |

Layer 3, B=32/row (9940 us):

| op | module | cores | count | us each | us total | % |
|---|---|---|---|---|---|---|
| AccumulationDeviceOperation (router cumsum) | MoE/router | 1 | 8 | 330 | 2644 | 26.6 |
| SparseMatmulDeviceOperation 72 x 32 x 4096 x 2048 (gate\|up, 2 blocks) | MoE/experts | 64 | 2 | 789 | 1577 | 15.9 |
| TernaryDeviceOperation | mHC | 120 | 38 | 16 | 607 | 6.1 |
| SparseMatmulDeviceOperation 72 x 32 x 1024 x 4096 (down, 2 blocks) | MoE/experts | 64 | 2 | 265 | 529 | 5.3 |
| UnaryDeviceOperation | MoE/experts | 120 | 10 | 45 | 454 | 4.6 |
| UntilizeWithUnpaddingDeviceOperation | mHC | 1 | 42 | 9 | 398 | 4.0 |
| BinaryNgDeviceOperation | MoE/router | 120 | 68 | 4 | 286 | 2.9 |
| MatmulDeviceOperation 32 x 16384 x 32 | mHC | 1 | 2 | 132 | 264 | 2.7 |
| PermuteDeviceOperation | MoE/router | 120 | 16 | 15 | 245 | 2.5 |
| BinaryNgDeviceOperation | mHC | 120 | 18 | 13 | 233 | 2.3 |
| ReduceDeviceOperation | mHC | 1 | 2 | 115 | 229 | 2.3 |
| ReduceScatterDeviceOperation (MoE all-reduce, skew) | MoE/CCL | 10 | 1 | 215 | 215 | 2.2 |
| TilizeWithValPaddingDeviceOperation | mHC | 1 | 42 | 4 | 176 | 1.8 |
| BinaryNgDeviceOperation | MoE/experts | 120 | 4 | 31 | 124 | 1.2 |

DSA attention at TP4 (16 local heads, chip (0,0)), B=4 -> B=32: q_a 32x4096x1536 68 us (48 cores), q_b
32x1536x4096 31 us, kv_a 32x4096x512 58 us (16 cores), the two indexer projections 32x4096x128 51 us each on 4
cores, absorbed q 16 heads x 32x256x512 61 -> 60 us (16 cores), `SdpaDecode` 73 us on 12 cores (3 cores per
user) -> 109 us on 120 cores, absorbed out 16 x 32x512x256 100 -> 99 us (8 cores), o_proj 32x4096x4096 76 us
(64 cores), 4 `PagedUpdateCache` + the pooled-key index maintenance ~40 us, reshapes 15 -> 90 us: 0.70 -> 0.87 ms
local. The flash-MLA decode itself is 1 % of the step at 64 users.

### CCL ops of one step, chip (0,0) (device us, B=4 / B=32)

| collective | lowering | layer 4 | layer 3 |
|---|---|---|---|
| attention o_proj `all_reduce_cols` (TP4, [1,1,B,4096] bf16) | ReduceScatter + AllGather over the row | 27 + 14 / 51 + 14 | 25 + 14 / 24 + 16 |
| experts `all_gather_rows` x2 (tokens, router weights; DP2) | AllBroadcast x2 at 8 rows (+ tilize/untilize/concat), AllGather x2 at 64 rows | 24 + 6 (+47 local) / 16 + 7 | 27 + 6 (+47 local) / 32 + 7 |
| experts `reduce_scatter_rows` (DP2 partials) | ReduceScatter (+ pad/slice at 8 rows) | 19 (+32 local) / 25 | 20 (+32 local) / 19 |
| MoE `all_reduce_cols` (TP4) | ReduceScatter + AllGather | 23 + 20 / 112 + 19 | 122 + 15 / 215 + 20 (skew on this chip) |

## Comparison with the single-chip decode tables (`doc/functional_decoder/tracy/`, B=1)

| op class (device us) | L4 single chip B=1 (10759, 471 ops) | L4 block B=4/row | L4 block B=32/row | L3 single chip B=1 (10861, 435 ops) | L3 block B=4/row | L3 block B=32/row |
|---|---|---|---|---|---|---|
| AccumulationDeviceOperation (router cumsum x8) | 2642 | 2642 | 2642 | 2644 | 2644 | 2644 |
| UnaryDeviceOperation | 1480 (x17) | 267 (x17) | 499 (x22) | 1420 (x12) | 246 (x12) | 474 (x17) |
| BinaryNgDeviceOperation | 1280 (x110) | 845 (x110) | 1984 (x112) | 950 (x92) | 589 (x92) | 662 (x94) |
| SparseMatmulDeviceOperation | 1111 (x2, 288 experts x 2048-wide) | 899 (x2, 72 x 1024-wide) | 2353 (x4) | 1112 (x2) | 802 (x2) | 2106 (x4) |
| MatmulDeviceOperation | 1033 (x12) | 676 (x13) | 686 (x14) | 1662 (x13) | 894 (x14) | 902 (x15) |
| TernaryDeviceOperation | 622 (x38) | 622 (x38) | 622 (x38) | 620 (x38) | 610 (x38) | 607 (x38) |
| everything else | 2591 | 2034 | 3662 | 2453 | 2000 | 2545 |
| step | 10759 | 7985 | 12448 | 10861 | 7785 | 9940 |

The TP4 x DP2 layout buys 2.8-3.1 ms per step at 8x the users: the expert activation unaries (1.48 -> 0.27 ms:
8x fewer elements over the local 72 x 1024-wide expert slice instead of 288 x 2048), the attention projections
(TP4 heads: matmuls 1.03 -> 0.68 ms on layer 4, 1.66 -> 0.89 ms on layer 3; KDA elementwise 1.28 -> 0.85 ms) and
the expert matmuls (1.11 -> 0.80-0.90 ms: 1/4 of the experts and 1/2 of the intermediate per chip, but the same
DRAM-bound 32-row block). What does NOT shrink is everything replicated on every chip: the router composite (3.4
ms, identical on all 8 chips and in all 6 columns of the table), the mHC hyper-connections (1.9 ms: 220 tiny ops,
two single-core 32x16384x32 matmuls and two single-core reduces), the norms. At 64 users the growth is the second
32-token expert block (+1.2-1.3 ms) and, on layer 4 only, the KDA recurrence (0.9 -> 3.2 ms, elementwise over
[32 users x 16 local heads, 128, 128] fp32); the DSA layer grows 2.2 ms for 8x the users (flash-MLA +0.04 ms).

## Design section 7 estimates vs measured

| estimate | measured (chip (0,0)) |
|---|---|
| per-layer all-reduce ~12 us on a ring of 4 | TP4 all-reduce = ReduceScatter + AllGather composite: 39-44 us of kernels at 8 rows, 40-65 us at 64 rows (plus the ring skew on one chip: 112-277 us), and 50-80 us of pad/tilize ops around the 8-row DP collectives; all collectives of a layer = 135-230 us (1.7-3.0 %) at B=4, 238-333 us (1.9-3.3 %) at B=32 -> not a target yet |
| experts streaming ~1.4 ms at 1 user | expert sublayer (local) 1.33-1.42 ms for one 32-token block at 8 users (sparse_matmul 0.80-0.90 ms = 0.59-0.67 gate\|up + 0.21-0.23 down over 72 local experts), 2.97-3.22 ms for two blocks at 64 users; the gate\|up matmul runs at ~35 % of DRAM peak (dtype gate note), bfp4 gate/up saves ~18 % of it |

## First optimization targets (both layers point the same way)

1. The exact fp32 router top-8 composite: 3.4 ms = 42-44 % of the 8-user step and 27-34 % at 64 users, replicated
   on all 8 chips, flat in B. 2.64 ms of it is the `ttnn.cumsum` first-match tie-break (one single-core
   `AccumulationDeviceOperation` of 330 us per round, 8 rounds), the rest 139 tiny ops (68 BinaryNg, 25 FillPad,
   25 Reduce, 16 Permute). Replace the per-round cumsum: `ttnn.argmax` (first index) + `eq(iota, idx)` one-hot, or
   a multicore cumsum / a single fp32 top-k; or a bf16 `moe_grouped_topk` with an fp32 re-check of the near-tie
   rows. Expected: -2.6 to -3.0 ms per MoE layer (8 users: 8.0 -> ~5.2 ms, i.e. 125 -> ~190 tok/s per user).
2. The mHC hyper-connections: 1.9-2.2 ms per layer over 220 ops (38 Ternary x 16 us, 42 Tilize + 42 Untilize + 64
   Slice on 1 core, two 32 x 16384 x 32 single-core matmuls of 131 us and two single-core Reduce of 115 us).
   Multicore program configs for the two matmuls / reduces (-0.4 ms) and fusing the per-stream slice/tilize/
   untilize/ternary chains (the `MhcSplitSinkhornDeviceOperation` kernel already covers the Sinkhorn part):
   -1 to -1.5 ms per layer, x45 layers.
3. KDA decode recurrence at 64 users: 3.2 ms (26 % of layer 4), 20 BinaryNg x 67 us + 13 ReshapeView x 42 us + 3
   Tilize x 98 us: a fused gated-delta decode kernel over [B*H_local, 128, 128] (the ttKDA prefill kernels have the
   chunked recurrence) or in-place elementwise without the reshape/tilize round trips: -2 ms at B=32 on the 34 KDA
   layers; at 8 users it is 0.9 ms.
4. Experts: 0.8-0.9 ms (one block) / 2.1-2.35 ms (two blocks) of sparse_matmul at ~35 % DRAM peak, plus 0.45 ms of
   activation unaries at two blocks; bfp4 gate/up (-18 % of the matmul, accuracy-gated), and at 64 tokens the two
   32-token blocks could become one 64-row sparse_matmul call. The padding ops around the 8-row DP collectives
   (80 us) disappear at B >= 32.
5. DSA small grids: the two indexer projections (32x4096x128 on 4 cores, 100 us) and the absorbed q/out matmuls on
   8-16 cores (160 us): -0.15 ms with wider grids. The flash-MLA kernel is not a target (73-109 us).
6. CCL: 1.7-3.3 % of the step; the all-reduce measures 39-65 us vs the 12 us estimate, only worth revisiting after
   1-3 (fused all-reduce instead of ReduceScatter + AllGather; the attention all-reduce could fold into the mHC
   combine).

## Caveats

- Tracy zone-hash collision (layer 3): the dense and sparse SDPA prefill kernels share `compute_streaming.hpp`
  (zone hash 0x6835 on main 8cfcaf754d3), so a run that compiles any prefill SDPA kernel throws at device close.
  The mesh test never prefills (seed-only decode), which is why both layers profile cleanly here; the single-chip
  layer-3 table needed `GLM53_PERF_SEED_ONLY=1` for the same reason.
- Device time is `DEVICE KERNEL DURATION` per op summed over the window (no op-to-op gaps: tt-perf-report shows
  ~300 us of gaps per step); wall time includes dispatch (and, for the signposted single replay, its first-replay
  latency: 8.4 / 16.0 / 8.2 / 10.4 ms).
- Per-chip time is lockstep: the spread over the 8 chips is < 15 us; the extra CCL "time" on one chip is skew
  absorbed as wait inside that kernel.
- Profiler buffers: one decode step fits the 12000-marker per-core buffers; un-drained replays overflow them (the
  test drains around every window and skips the wall loop under the profiler).
- `SparseMatmulDeviceOperation active=?/72`: the profiler does not record the number of active experts per block
  (tt-perf-report prints the DRAM / FLOP utilization only with `--active-experts`).
- Layer 3 decodes at position 1024 on the dense (flash-MLA) path with a random-seeded cache; the sparse path
  (position > 2050) is not wired on multi-row meshes and not profiled.

## Files

| directory | content |
|---|---|
| `layer4_kda_moe_B4/` | layer 4, B=4/row: `summary.md`, 8 x `dev<id>_r<r>c<c>_decode_perf_report.txt`, chip (0,0) summary CSV/PNG |
| `layer4_kda_moe_B32/` | layer 4, B=32/row |
| `layer3_dsa_moe_B4/` | layer 3, B=4/row |
| `layer3_dsa_moe_B32/` | layer 3, B=32/row |
