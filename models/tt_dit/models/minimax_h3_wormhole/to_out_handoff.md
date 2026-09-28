# MiniMax-H3 on Wormhole Galaxy: to_out AGMM optimization handoff (2026-09-25)

The to_out projection of the attention block runs as `all_gather_minimal_matmul_async` with the gated-residual
addcmul fused into its epilogue, and **that shipped form is unchanged**: nothing in this document has landed. This
is the map for whoever picks the op up next -- every lever proposed so far, whether it was run, what it measured or
is expected to gain, and whether it stays inside the current op (low risk) or replaces it (a different op, or kernel
rework). The per-op baseline write-up is [to_out.md](to_out.md); this file adds the 2026-09-25 experiments and the
protocol below.

## 0. How to evaluate a candidate (do all three, every time)

The op is 2.1% of the block, so an isolated-bench win has to be confirmed where it matters. For each lever, measure
baseline and candidate on the **same host, same session, same commit**, in this order:

1. **Isolated op, fast iteration.** `models/tt_dit/tests/models/minimax_h3/tools/transformer_op_mesh_bench.py`
   (host-timed, 10+ back-to-back calls, PCC / rel-RMSE against fp32 torch). Recipes in §4. Good for ranking
   variants, not for the final number.
2. **Tracy per-op breakdown of one transformer block**, and compare per op, not just the total:
   ```bash
   scripts/run_safe_pytest.sh --profile \
     "'models/tt_dit/tests/models/minimax_h3/test_transformer_minimax_h3.py::test_minimax_h3_transformer_block_perf[wormhole_b0-sp_sim1-test_prompt_text_tokens-15s_768p-4x8sp1tp0nl4_ring_is_fsdp1]'" \
     -s --timeout 3600            # prints "SAFE_PYTEST: PROFILER CSV: <csv>"
   python models/tt_dit/tests/models/minimax_h3/tools/block_profile_stats.py compare baseline=<csv> candidate=<csv>
   python models/tt_dit/tests/models/minimax_h3/tools/block_device_busy.py baseline=<csv> candidate=<csv>
   ```
   Read `to_out`'s row with §2 in mind: the "warm call" mean over devices includes ring waiting that is not the
   op's work. Compare the **first-call** to_out time, or the per-device minimum, between baseline and candidate,
   and check that the ops before and after it did not absorb what to_out gave up (`block_device_busy.py`'s union
   is the honest total).
3. **End-to-end generation, steady state.** The perf test's warm generation excludes kernel compilation, weight
   cache creation and the first-call costs; report its `ms/fwd` and denoise seconds:
   ```bash
   TT_DIT_CACHE_DIR=~/tt_dit_cache MINIMAX_H3_DIT_FSDP=1 RUN_VBENCH=0 scripts/run_safe_pytest.sh \
     "models/tt_dit/tests/models/minimax_h3/test_performance_minimax_h3.py::test_t2va_performance[wormhole_b0-4x8_WH-16x9_15s]" \
     -s --timeout 7200
   ```
   For a quicker steady-state A/B (10 scheduler steps, same prompt and seed):
   `H3_SWEEP_STEPS=10 ... test_parallel_sweep_minimax_h3.py::test_t2va_parallel_sweep[4x8_tp0_sp1]`. Run baseline
   and candidate back to back; the README's variance section puts run-to-run at ~0.1% for ms/fwd. Anything that
   changes numerics (fp32 dest, reduce order) also needs the CLIP gate from the full perf test (bar 33.0) and a
   look at the frames.

## 1. The op and where its time goes

| | |
|---|---|
| model call | `ColParallelLinear(7168, 5376)` with `addcmul_a = residual`, `addcmul_b = gate`, `attention_minimax_h3.py:151` (construction), `:640` (call); AGMM branch `layers/linear.py:436` |
| per device | M = 13664 rows, K = 7168 (K_local 1792 per device, gathered over the TP = 4 ring), N = 1344; weight N-sharded `[7168, 1344]` |
| config | bf16 HiFi2, fp32 dest, `packer_l1_acc`, `math_approx_mode`; 8x8 worker grid, blocking `grid_88_configs[(7168, 1344)]` (`utils/matmul.py:81`) = (8, 8, 6) subblock 2x2; 4 links, 2 workers per link |
| isolated | **5.29 ms device** (harness, real epilogue), 5.39 ms host-timed on the mesh bench (2026-09-25) |
| in the block | 5.2 ms on the first call on every device; **6.2 ms mean on the warm call** (§2) |
| roofline | compute 2.01 ms on 64 cores; fabric 1.47 ms at 4 links; DRAM 0.75 ms -- **compute-bound, not fabric-bound** |

Decomposition of the 5.3 ms (Tracy zones on the AGMM compute kernel, [to_out.md](to_out.md) §2):

| component | ms |
|---|---|
| FPU work at HiFi2 peak | 2.0 |
| grid padding (N 5.25 -> 6 tiles per core, M 54 -> 56) | 0.4 |
| 2x2 fp32 pipeline issue cost (47 vs 32 cycles per tile-MAC, same as ff1 / to_qkv) | 1.1 |
| **operand waits on the in-device store-and-forward relay** (`OPWAIT`, 25% of the K loop) | **1.1** |
| **two-pass addcmul epilogue** | **0.7** |

Why this AGMM waits and ff1 / to_qkv do not: N per core is 6 tiles, so each in0 byte relayed into a core feeds few
MACs; the kernel wants ~12.8 GB/s of in0 per core and the relay chain (in0 down a column of 8 cores, one semaphore
round trip per hop) delivers ~10. The inter-device fabric is not the limiter: at 4 links it has 0.5 ms of slack.

## 2. The 6.2 ms in the block is waiting, not work

Both block profiles (09-17 baseline `2026_09_17_21_33_20`, 09-24 all-on `2026_09_24_20_09_20`) show the same shape
for to_out (`INPUT_1 = [7168, 1344]` rows of the ops CSV, 32 devices x 2 calls):

| | call 1 | call 2 (the "warm" call the README table reports) |
|---|---|---|
| mean over devices | 5.20 ms | 6.22 ms |
| min .. max | 5.12 .. 5.31 | 5.12 .. 7.40 |

Call 2 is bimodal: devices 0-3 and 24-27 take 7.3-7.4 ms, the other 24 take 5.1-5.3. The op before to_out is an
FSDP weight all-gather; on a fast-to_out device it took 1.80 ms, on a slow-to_out device 0.31 ms. The sums agree
within 0.8 ms: device skew created upstream (the slow group spends 1.55 ms less in all-gathers over the iteration)
is absorbed by whichever ring collective comes next. With FSDP off (09-17 fsdp0 profile) the warm-call mean is
5.60 with the same 7.3 ms tail, so part of the skew is ring geometry, part FSDP traffic.

Consequences: (a) the op's own cost is 5.2-5.3 ms and that is the number to beat; (b) any replacement that is also a
collective (MM+RS, strided AGMM) will absorb the same skew in the block; (c) the block table's "mean over devices
for collectives" rule bakes skew into every collective's row -- compare first-call or per-device-minimum numbers.
The eight-device pattern (two mesh rows) is an FSDP all-gather asymmetry worth its own look; it is not a to_out fix.

## 3. The levers

### 3A. On the current op (least risk: no layout, cache or model change)

| # | lever | run? | result / expectation | effort, risk |
|---|---|---|---|---|
| A1 | **One-pass addcmul epilogue.** `add_bias_and_addcmul_block` (AGMM `device/kernels/compute.cpp:165`) makes two full passes over the fp32 intermediate through L1 because `unary_bcast_tile` did not work under fp32. Single-DST-pass form: `mul_tiles(interm, b)` -> DST0, scalar multiply only when scalar != 1.0, `copy_tile(a)` -> DST1, add, one pack. Reference: the single-device `dit_minimal_matmul_addcmul_fused` kernel costs 0.28 ms for the same math vs 0.87 here. | not run | **~ -0.35 ms**, no numerics change | kernel-only, a day; low risk. **Do this first**; it stacks with everything below and, if to_out ever moves to MM+RS, the same epilogue lives in the reduce-scatter's final write. |
| A2 | **fp32 dest off** (+ 4x2 subblock). The `MINIMAX_H3_MM_FP32_DEST` switch exists for ff1 and to_qkv (`attention_minimax_h3.py:218`, `transformer_block_minimax_h3.py:157`); to_out is not wired to it. | run (mesh bench, 2026-09-21) | 5.43 -> 5.24 ms (-3.5%); rel-RMSE 0.0056 -> 0.0086 | env switch + config; precision decision, needs the CLIP gate and frame check |
| A3 | **Blocking re-sweep with the real epilogue.** The 09-17 sweep (352 combos, best 0.5% better than shipped) used the harness's `plain` use case: no addcmul, `math_approx_mode=False`. | not run with the epilogue | <= 1% expected | `sweep_mm_block_sizes.py --use-case to_out --shape 13664,7168,1344`; low risk |
| A4 | Relay prefetch (`tools/agmm_relay_prefetch.patch`: request block k+1 before waiting on the downstream hop) | run | no gain (5.52 vs 5.43, within noise) | rejected |
| A5 | Fewer / larger hops: K_block 14; fewer in1 re-reads: M_block 16 | run | no gain (5.43 -> 5.43; 5.22 vs 5.23 fp32 off) | rejected; in0, not in1, is the delivered volume |
| A6 | 2 links instead of 4 | not possible | the factory asserts the 8-wide in0 axis forms exactly `num_links` groups of `num_workers_per_link`; 2 links needs 4 workers per link, and the single-row mux scheme asserts 2. Fabric is not the limiter anyway. | -- |
| A7 | **in0 (and in1) multicast inside the AGMM** instead of the store-and-forward relay: the chain's injector writes each block to every other core of its row / column in one NoC multicast, receivers signal the injector (`dm_in0_sender.cpp` / `dm_in1_sender_out.cpp` under `IN0_MCAST` / `IN1_MCAST`, rectangle and receiver count from the factory). Two companions found by zoning the dataflow kernels: deeper operand CBs (`TT_AGMM_CB_DEPTH`, default 2) and letting the injector cores defer their output write like every other core (`TT_AGMM_INJECTOR_DEFER`, stagger shifted one row). | **built and run 2026-09-25** (§7), opt-in env switches, bit-identical numerics | to_out on the shipped 8x8 (8,8,6): 5.38 -> **4.89 ms (-9.1%)** with all three; on 8x7: **4.71 (-12.5%)**, fp32 dest off 4x2 **4.64 (-13.8%)**. ff1 15.57 -> 14.95 (-4.0%), to_qkv unchanged. Multicast alone is a wash (5.37) and in1 alone 5.35: each chain masked the other, both together 5.21. | kernel + factory, done; needs the block-level confirmation (§7) and a decision on how the model turns it on (env today) |
| A8 | `num_buffers_per_channel` (48 today), workers per link (2) | not swept for to_out | small | knobs on the existing call (`linear.py:436`) |

### 3B. A different op, or major kernel rework

| # | lever | run? | result / expectation | effort, risk |
|---|---|---|---|---|
| B1 | **Row-parallel form: local matmul on K_local + fused reduce-scatter** (`RowParallelLinear.forward_fused_addcmul`, `linear.py:659`, the op ff2 ships: `minimal_matmul_strided_reduce_scatter_async`). Same FLOPs (M x 1792 x 5376 == M x 7168 x 1344), same fabric volume (147 MB of partials out instead of gathered in), but 21 N tiles per core instead of 6 so the relay problem disappears. | **run 2026-09-25** (bench only, §5) | best **5.23 vs 5.39 ms (-3.0%)** at equal precision; 5.10 with fp32 dest off. The reduce-scatter moves the same 147 MB as ff2's regardless of K: at 1 RS worker per direction it alone costs ~6.6 ms, so the matmul grid has to give up a row (8x6, 2 workers per direction, 4 links) to feed 16 workers, and the two halves then run neck and neck. `chunk_width_in_mm_blocks = 0` (one RS chunk per M block) was worth 0.2 ms. | model change: K-sharded weight (`mesh_axes` flip, new cache key), new forward path in the attention module, a `fused_mmrs_configs` entry (`utils/matmul.py:986`) to sweep; numerics change (bf16 partials summed on the ring; ff2 accepted this at PCC 1.0000). **Not worth it for 0.15 ms unless A7 is ruled out.** Open follow-up: ff2's shipped entry uses chunk width 1 -- try 0 there. |
| B2 | **Strided AGMM** (`strided_all_gather_minimal_matmul_async`): gather workers on the rows above the matmul grid write the remote K slices into a persistent DRAM buffer, in the matmul's consumption order, and the matmul reads in0 from it. The matmul's own in0 / in1 dataflow is the **same store-and-forward relay** as the AGMM's (`minimal_matmul/device/kernels/fabric_bound_dm_in0_sender.cpp`: the injector reads a block, every core forwards it one hop behind a semaphore handshake), so this lever changes the gather, not the operand waits. Wired into `ColParallelLinear` behind `get_fabric_agmm_config` (`utils/matmul.py:1253`); the addcmul form is what the LTX to_out already runs. | **run 2026-09-25** (§6) | Best at the model's 4 KB fabric payload: **5.44 vs 5.37 ms (+1.3%)**, 8x7 matmul grid, 4 links x 1 gather worker per direction, (8,4,6) 2x2 (K_block 7: 5.57, 8: 5.86; smaller gather chunks pipeline better). At the Wormhole payload cap (7616 B, 3 tiles per packet) the same form is **5.16 ms (-3.9%)**, but that payload slows the shipped AGMM to 5.98 (+11%, with 48, 24 or 16 channel buffers alike), so as a mesh-wide setting it is a net loss for the block unless the shipped AGMM's packetization is retuned or every AGMM moves. The op is gather-bound: the 16-core gather zone delivers 26 GB/s (4 KB) / 32 GB/s (7616 B) against the 100 GB/s of 4 links, ~3 GB/s per worker stream, and the 8x7 matmul alone is 4.24 ms. | measured and **parked**: nothing to land at the model's payload. The one open thread is the strided gather's per-worker rate (kernel work in `strided_all_gather_async/device/kernels/minimal_default_reader.cpp` / `_writer.cpp`: 2-3 tile packets, a 3-packet CB, one DRAM read per packet); at 45 GB/s from 16 cores the 8x7 form would be ~4.5 ms. |
| B3 | **FSDP skew** (§2): reorder or throttle the FSDP weight prefetch so the two outer mesh rows do not arrive early at every ring op | not run | up to ~1 ms of the *block's* to_out row, but it is a redistribution unless the block's critical path shortens; judge with `block_device_busy.py` | block-level scheduling, not a to_out change |

Recommended order: **A7 first, it is built** (§7: -9% on the shipped grid, -12.5% on 8x7, numerics untouched), then
A1, then A2 as a precision decision (it stacks: 4.64 on 8x7). B1 and B2 are both measured and parked (-3.0% and
+1.3% at the model's fabric payload); B2's second data point is that the strided gather, not the matmul, is what
needs work before that op can pay.

## 4. Recipes

```bash
# isolated baseline (shipped op, fused addcmul; PCC vs fp32 torch on 2048 rows)
python models/tt_dit/tests/models/minimax_h3/tools/transformer_op_mesh_bench.py --op to_out --iters 20
# fp32 dest off (A2)
python models/tt_dit/tests/models/minimax_h3/tools/transformer_op_mesh_bench.py --op to_out --fp32-dest 0 --blocks 8,8,6,4,2
# blocking re-sweep with the real epilogue (A3); shape ids end in _agmm_to_out
python models/tt_dit/utils/sweep_mm_block_sizes.py --device-config wh_4x8_ring --use-case to_out --shape 13664,7168,1344

# row-parallel form (B1): the bench-only spec `to_out_mmrs` (tools/minimax_h3_ops.py:204) runs the ff2 path at
# to_out's K/N; --mm-grid is the matmul grid, the reduce-scatter takes the rows above it
python models/tt_dit/tests/models/minimax_h3/tools/transformer_op_mesh_bench.py --op to_out_mmrs --fused \
  --mm-grid 8x6 --rs-workers 2 --window 2 --chunk-width 0 --blocks 6,7,8,2,2 --iters 20        # 5.23 ms
python models/tt_dit/tests/models/minimax_h3/tools/transformer_op_mesh_bench.py --op to_out_mmrs --fused \
  --mm-grid 8x6 --rs-workers 2 --window 2 --chunk-width 0 --blocks 6,7,8,2,2 --fp32-dest 0     # 5.10 ms

# strided AGMM (B2): --sagmm runs strided_all_gather_minimal_matmul_async the way the model's fabric path calls it;
# --mm-grid is the matmul grid, the gather takes the rows above it (--num-links x (--ag-workers + 1) x 2 cores)
python models/tt_dit/tests/models/minimax_h3/tools/transformer_op_mesh_bench.py --op to_out --sagmm \
  --mm-grid 8x7 --num-links 4 --ag-workers 1 --blocks 8,4,6,2,2 --iters 20                       # 5.44 ms
python models/tt_dit/tests/models/minimax_h3/tools/transformer_op_mesh_bench.py --op to_out --sagmm \
  --mm-grid 8x7 --num-links 4 --ag-workers 1 --blocks 8,4,6,2,2 --iters 20 --fabric-payload 7616  # 5.16 ms
python models/tt_dit/tests/models/minimax_h3/tools/transformer_op_mesh_bench.py --op to_out --iters 20 \
  --fabric-payload 7616                                                                            # shipped op there: 5.98 ms

# zones on the AGMM compute kernel (where the 5.3 ms goes): tools/agmm_compute_zones.py apply block|opwait|sampled,
# then the harness on shape 13664_7168_1344_8x8_agmm_to_out; revert the kernel afterwards (recipe in to_out.md §6)
```

Gotchas: the strided AGMM transposes its grid for every H3 shape (M > N), so N spreads over the matmul grid's
rows: 6 tiles per core on 8x7 (exact), 7 on 8x6 (a prime N_block, so no 2x2 subblock), 9 on 8x5; N_block must be the
whole per-core width or in0 is relayed once per N block. Its matmul-signal aggregators need 2 cores beyond the
mux/worker set and fall back (with a warning) to reader-signalled matmul when they do not fit. (8,14,6) overflows L1
on 8x7. The fused MM+RS rejects an M_block that does not divide by subblock_h; an L1 window shard that clashes with
the CB region fails at program build ("Statically allocated circular buffers ... clash with L1 buffers") -- M_block 8
and K_block 14 both did on the 8x6 / 8x7 grids with window 2. Rebuild the host library after any factory change
(`ninja -C build install`, never `build_metal.sh`); kernels are JIT-compiled per run.

## 5. Raw numbers, 2026-09-25, this galaxy (`UF-EV-B12-GWH02`), host-timed mesh bench, all passing the PCC bar

| variant | ms per call |
|---|---|
| **shipped AGMM**, 8x8, (8,8,6) 2x2, fused addcmul, 20 iterations | **5.39** |
| MM+RS fused, 8x7, 4 links, 1 RS worker per direction, (6,7,8) 2x2, window 2, chunk width 1 | 6.62 |
| same, (6,8,8) | 6.64 |
| same blocking, DRAM handoff (no window) | 8.05 |
| MM+RS fused, 8x7, 2 links, 3 workers | 6.15 |
| MM+RS fused, 8x6, 2 links, 5 workers | 6.04 |
| MM+RS fused, 8x6, 4 links, 2 workers, chunk width 1 | 5.44 |
| same, 4 buffers per channel | 5.47 |
| same, M_block 4 | 5.88 |
| same, chunk width 2 | 5.35 |
| **same, chunk width 0**, 20 iterations | **5.23** |
| same, chunk width 0, (6,8,8) | 5.28 |
| same, chunk width 0, 8 buffers per channel | 5.24 |
| same, chunk width 0, fp32 dest off, 2x2 (rel-RMSE 0.010 vs 0.005) | 5.10 |
| same, chunk width 0, fp32 dest off, 2x4 | 5.18 |
| MM+RS fused, 8x5, 4 links, 3 workers (matmul-bound) | 5.90 |
| MM+RS fused, 8x7, 1 worker, chunk width 0 | 6.25 |
| MM+RS unfused: minimal_matmul 8x9 + tuned reduce_scatter + addcmul | 7.03 |

## 6. Raw numbers, 2026-09-25 (later the same day): the strided AGMM (B2), this galaxy, host-timed mesh bench

Shipped op in the same session: **5.37 ms** at the model's 4096 B fabric payload, **5.98** at 7616 B (48, 24 or 16
channel buffers: 5.98 / 5.98 / 6.03). All strided rows pass the PCC bar with the shipped op's numerics (rel-RMSE
0.00559 at fp32 dest; 0.0085 with it off). The compute grid is 8x9, so the gather zone above an 8xR matmul grid is
8 x (9 - R) cores; "aggregators" = the 2 matmul-signal aggregator cores, used when they fit.

| matmul grid | gather zone | blocks | payload | ms |
|---|---|---|---|---|
| 8x7 | 2 links x 1 worker (8 cores, aggregators) | (8,8,6) 2x2 | 4096 | 8.98 (32 buffers/channel: 8.97) |
| 8x7 | 1 link x 2 workers (aggregators) | (8,8,6) 2x2 | 4096 | 10.08 |
| 8x7 | 1 link x 3 workers | (8,8,6) 2x2 | 4096 | 10.19 |
| 8x7 | 2 links x 3 workers (16 cores) | (8,8,6) 2x2 | 4096 | 6.27 |
| 8x7 | 2 links x 3 workers | (6,8,6) 2x2 | 4096 | 6.65 |
| 8x7 | 2 links x 3 workers | (8,7,6) 2x2 | 4096 | 6.09 |
| 8x7 | 2 links x 3 workers | (8,14,6) 2x2 | 4096 | L1 overflow (1.58 MB of CBs) |
| 8x7 | 2 links x 2 workers (aggregators) | (8,8,6) 2x2 | 4096 | 6.23 |
| 8x7 | 4 links x 1 worker (16 cores) | (8,8,6) 2x2 | 4096 | 5.86 |
| 8x7 | 4 links x 1 worker | (8,7,6) 2x2 | 4096 | 5.57 |
| **8x7** | **4 links x 1 worker** | **(8,4,6) 2x2** | **4096** | **5.44** |
| 8x7 | 4 links x 1 worker | (8,8,6) 2x2 | 7616 | 5.42 / 5.41 (16 buffers: 5.44; fp32 dest off: 5.28) |
| 8x7 | 4 links x 1 worker | (8,7,6) 2x2 | 7616 | 5.20 (fp32 dest off: 5.19) |
| **8x7** | **4 links x 1 worker** | **(8,4,6) 2x2** | **7616** | **5.16** |
| 8x6 | 2 links x 3 workers | (8,8,6) 2x2 (2 N blocks/core) | 4096 | 7.14 |
| 8x6 | 4 links x 1 worker | (8,8,6) 2x2 | 4096 | 7.53 |
| 8x6 | 2 links x 2 workers (aggregators) | (8,8,6) 2x2 | 4096 | 7.40 |
| 8x6 | 2 links x 3 workers | (8,8,7) 4x1 / 2x1 / (6,8,7) 2x1 | 4096 | 6.64 / 6.55 / 7.21 |
| 8x6 | 2 links x 3 workers | (8,8,7) 1x7, fp32 dest off | 4096 | 6.53 |
| 8x6 | 4 links x 2 workers (24 cores) | (8,8,7) 2x1 | 4096 | 6.06 |
| 8x5 | 4 links x 2 workers | (8,8,6) 2x2 (2 N blocks/core) | 4096 | 7.01 |
| 8x5 | 4 links x 2 workers | (6,8,9) 2x1 / 2x3 fp32 off | 4096 | 7.61 / 6.67 |
| 8x4 | 4 links x 3 workers (32 cores) | (8,8,6) 2x2 | 4096 | 7.70 |

Where the time goes, same session:

| component alone | ms | note |
|---|---|---|
| strided gather only, 2 links x 1 worker | 11.07 | 13 GB/s of remote K per device |
| strided gather only, 2 links x 2 workers | 5.87 | 25 GB/s |
| strided gather only, 2 links x 3 workers | 4.65 | 32 GB/s |
| strided gather only, 4 links x 1 worker | 5.60 (4096) / 4.64 (7616) / 4.59 (7616, K_block 7) | 26 / 32 GB/s; the 8x7 zone |
| strided gather only, 4 links x 2 workers | 3.26 (4096) / 3.38 (7616) | 45 GB/s; needs 24 cores = the 8x6 zone |
| matmul only (`dit_minimal_matmul_addcmul_fused`, gathered input, 8x7, (8,8,6) 2x2) | 4.24 | 56 cores, no padding |
| same on 8x8 | 4.40 | slower than 8x7: N pads 5.25 -> 6 tiles per core, one more relay hop |
| plain matmul, 8x7 | 4.00 | the addcmul epilogue costs 0.24 here |

Reading: ~3 GB/s per gather worker stream (2-3 tile packets, one DRAM read per packet, a 3-packet CB) is the limiter;
the fused 8x7 op sits 0.6-1.0 ms above whichever of its two halves is longer. The 8x7 matmul floor (4.24) is below
the shipped op's 8x8 floor (4.40): the shipped grid's N padding and eighth relay hop cost more than the row it
gains. That does not carry into the shipped AGMM, though: on 8x7 it runs 5.41 (8,8,6) / 5.55 (6,8,6) against 5.40
on 8x8 (`--mm-grid 8x7`), because the op is delivery-bound and the padded N tiles ride along for free. The rule for
this block: with the grid transposed (M over the 8 columns, N over the rows) a core holds ceil(N_tiles / rows) N
tiles, so 8x7 costs nothing only when ceil(Nt/7) == ceil(Nt/8). That holds for to_out (Nt 42: 6 either way) and for
no other AGMM here (to_qkv Nt 168: 21 vs 24; ff1 Nt 224: 28 vs 32, both +14% work per core on 8x7). Use 8x7 for
to_out only when the freed row buys something, as the strided gather's 16-core zone did. The standalone gather and matmul
floors came from two scratch scripts (`strided_all_gather_async` alone with the fused op's chunking; the single-device
addcmul matmul on a replicated gathered input); the fused rows are `--sagmm` on the mesh bench (§4).

## 7. A7 built: multicast delivery, deeper operand CBs, injector write deferral (2026-09-25, later)

All three are opt-in environment switches read by the AGMM program factory
(`all_gather_minimal_matmul_async_program_factory.cpp`) when it builds the program, so they apply to every AGMM of
the process (to_qkv, to_out, ff1) and are part of the kernel hash:

| switch | what it does |
|---|---|
| `TT_AGMM_IN0_MCAST=1` | in0 chain: the injector (row 0 of its column on the transposed grid) multicasts each block to rows 1..R-1 once all of them have reserved the CB slot; receivers `up` the injector's semaphore instead of their predecessor's and forward nothing. The fabric senders in the chain are untouched (every core still holds the block at the same CB address). `dm_in0_sender.cpp` under `IN0_MCAST`. |
| `TT_AGMM_IN1_MCAST=1` | the same for the in1 chain along each row (`dm_in1_sender_out.cpp` under `IN1_MCAST`; a partial N block goes row by row like the relay). The FSDP fabric relay reads from its own saved address, unaffected. |
| `TT_AGMM_CB_DEPTH=<n>` or `auto` | in0 / in1 CB depth (default 2). `auto` picks 4, 3 or 2 from the blocking so the CBs stay under 1.35 MB: to_out (8,8,6) gets 4, ff1 (8,7,10) 3, to_qkv (8,7,12) 2 (3 overflows L1 at 1.56 MB). |
| `TT_AGMM_INJECTOR_DEFER=1` | injector cores defer their output write like every other core (today `defer_write && !is_injector_core` makes them write synchronously at the M block end, so the row's in1 feed waits for the ~110 us epilogue plus the write); the host shifts the write stagger by one row so no core writes at K block 0. |

### Isolated op (mesh bench, host-timed, 20 calls, this galaxy; every row bit-identical to the shipped op)

| variant (to_out, fp32 dest unless noted) | 8x8 | 8x7 |
|---|---|---|
| shipped relay, (8,8,6) 2x2 | **5.38-5.40** | 5.41 |
| in0 multicast only | 5.37 | 5.27 |
| in1 multicast only | 5.35 | |
| in0 + in1 multicast | 5.21 | 5.09 |
| multicast, (12,8,6) / (6,8,6) / (8,7,6) / (8,14,6) / (6,14,6) / (12,7,6) / (12,4,6) | 5.11 / | 5.04 / 5.19 / 5.24 / 5.33 / 5.67 / 5.21 / 5.52 |
| multicast, (14,8,6) | 5.11 | |
| multicast + CB depth 3, (8,8,6) | 5.18 | 5.00 |
| multicast + CB depth 4, (8,8,6) | 5.06 | 4.93 |
| multicast + depth 3 (10,8,6) / depth 5 (6,8,6) / depth 4 (8,7,6) | | 4.94 / 4.94 / 5.06 |
| multicast + depth 3 (12,8,6) | | L1 overflow (1.59 MB) |
| relay + CB depth 3 | 5.32 | |
| relay + injector defer | 5.39 | |
| multicast + depth 2 + injector defer, (12,8,6) | | 5.02 |
| **multicast + depth 4 + injector defer, (8,8,6)** | **4.89** | **4.71** |
| same, fp32 dest off, 4x2 | | **4.64** (rel-RMSE 0.0086) |
| multicast + depth 4, fp32 dest off 4x2 (no defer) | | 4.83 |
| ff1 (8,7,10) 8x8: relay / multicast / multicast + depth 3 + defer | 15.57 / 15.22 / **14.95** | |
| to_qkv (8,7,12) 8x8: relay / multicast / multicast + defer (depth 3 overflows) | 11.04 / 11.05 / 11.01 | |

### Where the waits were (device zones, harness shape `13664_7168_1344_8x8_agmm_to_out`, (8,8,6))

Compute kernel (`agmm_compute_zones.py apply opwait`; its SwiGLU anchor was updated to the current call), per core:

| | relay | multicast |
|---|---|---|
| `TRISC-KERNEL` | 5,236 us | 5,001 us |
| `KLOOP` per M block | 659 us | 602 us |
| `OPWAIT` per K step, mean / p50 / p90 / p99 | 7.1 / 6.0 / 11.1 / 74.6 us | 5.0 / 3.0 / 10.3 / 59.6 us |

The wait by K index (M blocks 1-3, mean over 64 cores) is the actionable part: **K step 0 of every M block waits
42-52 us under both schemes** (the injector's synchronous output write behind the epilogue: the deferral switch),
the relay then waits 3-8 us on every step (the chain), the multicast 1-3 us in the first half of the K loop rising to
6-11 us in the second half and ~0.3 us on the last four (a bandwidth margin, absorbed by CB depth 3-4).

in0 injector zones (temporary, on M block 2 of the same run; per K step): DRAM read of the 128 KB block **5.8 us**,
ring-arrival wait (`compute_actual_k_block`) **0.2 us** -- the fabric is never the limiter -- and 7.2-7.3 us waiting
for the next hop (relay) or for all seven receivers (multicast) to free a slot, which is back-pressure, not loss.

### Block-level confirmation (protocol §0 step 2)

One block, `test_minimax_h3_transformer_block_perf[...4x8sp1tp0nl4_ring_is_fsdp1]` under Tracy, same session,
model grid and blockings (to_out 8x8 (8,8,6); every AGMM gets the switches, depth `auto`), FSDP on:

| | baseline | all three switches | delta |
|---|---|---|---|
| `AllGatherMinimalMatmulAsyncOp`, 3 calls (to_qkv + to_out + ff1), merged per `block_profile_stats.py` | 30.86 ms | 29.42 ms | **-1.44 ms** |
| every other op | unchanged within 0.2 ms (RMSNorm +0.17, AllBroadcast +0.07 absorb skew) | | |
| device only, whole block | 232.64 | 231.29 | -1.35 (-0.6%) |
| device-busy union (`block_device_busy.py`) | 229.87 | 228.58 | **-1.29 ms** |

The isolated numbers predicted -0.5 (to_out on 8x8) - 0.6 (ff1) - 0 (to_qkv) = -1.1 ms, so nothing was absorbed
upstream. The FSDP-fused in1 chain (weight gather threaded through the in1 relay) ran under `IN1_MCAST` without
incident. Profiles: `generated/profiler/reports/2026_09_25_23_49_52` (baseline) and `2026_09_25_23_51_37`.

### End to end (protocol §0 step 3), 2026-09-28

`test_t2va_performance[wormhole_b0-4x8_WH-16x9_15s]`, FSDP on, all four switches with `TT_AGMM_CB_DEPTH=auto`,
steady state over 48 denoise steps, same host and session:

| run | ms per step | CLIP prompt alignment (bar 33.0) |
|---|---|---|
| baseline | 12054 | 36.58 |
| baseline, second run | 12058 | |
| **A7** | **11993** | 36.67 |

-61 ms per step, which is the block-level -1.3 ms x 50 blocks. The videos are in `~/h3_t2va_artifacts/a7_2026_09_28`
and `baseline_2026_09_28` / `baseline2_2026_09_28`. Frames are **not** a bit-identity check at this level: two baseline
runs of the same commit differ by a mean 3.8 / 255 per pixel (max 248), and A7 vs baseline differs by 4.8, the same
band. The op-level comparison (PCC 0.9999889, rel-RMSE 0.00559 on every variant) is where the identical-numerics
claim is made; the pipeline's run-to-run spread comes from elsewhere (ring reduce order is the usual suspect) and
predates A7.

### Recipes

```bash
# to_out, shipped grid and blocking, everything on (numerics identical to the shipped op)
TT_AGMM_IN0_MCAST=1 TT_AGMM_IN1_MCAST=1 TT_AGMM_INJECTOR_DEFER=1 TT_AGMM_CB_DEPTH=4 \
  python models/tt_dit/tests/models/minimax_h3/tools/transformer_op_mesh_bench.py --op to_out --iters 20   # 4.89 ms
# same on 8x7 (the bench's --mm-grid now also applies to the shipped AGMM)
... --mm-grid 8x7                                                                                          # 4.71 ms
... --mm-grid 8x7 --blocks 8,8,6,4,2 --fp32-dest 0                                                          # 4.64 ms
# block profile with every AGMM on (depth from the L1 budget)
TT_AGMM_IN0_MCAST=1 TT_AGMM_IN1_MCAST=1 TT_AGMM_INJECTOR_DEFER=1 TT_AGMM_CB_DEPTH=auto scripts/run_safe_pytest.sh --profile ...
```

What is left on this op after A7: the one-pass epilogue (A1, ~0.3 ms, the K step 0 wait shrinks with it too), the
8x7 grid for to_out (0.18 ms, needs a `core_grid` for to_out in the attention module), fp32 dest off (A2, 0.07 ms
here, a precision decision), and the remaining 1-3 us per K step of multicast wait, which a prefetching injector
(issue the next DRAM read before waiting on the receivers) would take out.
