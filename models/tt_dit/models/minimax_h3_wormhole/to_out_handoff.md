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
| A7 | **in0 multicast inside the AGMM** instead of the store-and-forward relay (each hop re-writes 128 KB per iteration today) | not run | **up to -1.1 ms** (the whole `OPWAIT`), keeps all 64 cores | dataflow rework of the AGMM in0 sender / relay kernels (`all_gather_minimal_matmul_async/device/kernels/`); highest ceiling inside the op, highest risk of the 3A set |
| A8 | `num_buffers_per_channel` (48 today), workers per link (2) | not swept for to_out | small | knobs on the existing call (`linear.py:436`) |

### 3B. A different op, or major kernel rework

| # | lever | run? | result / expectation | effort, risk |
|---|---|---|---|---|
| B1 | **Row-parallel form: local matmul on K_local + fused reduce-scatter** (`RowParallelLinear.forward_fused_addcmul`, `linear.py:659`, the op ff2 ships: `minimal_matmul_strided_reduce_scatter_async`). Same FLOPs (M x 1792 x 5376 == M x 7168 x 1344), same fabric volume (147 MB of partials out instead of gathered in), but 21 N tiles per core instead of 6 so the relay problem disappears. | **run 2026-09-25** (bench only, §5) | best **5.23 vs 5.39 ms (-3.0%)** at equal precision; 5.10 with fp32 dest off. The reduce-scatter moves the same 147 MB as ff2's regardless of K: at 1 RS worker per direction it alone costs ~6.6 ms, so the matmul grid has to give up a row (8x6, 2 workers per direction, 4 links) to feed 16 workers, and the two halves then run neck and neck. `chunk_width_in_mm_blocks = 0` (one RS chunk per M block) was worth 0.2 ms. | model change: K-sharded weight (`mesh_axes` flip, new cache key), new forward path in the attention module, a `fused_mmrs_configs` entry (`utils/matmul.py:986`) to sweep; numerics change (bf16 partials summed on the ring; ff2 accepted this at PCC 1.0000). **Not worth it for 0.15 ms unless A7 is ruled out.** Open follow-up: ff2's shipped entry uses chunk width 1 -- try 0 there. |
| B2 | **Strided AGMM** (`strided_all_gather_minimal_matmul_async`, the "fabric-bound" op: gather workers write remote K slices into a DRAM buffer, the matmul reads in0 from DRAM with NoC multicast, no relay). Wired into `ColParallelLinear` behind `get_fabric_agmm_config` (`utils/matmul.py:1253`); supports the fused addcmul. | not run | estimate -0.9 ms at 2 links on an 8x7 matmul grid if the relay waits vanish and DRAM + multicast keep up; ~0 at 4 links on 8x6 (the gather workers cost `num_links x (workers + 1) x 2` cores plus 2 aggregators, in whole rows). | config plus a bench path mirroring `linear.py:282`; the only tuned entries are Blackhole 12x10 (`fabric_agmm_configs`, `utils/matmul.py:1219`); Wormhole nightly coverage is 1 link only; a 196 MB persistent DRAM buffer per device (ping-ponged) on the 12 GB part. Medium risk; an afternoon to measure. |
| B3 | **FSDP skew** (§2): reorder or throttle the FSDP weight prefetch so the two outer mesh rows do not arrive early at every ring op | not run | up to ~1 ms of the *block's* to_out row, but it is a redistribution unless the block's critical path shortens; judge with `block_device_busy.py` | block-level scheduling, not a to_out change |

Recommended order: A1, then A2 as a precision decision, then A7 if a bigger win is wanted; B2 is the cheap
experiment if someone wants a second data point on delivery; B1 is measured and parked.

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

# zones on the AGMM compute kernel (where the 5.3 ms goes): tools/agmm_compute_zones.py apply block|opwait|sampled,
# then the harness on shape 13664_7168_1344_8x8_agmm_to_out; revert the kernel afterwards (recipe in to_out.md §6)
```

Gotchas: the fused MM+RS rejects an M_block that does not divide by subblock_h; an L1 window shard that clashes with
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
