# MiniMax-H3 on Wormhole Galaxy: ff2 — matmul, reduce-scatter and the gated residual, fused

The second feed-forward linear, `RowParallelLinear(14336, 5376)`, followed by the ring reduce-scatter and the block's
gated residual `addcmul`. Measured 2026-09-17 (blocking sweep, numerics), 2026-09-21 (mesh bench) and 2026-09-22/23
(fused MM/RS re-evaluation, reduce-scatter hyperparameters, fused blocking sweep, landing) on the 4x8 Wormhole Galaxy.
Block-level context: [README.md](README.md).

## 1. The op and the baseline

| | |
|---|---|
| model call | `ParallelFeedForward.ff2 = RowParallelLinear(14336, 5376, bias=False)`. Unfused (`RowParallelLinear.forward`, `models/tt_dit/layers/linear.py`): `minimal_matmul` on the full grid, `ccl_manager.reduce_scatter`, then `ttnn.addcmul(residual, ff_out, gate)` in `transformer_block_minimax_h3.py`. Fused (`RowParallelLinear.forward_fused_addcmul`): one `minimal_matmul_strided_reduce_scatter_async`, taken when `has_mmrs_config` (`models/tt_dit/models/transformers/minimax_h3/mmrs_config.py`) resolves a blocking for the device grid |
| shape per device | M = 13664, K = 3584 (already per-device, no gather), N = 5376 before the reduce-scatter, 1344 after |
| config | bf16, HiFi2, `fp32_dest_acc_en = True`, `packer_l1_acc = True`, `math_approx_mode = True` (the block's shared matmul config) |
| unfused blocking | full 8x9 = 72 cores (grid transposed: M over the 8 columns = 54 tiles/core, N over the 9 rows = 19), `grid_89_configs[(13664, 3584, 5376)] = (8, 7, 10)` subblock 2x2, landed 2026-09-17 |
| **fused blocking (landed 2026-09-23)** | `fused_mmrs_configs[CoreCoord(8, 9)][(13664, 3584, 5376)] = FusedMMRSConfig(CoreCoord(8, 7), 6, 7, 8, 2, 2, None, 1)` (`models/tt_dit/utils/matmul.py`): matmul on 8x7 = 56 cores, never transposed (M over 7 rows = 61 tiles/core, N over 8 columns = 21, no padding), reduce-scatter on the 16 cores above it = 4 links x 2 directions x (1 worker + 1 mux), 2-block rolling L1 window for the matmul output, addcmul at the reduce-scatter's final write |
| roofline | matmul compute **3.57 ms** on 72 cores, 4.59 ms on 56 (0.53 TFLOP at 2048 FLOP/cycle/core, 1.0 GHz); reduce-scatter fabric 1.10 ms ((R-1)/R x 146.9 MB over 2 x 4 links x 12.5 GB/s); reduce-scatter worker data movement at 1 worker/direction 6.6 ms (§3.5) |
| unfused baseline (2026-09-17 block) | `MinimalMatmulDeviceOperation` 6.77 ms, `ReduceScatterMinimalAsync` 2.78 ms, the addcmul inside the 2.81 ms "remaining ops" bucket |
| mesh bench, host-timed, 10 back-to-back calls (`transformer_op_mesh_bench.py --op ff2`, 2026-09-22) | MM 7.45 ms; `--with-rs` 9.99; `--with-rs --with-addcmul` **10.70** (the whole ff2 tail the block ran); the same with the reduce-scatter tuned (`--rs-workers 4 --rs-chunks 32 --rs-buffers 4`) 10.17 |
| **fused, same bench** | `--fused --mm-grid 8x7 --window 2 --blocks 6,7,8,2,2`: **8.94 ms** (8.88 for the swept optimum (6,8,8), which does not fit the block's L1, §3.4); pcc 0.99993 vs fp32 torch on every device; 2000 back-to-back calls with no hang |
| **fused, in the block** (Tracy, `test_minimax_h3_transformer_block_perf` 15 s fsdp1, 2026-09-23) | one `MinimalMatmulStridedReduceScatterAsync` per layer, **8.61 ms** device-side, replacing `MinimalMatmul` 6.90 + `ReduceScatterMinimalAsync` 2.79 + `Ternary` 0.64 = 10.33 ms: **-1.88 ms per block device-only (244.45 -> 242.57, -0.8%)**, -2.96 ms with op gaps; every other op within noise (SDPA +0.03, the three AGMMs -0.30) |
| **fused, end to end** (10-step same-host A/B, `test_t2va_parallel_sweep[4x8_tp0_sp1]`, 15 s / 768P / 16:9, FSDP on, seed 0, one run each, 2026-09-23) | **12135 ms/fwd fused vs 12301 unfused: -166 ms/fwd (-1.35%)**; denoise 109.2 vs 110.7 s over 9 forwards. Video PCC fused vs unfused 0.836, mean abs diff 20/255, audio PCC 0.979 -- the same regime as the README's TP8/SP4 run (0.907 / 0.972, "what a changed bf16 reduction order looks like after 9 sampling steps"). CLIP prompt alignment (`clip_prompt_alignment`, 8 frames): fused **35.58** (min 34.54), unfused 35.71 (min 34.52); bar 33.0; the README's 10-step reference on this host 35.88 |
| numerics | unfused (8,7,10): pcc 1.0000000 vs torch fp32 on one device. Fused: pcc 0.9999323 / rel-RMSE 0.00474 on the full fused output (reduce-scattered sum with the addcmul) on all 4 devices, the same figures as every other fused configuration measured, so the residual is the bf16 partial-sum path, not the blocking |

Per layer per device the fused op saves **1.76 ms** against the path the model ran (10.70 -> 8.94), 1.23 ms against the
best tuned unfused path; over 50 layers that is **-88 ms/fwd (-0.73%)** / -62 ms/fwd on the 12.0 s forward.

## 2. Why the fused op was off, and why that was wrong

`eab3dfbd599` (2026-09-16) gated Wormhole off the fused path because every ff2 was falling to the warned default
config (8x7 grid, M_block 2, subblock 1x1) -- the best explanation for the mid-denoise hang
(`../MiniMaxH3_wormhole_hang.md`) and a measured 2-4% end-to-end loss. The 2026-09-17 follow-up swept the fused op at
three matmul grids and concluded "all worse than unfused; stays disabled". That sweep had four problems, found 2026-09-22:

1. It ran at **M = 4736** (the 5 s shape), not the 13664 the model is tuned for.
2. It compared the fused total against the **matmul alone** (2373 us), leaving the reduce-scatter and the addcmul out
   of the unfused side. At M = 13664 those are 2.54 + 0.71 ms host-timed -- a third of the unfused total.
3. Its candidate blockings were generated for the **transposed** grid (`get_per_core_dims` assumed M over columns),
   but the fused op never transposes: on 8x7 a core holds 61 x 21 tiles, not 54 x 24, so N_block 7 was never generated
   and the L1 filter was off.
4. The runner passed a **bias** the model does not have, and windowed every combo with >= 2 M blocks per core whether or
   not the window shard fit beside the CBs.

The 2-4% gain from `eab3dfbd599` came from leaving a 1x1-subblock, M_block-2 blocking, not from unfusing.

## 3. Experiments (2026-09-22/23)

All host-timed on the mesh bench unless marked device-side; 10 back-to-back calls with two alternating semaphore sets
and persistent buffers, as the model's `CCLManager` does; PCC against fp32 torch on 2048 rows of every device.

### 3.1 Like-for-like at M = 13664 (experiment 2, redone)

| path | ms per call | notes |
|---|---|---|
| MM alone, 8x9 (8,7,10) | 7.45 | matches the 2026-09-21 bench |
| MM + reduce-scatter (model form: persistent buffers, no barrier) | 9.99 | RS = 2.54 |
| **MM + RS + addcmul** (what the block runs) | **10.70** | addcmul = 0.71 |
| MM + RS + addcmul, RS tuned (§3.3) | 10.17-10.23 | |
| fused 8x7, 4 links, (8,7,6) 2x2, DRAM handoff | 10.61 | first fused point: already under the unfused total |
| fused 8x7, 4 links, (8,7,7) 4x1, DRAM handoff | 10.26 | |
| fused 8x7, 4 links, (8,14,7) 4x1, DRAM handoff | 9.52-9.54 | K_block 14 was the first step change |
| fused 8x7, 4 links, (4,7,6) 2x2, window 2 | 10.34 | the window only fits M_block <= 4 at K_block 7 |
| fused 8x8, 2 links, (8,7,6), DRAM handoff | 12.33 | 1 worker/direction on 4 workers total: worker-bound (§3.5) |
| fused 8x8, 2 links, (4,7,6), window 2 | 10.86 | |
| fused 8x6, 4 links, (8,7,6), DRAM handoff | 11.40 | 2 workers/direction but only 48 matmul cores |

Two knobs did nothing: `num_buffers_per_channel` 1 vs 2 vs 4 (10.26 / 10.30) and `chunk_width_in_mm_blocks` 1 vs
whole-block (10.26 / 10.49, the model's 1 is right).

### 3.2 Fused blocking sweep at M = 13664 (experiment 2b, device-side)

`sweep_mm_block_sizes.py`, use case `mmrs_nobias` (new: no bias, no ternary CBs charged to the matmul cores,
non-transposed per-core dims, window-vs-DRAM decided per combo by the L1 estimate), 8x7 matmul grid:

| links | combos | best | | | |
|---|---|---|---|---|---|
| 4 (1 worker/direction) | 149 | **(6,8,8) 2x2 window 8.508 ms** | (6,7,8) 2x2 window 8.539 | (8,7,12) 2x2 DRAM 8.948 | (8,14,7) 4x1 DRAM 9.414 |
| 2 (3 workers/direction) | 328 | (6,7,8) 2x2 window 8.445 ms | (6,8,8) 2x2 window 8.453 | (8,7,12) DRAM 8.710 | (6,4,8) window 8.803 |

The window is worth ~1 ms at M_block 6 (bench: (6,8,8) window 8.88 vs DRAM handoff 9.88). Two links with three
workers per direction ties four links with one (bench 8.89 vs 8.88), so the model keeps its four links and
`FusedMMRSConfig` needs no per-entry link count.

### 3.3 Control arm: the unfused reduce-scatter's hyperparameters (new)

`CCLManager.get_rs_hyperparams` fixes `chunks_per_sync=2, num_workers_per_link=2, num_buffers_per_channel=2` for every
shape. Swept on the unfused MM + RS + addcmul total (`--rs-workers/--rs-chunks/--rs-buffers`; workers are per
link-direction):

| workers/dir | chunks_per_sync 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| 2 | 10.64 | 10.71 | 10.62 | 10.67 | |
| 3 | 10.57 | 10.40 | 10.34 | 10.24 | |
| 4 | 10.54 | 10.41 | 10.28 | **10.26 (10.19 at 4 buffers)** | **10.17** |
| 6 / 8 | | | | 10.19 (8) | 10.28 (6) |

Buffers 2 vs 4 is noise. The same `4 / 16 / 4` setting at the other durations: 5 s (M = 4736) 4.33 -> 4.17 ms
(-3.7%), 10 s (M = 9184) 7.38 -> 7.00 ms (-5.1%). **Not landed**: `get_rs_hyperparams` is shared by every tt_dit
model and by H3's VAE reduce-scatters, none of which were measured; the 5 s and 10 s ff2 paths, which stay unfused
until swept, would take it through an H3-scoped override. Recorded here as a ready follow-up worth ~0.16-0.38 ms per
layer at those durations.

### 3.4 Landing, and the L1 headroom finding

The swept optimum (6,8,8) ran 2000 back-to-back calls on the bench and then failed in the transformer block:

    Statically allocated circular buffers in program 948 clash with L1 buffers on core range [0-0 - 7-6].
    L1 buffer allocated at 947712 and static circular buffer region ends at 957664

Its window shard (2 x 6 x 21 tiles x 2 KB = 504 KB) plus circular buffers need 1336 KB; the bench's L1 is otherwise
empty, the block's has a few KB of persistent buffers at the top (CCLManager counters and semaphores), and the shard
landed 10 KB inside the CB region. **(6,7,8) is landed instead**: 1280 KB, 0.4% slower in the sweep, 8.94 vs 8.88 ms
on the bench. The harness now keeps a 64 KB margin when it decides to window a combo (`MMRS_WINDOW_L1_MARGIN_KB`).

### 3.5 Roofline: the reduce-scatter worker bound

Fusion trades matmul cores for overlap. With r rows given to the reduce-scatter the matmul costs
`t_mm x r / (9 - r)` more, and the fused op wins when that is below the reduce-scatter + addcmul it hides (3.05 ms
host-timed after the last chunk's tail). One row (0.93 ms) and two rows (1.93 ms) pass; three (3.39 ms) do not, which
is the 8x6 result. But r rows only hold `r x 8 / (2 x links) - 1` workers per direction, and each worker moves far more
than its fabric share -- its input slices, the incoming intermediates, the sends, the output slice and both addcmul
operands, 3.25 x M x N x b = 477 MB per device in total. At the ~9 GB/s per worker the standalone reduce-scatter
achieves, that is 13.1 ms on the 4 workers one row can hold (the 8x8 / 2-link point measured 12.33), 6.6 ms on the 8
workers of two rows at four links (hidden under an 8.7 ms matmul), 3.3 ms on 16. Two rows is the interior optimum, and
the boundary is independent of M (every term is linear in it). The full model with the measured points is the
*Fused MM/RS Roofline* note.

## 4. Results

| step | ff2 tail per layer per device | status |
|---|---|---|
| unfused (8,8,8) default + RS + addcmul | ~10.9 ms (7.01 matmul) | before 2026-09-17 |
| unfused (8,7,10) + RS + addcmul | 10.70 ms (7.45 + 2.54 + 0.71, host-timed) | shipped 2026-09-17 to 2026-09-22 |
| unfused, RS tuned | 10.17 ms | measured, not landed (§3.3) |
| **fused 8x7 (6,7,8) 2x2, window 2** | **8.94 ms** (8.61 device-side in the block) | **landed 2026-09-23**: -1.76 ms/layer host-timed, -1.88 device-only / -2.96 with op gaps in the block profile; end to end **-166 ms/fwd (-1.35%)**, CLIP 35.58 vs 35.71 |
| fused (6,8,8) | 8.88 ms | swept optimum; does not fit the block's L1 (§3.4) |
| roofline, fused | 8.7 ms at the unfused matmul's 53% utilisation on 56 cores; 4.6 ms at peak | |

## 5. What is left

1. **VBench** on the fused output (the pipeline test's `gate_vbench`): needs `~/vbench_env`, absent on this host. The
   10-step A/B, CLIP and the block profile (§1) are done; a 50-step run with VBench is the last quality gate left.
2. **RS hyperparameters** for the unfused 5 s / 10 s paths (§3.3): an H3-scoped override, or sweep and fuse those
   durations too (same recipe as §3.2 with their M).
3. **A zone breakdown** of the fused op's matmul: port `agmm_compute_zones.py`'s anchors to the `minimal_matmul`
   compute kernel; the 56-core matmul at 53% is now the limiter, and the AGMM's fp32-off lever ([ff1.md](ff1.md) §3.1)
   should transfer.
4. **A faster reduce-scatter worker** is the lever that would unlock one RS row (8x8 matmul, 7.6 ms): 4 workers would
   have to move 477 MB in under that, ~16 GB/s each against ~9 today.
5. **PCC-validate (12,7,8) 4x2** for the unfused matmul (1.5%), relevant only to the 5 s / 10 s paths now.

## 6. Tooling and recipes

```bash
# like-for-like A/B on one TP ring, host-timed, PCC on every device
python models/tt_dit/tests/models/minimax_h3/tools/transformer_op_mesh_bench.py --op ff2 --with-rs --with-addcmul
python models/tt_dit/tests/models/minimax_h3/tools/transformer_op_mesh_bench.py --op ff2 --with-rs --with-addcmul --rs-workers 4 --rs-chunks 32 --rs-buffers 4
python models/tt_dit/tests/models/minimax_h3/tools/transformer_op_mesh_bench.py --op ff2 --fused --mm-grid 8x7 --window 2 --blocks 6,7,8,2,2
python models/tt_dit/tests/models/minimax_h3/tools/transformer_op_mesh_bench.py --op ff2 --fused --mm-grid 8x7 --window 2 --blocks 6,7,8,2,2 --iters 2000 --check-rows 512   # soak
# fused blocking sweep (device-side); the 2-link row runs on wh_4x8_ring_2links
MM_SWEEP_PROFILER_DUMP_EVERY=100000 python -m pytest models/tt_dit/utils/sweep_mm_block_sizes.py::test_mm_sweep \
  -k "13664_3584_5376_8x7_mm_mmrs_nobias and wh_4x8_ring" -s --timeout 14400
# block profile
scripts/run_safe_pytest.sh --profile \
  "'models/tt_dit/tests/models/minimax_h3/test_performance_minimax_h3.py::test_minimax_h3_transformer_block_perf[wormhole_b0-sp_sim1-15s_768p-4x8sp1tp0nl4_ring_is_fsdp1]'" \
  -s --timeout 3600
```

Bench flags: `--with-addcmul` (the block's gated residual after the RS), `--rs-workers/--rs-chunks/--rs-buffers`
(unfused RS hyperparameters), `--fused`, `--mm-grid WxH`, `--num-links`, `--window 0|2`, `--chunk-width`. A fused
`--rs-workers` is per link-direction and defaults to the `FusedMMRSConfig.get_params` zone formula.
