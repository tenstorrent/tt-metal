# MiniMax-H3 on Wormhole Galaxy: the ff2 matmul and reduce-scatter — baseline, experiments, results

The second feed-forward linear, `RowParallelLinear(14336, 5376)`: a plain `ttnn.experimental.minimal_matmul` on the
full 8x9 grid over this device's K shard, followed by a separate ring `reduce_scatter`. Measured 2026-09-17 (blocking
sweep, fused MM/RS sweep, numerics) and 2026-09-21 (mesh bench) on the 4x8 Wormhole Galaxy. Block-level context:
[README.md](README.md).

## 1. The op and the baseline

| | |
|---|---|
| model call | `ParallelFeedForward.ff2 = RowParallelLinear(14336, 5376, bias=False)`, `models/tt_dit/layers/linear.py` (`RowParallelLinear.forward`: `get_matmul_config` on the full grid, `minimal_matmul`, then `ccl_manager.reduce_scatter`) |
| shape per device | M = 13664, K = 3584 (already per-device, no gather), N = 5376 before the reduce-scatter, 1344 after |
| config | bf16, HiFi2, `fp32_dest_acc_en = True`, `packer_l1_acc = True`, `math_approx_mode = True` (the block's shared matmul config) |
| grid / blocking | full 8x9 = 72 cores; `grid_89_configs[(13664, 3584, 5376)] = (8, 7, 10)` subblock 2x2 (`models/tt_dit/utils/matmul.py`), landed 2026-09-17 |
| roofline | compute **3.57 ms** (0.53 TFLOP at 72 cores x 2048 FLOP/cycle x 1.0 GHz = 147.5 TFLOP/s), DRAM 0.47 ms; the reduce-scatter moves (R-1) x 2 x M x N / R / (2 x 4 links) per link = 1.10 ms at 12.5 GB/s |
| baseline | matmul **6.77 ms** (53% of peak) at the landed blocking; 7.01 ms at the (8,8,8) default before it. In the 2026-09-17 block: `MinimalMatmulDeviceOperation` 8.54 ms for 2 calls (ff2 + the small adaLN projection), `ReduceScatterMinimalAsync` 2.78 ms |
| mesh bench (`transformer_op_mesh_bench.py --op ff2`, host-timed, 10 calls, 2026-09-21) | 7.46 ms; with `--with-rs` 10.00 ms (the reduce-scatter adds 2.5 ms) |
| single device (`--op ff2`) | 7.40 ms plain |
| numerics | landed (8,7,10) pcc 1.0000000 vs torch fp32 on one device; vs (8,8,8) max diff 0.0156 (one bf16 ulp at 3.4), mean 0.0 |

## 2. Baseline breakdown by zone

No Tracy device zones have been taken on this kernel. The `minimal_matmul` compute kernel is the same code structure
as the AGMM's (same relay-chain data movement, same `matmul_blocks` K loop, a copy epilogue), so the ff1 K-loop
findings are expected to transfer: a 2x2 fp32 subblock issuing at ~47 cycles per tile-MAC against the nominal 32,
which alone would put the loop at ~5.2 ms of the 6.77. The only decomposition that exists is the blocking sweep in
§3.1. Taking zones here is a copy of the AGMM zone tool's anchors onto
`ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/compute.cpp`.

## 3. Experiments

### 3.1 Blocking sweep (experiment 1, 2026-09-17)

`sweep_mm_block_sizes.py` on `wh_4x8_ring` at 13664 rows/device, 322 combos on the 8x9 grid, timing only:

| blocking | device kernel | rank | status |
|---|---|---|---|
| (8, 8, 8) 2x2 (the hardcoded default before any Wormhole entry) | 7013.1 us | | replaced |
| **(8, 7, 10) 2x2** | **6770.7 us** | 2 | **landed** (`grid_89_configs`), -3.5%; PCC-validated |
| (12, 7, 8) 4x2 | 6668.2 us | 1 | not landed: 1.5% faster (~5 ms/step) but the sweep is timing-only and it has not been PCC-validated |

Why the entry is keyed on `(M, K, N)` rather than added to the `(K, N)`-keyed model table, and how `get_matmul_config`
falls back on equal `M_per_core` after an exact miss, is documented in `models/tt_dit/utils/matmul.py`. The
`M=13664` keying itself was the fix for the rows-per-device mismatch (the tables had been keyed on 13632, a length the pipeline never runs).

### 3.2 Fused MM/RS (experiment 2) — stays disabled

ff2 as `minimal_matmul_strided_reduce_scatter_async`, one entry per candidate matmul grid (the
reduce-scatter takes the rows the matmul leaves; at `num_links=4` that is 1 worker/link at 8x7, 2
at 8x6, 3 at 8x5):

| matmul grid | RS workers/link | best |
|---|---|---|
| 8x7 | 1 | 3134.7 us |
| 8x6 | 2 | 3610.2 us |
| 8x5 | 3 | 3996.7 us |

Monotonically worse as the RS zone grows: every core handed to the reduce-scatter costs the matmul
more than it returns. All are far off the 2373.0 us unfused matmul, so `eab3dfbd599` keeping
Wormhole off the fused path is right on tuned configs too, not just against the broken fallback (the accidental fused
path was the mid-denoise hang, closed in `eab3dfbd599`).
Not a like-for-like total — the unfused figure excludes the separate reduce-scatter and addcmul,
leaving them a 762 us budget — but combined with the measured 2.0-4.0% end-to-end gain from
disabling it, the conclusion holds.

### 3.3 Numerics (experiment 11)

The sweep never checks PCC, so the landed blocking was validated after the fact at M=13664 exactly: plain
`minimal_matmul` on one device vs torch fp32, pcc 1.0000000; against (8,8,8) the max difference is 0.0156 (one bf16
ulp at 3.4) and the mean difference 0.0, flat across N, so the partial trailing block of 9 tiles is handled correctly.

## 4. Results

| step | matmul per call | status |
|---|---|---|
| (8,8,8) default | 7.01 ms | before 2026-09-17 |
| **(8,7,10) landed** | **6.77 ms** (53%) | shipped; with ff1's blocking, -69.9 ms/fwd (-0.58%) end-to-end (README Part 4) |
| (12,7,8) 4x2 | 6.67 ms | needs PCC validation |
| roofline | 3.57 ms | |

The reduce-scatter (2.78 ms in the block, fabric-bound at 1.10 ms ideal) is a separate op and was not worked on.

## 5. What is left

1. **A zone breakdown**: port `agmm_compute_zones.py`'s anchors to the `minimal_matmul` compute kernel and run it
   through the harness on `13664_3584_5376_8x9_mm_ff2`; expect the AGMM pipeline pace and the fp32-off lever
   ([ff1.md](ff1.md) §3.1) to apply, without the operand waits (no gather, full grid).
2. **PCC-validate (12,7,8) 4x2** and land it if it holds (1.5%).
3. **Fused MM/RS** only if a Wormhole config appears that beats unfused; every measured grid was worse.
4. **The reduce-scatter**: 2.78 ms against a 1.10 ms fabric bound; a persistent-buffer variant exists in `CCLManager`
   (`use_persistent_buffer`) and has not been A/B'd here.

## 6. Tooling and recipes

```bash
python models/tt_dit/tests/models/minimax_h3/tools/transformer_op_mesh_bench.py --op ff2 [--with-rs] [--blocks 12,7,8,4,2]
python models/tt_dit/tests/models/minimax_h3/tools/transformer_op_single_device_bench.py --op ff2 --cases "8,7,10,2,2,1;12,7,8,4,2,1"
MM_SWEEP_PROFILER_DUMP_EVERY=100000 python -m pytest models/tt_dit/utils/sweep_mm_block_sizes.py::test_mm_sweep \
  -k "13664_3584_5376_8x9_mm_ff2 and wh_4x8_ring" -s --timeout 7200
```
