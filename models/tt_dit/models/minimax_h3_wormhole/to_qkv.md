# MiniMax-H3 on Wormhole Galaxy: the to_qkv AGMM — baseline, zones, experiments, results

The fused QKV projection of the attention block, `ttnn.experimental.all_gather_minimal_matmul_async` with
`chunks=3` (one matmul, output split into q | k | v by the writer). Measured 2026-09-21 on the 4x8 Wormhole Galaxy
`UF-EV-B12-GWH02`, branch `jameslee/exp_ring_sdpa_wh`, with the generic op tools written that day. Block-level
context: [README.md](README.md); the ff1 investigation this follows: [ff1.md](ff1.md).

## 1. The op and the baseline

| | |
|---|---|
| model call | `ColParallelLinear(5376, 3 * 7168, chunks=3, bias=False)`, `attention_minimax_h3.py:145-154`; AGMM branch `models/tt_dit/layers/linear.py` |
| shape per device | M = 13664 rows, K = 5376 (K_local 1344, gathered over the TP = 4 ring), N = 5376 = 3 chunks x 1792 (this device's q, k and v heads) |
| config | `chunks=3`, bias = False, bf16 in/out, HiFi2, `fp32_dest_acc_en = True`, `packer_l1_acc = True`, `math_approx_mode = True` |
| grid / blocking | 8x8 worker grid, `AGMM_BLOCK_SIZES[(5376, 5376)] = (8, 7, 12)`, subblock 2x2 (fixed by `get_matmul_config` for a `default_block_size`), 4 links, 2 workers per link |
| epilogue | `copy_block` (`compute.cpp:73`): the q/k/v split is done by the writer (`matmul_dataflow_common.hpp write_tile_to_chunk`, `dm_in1_sender_out.cpp`), never in the compute kernel |
| roofline | compute **6.03 ms** (0.79 TFLOP at 131.1 TFLOP/s), DRAM 0.71 ms, fabric 1.10 ms; compute-bound on Wormhole |
| blocking sweep (2026-09-17, `qkv` use case, 425 combos) | shipped (8,7,12) 10,401.8 us; best (8,6,12) 10,351.4 us (0.5%, within the run-to-run spread); default kept |
| baseline | **10.4 ms** in the 2026-09-17 block (58% of HiFi2 peak); device kernel 10,304 us on 2026-09-21 |
| mesh bench (`transformer_op_mesh_bench.py --op to_qkv`, host-timed, 10 calls) | fused (chunks=3) 11.30 ms / plain 11.22 ms: the chunked writer costs 0.08 ms |
| single device (`transformer_op_single_device_bench.py --op to_qkv`, `minimal_matmul_split`) | 11.34 / 11.26 ms |

## 2. Baseline breakdown by zone

Tracy device zones on the AGMM compute kernel through the sweep harness (`agmm_compute_zones.py apply block|opwait`,
shape id `13664_5376_5376_8x8_agmm_qkv`, combo (8,7,12) 2x2), per core; the harness always passes a bias, so the
epilogue zone it records is `EPILOGUE_BIAS` (`add_bias_block`) where the model runs `copy_block`.

| zone | per core, us | share | how |
|---|---|---|---|
| K loop (`KLOOP`; 336 iterations = 7 M blocks x 2 N blocks x 24 K blocks, 29.9 us each) | **10,053** | 98% | block zone |
| epilogue (`EPILOGUE_BIAS`, 14 blocks) | 168 | 1.6% | block zone |
| kernel (`TRISC-KERNEL`) | 10,228 | | |
| operand wait per iteration (`OPWAIT`, first ~100 iterations) | 0.9 us (3%) | | opwait zone |
| sampled 2x2 subblock | not sampled; ff1's 1321 / 1357 / 1236 cycles per 28 tile-MACs (47 per tile-MAC) assumed, and consistent with the K-loop total | | |

### Decomposition

| component | ms | how |
|---|---|---|
| FPU work at HiFi2 peak | 6.0 | roofline |
| grid padding (N per core 21 -> 24 tiles, M 54 -> 56) | 1.1 | issued tile-MACs / useful |
| 2x2 fp32 pipeline issue efficiency (47 vs 32 cycles per tile-MAC) | 2.6 | sampled zones, same pace as ff1 |
| operand waits on the store-and-forward relay | 0.4 | `OPWAIT` |
| epilogue (copy; 0.08 more for the chunked writer) | 0.2 | block zone / mesh A/B |
| **total** | **~10.3** | device 10.30 |

to_qkv is ff1 without the SwiGLU: the loop runs at the 2x2 fp32 pipeline pace and hardly waits on operands.

## 3. Experiments, by zone

### 3.1 The K loop

| lever | result | status |
|---|---|---|
| fp32 dest off, same (8,7,12) 2x2 | single device 11.22 -> 10.67 ms (-4.9%) | measured |
| fp32 dest off + 2x4 subblock | single device 10.80 | measured, worse than 4x2 |
| **fp32 dest off + 4x2 subblock, (8,7,12)** | single device 10.37; **mesh fused 11.30 -> 10.32 ms (-8.7%)**, PCC 1.0000, rel-RMSE 0.0044 -> 0.0107 (bar 0.02) | measured, not landed: precision decision |
| K_block 14: (8,14,12) 2x4 fp32 off | overflows L1 (`program.cpp:2097`) | rejected |
| LoFi (diagnostic; the 6.0 ms of math halves) | single device 11.22 -> 10.21: -1.0 ms for -3.0 ms of math | the loop is unpacker-paced, as ff1's |
| tt-llk `matmul_block` issue efficiency | not run | proposed, see [ff1.md](ff1.md) §3.1 |

### 3.2 Operand delivery

`OPWAIT` is 0.9 us of a 29.9 us iteration (3%). The relay-prefetch patch was not run here (nothing to recover); the
to_out result ([to_out.md](to_out.md)) shows it would not have added bandwidth anyway.

### 3.3 The epilogue and the chunked writer

Plain vs `chunks=3` on the mesh: 11.22 vs 11.30 ms, so the writer-side q/k/v split costs 0.08 ms. The model's
`copy_block` epilogue was not zoned (the harness runs `add_bias_block`, 168 us); it is bounded above by that.

## 4. Results

| step | per call | util | status |
|---|---|---|---|
| baseline, (8,7,12) 2x2 fp32 dest | 10.3 ms | 58% | shipped |
| fp32 dest off, (8,7,12) 4x2 | 9.3 ms (mesh host 10.32 vs 11.30) | ~65% | measured 2026-09-21; precision decision, model-level check needed; take together with ff1's |
| + tt-llk `matmul_block` issue efficiency | ~8 | ~75% | tt-llk work |
| roofline | 6.0 | 100% | |

Per forward: 1.0 ms per call is 50 ms of the 12.4 s (0.4%).

## 5. What is left

- **fp32 dest off with (8,7,12) 4x2**: -8.7% at 2.4x the accumulator error, the same precision decision as ff1's, to
  be taken for the two ops together (`ParallelFeedForward` / `Attention` hand one compute config to their linears;
  give the AGMMs their own). Validate at the model level.
- **Padding**: the N padding (21 -> 24 tiles per core) is 1.1 ms of issued MACs. An N_block of 7 (3 blocks, 21
  tiles, no padding) needs an odd subblock width, which the 2x2 / 4x2 shapes rule out; not measured.
- **tt-llk `matmul_block` issue efficiency**: the remainder of the loop, shared with ff1 and to_out.

## 6. Tooling and recipes

Everything below is shared with [ff1.md](ff1.md) §6 (mesh reproducer, sweep-harness timing, device zones, recovery);
only the op-specific invocations are listed here.

```bash
python models/tt_dit/tests/models/minimax_h3/tools/transformer_op_mesh_bench.py --op to_qkv [--no-fusion] [--fp32-dest 0 --blocks 8,7,12,4,2]
python models/tt_dit/tests/models/minimax_h3/tools/transformer_op_single_device_bench.py --op to_qkv --no-fusion --cases "8,7,12,2,2,1;8,7,12,4,2,0" [--fidelity LoFi]
# zones through the harness (apply, run, parse, revert):
python models/tt_dit/tests/models/minimax_h3/tools/agmm_compute_zones.py apply block   # or opwait / sampled
MM_SWEEP_EXPLICIT_COMBOS='[[8,7,12,2,2]]' MM_SWEEP_PROFILER_DUMP_EVERY=100000 python -m pytest \
  models/tt_dit/utils/sweep_mm_block_sizes.py::test_mm_sweep -k "13664_5376_5376_8x8_agmm_qkv and wh_4x8_ring" -s --timeout 7200
python models/tt_dit/tests/models/minimax_h3/tools/agmm_compute_zones.py parse generated/profiler/mm_sweep_wh_4x8_ring_13664_5376_5376_8x8_agmm_qkv/reports/<ts>/profile_log_device.csv
python models/tt_dit/tests/models/minimax_h3/tools/agmm_compute_zones.py revert
```
