# Panoptic-DeepLab on Quasar: gaps and other failures

Running log of everything the Panoptic-DeepLab Quasar bring-up (#60447) hit. Section 1 is the main
output: ttnn ops/paths that work on WH/BH but not on Quasar. Every workaround the op suite or the model
copy uses to get past a gap is listed with it, so no gap is hidden by a workaround.

Status values: `open` (not fixed in ttnn), `worked-around` (suite/model avoids it; still open in ttnn;
"(experimental op)" means routed to the stop-gap `ttnn.experimental.quasar` port of the base op),
`fixed` (fix landed; workaround removed), `unverified` (suspected, not yet reproduced),
`craq-sim only` (seen on craq-sim, not yet checked on the emulator — craq-sim's Quasar support is
"not validated", so a hang there may be a simulator bug).

How found: the captured-op suite `models/experimental/ops/quasar/tests/panoptic_deeplab_ops/` (161 cases
from one 512x1024 inference, captured on ttsim Blackhole 5x4), run on craq-sim `quasar` @ `889c806a`
(`libttsim.so` md5 `0e313844c8e0…`, `quasar_32_arch.yaml`, 8x4 grid), 2026-10-15. Without either
workaround below, 48 of 161 cases pass; every failure is Q1 or Q2.

Common environment for the repros:

```bash
export TT_METAL_SIMULATOR=/localdev/gchoudhary/quasar-sim/libttsim.so TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_DISABLE_SFPLOADMACRO=1
export TTNN_GRAPH_OPS_BF8_AS_BF16=1   # Q1 workaround, needed by every row below Q1
cd models/experimental/ops/quasar/tests/panoptic_deeplab_ops
```

## 1. Ops supported on WH/BH but not on Quasar

| # | Op / path | Quasar failure | Repro | Workaround | Status |
|---|---|---|---|---|---|
| Q1 | `bfloat8_b` / `bfloat4_b` tensors, any op. PDL keeps conv weights and most activations in bfloat8_b. | `TT_FATAL @ .../validation/resource/dfb.cpp:110: DFB '…' has data format 'Bfp8_b' which is not supported on architecture` — Quasar has no block-float formats (`is_supported_quasar` in `tt_backend_api_types.cpp`). 94 of 161 cases. | any bf8 case without `TTNN_GRAPH_OPS_BF8_AS_BF16`, e.g. `pytest test_slice.py` | suite: `TTNN_GRAPH_OPS_BF8_AS_BF16=1` builds them as bfloat16 (also `DataType::BFLOAT8_B` inside `Conv2dConfig`), same switch as PR #59575. Model: `tt/arch.py:block_float_dtype(device)` is bfloat16 on Quasar. Doubles weight/activation memory. MX formats are the eventual replacement. | worked-around |
| Q2 | Base `ttnn.conv2d`, `ttnn.add`, `ttnn.multiply`, `ttnn.move`, `ttnn.max_pool2d` | `TT_FATAL @ tt_metal/impl/kernels/kernel.hpp:532: DataMovementKernel is not supported on Quasar. Use QuasarDataMovementKernel instead.` (max_pool2d: `ComputeKernel is not supported on Quasar`, `kernel.hpp:738`) | `TTNN_GRAPH_OPS_QUASAR_FORKS=0 pytest test_add.py` (likewise `test_multiply.py`, `test_move.py`, `test_max_pool2d.py`, `test_conv2d.py`) | `op_utils.quasar_routed` calls `ttnn.experimental.quasar.<op>` on Quasar (same routing as the yolo suite, #60455): add 8/8, multiply 1/1, move 6/6 pass. conv2d and max_pool2d forks: see Q3–Q6. | worked-around (experimental op) |
| Q3 | `ttnn.experimental.quasar.conv2d`, 27 of 51 captured convs (e.g. `00_32x64x512`, `01_2048x512`, `04_2048x256_hs`, `06_32768x64`, `10_128x256x256`) | `TT_FATAL @ .../validation/resource/dfb.cpp: DataflowBuffer 'gather_scratch0' is self-looped by data-movement kernel 'reader0' (bound as both PRODUCER and CONSUMER). Self-loop DFBs are not supported for data-movement kernels on Gen2 architectures.` Independent of fp32 accumulation. | `pytest test_conv2d.py -k 00_32x64x512` | none | open |
| Q4 | `ttnn.experimental.quasar.conv2d` with `fp32_dest_acc_en=True` (the model's compute config for every conv), the other 24 convs (e.g. `02_2048x256_hs`, `07_32768x64`, `08_8192x128_hs`) | `TT_FATAL @ .../validation/hardware_config.cpp: Compute kernel 'compute' consumes FP32 DFB 'cb_intermed0' with enable_32_bit_dest=true, but provides no unpack_modes entry for this DFB. This configuration requires an explicit choice between UnpackMode::UnpackToSrc and UnpackMode::UnpackToDest.` With `fp32_dest_acc_en=False`, `07` passes, but `02` and `08` hit Q5 / Q6. | `pytest test_conv2d.py -k 07_32768x64` | none (turning fp32 accumulation off would change the model's numerics) | open |
| Q5 | `ttnn.experimental.quasar.conv2d`, `02_2048x256_bf8_hs-l1` with `fp32_dest_acc_en=False` | `TT_FATAL: DFB 4 device slot 4 out of range (num_dfbs 2)` | `pytest test_conv2d.py -k 02_2048x256` after setting `fp32_dest_acc_en=False` in `op_utils._conv_compute_config` (no switch for it) | none | open |
| Q6 | `ttnn.experimental.quasar.conv2d`, `08_8192x128_bf8_hs-l1` with `fp32_dest_acc_en=False` | hang: no progress after the first program's DFB config is written to all cores (killed after 30 min) | as Q5 with the `08` case | none | open, craq-sim only |
| Q7 | `ttnn.max_pool2d` → `ttnn.experimental.quasar.max_pool2d` (`00_131072x64`, stem max pool) | fork hangs after the first program's DFB config is written (killed after 30 min); base op fails with Q2 | `pytest test_max_pool2d.py` | none | open, craq-sim only |
| Q8 | `ttnn.avg_pool2d` (`00`, ASPP pooling branch, `[1,32,64,2048]` → 1x1) | hangs, both base (`kernels/compute/reduce.cpp`) and `ttnn.experimental.quasar.avg_pool2d` (`reduce_metal2.cpp`), after the first program's DFB config is written | `pytest test_avg_pool2d.py`; base: add `TTNN_GRAPH_OPS_QUASAR_FORKS=0` | none | open, craq-sim only |
| Q9 | Base `ttnn.matmul` with `MatmulMultiCoreReuseMultiCast1DProgramConfig` (final-upsample matmuls, both captured cases) | hang after the first program's DFB config is written (killed after 30 min) | `pytest test_matmul.py` | none; `ttnn.experimental.quasar.matmul` not tried yet | open, craq-sim only |
| Q10 | Base `ttnn.typecast` (`kernels/compute/eltwise_typecast.cpp`) | hang after the first program's DFB config is written (killed after 30 min). Note: with the Q1 switch the captured bf16 → bfloat8_b casts become bf16 → bf16, which may be what hangs. | `pytest test_typecast.py` | none; `ttnn.experimental.quasar.typecast` not tried yet | open, craq-sim only |

## 1b. Small-grid limits

Not yet run: the 2-core grid (emu2x3 stand-in, see 2.1) comes after the 8x4 results above.

## 2. Simulator and harness issues — not Quasar gaps

| # | Issue | Detail | Workaround | Status |
|---|---|---|---|---|
| S1 | craq-sim `quasar` @ `889c806a` cannot run a 2x3 descriptor | It hard-codes the 32-core layout (`src/tile.cpp` `coord_to_tile`): the emulator's `quasar-2x3_emu.yaml` dies at device start with `UnimplementedFunctionality: coord_to_tile: coord (0,1)`; UMD's `tests/soc_descs/quasar_simulation_2x3.yaml` fails in Metal with `RuntimeError: bad conversion`. | 8x4 build with `TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="3,2"` gives the emulator's 2x1 compute grid and 2 DRAM banks | open |
| S2 | pytest `--timeout` does not stop a craq-sim hang | the hang is inside the simulator, which the signal-based timeout cannot interrupt | external watchdog kills a test process after 30 min | worked-around |
