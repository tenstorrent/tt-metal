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

Re-run 2026-10-16 on tt-metal main @ `2014821daa6` (same craq-sim): 8x4 — 106 of 110 non-conv cases pass
(Q7–Q9 hang), all 51 conv2d cases fail (Q3/Q4). 2x1 stand-in (S1) — 88 pass, 18 skip (captured shard
grids wider than 2 cores), Q7–Q9 fail; conv2d 42 fail (Q3, Q4, and 13 small-grid limits, see 1b), 9 skip.

Re-run 2026-10-30 on tt-metal main @ `d7cec1a7241` (branch rebased; includes #62563, #62730, #62518 and the DFB
drain fixes), craq-sim `libttsim.so` md5 `63d56a28069144598cc481bc00fc8568`, watcher off. conv2d (Q3–Q6) not re-run.
8x4: Q8 passes, Q9 passes via `ttnn.experimental.quasar.matmul`, Q7 no longer hangs but returns wrong values.
2 cores ("3,2" stand-in): Q8 passes; Q7 gets past the mainline-slicing failure and then hits G4 (L1_SMALL), and with a
larger L1_SMALL returns wrong values; every other op file passes as before (add 4, concat 5, multiply 1, permute 2,
relu 5, reshape 12, slice 12, to_device 1, to_layout 14, to_memory_config 20, typecast 5, unsqueeze_to_4D 3, upsample 4;
G3 skips unchanged). Logs: `/localdev/gchoudhary/pdl_quasar_runs/recheck_2026-10-30/`.

Acceptance re-run 2026-11-02 on tt-metal main @ `de69037d3a0`, craq-sim md5 `63d56a28…` with `TT_METAL_WATCHER=10ms`,
`TT_METAL_LLK_ASSERTS=1`, `TT_METAL_WATCHER_DISABLE_NOC_SANITIZE=1` (`=10ms`, not `=1`: the watcher's 5 ms host poll
starves the craq-sim clock), and ttsim v1.15 Blackhole (md5 `951db8b8…`, no `reg=19` abort) on 5x4.
BH 5x4: op suite 161/161. Model tests (`-k "not 110_cores"`): aspp, semseg, tt_upsample pass; insemb and resnet fail
with the numbers main has (insemb center rel_err 32.6, resnet res5 rel_err 0.647); `test_tt_model` 20-core and
`test_pipeline_e2e` were cut short by the runner's timeouts (600 s pytest timeout; 1 h watchdog, 100 inputs) and are
not results. 8x4: 16 of 19 files pass (matmul 2/2 via the fork); avg_pool2d hits S3, max_pool2d Q12, conv2d Q13.
"3,2": 13 files pass, move/reshard skip (G3); matmul G1; avg_pool2d S3; max_pool2d Q12; conv2d Q3/G1/G2, then Q13.
Logs: `/localdev/gchoudhary/pdl_quasar_runs/full_2026-11-02/`.

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
| Q3 | `ttnn.experimental.quasar.conv2d`, 27 of 51 captured convs (e.g. `00_32x64x512`, `01_2048x512`, `04_2048x256_hs`, `06_32768x64`, `10_128x256x256`) | `TT_FATAL @ .../validation/resource/dfb.cpp: DataflowBuffer 'gather_scratch0' is self-looped by data-movement kernel 'reader0' (bound as both PRODUCER and CONSUMER). Self-loop DFBs are not supported for data-movement kernels on Gen2 architectures.` Independent of fp32 accumulation. Triggered by `config_tensors_in_dram=True`, which the model sets on every conv: the Quasar halo factory (`experimental/quasar/halo/device/untilize_with_halo_program_factory.cpp:467-477`) then binds `gather_scratch0/1` to the reader as PRODUCER and CONSUMER. 2x1: 10 of 51. | `pytest test_conv2d.py -k 00_32x64x512`; standalone repro in [#55526 (comment)](https://github.com/tenstorrent/tt-metal/issues/55526#issuecomment-6222213694) | none | open; reported on #55526 (DM self-loop umbrella). Not re-run on 2026-10-30 |
| Q4 | `ttnn.experimental.quasar.conv2d` with `fp32_dest_acc_en=True` (the model's compute config for every conv), the other 24 convs (e.g. `02_2048x256_hs`, `07_32768x64`, `08_8192x128_hs`) | `TT_FATAL @ .../validation/hardware_config.cpp: Compute kernel 'compute' consumes FP32 DFB 'cb_intermed0' with enable_32_bit_dest=true, but provides no unpack_modes entry for this DFB. This configuration requires an explicit choice between UnpackMode::UnpackToSrc and UnpackMode::UnpackToDest.` With `fp32_dest_acc_en=False`, `07` passes, but `02` and `08` hit Q5 / Q6. Comes from the Quasar mcast matmul factories that 1x1 convs lower to (no `unpack_modes` for the Float32 `cb_intermed0`). 2x1: 19 of 51. | `pytest test_conv2d.py -k 07_32768x64`; standalone repro in [#57683 (comment)](https://github.com/tenstorrent/tt-metal/issues/57683#issuecomment-6222214147) | none (turning fp32 accumulation off would change the model's numerics) | open; reported on #57683 (item 2). Not re-run on 2026-10-30 |
| Q5 | `ttnn.experimental.quasar.conv2d`, `02_2048x256_bf8_hs-l1` with `fp32_dest_acc_en=False` | `TT_FATAL: DFB 4 device slot 4 out of range (num_dfbs 2)` | `pytest test_conv2d.py -k 02_2048x256` after setting `fp32_dest_acc_en=False` in `op_utils._conv_compute_config` (no switch for it) | none | open. Not re-run on 2026-10-30 |
| Q6 | `ttnn.experimental.quasar.conv2d`, `08_8192x128_bf8_hs-l1` with `fp32_dest_acc_en=False` | hang: no progress after the first program's DFB config is written to all cores (killed after 30 min) | as Q5 with the `08` case | none | open, craq-sim only. Not re-run on 2026-10-30 |
| Q7 | `ttnn.max_pool2d` → `ttnn.experimental.quasar.max_pool2d` (`00_131072x64`, stem max pool, `output_layout=TILE`) | 2026-10-30 (main @ `d7cec1a7241`): the hang and the mainline-slicing failure are gone (#62563: the Quasar pool slices with the Quasar `padded_slice` / `slice_write`). It now returns **wrong values**: 8x4 PCC 0.176 (163 s; one slice fits, so `pool2d_DRAM` falls back to `pool2d_L1`), "3,2" PCC 0.352 (sliced; 1623 s, needs a larger L1_SMALL, see G4). Cause: the Quasar pool's **TILE-output path**. Stem-shaped probe (64x128, C=64, k3 s2 p1, DRAM input) on craq-sim: TILE output gives PCC 0.172 with TILE or ROW_MAJOR input, and ROW_MAJOR output is exact (PCC 1.0) with either input, 1 or 2 slices, and at the full 256x512 stem size. The unit case `tiled_out_max_k3x3_T1` (`tests/ttnn/nightly/unit_tests/operations/experimental/quasar/test_pool2d.py`, sim-skipped by default) mismatches too (PCC 0.089). The file documents this as a pre-existing Quasar bug, wrong on ZeBu RTL and exact on WH; craq-sim used to abort on it (`tensix_execute_pacr_stride`) and now reproduces it. Earlier: 8x4 hang after the first DFB config write; 2 cores: `ComputeKernel is not supported on Quasar` from the mainline `run_sliced_op`. | `pytest test_max_pool2d.py` (with the uncommitted suite change that accepts the pool's flattened `[1,1,N*H*W,C]` single-slice output in `graph_case._check_output`); probe: `QPOOL_NO_SIM_SKIP=1 QPOOL_ONLY=tiled_out_max_k3x3_T1 pytest tests/ttnn/nightly/unit_tests/operations/experimental/quasar/test_pool2d.py -k test_qpool_unit_cases` | none yet. Candidate: `output_layout=ROW_MAJOR` plus a separate tilize (exact in the probe; not yet tried in the suite or the model) | open: hang and slicing failure fixed; TILE-output values wrong (craq-sim and, per the unit test notes, ZeBu RTL) |
| Q8 | `ttnn.avg_pool2d` (`00`, ASPP pooling branch, `[1,32,64,2048]` → 1x1) | hangs, both base (`kernels/compute/reduce.cpp`) and `ttnn.experimental.quasar.avg_pool2d` (`reduce_metal2.cpp`), after the first program's DFB config is written | `pytest test_avg_pool2d.py`; base: add `TTNN_GRAPH_OPS_QUASAR_FORKS=0` | `ttnn.experimental.quasar.avg_pool2d` (routed) | fixed on craq-sim (fork): passes on main @ `d7cec1a7241`, 2026-10-30 — 8x4 1/1 (18 s; also with the previous `.so` `7f4a2163…`), "3,2" 1/1 (30 s). Earlier: hung on craq-sim and for 60 min on the emulator; emulator not re-checked. Base op not re-run |
| Q9 | Base `ttnn.matmul` with `MatmulMultiCoreReuseMultiCast1DProgramConfig` (final-upsample matmuls, both captured cases) | hang after the first program's DFB config is written (killed after 30 min) | `pytest test_matmul.py` (base: `TTNN_GRAPH_OPS_QUASAR_FORKS=0`) | suite routes `ttnn.matmul` to `ttnn.experimental.quasar.matmul` (`op_utils._QUASAR_FORKED_OPS`), rebuilding the captured config with the fork's own `ttnn._ttnn.operations.experimental.quasar.MatmulMultiCoreReuseMultiCast1DProgramConfig` (a separate type from `ttnn.…`). The fork's config has no `stream_in1`; the captured value is the default (0) and is dropped. 8x4: 2/2 pass (86 s, 2026-10-30). | worked-around (experimental op); base op not re-run on 2026-10-30. On 2 cores the routed cases still hit G1 |
| Q10 | Base `ttnn.typecast` (`kernels/compute/eltwise_typecast.cpp`) | hang after the first program's DFB config is written (killed after 30 min). Note: with the Q1 switch the captured bf16 → bfloat8_b casts become bf16 → bf16, which may be what hangs. | `pytest test_typecast.py` | none | fixed: passes on craq-sim after the rebase to `2014821daa6` (8x4 5/5, 2x1 5/5) and on the emulator (`00`, 57 s) |
| Q11 | Any Quasar compute kernel built with `TT_METAL_WATCHER=1`, e.g. `tilize_metal2` (device tilize used by `ttnn.to_layout` and by the TILE upload of `ttnn.add` inputs) | JIT link fails: `trisc3 build failed ... dataflow_buffer.inl:36: undefined reference to 'g_dfb_interface'`. `trisc3` is the isolated-SFPU TRISC (`UCK_CHLKC_ISOLATE_SFPU`); `firmware/src/tt-2xx/trisc.cc:67-74` defines `g_dfb_interface` only for the unpack/pack TRISCs, but the non-math `DataflowBuffer` constructor references it. Without the watcher the reference is apparently dropped and the kernel links (unconfirmed). Same on craq-sim and the emulator; `TT_METAL_LLK_ASSERTS=1` alone is fine. Note `TT_METAL_WATCHER=0` still enables the watcher — unset it. | `TT_METAL_WATCHER=1 pytest "test_to_layout.py::test_to_layout[00_1x256_bf16_int-dram-ttnn_mesh_device0]"` (craq-sim reproduces in seconds) | run without the watcher | fixed by #60860 (reported as #60768): the repro links and runs with `TT_METAL_WATCHER=1` on main @ `14fd6d745e3` (2026-10-19) |
| Q12 | `ttnn.experimental.quasar.max_pool2d` (`00_131072x64`) with LLK asserts on | `Neo0TRISC2 tripped an assert on line 57`, current kernel `experimental/quasar/pool_generic/device/kernels/compute/compute_pool_2d.cpp`; the watcher stops the device (`TT_THROW: Watcher detected tripped assert`). The watcher gives no file name; on the pack TRISC the line matches `tt_metal/hw/ckernels/quasar/metal/llk_io/llk_io_pack.h:57`, the TEN-4746 guard in `llk_push_tiles` (push of a DFB with no pack since `wait_for_free`), so the pool compute kernel likely pushes a reserved DFB it never packed (inferred, not confirmed). Fires before the Q7 values are checked. | `TT_METAL_WATCHER=10ms TT_METAL_LLK_ASSERTS=1 pytest test_max_pool2d.py` (8x4, 70 s; also on "3,2") | Cause (2026-11-03): in the tiled-output path `fast_tilize_cb` is a second view of the bytes packed through `pre_tilize_cb`; its `reserve_back` at the end of a tile row arms the guard and the next row's `push_back` follows with no pack on that DFB id, so the assert fires on the second tile row. Fix: `dummy_pack(fast_tilize_cb_id)` before that push (PR #64001). With it the case gets past the assert and reaches Q7 (8x4 PCC 0.176); BH ttsim still 1/1 | open until the fix merges (found 2026-11-02) |
| Q13 | `ttnn.experimental.quasar.conv2d` `07_32768x64` with the watcher and LLK asserts on | 8x4: hangs in a DFB `.finish()` drain (watcher waypoint AAW for the rest of the run; killed after 30 min, not debugged). "3,2": `Neo2TRISC0 tripped an assert on line 165` in `experimental/quasar/matmul/device/kernels/compute/unified_matmul_compute.cpp` (unpack TRISC; file not identified), which aborts the process. Without them this case failed fast with Q4. | `TT_METAL_WATCHER=10ms TT_METAL_LLK_ASSERTS=1 pytest test_conv2d.py -k 07_32768x64` | none | open (found 2026-11-02) |

## 1b. Small-grid limits

The captured cases come from a 20-core Blackhole run; on 2 cores (craq-sim 2x1 stand-in and the emulator) some
of them cannot run as captured. These are limits of the capture or of the grid, not missing Quasar support.

| # | Op / case | Failure on 2 cores | Status |
|---|---|---|---|
| G1 | `matmul` 00/01, 6 conv2d cases | `Out of Memory: Not enough space to allocate … L1 buffer across 2 banks` — an input or output the capture placed interleaved in L1 (8 MB for matmul `00`'s weight) fits across 20 cores but not 2 | expected; needs a 2-core capture (2026-10-30: same with the routed `ttnn.experimental.quasar.matmul`) |
| G2 | 7 conv2d cases (and the model's res5.2 3x3 conv on Blackhole 2x1) | `DRAM Auto slice could not find valid slice configuration. Tried up to N slices …` — even the maximum width slicing does not fit 3.86 MB (Quasar) / 1.4 MB (Blackhole) of L1 per core at 512x1024 | open: the 2-core model profile needs explicit (e.g. channel) slicing for these convs |
| G3 | sharded cases of add, move, reshard, to_memory_config, conv2d | skipped: the captured shard grid needs more than 2 cores | expected |
| G4 | `max_pool2d` 00 (stem pool, DRAM-sliced on 2 cores) | `TT_FATAL @ tt_metal/impl/allocator/bank_manager.cpp:514: Out of Memory: Not enough space to allocate 1152 B L1_SMALL buffer across 2 banks, where each bank needs to store 576 B, but bank size is 4096 B (allocated: 3648 B, …)`, from `pool2d_create_program_artifacts` (the sliding-window config tensors) under the Quasar `op_slicing::run_sliced_op`. PDL opens the device with `l1_small_size` = 4 KB (`PDL_L1_SMALL_SIZE`). On 2 cores the pool needs many slices, and 3.6 KB was already in use when this allocation failed, so the per-slice config tensors probably accumulate (not confirmed). With `TTNN_GRAPH_OPS_L1_SMALL_SIZE=32768` (uncommitted suite switch) the case runs to the end (see Q7 for the values; 27 min at a steady ~35 KHz). Repro: `TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="3,2" pytest test_max_pool2d.py` | open (found 2026-10-30): the 2-core model profile needs a larger `l1_small_size` |

## 1c. Quasar emulator (emu-quasar-2x3, 2 worker cores)

2026-10-15, tt-umd-simulators `f8191c7` (local 2-channel DRAM yaml), `TTNN_GRAPH_OPS_BF8_AS_BF16=1`,
`TT_METAL_LLK_ASSERTS=1`, slow dispatch, one case per pytest process. Six small DRAM-interleaved cases
of ops that passed on craq-sim, one per op:

| Case | Result | Time |
|---|---|---|
| `to_device` 00 (1x256) | pass | 75 s |
| `to_layout` 00 (1x256) | pass without the watcher; with `TT_METAL_WATCHER=1` fails to build (Q11) | 82 s |
| `reshape` 04 (1x2048) | pass | 70 s |
| `to_memory_config` 05 (1x2048) | pass | 224 s |
| `upsample` 01 (1x256 → 32x64) | pass | 86 s |
| `add` 01 (2048x256, `ttnn.experimental.quasar.add`) | pass without the watcher; Q11 with it | 76 s |

2026-10-16, tt-metal main @ `2014821daa6`, watcher off, the craq-sim hangs (one case each, capped at 60 min):

| Case | Result | Time |
|---|---|---|
| `typecast` 00 (32x64x256) | pass (Q10 was a craq-sim-only hang, gone after the rebase) | 57 s |
| `avg_pool2d` 00 | no progress for 60 min after the DFB config write; killed (Q8) | 60 min |
| `max_pool2d` 00 | fails: Quasar pool DRAM slicing calls mainline `padded_slice` (Q7) | 397 s |
| `matmul` 00 | cannot run on 2 cores: captured 8 MB L1 weight (G1) | 95 s |

## 2. Simulator and harness issues — not Quasar gaps

| # | Issue | Detail | Workaround | Status |
|---|---|---|---|---|
| S1 | craq-sim `quasar` @ `889c806a` cannot run a 2x3 descriptor | It hard-codes the 32-core layout (`src/tile.cpp` `coord_to_tile`): the emulator's `quasar-2x3_emu.yaml` dies at device start with `UnimplementedFunctionality: coord_to_tile: coord (0,1)`; UMD's `tests/soc_descs/quasar_simulation_2x3.yaml` fails in Metal with `RuntimeError: bad conversion`. | 8x4 build with `TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="3,2"` gives the emulator's 2x1 compute grid and 2 DRAM banks | open |
| S2 | pytest `--timeout` does not stop a craq-sim hang | the hang is inside the simulator, which the signal-based timeout cannot interrupt | external watchdog kills a test process after 15–30 min (emulator: 60 min) | worked-around |
| S3 | craq-sim aborts on `ttnn.experimental.quasar.avg_pool2d` (Q8 case) with the watcher and LLK asserts on | `UnimplementedFunctionality: decode_and_execute_custom_0: funct3=4 reg_index=40 cmd_buf=0 data=0x0`, about 20 s after device init, on 8x4 and "3,2" (md5 `63d56a28…`). Caused by the watcher (2026-11-03): with `TT_METAL_LLK_ASSERTS=1` alone the Q8 case passes on 8x4 (12 s), and the Quasar pool unit case `max_k3x3_s2_h2_tile_c40` (`test_pool2d.py -k test_qpool_dram_sliced`) passes with asserts alone and aborts the same way (`reg_index=16 cmd_buf=1`) with the watcher alone. | run pools with LLK asserts on and the watcher off | open (found 2026-11-02); filed as tenstorrent/craq-sim#543 |
