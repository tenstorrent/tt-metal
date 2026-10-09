# SDXL base UNet on Quasar (1024x1024)

Port of `models/demos/stable_diffusion_xl_base` (UNet + Euler scheduler, 1024x1024 only) to
Quasar, with every device op routed to `ttnn.experimental.quasar.*`. Mirrors the layout of
`models/demos/vision/classification/resnet50/quasar/`.

Status: **plumbing only**. The model code and all tests are in place; 13 of the 27 ops the
model needs do not exist under `ttnn.experimental.quasar` yet and the 14 that exist have not
been validated with these shapes / configs. See [docs/OPS.md](docs/OPS.md) for the inventory
and `tests/ops/` for one unit test per op.

## Layout

| path | what |
|---|---|
| `tt/qsr.py` | op router: `qsr.<op>(...)` -> `ttnn.experimental.quasar.<op>` resolved at call time; missing ops raise `NotImplementedError` naming the unit test. `SDXL_QSR_FALLBACK_MAINLINE=1` routes missing ops to mainline `ttnn` instead. `qsr.from_torch` = host tilize + `quasar.to_device`. |
| `tt/model_configs.py` | `load_model_optimisations((1024, 1024))` -> `ModelOptimisations1024x1024Quasar`, which inherits the Wormhole 1024x1024 configs (8x8 grid) as a starting point. Retune here for the 8x4 Quasar grid. |
| `tt/tt_*.py`, `tt/sdxl_utility.py` | copies of the Wormhole modules with `ttnn.<op>(` -> `qsr.<op>(`. Refiner / LoRA / Blackhole branches are untouched (inactive). |
| `tests/pcc/` | the module tests (attention, geglu, feedforward, transformer block / model, resnet block, down/up sample, down/mid/up blocks, embedding, timesteps, unet, scheduler, unet loop) with only the 1024x1024 parametrizations. Weights: HF cache if present, else random from the HF config (`sdxl_quasar_test_utils.load_torch_unet`). |
| `tests/ops/test_<op>.py` | generated per-op unit tests, one case per distinct call signature in the model. `tests/ops/op_case.py` executes a case (random inputs, torch golden, PCC). |
| `tests/ops/generate_cases.py` | regenerates `tests/ops/test_*.py` and `docs/OPS.md` from `docs/trace_1024.json.gz`. |
| `docs/trace_sdxl.py` | the shape tracer that produced `docs/trace_1024.json.gz` (runs the Wormhole model under a fake `ttnn`, no device needed). |

## Running on craq-sim

Same environment as the other Quasar model tests (slow dispatch, simulator paths), e.g.

```bash
export TT_METAL_HOME=$PWD TT_METAL_RUNTIME_ROOT=$PWD PYTHONPATH=$PWD:$PWD/ttnn \
       TT_METAL_SIMULATOR=<sim>/libttsim.so TT_METAL_SIMULATOR_HOME=<sim> \
       TT_METAL_SLOW_DISPATCH_MODE=1 ARCH_NAME=quasar
# one op case
pytest models/demos/stable_diffusion_xl_base/quasar/tests/ops/test_add_.py -k 00_1x1280
# every case of one op
pytest models/demos/stable_diffusion_xl_base/quasar/tests/ops/test_group_norm.py
# a module test (fails on the first op that is not ported, naming it)
pytest models/demos/stable_diffusion_xl_base/quasar/tests/pcc/test_module_tt_resnetblock2d.py -k "320-64-64"
```

Grid handling in the op tests (`SDXL_QSR_GRID`):

* `fit` (default): captured 8x8 shard grids are remapped onto the device grid (same strategy,
  shard shapes recomputed); matmul / layer-norm / SDPA program configs and the conv output-grid
  override are dropped so the op auto-configures; group-norm `core_grid` follows the remapped
  input grid. This checks the op works for the model's shapes on Quasar.
* `captured`: exactly the traced configs (needs an 8x8 grid).

Shrink knobs for simulator bring-up (the functional simulator is roughly serial over the work, so a
full-size case can take many minutes): `SDXL_QSR_CONV_SPATIAL_DIV=n` divides the conv spatial size,
`SDXL_QSR_SDPA_SEQ_DIV=n` divides the attention query sequence (and the key/value sequence for
self-attention; chunk sizes are clamped), `SDXL_QSR_SDPA_HEADS=n` caps the number of heads (one Q chunk
on one core with `SEQ_DIV=8 HEADS=1`). The shrunk shapes are only for bring-up; the recorded results below
are full size unless stated.

## Porting an op

1. `pytest tests/ops/test_<op>.py` lists the exact shapes / layouts / memory configs / kwargs
   the model needs (also in `docs/OPS.md`, with call counts and calling modules).
2. Add the op under `ttnn/cpp/ttnn/operations/experimental/quasar/<op>/` and bind it as
   `ttnn.experimental.quasar.<op>`; `qsr` picks it up with no model change.
3. Make the per-op cases pass, then the module test that uses it (`tests/pcc/`).

## Regenerating the op list

```bash
# trace the Wormhole model (no device; ~10 GB RAM for the random-weight UNet)
PYTHONPATH=$PWD:$PWD/ttnn python models/demos/stable_diffusion_xl_base/quasar/docs/trace_sdxl.py \
    models/demos/stable_diffusion_xl_base/quasar/docs/trace_1024.json.gz
python models/demos/stable_diffusion_xl_base/quasar/tests/ops/generate_cases.py \
    models/demos/stable_diffusion_xl_base/quasar/docs/trace_1024.json.gz
```

## Observed on craq-sim (2026-10-09, harness validation only)

| case | result |
|---|---|
| `test_add_.py -k 00_1x1280` (DRAM interleaved, fused SILU) | pass |
| `test_to_memory_config.py -k 02_1x1x16384x320_l1_il` (row-major, interleaved -> block sharded, grid remapped to 8x4) | pass |
| `test_sharded_to_interleaved.py -k 01_1x1x4096x320` | pass |
| `test_slice.py -k 00_`, `test_reshape.py -k 00_` | pass |
| `test_silu.py -k 00_` | `NotImplementedError` from `qsr` (op missing), as designed |
| `test_linear.py -k 00_1x320` | with bfp8_b weights: `TT_FATAL ... Bfp8_b ... not supported on architecture quasar`; with bf16 weights: `TT_FATAL: DataMovementKernel is not supported on Quasar` (this `[1,320] x [320,1280] + bias + silu` shape with no program config lands in a legacy matmul factory) |
| `test_move.py` (2), `test_unsqueeze.py` (4), `test_upsample.py` (2) | pass |
| `test_scaled_dot_product_attention.py`, all 4 cases at full size on the 8x4 grid | pass (sim time: case 00 q=k=4096 386 s, 01 13 s, 02 52 s, 03 9 s) |

## Quasar-native sync notes (learned while porting)

* `upsample` (sharded, nearest) and the tiled `reshape` behind `unsqueeze` use DFB **implicit sync**:
  `noc.async_read<NocOptions::TXN_ID>(accessor, dfb, {.page_id}, {})` / `async_write<TXN_ID>` plus
  `finish()`, no reserve/push/pop, credits posted by the DM0 ISR. `upsample` runs a reader and a writer
  kernel with `num_threads` DM threads each (default 2, `TT_METAL_QSR_UPSAMPLE_THREADS`, must divide the
  output shard height) over a plain L1 staging ring, no host config tensor.
* A DFB **borrowed from a tensor cannot be an implicit-sync endpoint**: the producer's `finish()` spins at
  waypoint `WTP2` forever because the ISR never posts (the in-tree borrowed-memory DFB test opts out of
  implicit sync for the same reason). Stage through a plain DFB and write the tensor with the accessor.
* Reads whose result the same kernel must parse (the reshape page map) keep explicit credits; opt a single
  DFB out with `config_2xx->disable_dfb_implicit_sync_for = {dfb}` instead of disabling it for the kernel.
* **`~DataflowBuffer()` drains on Quasar** (since upstream 18819a6cb52): only the object a DFB was first
  constructed from drains, copies never do. Every SDPA helper used to build its own `DataflowBuffer` per
  call, so each call ended in a drain waiting for the other thread (reader draining K before pushing Q,
  writer draining the identity-scale tile compute keeps all kernel) and all four SDXL SDPA cases hung
  (watcher: DMs `AAW`, unpack `UPMW` in the first QK matmul). Fix in
  `transformer/sdpa/device/kernels/dfb_registry.hpp`: kernels create one original per DFB with
  `sdpa_dfb::original(id)`, helpers take `sdpa_dfb::view(id)` (a non-draining copy), and the kernel ends with
  the pops of the entries it kept fronted plus `sdpa_dfb::finish_all()`. The originals live in static
  storage because ~20 of them on the stack overflowed the 4 KB TRISC local memory into the firmware TLS.
  `kernel_lib/reduce_helpers_dataflow` gained `DataflowBuffer&` overloads for the same reason.
* `TT_METAL_WATCHER` and `TT_METAL_DPRINT_CORES` cannot be on together on Quasar (pack TRISC firmware
  too big); DPRINT needs `TT_METAL_INSPECTOR` enabled (the print server resolves the kernel ELF through it).

## Known gaps / notes

* **No `bfloat8_b` on Quasar.** `tt_metal/common/tt_backend_api_types.cpp::is_supported_quasar`
  accepts Float16/Float16_b/Float32/Fp8_e4m3/Int*/MxFp*/MxInt* only; a Metal 2.0 op with a
  bfp8_b DFB hits `TT_FATAL ... not supported on architecture quasar` (seen with `linear`). The
  Wormhole model keeps attention / feed-forward / conv weights and the group-norm masks in
  bfp8_b. `ModelOptimisations1024x1024Quasar` therefore defaults all weight dtypes to bf16 and
  the op tests build bfp8_b tensors as bf16 (`SDXL_QSR_KEEP_BFP8=1` to keep them). bf16 weights
  are ~5.2 GB for the UNet; the craq-sim SoC has 2 x 1 GiB DRAM, so the full-UNet tests
  (`test_module_tt_unet.py`, `test_unet_loop.py`) will need weight streaming or MX formats on
  the simulator even once every op exists. The module tests fit.

* All configs are the Wormhole 8x8-grid ones. Quasar is 8x4 with 4 MB L1 per cluster; the
  module tests will need `ModelOptimisations1024x1024Quasar` retuned once the ops exist.
* `test_unet_loop.py` is reduced to UNet + scheduler with random prompt embeddings (no CLIP, no
  trace capture). `test_sdxl_clip_encoders.py` (tt_dit CLIP encoder) and the VAE / LoRA / refiner
  tests are not part of this port.
* The op tests for `conv2d` / `linear` / `matmul` run with the op's auto program config in `fit`
  mode; the traced program configs are kept in the cases for when the 8x4 configs are tuned.
* Inputs that the Wormhole tests built with on-device `permute` + `reshape` (NCHW -> [B,1,HW,C])
  are prepared in torch on host (`sdxl_quasar_test_utils.to_device_nhwc`), so `permute` is not
  in the op list.
