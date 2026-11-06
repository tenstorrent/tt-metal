# Kimi-K3 MoonViT-V2 vision encoder

The vision path of [Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3): patch embedding (14x14 px
patches), 27 encoder blocks, the final norm, the 2x2 patch merger and the projector to the language
model's width (7168). One still image per call; no packing, no video.

Layout: a ring of 4 Blackhole chips along mesh axis 1 (`(1, 4)` mesh, 2 links, `FABRIC_2D_TORUS_X`);
on a LoudBox (2x4) the tests use the first row. Patches are put in window order (each 2x2 merge
window's four patches consecutive), padded to 65,536 rows and split along the ring, so the merger is a
reshape on each chip. Attention is ring attention with the padding masked; everything else is
sequence parallel. Weights are bf16 and replicated. Activations are bf16 in DRAM, except the encoder's
residual stream, which is fp32 (`tt/vision_block.py`). Matmuls are HiFi4 with fp32 accumulation.

## Structure

The TT code needs only `ttnn` and `torch` (plus numpy and Pillow in `tt/image_prep.py`) and imports
nothing from `reference/`. The reference needs `transformers`.

| Path | Contents |
|---|---|
| `reference/modeling_kimi_k3.py`, `reference/configuration_kimi_k3.py` | Copied from [moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3); the language model is removed |
| `reference/kimi_k3_vision.py` | `KimiK3VisionModel` (upstream's vision tower, then `mm_projector`) and `load_vision_config` (selects upstream's `eager` attention) |
| `reference/checkpoint.py` | Reads `vision_tower.*` and `mm_projector.*` from a Kimi-K3 HF checkpoint |
| `reference/config.json`, `LICENSE-Kimi-K3` | The K3 `vision_config` and the upstream license |
| `reference/media_utils.py`, `reference/kimi_k3_vision_processing.py`, `reference/preprocessor_config.json` | Upstream's image processor and its config; the golden for `tt/image_prep.py` |
| `tt/image_prep.py` | PIL image to `(pixel_values, grid_thws)`, matching the HF processor: resize, black padding to a multiple of 28 px, normalize, patchify |
| `tt/host_prep.py` | Device inputs built on the host: patch rows, position embeddings, RoPE cos/sin, window order, padding to `max_patches` rows |
| `tt/model_config.py` | `MoonViTConfig`, the grid check, `VisionParallel`, the compute kernel config |
| `tt/module.py` | `Module`, `ModuleList`, `Parameter`, state-dict loading, `save`/`load`, `load_cached` |
| `tt/linear.py`, `tt/rms_norm.py` | The leaf layers |
| `tt/ccl.py` | `ParallelFactor` and `TT_CCL` (links, topology, ring-attention semaphores and buffers) |
| `tt/device_input.py` | `DeviceInput`: a per-image input allocated once and rewritten in place, so a trace can replay it |
| `tt/patch_embed.py`, `rope.py`, `vision_attention.py`, `vision_mlp.py`, `vision_block.py` | One TT module per encoder part |
| `tt/patch_merger.py`, `tt/projector.py` | `TT_Patch_Merger` (no weights) and `TT_Projector` |
| `tt/vision_model.py` | `TT_MoonViT_Vision_Tower` (patch embed, encoder, merger) and `TT_MoonViT_Vision_Model` (tower, projector) |
| `tests/` | Shared helpers (`common.py`, `conftest.py`), CPU tests, one device test file per part, the full-size tests, and `goldens.py` (their saved reference outputs) |

## Torch and ttnn

Each TT module's children use the reference's names, so it loads the reference's `state_dict()`
directly with `load_torch_state_dict`. `load_cached` caches the loaded weights as `.tensorbin` files.

```python
ccl = TT_CCL(mesh_device, num_links=2, topology=ttnn.Topology.Ring)
tt_model = TT_MoonViT_Vision_Model(config, mesh_device=mesh_device, sequence_parallel=ParallelFactor(4, 1), ccl=ccl)
tt_model.load_torch_state_dict(reference.state_dict())
pixel_values, grid_thws = image_to_patches(Image.open(path), config)  # tt/image_prep.py
out = tt_model.run(pixel_values, grid_thws)  # (h*w, 3, 14, 14), [[1, h, w]] -> (h*w/4, 7168)
```

Images must be RGB; other modes raise `ValueError`. An image whose HF grid exceeds `max_patches`
(65,536 patches) is rejected, never shrunk, since shrinking would diverge from HF. That is roughly
anything over 12.8 MP, plus some smaller images the 28 px padding pushes over (3590x3590 -> 258x258).

TODO (`tt/image_prep.py`):

- Support non-RGB modes as HF does: convert to RGB *after* the resize (Pillow resizes P and 1 with
  nearest neighbour and RGBA with premultiplied alpha, so converting first does not match HF). Test
  against `reference/kimi_k3_vision_processing.py` for RGBA, LA, PA, L, P, 1, CMYK and I;16.
- Support `transparent_bg_config` and `transparent_bg_fill_stage` (`reference/preprocessor_config.json`):
  composite images with alpha (RGBA, LA, PA, and P or L with `tRNS`) onto HF's chessboard after the
  resize and before the black pad, with the values read from the config.

## Weights

Tests that build the `reference` fixture are marked `@SYNTHETIC`, `@REAL` or both, and run once per
mark (ids `synthetic` and `real`; select with `-m synthetic` or `-m real`):

- `synthetic`: seeded random weights, with norm weights drawn around 1 so a norm that fails to load
  shows up.
- `real`: the checkpoint at `KIMI_K3_HF_MODEL`; skipped when it is unset. Only the shards with
  `vision_tower.*` and `mm_projector.*` keys are read, so `model.safetensors.index.json`,
  `config.json`, `model-00095-of-000096.safetensors` and `model-00096-of-000096.safetensors` are
  enough.

CPU tests use random weights only (`test_load_checkpoint.py` reads the checkpoint itself). On device,
only attention, the MLP, the projector and what contains them (block, full model, full size) use the
checkpoint.

Per-layer tests run layers 0, 13 and 26 (`@SAMPLED_LAYERS`), or only the layer in
`MOONVIT_TEST_LAYER`. The full-size layer test runs all 27 (`@ALL_LAYERS`) and ignores the variable.

## Tests

Each test runs a TT module and the reference on the same input and compares them with
`assert_quality` (`tests/common.py`): PCC and relative RMSE (RMS error over the reference's standard
deviation) are asserted, CCC and the largest differences are logged.

```bash
# CPU
pytest models/experimental/kimi_k3_moonvit_v2/tests/test_reference.py \
       models/experimental/kimi_k3_moonvit_v2/tests/test_image_prep.py \
       models/experimental/kimi_k3_moonvit_v2/tests/test_host_prep.py \
       models/experimental/kimi_k3_moonvit_v2/tests/test_load_checkpoint.py \
       models/experimental/kimi_k3_moonvit_v2/tests/test_module.py \
       models/experimental/kimi_k3_moonvit_v2/tests/test_goldens.py \
       models/experimental/kimi_k3_moonvit_v2/tests/test_quality.py

# Device, one P150: the parts that work on each row alone
pytest models/experimental/kimi_k3_moonvit_v2/tests/test_linear.py \
       models/experimental/kimi_k3_moonvit_v2/tests/test_rms_norm.py \
       models/experimental/kimi_k3_moonvit_v2/tests/test_rope.py \
       models/experimental/kimi_k3_moonvit_v2/tests/test_patch_embed.py \
       models/experimental/kimi_k3_moonvit_v2/tests/test_vision_mlp.py \
       models/experimental/kimi_k3_moonvit_v2/tests/test_patch_merger.py \
       models/experimental/kimi_k3_moonvit_v2/tests/test_projector.py

# Device, LoudBox ring: everything with attention
pytest models/experimental/kimi_k3_moonvit_v2/tests/test_vision_attention.py \
       models/experimental/kimi_k3_moonvit_v2/tests/test_vision_block.py \
       models/experimental/kimi_k3_moonvit_v2/tests/test_vision_model.py

# Simulator (ttsim): the `_small` variants of the row-wise tests; also run on hardware.
# With TT_METAL_SIMULATOR set, every other device test is skipped.
pytest models/experimental/kimi_k3_moonvit_v2/tests -m simulator
```

Ring tests (`ring1x4`) run on row 0 of any system mesh 4 chips wide (LoudBox 2x4, Galaxy 8x4) and skip
elsewhere. CI (Blaze models prefill, `kimi_k3_moonvit`) runs `-m synthetic -k real_rows_from_each_chip` on a Galaxy.
Row-wise tests run on one P150 (`p150`) standing in for one ring chip: it runs that chip's 16,384-row
quarter of the padded sequence (`MeshRows`).

Image sizes (`tests/common.py`):

| Tag | Used by | Images | Golden |
|---|---|---|---|
| `PADDING_LAYOUT_GRIDS` | Row-wise tests | `full` 256x256 and `max_side` 512x128 fill the padded length, `partial` 250x250 ends inside a tile, `small` 40x100 is mostly padding | The real rows of the quarter the P150 runs (the last ring chip holding real rows) |
| `ATTENTION_GRIDS` | Attention, block, full model | 64x64, 40x100; all real rows on chip 0 | Every row; kept small because the golden builds L x L scores |
| `REAL_ROWS_FROM_EACH_CHIP_ASSERTED_GRIDS` | `*_real_rows_from_each_chip` attention and block tests | `full`, `max_side`, `partial` | fp32 on `SAMPLED_ROWS_PER_CHIP` rows of each chip, checked chip by chip |
| `SIMULATOR_GRID` | `_small` tests | 6x20 in a config padded to 512 rows (128 per chip) | Every row |

Full size (`test_full_size.py`, LoudBox, checkpoint only): a 256x256 image fills all 65,536 rows.
Each layer, the full encoder and the full model are compared against reference outputs saved once
by `tests/goldens.py`. They are fp32 because the bf16 reference rounds more than the device in
deeper layers. The tests skip until the outputs exist:

```bash
export KIMI_K3_HF_MODEL=/path/to/Kimi-K3
export MOONVIT_GOLDEN_DIR=/path/to/goldens
python -m models.experimental.kimi_k3_moonvit_v2.tests.goldens --grid 256x256   # resumable
pytest models/experimental/kimi_k3_moonvit_v2/tests/test_full_size.py
```

The `_trace` variants (layer, encoder, model) capture the same forward in a trace, require the first
replay to match the untraced run exactly, then log the time per replay. The model one also logs a
host-in, host-out run.

```bash
pytest models/experimental/kimi_k3_moonvit_v2/tests/test_full_size.py -k trace
```
