# Kimi-K3 MoonViT-V2 vision encoder

The vision path of [Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3): patch embedding,
27 encoder blocks, the final norm, the 2x2 patch merger and the projector into the language
model's width (7168).

Scope: one still image per call. No packing, no video. The encoder takes 14x14 px patches; the
merger turns each 2x2 window into one 28 px token, and the projector maps it to a language-model
embedding.

Target layout: 4 Blackhole chips in a ring along mesh axis 1 (`(1, 4)` mesh, 2 links). On a LoudBox
(8 chips, 2x4) that is one row; the tests open all 8 chips and run on the first row
(`tests/conftest.py`), with `FABRIC_2D_TORUS_X`.
Each image's patches are put in window order (each 2x2 merge window's four patches consecutive),
padded to 65,536 rows and split along the ring, so the merger is a reshape on each chip. Attention is
ring attention with the padding masked; norms, MLP, merger and projector are sequence parallel.
Weights are bf16 and replicated, activations DRAM interleaved, matmuls HiFi4 with fp32 accumulation.
Activations are bf16 except the encoder's residual stream, which is fp32 (`tt/vision_block.py`).

## Structure

The TT code depends only on `ttnn` and `torch` (plus numpy and Pillow in `tt/image_prep.py`), and
imports nothing from `reference/`. The reference
needs `transformers`, as upstream does; tests also use `models.common.utility_functions`.

| Path | Contents |
|---|---|
| `reference/modeling_kimi_k3.py` | Upstream `modeling_kimi_k3.py` without the language model (`KimiK3PreTrainedModel`, `KimiK3ForConditionalGeneration` and their imports) |
| `reference/configuration_kimi_k3.py` | Upstream `configuration_kimi_k3.py`, unchanged |
| `reference/kimi_k3_vision.py` | `KimiK3VisionModel`: the vision path of `KimiK3ForConditionalGeneration` (tower, then `mm_projector`), and `load_vision_config`, which selects upstream's `eager` attention |
| `reference/checkpoint.py` | Reads `vision_tower.*` and `mm_projector.*` from a Kimi-K3 HF checkpoint |
| `reference/config.json`, `LICENSE-Kimi-K3` | The K3 `vision_config` and the license of the vendored code |
| `reference/media_utils.py`, `reference/kimi_k3_vision_processing.py`, `reference/preprocessor_config.json` | Upstream's image processor (`KimiK3VisionProcessor`) and its config, unchanged: the golden for `tt/image_prep.py` |
| `tt/image_prep.py` | A PIL image to `(pixel_values, grid_thws)` as the HF processor makes them: resize, black padding to a multiple of 28 px, normalize, patchify |
| `tt/host_prep.py` | Host prep for the device, torch only: patch rows, position embeddings, RoPE cos/sin, window order, padding to `max_patches` rows |
| `tt/model_config.py` | `MoonViTConfig`, the grid check, `VisionParallel`, the compute kernel config |
| `tt/module.py` | `Module`, `ModuleList`, `Parameter`, state-dict loading, `save`/`load`, `load_cached` |
| `tt/linear.py`, `tt/rms_norm.py` | The leaf layers |
| `tt/ccl.py` | `ParallelFactor` and `TT_CCL` (links, topology, ring-attention semaphores and buffers) |
| `tt/device_input.py` | `DeviceInput`: a per-image input allocated once and rewritten in place, so a trace can replay it |
| `tt/patch_embed.py`, `rope.py`, `vision_attention.py`, `vision_mlp.py`, `vision_block.py` | One TT module per encoder part |
| `tt/patch_merger.py`, `tt/projector.py` | `TT_Patch_Merger` (no weights) and `TT_Projector` |
| `tt/vision_model.py` | `TT_MoonViT_Vision_Tower` (patch embed, encoder, merger) and `TT_MoonViT_Vision_Model` (tower, projector) |
| `tests/` | `common.py` and `conftest.py`, CPU tests for the reference and host prep, one device test per part |

## Torch and ttnn

The TT modules are `tt/module.py` `Module`s whose children carry the reference's names, so each
loads the matching reference `state_dict()` with `load_torch_state_dict`. `load_cached` caches the
loaded weights as `.tensorbin` files. Tests build both, run the same input, and compare with
`assert_quality` (`tests/common.py`): PCC (`comp_pcc`) and relative RMSE (RMS error over the reference's
standard deviation) are asserted; CCC and the largest element differences are logged.

```python
ccl = TT_CCL(mesh_device, num_links=2, topology=ttnn.Topology.Ring)
tt_model = TT_MoonViT_Vision_Model(config, mesh_device=mesh_device, sequence_parallel=ParallelFactor(4, 1), ccl=ccl)
tt_model.load_torch_state_dict(reference.state_dict())
pixel_values, grid_thws = image_to_patches(Image.open(path), config)  # tt/image_prep.py
out = tt_model.run(pixel_values, grid_thws)  # (h*w, 3, 14, 14), [[1, h, w]] -> (h*w/4, 7168)
```

Image limit: an image is accepted when the HF processor's grid fits `max_patches` (65,536 patches).
That rejects nearly every image over about 12.8 MP, and some smaller ones that the 28 px padding
pushes over (3590x3590 -> 258x258). Over-limit images are rejected, never shrunk, because shrinking
would diverge from HF. Only RGB images are accepted; other modes (including anything with
transparency, which HF composites onto a chessboard) raise `ValueError`.

TODO (image pre-processing, `tt/image_prep.py`):

- Handle non-RGB images as HF does, instead of rejecting them: convert other modes to RGB *after*
  the resize (Pillow resizes P and 1 with nearest neighbour and RGBA with premultiplied alpha, so
  converting first does not match HF). Test against `reference/kimi_k3_vision_processing.py` for
  RGBA, LA, PA, L, P, 1, CMYK and I;16.
- Handle `transparent_bg_config` and `transparent_bg_fill_stage` (`reference/preprocessor_config.json`):
  composite images with alpha (RGBA, LA, PA, and P or L with `tRNS`) onto an 8 px chessboard
  (white 255, gray 180, white square top left), after the resize and before the black pad. Read the
  values from the config rather than hard-coding them.

## Weights

The `reference` (every layer) and `shallow_reference` (the first two layers, for the CPU
tests) fixtures in `tests/conftest.py` are parametrized `random` and `checkpoint`:

- `random`: the reference initialized under `torch.manual_seed(0)`, with norm weights drawn around
  one so a norm that fails to load shows up.
- `checkpoint`: the vision tower and projector of the checkpoint at `KIMI_K3_HF_MODEL`. The
  loader opens only the shards holding `vision_tower.*` and `mm_projector.*` keys, so a folder with `model.safetensors.index.json`,
  `config.json`, `model-00095-of-000096.safetensors` and `model-00096-of-000096.safetensors` is
  enough. These cases skip when the variable is unset.

Per-layer tests (attention, MLP, block, encoder linears and norms) take a `layer` parameter: layer
0 with random weights, and layers 0, 13 and 26 with the checkpoint. `MOONVIT_TEST_LAYERS=5` or
`MOONVIT_TEST_LAYERS=0,20` sets the layers for both.

## Tests

```bash
# CPU: reference, image and host prep, checkpoint loader, TT module names (device upload stubbed)
pytest models/experimental/kimi_k3_moonvit_v2/tests/test_reference.py \
       models/experimental/kimi_k3_moonvit_v2/tests/test_image_prep.py \
       models/experimental/kimi_k3_moonvit_v2/tests/test_host_prep.py \
       models/experimental/kimi_k3_moonvit_v2/tests/test_load_checkpoint.py \
       models/experimental/kimi_k3_moonvit_v2/tests/test_module.py

# Device, one P150 or a LoudBox: parts that work on each row alone
pytest models/experimental/kimi_k3_moonvit_v2/tests/test_linear.py \
       models/experimental/kimi_k3_moonvit_v2/tests/test_rms_norm.py \
       models/experimental/kimi_k3_moonvit_v2/tests/test_rope.py \
       models/experimental/kimi_k3_moonvit_v2/tests/test_patch_embed.py \
       models/experimental/kimi_k3_moonvit_v2/tests/test_vision_mlp.py \
       models/experimental/kimi_k3_moonvit_v2/tests/test_patch_merger.py \
       models/experimental/kimi_k3_moonvit_v2/tests/test_projector.py

# Device, a LoudBox: attention needs the ring
pytest models/experimental/kimi_k3_moonvit_v2/tests/test_vision_attention.py \
       models/experimental/kimi_k3_moonvit_v2/tests/test_vision_block.py \
       models/experimental/kimi_k3_moonvit_v2/tests/test_vision_model.py
```

Each row-wise test has a `ring1x4` case and a `p150` case, where one chip stands in for one chip of
the ring and gets the first quarter of the padded sequence (16,384 rows);
pytest skips the cases that need more chips than the machine has.

Image sizes (`tests/common.py`): the row-wise tests (`ROW_GRIDS`) compare against a CPU golden of
the full padded length, which takes seconds since nothing there scales with the square of the
length. Each size has one grid per mesh: `full` (256x256 on the ring, 128x128 on one chip) and
`max_side` (512x128, 512x32) fill the padded length exactly, `partial` (250x250, 122x130) ends
inside a tile, and `small` (40x100) is almost all padding. Attention, the block and the
full model (`ATTENTION_GRIDS`, 64x64 and 40x100) stay small: their CPU golden builds L x L scores per head.

Full size (`test_full_size.py`, LoudBox, checkpoint only): a 256x256 image fills all 65,536
rows, and each encoder layer on its own, the full 27-layer encoder and the full model (merger and
projector included) are compared against reference outputs saved once by `tests/goldens.py`. They are fp32, with attention computed in chunks of query
rows (~1.5 min and ~25 GB of RAM per layer, about 45 min for all 27): the bf16 reference rounds more
than the device from about layer 9 on, and upstream's eager attention in fp32 would need about 1 TB.
`--dtype bf16` writes the bf16 eager reference instead. The outputs go in `$MOONVIT_GOLDEN_DIR`
(about 7.3 GB for 256x256 fp32); the tests skip while it is unset or the outputs do not exist:

```bash
export KIMI_K3_HF_MODEL=/path/to/Kimi-K3
export MOONVIT_GOLDEN_DIR=/path/to/goldens
python -m models.experimental.kimi_k3_moonvit_v2.tests.goldens --grid 256x256   # resumable
pytest models/experimental/kimi_k3_moonvit_v2/tests/test_full_size.py
```

Traced perf tests, full size only (`test_full_size_layer_trace` for one layer,
`test_full_size_encoder_trace`): the forward runs once untraced, then is captured in one trace
(`Trace` in `tests/common.py`). The first replay must equal the untraced run exactly and pass the
untraced test's checks. Then `TRACE_ITERATIONS` replays run back to back with one wait at the end,
and the time per replay is logged:

```bash
pytest models/experimental/kimi_k3_moonvit_v2/tests/test_full_size.py -k trace
```

Device tests that need a forward are marked `xfail(raises=NotImplementedError, strict=True)`, so
each starts failing as a reminder once its forward runs.
