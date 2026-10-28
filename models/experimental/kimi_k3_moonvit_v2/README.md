# Kimi-K3 MoonViT-V2 vision encoder

The vision path of [Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3): patch embedding,
27 encoder blocks, the final norm, the 2x2 patch merger and the projector into the language
model's width (7168).

Scope: one still image per call. No packing, no video. The encoder takes 14x14 px patches; the
merger turns each 2x2 window into one 28 px token, and the projector maps it to a language-model
embedding.

Target layout: the 4 Blackhole chips of a QuietBox 2 in a ring along mesh axis 1 (`(1, 4)` mesh, `FABRIC_1D_RING`,
2 links). Each image's patches are put in window order (each 2x2 merge window's four patches
consecutive), padded to 65,536 rows and split along the ring, so the merger is a reshape on each chip. Attention is
ring attention with the padding masked; norms, MLP, merger and projector are sequence parallel. Weights are bf16 and
replicated, activations DRAM interleaved, matmuls HiFi4 with fp32 accumulation.

## Structure

The TT code depends only on `ttnn` and `torch`, and imports nothing from `reference/`. The reference
needs `transformers`, as upstream does; tests also use `models.common.utility_functions`.

| Path | Contents |
|---|---|
| `reference/modeling_kimi_k3.py` | Upstream `modeling_kimi_k3.py` without the language model (`KimiK3PreTrainedModel`, `KimiK3ForConditionalGeneration` and their imports) |
| `reference/configuration_kimi_k3.py` | Upstream `configuration_kimi_k3.py`, unchanged |
| `reference/kimi_k3_vision.py` | `KimiK3VisionModel`: the vision path of `KimiK3ForConditionalGeneration` (tower, then `mm_projector`), and `load_vision_config`, which selects upstream's `eager` attention |
| `reference/checkpoint.py` | Reads `vision_tower.*` and `mm_projector.*` from a Kimi-K3 HF checkpoint |
| `reference/config.json`, `LICENSE-Kimi-K3` | The K3 `vision_config` and the license of the vendored code |
| `tt/host_prep.py` | Host prep for the device, torch only: patch rows, position embeddings, RoPE cos/sin, window order, padding |
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
`comp_pcc` from `models.common.utility_functions`.

```python
ccl = TT_CCL(mesh_device, num_links=2, topology=ttnn.Topology.Ring)
tt_model = TT_MoonViT_Vision_Model(config, mesh_device=mesh_device, sequence_parallel=ParallelFactor(4, 1), ccl=ccl)
tt_model.load_torch_state_dict(reference.state_dict())
out = tt_model.run(pixel_values, grid_thws)  # (h*w, 3, 14, 14), [[1, h, w]] -> (h*w/4, 7168)
```

## Weights

The `reference` (every layer) and `shallow_reference` (the first two layers, for the full-model
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
# CPU: reference, host prep, checkpoint loader, TT module names (device upload stubbed)
pytest models/experimental/kimi_k3_moonvit_v2/tests/test_reference.py \
       models/experimental/kimi_k3_moonvit_v2/tests/test_host_prep.py \
       models/experimental/kimi_k3_moonvit_v2/tests/test_load_checkpoint.py \
       models/experimental/kimi_k3_moonvit_v2/tests/test_module.py

# Device, one P150 or 4 Blackhole chips: parts that work on each row alone
pytest models/experimental/kimi_k3_moonvit_v2/tests/test_linear.py \
       models/experimental/kimi_k3_moonvit_v2/tests/test_rms_norm.py \
       models/experimental/kimi_k3_moonvit_v2/tests/test_rope.py \
       models/experimental/kimi_k3_moonvit_v2/tests/test_patch_embed.py \
       models/experimental/kimi_k3_moonvit_v2/tests/test_vision_mlp.py \
       models/experimental/kimi_k3_moonvit_v2/tests/test_patch_merger.py \
       models/experimental/kimi_k3_moonvit_v2/tests/test_projector.py

# Device, 4 Blackhole chips: attention needs the ring
pytest models/experimental/kimi_k3_moonvit_v2/tests/test_vision_attention.py \
       models/experimental/kimi_k3_moonvit_v2/tests/test_vision_block.py \
       models/experimental/kimi_k3_moonvit_v2/tests/test_vision_model.py
```

Each row-wise test has a `ring1x4` case and a `p150` case, where one chip stands in for one chip of
the ring and gets the first quarter of the padded sequence (16,384 rows);
pytest skips the cases that need more chips than the machine has. Device tests that need a forward are marked `xfail(raises=NotImplementedError, strict=True)`, so
each starts failing as a reminder once its forward runs.
