# Dots OCR (TTNN)

TTNN-based implementation of the Dots OCR vision-language model family (HF: `rednote-hilab/dots.mocr`) with:

- Modular design (patch embedding, feature extractor, decoder)
- PyTorch reference implementation for correctness
- TTNN implementation optimized for Tenstorrent devices
- Demo script + performance benchmark

Run demo in TTNN:

```bash
python -m models.demos.dots_ocr.tests.demo.demo \
  --image models/demos/dots_ocr/tests/demo/image.png \
  --backend ttnn \
  --vision-backend ttnn
```

Run HF reference only:

```bash
python -m models.demos.dots_ocr.tests.demo.demo --image models/demos/dots_ocr/tests/demo/test12.png --backend hf
```

Text prefill PCC:

```bash
pytest models/demos/dots_ocr/tests/pcc/test_text_prefill_pcc.py
```

Vision Transformer PCC:

```bash
pytest models/demos/dots_ocr/tests/pcc/test_vision_transformer_pcc.py
```

### Supported Wormhole topologies

`dots.mocr` is GQA with `num_attention_heads=12` and `num_key_value_heads=2`.
Tensor parallelism shards heads along `cluster_shape[1]` and the base
`ModelArgs` asserts `n_kv_heads % cluster_shape[1] == 0`, so only TP degrees
that divide `gcd(12, 2) = 2` are supported: **1 or 2**.


| `MESH_DEVICE`                    | Mesh shape                                               | TP     | Status                                                                                                                  |
| -------------------------------- | -------------------------------------------------------- | ------ | ----------------------------------------------------------------------------------------------------------------------- |
| `N150`                           | 1x1                                                      | 1      | fully supported                                                                                                         |
| `N300`                           | 1x2                                                      | 2      | fully supported                                                                                                         |
| `T3K` (8-device Wormhole LLMBox) | physical `1×8`; dots.mocr runs on logical `1×1` or `1×2` | 1 or 2 | Default: open full 8-device mesh then submesh (`DOTS_T3K_OPEN_FULL_MESH=1`). Set `DOTS_T3K_TP=2` for TP across 2 chips. |
| `TG` / Galaxy                    | —                                                        | —      | not supported (needs DP, not implemented)                                                                               |


## Notes

- The full `rednote-hilab/dots.mocr` checkpoint is large; CI runs skip device tests unless `MESH_DEVICE` is set.
- Set `HF_MODEL` if not using the default repo id.
- Always pass `--confcutdir=models/demos/dots_ocr/tests` when running this folder’s tests so the repo root `conftest` does not override TTNN / device behavior.

### Eager attention vs `flash_attn` (and how `qwen25_vl` differs)

This demo targets **Tenstorrent device** inference and **does not require CUDA or the real
`flash_attn` wheel**. Reference logits use Hugging Face **eager** attention:

- `reference/hf_utils.py` loads with `_attn_implementation="eager"` and `use_safetensors=True`.

`**models/demos/qwen25_vl`** ships a **vendored** `reference/model.py`. It follows the usual
Transformers pattern: `if is_flash_attn_2_available(): import flash_attn … else: … = None`, and
maps `config._attn_implementation` to classes including `"eager"` and `"sdpa"` (see
`QWEN2_5_VL_ATTENTION_CLASSES` / vision equivalents). Demos often call `from_pretrained` with
`dtype` / `device_map` only; on GPU, the default may be SDPA.

**Dots OCR** uses **remote** Hub code (`trust_remote_code`). Transformers runs `**check_imports`**
on that file *before* `from_pretrained` kwargs take effect, so a top-level `import flash_attn` in
the checkpoint must still import. We register a minimal `**reference/flash_attention_shim.py`** so that
check passes; it is **not** FlashAttention at runtime — **eager** is. If the real `flash_attn`
package is installed, we do not replace it.
