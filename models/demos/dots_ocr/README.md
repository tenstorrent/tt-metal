# Dots OCR (TTNN)

TTNN-based implementation of the Dots OCR vision-language model family (HF: `rednote-hilab/dots.mocr`) with:
- Modular design (patch embedding, feature extractor, decoder)
- PyTorch reference implementation for correctness
- TTNN implementation optimized for Tenstorrent devices
- Unit tests (per-module) + end-to-end test with PCC validation
- Demo script + performance benchmark

## Quick start

Install extra dependencies:

```bash
pip install -r models/demos/dots_ocr/requirements.txt
```

Run unit tests (device-free subset by default):

```bash
pytest models/demos/dots_ocr/tests -q
```

Run demo with full TTNN vision (recommended):

```bash
MESH_DEVICE=N300 HF_MODEL=rednote-hilab/dots.mocr python -m models.demos.dots_ocr.demo.demo --image path/to/image.png --backend ttnn
```

Run HF reference only:

```bash
HF_MODEL=rednote-hilab/dots.mocr python -m models.demos.dots_ocr.demo.demo --image path/to/image.png --backend hf
```

Run HF reference OCR demo (image → text):

```bash
PYTHONPATH=$(pwd) python3 -m models.demos.dots_ocr.demo.reference_demo \
  --input models/demos/dots_ocr/demo/test12.png \
  --dtype fp32 \
  --use-slow-processor \
  --ocr-preset en \
  --num-beams 1 \
  --max-new-tokens 64
```

### Supported Wormhole topologies

`dots.mocr` is GQA with `num_attention_heads=12` and `num_key_value_heads=2`.
Tensor parallelism shards heads along `cluster_shape[1]` and the base
`ModelArgs` asserts `n_kv_heads % cluster_shape[1] == 0`, so only TP degrees
that divide `gcd(12, 2) = 2` are supported: **1 or 2**.

| `MESH_DEVICE` | Mesh shape | TP | Status |
|---|---|---|---|
| `N150` | 1x1 | 1 | fully supported |
| `N300` | 1x2 | 2 | fully supported |
| `T3K` (8-device Wormhole LLMBox) | physical `1×8`; dots.mocr runs on logical `1×1` or `1×2` | 1 or 2 | Default: open full 8-device mesh then submesh (`DOTS_T3K_OPEN_FULL_MESH=1`). Set `DOTS_T3K_TP=2` for TP across 2 chips. |
| `TG` / Galaxy | — | — | not supported (needs DP, not implemented) |

Use `models.demos.dots_ocr.tt.mesh.open_mesh_device()` — it reads `MESH_DEVICE` and,
on T3K-class systems, can open the full **8-device** mesh then `create_submesh` to the
supported logical shape (see `DOTS_T3K_OPEN_FULL_MESH`). Close with `close_dots_mesh_device`
when using that path. Logical mesh defaults to `1×1` unless `DOTS_T3K_TP=2` (then `1×2`).

```bash
export MESH_DEVICE=N150        # or N300, or T3K (8-device; logical 1x1 default, DOTS_T3K_TP=2 for 1x2)
export HF_MODEL=rednote-hilab/dots.mocr
# Optional: cap prefill/KV length to fit DRAM
export DOTS_MAX_SEQ_LEN=8192   # legacy name DOTS_MAX_SEQ_LEN_WH_LB still honored

PYTHONPATH=$(pwd) python -m models.demos.dots_ocr.demo.demo --image /path/to/page.png
```

**Full TTNN Vision**: Complete 42-layer TTNN `VisionTransformerTT` (no hybrid
HF `vision_tower`). Includes `PatchEmbedTT`, `VisionBlockTT` (post-norm), and
integration with the existing `PatchMergerTT`. Vision weights are currently
replicated across the mesh, so effective vision TP is 1 even on N300 / T3K.

The TTNN text decoder uses embeddings-based prefill with proper RoPE alignment.
See `tt/model.py`, `tt/vision_transformer.py`, and `tt/generator.py`.

## Notes
- The full `rednote-hilab/dots.mocr` checkpoint is large; CI runs skip device tests unless `MESH_DEVICE` is set.
- Set `HF_MODEL` if not using the default repo id.
- Unit tests: `pytest models/demos/dots_ocr/tests --confcutdir=models/demos/dots_ocr/tests` (avoids importing repo-wide `conftest` without TTNN).

### Eager attention vs `flash_attn` (and how `qwen25_vl` differs)

This demo targets **Tenstorrent device** inference and **does not require CUDA or the real
`flash_attn` wheel**. Reference logits use Hugging Face **eager** attention:

- `reference/hf_utils.py` loads with `_attn_implementation="eager"` and `use_safetensors=True`.

**`models/demos/qwen25_vl`** ships a **vendored** `reference/model.py`. It follows the usual
Transformers pattern: `if is_flash_attn_2_available(): import flash_attn … else: … = None`, and
maps `config._attn_implementation` to classes including `"eager"` and `"sdpa"` (see
`QWEN2_5_VL_ATTENTION_CLASSES` / vision equivalents). Demos often call `from_pretrained` with
`dtype` / `device_map` only; on GPU, the default may be SDPA.

**Dots OCR** uses **remote** Hub code (`trust_remote_code`). Transformers runs **`check_imports`**
on that file *before* `from_pretrained` kwargs take effect, so a top-level `import flash_attn` in
the checkpoint must still import. We register a minimal **`reference/_flash_attn_shim.py`** so that
check passes; it is **not** FlashAttention at runtime — **eager** is. If the real `flash_attn`
package is installed, we do not replace it.
