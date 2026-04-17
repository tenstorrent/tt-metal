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

### Supported Wormhole topologies

`dots.mocr` is GQA with `num_attention_heads=12` and `num_key_value_heads=2`.
Tensor parallelism shards heads along `cluster_shape[1]` and the base
`ModelArgs` asserts `n_kv_heads % cluster_shape[1] == 0`, so only TP degrees
that divide `gcd(12, 2) = 2` are supported: **1 or 2**.

| `MESH_DEVICE` | Mesh shape | TP | Status |
|---|---|---|---|
| `N150` | 1x1 | 1 | fully supported |
| `N300` | 1x2 | 2 | fully supported |
| `T3K` (on T3K hardware) | 1x2 submesh | 2 | supported via auto-clamp; 6 of 8 chips idle |
| `TG` / Galaxy | — | — | not supported (needs DP, not implemented) |

Use `models.demos.dots_ocr.tt.mesh.open_mesh_device()` in new code — it reads
`MESH_DEVICE` and clamps unsupported shapes to `1x2` with a warning.

```bash
export MESH_DEVICE=N150        # or N300, or T3K (clamped to 1x2)
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
