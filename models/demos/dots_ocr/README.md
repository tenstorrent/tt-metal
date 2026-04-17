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

### Wormhole LB (single chip)

**LB** here means a **single Wormhole** card (1×1 mesh), e.g. **N150** or **N300** — not T3K/Galaxy.

```bash
export MESH_DEVICE=N150   # or N300
export HF_MODEL=rednote-hilab/dots.mocr
# Optional: cap prefill/KV length to fit DRAM on WH
export DOTS_MAX_SEQ_LEN_WH_LB=8192

PYTHONPATH=$(pwd) python -m models.demos.dots_ocr.demo.demo --image /path/to/page.png
```

**Full TTNN Vision**: Complete 42-layer TTNN `VisionTransformerTT` (no hybrid HF `vision_tower`). Includes `PatchEmbedTT`, `VisionBlockTT` (post-norm), and integration with existing `PatchMergerTT`.

The TTNN text decoder uses embeddings-based prefill (from Step 2) with proper RoPE alignment. See `tt/model.py`, `tt/vision_transformer.py`, and `tt/generator.py`.

## Notes
- The full `rednote-hilab/dots.mocr` checkpoint is large; CI runs skip device tests unless `MESH_DEVICE` is set.
- Set `HF_MODEL` if not using the default repo id.
- Unit tests: `pytest models/demos/dots_ocr/tests --confcutdir=models/demos/dots_ocr/tests` (avoids importing repo-wide `conftest` without TTNN).
