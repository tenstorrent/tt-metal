# LFM2.5-VL-1.6B

## Platforms
Wormhole (n150, n300)

## Introduction
LFM2.5-VL-1.6B ([LiquidAI/LFM2.5-VL-1.6B](https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B)) is a compact vision-language model for OCR / document comprehension and general image+text understanding. It uses:

- LFM2 hybrid text backbone (ShortConv + full-attention layers)
- SigLIP2-NaFlex vision encoder
- Pixel-unshuffle multimodal projector

## Prerequisites
- Built tt-metal (`./build_metal.sh --build-type Release`)
- Python venv activated
- Extra deps:
```bash
pip install -r models/demos/multimodal/lfm25_vl/requirements.txt
```

## How to Run (N300)

```bash
export TT_METAL_HOME=/workdir/tt-metal
export PYTHONPATH=/workdir/tt-metal
export ARCH_NAME=wormhole_b0
source /opt/venv/bin/activate

cd /workdir/tt-metal
pip install -r models/demos/multimodal/lfm25_vl/requirements.txt

# Smoke with dummy weights (no HF download)
HF_MODEL=LiquidAI/LFM2.5-VL-1.6B MESH_DEVICE=N300 \
  pytest models/demos/multimodal/lfm25_vl/demo/vision_demo.py -k batch1-notrace --dummy_weights true -s

# Full run with real weights
HF_MODEL=LiquidAI/LFM2.5-VL-1.6B MESH_DEVICE=N300 \
  pytest models/demos/multimodal/lfm25_vl/demo/vision_demo.py -k batch1-notrace -s
```

Unit tests (after weights are available):
```bash
HF_MODEL=LiquidAI/LFM2.5-VL-1.6B MESH_DEVICE=N300 \
  pytest models/demos/multimodal/lfm25_vl/tests/test_load_checkpoints.py -s

HF_MODEL=LiquidAI/LFM2.5-VL-1.6B MESH_DEVICE=N300 \
  pytest models/demos/multimodal/lfm25_vl/tests/test_short_conv.py models/demos/multimodal/lfm25_vl/tests/test_mlp.py models/demos/multimodal/lfm25_vl/tests/test_projector.py -s
```

## Details
- Entry point: `tt/e2e_model.py` (`TtLfm25VlModel` / `Lfm25VlMultimodalGenerator`)
- Batch size: 1
- Early fusion at `image_token_id=396`
- Config/params: `models/tt_transformers/model_params/LFM2.5-VL-1.6B/config.json`

## Known limitations
- ShortConv decode state runs on host (not in device trace). Use `batch1-notrace` for bring-up.
- Optional `use_host_vision=True` debug path runs HF vision on host (requires `cache_hf=True`).

## Performance
See [PERF.md](PERF.md) (numbers TBD until hardware profiling).
