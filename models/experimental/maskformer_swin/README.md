# MaskFormer Swin‑B (TT‑NN)

Hybrid TTNN run for the Hugging Face checkpoint `facebook/maskformer-swin-base-coco`.

Coverage:
- CPU (Hugging Face): Swin‑B backbone + pixel decoder
- TT (TTNN): transformer decoder + heads

## Platforms
- Wormhole (N300)

## Prerequisites
- Follow `INSTALLING.md` at repo root to build TT‑Metal/TT‑NN and Python bindings.
- Hugging Face access for weights: `huggingface-cli login` or set `HF_TOKEN`.
- Python deps for demo/eval: `torch`, `transformers`, `pillow`, `huggingface_hub`, `safetensors`
- COCO eval deps: `numpy`, `pycocotools` (and `panopticapi` for PQ).

## Run: single image (overlay + perf header)

```bash
python -m models.experimental.maskformer_swin.demo.runner \
  --image models/sample_data/demo.jpeg \
  --weights facebook/maskformer-swin-base-coco \
  --device wormhole_n300 \
  --height 320 --width 320 \
  --save generated/maskformer_overlay.png \
  --dump-perf generated/maskformer_perf.json
```

This writes:
- Overlay PNG: `generated/maskformer_overlay.png`
- Perf JSON: `generated/maskformer_perf.json`
- Perf header JSON: `generated/maskformer_perf_header.json`

## Run: COCO evaluation (optional)

Requires a COCO root containing `val2017/` and panoptic annotations under `annotations/`.

```bash
python -m models.experimental.maskformer_swin.demo.runner \
  --image models/sample_data/demo.jpeg \
  --weights facebook/maskformer-swin-base-coco \
  --device wormhole_n300 \
  --coco-eval \
  --coco-dir /datasets/coco \
  --coco-max-images 50 \
  --coco-report generated/maskformer_coco_eval.json
```
