# MaskFormer Swin‑B (TT‑NN)

Status: hybrid TT run with overlay and perf header. COCO evaluation hook present.

## Platforms
Wormhole (N300)

## Introduction
MaskFormer is a transformer‑based segmentation model that predicts a fixed set of queries with class labels and binary masks. This implementation targets the Hugging Face checkpoint `facebook/maskformer-swin-base-coco` and provides a hybrid TT run (transformer decoder + heads on TT; optional TT mask projection) plus a CPU fallback path for validation.

## Prerequisites
- Build and environment: follow INSTALLING.md in the repository root to set up TT‑Metal/TT‑NN and their Python bindings.
- Hugging Face access for weights: `huggingface-cli login` or set `HF_TOKEN` in the shell.
- Optional Python packages (CPU helpers only): `transformers`, `pillow`, `huggingface_hub`, `safetensors`, `pycocotools` (and `panopticapi` for PQ). These support preprocessing/post‑processing and the HF fallback. TT execution uses TT‑NN.

## How to Run

### CPU fallback (end‑to‑end)
Runs the HF fallback pipeline to validate shapes and post‑processing:

```bash
pytest models/experimental/maskformer_swin/tests/test_e2e_demo.py::test_maskformer_fallback_end_to_end -q
```

### Hybrid TT demo (overlay + perf header)
Executes transformer decoder + heads on TT (optional TT mask projection). Saves overlay and perf artifacts under `generated/`:

```bash
python -m models.experimental.maskformer_swin.runner \
  --image models/sample_data/demo.jpeg \
  --weights facebook/maskformer-swin-base-coco \
  --device wormhole_n300 \
  --tt-run \
  --save generated/tt_overlay.png \
  --dump-perf generated/tt_perf.json
```

Notes: The pixel decoder’s 3×3 mask projection uses a conservative slice strategy. You can disable it by unsetting `MASKFORMER_TT_MASK_PROJ`.

### Optional: different preprocessing sizes
Use `--height/--width` to set the image processor’s resize (e.g., 320×320 or 640×640) for quick checks.

### COCO evaluation (optional)

```bash
# mIoU only (no panopticapi or GT):
python -m models.experimental.maskformer_swin.runner \
  --image models/sample_data/demo.jpeg \
  --weights facebook/maskformer-swin-base-coco \
  --device wormhole_n300 \
  --tt-run \
  --coco-eval \
  --coco-dir /datasets/coco \
  --coco-max-images 50 \
  --dump-perf generated/tt_perf.json

# mIoU + PQ (panopticapi installed)
python -m models.experimental.maskformer_swin.runner \
  --image models/sample_data/demo.jpeg \
  --weights facebook/maskformer-swin-base-coco \
  --device wormhole_n300 \
  --tt-run \
  --coco-eval \
  --coco-dir /datasets/coco \
  --coco-panoptic-json /datasets/coco/annotations/panoptic_val2017.json \
  --coco-panoptic-root /datasets/coco/annotations/panoptic_val2017 \
  --coco-max-images 50 \
  --coco-report generated/coco_eval_tt.json \
  --dump-perf generated/tt_perf.json
```

## Outputs
- Overlay PNG: `generated/tt_overlay.png`
- Perf JSON: `generated/tt_perf.json`
- Perf header JSON: `generated/tt_perf_header.json`
- Optional COCO report: `generated/coco_eval_tt.json`

## Tests
- End‑to‑end fallback demo: `pytest models/experimental/maskformer_swin/tests/test_e2e_demo.py::test_maskformer_fallback_end_to_end -q`
- Optional integration tests (download HF weights): set `MASKFORMER_RUN_WEIGHT_TESTS=1` and run `pytest models/experimental/maskformer_swin/tests -q`

## Details
- Entry points: `runner.py` (CLI demo), `fallback.py` (CPU pipeline)
- Modules: `backbone_swin.py`, `pixel_decoder.py`, `transformer_decoder.py`, `heads.py`, `weights.py`
