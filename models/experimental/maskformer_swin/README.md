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

## Parity helpers (optional)
- Patch embedding parity:

```bash
python -m models.experimental.maskformer_swin.runner \
  --image models/sample_data/demo.jpeg \
  --weights facebook/maskformer-swin-base-coco \
  --device wormhole_n300 \
  --patch-embed-parity
```

- Stage / decoder parity: replace `--patch-embed-parity` with `--stage1-parity`, `--stage2-parity`, or `--decoder-parity` to compare additional taps.

## Details
- Entry points: `runner.py` (CLI demo), `fallback.py` (CPU pipeline)
- Modules: `backbone_swin.py`, `pixel_decoder.py`, `transformer_decoder.py`, `heads.py`, `weights.py`

## Status & limitations
- TT coverage for bounty #30876 focuses on the transformer decoder and heads plus an optional TT mask‑projection conv. The Swin backbone and the rest of the pixel decoder currently run via the Hugging Face fallback path.
- The TT implementation is tuned and validated on Wormhole N300 (`--device wormhole_n300`) using `ttnn.WormholeComputeKernelConfig`. Blackhole/N150 support is not wired yet and would require dedicated profiling and kernel configuration.
- Decoder and heads keep weights and LayerNorm parameters in L1, while large activations (queries, memory, pixel features) stay in DRAM to avoid L1 OOM. This trades some on‑chip reuse for predictable correctness in Stage 1/2 bring‑up.
- Multi‑head attention in the decoder uses an explicit QK^T → softmax → AV path with per‑head reshaping instead of fused SDPA. Q scaling is folded into the Q projection, and additional fused/parallel variants can be added as Stage‑3 follow‑ups.
- TT behaviour is controlled via environment flags and the CLI: `--tt-run` enables `MASKFORMER_TT_DECODER=1` and `MASKFORMER_TT_MASK_PROJ=1`; advanced toggles like `MASKFORMER_TT_FUSE_LINEAR_ACT`, `MASKFORMER_TT_USE_LINEAR`, and `MASKFORMER_COMPARE_QUANTIZE` exist for experimentation but are not required for the bounty path.
