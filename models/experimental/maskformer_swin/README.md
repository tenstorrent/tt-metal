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
- Optional predictions JSON: `generated/predictions_tt.json` (via `--dump-predictions`)

## Tests
- End‑to‑end fallback demo: `pytest models/experimental/maskformer_swin/tests/test_e2e_demo.py::test_maskformer_fallback_end_to_end -q`
- Optional integration tests (download HF weights): set `MASKFORMER_RUN_WEIGHT_TESTS=1` and run `pytest models/experimental/maskformer_swin/tests -q`
- TT smoke test (skips without hardware): set `MASKFORMER_RUN_TT_TESTS=1` and run `pytest models/experimental/maskformer_swin/tests/test_tt_minimal.py -q`

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

## TT optimizations
- Coverage: decoder + heads on TT; optional 3×3 mask projection in the pixel decoder. Swin backbone and most pixel-decoder blocks remain on CPU.
- Attention/MLP configs: centralised in `tt_configs.py` (Matmul + SDPA program configs, 8×7 core-grid hints, L1 preference for small sequences). Decoder uses fused QKV + SDPA where supported and falls back to manual attention otherwise.
- Head/block parallelism: matmuls/attention respect the shared grid config so heads and batch shards spread across cores; stage activations stay in L1 when size permits.
- Fused dense ops: decoder MLPs and head projections route through `ttnn.linear` with bias/activation fusion when available. Program configs shard work across the Wormhole grid.
- CNN flow: mask projection conv uses TT-CNN builder with height slicing, TILE output, and L1-preferred inputs.
- Layout hygiene: tensors stay in TILE layout through decoder/heads; pixel embeddings are permuted once to NHWC before the projection matmul to reduce layout churn.
- Tunable knobs: `--height/--width` for 320×320 or 640×640, env vars (`MASKFORMER_TT_DECODER`, `MASKFORMER_TT_MASK_PROJ`, `MASKFORMER_TT_FUSE_LINEAR_ACT`, `MASKFORMER_TT_USE_LINEAR`), and program configs in `tt_configs.py`.
- Perf artifacts: example Wormhole N300 measurements for 320×320 and 640×640 live in `perf_wormhole_n300_{320,640}.json` plus headers `perf_header_wormhole_n300_{320,640}.json`. Regenerate with `--dump-perf` on hardware to refresh numbers for your firmware/runtime.
- Programmatic eval: `eval_utils.run_coco_eval` mirrors the CLI for mIoU/PQ; `dump_predictions_json` serializes per-query class confidences for quick inspection.
- Known limitation: TT mask projection will fall back to the HF conv if TT-CNN configs are unsupported on a given firmware/runtime; the fallback is logged and the rest of the TT path continues.

## Details
- Entry points: `runner.py` (CLI demo), `fallback.py` (CPU pipeline)
- Modules: `backbone_swin.py`, `pixel_decoder.py`, `transformer_decoder.py`, `heads.py`, `weights.py`

## Status & limitations
- Swin backbone + most pixel decoder layers are still CPU; TT coverage is decoder + heads + optional mask projection.
- Tuned for Wormhole N300; Blackhole/N150 not profiled.
- Perf headers for 320/640 reflect Wormhole N300 runs; refresh with `--dump-perf` if you change firmware or configs.
- Panoptic PQ requires `panopticapi` and GT paths; otherwise only mIoU is reported.
