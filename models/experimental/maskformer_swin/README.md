# MaskFormer Swin‑B (TT‑NN)

Full TTNN implementation of the Hugging Face checkpoint `facebook/maskformer-swin-base-coco`.

Coverage:
- TT (TTNN): Swin‑B backbone + pixel decoder + transformer decoder + heads

## Platform
- Tested on Wormhole (N300) only.

## Prerequisites
- Follow `INSTALLING.md` at repo root to build TT‑Metal/TT‑NN and Python bindings.
- Python deps (demo + tests + perf + COCO eval):

  ```bash
  pip install torch transformers huggingface_hub safetensors pillow numpy pycocotools loguru GitPython
  ```

- Optional (COCO PQ metric): `panopticapi`

  ```bash
  pip install git+https://github.com/cocodataset/panopticapi.git
  ```

## Hugging Face access (non-interactive)
Weights download uses `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN` if set (no interactive login required).

```bash
export HF_TOKEN=...               # or HUGGINGFACE_HUB_TOKEN=...
export HF_HOME=/path/to/hf_cache  # optional
export TRANSFORMERS_CACHE=/path/to/transformers_cache  # optional
```

## Artifact layout (bounty stage folders)

The demo runner and scripts are set up to write reproducible artifacts under:

```text
generated/maskformer_swin/
  stage1_baseline/
  stage2_opt/
  stage3_opt/
```

## Run: single image (verifiable artifacts + perf)

Stage 1 bring-up / baseline:

```bash
python -m models.experimental.maskformer_swin.demo.runner \
  --image models/sample_data/demo.jpeg \
  --weights facebook/maskformer-swin-base-coco \
  --device wormhole_n300 \
  --height 320 --width 320 \
  --optimization-stage stage1 \
  --tt-repeats 5 \
  --dump-perf generated/maskformer_swin/stage1_baseline/perf.json \
  --dump-perf-header generated/maskformer_swin/stage1_baseline/perf_header.json \
  --output-dir generated/maskformer_swin/stage1_baseline/demo_outputs
```

The `demo_outputs/` folder contains:
- `semantic_overlay.png` (semantic segmentation overlay)
- `panoptic_segmentation.png` + `panoptic_segments.json` (colored segments + labels/scores/areas)
- `instance_masks/*.png` + `instance_masks.json` (top-K binary instance masks, filenames include label + score)

Stage 2 / Stage 3 can be run by changing `--optimization-stage` and output paths:

```bash
python -m models.experimental.maskformer_swin.demo.runner \
  --image models/sample_data/demo.jpeg \
  --weights facebook/maskformer-swin-base-coco \
  --device wormhole_n300 \
  --height 320 --width 320 \
  --optimization-stage stage3 \
  --tt-repeats 5 \
  --dump-perf generated/maskformer_swin/stage3_opt/perf.json \
  --dump-perf-header generated/maskformer_swin/stage3_opt/perf_header.json \
  --output-dir generated/maskformer_swin/stage3_opt/demo_outputs
```

## PCC test (vs Torch reference)

```bash
python -m pytest models/experimental/maskformer_swin/tests/pcc/test_maskformer_swin.py -q
```

## Generate perf sheet (device ops CSV)

This generates the per-op device perf sheet CSV via `tools/tracy/profile_this.py`.

Notes:
- `profile_this.py` wraps `python3 -m tracy -m ...`, so the `-c` command should be a python **module path** (no leading `python -m`).
- For large models, increase profiler op support and enable mid-run dumps to avoid missing device logs.

```bash
PYTHONPATH=$(pwd)/ttnn \
TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=20000 \
TT_METAL_PROFILER_MID_RUN_DUMP=1 \
./tools/tracy/profile_this.py \
  -o generated/maskformer_swin/stage3_opt/perf_sheet \
  -n maskformer_swin_stage3_320 \
  -c "models.experimental.maskformer_swin.demo.runner \
        --image models/sample_data/demo.jpeg \
        --weights facebook/maskformer-swin-base-coco \
        --device wormhole_n300 \
        --height 320 --width 320 \
        --optimization-stage stage3 \
        --tt-repeats 1 \
        --output-dir generated/maskformer_swin/stage3_opt/profile_run"
```

The extracted CSV is written to:
- `generated/maskformer_swin/stage3_opt/perf_sheet/Ops_Perf.csv`

## COCO evaluation (accuracy)

Dataset layout (COCO panoptic):

```text
coco/
  val2017/
  annotations/
    panoptic_val2017.json
    panoptic_val2017/            # PNGs
```

Quick smoke (50 images, deterministic subset):

```bash
python -m models.experimental.maskformer_swin.demo.runner \
  --image models/sample_data/demo.jpeg \
  --weights facebook/maskformer-swin-base-coco \
  --device wormhole_n300 \
  --height 320 --width 320 \
  --coco-eval \
  --coco-dir /path/to/coco \
  --coco-max-images 50 \
  --coco-report generated/maskformer_swin/stage1_baseline/coco_eval/report_50.json
```

Longer run (example: 200 images):

```bash
python -m models.experimental.maskformer_swin.demo.runner \
  --image models/sample_data/demo.jpeg \
  --weights facebook/maskformer-swin-base-coco \
  --device wormhole_n300 \
  --height 320 --width 320 \
  --coco-eval \
  --coco-dir /path/to/coco \
  --coco-max-images 200 \
  --coco-report generated/maskformer_swin/stage1_baseline/coco_eval/report_200.json
```

Outputs:
- Report JSON at `.../coco_eval/report_*.json`
- If `panopticapi` is installed, panoptic predictions are also written under `.../coco_eval/panoptic_predictions/` and PQ is reported; otherwise PQ is `null` and mIoU is still computed.

## Tuning / known issues

- Program cache: disabled by default for N300 stability (`MASKFORMER_TT_DISABLE_PROGRAM_CACHE=1`). Set `MASKFORMER_TT_DISABLE_PROGRAM_CACHE=0` to override.
- SDPA gating in transformer decoder (stage3):
  - `MASKFORMER_TT_SDPA_MIN_SEQ` (default `192`) avoids SDPA on small sequences where it can regress.
  - `MASKFORMER_TT_FORCE_SDPA=1` forces SDPA regardless of sequence lengths (debug/tuning).
- Matmul/core-grid knobs:
  - `MASKFORMER_TT_DISABLE_CORE_GRID=1` disables explicit core grid for linear/matmul.
  - `MASKFORMER_TT_DISABLE_MATMUL_PC=1` disables matmul program config overrides.
