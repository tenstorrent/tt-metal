# MaskFormer Swin-B (TT-NN)

Full TTNN implementation of the Hugging Face checkpoint `facebook/maskformer-swin-base-coco`.

Coverage:
- TT (TTNN): Swin-B backbone + pixel decoder + transformer decoder + heads

## Platform
- Tested on Wormhole (N300) single chip (unit mesh / `device_id=0`; not data-parallel).

## Prerequisites
- Follow `INSTALLING.md` at repo root to build TT-Metal/TT-NN and Python bindings.
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

Note: `--device` is a reporting label; the runner currently opens `device_id=0` (single chip). Perf dumps include the detected `cluster_type` and actual `mesh_device_ids`/`mesh_num_devices`.

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

Optional TT-path debug/verification output (confirms backbone, pixel decoder, and transformer decoder tensors stay on TT):

```bash
MASKFORMER_TT_DEBUG_RUNNER=1 \
python -m models.experimental.maskformer_swin.demo.runner \
  --image models/sample_data/demo.jpeg \
  --weights facebook/maskformer-swin-base-coco \
  --device wormhole_n300 \
  --height 320 --width 320 \
  --optimization-stage stage1 \
  --tt-repeats 1 \
  --output-dir generated/maskformer_swin/stage1_baseline/tt_path_validation
```

## PCC test (vs Torch reference)

```bash
PYTHONPATH=$(pwd) pytest models/experimental/maskformer_swin/tests/pcc/test_maskformer_swin.py -q
```

## Ideal end-to-end perf (trace + 2CQ)

For "ideal" end-to-end perf (trace replay + 2 command queues), run:

```bash
PYTHONPATH=$(pwd) pytest models/experimental/maskformer_swin/tests/perf/test_perf_e2e_maskformer_swin.py -q
```

## Generate perf sheet (device ops CSV)

This generates the per-op device perf sheet CSV via `tools/tracy/profile_this.py`.

Notes:
- `profile_this.py` wraps `python3 -m tracy -m ...`, so the `-c` command should be a python **module path** (no leading `python -m`).
- For large models, increase profiler op support and enable mid-run dumps to avoid missing device logs.

```bash
PYTHONPATH=$(pwd)/ttnn \
TT_METAL_LOGS_PATH=/tmp/tt_logs \
TT_METAL_INSPECTOR=0 \
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
- Repro command is saved at:
  - `generated/maskformer_swin/stage3_opt/perf_sheet/command.txt`

## Measured performance (N300, single chip)

Measured on **N300** with a single 320x320 image (`models/sample_data/demo.jpeg`). See the raw runner dumps under:
- `generated/maskformer_swin/stage1_baseline/perf.json`
- `generated/maskformer_swin/stage2_opt/perf.json`
- `generated/maskformer_swin/stage3_opt/perf.json`

Summary (mean latency over 5 runs; includes full TT forward + explicit device synchronize in the timed loop):
- Stage 1: `1069.62 ms`
- Stage 2: `934.30 ms` (12.65% faster vs stage1)
- Stage 3: `904.63 ms` (15.43% faster vs stage1, 3.18% faster vs stage2)

640x640 stage1 sanity (single run):
- `generated/maskformer_swin/stage1_640_sanity/perf_640.json`
- latency: `2299.33 ms`
- outputs: `generated/maskformer_swin/stage1_640_sanity/demo_outputs/`

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

Example measured outputs from these runs:
- 50 images (`generated/maskformer_swin/stage1_baseline/coco_eval/report_50.json`)
  - mIoU: `0.3870658237494878`
  - PQ: `0.3301817310014115`
  - avg latency: `1332.2610265185358 ms`
- 200 images (`generated/maskformer_swin/stage1_baseline/coco_eval/report_200.json`)
  - mIoU: `0.4691255001343233`
  - PQ: `0.3826408539122837`
  - avg latency: `1272.736109829275 ms`

Outputs:
- Report JSON at `.../coco_eval/report_*.json`
- If `panopticapi` is installed, panoptic predictions are also written under `.../coco_eval/panoptic_predictions/` and PQ is reported; otherwise PQ is `null` and mIoU is still computed.

## Tuning / known issues

- Program cache: disabled by default for N300 stability (`MASKFORMER_TT_DISABLE_PROGRAM_CACHE=1`). Set `MASKFORMER_TT_DISABLE_PROGRAM_CACHE=0` to override.
- Stage 3 backbone attention optimizations (enabled by `--optimization-stage stage3`):
  - `MASKFORMER_TT_BACKBONE_TILE_MASK_ADD=1`: add shifted-window attention mask in TILE layout (B=1 path) to avoid ROW_MAJOR<->TILE conversions.
  - `MASKFORMER_TT_BACKBONE_INPLACE_ADDS=1`: use inplace elementwise ops (`add_`, `multiply_`) where supported to reduce allocations.
  - `MASKFORMER_TT_BACKBONE_REUSE_ATTN_MASK_CACHE=1`: reuse cached attention masks without cloning (reduces per-forward allocations).
- Fused mask+softmax (`MASKFORMER_TT_BACKBONE_FUSED_MASK_SOFTMAX=1`): available on some builds, but currently **not PCC-safe** for this model (kept disabled in tests and stage configs).
- Backbone L1 activations (`MASKFORMER_TT_BACKBONE_L1_ACT=1`): known to be unstable on N300 for this model; keep disabled.
- SDPA (experimental, transformer decoder):
  - `MASKFORMER_TT_ENABLE_SDPA=1` enables SDPA paths (currently disabled in the stage configs because it regressed on N300 in local tuning).
  - `MASKFORMER_TT_SDPA_MIN_SEQ` (default `192`) avoids SDPA on small sequences where it can regress.
  - `MASKFORMER_TT_FORCE_SDPA=1` forces SDPA regardless of sequence lengths (debug/tuning).
- Matmul/core-grid knobs:
  - `MASKFORMER_TT_DISABLE_CORE_GRID=1` disables explicit core grid for linear/matmul.
  - `MASKFORMER_TT_DISABLE_MATMUL_PC=1` disables matmul program config overrides.
