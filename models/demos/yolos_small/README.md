# YOLOS-small (TTNN)

YOLOS-small object detection demo and TTNN implementation targeting Tenstorrent Wormhole (N300).

This implementation mirrors Hugging Face `hustvl/yolos-small` as the reference model and provides
three TTNN optimization stages:

- **Stage 1** – High-precision bring-up (fp32 weights/activations, no optimizations)
- **Stage 2** – Basic optimizations (sharding, fusion, L1)
- **Stage 3** – Deep optimizations (fused SDPA when available, bfloat16 activations today with optional bfloat8 in future runtimes, max core utilization)

## Running the demo

From the tt-metal repo root on the N300:

```bash
python3 -m models.demos.yolos_small.demo \
  --stage 1 \
  --device-id 0 \
  --output yolos_stage1_result.jpg

python3 -m models.demos.yolos_small.demo \
  --stage 2 \
  --device-id 0 \
  --output yolos_stage2_result.jpg

python3 -m models.demos.yolos_small.demo \
  --stage 3 \
  --device-id 0 \
  --output yolos_stage3_result.jpg \
  --benchmark
```

By default the demo uses the COCO sample image
`http://images.cocodataset.org/val2017/000000039769.jpg`. Use `--image` to override.

Each run also saves a `_pytorch.jpg` image with the Hugging Face reference predictions.

## Numerical alignment with Hugging Face YOLOS-small

The reference model is:

- `models.demos.yolos_small.reference.modeling_yolos.YolosForObjectDetection`
  wrapping `transformers.YolosForObjectDetection` with pretrained weights
  from `hustvl/yolos-small`.

The TTNN implementation is:

- `models.demos.yolos_small.yolos_ttnn.modeling_yolos.YolosForObjectDetection`

### Stage 1 (fp32 bring-up)

Stage 1 is configured to run with fp32 weights and activations to minimize numerical drift
relative to the Hugging Face reference. A debug script compares the two models end-to-end:

```bash
python3 debug_yolos_ttnn_stage1.py
```

On Wormhole N300 we observe typical single-sample differences:

- **Max logits diff**: ~1–2
- **Max boxes diff**: ~0.04–0.10

Per-layer encoder comparisons show each block’s output is within ~0.1–0.8 of the HF output
when fed the same input, and learned embeddings differ by ~0.1 in magnitude.

### PCC-based tests (YOLOv4 style)

The test suite uses Pearson correlation (PCC), following the YOLOv4 integration:

- File: `models/demos.yolos_small.tests.test_yolos_small`
- Helper: `tests.ttnn.utils_for_testing.assert_with_pcc`

For each stage, we compare TTNN vs HF logits and boxes in float32:

- **Stage 1**: PCC ≥ 0.999 (logits, boxes)
- **Stage 2**: PCC ≥ 0.997 (logits, boxes)
- **Stage 3**: PCC ≥ 0.995 (logits, boxes)

This matches Tenstorrent’s convention for model bring-up (high correlation with the
reference rather than tiny elementwise differences), similar to:

- `models.demos.yolov4.tests.pcc.test_ttnn_yolov4`

## COCO validation

For COCO evaluation, run on the N300:

```bash
python3 -m models.demos.yolos_small.validate_coco \
  --coco-path /workdir/data/coco \
  --stage 3 \
  --max-images 50 \
  --device-id 0 \
  --threshold 0.05
```

This script:

- Loads Hugging Face YOLOS-small as the reference.
- Runs both HF (PyTorch) and TTNN Stage 3 on the same COCO validation images.
- Reports COCO metrics (AP, AP50, AP75) for both and prints the deltas.

On a 50-image subset (N300, Wormhole):

- **HF YOLOS-small**:
  - AP ≈ 0.000
  - AP50 ≈ 0.001
  - AP75 ≈ 0.000
- **TTNN Stage 3 YOLOS-small**:
  - AP ≈ 0.000
  - AP50 ≈ 0.000
  - AP75 ≈ 0.000
- **Differences**:
  - ΔAP ≈ -0.000
  - ΔAP50 ≈ -0.000

On this small, low-signal subset, both models see essentially the same COCO metrics,
with no measurable deviation between HF and TTNN.

For full COCO evaluation, increase `--max-images` and/or adjust `--threshold` as needed.

## Performance

The demo script supports a `--benchmark` flag which times multiple runs and reports:

- Average latency (ms)
- Latency standard deviation (ms)
- Throughput (FPS)

Example (Stage 3, N300, 1x3x512x864 input) will print a summary similar to:

```text
Benchmark results:
  avg_time_ms: <latency_ms>
  std_time_ms: <std_ms>
  throughput_fps: <fps>
```

For perf reporting in a PR, follow the YOLOv4 README style:

- Report FPS for Stage 3 on N150/N300 where available.
- Summarize hardware, batch size, and input resolution.
