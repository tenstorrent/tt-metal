# RT-DETR Validation

## Setup

| Item | Value |
|---|---|
| Hardware | Tenstorrent P150a |
| Git base commit | `b1d4b7a2106b4ebe16f6a42ebcc63755146b3df4` |
| Python | 3.10.19 |
| PyTorch | 2.11.0+cpu |
| Transformers | 5.12.1 |
| Input | Batch 1, 640×640, BF16 |
| Device configuration | `l1_small_size=16384`, `trace_region_size=1 << 26` |
| RT-DETR v1 checkpoint | `PekingU/rtdetr_r50vd` |
| RT-DETRv2 checkpoint | `PekingU/rtdetr_v2_r50vd` |
| Validation data | First 500 COCO `val2017` images in sorted filename order |

The recorded runs used the listed base commit with uncommitted RT-DETR validation changes.

### COCO Validation Data

The detection tests require the COCO `val2017` images. Ground-truth annotations are not required because the tests compare TTNN detections against the PyTorch reference.

From the RT-DETR directory:

```bash
mkdir -p datasets/coco
cd datasets/coco
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip
rm val2017.zip
cd ../..
```

The default image path is `./datasets/coco/val2017`. To use another location:

```bash
export COCO_VAL2017_IMAGES=<path-to-val2017>
```

## PCC

```bash
pytest tests/pcc.py::test_rtdetr_model -sv
pytest tests/pcc.py::test_rtdetr_v2_model -sv
```

Decoder queries shared by both implementations were aligned by proposal index before PCC comparison.

| Output | RT-DETR v1 | RT-DETRv2 |
|---|---:|---:|
| Last hidden state | 0.9693 | 0.9699 |
| Intermediate hidden states | 0.9667 | 0.9705 |
| Intermediate logits | 0.9375 | 0.9545 |
| Intermediate reference points | 0.9802 | 0.9512 |

## Detection Correctness

```bash
pytest tests/validate_detection.py::test_rtdetr_detection -sv
pytest tests/validate_detection.py::test_rtdetr_v2_detection -sv
```

PyTorch and TTNN detections used a confidence threshold of `0.5`. Boxes were greedily matched at IoU ≥ `0.5`.

| Metric | RT-DETR v1 | RT-DETRv2 |
|---|---:|---:|
| Images | 500 | 500 |
| PyTorch detections | 2,763 | 3,550 |
| TTNN detections | 2,752 | 3,558 |
| Matched detections | 2,644 | 3,377 |
| Matched ratio | 95.69% | 94.91% |
| Label agreement | 99.74% | 98.46% |
| Mean matched IoU | 0.9577 | 0.9521 |
| Minimum matched IoU | 0.5196 | 0.5383 |
| Mean confidence-score error | 0.0210 | 0.0221 |

## Model Latency

```bash
pytest tests/latency.py::test_rtdetr_batch_1_trace_latency -sv
pytest tests/latency.py::test_rtdetr_v2_batch_1_trace_latency -sv
```

Each test performs one warmup, captures the model, and measures eight trace replays on different images. Timing includes blocking trace execution and excludes preprocessing, input transfer, output transfer, and postprocessing. Program caching is enabled.

| Metric | RT-DETR v1 | RT-DETRv2 |
|---|---:|---:|
| Mean latency | 23.78 ms | 23.77 ms |
| Median latency | 23.78 ms | 23.78 ms |
| Throughput | 42.05 FPS | 42.07 FPS |

## Video Throughput

```bash
python demo.py --video <path-or-direct-url> --model_version 1
```

Video throughput includes decoding, preprocessing, input transfer, trace replay, output transfer, postprocessing, drawing, and H.264 encoding. Model initialization, warmup, and trace capture are excluded.

| Metric | Result |
|---|---:|
| Mean throughput | 27.83 FPS |
| Minimum throughput | 25.34 FPS |
| Maximum throughput | 29.19 FPS |
