# RT-DETR Validation

## Setup

| Item | Value |
|---|---|
| Hardware | Tenstorrent Blackhole P100a, full 11×10 grid |
| Git base commit | `02f968e1daa4dab70aadcc3f13fd97c037687000` |
| Python | 3.10.21 |
| PyTorch | 2.11.0+cpu |
| Transformers | 5.12.1 |
| Input | Batch 1, 640×640, BF16 |
| Device configuration | `l1_small_size=16384`, `trace_region_size=1 << 26` |
| RT-DETR v1 checkpoint | `PekingU/rtdetr_r50vd` |
| RT-DETRv2 checkpoint | `PekingU/rtdetr_v2_r50vd` |
| Validation data | First 500 COCO `val2017` images, in sorted filename order, whose license allows commercial use and modification |

The recorded runs used the RT-DETR commits of this change on top of the listed base commit, in a Release build.

### COCO Validation Data

The detection tests compare TTNN detections against the PyTorch reference, so they need no ground
truth, but they read the `val2017` annotations for each image's license. They use the first 500
images, in sorted filename order, whose license allows commercial use and modification (CC BY 2.0,
CC BY-SA 2.0, no known copyright restrictions or United States Government Work), and download any
of those images that are missing from the image directory.

From the RT-DETR directory:

```bash
mkdir -p datasets/coco
cd datasets/coco
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip annotations/instances_val2017.json
rm annotations_trainval2017.zip
cd ../..
```

The default paths are `./datasets/coco/annotations/instances_val2017.json` for the annotations and
`./datasets/coco/val2017` for the images. To use other locations:

```bash
export COCO_VAL2017_ANNOTATIONS=<path-to-instances_val2017.json>
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
| Shared top-k proposals | 298 / 300 | 297 / 300 |
| Last hidden state | 0.9963 | 0.9888 |
| Intermediate hidden states | 0.9962 | 0.9918 |
| Intermediate logits | 0.9942 | 0.9872 |
| Intermediate reference points | 0.9972 | 0.9848 |

All 25 tests in `tests/pcc.py` pass, on the full grid and on a 2-core grid (`TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE=1,0`). Thresholds are 0.999 for single ops, 0.99 for blocks and 0.95 for the full model.

## Detection Correctness

```bash
pytest tests/validate_detection.py::test_rtdetr_detection -sv
pytest tests/validate_detection.py::test_rtdetr_v2_detection -sv
```

PyTorch and TTNN detections used a confidence threshold of `0.5`. Boxes were greedily matched at IoU ≥ `0.5`. These results were recorded on base commit `ef2854b798f06e76028cc2c63987beb5b1354e0d`.

| Metric | RT-DETR v1 | RT-DETRv2 |
|---|---:|---:|
| Images | 500 | 500 |
| PyTorch detections | 2,752 | 3,578 |
| TTNN detections | 2,674 | 3,445 |
| Matched detections | 2,650 | 3,393 |
| Matched ratio | 96.29% | 94.83% |
| Label agreement | 99.66% | 99.17% |
| Mean matched IoU | 0.9622 | 0.9581 |
| Minimum matched IoU | 0.5650 | 0.5298 |
| Mean confidence-score error | 0.0114 | 0.0128 |

## Model Latency

```bash
pytest tests/latency.py::test_rtdetr_batch_1_trace_latency -sv
pytest tests/latency.py::test_rtdetr_v2_batch_1_trace_latency -sv
```

Each test performs one warmup, captures the model, and measures eight trace replays on different images. Timing includes blocking trace execution and excludes preprocessing, input transfer, output transfer, and postprocessing. Program caching is enabled.

| Metric | RT-DETR v1 | RT-DETRv2 |
|---|---:|---:|
| Mean latency | 24.53 ms | 24.53 ms |
| Median latency | 24.53 ms | 24.52 ms |
| Throughput | 40.77 FPS | 40.77 FPS |

## End-to-End Latency

```bash
pytest tests/latency.py::test_rtdetr_batch_1_e2e_latency -sv
pytest tests/latency.py::test_rtdetr_v2_batch_1_e2e_latency -sv
```

| Metric | RT-DETR v1 | RT-DETRv2 |
|---|---:|---:|
| Median latency | 33.70 ms | 33.07 ms |
| Throughput (median) | 29.67 FPS | 30.24 FPS |
