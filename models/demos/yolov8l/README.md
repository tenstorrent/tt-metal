# YOLOv8l

## Platforms
Wormhole (n150, n300, multi-device / T3K), Blackhole (p150).

## Introduction
YOLOv8l is the large Ultralytics YOLOv8 detection variant run on TT-NN with trace + 2 CQ. Demo/perf flows support **640x640** and **1280x1280**.

Resource: [Ultralytics YOLO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/model.py).

## Prerequisites
- [tt-metal](https://github.com/tenstorrent/tt-metal) checkout and [install](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md).
- For host reference / weights: `ultralytics` (PCC and demo load `yolov8l.pt` unless CI weights are used).

## How to verify

### 1) Correctness (PCC vs Torch)
Runs TT against Ultralytics reference on random **640x640** input (needs device):

```bash
pytest --disable-warnings models/demos/yolov8l/tests/pcc/test_yolov8l.py::test_yolov8l_640
```

Also available for **1280x1280**:

```bash
pytest --disable-warnings models/demos/yolov8l/tests/pcc/test_yolov8l.py::test_yolov8l_1280
```

### 2) Trace + 2 CQ performance smoke (throughput / FPS)

Measured end-to-end throughput (single device, batch=1, trace + 2 CQ):

| Device | Resolution | FPS |
|---|---|---|
| Blackhole P150 | 640×640 | **150** |
| Blackhole P150 | 1280×1280 | **30** |

Single device — the test runs **both** 640 and 1280; select one with `-k`:

```bash
# both resolutions
pytest --disable-warnings models/demos/yolov8l/tests/perf/test_e2e_performant.py::test_run_yolov8l_trace_2cqs_inference
# only 640x640
pytest --disable-warnings "models/demos/yolov8l/tests/perf/test_e2e_performant.py::test_run_yolov8l_trace_2cqs_inference" -k 640
# only 1280x1280
pytest --disable-warnings "models/demos/yolov8l/tests/perf/test_e2e_performant.py::test_run_yolov8l_trace_2cqs_inference" -k 1280
```

Multi-device / mesh (data-parallel, 1x8; effective batch 8):

```bash
pytest --disable-warnings models/demos/yolov8l/tests/perf/test_e2e_performant.py::test_run_yolov8l_trace_2cqs_dp_inference
```

Device (kernel) perf — reports `AVG DEVICE KERNEL SAMPLES/S` via the Tracy profiler (needs a profiler-enabled build, the default for `build_metal.sh`):

```bash
pytest --disable-warnings models/demos/yolov8l/tests/perf/test_perf_yolov8l.py::test_perf_device_yolov8l
```
> Only `b1_1280` is enabled by default (uncomment `b1_640` in the test for 640x640). On Blackhole the expected-perf threshold is auto-relaxed via the `is_wormhole_b0()` guard, so it reports without a hard fail.

### 3) End-to-end demo (images on disk)
Outputs under `models/demos/yolov8l/demo/runs/<model_type>/`.

`test_demo` and `test_demo_dp` are parameterized for both **640x640** and **1280x1280** (`res=640` / `res=1280` ids).

Single device:

```bash
pytest models/demos/yolov8l/demo/demo.py::test_demo[res0-True-tt_model-1-models/demos/yolov8l/demo/images-device_params0]
```

Multi-device:

```bash
pytest models/demos/yolov8l/demo/demo.py::test_demo_dp[wormhole_b0-res0-True-tt_model-1-models/demos/yolov8l/demo/images-device_params0]
```

Place test images in `models/demos/yolov8l/demo/images/`.

### 4) COCO subset eval (mAP-style metric vs FiftyOne val, 1280×1280)
Same harness as `yolov8s` / `yolov8x` in `models/demos/yolo_eval/evaluate.py` (subset size and metric definition match that script). Resolution and device buffers match the 1280² demo (`yolov8l_l1_small_size_for_res` / `yolov8l_trace_region_size_e2e_for_res`).

Bracket order is `res`-`device_params`-`model_type` (same as `test_yolov8s[...]`); `res` is id `1280`:

```bash
pytest models/demos/yolo_eval/evaluate.py::test_yolov8l[1280-device_params0-tt_model]
pytest models/demos/yolo_eval/evaluate.py::test_yolov8l[1280-device_params0-torch_model]
```

## Details
- **Entry point:** `models/demos/yolov8l/tt/ttnn_yolov8l.py`
- **Batch size:** 1 per device (effective batch scales with mesh in DP).
- **Resolution:** **(640, 640)** and **(1280, 1280)** in PCC/demo/perf.
- **Post-processing:** PyTorch (`models/demos/utils/common_demo_utils.py`).

## Inputs / outputs
- **Inputs:** `models/demos/yolov8l/demo/images/` for the pytest demo.
- **Outputs:** `models/demos/yolov8l/demo/runs/` (Torch vs TT subfolders per `model_type`).
