# YOLOv8l

## Platforms
Wormhole (n150, n300, multi-device / T3K).

## Introduction
YOLOv8l is the large Ultralytics YOLOv8 detection variant run on TT-NN at **640×640** per inference (letterboxed).

Resource: [Ultralytics YOLO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/model.py).

## Prerequisites
- [tt-metal](https://github.com/tenstorrent/tt-metal) checkout and [install](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md).
- For host reference / weights: `ultralytics` (PCC and demo load `yolov8l.pt` unless CI weights are used).

## How to verify

### 1) Correctness (PCC vs Torch)
Runs TT against Ultralytics reference on random **640×640** input (needs device):

```bash
pytest --disable-warnings models/demos/yolov8l/tests/pcc/test_yolov8l.py::test_yolov8l_640
```

### 2) Trace + 2 CQ performance smoke
Single device:

```bash
pytest --disable-warnings models/demos/yolov8l/tests/perf/test_e2e_performant.py::test_run_yolov8l_trace_2cqs_inference
```

Multi-device / mesh (data-parallel):

```bash
pytest --disable-warnings models/demos/yolov8l/tests/perf/test_e2e_performant.py::test_run_yolov8l_trace_2cqs_dp_inference
```

### 3) End-to-end demo (images on disk)
Outputs under `models/demos/yolov8l/demo/runs/<model_type>/`.

Single device:

```bash
pytest models/demos/yolov8l/demo/demo.py::test_demo[res0-True-tt_model-1-models/demos/yolov8l/demo/images-device_params0]
```

Multi-device:

```bash
pytest models/demos/yolov8l/demo/demo.py::test_demo_dp[wormhole_b0-res0-True-tt_model-1-models/demos/yolov8l/demo/images-device_params0]
```

Place test images in `models/demos/yolov8l/demo/images/`.

### 4) Large images (e.g. 1280×1280 on T3K)
TT still runs **640×640** tiles; use SAHI slicing (install `sahi`, plus a `ttnn`-compatible `numpy`/OpenCV stack):

```bash
python models/demos/yolov8l/sahi_ultralytics_eval.py --backend tt --tt-eth-dispatch \
  --tt-slice-parallel-devices 4 --pre-resize-to 1280 1280 \
  --slice-height 640 --slice-width 640 --overlap-height-ratio 0 --overlap-width-ratio 0 \
  --postprocess-type GREEDYNMM --postprocess-match-metric IOS --postprocess-match-threshold 0.1 \
  --confidence-threshold 0.55 --input /path/to/image.jpg
```

`--tt-model` defaults to **yolov8l**; omit it for this demo. Other values (`yolov8s`, `yolov8x`) select those TT runners for comparison only.

Throughput on **T3K (1×8 mesh):** eight **1280×1280** images at a time—each chip runs one image; SAHI runs **four** sequential batched forwards (tile 0…3), each forward batch **8**. Requires **8** images (directory or repeated inputs), `--tt-mesh-shape 1 8`, and `--tt-slice-dp-batch 8` (omit `--tt-slice-parallel-devices`). `summary.json` marks amortized per-image timings when this mode is on.

### 5) Offline tests (no device)
```bash
pytest models/demos/yolov8l/tests/test_sahi_parallel_chunks.py
```

## Details
- **Entry point:** `models/demos/yolov8l/tt/ttnn_yolov8l.py`
- **Batch size:** 1 per device (effective batch scales with mesh in DP).
- **Resolution:** **(640, 640)** per TT forward; larger scenes via `sahi_ultralytics_eval.py` above.
- **Post-processing:** PyTorch (`models/demos/utils/common_demo_utils.py`).

## Inputs / outputs
- **Inputs:** `models/demos/yolov8l/demo/images/` for the pytest demo.
- **Outputs:** `models/demos/yolov8l/demo/runs/` (Torch vs TT subfolders per `model_type`).
