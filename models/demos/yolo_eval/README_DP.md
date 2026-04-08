# Data-parallel YOLO mesh infer (`yolo_dp_mesh_infer.py`)

Small utility (no SAHI) to run **one** input image through YOLO with **batch = number of mesh devices**: the image is letterboxed to **640×640** once on the host, replicated across the batch dimension, and **sharded** so each wormhole chip gets the same tensor slice in one `runner.run()`. Use this to validate Galaxy / multi-chip **data parallel** bring-up, compare wall times, and optionally export per-slot annotated JPEGs.

Run from the **tt-metal repo root** (same as other demos).

## Example command (TT, 8×4 = 32 devices, YOLOv8s)

```sh
python models/demos/yolo_eval/yolo_dp_mesh_infer.py \
  --backend tt \
  --input models/demos/yolo_eval/sample_images/crowded_freeway.jpg \
  --batch-size 32 \
  --tt-mesh-shape 8 4 \
  --tt-model yolov8s
```

### Full system mesh (Galaxy / driver topology, ResNet DP-style)

Same sharding as above, but **`MeshShape` comes from `SystemMeshDescriptor()`** (like `sahi_ultralytics_eval.py` and multi-device ResNet bring-up). **`--batch-size` is adjusted** to the reported device count if it does not match (with a one-line note). **`--tt-mesh-shape` is ignored.**

```sh
python models/demos/yolo_eval/yolo_dp_mesh_infer.py \
  --backend tt \
  --input models/demos/yolo_eval/sample_images/crowded_freeway.jpg \
  --tt-use-system-mesh \
  --tt-model yolov8s
```

Steady-state **device** timing (warmup + multiple timed `runner.run()` calls; summary prints min/mean/max when `N>1`):

```sh
python models/demos/yolo_eval/yolo_dp_mesh_infer.py \
  --backend tt \
  --input models/demos/yolo_eval/sample_images/crowded_freeway.jpg \
  --tt-use-system-mesh \
  --tt-warmup-iters 2 \
  --tt-measured-iters 5 \
  --tt-model yolov8s
```

By default this **does not** write image files (inference + timing + a **TT data-parallel verify** block only). To save annotated outputs under `models/demos/yolo_eval/sample_images_output/dp32_glx/`:

```sh
python models/demos/yolo_eval/yolo_dp_mesh_infer.py \
  --backend tt \
  --input models/demos/yolo_eval/sample_images/crowded_freeway.jpg \
  --batch-size 32 \
  --tt-mesh-shape 8 4 \
  --tt-model yolov8s \
  --save-images
```

On large multi-chip systems you may need Ethernet dispatch (same idea as other tt-metal demos):

```sh
python models/demos/yolo_eval/yolo_dp_mesh_infer.py \
  --tt-eth-dispatch \
  --backend tt \
  --input models/demos/yolo_eval/sample_images/crowded_freeway.jpg \
  --batch-size 32 \
  --tt-mesh-shape 8 4 \
  --tt-model yolov8s
```

## Parameters

| Flag | Default | Meaning |
|------|---------|---------|
| **`--input`** | *(required)* | Path to one image file (BGR read via OpenCV). |
| **`--backend`** | `tt` | `tt` = Tenstorrent mesh + YOLOv8s/x performant runner; `cpu` = Ultralytics on the host. |
| **`--batch-size`** | `32` | Logical batch size. For **`--backend tt`** it **must** equal the number of opened mesh devices (**`ROWS × COLS`** from `--tt-mesh-shape`). Each batch row is one chip’s input (identical image in the typical smoke test). |
| **`--tt-mesh-shape ROWS COLS`** | `8 4` | `MeshShape(ROWS, COLS)` passed to `open_mesh_device`. **`ROWS * COLS`** must match **`--batch-size`**. Ignored if **`--tt-use-system-mesh`**. |
| **`--tt-use-system-mesh`** | off | Open the mesh from **`SystemMeshDescriptor()`** (full system topology). Aligns **`--batch-size`** to the device count. Use on Galaxy to match ResNet DP=32-style mesh open without hard-coding `8 4`. |
| **`--tt-warmup-iters N`** | `0` | Untimed **`runner.run()`** calls after setup to flush compile/trace effects before measured iters. |
| **`--tt-measured-iters N`** | `1` | Timed forwards; when **`N>1`**, prints **device(run)** min / mean / max across those iters. |
| **`--tt-model`** | `yolov8s` | TT demo variant: `yolov8s` or `yolov8x` (both use 640×640 internal resolution). |
| **`--save-images`** | off | If set, writes `{input_stem}_slot_{00..N-1}.jpg` under **`--output-dir`**. If omitted, no JPEGs are created (timing and verify prints still run). |
| **`--output-dir`** | `models/demos/yolo_eval/sample_images_output/dp{N}_glx` | Output folder when **`--save-images`** is set (`N` = `--batch-size`). Created only when saving. |
| **`--tt-device-id`** | `0` | Device id when using **`--tt-force-single-device`** (single chip). |
| **`--tt-l1-small-size`** | `24576` | TT device `l1_small_size` (YOLOv8 demos default). |
| **`--tt-trace-region-size`** | `6434816` | TT device `trace_region_size` (YOLOv8s demo default). |
| **`--tt-force-single-device`** | off | Open one chip with `open_device` instead of a mesh (only consistent with **`--batch-size 1`** for this script’s TT path). |
| **`--tt-eth-dispatch`** | off | Sets `TT_METAL_GTEST_ETH_DISPATCH=1` before ttnn init (often needed on multi-chip Ethernet dispatch setups). |
| **`--model`** | `yolov8s.pt` | Ultralytics weights path/name when **`--backend cpu`**. |

## Behavior notes

- **Letterbox:** TT path calls `preprocess([image], res=(640, 640))` **once**, then **`.repeat(batch_size, …)`** on the tensor—no per-device duplicate letterbox on the host.
- **Sharding:** Inputs use **`ShardTensorToMesh(..., dim=0)`**; weights **`ReplicateTensorToMesh`**; outputs **`ConcatMeshToTensor(dim=0)`**. One **`runner.run()`** per timed iteration, not a Python loop of N separate single-device forwards.
- **Timing (aligned with ResNet `ttnn_resnet` demo):**
  - **`device(run+sync)`** — **`runner.run(im)`** plus **`ttnn.synchronize_device(device)`** inside the same timer. This matches `models/demos/vision/classification/resnet50/ttnn_resnet/demo/demo.py` **`model_run_for_inference`** (`test_infra.run()` then **`synchronize_device`**). **`ttnn.to_torch`** and NMS/scale stay in **post**, like ResNet’s **`post_processing`** bucket.
  - **`setup`** — mesh open + performant runner construction (includes first-hit compile/trace effects).
  - **`pre`** — letterbox + host batch tensor.
  - **`post`** — D2H (`to_torch`) + NMS/scale + optional image saves.
  - With **`--tt-measured-iters > 1`**, the script prints min/mean/max of **`device(run+sync)`** across timed iterations.

### How `perf_device_resnet50` differs (CI / Tracy)

The pytest **`test_perf_device`** in `wormhole/tests/test_perf_device_resnet50.py` does **not** use that Python timer. It calls **`run_perf_device`** → **`run_device_perf`** in `models/perf/device_perf_utils.py`, which:

1. Clears Tracy/profiler artifacts.
2. Runs **`run_device_profiler(command, subdir, …)`** with a **pytest command** that executes the performant ResNet test (e.g. `test_run_resnet50_inference[...]`).
3. Post-processes the **device ops log** CSV (`post_process_ops_log`): aggregates **kernel duration** columns such as **`DEVICE KERNEL DURATION [ns]`**, optionally scoped by **signposts** (`start` / `stop`) in the trace.
4. Converts durations to **samples/s** via **`get_samples_per_s`** and compares to **`expected_perf`** (e.g. **6140** images/s for batch 16 in the wormhole test).

So **`perf_device_resnet50`** is **on-device kernel / profiler derived throughput**, not the same quantity as **`device(run+sync)`** wall time in this script or the ResNet demo’s **`time.time()`** profiler block. Use **demo timing** for host-wall **run+sync**; use **device perf tests** for **kernel-level** regression against published targets.

## CPU example

```sh
python models/demos/yolo_eval/yolo_dp_mesh_infer.py \
  --backend cpu \
  --input models/demos/yolo_eval/sample_images/crowded_freeway.jpg \
  --batch-size 32 \
  --model yolov8s.pt \
  --save-images
```

## See also

- [README_SAHI.md](README_SAHI.md) — SAHI sliced inference + `sahi_ultralytics_eval.py`
- [README.md](README.md) — broader yolo_eval / pytest evaluation
- [YOLOv8s README](../yolov8s/README.md) — Trace+2CQ (hand-rolled vs `tt_cnn` pipeline), Galaxy **8×4** pytest, `run_yolov8s_trace_2cqs_tt_cnn_pipeline_inference`
