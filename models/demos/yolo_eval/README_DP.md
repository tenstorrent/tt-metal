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

Same sharding as above, but `**MeshShape` comes from `SystemMeshDescriptor()**` (like `sahi_ultralytics_eval.py` and multi-device ResNet bring-up). `**--batch-size` is adjusted** to the reported device count if it does not match (with a one-line note). `**--tt-mesh-shape` is ignored.**

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


| Flag                            | Default                                                 | Meaning                                                                                                                                                                                                                    |
| ------------------------------- | ------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `**--input**`                   | *(required)*                                            | Path to one image file (BGR read via OpenCV).                                                                                                                                                                              |
| `**--backend*`*                 | `tt`                                                    | `tt` = Tenstorrent mesh + YOLOv8s/x performant runner; `cpu` = Ultralytics on the host.                                                                                                                                    |
| `**--batch-size**`              | `32`                                                    | Logical batch size. For `**--backend tt**` it **must** equal the number of opened mesh devices (`**ROWS × COLS`** from `--tt-mesh-shape`). Each batch row is one chip’s input (identical image in the typical smoke test). |
| `**--tt-mesh-shape ROWS COLS**` | `8 4`                                                   | `MeshShape(ROWS, COLS)` passed to `open_mesh_device`. `**ROWS * COLS**` must match `**--batch-size**`. Ignored if `**--tt-use-system-mesh**`.                                                                              |
| `**--tt-use-system-mesh**`      | off                                                     | Open the mesh from `**SystemMeshDescriptor()**` (full system topology). Aligns `**--batch-size**` to the device count. Use on Galaxy to match ResNet DP=32-style mesh open without hard-coding `8 4`.                      |
| `**--tt-warmup-iters N**`       | `0`                                                     | Untimed `**runner.run()**` calls after setup to flush compile/trace effects before measured iters.                                                                                                                         |
| `**--tt-measured-iters N**`     | `1`                                                     | Timed forwards; when `**N>1**`, prints **device(run)** min / mean / max across those iters.                                                                                                                                |
| `**--tt-model`**                | `yolov8s`                                               | TT demo variant: `yolov8s` or `yolov8x` (both use 640×640 internal resolution).                                                                                                                                            |
| `**--save-images**`             | off                                                     | If set, writes `{input_stem}_slot_{00..N-1}.jpg` under `**--output-dir**`. If omitted, no JPEGs are created (timing and verify prints still run).                                                                          |
| `**--output-dir**`              | `models/demos/yolo_eval/sample_images_output/dp{N}_glx` | Output folder when `**--save-images**` is set (`N` = `--batch-size`). Created only when saving.                                                                                                                            |
| `**--tt-device-id**`            | `0`                                                     | Device id when using `**--tt-force-single-device**` (single chip).                                                                                                                                                         |
| `**--tt-l1-small-size**`        | `24576`                                                 | TT device `l1_small_size` (YOLOv8 demos default).                                                                                                                                                                          |
| `**--tt-trace-region-size**`    | `6434816`                                               | TT device `trace_region_size` (YOLOv8s demo default).                                                                                                                                                                      |
| `**--tt-force-single-device**`  | off                                                     | Open one chip with `open_device` instead of a mesh (only consistent with `**--batch-size 1**` for this script’s TT path).                                                                                                  |
| `**--tt-eth-dispatch**`         | off                                                     | Sets `TT_METAL_GTEST_ETH_DISPATCH=1` before ttnn init (often needed on multi-chip Ethernet dispatch setups).                                                                                                               |
| `**--model**`                   | `yolov8s.pt`                                            | Ultralytics weights path/name when `**--backend cpu**`.                                                                                                                                                                    |


## Behavior notes

- **Letterbox:** TT path calls `preprocess([image], res=(640, 640))` **once**, then `**.repeat(batch_size, …)`** on the tensor—no per-device duplicate letterbox on the host.
- **Sharding:** Inputs use `**ShardTensorToMesh(..., dim=0)`**; weights `**ReplicateTensorToMesh**`; outputs `**ConcatMeshToTensor(dim=0)**`. One `**runner.run()**` per timed iteration, not a Python loop of N separate single-device forwards.
- **Timing (aligned with ResNet `ttnn_resnet` demo):**
  - `**device(run+sync)`** — `**runner.run(im)**` plus `**ttnn.synchronize_device(device)**` inside the same timer. This matches `models/demos/vision/classification/resnet50/ttnn_resnet/demo/demo.py` `**model_run_for_inference**` (`test_infra.run()` then `**synchronize_device**`). `**ttnn.to_torch**` and NMS/scale stay in **post**, like ResNet’s `**post_processing`** bucket.
  - `**setup**` — mesh open + performant runner construction (includes first-hit compile/trace effects).
  - `**pre**` — letterbox + host batch tensor.
  - `**post**` — D2H (`to_torch`) + NMS/scale + optional image saves.
  - With `**--tt-measured-iters > 1**`, the script prints min/mean/max of `**device(run+sync)**` across timed iterations.

### How `perf_device_resnet50` differs (CI / Tracy)

The pytest `**test_perf_device**` in `wormhole/tests/test_perf_device_resnet50.py` does **not** use that Python timer. It calls `**run_perf_device`** → `**run_device_perf**` in `models/perf/device_perf_utils.py`, which:

1. Clears Tracy/profiler artifacts.
2. Runs `**run_device_profiler(command, subdir, …)**` with a **pytest command** that executes the performant ResNet test (e.g. `test_run_resnet50_inference[...]`).
3. Post-processes the **device ops log** CSV (`post_process_ops_log`): aggregates **kernel duration** columns such as `**DEVICE KERNEL DURATION [ns]`**, optionally scoped by **signposts** (`start` / `stop`) in the trace.
4. Converts durations to **samples/s** via `**get_samples_per_s`** and compares to `**expected_perf**` (e.g. **6140** images/s for batch 16 in the wormhole test).

So `**perf_device_resnet50`** is **on-device kernel / profiler derived throughput**, not the same quantity as `**device(run+sync)`** wall time in this script or the ResNet demo’s `**time.time()**` profiler block. Use **demo timing** for host-wall **run+sync**; use **device perf tests** for **kernel-level** regression against published targets.

## CPU example

```sh
python models/demos/yolo_eval/yolo_dp_mesh_infer.py \
  --backend cpu \
  --input models/demos/yolo_eval/sample_images/crowded_freeway.jpg \
  --batch-size 32 \
  --model yolov8s.pt \
  --save-images
```

## Mesh vs Submesh vs Independent — 32-chip comparison

Model: YOLOv8s, 640×640, WH Galaxy (32 chips), `--tt-warmup-iters 2 --tt-measured-iters 5`.

All three scripts use the same **split timing** approach (matching `test_e2e_performant`):
`prepare_host_input` (once, outside loop) → `push_host_input_to_device_dram` (H2D) → `execute_reshard_and_trace` (compute).

### 1. Mesh DP (BS=32, 1 runner) — `yolo_dp_mesh_infer.py`

| Stage | Total | What this includes |
|:------|------:|:-------------------|
| **Setup** | 7.39 s | Open mesh (32 devices), load/preprocess weights, DRAM input, trace capture |
| **Pre** | 0.084 s | Read/normalize image to 640×640 = 0.007 s |
| | | `prepare_host_input` — split batch for 32 shards (once, not repeated each iter) = 0.078 s |
| **Device** (bs=32, dp) | **0.0064 s** | host input → device DRAM (`push_host_input_to_device_dram`) = 1.0 ms |
| | | DRAM → L1 reshard, run captured trace, blocking=True, `synchronize_device` = 5.5 ms |
| **Post** | 0.023 s | `ttnn.to_torch` + composer = 0.013 s |
| | | NMS, scale boxes to original image coords = 0.010 s |
| **TOTAL** | **7.55 s** | |

**Total Device Time = 0.0064 s = 6.4 ms → 1000/6.4 = 156 fps × 32 = 4,992 fps**

### 2. Submesh `--parallel` (32× 1×1, threads) — `yolo_dp_submesh_infer.py --parallel`

| Stage | Total | What this includes |
|:------|------:|:-------------------|
| **Setup** | 3.60 s (mesh) + 20.7 s (runners) | Open parent mesh + `create_submeshes(1,1)` = 3.60 s |
| | | Build 32 `YOLOv8sPerformantRunner` concurrently (mean per slot) = 20.7 s |
| **Pre** | 0.09 s | Read/normalize image to 640×640 (shared across slots, done once) |
| | | `prepare_host_input` per slot (mean) = 3.0 ms |
| **Device** (per slot, mean) | **0.017 s** | host input → device DRAM (`push_host_input_to_device_dram`) = 0.3 ms |
| | | DRAM → L1 reshard, run captured trace, `synchronize_device` = 16.5 ms |
| | | *min–max across 32 slots × 5 iters: 4.5–91.1 ms (shared CQ contention)* |
| **Post** (per slot, mean) | 0.86 s | `ttnn.to_torch` = 0.025 s |
| | | NMS, scale boxes = 0.84 s (GIL contention across 32 threads) |
| **TOTAL** | **28.7 s** | |

**Total Device Time = 0.017 s = 16.8 ms per slot → 1000/16.8 = 60 fps × 32 = 1,905 fps (limited by shared CQ)**

### 3. Independent (32 subprocesses) — `yolo_dp_32_independent_infer.py`

| Stage | Total (mean per worker) | What this includes |
|:------|------------------------:|:-------------------|
| **Setup** | 105 s | `TT_VISIBLE_DEVICES=i`, `open_device(0)`, build runner (each subprocess) |
| **Pre** | 0.08 s | Read/normalize image to 640×640 = 0.031 s |
| | | Letterbox + tensor = 0.052 s |
| | | `prepare_host_input` = 6.6 ms |
| **Device** (per chip, mean) | **0.011 s** | host input → device DRAM (`push_host_input_to_device_dram`) = 1.6 ms |
| | | DRAM → L1 reshard, run captured trace, `synchronize_device` = 9.3 ms |
| | | *min–max across 32 chips × 5 iters: 4.3–40.8 ms (PCIe/memory contention)* |
| **Post** (mean per worker) | 0.26 s | `ttnn.to_torch` = 0.029 s |
| | | NMS, scale boxes = 0.234 s |
| **TOTAL** | **115 s** | |

**Total Device Time = 0.011 s = 11.0 ms per chip → 1000/11.0 = 91 fps × 32 = 2,909 fps (isolated CQs)**

### Side-by-side summary

| | Mesh (bs=32) | 32 Independant Runners (1x1 Submesh parallel) | Independent |
|:--|-------------:|---------------------:|------------:|
| **Device compute** (reshard+trace+sync) | **5.5 ms** | 16.5 ms mean | 9.3 ms mean |
| **Device total** (h2d+compute) | **6.4 ms** | 16.8 ms mean | 11.0 ms mean |
| **Effective FPS** (device only × 32) | **4,992** | 1,905 | 2,909 |
| **Total wall** | **7.55 s** | 28.7 s | 115 s |
| **Setup** | 7.39 s | 3.60 s + 20.7 s | 105 s |

### Why submesh `--parallel` compute is higher than independent

The per-chip compute floor is ~4.5 ms (visible in min-of-mins across all approaches). The submesh mean (~16.5 ms) is inflated by:

1. **Shared command queues (C++ mutexes):** All 32 submeshes share the parent mesh's `FDMeshCommandQueue`. Each CQ operation acquires a C++ mutex (`lock_api_function_`), so `push_dram`, `reshard`, and `execute_trace` serialize at the C++ level. Independent subprocesses each have their own `open_device` and CQ — no mutex sharing. This is the dominant bottleneck.

2. **Python GIL:** `performant_runner.py` calls 6 ttnn functions in sequence per iteration (`wait_for_event` → `copy_host_to_device_tensor` → `record_event` → `reshard` → `execute_trace` → `synchronize_device`). The GIL serializes these calls across 32 threads. Releasing the GIL during C++ calls was tested but made things *worse* — threads rushed into the C++ code simultaneously and created more contention on the shared CQ mutex (lock convoys). The GIL's orderly serialization actually reduces CQ mutex contention.

To make submesh `--parallel` faster, the fundamental fix is **per-submesh command queues** at the C++ level, so each 1×1 submesh has its own `FDMeshCommandQueue` independent of the parent mesh.

## See also

- [README_SAHI.md](README_SAHI.md) — SAHI sliced inference + `sahi_ultralytics_eval.py`
- [README.md](README.md) — broader yolo_eval / pytest evaluation
- [YOLOv8s README](../yolov8s/README.md) — Trace+2CQ (hand-rolled vs `tt_cnn` pipeline), Galaxy **8×4** pytest, `run_yolov8s_trace_2cqs_tt_cnn_pipeline_inference`
