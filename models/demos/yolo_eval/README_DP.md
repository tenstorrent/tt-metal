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
| **`--tt-mesh-shape ROWS COLS`** | `8 4` | `MeshShape(ROWS, COLS)` passed to `open_mesh_device`. **`ROWS * COLS`** must match **`--batch-size`**. Adjust for your machine (e.g. `1 8` on an 8-chip row mesh). |
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
- **Sharding:** Inputs use **`ShardTensorToMesh(..., dim=0)`**; weights **`ReplicateTensorToMesh`**; outputs **`ConcatMeshToTensor(dim=0)`**. One **`runner.run()`** per invocation, not a Python loop of 32 separate forwards.
- **Timing:** After the run, the script prints **setup** (mesh open + runner build), **pre** (letterbox + tensor), **device** (`runner.run`), **post** (D2H + NMS/scale + optional saves), and **total**. First run includes compile/trace cost in **setup**.

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
