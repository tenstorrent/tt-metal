# SAHI + Ultralytics quick test

This adds a lightweight test harness to evaluate whether sliced inference with SAHI helps your use case (for example, large images with small objects).

## 1) Install dependencies

From repo root, use the same Python interpreter you use for TT Metal / demos (for example `tt-metal/python_env/bin/python`).

### If `pip` is missing (`No module named pip`)

Bootstrap and upgrade `pip`:

```bash
python -m ensurepip --upgrade && python -m pip install -U pip
```

### Core packages

```bash
python -m pip install -U ultralytics sahi
```

If your environment already uses `tt-metal/tt_metal/python_env/requirements-dev.txt`, Ultralytics is pinned there and you can install only SAHI:

```bash
python -m pip install -U sahi
```

### Headless OpenCV (servers / no GUI)

Prefer the headless OpenCV wheel. After installing SAHI, remove the full `opencv-python` package if it is present, then install `opencv-python-headless`:

```bash
python -m pip install -U sahi && python -m pip uninstall -y opencv-python || true
python -m pip install -U opencv-python-headless
```

### ttnn / NumPy / OpenCV pins

Upgrading `opencv-python-headless` without a cap can pull **NumPy 2.x**, which conflicts with **ttnn** (`numpy<2`) and common **Ultralytics** pins. If you hit that, constrain NumPy and OpenCV:

```bash
python -m pip install -U "numpy>=1.24.4,<2" "opencv-python-headless<=4.11.0.86"
```

### Example: explicit `python_env` path

Replace `/path/to/tt-metal` with your checkout (for example `/home/you/tt-metal`):

```bash
"/path/to/tt-metal/python_env/bin/python" -m ensurepip --upgrade && \
"/path/to/tt-metal/python_env/bin/python" -m pip install -U pip

"/path/to/tt-metal/python_env/bin/python" -m pip install -U sahi && \
"/path/to/tt-metal/python_env/bin/python" -m pip uninstall -y opencv-python || true
"/path/to/tt-metal/python_env/bin/python" -m pip install -U opencv-python-headless

"/path/to/tt-metal/python_env/bin/python" -m pip install -U "numpy>=1.24.4,<2" "opencv-python-headless<=4.11.0.86"
```

Pip may warn that some packages list a dependency on the `opencv-python` *distribution*; `opencv-python-headless` still provides `cv2` and is the right choice for headless machines.

## 2) Run baseline vs sliced inference

### Tenstorrent (`--backend tt`) behavior (current implementation)

- **Full-image path:** The image is loaded with **PIL + EXIF orientation** (`ImageOps.exif_transpose`) and converted to BGR for the TT runner, so full-image boxes line up with SAHI’s tiling and `export_visuals` (OpenCV `imread` alone does not apply EXIF).
- **Sliced path:** Per-slice boxes are shifted to **full-image coordinates before** SAHI merge postprocess (NMS / NMM / GREEDYNMM / LSNMS), matching `sahi.predict.get_sliced_prediction`. Running merge on slice-local boxes only breaks overlap logic and can misalign overlays.
- **`--tt-slice-parallel-devices N`:** Opens a dedicated **`MeshShape(1, N)`** mesh (first N devices). Each forward can run **up to N different** 640×640 tiles—**one distinct slice per chip**—with black padding if the last chunk has fewer than N tiles. Omit this flag to run slices **sequentially**; in that mode each slice is **replicated** on all devices in the batch (same input on every chip per forward).
- **Teardown:** Slice-parallel mode does **not** use `create_submesh()` off a larger parent mesh, avoiding Metal errors when closing meshes that share command queues.

Single image:

```bash
python models/demos/yolo_eval/sahi_ultralytics_eval.py \
  --input /absolute/path/to/image.jpg \
  --model yolo11n.pt \
  --pre-resize-to 640 640 \
  --slice-height 512 \
  --slice-width 512 \
  --overlap-height-ratio 0.2 \
  --overlap-width-ratio 0.2 \
  --postprocess-type NMS \
  --postprocess-match-metric IOU \
  --postprocess-match-threshold 0.5 \
  --save-visuals \
  --save-slice-grid-overlay
```

Optional quick sanity image:

```bash
python -c "from PIL import Image; Image.new('RGB',(1024,768),(120,80,200)).save('/tmp/sahi_test_image.png')"
```

Then run:

```bash
python models/demos/yolo_eval/sahi_ultralytics_eval.py --input /tmp/sahi_test_image.png --model yolov8x.pt --slice-height 320 --slice-width 320 --overlap-height-ratio 0.2 --overlap-width-ratio 0.2
```

Directory of images:

```bash
python models/demos/yolo_eval/sahi_ultralytics_eval.py \
  --input /absolute/path/to/images_dir \
  --model yolo11n.pt \
  --slice-height 640 \
  --slice-width 640 \
  --overlap-height-ratio 0.25 \
  --overlap-width-ratio 0.25 \
  --device cuda:0 \
  --save-visuals
```

TT backend, **sequential slices** (default multi-chip behavior: one slice at a time, replicated on all devices in the mesh):

```bash
python models/demos/yolo_eval/sahi_ultralytics_eval.py \
  --backend tt \
  --tt-model yolov8x \
  --input /absolute/path/to/large_image_or_dir \
  --pre-resize-to 1280 1280 \
  --slice-height 640 \
  --slice-width 640 \
  --overlap-height-ratio 0 \
  --overlap-width-ratio 0 \
  --postprocess-type GREEDYNMM \
  --postprocess-match-metric IOS \
  --postprocess-match-threshold 0.1 \
  --confidence-threshold 0.55 \
  --save-visuals \
  --save-slice-grid-overlay
```

**Multi-chip / Ethernet dispatch:** On systems such as T3K (1×8), you may want dispatch on Ethernet cores (same idea as metal unit tests). Either export before Python:

```bash
export TT_METAL_GTEST_ETH_DISPATCH=1
```

or pass the script flag (sets the variable before `ttnn` loads):

```bash
python models/demos/yolo_eval/sahi_ultralytics_eval.py --tt-eth-dispatch --backend tt ...
```

**Sample: T3K-style 4-way slice parallel (YOLOv8s, 1280×1280 → four 640×640 tiles, no overlap)**
Use this when you want **four different slices per forward** on a **1×4** mesh. Adjust `--input` and `--output-dir` to your paths; from repo root:

```bash
python models/demos/yolo_eval/sahi_ultralytics_eval.py \
  --backend tt \
  --tt-model yolov8s \
  --tt-eth-dispatch \
  --tt-slice-parallel-devices 4 \
  --pre-resize-to 1280 1280 \
  --slice-height 640 \
  --slice-width 640 \
  --overlap-height-ratio 0 \
  --overlap-width-ratio 0 \
  --postprocess-type GREEDYNMM \
  --postprocess-match-metric IOS \
  --postprocess-match-threshold 0.1 \
  --confidence-threshold 0.55 \
  --save-visuals \
  --save-slice-grid-overlay \
  --input /absolute/path/to/image.jpg \
  --output-dir models/demos/yolo_eval/sahi_outputs/my_run
```

Same flags with a typical **local** layout (repo at `~/sdawle/tt-metal`, sample images under `~/sdawle/sample_images/`; change paths as needed):

```bash
python models/demos/yolo_eval/sahi_ultralytics_eval.py \
  --backend tt \
  --tt-model yolov8s \
  --tt-eth-dispatch \
  --tt-slice-parallel-devices 4 \
  --pre-resize-to 1280 1280 \
  --slice-height 640 \
  --slice-width 640 \
  --overlap-height-ratio 0 \
  --overlap-width-ratio 0 \
  --postprocess-type GREEDYNMM \
  --postprocess-match-metric IOS \
  --postprocess-match-threshold 0.1 \
  --confidence-threshold 0.55 \
  --save-visuals \
  --save-slice-grid-overlay \
  --input "$HOME/sdawle/sample_images/people/pedestrians_03.jpg" \
  --output-dir models/demos/yolo_eval/sahi_outputs/citytraffic_yolov8s_tt
```

If the last batch has fewer than N slices, the remainder is padded with black images; padded slots are not merged into SAHI output. Omit **`--tt-slice-parallel-devices`** for sequential slicing (each slice replicated across the opened mesh per forward).

## 3) Outputs

- Printed per-image comparison:
  - full-image detection count + latency
  - sliced detection count + latency
  - detection delta (sliced - full)
- JSON summary:
  - `<output-dir>/summary.json` (default output dir: `models/demos/yolo_eval/sahi_outputs`)
- Optional visual outputs:
  - `<image>_full.*`
  - `<image>_sliced.*`
  - `<image>_slice_grid.png` (original image with SAHI tile boundaries)

## Notes for tuning

- Start with `slice-height/width` in the range `512-768`.
- Increase overlap (`0.2 -> 0.3`) when objects are often cut at tile boundaries.
- Expect sliced inference to be slower; the tradeoff is often better recall on small objects.
- If you see many duplicate boxes, use stricter merge settings:
  - `--postprocess-type NMS --postprocess-match-metric IOU --postprocess-match-threshold 0.5`
  - optionally raise `--confidence-threshold` to `0.35` or `0.4`
- If you need an exact fixed-size tiling setup, use `--pre-resize-to WIDTH HEIGHT`.
  - Example: `--pre-resize-to 640 640 --slice-height 320 --slice-width 320 --overlap-height-ratio 0 --overlap-width-ratio 0`
  - This yields exactly `4` tiles per image.

## Offline tests

Chunk batching for `--tt-slice-parallel-devices` (no device):

```bash
pytest models/demos/yolo_eval/test_sahi_parallel_chunks.py
```
