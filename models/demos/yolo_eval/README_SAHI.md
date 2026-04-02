# SAHI + Ultralytics / Tenstorrent YOLO eval

This harness compares **full-image** vs **SAHI sliced** inference (large images, small objects). It supports **`--backend ultralytics`** (PyTorch/GPU/CPU) and **`--backend tt`** (Tenstorrent YOLOv8s / YOLOv8x performant runners).

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

### libGL / `ImportError: libGL.so.1`

On minimal Linux images, install Mesa GL user libraries:

```bash
sudo apt-get install -y libgl1
```

(or use only `opencv-python-headless` and avoid pulling in the full `opencv-python` GUI stack.)

### ttnn / NumPy / OpenCV pins

Upgrading `opencv-python-headless` without a cap can pull **NumPy 2.x**, which conflicts with **ttnn** (`numpy<2`) and common **Ultralytics** pins. If you hit that, constrain NumPy and OpenCV:

```bash
python -m pip install -U "numpy>=1.24.4,<2" "opencv-python-headless<=4.11.0.86"
```

### Example: explicit `python_env` path

Replace `/path/to/tt-metal` with your checkout (for example `/home/cust-team/sdawle/image_slicing/tt-metal`):

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

Set **`PYTHONPATH`** and **`TT_METAL_HOME`** to the repo root when running TT (same as other demos):

```bash
export TT_METAL_HOME="$(pwd)"
export PYTHONPATH="$(pwd)"
```

### Tenstorrent (`--backend tt`) — current behavior

- **Full-image path:** Image is loaded with **PIL + EXIF** (`read_image_as_pil`), converted to BGR for the TT runner. Inference uses a **640×640 letterbox** (`preprocess`); boxes are scaled back to the original resolution via `scale_boxes` in `common_demo_utils.postprocess`.
- **Sliced path:** SAHI `slice_image` builds tiles. Each tile is inferred; predictions are **shifted to full-image coordinates** before SAHI merge (NMS / NMM / GREEDYNMM / LSNMS), matching `sahi.predict.get_sliced_prediction` semantics.
- **`--tt-slice-parallel-devices N`:** Opens **`MeshShape(1, N)`** — **N** chips in one row. Each forward can run **up to N distinct** 640×640 crops (**one slice per chip**). The last chunk is padded with black if fewer than N tiles remain (padded slots are not merged). **`N` must be ≤ the number of devices** reported by the system (or configured with `--tt-mesh-shape`). Omit this flag for **sequential** per-slice inference; without it, a single slice may be **replicated** across the opened mesh per forward.
- **Teardown:** Slice-parallel mode does **not** use `create_submesh()` off a larger parent mesh (avoids CQ sharing issues on close).
- **Verification:** On TT init the script prints **`TT device verify:`** with `num_devices`, `mesh_shape`, `device_batch_size`, and how the device was opened. **`summary.json`** includes **`config.tt_device_verify`** with the same fields.
- **Timing:** For each image, stdout shows **full** and **sliced** lines with **`pre` / `device` / `post`** (seconds). Per-image and aggregate breakdowns are in **`summary.json`** (`full_timing_sec`, `sliced_timing_sec`, `aggregate.mean_*_timing_sec`).
- **`--save-slice-images`:** After each slice’s inference, saves **`OUTPUT_DIR/<stem>_slices/slice_NNN_xX_yY.png`** with **detections drawn** (slice-local boxes, **before** cross-tile merge). No separate script.

### Multi-chip / Ethernet dispatch

On systems such as T3K (1×8), you may need Ethernet dispatch (same idea as some Metal tests). Either:

```bash
export TT_METAL_GTEST_ETH_DISPATCH=1
```

or pass **`--tt-eth-dispatch`** (the script sets the env var before `ttnn` loads if this flag appears early on the command line).

---

### Example: TT reference command (YOLOv8s, 1280×1280, 640 tiles, NMM, parallel mesh)

Below is a **copy-paste** command using paths under this repo. Adjust **`--tt-slice-parallel-devices`** to match your hardware (e.g. **`4`** for a 2×2 grid on 1280×1280 with 640×640 tiles and zero overlap). **`32`** is only valid if the machine exposes **at least 32** mesh devices; otherwise the script raises an error.

```bash
python models/demos/yolo_eval/sahi_ultralytics_eval.py \
  --backend tt \
  --tt-model yolov8s \
  --pre-resize-to 1280 1280 \
  --slice-height 640 \
  --slice-width 640 \
  --overlap-height-ratio 0 \
  --overlap-width-ratio 0 \
  --postprocess-type NMM \
  --postprocess-match-metric IOU \
  --postprocess-match-threshold 0.2 \
  --confidence-threshold 0.50 \
  --save-visuals \
  --save-slice-grid-overlay \
  --tt-slice-parallel-devices 32 \
  --input '/home/cust-team/sdawle/image_slicing/tt-metal/models/demos/yolo_eval/sample_images/crowded_freeway.jpg' \
  --output-dir '/home/cust-team/sdawle/image_slicing/tt-metal/models/demos/yolo_eval/sample_images_output'
```

**Typical 4-tile 1280×1280 run** (replace `--tt-slice-parallel-devices 32` with **`4`** on a 4-chip or 1×4 mesh):

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
  --postprocess-type NMM \
  --postprocess-match-metric IOU \
  --postprocess-match-threshold 0.2 \
  --confidence-threshold 0.50 \
  --save-visuals \
  --save-slice-grid-overlay \
  --input '/home/cust-team/sdawle/image_slicing/tt-metal/models/demos/yolo_eval/sample_images/crowded_freeway.jpg' \
  --output-dir '/home/cust-team/sdawle/image_slicing/tt-metal/models/demos/yolo_eval/sample_images_output'
```

Optional: add **`--save-slice-images`** to export per-tile PNGs with boxes (same folder pattern as above).

---

### Ultralytics backend (single image)

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

Quick synthetic image:

```bash
python -c "from PIL import Image; Image.new('RGB',(1024,768),(120,80,200)).save('/tmp/sahi_test_image.png')"
python models/demos/yolo_eval/sahi_ultralytics_eval.py --input /tmp/sahi_test_image.png --model yolov8x.pt --slice-height 320 --slice-width 320 --overlap-height-ratio 0.2 --overlap-width-ratio 0.2
```

Directory of images (Ultralytics):

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

TT backend, **sequential slices** (omit `--tt-slice-parallel-devices`; multi-chip mesh may still replicate one slice per forward):

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

## 3) Outputs

- **Stdout (per image):**
  - Detection counts and wall times for full vs sliced.
  - Two lines with **`pre` / `device` / `post`** timing for **full (single)** and **sliced** (seconds; see `sahi_ultralytics_eval.py` for definitions).
  - If **`--save-slice-images`**, an extra line reports **`slice_export`** I/O time.
- **`summary.json`** (under `--output-dir`):
  - **`config`:** run settings, **`tt_device_verify`** (TT only), etc.
  - **`per_image`:** `full_timing_sec`, `sliced_timing_sec`, detection deltas.
  - **`aggregate`:** means including **`mean_full_timing_sec`**, **`mean_sliced_timing_sec`**.
- **Visuals (optional):**
  - `<stem>_full.*`, `<stem>_sliced.*`
  - `<stem>_slice_grid.png` — tile boundaries on the input.
  - `<stem>_slices/*.png` — per-tile predictions if **`--save-slice-images`**.

Default output directory if omitted: `models/demos/yolo_eval/sahi_outputs`.

## Notes for tuning

- Start with `slice-height/width` in the range **512–768** (or **640** to match YOLO native size).
- Increase overlap (`0.2 → 0.3`) when objects are often cut at tile boundaries.
- Sliced inference is usually slower; the tradeoff is often better recall on small objects.
- Duplicate boxes across tiles: try **`--postprocess-type NMS`** with **`--postprocess-match-metric IOU`** / threshold tuning, or stricter **`--confidence-threshold`**.
- Fixed tiling: **`--pre-resize-to W H`** with **`--overlap-*-ratio 0`** gives a predictable grid (e.g. 1280×1280 + 640×640 + no overlap ⇒ **4** tiles).

## Offline tests

Chunk batching for **`--tt-slice-parallel-devices`** (no device):

```bash
pytest models/demos/yolo_eval/test_sahi_parallel_chunks.py
```
