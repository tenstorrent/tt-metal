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
- **`--tt-slice-parallel-devices N`:** Opens a mesh of **N** chips for slice batching (default topology **`MeshShape(1, N)`** — one row). Each forward runs **up to N distinct** 640×640 crops (**one real slice per chip**). The last chunk is padded with black if fewer than N tiles remain (padded slots are not merged). **`N` must be ≤** the device count implied by **`--tt-mesh-shape`** when set, or the system mesh descriptor. Omit this flag for **sequential** per-slice inference; without it, a single slice may be **replicated** across the opened mesh per forward.
- **`--tt-slice-parallel-mesh-shape ROWS COLS`** *(optional):* Use **`MeshShape(ROWS, COLS)`** instead of **`(1, N)`**, as long as **`ROWS * COLS == N`** (e.g. **`8 3`** with **`--tt-slice-parallel-devices 24`** on Galaxy-style layouts). Requires **`--tt-slice-parallel-devices`**. If omitted, behavior is unchanged: **`MeshShape(1, N)`**.
- **`--tt-mesh-shape ROWS COLS`** *(optional):* Caps / configures how many devices are considered available when validating **`N`** for slice-parallel mode; the slice-parallel path still opens exactly **`N`** devices with the topology above (not the full **`8×4`** mesh unless **`N`** equals 32 and you choose a matching shape).
- **Teardown:** Slice-parallel mode does **not** use `create_submesh()` off a larger parent mesh (avoids CQ sharing issues on close).
- **Verification:** On TT init the script prints **`TT device verify:`** with `num_devices`, `mesh_shape`, `device_batch_size`, `tt_slice_parallel_mesh_shape`, and how the device was opened. **`summary.json`** includes **`config.tt_device_verify`** (and **`config.tt_slice_parallel_mesh_shape`**) with the same fields.
- **Timing:** Per-image stdout and **`summary.json`** timing fields are documented in **[Timing breakdown](#timing-breakdown-full-image-vs-sliced-tt-vs-ultralytics)** below (coarse **`pre` / `device` / `post`**, detail-line labels, JSON keys, and what is **not** bucketed).
- **`--save-slice-images`:** After each slice’s inference, saves **`OUTPUT_DIR/<stem>_slices/slice_NNN_xX_yY.png`** with **detections drawn** (slice-local boxes, **before** cross-tile merge). No separate script.

### Multi-chip / Ethernet dispatch

On systems such as T3K (1×8), you may need Ethernet dispatch (same idea as some Metal tests). Either:

```bash
export TT_METAL_GTEST_ETH_DISPATCH=1
```

or pass **`--tt-eth-dispatch`** (the script sets the env var before `ttnn` loads if this flag appears early on the command line).

---

### Example: TT reference command (YOLOv8s, 1280×1280, 640 tiles, NMM, parallel mesh)

Below is a **copy-paste** command using paths under this repo. Adjust **`--tt-slice-parallel-devices`** to match your hardware (e.g. **`4`** for a 2×2 grid on 1280×1280 with 640×640 tiles and zero overlap). **`32`** is only valid if the machine exposes **at least 32** mesh devices; otherwise the script raises an error. For **`--tt-slice-parallel-mesh-shape`**, **`ROWS*COLS`** must equal **`N`**.

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

### Example: 3840×2160 UHD, 6×4 = 24 tiles, slice-parallel 24, mesh 8×3

Resize input to **3840×2160**, tile with **640×640** and **no overlap** → **⌈3840/640⌉ × ⌈2160/640⌉ = 6×4 = 24** slices. Use **`--tt-slice-parallel-devices 24`** so one forward processes up to 24 distinct tiles. **`--tt-slice-parallel-mesh-shape 8 3`** opens **`MeshShape(8,3)`** instead of **`(1,24)`**; **`--tt-mesh-shape 8 4`** ensures the configured device budget is at least **32** so **`24 ≤ max_devices`**. Adjust paths to your checkout.

```bash
python models/demos/yolo_eval/sahi_ultralytics_eval.py \
  --backend tt \
  --tt-model yolov8s \
  --tt-eth-dispatch \
  --tt-slice-parallel-devices 24 \
  --tt-slice-parallel-mesh-shape 8 3 \
  --tt-mesh-shape 8 4 \
  --pre-resize-to 3840 2160 \
  --slice-height 640 \
  --slice-width 640 \
  --overlap-height-ratio 0 \
  --overlap-width-ratio 0 \
  --postprocess-type NMM \
  --postprocess-match-metric IOU \
  --postprocess-match-threshold 0.2 \
  --confidence-threshold 0.55 \
  --save-visuals \
  --save-slice-grid-overlay \
  --save-slice-images \
  --input models/demos/yolo_eval/sample_images/crowded_freeway_3840x2160.jpg \
  --output-dir models/demos/yolo_eval/sample_images_output
```

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
  - Two lines with coarse **`pre` / `device` / `post`** timing for **full (single)** and **sliced** (seconds).
  - Extra **detail** lines with granular timers when non-zero.
  - See **[Timing breakdown](#timing-breakdown-full-image-vs-sliced-tt-vs-ultralytics)** for definitions, the print-label ↔ JSON table, and unbucketed work.
- **`summary.json`** (under `--output-dir`):
  - **`config`:** run settings, **`tt_device_verify`** (TT only), **`tt_slice_parallel_mesh_shape`**, etc.
  - **`per_image`:** `full_timing_sec`, `sliced_timing_sec` (including granular timing keys), detection deltas.
  - **`aggregate`:** means including **`mean_full_timing_sec`**, **`mean_sliced_timing_sec`** over the same timing key set.
- **Visuals (optional):**
  - `<stem>_full.*`, `<stem>_sliced.*`
  - `<stem>_slice_grid.png` — tile boundaries on the input.
  - `<stem>_slices/*.png` — per-tile predictions if **`--save-slice-images`**.

Default output directory if omitted: `models/demos/yolo_eval/sahi_outputs`.

## Timing breakdown (full-image vs sliced, TT vs Ultralytics)

This section describes what **`sahi_ultralytics_eval.py`** prints and what each **`full_timing_sec` / `sliced_timing_sec`** field means. All times are wall-clock seconds unless noted.

### Per-image stdout

1. **First line:** Detection counts and **end-to-end** wall time for full vs sliced (`full_time_sec` / `sliced_time_sec` in JSON). This span includes any work that is **not** assigned to the coarse buckets below (see **Gaps** at the end of this section).
2. **Next two lines (coarse triple):** For **full (single)** and **sliced**, labels **`pre`**, **`device`**, **`post`**:
   - **`pre`** = `host_slice_and_preprocess_sec` (derived). **Sliced:** `host_read_image_sec` + `host_sahi_slice_sec` + `host_preprocess_before_device_sec`. **Full single image:** `host_preprocess_before_device_sec` only (for TT full, that already includes **image load** + letterbox/tensor prep—there is no separate add of `host_read_image_sec` in this sum).
   - **`device`** = `device_inference_sec`. **TT:** sum of `runner.run(...)` only (includes host→device / `ttnn` setup inside that call—not split further). **Ultralytics + SAHI sliced:** SAHI’s **`prediction`** duration (model **plus** per-slice host work inside SAHI’s loop—not “device-only”). **Ultralytics full:** SAHI **`prediction`** ≈ model forward.
   - **`post`** = `host_postprocess_and_sahi_merge_sec`. **TT full:** `ttnn.to_torch` + `postprocess` (NMS, `scale_boxes`, etc.)—no SAHI merge. **TT sliced:** per-tile `to_torch` + `postprocess` (summed) + **shift to full image** + **SAHI merge** (NMM/NMS). **Ultralytics / SAHI:** SAHI **`postprocess`** (and merge where applicable).

3. **Detail lines (optional):** A second line per row when any **granular** field is non-zero. Labels are **short**; the canonical names are the **JSON keys** in the table below.

### Detail print label → JSON key → meaning (TT-focused)

| Print label | JSON key | Meaning (TT) |
|-------------|----------|----------------|
| `read` | `host_read_image_sec` | PIL read (+ EXIF) for sliced TT; full TT path uses `load_image_bgr_sahi` and still records this field, while the coarse **`pre`** triple for **full** folds load into `host_preprocess_before_device_sec` (so do not double-count `read` + `pre` on full). |
| `sahi_slice` | `host_sahi_slice_sec` | `slice_image(...)` wall time. |
| `cpu_prep` | `host_cpu_prep_letterbox_sec` | **Sliced TT:** RGB→BGR + `preprocess` (letterbox, tensor) **summed over all slices**; detail may show `(Nt~mean/t)` using `host_cpu_prep_mean_per_slice_sec`. The key name suggests letterbox only, but the timed region is **wider** on the sliced path. |
| `tt_run` | `device_ttnn_run_sec` | Same as summed `runner.run` for TT (includes anything inside that call, e.g. host tensor → device, not broken out). |
| `to_torch` | `host_ttnn_to_torch_sec` | `ttnn.to_torch` on outputs. |
| `tt_post` | `host_tt_torch_postprocess_sec` | `postprocess(...)` on host after torch tensors. |
| `sahi_per_slice` | `sahi_ultralytics_per_slice_host_sec` | **Ultralytics + SAHI** path: sum of SAHI’s per-slice **`postprocess`** timers (not used the same way on pure TT). |
| `sahi_shift` | `sahi_shift_to_full_sec` | SAHI **`get_shifted_object_prediction()`** over all slice predictions—boxes moved from **tile coordinates** to **full-image** coordinates before merge. This is SAHI’s API, matching `get_sliced_prediction` semantics. |
| `sahi_merge` | `sahi_merge_sec` | SAHI merge postprocess (NMM/NMS / configured matcher). |
| `slice_export` | `host_slice_image_export_sec` | Saving tagged per-slice PNGs when **`--save-slice-images`** (or equivalent) is enabled. |
| `sahi_pred` | *(uses `device_inference_sec`)* | Shown when there is **no** `tt_run` but **sliced** Ultralytics/SAHI timing: SAHI **`prediction`** bucket (not pure GPU). |
| `ultra_decode` | *(uses `host_postprocess_and_sahi_merge_sec`)* | **Full-image Ultralytics:** host decode / post bucket from SAHI **`postprocess`**. |

**Sliced path `read` vs SAHI’s own timers:** TT and Ultralytics **tagged** slice flows preload with **`read_image_as_pil`** and pass PIL into **`slice_image`**, so **`read`** and **`sahi_slice`** are separate. Plain SAHI **`get_sliced_prediction`** durations keep SAHI’s internal **`slice`** bucket only.

### `summary.json` timing fields

- **`per_image`:** `full_timing_sec`, `sliced_timing_sec` — same keys as above, plus aggregates such as **`host_slice_and_preprocess_sec`**, **`total_wall_sec`**, and **`extra`** (e.g. **`tt_num_slices`** for sliced TT).
- **`aggregate`:** **`mean_full_timing_sec`**, **`mean_sliced_timing_sec`** — means over the same key set.

### Gaps

Work that still contributes to **`total_wall_sec`** / the first-line wall times but is **not** a separate JSON field:

- **`result_to_object_predictions`** after each forward (full TT: after `infer_*_timed`; sliced TT: inside the per-slice loop).
- **`build_postprocess(...)`** (usually negligible).
- Python overhead between timed regions (list growth, indexing).
- **Inside `tt_run`:** no separate line for `ttnn.from_torch` / H2D vs device execute—everything stays in **`device_run_sec`** / **`device_ttnn_run_sec`**.
- **Visuals not in timing dict:** **`maybe_export_visuals`**, **`save_slice_grid_overlay`** when those flags are on.

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
