# ATSS Swin-L DyHead (TTNN)

TTNN implementation of **ATSS** (Adaptive Training Sample Selection) object detection
with **Swin-L** backbone, **FPN** neck, **DyHead** dynamic head, and **ATSS Head**.

## Architecture

```
Input Image [1, 3, 1200, 2000]
  │
  ▼
Swin-L Backbone (TTNN)      → 3 feature maps (stages 1, 2, 3)
  │                            [384, 768, 1536] channels
  ▼
FPN Neck (TTNN)              → 5 feature maps (P3..P7)
  │                            all 256 channels
  ▼
DyHead Neck (Hybrid*)        → 5 refined feature maps
  │                            6 blocks, DCNv2 + scale/spatial/task attention
  ▼
ATSS Head (TTNN)             → per-level: cls (80), reg (4), centerness (1)
  │
  ▼
Post-processing (CPU)        → bboxes, scores, labels
```

*DyHead runs in hybrid mode by default:
- **Spatial attention** (DCNv2) runs on CPU — no native TTNN kernel yet
- **Scale-aware attention** (AvgPool + Conv + hardsigmoid) runs on TTNN
- **Task-aware attention / DyReLU** (AvgPool + FC + hardsigmoid + element-wise) runs on TTNN

Set `hybrid_dyhead=False` in `from_checkpoint()` to run the entire DyHead on CPU.

## Module Structure

```
models/experimental/atss_swin_l_dyhead/
├── common.py                           # ATSS constants + checkpoint auto-download
├── weights/                            # Checkpoint and config (auto-downloaded)
│   ├── atss_swin-l-p4-w12_fpn_dyhead_mstrain_2x_coco_*.pth
│   └── atss_swin-l-p4-w12_fpn_dyhead_ms-2x_coco.py
├── reference/                          # PyTorch reference implementations
│   ├── swin_transformer.py            #   Swin-L backbone
│   ├── fpn.py                         #   Feature Pyramid Network
│   ├── dyhead.py                      #   Dynamic Head (DCNv2)
│   ├── atss_head.py                   #   ATSS detection head
│   ├── postprocess.py                 #   Anchor gen, bbox decode, NMS
│   └── model.py                       #   Full model assembly + weight loading
├── tt/                                 # TTNN implementations
│   ├── tt_fpn.py                      #   TTNN FPN
│   ├── tt_dyhead.py                   #   Hybrid DyHead (scale/task on TTNN)
│   ├── tt_atss_head.py                #   TTNN ATSS Head
│   ├── tt_atss_model.py               #   Full TTNN model (hybrid)
│   └── weight_loading.py              #   Weight loading for FPN/DyHead/Head
├── tests/pcc/                          # PCC validation tests
│   ├── conftest.py                    #   Shared fixtures
│   ├── test_reference_model.py        #   Reference model vs mmdet
│   ├── test_ttnn_fpn.py               #   TTNN FPN vs reference
│   ├── test_ttnn_atss_head.py         #   TTNN ATSS Head vs reference
│   └── test_ttnn_e2e.py               #   Full E2E TTNN vs reference
├── tests/perf/                         # Performance tests
│   ├── test_atss_swin_l_dyhead_device_perf.py   # Device perf
│   └── test_atss_swin_l_dyhead_e2e_perf.py      # E2E pipeline perf (2CQ + no trace)
├── demo/                               # Demo scripts
│   ├── demo_inference.py              #   Single-image inference + visualization
│   ├── demo_batch.py                  #   Multi-image batch inference
│   ├── demo_perf.py                   #   Performance benchmark (PyTorch vs TTNN)
│   ├── demo_slice_4dev.py             #   4-device slice demo (1280x1280 → 2x2 of 640, 1x4 mesh)
│   └── sweep_merge_4dev.py            #   Sweep cross-tile merge configs (CPU-only post-process)
└── README.md

models/experimental/swin_l/             # Reusable Swin-L backbone (separate module)
├── tt/                                 #   TTNN backbone
│   ├── tt_backbone.py                 #     TtSwinLBackbone
│   ├── tt_swin_attention.py           #     Shifted window attention
│   ├── tt_swin_mlp.py                 #     MLP
│   ├── tt_swin_block.py               #     Transformer block
│   ├── tt_swin_patch_merge.py         #     Patch merging (downsample)
│   └── model_preprocessing.py         #     Backbone weight loading
└── common.py                          #   Swin-L architecture constants
```

## Quick Start

### Prerequisites

```bash
cd $TT_METAL_HOME
source python_env/bin/activate
export ARCH_NAME=wormhole_b0
export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd):$HOME/.local/lib/python3.10/site-packages
```

### Checkpoint

The checkpoint is stored inside the model folder at `weights/`. If not present,
it will be auto-downloaded via `mim download mmdet` when you first import `common.py`.

To download manually:
```bash
pip install openmim && mim install mmdet mmengine
mim download mmdet --config atss_swin-l-p4-w12_fpn_dyhead_ms-2x_coco \
    --dest models/experimental/atss_swin_l_dyhead/weights/
```

#### Environment variable overrides

| Variable | Description |
|---|---|
| `ATSS_CHECKPOINT` | Path to the `.pth` checkpoint file. Skips auto-download when set. |
| `ATSS_CONFIG` | Path to the mmdet config `.py` file. |

```bash
export ATSS_CHECKPOINT=/data/models/atss_swin-l.pth
export ATSS_CONFIG=/data/models/atss_swin-l-config.py
```

### Run PCC Tests

```bash
# Reference model tests (PyTorch only)
pytest models/experimental/atss_swin_l_dyhead/tests/pcc/test_reference_model.py -v

# TTNN FPN test
pytest models/experimental/atss_swin_l_dyhead/tests/pcc/test_ttnn_fpn.py -v

# TTNN ATSS Head test
pytest models/experimental/atss_swin_l_dyhead/tests/pcc/test_ttnn_atss_head.py -v

# Full E2E test
pytest models/experimental/atss_swin_l_dyhead/tests/pcc/test_ttnn_e2e.py -v
```

### Run performance tests

```bash
# Device kernel perf (tracy, single device, no trace — reports DEVICE KERNEL DURATION).
# This is the "pure compute" number used in the perf breakdown below.
pytest models/experimental/atss_swin_l_dyhead/tests/perf/test_atss_swin_l_dyhead_device_perf.py -v

# E2E single-device pipeline, 2 CQs, no trace (baseline pipeline measurement).
pytest models/experimental/atss_swin_l_dyhead/tests/perf/test_atss_swin_l_dyhead_e2e_perf.py::test_atss_swinl_dyhead_perf_single_device_2cq -v

# E2E single-device pipeline, 2 CQs + TRACE (matches what demo_slice_4dev.py runs per-device).
pytest models/experimental/atss_swin_l_dyhead/tests/perf/test_atss_swin_l_dyhead_e2e_perf.py::test_atss_swinl_dyhead_perf_single_device_trace_2cq -v

# E2E multi-device pipeline, 2 CQs, no trace.
pytest models/experimental/atss_swin_l_dyhead/tests/perf/test_atss_swin_l_dyhead_e2e_perf.py::test_atss_swinl_dyhead_perf_multi_device_2cq -v

# E2E multi-device pipeline, 2 CQs + TRACE (matches demo_slice_4dev.py end-to-end).
# Reports per-iteration host roundtrip and total-mesh FPS.
pytest models/experimental/atss_swin_l_dyhead/tests/perf/test_atss_swin_l_dyhead_e2e_perf.py::test_atss_swinl_dyhead_perf_multi_device_trace_2cq -v
```

**Trace note:** the slice demo (`demo_slice_4dev.py`) runs with **trace enabled by default**
(`use_trace=True`, plus a 400 MB trace region). That's what makes the steady-state host
roundtrip land around 172 ms instead of the ~210 ms FW duration the no-trace perf test
reports — trace replay skips per-op dispatch firmware overhead. Pass `--no-trace` to the
demo if you want to compare against the no-trace numbers.

### Demos

The inference demos require user-supplied images (no test images are bundled).
The performance benchmark and report scripts use synthetic random inputs.

The model can run on either a **single Wormhole device** (full 1280×1280 input on one
chip — the default for most demos) or on a **1×4 sub-mesh** (input sliced into 4×640×640
tiles, one per device, all running in parallel). Single-device is the simplest path;
the slicing demo is for the multi-device throughput story.

#### Single-device demos (one Wormhole chip)

```bash
# Single-image inference (--image is required). Runs the full model on one device.
python models/experimental/atss_swin_l_dyhead/demo/demo_inference.py \
    --image path/to/your/image.jpg

# Batch inference on a directory of images (--image-dir is required).
python models/experimental/atss_swin_l_dyhead/demo/demo_batch.py \
    --image-dir path/to/your/image_directory/

# Performance benchmark (no image needed — uses random tensor input).
python models/experimental/atss_swin_l_dyhead/demo/demo_perf.py

# PCC + performance report (no image needed — uses random tensor input).
python models/experimental/atss_swin_l_dyhead/demo/generate_report.py
```

Output is saved to `atss_swin_l_dyhead/results/` by default. Override with `--output-dir`.

#### Multi-device slice demo (1×4 mesh)

`demo_slice_4dev.py` runs sliced inference on a **1×4 Wormhole sub-mesh** (Galaxy or T3K).
The input image is resized to 1280×1280, sliced into a 2×2 grid of 640×640 tiles, and
all 4 tiles run **in parallel** across the 4 devices with 2 command queues + trace.
Per-tile detections are post-processed on the host and deduped across tiles into a
single 1280×1280 frame.

The defaults (`--overlap 128 --merge-mode nms --merge-match iou --merge-iou 0.55
--max-per-tile 300 --max-per-frame 500`) were tuned on dense aerial harbor scenes and
are the recommended starting point — they roughly double the visible detection count
over the previous defaults (`overlap=0`, `IoS-NMM`, per-image cap of 100) while
eliminating the wrap-around "clubbed" boxes that the IoS merger produced.

```bash
# Recommended defaults — 128 px overlap, plain NMS dedup, IoU=0.55, caps 300/500.
# Best for dense aerial/top-down scenes (harbors, parking lots, crowds from above).
python models/experimental/atss_swin_l_dyhead/demo/demo_slice_4dev.py \
    --image path/to/1280x1280_image.jpg \
    --output-dir path/to/output_dir

# Exact 2x2 (no overlap, no seam-merge) — faster (skips the pre-tile resize) but
# misses objects sitting on tile seams. Use when you know objects don't cross seams.
python models/experimental/atss_swin_l_dyhead/demo/demo_slice_4dev.py \
    --image path/to/1280x1280_image.jpg \
    --overlap 0 \
    --output-dir path/to/output_dir

# Clipped-half regime: switch to NMM so half-detections from adjacent tiles are
# unioned into the full extent. Best for street-level scenes where a single large
# object (person, bus) lands as two half-boxes from neighbouring tiles.
python models/experimental/atss_swin_l_dyhead/demo/demo_slice_4dev.py \
    --image path/to/1280x1280_image.jpg \
    --merge-mode nmm --merge-match iou --merge-iou 0.55 \
    --output-dir path/to/output_dir

# Batch over a folder of images.
for img in path/to/your/images/*.jpg; do
  name=$(basename "$img" .jpg)
  python models/experimental/atss_swin_l_dyhead/demo/demo_slice_4dev.py \
      --image "$img" \
      --output-dir path/to/output_dir/$name
done
```

**Key flags:**

| Flag | Default | Description |
|---|---|---|
| `--image` | *(required)* | Input image (auto-resized to 1280×1280 if needed). |
| `--overlap` | `128` | Tile overlap in px. `0` = exact 2×2 of 640×640 (faster but loses objects on seams). When `> 0`, image is resized to `(2*640 - overlap)` so tiles overlap by that many px in each axis. |
| `--merge-mode` | `nms` | Cross-tile dedup strategy. `nms` runs `torchvision.batched_nms` and drops duplicates (safest for dense same-class scenes — structurally cannot produce wrap-around boxes). `nmm` runs greedy non-max merging and unions overlapping detections (recovers clipped halves but at low thresholds / `ios` matching can union unrelated objects). |
| `--merge-iou` | `0.55` | IoU (or IoS, see below) threshold for the cross-tile merge. |
| `--merge-match` | `iou` | `iou` or `ios` (intersection-over-smaller). Only used by `--merge-mode nmm`. `ios` is more aggressive — useful for clipped-half merging but biases toward wrap-around boxes in dense scenes. |
| `--max-per-tile` | `300` | Per-tile post-NMS detection cap. Larger than the single-image `ATSS_MAX_PER_IMG=100` because each tile is a separate 640×640 forward and dense scenes routinely saturate the cap. |
| `--max-per-frame` | `500` | Final per-frame detection cap after cross-tile dedup. |
| `--seam-merge` | off | Run a second merge pass that knows the exact seam positions. Catches objects split across a tile seam that don't share enough IoU to merge normally. **Off by default** because in dense same-class scenes (e.g. boats moored parallel to a seam) it can reintroduce wrap-around boxes. Use when you have known large objects spanning seams and few neighbours nearby. |
| `--seam-tol` | `-1` (auto) | Abutting-edge tolerance in px for seam-merge. `-1` auto-picks `max(overlap, 20)`. |
| `--score-thr` | `0.3` | Visualization-only score threshold (boxes drawn). Post-process internally uses `ATSS_SCORE_THR=0.05`; raising it does not help recall, lowering it just draws more low-confidence boxes. |
| `--class-agnostic` | off | If set, the merger ignores class labels (NMS/NMM treats all classes as one). |
| `--no-trace` | off | Disable trace (2CQ only). Adds ~40 ms per iteration; useful for matching the no-trace perf numbers. |
| `--overlay` | off | Draw a short title bar (`atss_swin_l_dyhead \| 4 devices \| infer: XX ms`) and the four tile-seam rectangles onto the saved JPEG. By default the output is just the image with bounding boxes. |
| `--checkpoint` | auto | Override the mmdet `.pth` checkpoint path. Defaults to the auto-downloaded one in `weights/`. |
| `--output-dir` | `results/slice_4dev/` | Output directory for the annotated JPEG (`atss_slice_4dev_detections.jpg`). |

**Choosing flags by scene type:**

- **Dense aerial / top-down with many same-class objects (harbors, parking lots, crowds)**:
  *use the defaults* — `--overlap 128 --merge-mode nms --merge-match iou --merge-iou 0.55
  --max-per-tile 300 --max-per-frame 500`. The 128 px overlap recovers boats/cars that
  sit on tile seams, and plain NMS cannot fuse two distinct objects into a single
  wrap-around box. Keep `--seam-merge` **off** here.
- **Street-level with large objects spanning seams (persons, buses, single subject)**:
  add `--merge-mode nmm --merge-match iou --merge-iou 0.55` and optionally
  `--seam-merge`. NMM unions half-detections from adjacent tiles into the full extent;
  seam-merge picks up the residual cases (e.g. a person tall enough that head + body
  land in different tiles). The risk of unioning unrelated objects is low when objects
  are sparse.
- **Speed-first**: `--overlap 0` skips the input resize (and uses the input as-is at
  1280×1280) and gives marginally faster end-to-end time. Per-device compute is
  identical at all overlap settings — the only cost of overlap is the resize and a
  slightly busier CPU post-process.

#### Sweeping merge configurations

`sweep_merge_4dev.py` opens the mesh **once**, runs ATSS inference per image at the
overlap settings you ask for, then sweeps cross-tile merge configurations on CPU
only — every config gets its own annotated JPEG and a row in a JSON summary. This
is what the recommended defaults above were tuned on.

```bash
python models/experimental/atss_swin_l_dyhead/demo/sweep_merge_4dev.py \
    --images path/to/img1.jpg path/to/img2.jpg \
    --output-dir path/to/sweep_out
```

**Expected performance** on a healthy T3K 1×4 sub-mesh (measured on a dense
harbor-shot frame ~90 detections, after merging adding DCN addalpha fusion,
Swin softmax numeric_stable=False, Swin attn untilize-before-pad + ROW_MAJOR
window partition + bfp8_b attn bias, and tuned
Stage 0/1/2 fc1/fc2 program_configs):

| Metric | Time |
|---|---|
| `DEVICE KERNEL DURATION` (pure compute, per tile, parallel × 4) | ~163 ms (6.12 samples/s) |
| `DEVICE FW DURATION` (kernel + per-op dispatch overhead, no-trace pipeline) | ~210 ms (4.77 samples/s) |
| `E2E DURATION` (kernel + DMA, trace replay skips dispatch) | ~172 ms (5.81 samples/s) |

Per-device compute is invariant to `--overlap`: every tile is still a 640×640 forward,
so `overlap=0` and `overlap=128` both report ~172 ms steady-state E2E on a 4-device run
(measured median over 10 harbor shots). Raising `--max-per-tile`/`--max-per-frame`
only affects the CPU post-process, which is negligible relative to device time.

### Usage

```python
from models.experimental.atss_swin_l_dyhead.common import ATSS_CHECKPOINT
from models.experimental.atss_swin_l_dyhead.tt import TtATSSModel

model = TtATSSModel.from_checkpoint(ATSS_CHECKPOINT, device)
results = model.predict(image_tensor, img_shape=(H, W))
```
