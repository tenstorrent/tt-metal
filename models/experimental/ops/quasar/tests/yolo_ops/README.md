# YOLOv8 per-op tests (`tests/yolo_ops/`)

Isolated tests for **every ttnn op the YOLOv8 model executes**, with the shapes the
model uses (yolov8l + yolov8s, 640×640, batch 1).

## Licensing

The YOLOv8 model **is not** in main — it lives on branch `sdawle/yolov8_bh` and can't
be merged (licensing). These tests import **no** YOLOv8 code: each calls the raw `ttnn`
op directly with shapes extracted read-only from that branch
(`models/demos/yolov8{l,s}/tt/*`), cited by `file:line` in each test's docstring. So this
suite demonstrates the ops run on Quasar without landing any model code in main.

## Running

```bash
# Full suite (real Blackhole — most feature maps are large):
pytest models/experimental/ops/quasar/tests/yolo_ops/ -v

# Emulator subset (small feature maps / tensors that fit the 2-node emulator):
pytest models/experimental/ops/quasar/tests/yolo_ops/ -m emulator -v
```

## Conventions

- Shared helpers/constants in `op_utils.py` (re-exports the generic `tests/ops/op_utils`
  helpers + YOLO dims + `nhwc_to_tt`). One file per op.
- `conftest.py` adds the `emulator` marker from each case's params: `hw` ≤ 40 (feature-map
  side) or a `shape`/`out_shape` tuple under `EMU_MAX_ELEMS`, single-device only. YOLO's
  large 320/160/80² maps are excluded (they run on Blackhole); small detect/anchor tensors,
  20/40² maps, and layout round-trips are included. Thresholds in `op_utils.py` are
  heuristics — tune on the emulator.
- Correctness: `assert_pcc` vs a torch reference where one exists (conv → `F.conv2d`,
  pool/upsample, elementwise, and value-preserving data-movement round-trips); no op is
  shape/dtype-only here.

## Status

Authored from static analysis of the branch (no device at authoring). A few shapes are
noted inline as approximate where they come from runtime config dicts not recoverable from
the source (some yolov8s neck channel widths, one neck concat width, the bottleneck residual
channel) — harmless for the channel-independent value-preserving ops. `conv2d` covers 69
distinct shapes; expect first-run touch-ups on conv/sharded cases.
