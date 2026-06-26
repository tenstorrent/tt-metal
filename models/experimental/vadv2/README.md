## VADv2

TTNN port of **VADv2** — the probabilistic-planning successor to VAD
(*Vectorized Scene Representation for Efficient Autonomous Driving*). The model
takes multi-camera images, lifts them into a Bird's-Eye-View (BEV) feature map,
and jointly predicts object detection, vectorized map elements, agent motion,
and the ego trajectory. This port targets the **Tiny** variant.

## Supported platforms
- Wormhole (n150, n300)
- Blackhole (p150b)

## Setup

Download the pretrained weights before running any test (they are not checked
into the repo):

```
bash models/experimental/vadv2/weights_download.sh
```

This fetches `weights.zip` via `gdown`, unzips it into
`models/experimental/vadv2/`, and cleans up. The reference PyTorch dumps used
for PCC checks live in `models/experimental/vadv2/reference/dumps/`.

## How to Run

Full-model end-to-end PCC test:

```
pytest models/experimental/vadv2/tests/pcc/test_tt_vad.py
```

To reproduce the warm-wall perf measurements, enable the timing / warm-iteration
knobs the test reads from the environment:

```
TT_VISIBLE_DEVICES=0 TT_VADV2_TIMING=1 TT_VADV2_WARM_ITERS=2 \
    pytest models/experimental/vadv2/tests/pcc/test_tt_vad.py -s
```

- `TT_VADV2_WARM_ITERS=N` runs `N` extra warm forwards after the cold `call#1`. Anchor `call#3` (the second warm pass) is the production-relevant warm-wall number.
- `TT_VADV2_TIMING=1` logs the per-call wall time and the warm anchor.
- `TT_VADV2_MEMORY_REPORT=1` (optional, with `TT_VADV2_MEMORY_REPORT_PATH=/tmp/vadv2_graph.json`) captures a `ttnn.graph` report for the run.

### Per-component PCC tests

Each major module has its own test that validates the TTNN implementation
against the PyTorch reference in isolation — useful when bisecting a PCC
regression before running the full model:

| Test | Component | PCC floor |
| --- | --- | --- |
| `tests/pcc/test_tt_backbone.py` | ResNet-50 backbone + FPN neck | 0.96 |
| `tests/pcc/test_tt_temporal_self_attention.py` | Temporal Self-Attention (TSA) | 0.99 |
| `tests/pcc/test_tt_spatial_cross_attention.py` | Spatial Cross-Attention (SCA) | 0.99 |
| `tests/pcc/test_tt_transformer.py` | Perception transformer (BEV encoder + decoders) | 0.98–0.99 |
| `tests/pcc/test_tt_head.py` | `TtVADHead` (detection / map / motion / planning) | 0.98–0.99 |
| `tests/pcc/test_tt_vad.py` | Full model, all 9 output keys | per-key (see below) |

## Architecture

The forward pipeline (entry point `tt/tt_vad.py`, class `TtVAD`):

1. **Image backbone** — `TtResnet50` (`tt/tt_backbone.py`) extracts multi-scale
   features from the multi-camera input `(384, 640)`.
2. **Image neck** — `TtFPN` (`tt/tt_fpn.py`) fuses the backbone scales into a
   single feature level.
3. **Detection head** — `TtVADHead` (`tt/tt_head.py`) is the bulk of the work:
   - **Perception transformer** (`TtVADPerceptionTransformer`,
     `tt/tt_transformer.py`): the **BEV encoder** lifts image features into a
     BEV grid via **Temporal Self-Attention** (`tt/tt_temporal_self_attention.py`)
     and **Spatial Cross-Attention** (`tt/tt_spatial_cross_attention.py`), both
     built on multi-scale deformable attention (`tt/tt_deformable_attention.py`).
     The BEV runs `Q = 10000` queries, which is why the deformable-attention path
     dominates the perf profile.
   - **Detection / map branches**: object class + box predictions and vectorized
     map element predictions off the BEV features.
   - **Motion decoders** (`TtCustomTransformerDecoder` + `TtLaneNet` lane
     encoder): agent trajectory prediction over `bsz * num_heads` batched
     multi-head attention.
   - **Planning head**: produces the ego future trajectory (`ego_fut_preds`).

The nine outputs (`bev_embed`, `all_cls_scores`, `all_bbox_preds`,
`all_traj_preds`, `all_traj_cls_scores`, `map_all_cls_scores`,
`map_all_bbox_preds`, `map_all_pts_preds`, `ego_fut_preds`) are checked against
the reference dumps.

## Model details
- Model Type: VAD (Vectorized Autonomous Driving) Tiny variant
- Input Resolution: `(384, 640)` (Height, Width)
- Batch Size: 1

## Performance

Measured on Blackhole p150b with the command above.

| Metric | Value |
| --- | --- |
| Warm wall (anchor `call#3`) | **~486 ms** (down from ~2305 ms initial, −79%) |
| Cold wall (`call#1`, incl. JIT compile + first-touch) | ~7030 ms |
| PCC | all 9 output keys pass the per-key floors in `test_tt_vad.py` |

Per-key PCC floors (observed correlation in parentheses): `bev_embed` 0.96
(0.980), `all_cls_scores` 0.92 (0.956), `all_bbox_preds` 0.99 (0.999),
`all_traj_preds` 0.93 (0.960), `all_traj_cls_scores` 0.93 (0.965),
`map_all_cls_scores` 0.92 (0.960), `map_all_bbox_preds` 0.98 (0.996),
`map_all_pts_preds` 0.99 (0.998), `ego_fut_preds` 0.99 (0.999).

### Optimization journey

| Lever | Effect |
| --- | --- |
| (baseline) | warm wall ~2305 ms |
| Fused MSDA op (`num_levels==1`) | warm wall 989 → 808 ms |
| Batched MHA core-grid spread | matmul device time 47.3 → 15.5 ms (wall 808 → 771 ms) |
| `offset_normalizer` fold | warm wall −83 ms |
| Host-side cost removal (on-device untilize) | host untilize 240 → 2 ms |
| **Current** | **warm wall ~486 ms** |

### Key optimizations

- **SpatialCrossAttention `nonzero` index fix (correctness).** `ttnn.nonzero`
  now returns `[count, 4]` coordinate tuples; the model still assumed the old
  flat-index layout, silently corrupting the SCA gather/scatter. Extracting the
  final coordinate column restored `bev_embed` PCC 0.14 → 0.98 (and lifted the
  downstream keys), which had been masked by vacuous PCC floors.

- **Fused MSDA op.** The `num_levels==1` path routes through the fused
  `ttnn.experimental.multi_scale_deformable_attn` op (grid_sample + weighted
  sum in one kernel) when `N*Q` clears an amortization threshold — the largest
  single lever (warm wall 989 → 808 ms) since the BEV encoder runs `Q=10000`.
  Smaller shapes fall back to the decomposed grid_sample chain.

- **Batched MHA core-grid spread.** The motion/map decoder multi-head attention
  runs batched matmuls with a large batch (`bsz * num_heads`, up to ~14400) but
  a single query row, which ttnn's matmul heuristic collapses onto one core.
  `TtMultiheadAttention` passes an explicit `core_grid` for that regime so the
  batches spread across the full grid (`q@kᵀ` and `attn@v`: 1 → 130 cores),
  dropping warm matmul device time 47.3 → 15.5 ms with bit-identical output.

- **`offset_normalizer` fold.** The constant `1/offset_normalizer` reciprocal is
  folded into the `sampling_offsets` Linear weights at load time, eliminating a
  large per-call elementwise divide in the deformable-attention path (warm wall
  −83 ms, bit-identical).

- **Host-side cost removal.** Warm-path host syncs in the detection head were
  removed, and `bev_embed` is untilized on-device before the host transfer — its
  size-1 second-to-last dim pads to a full tile, so the *host* untilize was doing
  ~32× the work (240 → 2 ms). Combined with the MSDA value-prep layout collapse
  (one permute) these recovered the bulk of the remaining warm-wall time.

### Performance regime (device-bound)

The warm head is **device-bound**, not dispatch-bound: ~93% of the wall is the
`pts_bbox_head` forward, and a Metal Trace capture of that region — though it
captures and replays PCC-bit-identical — moves the wall only −8 ms. The
~0.24 ms/op cost is on-device (kernel launch + compute), so the levers that move
the wall are **device-kernel reduction** and **op-count / layout-churn
reduction**, not removing host dispatch.

### Known follow-ups

- **Metal Trace replay is not a lever.** A trace of `pts_bbox_head` was
  implemented and verified bit-identical, but yields only −8 ms because the head
  is device-bound (see the regime note above). Not worth pursuing further.
- **Op-count / layout-churn reduction.** Layout ops (reshape / permute /
  transpose / tilize / untilize) are ~42% of warm ops; cutting them reduces the
  on-device per-op cost ~1:1 and is the main remaining lever.
- **bf8b weight quantization on memory-bound matmuls.** Previously neutral or
  negative for compute-bound matmuls — worth a re-check now that
  `linear_flatten_batch` has changed M dimensions; some matmuls that were
  compute-bound may have become memory-bound at the new core grid.
