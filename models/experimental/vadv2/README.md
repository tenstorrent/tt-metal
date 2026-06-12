## VADV2

## Platforms:
    Wormhole (n150, n300)
    Blackhole (p150b)

## How to Run
- Use the following command to run the vadv2 test:
```
pytest models/experimental/vadv2/tests/pcc/test_tt_vad.py
```

- To reproduce the warm-wall perf measurements, enable the timing /
  warm-iteration knobs the test reads from the environment:

```
TT_VISIBLE_DEVICES=0 TT_VADV2_TIMING=1 TT_VADV2_WARM_ITERS=2 \
    pytest models/experimental/vadv2/tests/pcc/test_tt_vad.py -s
```

  - `TT_VADV2_WARM_ITERS=N` runs `N` extra warm forwards after the cold
    `call#1`. Anchor `call#3` (the second warm pass) is the
    production-relevant warm-wall number.
  - `TT_VADV2_TIMING=1` logs the per-call wall time and the warm anchor.
  - `TT_VADV2_MEMORY_REPORT=1` (optional, with
    `TT_VADV2_MEMORY_REPORT_PATH=/tmp/vadv2_graph.json`) captures a
    `ttnn.graph` report for the run.

## Details
- The entry point to vadv2 model is in `models/experimental/vadv2/tt/tt_vad.py`.
- Model Type: VAD (Video-based Autonomous Driving) Tiny variant
- Input Resolution - (384,640) (Height,Width)
- Batch Size : 1
- Inference steps for both GPU and CPU : [https://docs.google.com/document/d/1mcqm_TXuZpPpvtnT19BNeKqP-ilQfGqSBGcEF_X9onk/edit?usp=sharing]
- GPU and CPU evaluation metrics on nuscenes mini dataset are here : [https://drive.google.com/file/d/1p5ESawe79n4SPgt3ZCPO4fOQ4sxVufhU/view?usp=sharing]

## Performance

Measured on Blackhole p150b with the command above.

- Warm wall (anchor `call#3`): **~771 ms** (sub-1-second).
- Cold wall (`call#1`, includes JIT compile + first-touch): ~7030 ms.
- PCC: all 9 output keys (`bev_embed`, `all_cls_scores`, `all_bbox_preds`,
  `all_traj_preds`, `all_traj_cls_scores`, `map_all_cls_scores`,
  `map_all_bbox_preds`, `map_all_pts_preds`, `ego_fut_preds`) pass the
  per-key floors pinned in `test_tt_vad.py`.

The `num_levels==1` MSDA path routes through the fused
`ttnn.experimental.multi_scale_deformable_attn` op (grid_sample + weighted
sum in one kernel) when `N*Q` clears an amortization threshold; this was the
last large lever (warm wall 989 → 808 ms) since the BEV encoder runs
`Q=10000`. Smaller shapes fall back to the decomposed grid_sample chain.

The motion/map decoder multi-head attention runs batched matmuls with a
large batch (`bsz * num_heads`, up to ~14400) but a single query row, which
ttnn's matmul heuristic collapses onto one core. `TtMultiheadAttention`
passes an explicit `core_grid` for that regime so the batches spread across
the full grid (`q@kᵀ` and `attn@v`: 1 → 130 cores), dropping warm matmul
device time 47.3 → 15.5 ms and warm wall 808 → 771 ms with bit-identical
output.

### Known follow-ups

- **Metal Trace replay.** Every model-side blocker is cleared
  (persistent buffers for `shift`/`can_bus`, static zeros caches for
  `bev_mask`/`bev_pos`/`level_start_index`/`slots`/`sentinel_row`), but
  `ttnn.embedding` itself does internal host→device writes on every
  call (TT-Metal layer; FATAL inside trace capture at
  `fd_mesh_command_queue.cpp:595`). Needs an upstream fix.
- **bf8b weight quantization on memory-bound matmuls.** Previously
  neutral or negative for compute-bound matmuls — worth a re-check now
  that `linear_flatten_batch` has changed M dimensions; some matmuls
  that were compute-bound may have become memory-bound at the new core
  grid.
