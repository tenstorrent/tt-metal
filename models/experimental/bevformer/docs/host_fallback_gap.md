# Host fallback ops and the remaining layer gap

Snapshot of the visualizer run
`ctr-mmicic-workspace/perf_profile_visualizer/output/2026_08_28_14_01_32`
(BEVFormer round 2, fused MSDA). The question this file answers: **how much of
the remaining op-to-op gap is the generation of TTNN host-fallback operations,
and which host-side operations actually sit on the forward path?**

Related: [PERF.md](PERF.md) (gap-column caveats), and candidate 1 in
[perf_optimization_candidates.md](perf_optimization_candidates.md) — the device
transfers landed (1a–1d), the surviving host work is inventoried in
[1g](perf_optimization_candidates.md#1g-what-is-still-host-side), and the one
item with a reason to move is the `max_len` shape,
[1e](perf_optimization_candidates.md#1e-an-empirical-high-water-mark-for-max_len).

**This file is the independent second capture of that residue.** 1g attributes
per-op gap on stage-05 runs; this is stage 04 with a different harness and a
different visualizer, and the two agree site for site (see
[the 1e/1f split](#what-1e-can-and-cannot-take-from-this)). Where they differ is
harness scope, not measurement.

## Capture

| | |
|---|---|
| report | `bevformer-round-2-msda-kernel-op` |
| tt-metal | [`2bc69b90e3c`](https://github.com/tenstorrent/tt-metal/commit/2bc69b90e3c1472e29428f40542bac47847cbf36) — stage 04, fused `MSDAOperation` |
| raw CSV | `generated/profiler/reports/2026_08_27_23_43_40/ops_perf_results_2026_08_27_23_43_40.csv` |
| window | signposts `start` → `stop`, CSV rows 441–588, **129 device ops** |
| device | N150, one encoder layer, `nuscenes_base`, BEV 100×100 |

Visualizer totals (`generation.json` → `result`):

| | µs | ms |
|---|---:|---:|
| summed device kernel | 489944 | **489.94** |
| op-to-op gaps (tt-perf-report) | 7212 | **7.21** |
| measured forward elapsed | 497156 | **497.16** |
| theory (adjusted prefill TTFT) | — | **1.11** |

`tt-perf-report` was invoked with `--no-host-ops`. That flag did not hide
anything here: the raw CSV contains **no** `python_fallback` / `tt_dnn_cpu`
rows at all, only `tt_dnn_device` (523 across the whole file, 129 in the
window) and `signpost`.

## What "host fallback" means in this stack

Three different things get called "host" and they are not interchangeable.

| Kind | How it is recorded | Present in this forward? |
|---|---|---|
| **TTNN host fallback** (`python_fallback`) | `decorate_external_operation` → Tracy zone `TT_DNN_FALLBACK_OP` → CSV `OP TYPE = python_fallback`. This is also `tt_lib.fallback_ops` (torch on host, wrap with `convert_tt_tensors_wrapper`). | **No. Zero rows, zero call sites.** |
| **Host transfer / host compute in the model** | `ttnn.to_torch` / `ttnn.from_torch` / torch index math / `ttnn.zeros` via `full_impl`. No device op is dispatched; the stall is charged as `OP TO OP LATENCY` on the *next* device op. | **Yes.** This is the remaining host cost. |
| **Device-op host dispatch** | Creating and enqueueing a real device program. Shows up as a small `OP TO OP LATENCY` plus `HOST DURATION` on the device row itself. | Yes, and it is tiny (`HOST DURATION` sums to 0.71 ms over 129 ops). |

BEVFormer never imports `tt_lib.fallback_ops` and never calls
`decorate_external_operation`. Nothing on the forward path is a TTNN host
fallback in the profiler sense.

## Answer: how much of the remaining gap is fallback generation?

**0 ms, 0%.**

The remaining gap is not produced by generating host-fallback operations,
because none are generated.

Against the two ways "remaining gap" is read:

| Reading | Amount | Of which host-fallback generation |
|---|---:|---:|
| Visualizer op-to-op gaps (tt-perf-report) | 7.21 ms | **0 ms** |
| Raw `OP TO OP LATENCY` sum over the 129 device ops | 13.59 ms | **0 ms** |
| Measured elapsed − adjusted theory (497.16 − 1.11) | 496.05 ms | **0 ms** (489.94 of this is kernel) |

The 6.38 ms difference between the raw sum and the visualizer gap is the
**region-entry** stall on the first device op after `start` (idx 0,
`CloneOperation`). tt-perf-report drops that first gap; [PERF.md](PERF.md#the-gap-column-carries-region-entry-cost)
already documents it. It is host time spent entering the signposted region, not
fallback generation.

Gap numbers on this harness do not reproduce run-to-run (14 / 93 / 151 ms on
identical stage-05 code). This file quotes one capture. The **zero fallback
rows** finding does not depend on the gap column being stable.

## Where the 13.59 ms raw gap actually sits

`OP TO OP LATENCY` is the host stall *before* the named op, never work that op
did. Eight ops hold 13.47 ms of the 13.59 ms; the other 121 hold 0.12 ms
combined.

| vis idx | CSV | gap ms | next device op | What the stall is |
|--:|--:|---:|---|---|
| 0 | 446 | **6.380** | `CloneOperation` (TSA `ttnn.clone`) | Region entry: host sync + empty command queue after `start`. Dropped by tt-perf-report. |
| 33 | 485 | **2.245** | `UnaryDeviceOperation` (`ttnn.clamp` of `reference_points_cam`) | `ttnn.to_torch(bev_mask)` readback + `torch.nonzero` / `torch.full` index construction inside `build_rebatch_plan`. |
| 39 | 491 | **1.417** | `RepeatCodegenDeviceOperation` (`ttnn.repeat` of the scatter index) | `ttnn.from_torch(query_ids)` — host tilize + DMA write. |
| 2 | 448 | 1.182 | `BinaryNgDeviceOperation` (TSA `query + query_pos`) | Device-op dispatch after the two clones. No host transfer at this site. |
| 113 | 567 | 0.867 | first permute of `ttnn.scatter_add` | `ttnn.zeros(...)` host allocation (`full_impl`) plus scatter dispatch. |
| 4 | 451 | 0.654 | `ReshapeViewDeviceOperation` (value head-split) | Device-op dispatch. |
| 3 | 450 | 0.454 | `MatmulDeviceOperation` (value projection) | Device-op dispatch. |
| 40 | 492 | **0.274** | `ReshapeViewDeviceOperation` (`unsqueeze` of `count`) | `ttnn.from_torch(count)` upload. |
| *rest* | — | 0.120 | 121 ops | Noise-level dispatch (~1 µs each). |

Of the **7.21 ms** visualizer gap (raw minus idx 0):

| Bucket | ms | % of 7.21 ms |
|---|---:|---:|
| Host transfers + torch index math in `build_rebatch_plan` (idx 33, 39, 40) | **3.94** | **54.6%** |
| `ttnn.zeros` host alloc + scatter dispatch (idx 113) | 0.87 | 12.0% |
| Ordinary device-op dispatch (idx 2, 3, 4 + tail) | 2.41 | 33.4% |
| **Generating `python_fallback` / `fallback_ops`** | **0.00** | **0%** |

The 3.94 ms host-transfer block is an artifact of the **single-layer** harness.
`rebatch_plan` arrives as `None`, so `TTSpatialCrossAttention.forward` builds it
inside the profiled window (mapping caveat, idx 33–40). The full encoder builds
the plan once above the layer loop ([tt_encoder.py](../tt/tt_encoder.py) around
the `build_rebatch_plan` call) and those four milliseconds leave the layer.

## What 1e can and cannot take from this

This capture prices the `max_len` block per site, which is exactly what
[1e](perf_optimization_candidates.md#1e-an-empirical-high-water-mark-for-max_len)
needs to be ranked. Splitting the 3.94 ms by whether a grow-only high-water mark
would remove it:

| idx | site | ms | Does 1e remove it? |
|--:|---|---:|---|
| 33 | `to_torch(bev_mask)` readback | *part of 2.245* | **Yes on steady-state frames** — the mark has not grown, so nothing is read back. |
| 33 | `torch.full` + per-camera `torch.nonzero` | *rest of 2.245* | **No.** The valid query set moves every frame with the ego-motion term; only the *shape* stabilizes. Needs `ttnn.nonzero` on top. |
| 39 | `from_torch(query_ids)` → scatter index | 1.417 | **No.** Contents change per frame. |
| 40 | `from_torch(count)` | 0.274 | **No.** Contents change per frame. |

The two sub-items at idx 33 cannot be separated without instrumenting the block —
the profiler charges both to the same stall. So **1e's measurable ceiling is
strictly under 2.245 ms and probably well under it**, because the `nonzero` loop
is the part of that stall that survives. The 1.417 + 0.274 ms of uploads are
untouched by 1e and belong to
[site 2's `ttnn.nonzero` follow-up](perf_optimization_candidates.md#1g-what-is-still-host-side).

**And this is per frame, not per layer.** [1c](perf_optimization_candidates.md#1c-hoist-index-computation-above-the-layer-loop)
hoisted `build_rebatch_plan` above the layer loop, so on a real encoder forward
the whole block runs once per frame — ~0.66 ms per layer amortized over six
layers, against 456.8 ms of layer kernel. It shows up at full size here only
because the single-layer harness gets `rebatch_plan = None` and rebuilds it
inside the window.

**Conclusion for ranking: 1e is not a gap optimization.** Its case is
[candidate 5](perf_optimization_candidates.md#candidate-5--trace-capture) — a
`rebatch_len` that is constant across replays — and this capture is the evidence
that the milliseconds will not carry it.

Cross-check against 1g, which measured the same sites on stage-05 runs with a
different attribution method:

| site | this capture (stage 04) | 1g (stage 05) |
|---|---:|---:|
| mask readback + index construction | 2.245 ms | 2.1 ms |
| scatter-index upload | 1.417 ms | 1.4 ms |

Two harnesses, two stages, same numbers. That is the part of the gap column that
*is* reproducible — unlike the totals, which move 14 / 93 / 151 ms run to run.

## Host-side operations on the forward path

None of the rows below is a TTNN host fallback. They are listed because they
are the host work that still exists, and because several of them are easy to
mistake for fallbacks (they run torch on the host and emit no device op).

### Inside the profiled layer window

Taken from the op→source mapping at `2bc69b90e3c`
(`ttnn_op_mapping.bevformer-round-2-msda-kernel-op.json`). Sites with
`ops: ""` emit no device row.

#### Runs every profiled layer (single-layer harness)

| Site | What it does | Cost in this capture |
|---|---|---|
| [`tt_spatial_cross_attention.py:73`](../tt/tt_spatial_cross_attention.py) `ttnn.to_torch(bev_mask).sum(-1) > 0` | Device→host readback of the validity mask. Forced: `rebatch_len` must be a Python int — until [1e](perf_optimization_candidates.md#1e-an-empirical-high-water-mark-for-max_len) makes the int stop changing. | Charged on idx 33 (**2.245 ms**), together with the next two rows. |
| `:102–106` `torch.full` + per-camera `torch.nonzero` | Builds padded `query_ids` on host from the read-back mask. | Same stall. |
| `:139` `ttnn.from_torch(query_ids.reshape(...))` | Uploads the scatter index (one id per row; widened on device by `ttnn.repeat`). | idx 39, **1.417 ms**. |
| `:150` `ttnn.from_torch(count)` | Uploads per-query camera-hit counts. | idx 40, **0.274 ms**. |
| `:154` `_flat_row_index(...)` | Torch index arithmetic + `from_torch` of the query-gather ids. | Host-side; folded into the idx 33–40 block. |
| `:391` `ttnn.zeros((bs, num_queries+1, embed_dims), layout=ROW_MAJOR)` | `full_impl` host allocation of the scatter target. Not a device fill. | Part of idx 113, **0.867 ms**. |
| `:366–369` `if isinstance(queries_output, torch.Tensor): from_torch(...)` | Defensive re-upload. **Not taken** — the fused op returns a `ttnn.Tensor`. | 0 |

#### Cached / skipped inside the window (no device op, no measurable stall)

| Site | Why it is free |
|---|---|
| [`tt_encoder.py:164`](../tt/tt_encoder.py) `from_torch(bev_reference_points)` | Cached on the source-tensor identity. |
| [`tt_temporal_self_attention.py:167`](../tt/tt_temporal_self_attention.py) `from_torch(level_start_index)` | Cached on the tensor contents (`[0]`). |
| [`tt_ms_deformable_attention.py`](../tt/tt_ms_deformable_attention.py) `from_torch(spatial_shapes)` → `offset_normalizer` | Cached on the feature-pyramid shapes (stage 04). Stage 05 deletes the runtime divide and folds the scale into the Linear; the remaining `from_torch(scale)` is first-call only. |

Free views (`reshape` / `unsqueeze` / one-level `split` / already-matching
`typecast` / already-TILE `to_layout`) are host metadata only. They are not
fallbacks and they do not move the gap column.

### Outside the profiled window (encoder / first frame)

The layer test's `start` is after point sampling. These run on a full encoder
forward and on the first frame; they are invisible in this report.

| Site | When | What |
|---|---|---|
| [`tt_point_sampling_3d_2d.py:58–68`](../tt/tt_point_sampling_3d_2d.py) `torch_generate_reference_points` + `from_torch` | First time a `(bev_h, bev_w, bs)` grid is seen; then cached on the encoder | CPU grid, then one upload. |
| `:113–119` `from_torch(reference_points)` / `from_torch(lidar2img)` | Per encoder forward if the caller passed torch | Uploads. `lidar2img` is per-frame input. |
| [`tt_encoder.py:445–468`](../tt/tt_encoder.py) `torch.stack(lidar2img_list)` / `.cpu()` | Per encoder forward | Host-side metadata assembly, not a ttnn op. |
| `:472–478` intrinsic @ extrinsic | Only if `lidar2img` is missing from `img_metas` | Host matmul fallback for the transform, not a TTNN fallback op. |
| `:496` `build_rebatch_plan(...)` | Once per encoder forward | The same `to_torch` / `nonzero` / `from_torch` block that the layer harness pays *inside* the window. |
| `:510` `torch.tensor([[bev_h, bev_w]])` | Every layer | Tiny host tensor; TSA then cache-hits the upload. |
| [`model_preprocessing.py`](../tt/model_preprocessing.py) weight `from_torch` | Module init only | Not on the timed forward. |

CSV rows 0–123 of this capture are the point-sampling block. They sit before
`start` and were deliberately left unmapped.

### Still not taken

| Site | Reason |
|---|---|
| `tt_lib.fallback_ops.*` | No import in `models/experimental/bevformer/`. |
| `ttnn.decorate_external_operation` | No call site. Tracy would have emitted `TT_DNN_FALL_BACK_OP` / `python_fallback` if anything had used it. |
| `queries_output` torch branch in SCA | Fused MSDA returns a device tensor. |

## What this does *not* say

- It does not say the remaining 7 ms of gap is free. About half of it (on this
  capture, in the layer harness) is the `bev_mask` readback and the index
  uploads that follow it. That work is real; it is just not fallback
  generation, and the encoder already hoists it out of the layer loop.
- It does not say that removing the readback removes that half. Only the
  readback goes; the `torch.nonzero` loop and both uploads stay, because their
  *contents* are per-frame even when the *shape* is not. See
  [the 1e/1f split](#what-1e-can-and-cannot-take-from-this).
- It does not resurrect the gap column as a stage-to-stage metric. See
  [PERF.md](PERF.md#the-gap-column-is-not-reliable).
- It does not claim the 496 ms vs-theory remainder is host time. 489.94 ms of
  that is device kernel, 36.7% of which at stage 05 is `MSDAOperation` itself.

## How to re-check

```bash
python3 - <<'PY'
import csv
from collections import Counter
p = "generated/profiler/reports/2026_08_27_23_43_40/ops_perf_results_2026_08_27_23_43_40.csv"
rows = list(csv.DictReader(open(p)))
print(Counter(r["OP TYPE"] for r in rows))
# expect: tt_dnn_device, signpost only — no python_fallback, no tt_dnn_cpu
PY
```

A future capture that actually generated host fallbacks would show
`OP TYPE = python_fallback` rows and Tracy messages starting
`` `TT_DNN_FALL_BACK_OP: ``. Those rows are what `--no-host-ops` drops. They
are absent here even before that flag.
