# Host fallback ops and the remaining layer gap

**The question: how much of the remaining op-to-op gap is TTNN host-fallback generation? Answer:
0 ms, 0%. None are generated.** BEVFormer never imports `tt_lib.fallback_ops` and never calls
`decorate_external_operation`, and the raw CSV contains zero `python_fallback` / `tt_dnn_cpu` rows —
only `tt_dnn_device` (523 in the file, 129 in the window) and `signpost`.

This is the **independent second capture** of the host residue that
[1g](perf_optimization_candidates.md#1g-what-is-still-host-side) inventories on stage-05 runs. This
one is stage 04, a different harness and a different visualizer, and the two agree site for site.
Where they differ is harness scope, not measurement.

| | |
|---|---|
| report | `bevformer-round-2-msda-kernel-op`, visualizer output `2026_08_28_14_01_32` |
| tt-metal | [`2bc69b90e3c`](https://github.com/tenstorrent/tt-metal/commit/2bc69b90e3c1472e29428f40542bac47847cbf36) — stage 04, fused `MSDAOperation` |
| raw CSV | `generated/profiler/reports/2026_08_27_23_43_40/` |
| window | signposts `start` → `stop`, CSV rows 441–588, **129 device ops** |
| device | N150, one encoder layer, `nuscenes_base`, BEV 100×100 |
| totals | kernel **489.94 ms** · visualizer gap **7.21 ms** · elapsed 497.16 ms · `HOST DURATION` sum 0.71 ms |

## Three different things get called "host"

| Kind | How it is recorded | Present? |
|---|---|---|
| **TTNN host fallback** (`python_fallback`) | `decorate_external_operation` → Tracy `TT_DNN_FALLBACK_OP` → CSV `OP TYPE = python_fallback`; also `tt_lib.fallback_ops` | **No. Zero rows, zero call sites.** |
| **Host transfer / host compute in the model** | `ttnn.to_torch` / `from_torch` / torch index math / `ttnn.zeros` via `full_impl`. No device op dispatched; the stall is charged as `OP TO OP LATENCY` on the *next* device op. | **Yes** — this is the remaining host cost |
| **Device-op host dispatch** | Creating and enqueueing a real program. Small `OP TO OP LATENCY` plus `HOST DURATION` on the row itself. | Yes, and tiny (0.71 ms over 129 ops) |

Gap totals on this harness do not reproduce run to run (14 / 93 / 151 ms on identical stage-05 code,
[PERF.md](PERF.md#the-gap-column-is-not-reliable)). This file quotes one capture; the **zero
fallback rows** finding does not depend on the gap column being stable.

## Where the 13.59 ms of raw gap sits

`OP TO OP LATENCY` is the host stall *before* the named op, never work that op did. Eight ops hold
13.47 ms; the other 121 hold 0.12 ms combined.

| idx | gap ms | next device op | what the stall is |
|--:|---:|---|---|
| 0 | **6.380** | `Clone` (TSA `ttnn.clone`) | Region entry: host sync + empty queue after `start`. Dropped by tt-perf-report, which is the 6.38 ms difference between 13.59 and 7.21. |
| 33 | **2.245** | `Unary` (`clamp` of `reference_points_cam`) | `to_torch(bev_mask)` readback + `torch.nonzero` / `torch.full` index construction in `build_rebatch_plan` |
| 39 | **1.417** | `RepeatCodegen` (scatter index) | `from_torch(query_ids)` — host tilize + DMA write |
| 2 | 1.182 | `BinaryNg` (`query + query_pos`) | device-op dispatch, no host transfer |
| 113 | 0.867 | first permute of `scatter_add` | `ttnn.zeros(...)` host allocation (`full_impl`) + scatter dispatch |
| 4, 3 | 0.654, 0.454 | `ReshapeView`, `Matmul` | device-op dispatch |
| 40 | **0.274** | `ReshapeView` (`unsqueeze` of `count`) | `from_torch(count)` upload |
| rest | 0.120 | 121 ops | ~1 µs each |

Of the 7.21 ms visualizer gap: **3.94 ms (54.6%)** is the host-transfer block in
`build_rebatch_plan` (idx 33/39/40), 0.87 ms (12.0%) is the `zeros` alloc, 2.41 ms (33.4%) is
ordinary dispatch, and **0.00 ms is fallback generation**.

**The 3.94 ms is a single-layer harness artifact.** `rebatch_plan` arrives as `None`, so
`TTSpatialCrossAttention.forward` builds it inside the profiled window. The encoder builds it once
above the layer loop ([1c](perf_optimization_candidates.md#1c-hoist-index-computation-above-the-layer-loop)),
so those four milliseconds leave the layer — ~0.66 ms per layer amortized over six, against 456.8 ms
of layer kernel.

## What 1e can and cannot take from this

| idx | site | ms | does [1e](perf_optimization_candidates.md#1e-an-empirical-high-water-mark-for-max_len) remove it? |
|--:|---|---:|---|
| 33 | `to_torch(bev_mask)` readback | *part of 2.245* | **Yes on steady-state frames** — the mark has not grown, nothing is read back |
| 33 | `torch.full` + per-camera `torch.nonzero` | *rest of 2.245* | **No.** The valid query set moves every frame with ego motion; only the *shape* stabilizes. Needs `ttnn.nonzero` on top. |
| 39 | `from_torch(query_ids)` | 1.417 | **No** — contents change per frame |
| 40 | `from_torch(count)` | 0.274 | **No** — contents change per frame |

The two sub-items at idx 33 cannot be separated without instrumenting the block; the profiler charges
both to the same stall. So **1e's measurable ceiling is strictly under 2.245 ms and probably well
under it**, and it is per frame, not per layer.

**Conclusion for ranking: 1e is not a gap optimization.** Its case is
[candidate 9](perf_optimization_candidates.md#candidate-9--trace-capture) — a `rebatch_len` constant
across replays — and this capture is the evidence that the milliseconds will not carry it.

Cross-check against 1g, same sites, different stage and attribution method: mask readback + index
construction **2.245 ms here vs 2.1 ms**, scatter-index upload **1.417 vs 1.4**. Two harnesses, two
stages, same numbers — that is the part of the gap column that *is* reproducible.

## Host-side sites on the forward path

None is a TTNN host fallback. They are listed because several are easy to mistake for one: they run
torch on the host and emit no device row. Mapping from `ttnn_op_mapping.…json` at `2bc69b90e3c`.

**Inside the window, every profiled layer** — all in [`tt_spatial_cross_attention.py`](../tt/tt_spatial_cross_attention.py):

| line | what | cost |
|---|---|---|
| `:73` | `to_torch(bev_mask).sum(-1) > 0` — forced, `rebatch_len` must be a Python int | idx 33, **2.245 ms**, with the two below |
| `:102–106` | `torch.full` + per-camera `torch.nonzero` build padded `query_ids` | same stall |
| `:139` | `from_torch(query_ids)` — scatter index, widened on device by `repeat` | idx 39, **1.417 ms** |
| `:150` | `from_torch(count)` — per-query camera-hit counts | idx 40, **0.274 ms** |
| `:154` | `_flat_row_index(...)` torch arithmetic + upload | folded into idx 33–40 |
| `:391` | `ttnn.zeros(..., ROW_MAJOR)` — `full_impl` host alloc, not a device fill | part of idx 113, **0.867 ms** |
| `:366–369` | defensive `from_torch` if the inner call returned torch — **not taken**, the fused op returns a device tensor | 0 |

**Cached or skipped inside the window** (no device op, no measurable stall): `bev_reference_points`
(cached on source-tensor identity), `level_start_index` (cached on contents), `offset_normalizer`
(cached on pyramid shapes at stage 04; stage 05 deletes the divide and the remaining `from_torch` is
first-call only). Free views — `reshape` / `unsqueeze` / one-level `split` / matching `typecast` /
already-TILE `to_layout` — are host metadata only.

**Outside the window** (before `start`, or encoder-level / first-frame only): reference-point grid
generation and upload in [`tt_point_sampling_3d_2d.py:58–68`](../tt/tt_point_sampling_3d_2d.py) (then
cached per `(bev_h, bev_w, bs)`), `from_torch(reference_points)` / `from_torch(lidar2img)` at
`:113–119`, `torch.stack(lidar2img_list).cpu()` in
[`tt_encoder.py:445–468`](../tt/tt_encoder.py), the intrinsic @ extrinsic host matmul at `:472–478`
(only when `lidar2img` is absent from `img_metas`), `build_rebatch_plan` at `:496` (once per encoder
forward), `torch.tensor([[bev_h, bev_w]])` at `:510`, and weight uploads in
[`model_preprocessing.py`](../tt/model_preprocessing.py) (module init). CSV rows 0–123 are the
point-sampling block and were deliberately left unmapped.

## What this does not say

- **Not that the remaining gap is free.** About half of it in this harness is the `bev_mask` readback
  and the uploads after it. That work is real — it is just not fallback generation, and the encoder
  already hoists it out of the layer loop.
- **Not that removing the readback removes that half.** Only the readback goes; the `nonzero` loop and
  both uploads stay, because their *contents* are per-frame even when the *shape* is not.
- **Not a resurrection of the gap column** as a stage-to-stage metric.
- **Not that the 496 ms vs-theory remainder is host time** — 489.94 ms of it is device kernel.

## How to re-check

```bash
python3 -c "
import csv, collections
p='generated/profiler/reports/2026_08_27_23_43_40/ops_perf_results_2026_08_27_23_43_40.csv'
print(collections.Counter(r['OP TYPE'] for r in csv.DictReader(open(p))))"
# expect tt_dnn_device and signpost only — no python_fallback, no tt_dnn_cpu
```

A capture that did generate host fallbacks would show `OP TYPE = python_fallback` rows and Tracy
messages starting `` `TT_DNN_FALL_BACK_OP: ``. Those are what `--no-host-ops` drops; they are absent
here even before the flag.
