# Stage: 01-sca-rebatch-on-device

- source commit: [`4048ef2bbf1`](https://github.com/tenstorrent/tt-metal/commit/4048ef2bbf14158b14d27a58911944d604b0c926)
- harness: `tests/perf/test_layer_perf.py`, one encoder layer, N150
- kernel time: **683.0 ms** (+27.4 ms)
- op-to-op gap: **44.4 ms** (−2372.1 ms) — corrected, see below
- wall: **727.4 ms** (−2344.7 ms, **−76.5%**)
- device ops in the signposted region: **146** (+15)
- PCC gate: **0.999608** — identical to the baseline's 0.999608
- CSV: `generated/profiler/reports/2026_08_27_08_56_58/`

> **Correction (gap only).** This report first published 218.3 ms of gap. That does not reproduce.
> Re-measured 2026-08-27 on the same `tt/` sources: **44.4 ms**, same 146 device ops, kernel within
> 1.0 ms, PCC identical. The original CSV has been deleted, so the higher figure cannot be
> re-audited. Header numbers above are the corrected ones.
>
> Most of the 44.4 ms is [region-entry cost](../PERF.md#the-gap-column-carries-region-entry-cost),
> not per-op latency. A two-iteration pass on this tree puts steady-state gap at **30.9 ms**
> (`2026_08_27_20_58_53`).

## What this change was

**The SCA rebatch and scatter-back run on device.** `query`, `reference_points_cam`, the rebatch
accumulators and `slots` no longer cross the bus; what crosses now is the row-index tensors, which
are three orders of magnitude smaller.

The `TODO: Currently done on CPU, to be modified once TTNN supports required indexing ops` comments
were stale — every op they waited on exists in this build. Both are deleted rather than edited.

| Step | Before — host (torch) | After — device (ttnn) | What the ttnn op does |
|---|---|---|---|
| Rebatch the queries | `query_torch[j, valid_indices]` | `ttnn.embedding(query_index, query_rows)` | Table lookup: one index per output **row**, output row `k` is table row `index[k]`. Exactly a row gather. |
| Rebatch the reference points | `ref_points_torch[i, j, valid_indices]` | `ttnn.embedding(ref_index, ref_rows)` | Same, over the `[num_cams × bs × num_queries, 8]` table. |
| Scatter the results back | `slots_torch[j, valid] += queries_output_torch[j, i, :n]` | `ttnn.scatter_add(zeros, dim=1, index, src)` | Adds `src` rows into `input` at `index`, **accumulating repeated indices** — which is the multi-view sum, so all six cameras land in one call. |
| Widen the scatter index | — | `ttnn.repeat` | `scatter_add` indexes per element, so the row id is broadcast across `embed_dims` on device rather than uploaded that wide. |

Both rebatches are row gathers, and `embedding` takes one index per row rather than one per element,
so the index stays small. Folding the leading dimensions into the row id
(`camera × num_queries + row`) lets each run as a **single call** covering every batch item and
camera, instead of one call per camera plus a `concat` to stack them.

- **`rebatch_len`** is `max_len` rounded up to a tile boundary (2484 → 2496), which keeps the
  `(num_cams, rebatch_len)` merges and splits as views on a tiled tensor instead of re-layouts. The
  extra rows cost 0.5% more deformable-attention compute and nothing else.
- **Padded slots** address a sentinel row one past `num_queries`, which is sliced off after the
  scatter. Whatever deformable attention computed for them is discarded without their value having
  to be anything in particular.

`max_len` is still read back from `bev_mask` — it is a data-dependent shape, not an indexing
problem. That readback, and the `count` normalisation computed from the same host copy, are what
remain; see candidates 1b and 1c.

## Where the time went

| region | ops | kernel | gap | Δ gap |
|---|---:|---:|---:|---:|
| SCA — deformable attention | 72 | 532.3 ms | 0.3 ms | +0.0 |
| TSA — deformable attention | 39 | 91.0 ms | 27.0 ms | −127.5 |
| SCA — rebatch / scatter-back | 11 | 44.2 ms | **8.9 ms** | **−1908.1** |
| TSA — forward, outside MSDA | 3 | 0.2 ms | 180.0 ms | −87.7 |
| MSDA exit | 14 | 12.8 ms | 2.2 ms | −66.1 |
| FFN | 5 | 1.3 ms | 0.0 ms | −7.6 |
| rest | 2 | 0.2 ms | 0.0 ms | −1.1 |

The 1917 ms stall is gone and nothing replaced it: the largest remaining gap in the layer is ~175 ms,
on a `Clone` at TSA entry. **SCA's own host gap is now ~11 ms**, down from 1917. What is left of the
layer's gap belongs to TSA, not SCA.

## The cost side: +26.4 ms of kernel

Paid knowingly, against 2198 ms of gap removed:

| Op | inst | kernel | note |
|---|---:|---:|---|
| ScatterDeviceOperation | 1 | 10.50 ms | the whole scatter-back, all cameras |
| RepeatCodegenDeviceOperation | 1 | 0.97 ms | widens the scatter index across embed_dims on device |
| EmbeddingsDeviceOperation | 2 | 0.27 ms | **both** rebatch gathers |
| everything else | — | ~+15 ms | `rebatch_len` 2484 → 2496 rippling through MSDA, the sentinel slice, and the extra layout conversions |

The two gathers costing 0.27 ms combined is the headline: the rebatch that used to serialize the
pipeline for nearly two seconds is now the cheapest thing in the layer.

### A rejected first attempt worth recording

The reference-point rebatch was first written with `ttnn.gather` — the general op, index materialized
across the `num_depth_levels × 2` columns. It measured **97.59 ms in a single call**, making it the
fifth most expensive op in the layer. `gather` transposes the gather dimension to last internally,
and at `[6, 1, 10000, 8]` that transpose is the entire cost.

Switching it to `embedding` with folded row ids took the same work from 97.59 ms to 0.12 ms. When a
gather is over whole rows, `embedding` is the op; `gather` is for when it genuinely is not.

## Kernel time by op code

| Op | inst | ms |
|---|---:|---:|
| ReshapeViewDeviceOperation | 23 | 156.90 |
| GridSampleOperation | 5 | 116.19 |
| ConcatDeviceOperation | 3 | 115.33 |
| PermuteDeviceOperation | 27 | 105.50 |
| BinaryNgDeviceOperation | 18 | 86.42 |
| SliceDeviceOperation | 13 | 29.44 |
| UntilizeWithUnpaddingDeviceOperation | 18 | 28.95 |
| TilizeWithValPaddingDeviceOperation | 8 | 14.58 |
| ScatterDeviceOperation | 1 | 10.50 |
| MatmulDeviceOperation | 11 | 4.68 |
| ReduceDeviceOperation | 2 | 4.47 |
| FillPadDeviceOperation | 2 | 4.36 |
| UnaryDeviceOperation | 2 | 1.56 |
| SoftmaxDeviceOperation | 2 | 1.22 |
| RepeatCodegenDeviceOperation | 1 | 0.97 |
| EmbeddingsDeviceOperation | 2 | 0.27 |
| LayerNormDeviceOperation | 3 | 0.26 |
| TransposeDeviceOperation | 2 | 0.14 |
| CloneOperation | 2 | 0.10 |
| UntilizeCodegenDeviceOperation | 1 | 0.08 |

## What this changes about the plan

Kernel is now **94% of wall clock** (683.0 of 727.4 ms) and 623 ms of it is the two MSDA calls. The
host side is no longer the story: candidates 2, 3 and 4 are, and the remaining gap is concentrated
in TSA rather than SCA.

## Correctness

- SCA PCC suite: **7 passed** (including `bs=2` and two new blank-camera params). `200×200` is
  deselected for a pre-existing MSDA OOM, confirmed identical on the baseline commit.
- Layer + encoder PCC suites: **9 passed**, including the full 6-layer encoder across all configs.
- Perf-harness PCC gate: 0.999608, unchanged from baseline.

### What review changed

Five reviewers ran against the first working version. What they moved:

- **The pad-row aliasing.** Padded slots originally carried index 0 and were neutralised by a mask
  multiply, which made correctness depend on that multiply producing exact zero — one non-finite
  value out of deformable attention would have poisoned query 0 for the whole batch item, and no
  aggregate PCC gate would have caught a single wrong row out of 10000. The sentinel row removes the
  dependency instead of testing around it, and deletes the mask op.
- **A camera that sees nothing.** The old host loop guarded `len(valid_indices) > 0` and skipped the
  camera; the device path always runs it. No existing test reached that case — the SCA tests build
  `bev_mask` by uniform random masking, which leaves every camera populated, and the layer/encoder
  tests use a rig where all six cameras see the grid. Two params now cover it, including one where
  only one batch item's camera is blank.
- **A reference-point reshape done in the wrong layout**, splitting `8 → (4, 2)` on a tiled tensor,
  which is a re-layout rather than a view and pads those 8 columns to a full tile on the way.
- **Unasserted dtype preconditions.** `embedding` requires a bfloat16 table and `scatter_add`
  requires accumulator and source to agree, so the rebatch path is pinned to bfloat16 where the old
  host round-trip was dtype-agnostic. Asserted at the boundary rather than left to fail as a
  TT_FATAL inside a device op.
