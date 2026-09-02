# Stage: 01-sca-rebatch-on-device

| | |
|---|---|
| commit | [`4048ef2bbf1`](https://github.com/tenstorrent/tt-metal/commit/4048ef2bbf14158b14d27a58911944d604b0c926) |
| candidate | [1a](../perf_optimization_candidates.md#1a-rebatch-and-scatter-back-on-device) |
| profile | **683.0 ms kernel** (+27.4) / 44.4 ms gap (−2372.1) / **727.4 ms wall** (**−2344.7 ms, −76.5%**), 146 ops (+15) |
| CSV | `generated/profiler/reports/2026_08_27_08_56_58/` |
| PCC | **0.999608** — identical to baseline |
| suites | SCA 7 passed · layer + encoder 9 passed |

> **Correction (gap only).** First published as 218.3 ms of gap; that does not reproduce.
> Re-measured 2026-08-27 on the same sources: **44.4 ms**, same 146 ops, kernel within 1.0 ms, PCC
> identical. The original CSV was deleted, so the higher figure cannot be re-audited. Most of the
> 44.4 ms is [region-entry cost](../PERF.md#the-gap-column-carries-region-entry-cost); a
> two-iteration pass puts steady-state gap at **30.9 ms** (`2026_08_27_20_58_53`).

## What changed

`query`, `reference_points_cam`, the rebatch accumulators and `slots` stop crossing the bus. What
crosses now is the row-index tensors, three orders of magnitude smaller. The
`TODO: … once TTNN supports required indexing ops` comments were stale — every op they waited on
exists in this build — and were deleted.

| Step | Was (host torch) | Now (device ttnn) |
|---|---|---|
| Rebatch queries | `query_torch[j, valid_indices]` | `ttnn.embedding(query_index, query_rows)` — one index per output **row** |
| Rebatch reference points | `ref_points_torch[i, j, valid_indices]` | `ttnn.embedding` over the `[num_cams × bs × num_queries, 8]` table |
| Scatter back | `slots_torch[j, valid] += …` | `ttnn.scatter_add(zeros, dim=1, index, src)` — repeated indices accumulate, so the multi-view sum is **one call for all six cameras** |
| Widen the scatter index | — | `ttnn.repeat` on device, rather than uploading it `embed_dims` wide |

Folding the leading dims into the row id (`camera × num_queries + row`) lets each gather run as a
single call over every batch item and camera, instead of one per camera plus a `concat` to stack.

- **`rebatch_len`** is `max_len` tile-rounded (2484 → 2496), which keeps the `(num_cams, rebatch_len)`
  merges and splits as views rather than re-layouts. The extra rows cost 0.5% more deformable
  attention and nothing else.
- **Padded slots** address a sentinel row one past `num_queries`, sliced off after the scatter, so
  whatever deformable attention computed for them is discarded without needing a particular value.

`max_len` is still read back from `bev_mask` — a data-dependent shape, not an indexing problem. That
readback and the `count` normalisation from the same host copy are what remain (candidates 1b, 1c).

## Where the time went

| region | ops | kernel | gap | Δ gap |
|---|---:|---:|---:|---:|
| SCA — deformable attention | 72 | 532.3 ms | 0.3 ms | +0.0 |
| TSA — deformable attention | 39 | 91.0 ms | 27.0 ms | −127.5 |
| SCA — rebatch / scatter-back | 11 | 44.2 ms | **8.9 ms** | **−1908.1** |
| TSA — outside MSDA | 3 | 0.2 ms | 180.0 ms | −87.7 |
| MSDA exit | 14 | 12.8 ms | 2.2 ms | −66.1 |
| FFN + rest | 7 | 1.5 ms | 0.0 ms | −8.7 |

The 1917 ms stall is gone and nothing replaced it — the largest remaining gap is ~175 ms on a `Clone`
at TSA entry. **SCA's own host gap is ~11 ms**, down from 1917; what is left belongs to TSA.

## The cost side: +26.4 ms of kernel

Paid knowingly, against 2198 ms of gap removed.

| Op | inst | kernel | note |
|---|---:|---:|---|
| Scatter | 1 | 10.50 ms | the whole scatter-back, all cameras |
| RepeatCodegen | 1 | 0.97 ms | widens the scatter index across embed_dims |
| Embeddings | 2 | **0.27 ms** | **both** rebatch gathers |
| everything else | — | ~+15 ms | `rebatch_len` 2484 → 2496 rippling through MSDA, the sentinel slice, extra layout conversions |

Two gathers at 0.27 ms combined is the headline: the rebatch that serialized the pipeline for two
seconds is now the cheapest thing in the layer. Full op table:

| Op | inst | ms | | Op | inst | ms |
|---|---:|---:|---|---|---:|---:|
| ReshapeView | 23 | 156.90 | | Scatter | 1 | 10.50 |
| GridSample | 5 | 116.19 | | Matmul | 11 | 4.68 |
| Concat | 3 | 115.33 | | Reduce | 2 | 4.47 |
| Permute | 27 | 105.50 | | FillPad | 2 | 4.36 |
| BinaryNg | 18 | 86.42 | | Unary | 2 | 1.56 |
| Slice | 13 | 29.44 | | Softmax | 2 | 1.22 |
| UntilizeWithUnpadding | 18 | 28.95 | | RepeatCodegen | 1 | 0.97 |
| TilizeWithValPadding | 8 | 14.58 | | Embeddings / LayerNorm / Transpose / Clone / UntilizeCodegen | 9 | 0.85 |

## `ttnn.gather` was the wrong op — 97.59 ms vs 0.12 ms

The reference-point rebatch was first written with `ttnn.gather`, index materialized across the
`num_depth_levels × 2` columns: **97.59 ms in one call**, the fifth most expensive op in the layer.
`gather` transposes the gather dimension to last internally, and at `[6, 1, 10000, 8]` that transpose
is the entire cost. `embedding` with folded row ids does the same work in **0.12 ms**. When the
gather is over whole rows, `embedding` is the op. Full entry:
[DEAD_ENDS 2](DEAD_ENDS.md#2-ttnngather-for-the-reference-point-rebatch).

## What review changed

Five reviewers ran against the first working version:

- **Pad-row aliasing.** Padded slots originally carried index 0, neutralised by a mask multiply — so
  correctness depended on that multiply producing exact zero, and one non-finite value out of
  deformable attention would have poisoned query 0 for the whole batch item, invisibly to an
  aggregate PCC gate. The sentinel row removes the dependency and deletes the mask op.
- **A camera that sees nothing.** The old host loop guarded `len(valid_indices) > 0` and skipped the
  camera; the device path always runs it. No test reached that case — SCA tests build `bev_mask` by
  uniform random masking, and the layer/encoder tests use a rig where all six cameras see the grid.
  Two params now cover it, one with only a single batch item's camera blank.
- **A reference-point reshape in the wrong layout**, splitting `8 → (4, 2)` on a tiled tensor: a
  re-layout, not a view, padding those 8 columns to a full tile on the way.
- **Unasserted dtype preconditions.** `embedding` needs a bfloat16 table and `scatter_add` needs
  accumulator and source to agree, so the path is pinned to bfloat16 where the host round-trip was
  dtype-agnostic. Asserted at the boundary instead of failing as a TT_FATAL inside a device op.

## What this changes about the plan

Kernel is **94% of wall clock** (683.0 of 727.4 ms) and 623 ms of it is the two MSDA calls. The host
side is no longer the story; candidates 2/3/4 are, and the remaining gap sits in TSA, not SCA.
`200×200` stays deselected for a pre-existing MSDA OOM, confirmed identical on the baseline commit.
