# BEVFormer encoder — performance optimization candidates

Status: draft list of candidates, no measurements attached yet. Baseline profiling
(`models/experimental/bevformer/tests/test_encoder_profile.py` + `tt:profiler`) still has to
be run before any of these are ranked by actual device/host time.

## Candidate 1 — host round-trips (`ttnn.to_torch` / `ttnn.from_torch`) in the encoder

Every `to_torch` is a device→host readback that serializes the whole pipeline: it flushes the
op queue, blocks on the readback, and kills any chance of tracing the encoder. The matching
`from_torch` pushes back down. In BEVFormer these are not one-off setup costs — several sit
inside per-layer, per-camera loops.

### 1a. Spatial cross-attention rebatching loop — the worst offender

[tt_spatial_cross_attention.py:206-233](../tt/tt_spatial_cross_attention.py#L206-L233)

Inside the `bs × num_cams` loop, per iteration:

- `ttnn.to_torch(index_query_per_img[j])` (L208)
- `ttnn.to_torch(query)` and `ttnn.to_torch(reference_points_cam)` (L217-218) — re-read of the
*same* tensors on every camera
- `ttnn.to_torch(queries_rebatch)` / `ttnn.to_torch(reference_points_rebatch)` (L220-221) —
read back the accumulator that was just written
- `ttnn.from_torch(...)` ×2 (L228-233) — full re-upload of both accumulators

So for 6 cameras that is ~36 host transfers of tensors that never had to leave host memory in
the first place. Steps:

1. Cheap fix, no new ops: hoist the invariant `to_torch(query)` / `to_torch(reference_points_cam)`
  out of the loop, build `queries_rebatch_torch` / `ref_rebatch_torch` entirely on host, and do a
   single `from_torch` after the loop. Removes ~30 transfers with zero semantic change.
2. Real fix: keep it on device — see below.

#### On the `TODO: Currently done on CPU, to be modified once TTNN supports required indexing ops`

That comment ([L205](../tt/tt_spatial_cross_attention.py#L205), and the twin at
[L298](../tt/tt_spatial_cross_attention.py#L298)) is stale. TTNN has the indexing ops now; what is
still missing is something else, and the distinction changes how the rewrite is scoped.

**What exists today:**

| Need | Op | Constraints |
|---|---|---|
| Row gather `query[j, valid_indices]` | `ttnn.embedding(indices, weight)` | `weight` ROW_MAJOR bfloat16 with leading dims 1 (i.e. `[1, 1, num_queries, embed_dims]`); `indices` UINT32 or BFLOAT16; both INTERLEAVED. Output `[..., max_len, embed_dims]`. Natural fit — the gather is over whole rows. |
| Generic gather along a dim | `ttnn.gather(input, dim, index)` | Index must have the **same rank** as input and the output takes the index's shape, so the index has to be materialized at `[bs, max_len, embed_dims]` (row id broadcast across `E`). TILE, UINT16/UINT32 index, interleaved. Works, but wastes bandwidth vs. `embedding`. |
| Scatter-back accumulation (1c) | `ttnn.scatter_add(input, dim, index, src)` | Interleaved, on device. Same index-materialization caveat as `gather`. |
| Mask → indices | `ttnn.nonzero(input)` | **1D, ROW_MAJOR only.** Returns `(count, indices)`, `indices` padded to input length, only the first `count` entries meaningful. |

So `queries_rebatch` / `reference_points_rebatch` can be built with `embedding`, and the L302-311
scatter-back with `scatter_add`. There is no missing op there.

**What is actually blocking:**

1. **`max_len` is a data-dependent shape.** `ttnn.nonzero` returns `count` as a *device* tensor;
   `max_len = max over cameras of count` has to be a Python int to size `queries_rebatch`. Reading
   it back is a host sync — exactly what we are trying to remove. This is a dynamic-shape gap, not
   an indexing-op gap, and TTNN is not going to close it for us. The fix is on our side: **bound
   `max_len` statically.** Worst case is `num_queries`; a calibrated per-config bound is better.
   Padding rows are already zeroed and already contribute nothing (the scatter-back only touches
   `valid_indices`), so an over-large `max_len` costs wasted MSDA compute, not correctness. Worth
   it — a static `max_len` is what makes the encoder trace-capturable.
2. **`nonzero` is 1D + ROW_MAJOR**, so producing the index tensors means one call per camera on a
   `[num_queries]` mask, plus a layout conversion, plus assembling the padded `[bs, max_len]` index
   tensor. Mechanical, but this is where the rewrite effort actually goes.
3. **The index set is frame-dependent but not layer-dependent.** `bev_mask` comes from
   `point_sampling_3d_to_2d`, i.e. it depends on `lidar2img` — it changes per frame, but it is
   identical across all encoder layers within a frame. Compute the index tensors **once per
   forward, above the layer loop**, and pass them down. Right now they are recomputed inside every
   layer's SCA. That alone divides the cost by `num_layers`, independent of whether anything moves
   to device.

Recommended order: (3) first — pure refactor, no op risk, immediate win. Then (1), the decision
that unblocks everything else. Then (2) with `embedding` + `scatter_add`.

When this lands, delete both TODO comments rather than editing them — they describe a constraint
that no longer holds.



### 1b. `bev_mask` readback and validity/count computation

[tt_spatial_cross_attention.py:155](../tt/tt_spatial_cross_attention.py#L155),
[:299-328](../tt/tt_spatial_cross_attention.py#L299-L328)

- `bev_mask_torch = ttnn.to_torch(bev_mask)` (L155) drives `indexes`, `max_len`, and later `count`.
- `count` (L322-327) is `sum(-1) > 0 → permute → sum(-1) → clamp` — pure elementwise/reduce work
that maps 1:1 onto ttnn ops. No reason for it to be on host.

`max_len` is the genuine blocker: it is a data-dependent shape, so it forces a host sync. Options:
(a) accept one sync per frame and move everything else on device; (b) bound `max_len` statically
(worst case = `num_queries`, or a calibrated upper bound) and drop the sync entirely, trading some
wasted compute for a fully-device path. (b) is what unlocks trace capture.

### 1c. Scatter-back / aggregation loop

[tt_spatial_cross_attention.py:299-315](../tt/tt_spatial_cross_attention.py#L299-L315)

`to_torch(slots)`, `to_torch(queries_output)`, per-camera `to_torch(index...)`, host `+=`
accumulation, one `from_torch`. Same shape of problem as 1a, inverted: a scatter-add over the
same index sets. Solvable with the same index tensors — `ttnn.scatter_add` / masked
`ttnn.where` + `ttnn.sum` over the camera dim.

Note 1a and 1c share the index computation. Fixing 1b first makes both cheap.

### 1d. Per-call constant uploads

These are small, but they run every layer / every forward:


| Location                                                                         | Tensor                                                  | Fix                                                                                                                           |
| -------------------------------------------------------------------------------- | ------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| [tt_encoder.py:157](../tt/tt_encoder.py#L157)                                    | `bev_reference_points`                                  | Precompute once at module init, cache as a device tensor. It only depends on `reference_points_3d`, which is frame-invariant. |
| [tt_ms_deformable_attention.py:283](../tt/tt_ms_deformable_attention.py#L283)    | `spatial_shapes_tt` (→ `offset_normalizer`)             | Config-derived, identical every call. Cache the `offset_normalizer` itself, not just `spatial_shapes`.                        |
| [tt_point_sampling_3d_2d.py:69](../tt/tt_point_sampling_3d_2d.py#L69)            | reference points from `torch_generate_reference_points` | Already marked `TODO: Calculate during initialization`. Do it.                                                                |
| [tt_point_sampling_3d_2d.py:115-120](../tt/tt_point_sampling_3d_2d.py#L115-L120) | `reference_points`, `lidar2img`                         | `lidar2img` is per-frame input, unavoidable; the rest is cacheable.                                                           |
| [tt_temporal_self_attention.py:163](../tt/tt_temporal_self_attention.py#L163)    | `level_start_index`                                     | Constant `[0]` in the default path. Build once.                                                                               |
| [tt_spatial_cross_attention.py:245](../tt/tt_spatial_cross_attention.py#L245)    | `to_torch(spatial_shapes)`                              | Only used for an assert. Guard it behind a debug flag or keep a host-side copy.                                               |


Weight preprocessing in [model_preprocessing.py](../tt/model_preprocessing.py) is one-time setup —
not a candidate.

## Candidate 2 — replace the hand-rolled MSDA with `ttnn.experimental.multi_scale_deformable_attn`

A fused multi-scale deformable attention op has landed
([PR #52380](https://github.com/tenstorrent/tt-metal/pull/52380)) and is present in this tree at
[ttnn/cpp/ttnn/operations/experimental/multi_scale_deformable_attn/](../../../../ttnn/cpp/ttnn/operations/experimental/multi_scale_deformable_attn/).
Performance numbers are not published yet — measuring it against the current decomposition is
itself the first work item.

### What it replaces

`[multi_scale_deformable_attn_ttnn](../tt/tt_ms_deformable_attention.py#L37-L130)` currently
decomposes into, per level: `to_layout` ×2, `permute` ×3, `reshape` ×4, `grid_sample`, `squeeze`,
plus a trailing `stack` + `reshape` + `mul` + `sum` + `reshape` + `permute`. The fused op collapses
the sample-weight-reduce chain into one kernel and drops the layout churn around it.

### Op contract (from the nanobind doc + device-op validation)

```
ttnn.experimental.multi_scale_deformable_attn(value, grid, attn, *,
                                              memory_config=None, align_corners=False)
```

- `value`: `(N, h_in, w_in, D)`, ROW_MAJOR, bfloat16, `N = B * num_heads`
- `grid`:  `(N, Q*P, 1, 2)`, ROW_MAJOR, bfloat16, normalized to `[-1, 1]`
- `attn`:  `(N, Q, P)`, ROW_MAJOR, bfloat16
- returns `(N, Q, D)`, ROW_MAJOR, bfloat16

Hard constraints from
[multi_scale_deformable_attn_device_operation.cpp](../../../../ttnn/cpp/ttnn/operations/experimental/multi_scale_deformable_attn/device/multi_scale_deformable_attn_device_operation.cpp):

- all three inputs **and** the output must be INTERLEAVED (no sharded inputs)
- all bfloat16, all ROW_MAJOR
- `D % 16 == 0` — holds for BEVFormer (`embed_dims=256`, `num_heads=8` → `D=32`)
- `grid.shape[1] == Q * P`



### Fit and blockers

1. `num_levels == 1` **fast path only.** The op's doc line says so explicitly. Temporal
  self-attention runs `num_levels=1` and is a direct drop-in. Spatial cross-attention runs
   multi-level — it needs either a per-level call plus host-side weighted sum (still worth it,
   the per-level fusion is the expensive part), or a follow-up to the op. Decide this by
   measuring the TSA path first.
2. **Layout.** Current code already converts to ROW_MAJOR before `grid_sample`
  ([L81, L86](../tt/tt_ms_deformable_attention.py#L81-L86)), so inputs align. Output is
   ROW_MAJOR `(N, Q, D)` and the next consumer is a `ttnn.linear` needing TILE — one
   `to_layout` at the boundary, same as today.
3. **Grid normalization.** The op expects `[-1, 1]`; the existing `mul(2) - 1`
  ([L75-76](../tt/tt_ms_deformable_attention.py#L75-L76)) already produces that. Keep it.
4. `align_corners`**.** Default `False` matches mmcv, which is what the reference
  BEVFormer/mmcv implementation uses. Do not flip it without a PCC check.
5. **Reshape overhead.** `attn` is wanted as `(N, Q, P)`; the current code builds
  `(N, 1, Q, L*P)` for the elementwise-mul path ([L110-111](../tt/tt_ms_deformable_attention.py#L110-L111)).
   Different target shape — the permute is the same, only the trailing reshape changes.



### Work items

- [ ] Confirm the op is built in the current `python_env` (`ttnn.experimental.multi_scale_deformable_attn`).
- [ ] Standalone microbenchmark: fused op vs. the current decomposition at BEVFormer's TSA shapes.
- [ ] Swap the TSA (`num_levels=1`) path, gate on PCC against the torch reference.
- [ ] Decide the multi-level SCA strategy based on the measured per-level win.



## Ordering

Profile first. My expectation, to be falsified by data: the SCA host round-trips (1a/1b/1c)
dominate wall-clock by a wide margin because they serialize the pipeline, while candidate 2 is
the larger *device-time* win. Fixing 1b is also a prerequisite for trace capture, which is likely
the single biggest end-to-end lever — so it wins on ordering even if the fused op wins on
device time.
