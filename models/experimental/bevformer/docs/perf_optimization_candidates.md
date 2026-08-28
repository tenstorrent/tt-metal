# BEVFormer encoder — performance optimization candidates

The **backlog**. Measured results are not here — they live in [PERF.md](PERF.md) and
[`perf_reports/`](perf_reports/). When a candidate lands, it gets a row in PERF.md and its status
here becomes `landed` with a link to its report.

Numbers quoted below as *cost* come from the baseline profile
([00-baseline.md](perf_reports/00-baseline.md)): one encoder layer, N150, 655.6 ms kernel +
2416.5 ms host gap = 3072.1 ms wall. The layer today is **489.5 ms kernel + 14.0 ms gap** after
[stage 04](perf_reports/04-fused-msda.md); costs quoted against the baseline are historical.

## Candidates

| # | candidate | targets | measured cost at baseline | effort | risk | status |
|--:|---|---|---|---|---|---|
| [1](#candidate-1--host-round-trips) | remove host round-trips from SCA | gap | **1917 ms** (62% of wall) | — | — | **complete** |
| [1a](#1a-rebatch-and-scatter-back-on-device) | rebatch + scatter-back on device | gap | **−2344.7 ms wall (−76%)** | L | med | **landed — [01](perf_reports/01-sca-rebatch-on-device.md)** |
| [1b](#1b-bound-max_len-statically) | bound `max_len` statically | gap, trace | +129 ms kernel to unlock ~9 ms gap | M | high | **[rejected](perf_reports/DEAD_ENDS.md#3-a-static-bound-on-max_len)** |
| [1c](#1c-hoist-index-computation-above-the-layer-loop) | rebatch plan once per frame, not per layer | gap | −94.6 ms encoder wall (−2.2%) | S | low | **landed — [02](perf_reports/02-rebatch-plan-hoisted.md)** |
| [1d](#1d-per-call-constant-uploads) | cache what is frame-invariant | gap | −56.4 ms encoder wall (−1.3%) | S | none | **landed — [03](perf_reports/03-constant-uploads-cached.md)** |
| [2](#candidate-2--fused-msda) | fused `multi_scale_deformable_attn` | kernel | **−191.6 ms kernel (−28.1%)** | M | med | **landed — [04](perf_reports/04-fused-msda.md)** |
| [3](#candidate-3--tile-padding-waste) | fold the offset normalizer into the Linear | kernel | ~24 ms | S | low | todo — **rescoped by 2** |
| [4](#candidate-4--the-msda-concat) | replace the per-level concat | kernel | — | — | — | **moot — deleted by [04](perf_reports/04-fused-msda.md)** |
| [5](#candidate-5--trace-capture) | trace capture the encoder | gap | ≤9 ms/layer | M | low | parked behind 1b |
| [6](#candidate-6--msdaoperation-itself) | `MSDAOperation` device time | kernel | **143 ms** | ? | ? | todo — upstream |

Ordering rationale is at the [bottom](#ordering).

Things tried and rejected, with their numbers, are in
[perf_reports/DEAD_ENDS.md](perf_reports/DEAD_ENDS.md).

---

## Candidate 1 — host round-trips

`ttnn.to_torch` is a device→host readback that serializes the pipeline: it flushes the op queue,
blocks on the readback, and makes trace capture impossible. The matching `from_torch` pushes back
down. In BEVFormer these are not one-off setup costs — several sit inside per-layer, per-camera
loops.

The baseline confirms the cost precisely: a **single 1.917 s gap** on the first op after the SCA
rebatch loop. Two-thirds of the layer's wall clock, one Python loop.

### 1a. Rebatch and scatter-back on device

**Landed** — [stage 01](perf_reports/01-sca-rebatch-on-device.md), −2344.7 ms wall (−76%), PCC
unchanged at 0.999608. Both rebatch gathers run as a single `ttnn.embedding` each and cost 0.27 ms
combined; the scatter-back is one `ttnn.scatter_add` at 10.50 ms. What it was:

The `TODO: Currently done on CPU, to be modified once TTNN supports required indexing ops` comments
([L189](../tt/tt_spatial_cross_attention.py#L189),
[L281](../tt/tt_spatial_cross_attention.py#L281)) are **stale**. Every indexing op they wait on
exists in this build — verified against the installed `ttnn`, not against documentation:

| Need | Op | Constraints (read from the device-op validation) |
|---|---|---|
| Row gather `query[j, valid_indices]` | `ttnn.embedding(indices, weight)` | `weight` ROW_MAJOR bfloat16, `padded_shape[0] == padded_shape[1] == 1` → `[1, 1, num_queries, E]`; `indices` UINT32 or BFLOAT16, ROW_MAJOR needs `padded_shape[1] == padded_shape[2] == 1` → `[B, 1, 1, S]`; both INTERLEAVED. Output `[B, 1, S, E]`. A **tilized** output additionally needs `E % 32 == 0` *and* `S % 32 == 0`. Also takes a `padding_idx`. |
| Generic gather along a dim | `ttnn.gather(input, dim, index)` | Index must have the **same rank** as input and the output takes the index's shape, so the index has to be materialized at `[bs, max_len, E]` (row id broadcast across `E`). Works, but wastes bandwidth vs. `embedding`. |
| Scatter-back accumulation | `ttnn.scatter_add(input, dim, index, src)` | `index` dtype INT32/UINT8/UINT16/UINT32; input, index and src all on device and **not sharded**; `input.dtype == src.dtype`; output is ROW_MAJOR. Torch semantics — `index` and `src` share a shape, so the row id materializes across `E` here too. |
| Mask → indices | `ttnn.nonzero(input)` | **1D, ROW_MAJOR only.** Returns `(count, indices)`, `indices` padded to input length, only the first `count` entries meaningful — and `count` is a *device* tensor. |

So `queries_rebatch` / `reference_points_rebatch` build with `embedding`, and the scatter-back with
`scatter_add`. **There is no missing op.** What the comment should have said is that `max_len` is a
data-dependent shape (see 1b) — a dynamic-shape gap, not an indexing-op gap.

**Scope of this step.** The index sets are derived from `bev_mask`, which is synced to host anyway
for `max_len`; the indices themselves are tiny (`[bs, num_cams, max_len]` ints — ~2.5 K values at
100×100 against 2.56 M for `query`). So 1a computes the indices on host and does the **gather and
scatter on device**: `query`, `reference_points_cam`, the rebatch accumulators and `slots` all stop
crossing the bus. Moving the index derivation itself onto `ttnn.nonzero` buys nothing while the
mask is synced regardless, and is only worth doing once 1b removes that sync.

`count` ([L302-304](../tt/tt_spatial_cross_attention.py#L302-L304)) is
`sum(-1) > 0 → permute → sum(-1) → clamp` — pure elementwise/reduce, maps 1:1 onto ttnn ops.

Both TODO comments were deleted rather than edited — they describe a constraint that never held in
this build.

### 1b. Bound `max_len` statically

**Investigated and rejected** — [DEAD_ENDS entry 3](perf_reports/DEAD_ENDS.md#3-a-static-bound-on-max_len),
full data in that entry.

`max_len` is a data-dependent shape, so it forces the `bev_mask` readback that keeps the encoder from
being trace-capturable. A bound is derivable — the coverage ratio is a stable camera-FOV property —
but cost is exactly linear in it, the naive bound does not fit in DRAM at all, and a sensible one
costs +129 ms of kernel against the ~9 ms of gap trace capture could return. This entry expected
candidate 2 to relieve it, on the theory that 2 owned both the memory ceiling and most of the
per-row cost.

**It does not — tested, not assumed.** [Stage 04](perf_reports/04-fused-msda.md) left the ceiling
exactly where it was: SCA at `bev_size=(200,200)` fails with the same
`Out of Memory: … 2969567232 B DRAM buffer`, byte-for-byte identical before and after the fused op.
The allocation is in the sampling-location math upstream of the op, which 2 never touched. 1b stays
rejected on its original terms, and stays rejected for the same reason.

**1b gates candidate 5**, and nothing else in candidate 1 needs it.

### 1c. Hoist index computation above the layer loop

**Landed** — [stage 02](perf_reports/02-rebatch-plan-hoisted.md), −94.6 ms encoder wall (−2.2%,
read as ~1–2%: the encoder is measured end-to-end, not profiled). No numerical change.

More was invariant than this entry originally claimed: not just the index derivation but the
**entire reference-point rebatch**, which never touches `query`. Six identical gathers were being
computed and discarded. What remains per layer is only the query gather. Five of six `bev_mask`
readbacks are gone and the layer loop no longer contains one — the structural result, worth more
than the 2%.

Why it is available at all: `bev_mask` depends on `lidar2img`, so it changes per frame but is
identical across all encoder layers within a frame, and the encoder already computes it once above
the layer loop ([tt_encoder.py](../tt/tt_encoder.py#L464)). Everything SCA derives from it is
therefore per-forward work that was being repeated per layer.

### 1d. Per-call constant uploads

**Landed** — [stage 03](perf_reports/03-constant-uploads-cached.md), −56.4 ms encoder wall (−1.3%)
and −25 device ops per layer, taking layer gap to 8.9 ms. No numerical change. A steady-state win:
the caches hold across forwards, not just across layers, so the first frame still pays.

They were small individually, but ran every layer / every forward:

| Location | Tensor | Fix |
|---|---|---|
| [tt_encoder.py:157](../tt/tt_encoder.py#L157) | `bev_reference_points` | Precompute once at module init, cache as a device tensor. Depends only on `reference_points_3d`, which is frame-invariant. |
| [tt_ms_deformable_attention.py:283](../tt/tt_ms_deformable_attention.py#L283) | `spatial_shapes_tt` (→ `offset_normalizer`) | Config-derived, identical every call. Cache the `offset_normalizer` itself, not just `spatial_shapes`. |
| [tt_point_sampling_3d_2d.py:69](../tt/tt_point_sampling_3d_2d.py#L69) | reference points from `torch_generate_reference_points` | Already marked `TODO: Calculate during initialization`. Do it. |
| [tt_point_sampling_3d_2d.py:115-120](../tt/tt_point_sampling_3d_2d.py#L115-L120) | `reference_points`, `lidar2img` | `lidar2img` is per-frame input, unavoidable; the rest is cacheable. |
| [tt_temporal_self_attention.py:163](../tt/tt_temporal_self_attention.py#L163) | `level_start_index` | Constant `[0]` in the default path. Build once. |
| [tt_spatial_cross_attention.py:245](../tt/tt_spatial_cross_attention.py#L245) | `to_torch(spatial_shapes)` | Only used for an assert. Guard behind a debug flag or keep a host-side copy. |

Weight preprocessing in [model_preprocessing.py](../tt/model_preprocessing.py) is one-time setup —
not a candidate.

---

## Candidate 2 — fused MSDA

**Landed** — [stage 04](perf_reports/04-fused-msda.md), −191.6 ms kernel (−28.1%), −190.6 ms layer
wall, PCC 0.999611 against 0.999608. `GridSample` and `Concat` are both gone from the profile.

Two things the entry below got wrong, corrected by the measurement:

- **SCA does not need a host-side weighted sum.** `attention_weights` is softmaxed jointly over
  `L*P` and thereafter only summed, so the joint sum decomposes exactly into per-level sums. Four
  fused calls plus an L-way `ttnn.add` on device, exact.
- **The fused op is slower than the sampling it replaces** — 24.4 ms vs 16.8 ms at TSA, 143.3 ms
  vs 99.2 ms at SCA. The entire win is deleting the `stack`/`mul`/`sum` tail it makes unnecessary.
  That cost is now [candidate 6](#candidate-6--msdaoperation-itself).

Prior art that would have saved the derivation: `models/experimental/vadv2/tt/tt_utils.py` already
runs this op, with the tensor-shuffle recipe and a measured `N*Q >= 1024` floor below which the
decomposition wins.

What it was:

Replace the hand-rolled multi-scale deformable attention with
`ttnn.experimental.multi_scale_deformable_attn`
([PR #52380](https://github.com/tenstorrent/tt-metal/pull/52380)), present in this tree at
[ttnn/cpp/ttnn/operations/experimental/multi_scale_deformable_attn/](../../../../ttnn/cpp/ttnn/operations/experimental/multi_scale_deformable_attn/).

**Why it matters:** the two MSDA calls are 613 ms of the layer's 656 ms kernel time — SCA 522 ms,
TSA 92 ms. Published performance numbers for the fused op do not exist; measuring it against the
current decomposition is the first work item.

### What it replaces

[`multi_scale_deformable_attn_ttnn`](../tt/tt_ms_deformable_attention.py#L37-L130) decomposes into,
per level: `to_layout` ×2, `permute` ×3, `reshape` ×4, `grid_sample`, `squeeze`, plus a trailing
`stack` + `reshape` + `mul` + `sum` + `reshape` + `permute`. The fused op collapses the
sample-weight-reduce chain into one kernel and drops the layout churn around it — which is exactly
where the baseline's reshape (155 ms), concat (114 ms) and permute (105 ms) time sits.

### Op contract

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

1. **`num_levels == 1` fast path only.** The op's doc line says so explicitly. TSA runs
   `num_levels=1` and is a direct drop-in — but TSA is only 92 ms of the 613 ms. SCA runs
   multi-level and holds the other 522 ms; it needs either a per-level call plus a host-side
   weighted sum (still worth it — per-level fusion is the expensive part) or a follow-up to the op.
   Measure the TSA path first, then decide.
2. **Layout.** Current code already converts to ROW_MAJOR before `grid_sample`
   ([L81, L86](../tt/tt_ms_deformable_attention.py#L81-L86)), so inputs align. Output is ROW_MAJOR
   `(N, Q, D)` and the next consumer is a `ttnn.linear` needing TILE — one `to_layout` at the
   boundary, same as today.
3. **Grid normalization.** The op expects `[-1, 1]`; the existing `mul(2) - 1`
   ([L75-76](../tt/tt_ms_deformable_attention.py#L75-L76)) already produces that. Keep it.
4. **`align_corners`.** Default `False` matches mmcv, which is what the reference BEVFormer
   implementation uses. Do not flip it without a PCC check.
5. **Reshape overhead.** `attn` is wanted as `(N, Q, P)`; the current code builds `(N, 1, Q, L*P)`
   for the elementwise-mul path
   ([L110-111](../tt/tt_ms_deformable_attention.py#L110-L111)). Different target shape — the permute
   is the same, only the trailing reshape changes.

### Work items

- [ ] Confirm the op is built in the current `python_env`.
- [ ] Standalone microbenchmark: fused op vs. the current decomposition at BEVFormer's TSA shapes.
- [ ] Swap the TSA (`num_levels=1`) path, gate on PCC against the torch reference.
- [ ] Decide the multi-level SCA strategy based on the measured per-level win.

---

## Candidate 3 — tile padding waste

**Rescoped by [stage 04](perf_reports/04-fused-msda.md): ~60 ms → ~24 ms, and the fix changed.**
The two `Permute` sites below were inside the MSDA decomposition and no longer exist; `ReshapeView`
dropped 157 → 77 ms for the same reason. What survives is the `BinaryNg` site — the `div` by
`offset_normalizer`, 23.9 ms, profiler row 506 of `2026_08_27_23_24_32`. It sits *upstream* of the
fused op, in the sampling-location math, which candidate 2 did not touch.

**The better fix is already in the tree.** `fold_offset_normalizer_into_weight`
(`models/experimental/vadv2/tt/tt_utils.py`) pre-scales the `sampling_offsets` Linear weight and
bias by `1/[W, H]`, so the division never runs at all — rather than un-padding it as this entry
proposes. Exact (`s·(Wx+b) == (Wx+b)/norm`), computed once. It is written for `num_levels == 1`, but
the Linear output is laid out `(L, P, 2)`, so a per-`(level, axis)` static scale vector generalizes
it to SCA.

The original entry, for the record:

Exposed by the baseline, not previously listed. Several TILE ops compute almost entirely on padding:

| op | logical shape | padded to | waste | cost |
|---|---|---|---:|---:|
| BinaryNg | `[80000, 1, 4, 2]` | `[80000, 1, 32, 32]` | 128× | 23.0 ms |
| Permute ×2 | `[6, 30125, 1, 256]` | `[6, 30125, 32, 256]` | 32× | 38.5 ms |
| ReshapeView | `[1, 6, 30125, 256]` | `…30144…` → `[1, 180750, 8, 32]` | 4× on the last axis | 21.2 ms |

The pattern is the same each time: a coordinate or offset axis of extent 2 or 4, or a broadcast axis
of extent 1, sitting in one of the two tiled dimensions. Roughly 60 ms of kernel time — 9% of the
layer — is arithmetic on zeros.

Fixes, in increasing order of intrusiveness: keep these tensors ROW_MAJOR where the consumer
allows; fold the length-2 coordinate axis into the last dimension so it tiles alongside real data;
reorder the permutes so the degenerate axis is never one of the trailing two. Needs a per-site read
of [tt_ms_deformable_attention.py](../tt/tt_ms_deformable_attention.py) — the shapes above name the
sites.

Candidate 2 may delete some of these sites outright. Sequence 3 after 2.

## Candidate 4 — the MSDA concat

**Moot — deleted by [stage 04](perf_reports/04-fused-msda.md).** The fused op returns `(N, Q, D)`
already reduced over the sampling points, so there is no per-level list to stack. `Concat` went
115.5 ms → **0**, and the 74.9 ms reshape that followed it went with it.

This entry was ordered *before* candidate 2 on the grounds that it was cheap and self-contained.
That was wrong: 2 subsumes it, and doing 4 first would have been discarded work. When one candidate
rewrites the region another one lives in, sequence the rewrite first.

What it was:

The single most expensive op in the layer: **113.5 ms**, one `ConcatDeviceOperation` stacking four
`[32, 2484, 1, 4]` ROW_MAJOR tensors into `[32, 2484, 4, 4]` on 64 cores. That is 1.27 MB of output
for 113 ms — it is not moving data, it is running badly.

This is the per-level `stack` in the MSDA decomposition
([tt_ms_deformable_attention.py](../tt/tt_ms_deformable_attention.py#L37-L130)). Options: preallocate
the output and write each level into a slice; concat along a different (non-degenerate) axis and
reshape; or let candidate 2 remove the stack entirely. Cheap to try, large single-op win, and worth
a standalone microbenchmark either way — a 113 ms concat at this size may well be an op bug worth
reporting upstream.

## Candidate 5 — trace capture

Trace capture needs shapes that are static across replays, and `rebatch_len` is the one shape that is
not — it is redecided every frame. 1a and 1c removed the transfers and moved the last readback out of
the layer loop, but neither makes that shape constant; only 1b would, and 1b is
[rejected](perf_reports/DEAD_ENDS.md#3-a-static-bound-on-max_len) on its own terms.

So 5 is parked with it — and re-measurement has collapsed what it was worth. Per-layer gap is
**8.9 ms**, not the 218 ms this entry was written against. Trace capture cannot recover more than
that at the layer level, and against 1b's +129 ms of kernel the trade is not close.

Candidate 5 keeps a reason to exist at the *encoder* level, where per-forward host work is not
hidden behind device time — encoder wall is 4234.5 ms against 6 × 691 = 4146 ms of steady-state
layer time, so ~90 ms sits outside the layers. That case has to be made on the encoder harness and nobody
has measured it. Treat 5 as low-value, not as blocked high-value.

Both perf harnesses document the current blocker in their module docstrings; update them when it
lifts.

## Candidate 6 — `MSDAOperation` itself

New, created by [stage 04](perf_reports/04-fused-msda.md). The fused op is now **the single largest
cost in the layer**: 167.6 ms total, 143.3 ms of it the four SCA calls at ~35.9 ms each.

It is more expensive than the sampling it absorbed:

| | old `GridSample` | new `MSDAOperation` |
|---|---:|---:|
| TSA (1 level) | 16.8 ms | 24.4 ms |
| SCA (4 levels) | 99.2 ms | 143.3 ms |

+45% per sample. The op still wins because it deletes 215 ms of tail, but that says the tail was
bad, not that the kernel is good. VADv2 independently measured a floor of `N*Q >= 1024` below which
the decomposition beats it — consistent with a real launch/packing overhead.

This is an **upstream** item, not a model-side one:

- [ ] Standalone microbenchmark of `ttnn.experimental.multi_scale_deformable_attn` at BEVFormer's
      SCA shapes (`N=48, Q=2496, P=4, D=32`) against a bare `ttnn.grid_sample` at the same shapes.
- [ ] If the gap reproduces, file it against the op with the numbers.

Also in the region and cheaper to attack from this side: **~103 ms of per-level layout prep**
(`Untilize`/`Transpose`/`Slice`/`Permute`/`Tilize`, ×4). Some is genuinely per-level; the
tilize↔untilize churn around each call is not obviously necessary and is model-side work.

---

## Ordering

1. ~~**1a**~~ — landed, −2344.7 ms.
2. ~~**1b**~~ — [rejected](perf_reports/DEAD_ENDS.md#3-a-static-bound-on-max_len): +129 ms of kernel
   to unlock ~9 ms of gap, and candidate 2 owns the memory ceiling that caps it.
3. ~~**1c**~~ — landed, −94.6 ms encoder wall.
4. ~~**1d**~~ — landed, −56.4 ms encoder wall. **Candidate 1 is complete.**
5. ~~**2**~~ — landed, −191.6 ms kernel (−28.1%). **Was ordered after 4 and should not have been.**
6. ~~**4**~~ — never run; [deleted by 2](perf_reports/04-fused-msda.md).
7. **6** — 143 ms, the largest single cost now. Microbenchmark before anything else; the answer may
   belong upstream rather than in this model.
8. **3** — ~24 ms after rescoping, and the fix (fold the normalizer into the Linear weight) is
   already written in `vadv2`. Cheapest remaining model-side win.
9. **5** — needs 1b, and is worth ≤9 ms/layer rather than the 218 ms first claimed. Only revisit if
   an encoder-harness measurement shows per-forward host time the layer profile does not.

The baseline settled what was previously a guess: host round-trips dominated wall clock 4:1 over
kernel time, and within kernel time it is layout churn, not arithmetic — matmul is 0.7%. Nothing in
the matmul-tuning playbook applies here. Stage 01 inverted the ratio; stage 04 then took −28% of
kernel by deleting the layout churn around the sampler rather than by making arithmetic faster.

**Kernel is 97% of layer wall clock and the two MSDA calls are 88% of the kernel** — still the only
region worth working, but the character has changed. The churn is largely gone and what remains is
one expensive device op. That makes 6 the live question and it is an op-level one, which is a
different kind of work than everything above it in this list.

Two lessons this backlog got wrong and should not repeat:

- **Sequence rewrites before the cleanups inside them.** 4 was ranked first for being cheap; 2
  deleted it.
- **Grep the tree for prior art before deriving an op contract.** `vadv2` had a working
  `multi_scale_deformable_attn` call site, a measured shape floor, and the offset-normalizer fold —
  all three relevant, none referenced here until stage 04.
