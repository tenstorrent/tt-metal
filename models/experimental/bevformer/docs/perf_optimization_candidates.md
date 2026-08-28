# BEVFormer encoder — performance optimization candidates

The **backlog**. Measured results are not here — they live in [PERF.md](PERF.md) and
[`perf_reports/`](perf_reports/). When a candidate lands, it gets a row in PERF.md and its status
here becomes `landed` with a link to its report.

Numbers quoted below as *cost* come from the baseline profile
([00-baseline.md](perf_reports/00-baseline.md)): one encoder layer, N150, 655.6 ms kernel +
2416.5 ms host gap = 3072.1 ms wall.

## Candidates

| # | candidate | targets | measured cost at baseline | effort | risk | status |
|--:|---|---|---|---|---|---|
| [1](#candidate-1--host-round-trips) | remove host round-trips from SCA | gap | **1917 ms** (62% of wall) | — | — | partly landed |
| [1a](#1a-rebatch-and-scatter-back-on-device) | rebatch + scatter-back on device | gap | **−2171.9 ms wall (−71%)** | L | med | **landed — [01](perf_reports/01-sca-rebatch-on-device.md)** |
| [1b](#1b-bound-max_len-statically) | bound `max_len` statically | gap, trace | the last readback + trace capture | M | high | investigation |
| [1c](#1c-hoist-index-computation-above-the-layer-loop) | index computation once per frame, not per layer | gap | ÷6 at encoder level | S | low | todo |
| [1d](#1d-per-call-constant-uploads) | move to `__init__` what is frame-invariant | gap | small, every layer | S | none | todo |
| [2](#candidate-2--fused-msda) | fused `multi_scale_deformable_attn` | kernel | up to 613 ms | M | med | **landed — [02](perf_reports/02-fused-msda.md)** |
| [3](#candidate-3--tile-padding-waste) | kill tile padding on degenerate dims | kernel, DRAM | **−177.3 ms, and 200×200 now runs** | M | low | **landed — [03](perf_reports/03-camera-fold.md), [04](perf_reports/04-flat-sampling-chain.md)** |
| [4](#candidate-4--the-msda-concat) | replace the per-level concat | kernel | **114 ms** (single op) | S | low | closed — deleted by 2 |
| [5](#candidate-5--trace-capture) | trace capture the encoder | gap | all remaining gap | M | low | blocked on 1b |
| [6](#candidate-6--the-fused-msda-op-itself) | the fused MSDA op itself | kernel | **−138.3 ms**, and it was never the op | L | med | **landed — [06](perf_reports/06-sfpu-geometry.md)** |
| [7](#candidate-7--an-msda-head-reshape-op) | an MSDA head-reshape op | kernel | — | L | med | closed — wrong shape, see 9 |
| [9](#candidate-9--axes-as-addresses-not-data) | axes as addresses, not data | kernel | **~22 ms left** of ~45 | M | low | **in progress — [08](perf_reports/08-packed-value-heads.md)** |
| [8](#candidate-8--the-grids-point-axis-in-its-page) | fold the grid's point axis into its page | kernel | **−12.9 ms**, and the page was the story | S | low | **landed — [07](perf_reports/07-folded-grid-page.md)** |

Ordering rationale is at the [bottom](#ordering).

**Rejected:** *hoist the invariant reads out of the host rebatch loop.* Measured at −1721 ms wall
(−56%) and dropped anyway: it optimizes the number of host transfers instead of removing the reason
for having any, and 1a deletes the loop it optimizes. Reducing a cost is not the same as removing
it; when both are available at comparable effort, the removal wins.

---

## Candidate 1 — host round-trips

`ttnn.to_torch` is a device→host readback that serializes the pipeline: it flushes the op queue,
blocks on the readback, and makes trace capture impossible. The matching `from_torch` pushes back
down. In BEVFormer these are not one-off setup costs — several sit inside per-layer, per-camera
loops.

The baseline confirms the cost precisely: a **single 1.917 s gap** on the first op after the SCA
rebatch loop. Two-thirds of the layer's wall clock, one Python loop.

### 1a. Rebatch and scatter-back on device

**Landed** — [stage 01](perf_reports/01-sca-rebatch-on-device.md), −2171.9 ms wall (−71%), PCC
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

[tt_spatial_cross_attention.py:157](../tt/tt_spatial_cross_attention.py#L157),
[:167](../tt/tt_spatial_cross_attention.py#L167)

**An investigation, not a scheduled change.** `max_len = max over cameras of (valid query count)` is
a **data-dependent shape**. It sizes `queries_rebatch`, so it must be a Python int, so `bev_mask`
must come back to host. TTNN has no dynamic shapes and will not grow them for us; the fix, if there
is one, is on our side.

Padding slots address a sentinel row that is sliced off after the scatter, so an over-large
`max_len` costs wasted MSDA compute, not correctness. At
100×100 the measured `max_len` is **2484** — the MSDA signpost reports it directly as
`query.shape[1]` — against `num_queries = 10000`. Bounding at the worst case is a **4× increase** on
the 521 ms MSDA kernel, roughly 2.1 s per layer against a current whole-layer wall clock of 900 ms.
Not a non-starter, but nowhere near free.

`max_len` is already ~25% of `num_queries` at this grid size, so the headroom a bound has to cover
is the gap between typical and worst-case camera coverage, not between a handful of queries and all
of them. That is what makes a calibrated bound plausible here and worth measuring.

So the investigation is: how tight can a *calibrated* per-config bound be, and what does it cost?
Two things have to come out of it:

1. **The compute curve.** MSDA time against `max_len`, so the price of any candidate bound is known
   rather than guessed.
2. **The safety argument.** A bound that some frame exceeds silently drops queries. Whatever bound
   is chosen needs either a proof from the BEV grid and camera FOV geometry, or a runtime assert
   plus a fallback path — and the assert reintroduces a sync unless it is debug-gated.

The alternative that avoids the whole question: **one sync per forward instead of six** — see 1c.
That gets nearly all of the remaining gap without a bound, and it is on the way here rather than a
detour. 1b's real payoff is candidate 5 (trace capture), which needs static shapes across replays.

**1b gates candidate 5.** Nothing else in candidate 1 needs it.

### 1c. Hoist index computation above the layer loop

The index set is **frame-dependent but not layer-dependent**. `bev_mask` comes from
`point_sampling_3d_to_2d`, i.e. it depends on `lidar2img`: it changes per frame, but it is identical
across all encoder layers within a frame. `reference_points_cam` and `bev_mask` are already computed
once per forward above the layer loop ([tt_encoder.py:464](../tt/tt_encoder.py#L464)) — but SCA
re-derives `indexes`, `max_len` and `valid_indices` from that mask inside **every** layer.

Hoisting that derivation to the encoder and passing it down means **one host sync per frame instead
of six**, and leaves each layer's SCA fully device-side. No static bound, no correctness risk, no
MSDA blow-up. The layer harness cannot show this win (it runs one layer); it is worth 6× at encoder
level, and it is the pragmatic alternative to 1b.

### 1d. Per-call constant uploads

Small, but they run every layer / every forward:

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

Blocked on 1b specifically: trace capture needs shapes that are static across replays, and `max_len`
is the one shape that is not. 1a and 1c remove the transfers but not the sync that decides the
shape. Once no host readback decides a downstream shape, the layer becomes trace-capturable and
**all** remaining op-to-op gap collapses — 2416 ms per layer at baseline.

Both perf harnesses document the current blocker in their module docstrings; update them when it
lifts.

---

## Ordering

1. ~~**1a**~~ — landed, −2171.9 ms wall.
2. ~~**2**~~ — landed, −194.1 ms kernel. It deleted candidate 4 as a side effect.
3. ~~**4**~~ — closed. The concat only existed to hold the per-level results the fused op now
   reduces itself; 115.4 ms → 0.00 ms with no work of its own.
4. ~~**3**~~ — landed in two parts, −177.3 ms kernel total, and `200×200` now runs. It was two
   independent defects, not one: a degenerate batch axis tiled in the wrong position (stage 03,
   −36.8 ms) and a trailing `(num_points, 2)` padding to 128× (stage 04, −140.5 ms).
5. ~~**the residual layout churn**~~ — done, −46.5 ms
   ([05](perf_reports/05-hoisted-layout-ops.md)). Two per-level ops hoisted out of the level loop.
   What is left is ~52 ms with no common cause, and a profile read op by op found no further target
   of that kind. The Python-side layout work is closed.
6. ~~**6**~~ — landed, −138.3 ms ([06](perf_reports/06-sfpu-geometry.md)). The premise was wrong:
   the op was not slow, the compute kernel was idle waiting on a reader doing soft-float geometry.
   Moving that onto the SFPU collapsed it 167.6 → 29.5 ms without touching the sampling kernel.
7. ~~**7**~~ — closed unstarted. A fused head-reshape op still writes the head-major tensor it
   exists to hand over; the cost was the page, not the call count.
8. ~~**8**~~ — landed, −12.9 ms ([07](perf_reports/07-folded-grid-page.md)).
9. **9** — the generalisation of 7 and 8: each of value, grid, attn and output stops being copied
   into head-major or level-major form and is addressed by offset instead. Value landed
   ([08](perf_reports/08-packed-value-heads.md), −20.5 ms); attn, grid and output remain.
7. **1c** — hoist the index derivation to the encoder: one sync per frame instead of six. Pure
   refactor, no op risk.
8. **1d** — move to `__init__` what is genuinely frame-invariant.
9. **1b** — the `max_len` investigation. Documented as a study with a compute curve and a safety
   argument, not as a change to land blind.
10. **5** — needs 1b.

The baseline settled what was previously a guess: host round-trips dominated wall clock 4:1 over
kernel time, and within kernel time it is layout churn, not arithmetic — matmul is 0.7%. Nothing in
the matmul-tuning playbook applies here.

Stages 02–05 settled the rest. Measured on Release, kernel is **86%** of wall clock and total gap is
41.8 ms — so 1b, 1c and 1d together cannot recover more than that, and they sit below the op itself.
Stage 01's 218.3 ms gap was a Debug-build artifact; see
[02](perf_reports/02-fused-msda.md) § *Baseline this is measured against*.

After four device-time stages the layer is at **263.6 ms of kernel, down 61.3%**, and the shape of
the problem has changed again: it is now one op, not a hundred small ones.

---

## Candidate 6 — the fused MSDA op itself

`ttnn.experimental.multi_scale_deformable_attn` is **167.8 ms across 5 calls, 64% of kernel time**.
Everything else in the layer put together is 96 ms.

A bandwidth estimate for one SCA level call — `48 × 2496` queries × 4 points × 4 bilinear taps × 32
channels × 2 B ≈ 123 MB, plus ~11 MB of grid, weights and output — puts the DRAM roof at **0.46 ms**
against a measured **36 ms**. TSA's call reproduces the ratio independently: 88 MB, 0.31 ms roof,
24.35 ms measured. Both land at **78–79× above the roof**, or ~1.3% of it.

Two further observations point the same way:

- Cost is **flat across levels** whose `value` tensors differ 64-fold in size (`200×113` down to
  `25×15`). It tracks the sample-point count, not the data read.
- ~4800 cycles per sampled point per core, for a bilinear fetch of 32 channels — 128 reads and 128
  multiplies.

**Landed** — [stage 06](perf_reports/06-sfpu-geometry.md), 167.6 → **29.5 ms**, PCC unchanged.

Every number above is correct and every conclusion drawn from them was wrong. The op was 78× above
its DRAM roof because it was not reading DRAM: `CB-COMPUTE-WAIT-FRONT` measured 36.1 ms on a 36.0 ms
call, so the compute kernel was idle for the whole op. The reader was deriving the sampling geometry
in floating point on a dataflow core that has **no FPU** — ~140 cycles an operation of soft-float
emulation — which is also why cost tracked point count rather than data read, and why it was flat
across levels.

The fix was to move that arithmetic onto the SFPU and leave the sampling kernel alone. It needed no
upstream change.

**What to take from this.** A roofline says an op is not bound by the resource you modelled. It does
not say which resource binds it. "78× above the DRAM roof" was read as *the kernel is slow* when it
meant *this op is not doing memory work*. The distinction cost four stages of looking in the wrong
place, and one CB-wait zone settled it.

---

## Candidate 7 — an MSDA head-reshape op

`Reshape` + `Permute` + `Slice` + `Untilize` is **71.7 ms, 57% of kernel** — three times the fused
op. Read against the stage-06 profile it is not shapeless:

| group | ops | cost |
|---|---|---:|
| `value` → per-level heads | untilize, split embed_dims into (num_heads, head_dim), permute heads ahead of the spatial axis, slice the level | **~24 ms** |
| grid and attn prep | reshape, untilize, head-major permute, per-level slice | **~31 ms** |
| output concat heads | one permute back | ~1 ms |

`ttnn/cpp/ttnn/operations/experimental/transformer/` already carries **19** ops that exist for
exactly this: `nlp_create_qkv_heads` and `nlp_concat_heads` plus model-specific members (`_vit`,
`_segformer`, `_falcon7b`, `_boltz`, `_decode`). Each fuses the layout plumbing that surrounds an
attention op into one kernel because no generic op does it well. What is missing is the
deformable-attention member of that family.

The first group is the direct analogue of `nlp_create_qkv_heads`: one op taking `(B, L, embed_dims)`
and the level shapes, emitting per-level `(B*num_heads, H, W, head_dim)` ready for the fused op. The
second has no existing analogue — it is deformable-specific — but it is the larger half.

**Two cheaper things to try first**, both Python-side and both measurable in an afternoon:

- The largest single reshape, `1x1x119808x32 → 1x1x1916928x2` at **7.4 ms**, is a pure ROW_MAJOR
  view that splits the last axis. ttnn materialises it.
- The largest permute, **9.96 ms** moving 69.4 MB, is bandwidth-bound rather than per-call overhead.

Stage 05 measured both as a wash and reverted them. It measured them against a kernel twice as slow,
and against a critical path that has since moved.

---

## Candidate 8 — the grid's point axis in its page

**Landed** — [stage 07](perf_reports/07-folded-grid-page.md), kernel −12.9 ms, wall −16.7 ms, PCC
unchanged.

A ROW_MAJOR page is the last dimension, and the buffer rounds it up to the 32-byte DRAM alignment.
The fused op took its grid as `(N, Q*P, 1, 2)` — a 4-byte page in a 32-byte slot, so the SCA grid
held 3.83 M points in **122 MB** for 7.67 MB of data, and the reader issued P four-byte NoC reads
per query. The op now also accepts `(N, Q, 1, P*2)`: one read, a quarter of the pages, 30.6 MB.

This is what the layout profile had been saying all along. Effective bandwidth sorted the ops by
page width, not by size — 512 B at 38 GB/s, 64 B at 14 GB/s, 4 B at 2 GB/s — and the two slowest
reshapes in the layer were the two moving the least data.

Stage 05 tried the Python half alone and correctly measured nothing: without the op change the grid
still reaches the op at width 2, so the 4-byte page is still written and only the op writing it
moves. Any divisor of P per page is accepted, so `(N, Q*P, 1, 2)` still works and VADv2 is
untouched.

---

## Candidate 9 — axes as addresses, not data

**In progress.** Value landed in [stage 08](perf_reports/08-packed-value-heads.md), −20.5 ms of
kernel. Attn, grid and output remain, together worth roughly 22 ms.

Candidate 7 proposed the deformable member of the `nlp_create_qkv_heads` family. That was the wrong
shape: a fused head-reshape op still has to **produce** the head-major tensor for MSDA to read —
92.6 MB written, 92.6 MB read. It removes per-call overhead, and per-call overhead is not what this
costs. The permute ran at 14 GB/s because both its pages were 64 bytes.

The head is not data. It is a byte offset inside a stick, and the level is a byte offset too. Given
`num_heads`, the op derives `b = n / num_heads` and `h = n % num_heads` and reads:

| input | copied form | addressed form | offset |
|---|---|---|---|
| value | `(B*nh, H, W, D)` | `(B, H, W, nh*D)` | `h*D*2` — **landed, −20.5 ms** |
| attn | `(B*nh, Q, P)` | `(B, Q, nh*L*P)` | `(h*L*P + l*P)*2` — **~15 ms** |
| grid | `(B*nh, Q, 1, P*2)` | `(B, Q, nh*L*P*2)` | `(h*L*P + l*P)*2*2` — **~5 ms** |
| output | `(B*nh, Q, D)` | `(B, Q, nh*D)` | `h*D*2` in the writer — **~2 ms** |

Each input is independent: `N_work` is `B*num_heads` either way, so one can be widened while the
others stay in the form the caller already produces. Every step is one stage, one measurement.

`attn` is the largest remaining piece because its copied form carries a trailing `(L, P) = (4, 4)`
that pads to a full `(32, 32)` tile — 64× — which is what the 3.6 ms untilize pays for.

Not yet verified: the writer passes `{.offset_bytes = 0}` in its destination args
(`writer_msda.cpp:87`). Whether the destination struct accepts an offset the way the source struct
does needs checking before the output step.
