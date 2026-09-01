# BEVFormer encoder — performance optimization candidates

The **backlog**. Measured results are not here — they live in [PERF.md](PERF.md) and
[`perf_reports/`](perf_reports/). When a candidate lands, it gets a row in PERF.md and its status
here becomes `landed` with a link to its report.

Numbers quoted below as *cost* come from the baseline profile
([00-baseline.md](perf_reports/00-baseline.md)): one encoder layer, N150, 655.6 ms kernel +
2416.5 ms host gap = 3072.1 ms wall. The layer today is **456.8 ms kernel** after
[stage 05](perf_reports/05-offset-normalizer-folded.md); costs quoted against the baseline are
historical. Gap/wall figures are not comparable across runs — see
[PERF.md](PERF.md#the-gap-column-is-not-reliable).

**Accuracy budget.** Everything landed up to stage 05 is correctness-preserving: PCC moved
0.999608 → 0.999611 across five stages, i.e. not at all. Two items now on the list are not —
[1e's statistical sibling](#a-statistical-bound-instead-of-a-worst-case-one) drops queries, and
[candidate 13](#candidate-13--dtype-and-math-fidelity) spends mantissa — so what they are allowed to
cost has to come from somewhere.

It already does. The gates live in [`tests/pcc/`](../tests/pcc/), per module and per config, each
case carrying `expected_pcc` plus `expected_abs_error`, `expected_rel_error` and
`expected_high_error_ratio`. For the profiled configuration:

| gate | `expected_pcc` | measured at stage 05 | headroom |
|---|---:|---:|---:|
| [`test_encoder.py`](../tests/pcc/test_encoder.py) `nuscenes_base`, 100×100, 6 layers | **0.997** | 0.999611 | **0.0026** |
| [`test_layer.py`](../tests/pcc/test_layer.py) `nuscenes_base`, 100×100 | 0.997 | — | — |
| [`test_spatial_cross_attention.py`](../tests/pcc/test_spatial_cross_attention.py) `nuscenes_base`, 100×100 | 0.999 | — | — |

So a lossy change is ranked against **0.997 at the encoder**, and the three error columns bound the
shape of the deviation, not just its correlation. Two rules follow, and neither is optional: report
PCC **per change** so a batch cannot hide which one spent the budget, and **do not relax a threshold
to make a change pass** — a lowered gate is the change failing, recorded as if it had succeeded. Note
the tighter per-module gates (SCA at 0.999) bind before the encoder's 0.997 does.

## Candidates

| # | candidate | targets | measured cost at baseline | effort | risk | status |
|--:|---|---|---|---|---|---|
| [1](#candidate-1--host-round-trips) | remove host round-trips from SCA | gap | **1917 ms** (62% of wall) | — | — | **device transfers done — [residue](#1g-what-is-still-host-side); [1e](#1e-an-empirical-high-water-mark-for-max_len)/[1f](#1f-derive-max_len-where-the-mask-is-produced) open** |
| [1a](#1a-rebatch-and-scatter-back-on-device) | rebatch + scatter-back on device | gap | **−2344.7 ms wall (−76%)** | L | med | **landed — [01](perf_reports/01-sca-rebatch-on-device.md)** |
| [1b](#1b-bound-max_len-statically) | bound `max_len` statically | gap, trace | +129 ms kernel to unlock ~9 ms gap | M | high | **[rejected](perf_reports/DEAD_ENDS.md#3-a-static-bound-on-max_len)** — [statistical variant unexplored](#a-statistical-bound-instead-of-a-worst-case-one) |
| [1c](#1c-hoist-index-computation-above-the-layer-loop) | rebatch plan once per frame, not per layer | gap | −94.6 ms encoder wall (−2.2%) | S | low | **landed — [02](perf_reports/02-rebatch-plan-hoisted.md)** |
| [1d](#1d-per-call-constant-uploads) | cache what is frame-invariant | gap | −56.4 ms encoder wall (−1.3%) | S | none | **landed — [03](perf_reports/03-constant-uploads-cached.md)** |
| [1e](#1e-an-empirical-high-water-mark-for-max_len) | grow `rebatch_len` monotonically instead of per-frame | gap, trace | ≤2.1 ms gap; unblocks 9 | M | med | todo |
| [1f](#1f-derive-max_len-where-the-mask-is-produced) | move the `max_len` reduce to the mask producer | — | 0 ms — structural | S | low | todo — sequence after 1e |
| [1g](#1g-what-is-still-host-side) | inventory of the surviving host work | gap | ~3.5 ms of layer gap | — | — | **inventory — no action on its own numbers** |
| [2](#candidate-2--fused-msda) | fused `multi_scale_deformable_attn` | kernel | **−191.6 ms kernel (−28.1%)** | M | med | **landed — [04](perf_reports/04-fused-msda.md)** |
| [3](#candidate-3--tile-padding-waste) | fold the offset normalizer into the Linear | kernel | **−32.7 ms kernel** | S | low | **landed — [05](perf_reports/05-offset-normalizer-folded.md)** |
| [4](#candidate-4--the-msda-concat) | replace the per-level concat | kernel | — | — | — | **moot — deleted by [04](perf_reports/04-fused-msda.md)** |
| [5](#candidate-5--data-movement-vs-compute) | shrink the data-movement / compute ratio | kernel | **220.8 ms** DM vs **220.7 ms** compute (~1:1) | M | low | todo — analysis |
| [6](#candidate-6--permutereshape-by-reformulation) | drop permute/reshape by reformulation or weight reorder | kernel | **139.8 ms** reshape+permute | M | med | todo |
| [7](#candidate-7--l1-vs-dram) | place operands in L1 instead of DRAM | kernel | unknown; expected small | S | low | todo — likely small |
| [8](#candidate-8--fuse-binaryng) | fuse `BinaryNg` into its producers | kernel | **48.2 ms** (10.6%) | M | med | todo |
| [9](#candidate-9--trace-capture) | trace capture the encoder | gap | ≤9 ms/layer | M | low | parked behind 1b/1e |
| [10](#candidate-10--msdaoperation-itself) | `MSDAOperation` device time | kernel | **167.8 ms** (36.7% of layer) | ? | ? | todo — upstream |
| [11](#candidate-11--absorb-msda-layout-prep) | absorb MSDA permute/reshape into the op | kernel | **~103 ms** per-level layout prep | L | med | todo — after 6 |
| [12](#candidate-12--one-fused-call-for-all-levels) | multi-level fused op: 4 SCA launches → 1 | kernel | 4× launch + 4× per-level prep | L | med | todo — upstream, with 10 |
| [13](#candidate-13--dtype-and-math-fidelity) | bfloat8_b weights, bfloat16 activations | kernel | unmeasured; **spends accuracy** | S | med | todo — needs an accuracy budget |

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

**1b gates candidate 9**, and nothing else in candidate 1 needs it. It is not the only way to gate it,
though: [1e](#1e-an-empirical-high-water-mark-for-max_len) reaches the same static
shape by learning the bound at runtime instead of deriving it, which is what makes the +129 ms
inapplicable. Read 1b as rejecting the *analytic* bound specifically.

#### A statistical bound instead of a worst-case one

**Unexplored, and the only line left in 1b that has not been measured.** What was rejected above is a
*sound* bound — one large enough that no frame can exceed it. Cost is linear in the bound, and a
sound bound has to cover the worst camera geometry the dataset can produce, which is where the
+129 ms comes from.

The alternative is to derive the bound **statistically for a fixed camera rig**, not analytically for
the worst case. `max_len` is a function of `lidar2img` and the BEV grid; for a given sensor setup
(nuScenes' six-camera ring, fixed intrinsics and mounting) it is a narrow distribution, not a wide
one — and [1e](#why-max_len-moves-at-all) now names the mechanism: intrinsics and extrinsics are
constant per rig, so the entire spread comes from the per-frame ego-motion compensation term. The experiment is cheap and does not need device time:

- [ ] Sweep the dataset, record `max_len` per frame per camera rig, and report the distribution —
      mean, p50, p99, max, and the spread across rigs. `build_rebatch_plan` already computes it;
      logging it costs nothing.
- [ ] If p99 sits well under the sound bound, price `rebatch_len = ceil_tile(p99)` — that is the
      kernel cost of the bound, and it is what decides whether the trade beats +129 ms.
- [ ] Decide what happens on the frames that exceed it. A static shape that is sometimes too small is
      only usable with an escape hatch: either fall back to the dynamic path for those frames (which
      keeps the readback and so keeps trace capture out of reach — pointless), or **drop** the
      overflow queries and accept that some BEV cells lose a camera's contribution on rare frames.
      The second is a numerical change and needs a PCC number per percentile, not an argument.

The third bullet is the one that decides this. A statistical bound is not a correctness-preserving
transform the way the rest of this backlog is, so it can only be justified against a measured
accuracy cost. Until that is measured, treat this as a research item, not an optimization.

Note the payoff is unchanged and small — trace capture is worth ≤9 ms of layer gap
([candidate 9](#candidate-9--trace-capture)), and the current host-fallback residue is ~3.5 ms
([1g](#1g-what-is-still-host-side)). This is a structural item (trace-capturability), not a
performance one.

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

### 1e. An empirical high-water mark for `max_len`

**The third option in the `max_len` problem, and the only one that is both sound and
correctness-preserving.** [1b](#1b-bound-max_len-statically) rejected an *analytic* bound (+129 ms of
kernel, and the naive version does not fit DRAM at all); the
[statistical bound](#a-statistical-bound-instead-of-a-worst-case-one) is cheap but drops queries on
overflow frames, so it is a numerical change. This one is neither: let the bound be **learned at
runtime and never shrink**.

#### Why `max_len` moves at all

Established by a separate analysis of the input pipeline, and it narrows the problem considerably.
`bev_mask` is a function of constant configuration plus `lidar2img`, and `lidar2img` decomposes into
three factors:

| factor | varies |
|---|---|
| camera intrinsics | **no** — fixed per sensor rig |
| camera extrinsics (mounting) | **no** — fixed per sensor rig, i.e. constant within a scene |
| ego-motion compensation | **yes** — per frame |

So the *only* source of frame-to-frame variation is the ego-motion compensation term, which exists
because the cameras do not expose simultaneously and each one's pose has to be corrected for where
the vehicle had moved by its own capture time. That is a small correction, not a free variable —
which is the mechanism behind the expectation that the distribution is narrow, and it is why a
high-water mark is expected to settle quickly rather than creep indefinitely.

It also says what a bound is a property *of*: the rig, not the dataset. Two rigs are two bounds.

#### Four routes, and which one this is

The same analysis enumerated the options. Recorded here so this entry is not re-derived:

1. **Dataset-wide upper bound per camera** — sweep every scene, take the max. Sound, but this is
   [1b](#1b-bound-max_len-statically) with measured input instead of analytic input, and it inherits
   1b's cost problem: the bound is set by the worst frame in the corpus.
2. **Intra-scene variance + a buffer** — measure how much `max_len` moves *within* a scene, then take
   the first frame's value plus a few percent. Cheap, and the ego-motion argument above says the
   spread should be small — but it can still be exceeded, so it needs the same overflow decision as
   the [statistical bound](#a-statistical-bound-instead-of-a-worst-case-one).
3. **Grow-only: re-cut the tensors only when `max_len` increases frame to frame.** This entry. The
   one route that never under-allocates.
4. **A combination** — seed the mark with route 2's per-scene estimate so the first frames do not pay
   a re-cut on every frame, then let route 3 handle the tail.

Route 3 is the one to build; route 4 is route 3 plus a warm-start and can be added later without
changing the mechanism. **Note the test dataset is not yet fixed**, so an acceptable interim is to
pass `max_len` in as an explicit parameter and record the dynamic behaviour as a known gap — which is
what this entry is.

#### The shape

`build_rebatch_plan` ([tt_spatial_cross_attention.py:90](../tt/tt_spatial_cross_attention.py#L90))
recomputes `rebatch_len` from scratch every frame. Instead, keep it on the module as a high-water
mark:

- compute `max_len` **on device** — `valid_per_cam` is `sum(-1) > 0 → sum(-1) → max`, all
  elementwise/reduce, no readback needed to produce it;
- compare it against the cached mark with `ttnn.gt` / `ttnn.lt`;
- **only when it grows**, read the scalar back and re-cut the shapes to the new
  `ceil_tile(max_len)`; otherwise keep the shape from the previous frame.

Cost is then the **observed** maximum over frames seen so far, not the analytic worst case — which is
why the +129 ms of 1b does not apply. Steady state after warm-up: the mark stops moving, and every
frame runs at a shape that has not changed since.

Note `rebatch_len` is already tile-rounded, so the mark only has to move in units of 32. Growth
within a tile costs nothing at all — a further reason to expect it to settle fast.

#### What it does not do

**It does not remove the host sync, and it does not remove the `torch.nonzero` loop.** Be precise
about why, because the naive framing of this idea overclaims both:

1. `ttnn.gt` returns a **device** tensor. Branching on it in Python still requires reading a scalar
   back, and the readback is a pipeline flush regardless of payload size. What shrinks is the
   *volume* — one scalar instead of the whole `bev_mask` — not the *flush*. The flush is what costs.
2. `max_len` stabilizing does **not** stabilize `query_ids`. Its *shape* becomes constant; its
   *contents* change every frame, because the valid query set moves with the ego-motion term above.
   So the per-frame index derivation
   ([sca:102-107](../tt/tt_spatial_cross_attention.py#L102-L107)) still runs. Only then does
   replacing that loop with `ttnn.nonzero` (contract in
   [1a](#1a-rebatch-and-scatter-back-on-device)) become worth doing — it is the remaining reason to
   move the mask to host at all, which is site 2 of the
   [host-fallback inventory](#1g-what-is-still-host-side).

So the direct gap payoff is bounded by the **2.1 ms** measured for sites 1–3 and is smaller than
that. [host_fallback_gap.md](host_fallback_gap.md#what-1e-can-and-cannot-take-from-this) tightens it
further: the readback shares one unsplittable stall with the `torch.nonzero` loop that 1e keeps, and
the whole block is **per frame** — ~0.66 ms per layer amortized over six layers, against 456.8 ms of
layer kernel. Do not sell this as a gap win.

#### Why it is worth doing anyway

**The payoff is [candidate 9](#candidate-9--trace-capture): a shape that is constant across replays.**
That is exactly what trace capture needs and exactly what 1a/1c did not deliver. A high-water mark
gives it without the +129 ms and without the accuracy cost, at the price of **invalidating the
captured trace whenever the mark grows** — rare after the first frames, but it has to be handled
rather than assumed away.

It also does not regress the DRAM ceiling the way 1b does: the shape never exceeds what today's
dynamic path already allocates for the frames seen so far, so the `bev_size=(200,200)` OOM
[documented in 1b](#1b-bound-max_len-statically) is unchanged, not made worse.

#### Work items

- [ ] Move the `max_len` reduce onto device ops and confirm it matches the host value exactly
      (integer counts through bfloat16 reductions — check, do not assume).
- [ ] Add the monotone cache and measure how many frames it takes for the mark to settle, **per
      scene and per rig** — the ego-motion argument predicts fast, and a slow settle would falsify
      it. This is the same dataset sweep the
      [statistical bound](#a-statistical-bound-instead-of-a-worst-case-one) needs; run it once and
      answer both, and report intra-scene spread separately from cross-scene spread since only the
      first is what route 2 above would key on.
- [ ] Price the steady-state shape against today's per-frame shape. If the settled mark is much above
      the per-frame median, this buys shape stability at a real kernel cost and has to be weighed
      against candidate 9's ≤9 ms.
- [ ] Only then: replace site 2's `torch.nonzero` loop with `ttnn.nonzero` and check whether the
      scalar readback can be deferred (compare-and-continue, resize on the *next* frame) — which
      would make the growth check asynchronous rather than a flush.

The last bullet is the interesting one and it is speculative: deferring the resize by one frame means
a frame occasionally runs with a shape one step too small, which is the statistical bound's overflow
problem again. Not free. Do not fold it in without deciding what happens to the dropped queries.

### 1f. Derive `max_len` where the mask is produced

**Structural, zero measured value on its own.** `point_sampling_3d_to_2d_ttnn` and
`build_rebatch_plan` are called back to back in
[tt_encoder.py:483-499](../tt/tt_encoder.py#L483-L499): the first produces `bev_mask`, the second
immediately reduces it to `max_len`. The reduce could live in the producer.

**It saves nothing today.** There is exactly one readback either way, one reduce either way, and
`bev_mask` is passed to `build_rebatch_plan` regardless because the index derivation needs
`valid_per_cam`, not just the scalar. This is code movement, not optimization — do not expect a
number from it.

What it is worth is deciding where the mask's *reductions* live once
[1e](#1e-an-empirical-high-water-mark-for-max_len) moves them onto the device. At that point
`valid_per_cam` (`sum(-1) > 0`) is consumed by three things — `max_len`, the `count` tensor
([sca:149](../tt/tt_spatial_cross_attention.py#L149)), and the index derivation — and computing it
once next to the mask that produces it is the natural home. Doing it *before* 1e fixes the layout of
code that 1e is about to rewrite.

- [ ] Sequence strictly after 1e. Alone it is a refactor with no measurement to justify it.
- [ ] If taken, return `valid_per_cam` (or the counts) alongside `bev_mask` rather than returning
      `max_len` — the scalar is the least useful of the three consumers' needs.
- [ ] `point_sampling_3d_to_2d_ttnn` is also where `lidar2img` enters, so it is the natural place to
      key a per-rig or per-scene warm start (route 4 in
      [1e](#four-routes-and-which-one-this-is)) if that is ever added.

### 1g. What is still host-side

**The device transfers are done; this is what survives.** The inventory of host work at stage 05,
what each item costs, and why it is still there. None of it is worth attacking on its own numbers
today — the one item with a reason to move is site 1, which is
[1e](#1e-an-empirical-high-water-mark-for-max_len).

Independently confirmed by [host_fallback_gap.md](host_fallback_gap.md), a stage-04 capture on a
different harness: it prices the same block at 2.245 ms (readback + index construction) and 1.417 ms
(scatter-index upload) against the 2.1 / 1.4 ms below. It also settles a question this inventory
raised and did not answer — **none of this is a TTNN `python_fallback`**; the CSV contains zero such
rows. "Host fallback" here means host transfer and host torch, charged as `OP TO OP LATENCY` on the
next device op.

Attribution is per-op `OP TO OP LATENCY` from the stage-05 measured region, both runs:
`2026_08_28_10_23_13` (93.4 ms gap) and `2026_08_28_10_30_24` (151.2 ms gap), segmented between the
`TTNN BEVFormerLayer Forward Start` / `End` signposts of the final layer forward.

#### The gap is region entry, not host fallback

Attributing the gap before listing the sites, because it changes what the sites are worth:

| what | run `10_23_13` | run `10_30_24` | share of that run's gap |
|---|---:|---:|---|
| region-entry smear (first 1–2 ops after the signpost) | 36.3 ms | 146.8 ms | **38.9% / 97.1%** |
| host-fallback-attributable (see table below) | ≤3.5 ms | 3.5 ms | **≤3.7% / 2.3%** |
| unattributed dispatch (SCA exit permute) | 0.7 ms | 0.7 ms | 0.8% / 0.5% |
| everything else — 120 of 127 ops | **0.00 ms** | **0.00 ms** | 0% |

Read the second run for the host-fallback figure. In `10_23_13` the entry smear landed on two ops
(`Clone` 6.6 ms, `BinaryNg` 29.7 ms); in `10_30_24` it landed almost entirely on one (`Clone` 6.6 ms,
`Matmul` 140.2 ms) and left the rebatch block clean at 2.1 ms. The 27.3 + 26.6 ms charged to the
rebatch block in `10_23_13` is the same smear finding different ops, not real host time — which is
why the host-fallback row reads "≤" for that run. Consistent with
[PERF.md](PERF.md#the-gap-column-carries-region-entry-cost): entry cost is 6.4–38.1 ms and lands on
whichever op is first in the queue.

**So: ~3.5 ms of the layer's gap belongs to host fallback — 2.3% of the reported gap, 0.8% of the
456.8 ms kernel.** 120 of 127 ops record exactly zero gap. Host dispatch is fully hidden behind
device time; there is no host-side win left at the layer level. The percentages below are against
run `10_30_24`'s 151.2 ms gap.

#### The sites

| # | site | what runs on host | frequency | gap | % of layer gap |
|--:|---|---|---|---:|---:|
| 1 | [sca:73-74](../tt/tt_spatial_cross_attention.py#L73-L74) | `to_torch(bev_mask)` → `max_len`. **The only device→host sync left in the encoder.** | per forward | *see 3* | *see 3* |
| 2 | [sca:102-107](../tt/tt_spatial_cross_attention.py#L102-L107) | `bs × num_cams` Python loop of `torch.nonzero` building `query_ids` | per forward | *see 3* | *see 3* |
| 3 | [sca:122-127](../tt/tt_spatial_cross_attention.py#L122-L127), [149-150](../tt/tt_spatial_cross_attention.py#L149-L150) | torch index arithmetic + `from_torch` of `ref_index` and `count` | per forward | **2.1** (1+2+3) | **1.4%** |
| 4 | [sca:138-146](../tt/tt_spatial_cross_attention.py#L138-L146) | `from_torch` of `scatter_index` (widened on device by `repeat`) | per forward | 1.4 | 0.9% |
| 5 | [sca:342](../tt/tt_spatial_cross_attention.py#L342) | `spatial_shapes.prod(dim=1).sum().item()` — assert only | per layer | 0.0 | 0% |
| 6 | [msda:201](../tt/tt_ms_deformable_attention.py#L201) | `tuple(spatial_shapes.flatten().tolist())` as the fold cache key | **per MSDA call** (5×/layer) | 0.0 | 0% |
| 7 | [msda:214-227](../tt/tt_ms_deformable_attention.py#L214-L227) | builds the fold scale on host, then 2 device `mul`s — see [candidate 3](#candidate-3--tile-padding-waste) | first call only, then cached | 0.0 | 0% |
| 8 | [msda:271](../tt/tt_ms_deformable_attention.py#L271), [289](../tt/tt_ms_deformable_attention.py#L289) | `spatial_shapes.prod(dim=1).sum()` for the signpost header and a shape check | per MSDA call | 0.0 | 0% |
| 9 | [tsa:159-168](../tt/tt_temporal_self_attention.py#L159-L168) | `torch.tensor([0])` + `tolist()` cache key for `level_start_index` | per layer | 0.0 | 0% |
| 10 | [encoder:466-470](../tt/tt_encoder.py#L466-L470) | `lidar2img` torch `stack` + `.cpu()` | per frame | n/a — outside the layer | — |
| 11 | [encoder:510](../tt/tt_encoder.py#L510) | `bev_shape = torch.tensor([[bev_h, bev_w]])`, rebuilt per layer | per layer | 0.0 | 0% |

Sites 1–4 are one host block — the profiler charges it to whichever op dispatches next, so they
cannot be separated further without instrumenting the block itself. Together they are the whole
measurable residue.

#### What this means for a future pass

- **Only site 1 is a sync.** Everything else is host-only Python/torch work that costs dispatch
  latency, not a pipeline flush. That is the distinction that matters: sites 5–11 are hidden behind
  device time and will stay hidden as long as the layer runs 456 ms of kernel. Removing them buys
  nothing measurable; removing site 1 is what unlocks trace capture, and that is
  [1b](#1b-bound-max_len-statically).
- **Sites 5, 6, 8, 9, 11 are per-layer or per-call re-derivations of constants.** They are free
  today and worth deleting only as a by-product of other work — e.g. if `spatial_shapes` ever moves
  to a device tensor, sites 5, 6 and 8 have to be reworked anyway.
- **Site 2 is the one that scales badly.** The `torch.nonzero` loop is `O(bs × num_cams)` Python
  iterations, invisible at `bs=1` and 6 cameras. It is the first thing to check if batch size or
  camera count goes up, and `ttnn.nonzero` (contract in [1a](#1a-rebatch-and-scatter-back-on-device))
  is the replacement — but only after site 1 stops forcing the readback, since while the mask is
  synced anyway the host loop is free.
- **Site 7 belongs to candidate 3** — see the [open item](#the-fold-runs-in-forward-not-in-init)
  there.

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
  That cost is now [candidate 10](#candidate-10--msdaoperation-itself).

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

**Landed** — [stage 05](perf_reports/05-offset-normalizer-folded.md), −32.7 ms kernel, PCC
unchanged at 0.999611. Beat its own rescoped estimate: ~24 ms was the SCA divide alone, and TSA
carried the same divide for another 9 ms.

### The fold runs in forward, not in init

**Open item, left deliberately.** The landed fold is lazy: `_folded_sampling_offsets`
([tt_ms_deformable_attention.py:188-230](../tt/tt_ms_deformable_attention.py#L188-L230)) is called
from `forward` ([L312](../tt/tt_ms_deformable_attention.py#L312)) and caches on
`tuple(spatial_shapes.flatten().tolist())`. So the first forward pays a host-side torch scale build
plus **two device `ttnn.mul`s over the weight and bias**, and every subsequent call pays a
`.tolist()` and a dict lookup. This is site 6/7 in the
[host-fallback inventory](#1g-what-is-still-host-side) — 0.0 ms of measured gap, so it is not a
performance problem; it is a structural one.

Why it is lazy: `spatial_shapes` is a **forward argument**, not config. `TTMSDeformableAttention`
does not know the feature-pyramid shapes at construction time — TSA is handed `bev_shape` and SCA the
real multi-level shapes, both per call. Moving the fold to init means giving the module the shapes at
init, which is a constructor-signature change across both attention modules and their callers.

To investigate:

- [ ] Are `spatial_shapes` genuinely constant per module instance for the whole model lifetime?
      They are config-derived (`nuscenes_base` feature pyramid) and `bev_shape` is grid geometry, so
      almost certainly yes — but the cache is keyed on contents precisely because nobody checked.
      Confirm from the call chain in [tt_encoder.py](../tt/tt_encoder.py), then the key can go.
- [ ] If constant: fold in `__init__` (or in
      [model_preprocessing.py](../tt/model_preprocessing.py), which is where the weight is already
      transposed and uploaded — arguably the right home, since the fold is a weight transform, not a
      runtime one). The two device muls then leave the model graph entirely.
- [ ] If not constant per instance: keep the cache, but hoist the key derivation so `.tolist()` runs
      once per forward rather than 5×.

The payoff is not device time — it is that the first forward stops differing from the rest, which
matters for anyone measuring cold-start or capturing a trace. Sequence it with
[candidate 9](#candidate-9--trace-capture), not against candidate 10.

### The original scoping

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

## Candidate 5 — data-movement vs compute

**Analysis first, not a fix.** After [stage 05](perf_reports/05-offset-normalizer-folded.md) the
layer is no longer host-bound and no longer concat/grid-sample-bound. What is left splits almost
evenly between moving tensors and computing on them:

| class | ops in the stage-05 table | kernel | % of 456.8 ms |
|---|---|---:|---:|
| data movement | ReshapeView, Permute, Untilize, Tilize, Slice, Transpose | **220.8 ms** | **48.3%** |
| compute | MSDAOperation, BinaryNg, Matmul | **220.7 ms** | **48.3%** |
| other | Scatter, rest | 15.3 ms | 3.3% |

Matmul is 1.0%. The compute side is the fused sampler and elementwise, not GEMM. The data-movement
side is layout: getting a tensor into the contract the next op accepts, then converting it back.
A change that deletes a permute and inserts an untilize has not moved this ratio.

The ratio is how you decide which of [6](#candidate-6--permutereshape-by-reformulation) /
[8](#candidate-8--fuse-binaryng) / [11](#candidate-11--absorb-msda-layout-prep) is worth doing, and it is
how you know you are done.

### How to measure it

Do not count ops. A cheap view and a 30 ms permute are both one row. Classify every row in the
stage-05 CSV (`2026_08_28_10_23_13`) and sum **kernel ms**:

1. **Tag each op** as compute, data-movement, or mixed. Compute is anything that reads values and
   writes a function of them (MSDA, BinaryNg, Matmul, Softmax, Embedding, ScatterAdd). Data-movement
   is layout/shape/memory only (ReshapeView, Permute, Transpose, Tilize, Untilize, Slice, Clone,
   Typecast). Mixed is the rest — Scatter is the one that matters here.
2. **Attribute each data-movement op to the compute that required it.** The useful question is not
   "how many permutes" but "which compute's contract forced this shuffle." Group by producer →
   consumer pair, using the signposts already in the layer (`TT MS Deformable Attn Module`,
   `MSDA Fused Core`, SCA rebatch). The ~103 ms of per-level prep around the fused op
   ([candidate 10](#candidate-10--msdaoperation-itself) / [11](#candidate-11--absorb-msda-layout-prep))
   should fall out as one group.
3. **Split necessary from accidental.** Necessary = the next compute's device-op validation
   rejects any other layout (the fused MSDA's ROW_MAJOR / INTERLEAVED / `(N, H, W, D)` contract is
   the current example). Accidental = the tensor was stored in an order that a weight reorder or a
   different reshape would have avoided ([candidate 6](#candidate-6--permutereshape-by-reformulation)).
   Only the accidental bucket is a model-side win.
4. **Optional: bound vs compute.** If a `noc.csv` / Tracy capture is available, mark each
   data-movement group as DRAM-bandwidth-bound or just launch-heavy. Bandwidth-bound groups are
   the ones that would move if the tensor lived in L1 ([candidate 7](#candidate-7--l1-vs-dram));
   launch-heavy groups are the ones that disappear only when the op count drops (fusion, [8](#candidate-8--fuse-binaryng)
   and [11](#candidate-11--absorb-msda-layout-prep)).
5. **Re-measure the ratio after every landed change.** The headline number is
   `data-movement_ms / compute_ms`. Stage 05 is ~1.00. A real win moves it down; a shuffle that
   trades Permute for Untilize leaves it where it is.

Work items:

- [ ] Produce the classified table from the stage-05 CSV, one row per op-code, with the
      producer→consumer grouping. That table is the input to 6, 8 and 11 — do not start those
      without it.
- [ ] Call out the largest accidental group. If it is the MSDA pre/post shuffle, 6 and 11 own it;
      if it is something else (SCA key/value permute, reference-point unsqueezes), write that down
      before inventing a kernel.

## Candidate 6 — permute/reshape by reformulation

**139.8 ms** is ReshapeView (77.1) + Permute (62.7) at stage 05 — 30.6% of the layer, and the
largest model-side item once [candidate 10](#candidate-10--msdaoperation-itself) is set aside as
upstream. Some of those reshapes are views. In ROW_MAJOR a trailing split of `H*W` into `(H, W)`
is free ([`_fused_msda_level`](../tt/tt_ms_deformable_attention.py#L48-L52) already relies on
that). The rest are real: they change which axis is tiled, or they move `num_heads` between
positions 1 and 2.

The question is whether the tensors can be **born** in the layout the consumer wants, so the
shuffle never runs. Same arithmetic, different storage order.

### Sites

The expensive shuffles sit on the fused-op boundary. Native BEVFormer layout is
`(bs, Q_or_K, num_heads, …)`; the op wants `num_heads` fused into the batch (`N = bs * num_heads`)
and `Q` in a different position:

| tensor | born as | op contract | today's shuffle |
|---|---|---|---|
| `value` | `(bs, H*W, heads, D)` after the Linear | `(N, H, W, D)` | permute `(0, 2, 1, 3)` → RM → reshape |
| `grid` | `(bs, Q, heads, L, P, 2)` | `(N, Q*P, 1, 2)` per level | slice L → permute `(0, 2, 1, 3, 4)` → reshape |
| `attn` | `(bs, Q, heads, L, P)` after softmax | `(N, Q, P)` per level | slice L → permute `(0, 2, 1, 3)` → reshape → RM |
| output | `(N, Q, D)` | `(bs, Q, embed)` into `output_proj` | reshape → TILE → permute `(0, 2, 1, 3)` → reshape |

Three Linears produce `value`, `sampling_offsets`, and `attention_weights`. Their weights already
go through [`preprocess_linear_weight`](../tt/model_preprocessing.py) (a transpose + upload). An
extra static permutation of the *output* channels — grouping by head rather than by query — is
the same class of transform as the offset-normalizer fold in
[candidate 3](#candidate-3--tile-padding-waste): done once, exact, no runtime op.

A reshape of a Linear's output that only splits a trailing multiple (`embed → (heads, D)`,
`heads*L*P → (heads, L, P)`) is a view in the right layout and is not the cost. The cost is the
**axis move** that follows it. If the Linear emits head-major, that axis move is gone.

### What a weight reorder cannot reach

**The untilize on `value` is forced by the op, not by BEVFormer's storage order.** `value` comes out
of `value_proj`, a matmul, so it is TILE; the fused op enforces ROW_MAJOR with `TT_FATAL`
([tt_ms_deformable_attention.py:43-44](../tt/tt_ms_deformable_attention.py#L43-L44)). The
`to_layout` at [L51](../tt/tt_ms_deformable_attention.py#L51) therefore survives every reorder this
candidate can make — no weight permutation makes a matmul emit ROW_MAJOR — and `value` is the ~92 MB
tensor in [candidate 7](#candidate-7--l1-vs-dram)'s size table.

So scope 6 to the permute at [L50](../tt/tt_ms_deformable_attention.py#L50), not the untilize behind
it. Killing that untilize needs the op to accept TILE input, which is
[candidate 11](#candidate-11--absorb-msda-layout-prep) route 1 — and this is the strongest argument
for route 1, stronger than the one 11 currently makes for itself.

### Free hoists in the same function, worth taking either way

Not reformulation, just work sitting inside the level loop that has no reason to be there
([`_fused_msda_level`](../tt/tt_ms_deformable_attention.py#L36-L70)):

- `attn`'s `to_layout(ROW_MAJOR)` runs **per level** ([L61](../tt/tt_ms_deformable_attention.py#L61))
  while `sampling_grids`' identical conversion is already hoisted above the loop
  ([L117](../tt/tt_ms_deformable_attention.py#L117)). Same asymmetry for the three `typecast` guards
  at [L63-68](../tt/tt_ms_deformable_attention.py#L63-L68).
- `grid` and `attn` are **sliced, then permuted**, per level. Permuting the full tensor once and
  slicing after is the same data through a quarter of the launches.

Same bytes, fewer launches. Whether that shows up depends on whether these groups are launch-bound
or bandwidth-bound — [candidate 5](#candidate-5--data-movement-vs-compute)'s optional noc pass is
what tells you, and this is a cheap place to test that classification before betting 11 on it.

### What this is not

This is not [candidate 11](#candidate-11--absorb-msda-layout-prep). 6 asks whether the shuffle can be
deleted by changing how weights and activations are stored. 11 asks what to do with the shuffle
that remains after that — fold it into a specialized kernel or into the fused op. Sequence 6
first. Writing a new reshape op for a permute that a weight reorder would have deleted is the
same mistake as ranking candidate 4 ahead of 2.

Work items:

- [ ] For each site above, write the output-channel permutation of the Linear that makes the
      native layout match the next consumer. `value_proj` and `output_proj` are the two that
      close a permute pair (heads forward, then heads back); `attention_weights` is one-way into
      the op. Confirm the softmax dim still lands on `L*P` after the reorder — if it does not,
      the reorder is not free.
- [ ] Check which of today's `ttnn.reshape` calls are already views (ROW_MAJOR, no padded-shape
      change on a tiled dim). Those are not in the 139.8 ms and must not be counted as a win
      when they disappear.
- [ ] Microbenchmark the remaining real permutes at the SCA shapes after the reorder. Whatever
      is left is the input to 11.

## Candidate 7 — L1 vs DRAM

**Expected small, and the reason is size.** Most of the tensors the layer actually spends time
on do not fit in usable L1. The exercise is a one-pass inventory so the idea is rejected (or
kept) with numbers, not intuition.

Wormhole L1 is ~1.5 MB per Tensix core; an N150 has 64 cores, but a large fraction is reserved
for kernels, circular buffers and allocator overhead. Interleaved L1 across the chip is tens of
MB at most, not hundreds. Against that:

| tensor | rough bytes (bf16) | fits usable L1? |
|---|---:|---|
| SCA `value` (`6 × ~30125 × 256`) | ~92 MB | no |
| TSA `query` (`10000 × 256`) | ~5 MB | maybe, tight |
| SCA rebatched `query` (`2484 × 256`) | ~1.3 MB | maybe |
| per-level `attn` (`N × Q × P`) | small | yes |
| `sampling_grids` at SCA | large (Q × heads × L × P × 2) | no |

A DRAM→L1 move only pays when the tensor is reused enough times to amortize the extra copy, or
when the consumer is bandwidth-bound on that operand and the working set already lives in L1.
The fused MSDA reads `value` / `grid` / `attn` once per call; the Linears read their weights
once. There is no obvious reuse that would turn a copy-up into a win on the large operands.

Work items:

- [ ] From the stage-05 CSV (or the device-op args), list each operand's byte size and current
      `BufferType` (DRAM vs L1). The classification in
      [candidate 5](#candidate-5--data-movement-vs-compute) already names the consumers.
- [ ] Keep only the operands that fit and are read more than once, or that feed a
      bandwidth-bound op. Price a `ttnn.to_memory_config(..., L1)` on those alone.
- [ ] If nothing survives the filter — the expected outcome — record it in
      [DEAD_ENDS](perf_reports/DEAD_ENDS.md) with the size table so this is not re-derived.

Do not hold any other candidate behind this one.

## Candidate 8 — fuse `BinaryNg`

**48.2 ms, 17 instances, 10.6% of the layer** after [stage 05](perf_reports/05-offset-normalizer-folded.md)
removed the two divides. What remains is add / mul / sub, not the tile-padded `div` candidate 3
already killed.

Known sites in the MSDA / attention path, none of them profiled in isolation:

| site | ops | fuses into |
|---|---|---|
| `query + query_pos` | 1 add, TSA and SCA | the first Linear that consumes `query` — but three Linears read the sum, so this is a hoist, not an epilogue, and it is already hoisted |
| `sampling_grids = loc * 2 - 1` | mul + sub | the preceding `ref + offsets` add: `2*(ref+off) - 1` is one scale-and-shift, or a static fold into the folded `sampling_offsets` weight (it already has the `1/[W,H]` scale) |
| `sampling_locations = ref + offsets` | 1 add | see the row above |
| per-level `ttnn.add` of fused-op outputs | 3 adds at SCA (4 levels) | an accumulate-into-output on the fused op, or a small add-tree kernel |
| residual `output + identity` | 1 add | epilogue of `output_proj` |

That is not yet 17. The rest live outside this file — SCA key/value path, encoder, FFN — and
have to come from the CSV, not from reading the attention module.

### How to look for fusion

Walk producer → consumer pairs, not op names:

1. Take every `BinaryNgDeviceOperation` row. If one operand is the previous op's output and the
   other is a broadcast / scalar / residual, it is a fusion candidate.
2. If the producer is a `Matmul` / `Linear`, the binary is an epilogue (bias-style add, residual
   add after `output_proj`). ttnn already fuses bias; residual is the one that is still a
   separate launch.
3. If the producer is another `BinaryNg`, it is a tree (`mul` then `sub` on `sampling_grids` is
   the current example). One kernel, or fold the constants away entirely.
4. Do not fuse across a layout change. A `BinaryNg` that sits between an Untilize and a Tilize
   is not an epilogue — the layout ops are the cost, and
   [candidate 5](#candidate-5--data-movement-vs-compute) / [6](#candidate-6--permutereshape-by-reformulation)
   own those, not this entry.
5. Check whether the binary is already a candidate for deletion by a weight fold, the way
   candidate 3 deleted the `div`. `* 2 - 1` on the sampling grid is the obvious next fold: the
   offset Linear already carries a static per-channel scale.

Work items:

- [ ] Inventory the 17 remaining `BinaryNg` rows from the stage-05 CSV: producer, both operands,
      shape, and whether a layout op sits between producer and binary. That list is the scope;
      do not fuse blindly from the table above.
- [ ] Price the `* 2 - 1` fold into the already-folded `sampling_offsets` weight. Exact
      (`2·(x + off) - 1` vs `(2x - 1)` after the add) and it removes two binaries plus,
      possibly, their tile-padding waste if the extent-2 coordinate axis is still one of the
      tiled dims.
- [ ] Residual-into-`output_proj` and per-level accumulate are the next two; sequence them after
      the inventory, not before.

## Candidate 9 — trace capture

Trace capture needs shapes that are static across replays, and `rebatch_len` is the one shape that is
not — it is redecided every frame. 1a and 1c removed the transfers and moved the last readback out of
the layer loop, but neither makes that shape constant. Two things would: 1b, which is
[rejected](perf_reports/DEAD_ENDS.md#3-a-static-bound-on-max_len) on its own terms, and
[1e](#1e-an-empirical-high-water-mark-for-max_len), which is not — at the price of
re-capturing the trace whenever the high-water mark grows.

So 9 is parked behind 1e now rather than dead behind 1b — and re-measurement has collapsed what it was worth. Per-layer gap is
**8.9 ms**, not the 218 ms this entry was written against. Trace capture cannot recover more than
that at the layer level, and against 1b's +129 ms of kernel the trade is not close.

Candidate 9 keeps a reason to exist at the *encoder* level, where per-forward host work is not
hidden behind device time — encoder wall is 4234.5 ms against 6 × 691 = 4146 ms of steady-state
layer time, so ~90 ms sits outside the layers. That case has to be made on the encoder harness and nobody
has measured it. Treat 9 as low-value, not as blocked high-value.

Both perf harnesses document the current blocker in their module docstrings; update them when it
lifts.

## Candidate 10 — `MSDAOperation` itself

New, created by [stage 04](perf_reports/04-fused-msda.md). The fused op is **the single largest
cost in the layer**: 167.8 ms total, 143.3 ms of it the four SCA calls at ~35.9 ms each. After
[stage 05](perf_reports/05-offset-normalizer-folded.md) it is **36.7% of layer kernel time** and
more than three times the next model-side item.

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
tilize↔untilize churn around each call is not obviously necessary and is model-side work. That
prep is now [candidate 5](#candidate-5--data-movement-vs-compute) (measure),
[candidate 6](#candidate-6--permutereshape-by-reformulation) (delete by storage order), and
[candidate 11](#candidate-11--absorb-msda-layout-prep) (fold what remains into a kernel). This entry
owns the fused op's device time, not the shuffle around it.

## Candidate 11 — absorb MSDA layout prep

New, and last, because it is a kernel — the same class of work as
[candidate 10](#candidate-10--msdaoperation-itself), aimed at the shuffle that 10 left on the
table. Do not start it before [candidate 6](#candidate-6--permutereshape-by-reformulation) has tried to
delete the permutes by changing how the tensors are stored. Whatever 6 cannot remove is the
scope here.

[`_fused_msda_level`](../tt/tt_ms_deformable_attention.py#L36-L70) and the post-loop pack
([L137-L140](../tt/tt_ms_deformable_attention.py#L137-L140)) are the site. Before each of the
five fused calls (1 TSA + 4 SCA):

```
value:  permute (0, 2, 1, 3) → to_layout RM → reshape (N, H, W, D)
grid:   slice level → permute (0, 2, 1, 3, 4) → reshape (N, Q*P, 1, 2)
attn:   slice level → permute (0, 2, 1, 3) → reshape (N, Q, P) → to_layout RM
```

and after the level loop:

```
output: reshape (bs, heads, Q, D) → to_layout TILE → permute (0, 2, 1, 3) → reshape (bs, Q, embed)
```

[Stage 04](perf_reports/04-fused-msda.md) priced the per-level half at **~103 ms**. The post-loop
half is in the remaining Reshape / Permute / Tilize total (77.1 + 62.7 + 17.1 ms) and has not
been split out. Together they are the reason data-movement still matches compute
([candidate 5](#candidate-5--data-movement-vs-compute)) after the fused op deleted the old
sample-weight-reduce tail.

Two routes, in the order they should be tried:

Route 1 is worth more than this entry originally argued: [candidate 6](#what-a-weight-reorder-cannot-reach)
establishes that the untilize of the ~92 MB `value` tensor **cannot** be deleted by any weight
reorder, because a matmul emits TILE and the op demands ROW_MAJOR. Accepting TILE input is the only
thing that removes it. And [candidate 12](#candidate-12--one-fused-call-for-all-levels) wants a
contract change to the same op — take both to upstream in one conversation.

1. **Relax the fused op's input/output contract** so it accepts the native BEVFormer layouts —
   `(bs, H*W, heads, D)`, `(bs, Q, heads, P, 2)`, `(bs, Q, heads, P)` in, `(bs, Q, embed)` out —
   and does the head-batch merge, the `H*W → (H, W)` split, and the output pack inside the
   kernel. One launch, no intermediate tensors. This is the better landing if the op can take
   it: the shuffle is addressing, not arithmetic, and the kernel already touches every element.
2. **A specialized MSDA-reshape op** that takes the three native tensors and emits the three
   contract tensors (and the inverse on the way out) in one pass. Fallback if the fused op
   cannot grow its contract — e.g. because VADv2 and other callers depend on today's shapes.
   Cheaper to land than a contract change, but it is still a launch and still writes the
   shuffled tensors out, so it only wins if today's many small permutes/untilizes are
   launch-bound rather than bandwidth-bound. Candidate 5's optional noc pass is what decides
   that.

Either route has to keep the fused op's hard constraints (INTERLEAVED, bfloat16, `D % 16 == 0`,
`grid.shape[1] == Q*P`) or move them to the new wrapper. Do not silently drop them.

Work items:

- [ ] Wait on 6. If the Linear reorder deletes the heads-axis permutes, this entry rescopes to
      the leftover `to_layout` / slice / output pack — a smaller kernel, possibly not worth
      writing.
- [ ] If 6 leaves the shuffles: microbenchmark route 1 vs route 2 vs today's decomposition at
      the SCA per-level shapes (`N=48, Q=2496, P=4, D=32`, four levels). Gate on PCC against
      the current fused path, not against torch — this is a layout change, not a math change.
- [ ] Check `vadv2` before picking a contract. If it already feeds the op in today's layout,
      a new reshape op (route 2) does not break it; a contract change (route 1) needs a
      compatible default.

## Candidate 12 — one fused call for all levels

**Untracked until now, and it may be the largest single item left.** SCA runs **four**
`multi_scale_deformable_attn` launches, one per pyramid level
([tt_ms_deformable_attention.py:122-132](../tt/tt_ms_deformable_attention.py#L122-L132)), because the
op is `num_levels == 1` only. That one constraint costs three things at once:

| what | why the level count drives it |
|---|---|
| 4 launches instead of 1 | [candidate 10](#candidate-10--msdaoperation-itself) already shows the op carries real launch/packing overhead — +45% per sample vs bare `grid_sample`, and VADv2's measured `N*Q >= 1024` floor |
| **4× the per-level prep** | the ~103 ms in [candidate 11](#candidate-11--absorb-msda-layout-prep) is *per level*; one level is one prep |
| 3 `ttnn.add`s | combining the per-level outputs — row 4 of [candidate 8](#candidate-8--fuse-binaryng) |

### Why it fell through the cracks

[Candidate 2](#candidate-2--fused-msda) recorded it as blocker 1 ("a per-level call plus a host-side
weighted sum … or a follow-up to the op"), took the per-level route, and nothing inherited the
follow-up. 10 owns the op's device time but scopes itself to the single-level kernel; 11 asks the op
to relax its *layout* contract but never its *level* contract. Three entries, and the item sat
between all of them.

### What the op would have to grow

The reference MSDA formulation flattens all levels into one value tensor and indexes with
`level_start_index`; this op instead takes `(N, H, W, D)`, so the level count is baked into the
operand shape. A multi-level contract means either a list of value tensors plus their shapes, or the
flattened value plus `level_start_index` and `spatial_shapes` — the second is what mmcv does and what
`sampling_locations` is already normalized for. `attention_weights` needs no change at all: it is
already softmaxed jointly over `L*P` ([stage 04](perf_reports/04-fused-msda.md) established that the
joint sum decomposes exactly), so a multi-level kernel just stops decomposing it.

This is **upstream** work, same as 10, and it should be sequenced with 10's microbenchmark: if the
launch overhead is what 10 suspects, collapsing four launches into one is where that finding cashes
out. File them together, with one set of numbers.

- [ ] Fold into 10's microbenchmark: measure 4 single-level calls at SCA's per-level shapes against
      one call over the same total sampled points. That delta is what 12 is worth, and it is
      measurable **before** any op change.
- [ ] Only if the delta is real: raise the multi-level contract upstream, together with 11's
      layout-contract ask. Two contract changes to one op are one negotiation, not two.
- [ ] Check in-tree callers before proposing a signature change — `vadv2` runs `num_levels == 1` and
      must keep working.

## Candidate 13 — dtype and math fidelity

**Unpriced, and the first item on this list that spends accuracy.** Everything today is bfloat16:
`DEFAULT_DTYPE` in [model_preprocessing.py](../tt/model_preprocessing.py) sets weights and biases,
and every activation follows. The working hypothesis from the precision discussion is
**bfloat16 activations + bfloat8_b weights + HiFi2**, validated afterwards against PCC and device
time.

### One part of that hypothesis is already true

**BEVFormer's matmuls already run HiFi2.** No call site passes `compute_kernel_config` — none of the
seven `ttnn.linear` / `ttnn.matmul` sites carries one — and with bf16 inputs, no `program_config` and
no user core grid, matmul defaults to `MathFidelity::HiFi2`
([matmul_device_operation.cpp:2798-2799](../../../../ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp#L2798-L2799)).
So "move to HiFi2" is a no-op here, not a change. Setting it explicitly is still worth doing — as
documentation, and because the default is conditional — but do not book a win against it.

**And there is a trap in the same code.** Fidelity drops to `LoFi` when `are_inputs_low_precision_df`
holds, which requires **both** operands to be bfloat8_b/bfloat4_b. bf8_b weights against bf16
activations keeps HiFi2, which is the hypothesis. But if activations are ever lowered too, fidelity
silently drops to LoFi in the same step — two precision changes landing as one, with one PCC number
between them. Pin `compute_kernel_config` explicitly before touching activation dtype.

### Where the win would actually come from

Not from matmul. **Matmul is 1.0% of the layer** — the arithmetic is not the cost, and that has been
true since the baseline. The reason to care about dtype is
[candidate 5](#candidate-5--data-movement-vs-compute): **48.3% of the layer is data movement**, which
is bytes, and bfloat8_b is half the bytes of bfloat16. Every Permute, Untilize, Tilize and Slice on a
bf8_b tensor moves half as much.

The catch is the fused op, which requires **bfloat16** (`TT_FATAL`, see
[candidate 2's contract](#op-contract)). `value`, `grid` and `attn` must be bf16 at the op boundary,
so a bf8_b `value` buys a cheaper permute and then pays a typecast — and the typecast reads and
writes the whole tensor, which is roughly what the permute cost in the first place. **Expect that to
cancel.** The honest scope is weights, plus the tensors that never reach the op: the FFN pair and
`output_proj`.

So: small, and worth pricing precisely because "half the bytes on 48% of the layer" sounds much
bigger than it is going to be.

### Work items

- [ ] Set `compute_kernel_config` explicitly at all seven matmul sites at today's dtypes. Expect
      **zero** change — that is the point. It pins the conditional default and makes every later
      comparison honest.
- [ ] Flip `DEFAULT_DTYPE` for weights only (bfloat8_b), keep activations bfloat16, measure PCC and
      kernel time. This is the cheap half of the hypothesis and it is one line; weights are small, so
      expect a small number.
- [ ] Per-tensor, not global: list every tensor that never crosses the fused op's bf16 boundary and
      price bf8_b on those alone. A global flip either hits the op's `TT_FATAL` or silently inserts
      typecasts — both make the measurement meaningless.
- [ ] Report PCC **per change**, never for the batch, against the gates in
      [`tests/pcc/`](../tests/pcc/) — encoder 0.997, SCA 0.999, and the abs/rel/high-error columns
      alongside them (see [accuracy budget](#candidates)). Stage 05 measures 0.999611, so the encoder
      headroom is 0.0026 and the per-module gates bind first. Do not relax a threshold to make a
      dtype change pass.

Do not sequence any other candidate behind this one. It is independent of 5/6/11 and, unlike them, it
is not correctness-preserving.

---

## Ordering

1. ~~**1a**~~ — landed, −2344.7 ms.
2. ~~**1b**~~ — [rejected](perf_reports/DEAD_ENDS.md#3-a-static-bound-on-max_len): +129 ms of kernel
   to unlock ~9 ms of gap, and candidate 2 owns the memory ceiling that caps it.
3. ~~**1c**~~ — landed, −94.6 ms encoder wall.
4. ~~**1d**~~ — landed, −56.4 ms encoder wall. **The device transfers are done** — the surviving host
   work is inventoried in [1g](#1g-what-is-still-host-side) and is worth ~3.5 ms of layer gap, 0.8%
   of kernel. Nothing there is actionable on its own numbers; what is left of candidate 1 is the
   `max_len` shape, not the transfers.
5. ~~**2**~~ — landed, −191.6 ms kernel (−28.1%). **Was ordered after 4 and should not have been.**
6. ~~**4**~~ — never run; [deleted by 2](perf_reports/04-fused-msda.md).
7. ~~**3**~~ — landed, −32.7 ms kernel. Taken before 10 because it sits upstream of the fused region
   and its value does not depend on what 10 concludes.
8. **1e** — the empirical high-water mark on `max_len`. Ranked here because it is the only route to a
   static `rebatch_len` that costs neither +129 ms nor accuracy, and it is what makes 9 answerable.
   Its own gap payoff is ≤2.1 ms — rank it on the trace-capturability, not the milliseconds.
9. **1f** — refactor only, and only as a by-product of 1e. Never worth doing on its own.
10. **5** — classify the remaining 456.8 ms as data-movement vs compute and attribute each shuffle
    to the compute that required it. Cheap, and it is the input to 6 / 8 / 11. Do not start those
    without the table.
11. **6** — drop permute/reshape by emitting tensors in the consumer's layout (weight reorder /
    reformulation). 139.8 ms on the table; sequence before 11 so a new kernel is not written for
    a shuffle a storage change would delete.
12. **8** — fuse the remaining 48.2 ms of `BinaryNg` into producers, starting with the inventory
    and the `* 2 - 1` fold. Independent of 6; can run in parallel once 5 has named the sites.
13. **7** — L1 vs DRAM. Expected small; do not block anything on it. Reject-with-numbers or keep
    the operands that actually fit.
14. **10 + 12** — **167.8 ms, 36.7% of the layer**, and four launches where one would do. One
    microbenchmark answers both: per-sample cost against bare `grid_sample`, and 4 single-level
    calls against one call over the same points. Independent of 5–8, and the answer likely belongs
    upstream rather than in this model.
15. **11** — absorb whatever shuffle 6 could not delete into the fused op or a specialized
    MSDA-reshape kernel. After 6, and after 5's optional noc pass has said whether that shuffle
    is launch-bound or bandwidth-bound. Its route 1 also carries 6's forced untilize and 12's level
    contract — one upstream conversation, three asks.
16. **13** — dtype. Independent of everything above and cheap to try, but it is the first lossy
    change on the list, so it is ranked last on purpose: land the correctness-preserving work first
    and spend the 0.0026 of encoder [PCC headroom](#candidates) knowingly. Note one third of the
    hypothesis (HiFi2) is already the running default.
17. **9** — needs 1b or 1e, and is worth ≤9 ms/layer rather than the 218 ms first claimed. Only revisit if
   an encoder-harness measurement shows per-forward host time the layer profile does not — and that
   harness would have to fix the gap column first (see [PERF.md](PERF.md#the-gap-column-is-not-reliable)).

Parked, not ranked — no device-time case, kept because they are structural:

- [**a statistical `max_len` bound**](#a-statistical-bound-instead-of-a-worst-case-one) — needs a
  dataset sweep and a PCC-vs-percentile curve, not device time. Not correctness-preserving, so it
  cannot be ranked against the rest of this list. It is no longer the *only* untried route to
  trace-capturability — [1e](#1e-an-empirical-high-water-mark-for-max_len) is the correctness-
  preserving one, and it needs the same sweep. Run the sweep once for both.
- [**fold the offset normalizer at init**](#the-fold-runs-in-forward-not-in-init) — moves two device
  muls and a per-call `.tolist()` out of `forward`. Zero measured gap; the point is that the first
  forward stops differing from the rest.
- [**the host-fallback residue**](#1g-what-is-still-host-side) — ~3.5 ms of layer gap across four
  sites in one host block, plus seven sites at exactly 0.0 ms. Revisit if batch size or camera count
  grows: site 2's `torch.nonzero` loop is the only item that scales with either.

The baseline settled what was previously a guess: host round-trips dominated wall clock 4:1 over
kernel time, and within kernel time it is layout churn, not arithmetic — matmul is 0.7%. Nothing in
the matmul-tuning playbook applies here. Stage 01 inverted the ratio; stage 04 then took −28% of
kernel by deleting the layout churn around the sampler rather than by making arithmetic faster.

**Kernel is 97% of layer wall clock.** The fused op is 36.7% of it and is still the largest single
item ([candidate 10](#candidate-10--msdaoperation-itself)), but it is no longer the only live
question: data-movement and compute are now ~1:1, and the model-side half of that (permute,
reshape, BinaryNg, the MSDA pre/post shuffle) is [5](#candidate-5--data-movement-vs-compute)–[8](#candidate-8--fuse-binaryng)
and [11](#candidate-11--absorb-msda-layout-prep). 10 is an op-level question; those five are not.

Four lessons this backlog got wrong and should not repeat:

- **Sequence rewrites before the cleanups inside them.** 4 was ranked first for being cheap; 2
  deleted it.
- **Grep the tree for prior art before deriving an op contract.** `vadv2` had a working
  `multi_scale_deformable_attn` call site, a measured shape floor, and the offset-normalizer fold —
  all three relevant, none referenced here until stage 04.
- **A blocker deferred inside a landed entry needs a home, or it disappears.** 2 recorded the
  single-level limitation, shipped the per-level workaround, and no entry inherited it; 10 and 11
  each assumed the other owned it. It is [12](#candidate-12--one-fused-call-for-all-levels) now, and
  it may be the largest item left. When a candidate lands *around* a constraint, open the entry for
  the constraint in the same commit.
- **Read the op's defaults before proposing to change them.** HiFi2 was proposed as a change;
  matmul has been running HiFi2 all along, by a conditional default no call site overrides
  ([13](#candidate-13--dtype-and-math-fidelity)). Verify the current value, then propose.
