# BEVFormer encoder — performance optimization candidates

The **backlog**. Measured results live in [PERF.md](PERF.md) and [`perf_reports/`](perf_reports/);
when a candidate lands, its status here becomes `landed` with a link to its report and the detail
moves there. Rejected items with their numbers are in
[perf_reports/DEAD_ENDS.md](perf_reports/DEAD_ENDS.md).

The layer is **280.2 ms kernel** after [stage 10](perf_reports/10-value-head-split-unpadded.md),
from 655.6 ms + 2416.5 ms gap at the [baseline](perf_reports/00-baseline.md). Costs quoted against
the baseline are historical. Gap and wall figures are not comparable across runs —
[PERF.md](PERF.md#the-gap-column-is-not-reliable).

**Accuracy budget.** Everything landed so far is correctness-preserving: PCC moved 0.999608 →
**0.999651**, i.e. it improved. Two items on the list are not correctness-preserving —
[1b's statistical sibling](#a-statistical-bound-instead-of-a-worst-case-one) drops queries, and
[candidate 13](#candidate-13--dtype-and-math-fidelity) spends mantissa.

The gates live in [`tests/pcc/`](../tests/pcc/), per module and per config, each case carrying
`expected_pcc` plus `expected_abs_error`, `expected_rel_error` and `expected_high_error_ratio`. For
the profiled configuration the binding gates are **encoder 0.997** (measured 0.999651, headroom
0.0026) and **SCA 0.999** — the per-module gates bind first. Two rules, neither optional: report PCC
**per change**, so a batch cannot hide which change spent the budget; and **never relax a threshold
to make a change pass** — a lowered gate is the change failing, recorded as if it had succeeded.

## Candidates

| # | candidate | ticket | issue | targets | measured | effort | risk | status |
|--:|---|---|---|---|---|---|---|---|
| [1](#candidate-1--host-round-trips) | remove host round-trips from SCA | `01` | [#55191](https://github.com/tenstorrent/tt-metal/issues/55191) | gap | **1917 ms** (62% of baseline wall) | — | — | **transfers done — [residue](#1g-what-is-still-host-side); [1e](#1e-an-empirical-high-water-mark-for-max_len)/[1f](#1f-derive-max_len-where-the-mask-is-produced) open** |
| [1a](#1a-rebatch-and-scatter-back-on-device) | rebatch + scatter-back on device | `01.01` | [#55192](https://github.com/tenstorrent/tt-metal/issues/55192) | gap | **−2344.7 ms wall (−76%)** | L | med | **landed — [01](perf_reports/01-sca-rebatch-on-device.md)** |
| [1b](#1b-bound-max_len-statically) | bound `max_len` statically | — | — | gap, trace | +129 ms kernel for ~9 ms of gap | M | high | **[rejected](perf_reports/DEAD_ENDS.md#3-a-static-bound-on-max_len)** — but its [DRAM argument is gone](#the-memory-half-of-the-rejection-no-longer-holds) |
| [1c](#1c-hoist-index-computation-above-the-layer-loop) | rebatch plan once per frame, not per layer | `01.02` | [#55193](https://github.com/tenstorrent/tt-metal/issues/55193) | gap | −94.6 ms encoder wall (−2.2%) | S | low | **landed — [02](perf_reports/02-rebatch-plan-hoisted.md)** |
| [1d](#1d-per-call-constant-uploads) | cache what is frame-invariant | `01.03` | [#55194](https://github.com/tenstorrent/tt-metal/issues/55194) | gap | −56.4 ms encoder wall (−1.3%) | S | none | **landed — [03](perf_reports/03-constant-uploads-cached.md)** |
| [1e](#1e-an-empirical-high-water-mark-for-max_len) | grow `rebatch_len` monotonically, never shrink | `01.02.01` | [#55195](https://github.com/tenstorrent/tt-metal/issues/55195) | trace | ≤2.1 ms gap; unblocks 9 | M | med | todo — rank on trace-capturability, not ms |
| [1f](#1f-derive-max_len-where-the-mask-is-produced) | move the `max_len` reduce to the mask producer | `01.02.02` | [#55196](https://github.com/tenstorrent/tt-metal/issues/55196) | — | 0 ms — structural | S | low | todo — strictly after 1e |
| [1g](#1g-what-is-still-host-side) | inventory of the surviving host work | — | — | gap | ~3.5 ms of layer gap | — | — | **inventory — no action on its own numbers** |
| [2](#candidate-2--fused-msda) | fused `multi_scale_deformable_attn` | `05` | [#55198](https://github.com/tenstorrent/tt-metal/issues/55198) | kernel | **−191.6 ms kernel (−28.1%)** | M | med | **landed — [04](perf_reports/04-fused-msda.md)** |
| [3](#candidate-3--tile-padding-waste) | fold the offset normalizer into the Linear | `02` | [#55197](https://github.com/tenstorrent/tt-metal/issues/55197) | kernel | **−32.7 ms kernel** | S | low | **landed — [05](perf_reports/05-offset-normalizer-folded.md)** |
| [4](#candidate-4--the-msda-concat) | replace the per-level concat | — | — | kernel | — | — | — | **moot — deleted by [04](perf_reports/04-fused-msda.md)** |
| [5](#candidate-5--data-movement-vs-compute) | classify the layout churn, then delete it by shape and order | `03` | [#55202](https://github.com/tenstorrent/tt-metal/issues/55202) | kernel | **−176.3 ms (−38.6%)**, 456.5 → 280.2 ms | — | — | **closed — all six landed, [result](#result--candidate-5-is-closed)** |
| [5a](#5a-delete-the-key-permute) | delete the dead SCA `key` permute | `03.01` | [#55203](https://github.com/tenstorrent/tt-metal/issues/55203) | kernel | **−18.1 ms (−4.0%)** | XS | none | **landed — [06](perf_reports/06-sca-key-permute-deleted.md)** |
| [5b](#5b-sampling-location-math-in-row_major) | sampling-location math in ROW_MAJOR; `2/[W,H]` and `2·ref−1` folded | `03.02` | [#55204](https://github.com/tenstorrent/tt-metal/issues/55204) | kernel | **−82.2 ms (−18.8%)**; PCC **improved**; cleared the 200×200 OOM | M | low | **landed — [07](perf_reports/07-sampling-grid-in-row-major.md)** |
| [5c](#5c-prepare-attn-once-not-per-level) | prepare `attn` once per call, not per level | `03.03` | [#55205](https://github.com/tenstorrent/tt-metal/issues/55205) | kernel | **−44.9 ms (−12.6%)** | S | low | **landed — [08](perf_reports/08-attn-prepared-once-per-call.md)** |
| [5d](#5d-value-head-split-without-the-padding) | `value` head split without the tile padding | `03.05` | [#55207](https://github.com/tenstorrent/tt-metal/issues/55207) | kernel | **−6.6 ms (−2.3%)** — benchmarked at 0.84× first | M | med | **landed — [10](perf_reports/10-value-head-split-unpadded.md)** |
| [5e](#5e-head-major-grid-with-a-tile-transpose) | build the grid head-major with a TILE transpose | `03.04` | [#55206](https://github.com/tenstorrent/tt-metal/issues/55206) | kernel | **−24.5 ms (−7.9%)** — deletes all five per-level grid permutes | S | low | **landed — [09](perf_reports/09-head-major-sampling-grid.md)** |
| [5f](#5f-the-per-level-dtype-guards-are-free) | the per-level dtype guards | — | — | kernel | **0 ms** — zero `Typecast` rows | XS | none | **verified-zero — no code change** |
| [6](#candidate-6--permutereshape-by-reformulation) | **[6a](#6a-hoist-the-sca-camera-permute-out-of-the-layer-loop)** only: hoist the SCA camera permute out of the layer loop | `03.06` | [#55208](https://github.com/tenstorrent/tt-metal/issues/55208) | kernel | **19.3 ms/layer**, ~96 ms/frame — runs 6× on identical data | S | low | **re-scoped — the weight reorder is [dead](#why-the-weight-reorder-is-dead); the rest is [11](#candidate-11--absorb-msda-layout-prep)'s** |
| [7](#candidate-7--l1-vs-dram) | place operands in L1 instead of DRAM | `03.07` | [#55209](https://github.com/tenstorrent/tt-metal/issues/55209) | kernel | unknown; expected small | S | low | todo — likely small |
| [8](#candidate-8--fuse-binaryng) | fuse `BinaryNg` into its producers | — | — | kernel | **2.9 ms (1.0%)** at stage 10, was 48.2 | — | — | **closed — [5b](#5b-sampling-location-math-in-row_major) deleted the cost instead of fusing it** |
| [9](#candidate-9--trace-capture) | trace capture the encoder | `06` | [#55210](https://github.com/tenstorrent/tt-metal/issues/55210) | gap | ≤9 ms/layer | M | low | parked behind 1e |
| [10](#candidate-10--msdaoperation-itself) | `MSDAOperation` device time | `05.01` | [#55199](https://github.com/tenstorrent/tt-metal/issues/55199) | kernel | **167.9 ms — 59.9% of the layer** | ? | ? | todo — upstream, the largest item left |
| [11](#candidate-11--absorb-msda-layout-prep) | absorb MSDA permute/reshape into the op | `05.02` | [#55200](https://github.com/tenstorrent/tt-metal/issues/55200) | kernel | **~40 ms** ([inventory](#the-inventory-at-stage-10)) | L | med | todo — after 6a; **no longer gated on 6** |
| [12](#candidate-12--one-fused-call-for-all-levels) | multi-level fused op: 4 SCA launches → 1 | `05.03` | [#55201](https://github.com/tenstorrent/tt-metal/issues/55201) | kernel | 4× launch + 4× per-level prep | L | med | todo — upstream, with 10 |
| [13](#candidate-13--dtype-and-math-fidelity) | bfloat8_b weights, bfloat16 activations | `07` | [#55211](https://github.com/tenstorrent/tt-metal/issues/55211) | kernel | unmeasured; **spends accuracy** | S | med | todo — needs an accuracy budget |
| [14](#candidate-14--sfpu-sampling-geometry) | move MSDA sampling geometry from the reader to the SFPU | `05.04` | [#55231](https://github.com/tenstorrent/tt-metal/issues/55231) | kernel | **167.6 → 29.5 ms** on the profiling branch (N150) | M | med | todo — **the largest single item; obsoletes 10's premise** |
| [15](#candidate-15--axes-as-addresses) | address MSDA head/level by byte offset instead of materializing them | `05.05` | [#55232](https://github.com/tenstorrent/tt-metal/issues/55232) | kernel | layout plumbing **60.5 → 15.4 ms** (N150) | L | med | todo — four additive op-contract changes |
| [15a](#candidate-15--axes-as-addresses) | fold the grid point axis into its page | `05.05.01` | [#55233](https://github.com/tenstorrent/tt-metal/issues/55233) | kernel | −12.9 ms (N150) | S | low | todo |
| [15b](#candidate-15--axes-as-addresses) | value head by byte offset (`num_heads`) | `05.05.02` | [#55234](https://github.com/tenstorrent/tt-metal/issues/55234) | kernel | −20.5 ms (N150) | M | med | todo |
| [15c](#candidate-15--axes-as-addresses) | attn level run by byte offset | `05.05.03` | [#55235](https://github.com/tenstorrent/tt-metal/issues/55235) | kernel | −17.9 ms (N150) | M | med | todo — 32-byte NoC alignment |
| [15d](#candidate-15--axes-as-addresses) | rank-3 packed grid (head + level) | `05.05.04` | [#55236](https://github.com/tenstorrent/tt-metal/issues/55236) | kernel | −4.6 ms (N150) | M | med | todo — after 15a |

Ordering rationale is at the [bottom](#ordering).

---

## Candidate 1 — host round-trips

`ttnn.to_torch` is a device→host readback that flushes the op queue, blocks, and makes trace capture
impossible; the matching `from_torch` pushes back down. At the baseline these were not setup costs —
several sat inside per-layer, per-camera loops, and the profile showed **a single 1.917 s gap** on
the first op after the SCA rebatch loop: two-thirds of the layer's wall clock in one Python loop.

### 1a. Rebatch and scatter-back on device

**Landed** — [stage 01](perf_reports/01-sca-rebatch-on-device.md), −2344.7 ms wall (−76%), PCC
unchanged. Both gathers run as one `ttnn.embedding` each and cost 0.27 ms combined; the scatter-back
is one `ttnn.scatter_add` at 10.50 ms. The `TODO: … once TTNN supports required indexing ops`
comments were stale — **there was no missing op**, only a data-dependent shape (1b).

The op contracts, verified against the installed `ttnn` rather than documentation, because
[1e](#1e-an-empirical-high-water-mark-for-max_len) still needs them:

| need | op | constraints |
|---|---|---|
| row gather | `ttnn.embedding(indices, weight)` | `weight` ROW_MAJOR bfloat16 with `padded_shape[0] == padded_shape[1] == 1`; `indices` UINT32/BFLOAT16, ROW_MAJOR needs `padded_shape[1] == padded_shape[2] == 1`; both INTERLEAVED. Output `[B, 1, S, E]`; a **tilized** output also needs `E % 32 == 0` *and* `S % 32 == 0`. Takes a `padding_idx`. |
| generic gather | `ttnn.gather(input, dim, index)` | index must match input rank and the output takes the index's shape, so the row id materializes across `E`. Works, wastes bandwidth — and measured **800× slower** here ([DEAD_ENDS 2](perf_reports/DEAD_ENDS.md#2-ttnngather-for-the-reference-point-rebatch)). |
| scatter accumulation | `ttnn.scatter_add(input, dim, index, src)` | `index` INT32/UINT8/UINT16/UINT32; input, index, src on device and **not sharded**; `input.dtype == src.dtype`; output ROW_MAJOR. Torch semantics, so `index` and `src` share a shape. |
| mask → indices | `ttnn.nonzero(input)` | **1D, ROW_MAJOR only.** Returns `(count, indices)`, `indices` padded to input length, only the first `count` meaningful — and `count` is a **device** tensor. |

### 1b. Bound `max_len` statically

**Rejected** — [DEAD_ENDS 3](perf_reports/DEAD_ENDS.md#3-a-static-bound-on-max_len) has the data.
`max_len` is a data-dependent shape, so it forces the `bev_mask` readback that keeps the encoder from
being trace-capturable. A bound is derivable — the coverage ratio is a stable camera-FOV property —
but cost is exactly linear in it: **+129 ms of kernel against the ~9 ms of gap** trace capture could
return, and a frame exceeding the bound silently drops queries.

Read 1b as rejecting the *analytic* bound specifically.
[1e](#1e-an-empirical-high-water-mark-for-max_len) reaches the same static shape by learning the
bound at runtime, which is what makes the +129 ms inapplicable.

#### The memory half of the rejection no longer holds

1b was rejected on cost **and** on a DRAM ceiling — the naive bound did not fit at all, and SCA at
`bev_size=(200,200)` failed with `Out of Memory: … 2969567232 B DRAM buffer`.
[Stage 04](perf_reports/04-fused-msda.md) tested whether the fused op relieved it and it did not.
**[Stage 07](perf_reports/07-sampling-grid-in-row-major.md) did**, by deleting the tile-padded
`(bs*Q*heads, L, P, 2)` intermediate that was the allocation — verified by bisection, and
`tests/pcc/` went from 32 passed / 1 failed to **33 passed**.

So the ceiling is gone and the +129 ms argument stands alone. It has **not** been re-priced against a
280 ms layer; do that before quoting 1b as rejected on cost.

#### A statistical bound instead of a worst-case one

**Unexplored, and not correctness-preserving.** What 1b rejected is a *sound* bound, large enough
that no frame can exceed it — which has to cover the worst camera geometry the dataset can produce.
The alternative is to derive it **statistically for a fixed camera rig**: `max_len` is a function of
`lidar2img` and the BEV grid, and [1e](#why-max_len-moves-at-all) names the mechanism — intrinsics
and extrinsics are constant per rig, so the whole spread is the per-frame ego-motion term. Cheap to
test, no device time needed:

- [ ] Sweep the dataset; record `max_len` per frame per rig; report mean, p50, p99, max and the
      spread across rigs. `build_rebatch_plan` already computes it — logging it costs nothing.
- [ ] If p99 sits well under the sound bound, price `rebatch_len = ceil_tile(p99)`. That is the
      kernel cost of the bound and it is what decides whether the trade beats +129 ms.
- [ ] **Decide what happens on frames that exceed it.** Falling back to the dynamic path keeps the
      readback and so keeps trace capture out of reach — pointless. **Dropping** the overflow queries
      is a numerical change and needs a PCC number per percentile, not an argument.

The third bullet decides this. Until it is measured, treat this as a research item, not an
optimization — and note the payoff is small either way: trace capture is worth ≤9 ms of layer gap.

### 1c. Hoist index computation above the layer loop

**Landed** — [stage 02](perf_reports/02-rebatch-plan-hoisted.md), −94.6 ms encoder wall (read as
~1–2%), no numerical change. More was invariant than the entry claimed: not just the index derivation
but the **entire reference-point rebatch**, which never touches `query` — six identical gathers were
computed and discarded. The structural result is worth more than the 2%: five of six `bev_mask`
readbacks are gone and the layer loop no longer contains one.

### 1d. Per-call constant uploads

**Landed** — [stage 03](perf_reports/03-constant-uploads-cached.md), −56.4 ms encoder wall,
−25 device ops per layer, layer gap to 8.9 ms. Five sites, all deriving something fixed by config or
by the BEV grid inside a per-layer path; the report has the table. Two caches are keyed on **tensor
identity** rather than contents, which is what makes them hit across forwards and not just across
layers. A steady-state win — the first frame still pays.

### 1e. An empirical high-water mark for `max_len`

**The only route to a static `rebatch_len` that is both sound and correctness-preserving.**
[1b](#1b-bound-max_len-statically) rejected an analytic bound; the
[statistical bound](#a-statistical-bound-instead-of-a-worst-case-one) drops queries. This one does
neither: let the bound be **learned at runtime and never shrink**.

#### Why `max_len` moves at all

`bev_mask` is a function of constant configuration plus `lidar2img`, and `lidar2img` decomposes into
camera intrinsics (**fixed per rig**), camera extrinsics/mounting (**fixed per rig**, so constant
within a scene) and ego-motion compensation (**per frame**). So the *only* source of frame-to-frame
variation is the ego-motion term, which exists because the cameras do not expose simultaneously and
each pose is corrected for where the vehicle had moved by its own capture time.

That is a small correction, not a free variable — which is why the distribution is expected to be
narrow and a high-water mark to settle quickly rather than creep. It also says what a bound is a
property *of*: **the rig, not the dataset.** Two rigs are two bounds.

#### Four routes, and which one this is

1. **Dataset-wide max per camera** — sound, but this is 1b with measured input instead of analytic
   input, and inherits its cost problem: the bound is set by the worst frame in the corpus.
2. **Intra-scene variance + a buffer** — cheap, and the ego-motion argument says the spread is small,
   but it can still be exceeded, so it needs the statistical bound's overflow decision.
3. **Grow-only: re-cut only when `max_len` increases.** This entry — the one route that never
   under-allocates.
4. **Combination** — seed the mark with route 2's per-scene estimate, then let route 3 handle the
   tail. Route 3 plus a warm start; addable later without changing the mechanism.

**Note the test dataset is not yet fixed**, so an acceptable interim is to pass `max_len` in as an
explicit parameter and record the dynamic behaviour as a known gap.

#### The shape

`build_rebatch_plan` ([sca:90](../tt/tt_spatial_cross_attention.py#L90)) recomputes `rebatch_len`
every frame. Instead keep it on the module as a high-water mark: compute `max_len` **on device**
(`valid_per_cam` is `sum(-1) > 0 → sum(-1) → max`, all elementwise/reduce), compare against the
cached mark with `ttnn.gt`, and **only when it grows** read the scalar back and re-cut to
`ceil_tile(max_len)`. Cost is then the **observed** maximum over frames seen so far, not the analytic
worst case — which is why 1b's +129 ms does not apply. `rebatch_len` is already tile-rounded, so the
mark moves only in units of 32 and growth within a tile costs nothing.

#### What it does not do

The naive framing of this idea overclaims twice:

1. **It does not remove the host sync.** `ttnn.gt` returns a *device* tensor; branching on it in
   Python still reads a scalar back, and the readback is a pipeline flush regardless of payload size.
   What shrinks is the volume, not the flush — and the flush is what costs.
2. **It does not remove the `torch.nonzero` loop.** `max_len` stabilizing makes the *shape* constant,
   not the *contents*: the valid query set moves every frame with the ego-motion term, so the
   per-frame index derivation ([sca:102-107](../tt/tt_spatial_cross_attention.py#L102-L107)) still
   runs. Only then is replacing it with `ttnn.nonzero` worth doing.

So the gap payoff is bounded by the **2.1 ms** measured for sites 1–3 of
[1g](#1g-what-is-still-host-side) and is smaller than that.
[host_fallback_gap.md](host_fallback_gap.md#what-1e-can-and-cannot-take-from-this) tightens it
further: the readback shares one unsplittable stall with the `nonzero` loop that 1e keeps, and the
whole block is **per frame** — ~0.66 ms per layer amortized. **Do not sell this as a gap win.**

#### Why it is worth doing anyway

**The payoff is [candidate 9](#candidate-9--trace-capture): a shape constant across replays** —
exactly what trace capture needs and exactly what 1a/1c did not deliver, without 1b's +129 ms and
without an accuracy cost. The price is **invalidating the captured trace whenever the mark grows** —
rare after the first frames, but it has to be handled rather than assumed away. It also cannot
regress DRAM: the shape never exceeds what the dynamic path already allocates for frames seen so far.

- [ ] Move the `max_len` reduce onto device ops and confirm it matches the host value **exactly** —
      integer counts through bfloat16 reductions; check, do not assume. Note host and device already
      [disagree by up to 12 queries](pcc_drop_after_deterministic_lidar2img.md#secondary-and-still-open).
- [ ] Add the monotone cache and measure how many frames the mark takes to settle, **per scene and
      per rig**. The ego-motion argument predicts fast; a slow settle falsifies it. This is the same
      sweep the [statistical bound](#a-statistical-bound-instead-of-a-worst-case-one) needs — run it
      once, answer both, and report intra-scene spread separately from cross-scene spread.
- [ ] Price the settled shape against today's per-frame shape. If the mark settles well above the
      per-frame median, this buys shape stability at a real kernel cost.
- [ ] Only then: replace the `torch.nonzero` loop with `ttnn.nonzero`, and check whether the scalar
      readback can be deferred (compare-and-continue, resize on the *next* frame) to make the growth
      check asynchronous. **Speculative and not free** — a deferred resize means a frame occasionally
      runs one step too small, which is the overflow problem again.

### 1f. Derive `max_len` where the mask is produced

**Structural, zero measured value on its own.** `point_sampling_3d_to_2d_ttnn` and
`build_rebatch_plan` are called back to back
([tt_encoder.py:483-499](../tt/tt_encoder.py#L483-L499)): the first produces `bev_mask`, the second
immediately reduces it. There is one readback either way and one reduce either way, and `bev_mask` is
passed to `build_rebatch_plan` regardless because the index derivation needs `valid_per_cam`, not the
scalar. **Code movement, not optimization.**

Its value is deciding where the mask's reductions live once 1e moves them onto the device — at that
point `valid_per_cam` feeds three consumers (`max_len`, the `count` tensor, the index derivation) and
computing it next to the mask is the natural home.

- [ ] Sequence **strictly after 1e**. Alone it is a refactor with no measurement to justify it.
- [ ] If taken, return `valid_per_cam` (or the counts) alongside `bev_mask` rather than `max_len` —
      the scalar is the least useful of the three consumers' needs.
- [ ] `point_sampling_3d_to_2d_ttnn` is where `lidar2img` enters, so it is also the natural place to
      key a per-rig warm start (route 4 above).

### 1g. What is still host-side

**The device transfers are done; this is what survives, and none of it is worth attacking on its own
numbers.** Attribution is per-op `OP TO OP LATENCY` from the stage-05 runs `2026_08_28_10_23_13` and
`2026_08_28_10_30_24`.

**~3.5 ms of the layer's gap is host-attributable — 2.3% of the reported gap, 0.8% of kernel — and
120 of 127 ops record exactly zero gap.** The rest of the reported gap is region-entry smear
(36.3 ms and 146.8 ms in the two runs), which lands on whichever op is first in the queue; see
[PERF.md](PERF.md#the-gap-column-carries-region-entry-cost). Host dispatch is otherwise fully hidden
behind device time.

| # | site | what runs on host | frequency | gap |
|--:|---|---|---|---:|
| 1 | [sca:73-74](../tt/tt_spatial_cross_attention.py#L73-L74) | `to_torch(bev_mask)` → `max_len`. **The only device→host sync left in the encoder.** | per forward | *see 3* |
| 2 | [sca:102-107](../tt/tt_spatial_cross_attention.py#L102-L107) | `bs × num_cams` Python loop of `torch.nonzero` building `query_ids` | per forward | *see 3* |
| 3 | [sca:122-127](../tt/tt_spatial_cross_attention.py#L122-L127), [149-150](../tt/tt_spatial_cross_attention.py#L149-L150) | torch index arithmetic + `from_torch` of `ref_index` and `count` | per forward | **2.1** (1+2+3) |
| 4 | [sca:138-146](../tt/tt_spatial_cross_attention.py#L138-L146) | `from_torch` of `scatter_index` | per forward | **1.4** |
| 5–9 | [sca:342](../tt/tt_spatial_cross_attention.py#L342), [msda:201](../tt/tt_ms_deformable_attention.py#L201), [msda:271](../tt/tt_ms_deformable_attention.py#L271)/[289](../tt/tt_ms_deformable_attention.py#L289), [tsa:159-168](../tt/tt_temporal_self_attention.py#L159-L168) | `spatial_shapes` asserts and signpost headers, `.tolist()` cache keys | per layer or per call | **0.0** |
| 10–11 | [encoder:466-470](../tt/tt_encoder.py#L466-L470), [encoder:510](../tt/tt_encoder.py#L510) | `lidar2img` `stack`/`.cpu()`; `bev_shape` rebuilt per layer | per frame / per layer | **0.0** |

Sites 1–4 are one host block — the profiler charges it to whichever op dispatches next, so they
cannot be separated further without instrumenting the block. Independently confirmed by
[host_fallback_gap.md](host_fallback_gap.md), a stage-04 capture on a different harness: 2.245 ms and
1.417 ms against the 2.1 / 1.4 ms here. That capture also settles what this inventory could not:
**none of this is a TTNN `python_fallback`** — the CSV contains zero such rows. "Host fallback" here
means host transfer and host torch charged as latency on the next device op.

What this means for a future pass:

- **Only site 1 is a sync.** Everything else costs dispatch latency, not a pipeline flush, and stays
  hidden as long as the layer runs hundreds of ms of kernel. Removing site 1 is what unlocks trace
  capture — [1e](#1e-an-empirical-high-water-mark-for-max_len).
- **Sites 5–11 are per-layer re-derivations of constants.** Free today; delete them only as a
  by-product of other work — e.g. if `spatial_shapes` ever becomes a device tensor they all have to
  be reworked anyway.
- **Site 2 is the one that scales badly.** `O(bs × num_cams)` Python iterations, invisible at `bs=1`
  and 6 cameras. First thing to check if batch size or camera count grows.

---

## Candidate 2 — fused MSDA

**Landed** — [stage 04](perf_reports/04-fused-msda.md), −191.6 ms kernel (−28.1%), PCC 0.999611.
`GridSample` and `Concat` are both gone from the profile. Replaces the hand-rolled decomposition with
`ttnn.experimental.multi_scale_deformable_attn`
([PR #52380](https://github.com/tenstorrent/tt-metal/pull/52380)).

Two things the entry got wrong, corrected by measurement:

- **SCA does not need a host-side weighted sum.** `attention_weights` is softmaxed jointly over `L*P`
  and thereafter only summed, so the joint sum decomposes exactly into per-level sums — four fused
  calls plus an L-way `ttnn.add` on device, exact.
- **The fused op is slower than the sampling it replaces** — 24.4 vs 16.8 ms at TSA, 143.3 vs 99.2 ms
  at SCA. The entire win is deleting the `stack`/`mul`/`sum` tail it makes unnecessary. That cost is
  now [candidate 10](#candidate-10--msdaoperation-itself).

Prior art that would have saved the derivation: `models/experimental/vadv2/tt/tt_utils.py` already
ran this op, with the tensor-shuffle recipe and a measured `N*Q >= 1024` floor.

### Op contract

Kept here because [11](#candidate-11--absorb-msda-layout-prep) and
[12](#candidate-12--one-fused-call-for-all-levels) both negotiate against it, and
[13](#candidate-13--dtype-and-math-fidelity) is blocked by it.

```
ttnn.experimental.multi_scale_deformable_attn(value, grid, attn, *,
                                              memory_config=None, align_corners=False)
```

`value` `(N, h_in, w_in, D)` with `N = B * num_heads` · `grid` `(N, Q*P, 1, 2)` normalized to
`[-1, 1]` · `attn` `(N, Q, P)` · returns `(N, Q, D)`.

Hard constraints from
[multi_scale_deformable_attn_device_operation.cpp](../../../../ttnn/cpp/ttnn/operations/experimental/multi_scale_deformable_attn/device/multi_scale_deformable_attn_device_operation.cpp):
all three inputs **and** the output INTERLEAVED (no sharding), **all bfloat16, all ROW_MAJOR**,
`D % 16 == 0` (holds: `embed_dims=256`, `num_heads=8` → `D=32`), `grid.shape[1] == Q * P`, and
**`num_levels == 1` only** — which is [candidate 12](#candidate-12--one-fused-call-for-all-levels).
`align_corners=False` matches mmcv, which the reference implementation uses; do not flip it without a
PCC check.

---

## Candidate 3 — tile padding waste

**Landed** — [stage 05](perf_reports/05-offset-normalizer-folded.md), −32.7 ms kernel, PCC unchanged.
Beat its own rescoped estimate: ~24 ms was the SCA divide alone and TSA carried the same divide for
another 9 ms. [Stage 07](perf_reports/07-sampling-grid-in-row-major.md) later folded the `2×` grid
rescale into the same constant.

### The fold runs in forward, not in init

**Open item, left deliberately.** `_folded_sampling_offsets`
([msda:188-230](../tt/tt_ms_deformable_attention.py#L188-L230)) is called from `forward` and caches on
`tuple(spatial_shapes.flatten().tolist())`, so the first forward pays a host scale build plus **two
device `ttnn.mul`s over the weight and bias**, and every later call pays a `.tolist()` and a dict
lookup. Zero measured gap — it is a structural problem, not a performance one.

Why it is lazy: `spatial_shapes` is a **forward argument**, not config. `TTMSDeformableAttention` does
not know the pyramid shapes at construction — TSA is handed `bev_shape`, SCA the real multi-level
shapes, both per call. Moving the fold to init is a constructor-signature change across both
attention modules and their callers.

- [ ] Are `spatial_shapes` genuinely constant per module instance for the model's lifetime? They are
      config-derived and `bev_shape` is grid geometry, so almost certainly yes — but the cache is
      keyed on contents precisely because nobody checked. Confirm from the call chain, then the key
      can go.
- [ ] If constant: fold in `__init__`, or better in
      [model_preprocessing.py](../tt/model_preprocessing.py) where the weight is already transposed
      and uploaded — the fold is a weight transform, not a runtime one. The two muls then leave the
      graph entirely.
- [ ] If not: keep the cache but hoist the key derivation so `.tolist()` runs once per forward
      rather than 5×.

The payoff is that the first forward stops differing from the rest, which matters for cold-start
measurement and for trace capture. Sequence with [candidate 9](#candidate-9--trace-capture).

## Candidate 4 — the MSDA concat

**Moot — deleted by [stage 04](perf_reports/04-fused-msda.md).** The fused op returns `(N, Q, D)`
already reduced over sampling points, so there is no per-level list to stack: `Concat` went
115.5 ms → **0**, and the 74.9 ms reshape after it went with it. It had been the single most expensive
op in the layer at 113.5 ms.

**The lesson is the ordering, not the op.** This entry was ranked *before* candidate 2 for being cheap
and self-contained. That was wrong — 2 subsumed it, and doing 4 first would have been discarded work.
**When one candidate rewrites the region another lives in, sequence the rewrite first.**

## Candidate 5 — data-movement vs compute

**Closed.** The entry was "classify the CSV, then decide what 6 / 8 / 11 are worth." The
classification produced six shape and ordering changes, all landed, none needing a new kernel, an
upstream change, or any accuracy.

### Result — candidate 5 is closed

| item | predicted | measured | report |
|---|---:|---:|---|
| [5a](#5a-delete-the-key-permute) `key` permute deleted | −19.3 | `03.01` | [#55203](https://github.com/tenstorrent/tt-metal/issues/55203) | **−18.1** | [06](perf_reports/06-sca-key-permute-deleted.md) |
| [5b](#5b-sampling-location-math-in-row_major) sampling grid in ROW_MAJOR | −90 | `03.02` | [#55204](https://github.com/tenstorrent/tt-metal/issues/55204) | **−82.2** | [07](perf_reports/07-sampling-grid-in-row-major.md) |
| [5c](#5c-prepare-attn-once-not-per-level) `attn` prepared once | −35 | `03.03` | [#55205](https://github.com/tenstorrent/tt-metal/issues/55205) | **−44.9** | [08](perf_reports/08-attn-prepared-once-per-call.md) |
| [5e](#5e-head-major-grid-with-a-tile-transpose) head-major grid in TILE | −13 | `03.04` | [#55206](https://github.com/tenstorrent/tt-metal/issues/55206) | **−24.5** | [09](perf_reports/09-head-major-sampling-grid.md) |
| [5d](#5d-value-head-split-without-the-padding) `value` head split unpadded | −10…−20 | `03.05` | [#55207](https://github.com/tenstorrent/tt-metal/issues/55207) | **−6.6** | [10](perf_reports/10-value-head-split-unpadded.md) |
| [5f](#5f-the-per-level-dtype-guards-are-free) per-level guards | 0 | — | — | **0** (verified) | — |

**456.5 → 280.2 ms, −176.3 ms, −38.6%.** Ops 127 → 106. PCC 0.999611 → **0.999651** — it improved,
because 5b removed two bf16 rounding steps per coordinate. `tests/pcc/` went from 32 passed / 1 failed
to **33 passed**: 5b cleared the 200×200 DRAM OOM.

The ratio this entry existed to track:

| class | stage 05 | stage 10 |
|---|---:|---:|
| data movement | 220.8 ms | **90.8 ms** |
| compute | 220.7 ms (of which ~46 was padding) | **178.8 ms** |
| Scatter | 15.3 ms | 10.5 ms |
| **ratio** | 1.00 as measured, 1.51 corrected | **0.51** |

Compute barely moved — 178.8 against the ~175 ms the entry predicted was the real figure once padded
`BinaryNg` was reclassified. **Every one of the 176 ms came off the data-movement side**, and
`MSDAOperation` alone is 167.9 of the 178.8 ms of compute, i.e. **59.9% of the layer**.

### What this entry got wrong, and what it taught

Three of the five estimates were wrong in ways that were systematic, not random:

1. **"Op count" was the wrong frame.** The entry opened asking for fewer launches. 5b *added* five ops
   and took 82 ms off; 5d removed six and took 6.6 ms. What is paid for is **padded bytes touched**.
2. **"A ROW_MAJOR reshape is a free view" is false in ttnn** unless the last dimension is unchanged.
   Stage 07 shipped a 7.09 ms "view" and stage 08 a 2.59 ms one. The replacement rule, from
   [stage 09](perf_reports/09-head-major-sampling-grid.md): **do axis moves in TILE — a transpose
   swaps whole tiles, ~0.4 ms — and in ROW_MAJOR do only elementwise work and constant-row-width row
   regrouping.** 5d is the exception that shows the rule's scope: a ROW_MAJOR permute won there only
   because reaching the TILE-transposable shape needed a 4×-padded re-layout of 92 MB.
3. **The one entry flagged "may lose" was the one that needed measuring, and measuring worked.** 5d
   was benchmarked as two whole chains on device before a line was written — 0.84× and bit-identical,
   predicting −5.6 ms against −6.6 measured. Its own guess had been −10…−20.

### The six items

Bodies are in the reports; these are the one-line summaries and the sites.

#### 5a. Delete the `key` permute

19.3 ms per layer of work whose result was never read: SCA built `key_reshaped`
([sca:348-352](../tt/tt_spatial_cross_attention.py#L348-L352)) and passed it as `key=`, but
`TTMSDeformableAttention.forward` has no `key` parameter — it landed in `**kwargs`. And since the
encoder sets `value = key`, it was the same tensor permuted twice.
**Landed: −18.1 ms** ([06](perf_reports/06-sca-key-permute-deleted.md)).

#### 5b. Sampling-location math in ROW_MAJOR

104.4 ms across SCA and TSA computing on tensors tile-padded 128× — trailing axes of extent 4 and 2
in the two tiled dimensions. Moved onto the `(bs, Q, 256)` shape the Linear emits, with `2/[W,H]`
folded into the weight and `2·ref − 1` into a precomputed bias so one add replaces add/mul/sub.
**Landed: −82.2 ms, `BinaryNg` 49.3 → 2.3 ms, PCC improved, 200×200 OOM cleared**
([07](perf_reports/07-sampling-grid-in-row-major.md)).

#### 5c. Prepare `attn` once, not per level

43.0 ms to feed the op a 0.5 MB tensor: slicing a tile-padded `(4, 4)` tail forced three of four
levels through an untilize/slice/re-tilize round trip. Hoisted above the level loop, leaving one
slice per level. **Landed: −44.9 ms, ops 131 → 113** ([08](perf_reports/08-attn-prepared-once-per-call.md)).

#### 5e. Head-major grid with a TILE transpose

The entry's own proposal — permute once above the loop — was worth ~4 ms, because the per-level
permute already runs on a sliced tensor. What landed instead moves the head axis **while the tensor is
still tiled**, which deletes all five per-level permutes.
**Landed: −24.5 ms** ([09](perf_reports/09-head-major-sampling-grid.md)).

#### 5d. `value` head split without the padding

Splitting 256 channels into `(heads, head_dim)` inside TILE puts `heads = 8` in a tiled dimension and
re-lays out ~92 MB at 4× its real volume. Folding heads into the row axis instead leaves the padding
alone. **Landed: −6.6 ms** ([10](perf_reports/10-value-head-split-unpadded.md)); the remaining 33.8 ms
of `Permute` is not reachable by reordering and belongs to
[11](#candidate-11--absorb-msda-layout-prep).

#### 5f. The per-level dtype guards are free

**Verified, no code change.** The stage-10 CSV has **zero `Typecast` rows** in the layer and zero in
the whole capture, so the three guards at
[msda:63-68](../tt/tt_ms_deformable_attention.py#L63-L68) never fire. They stay: they are what stands
between a future [dtype change](#candidate-13--dtype-and-math-fidelity) and a `TT_FATAL` from inside
the device op. Do not count their removal as a win.

### Follow-ups this entry created

- **[1b](#1b-bound-max_len-statically)'s memory argument is dead** — 5b removed the allocation that
  was the DRAM ceiling. The +129 ms cost argument stands and has not been re-priced.
- **[Candidate 6](#candidate-6--permutereshape-by-reformulation) is re-scoped and
  [8](#candidate-8--fuse-binaryng) is closed.** 6's premise — that a channel reorder could delete the
  shuffles — is dead: a channel permutation cannot move a channel past a row axis, which is what every
  surviving shuffle does. 8's 48.2 ms was padding 5b deleted, not arithmetic to fuse.
- **The harness's per-layer `build_rebatch_plan`** (6.8 ms at stage 05) is in every layer number here;
  the encoder hoists it. Fix or subtract it before quoting these as encoder-path figures.

### What this entry no longer needs

The optional `noc.csv` pass. It was there to decide whether the data-movement groups are
bandwidth-bound or launch-heavy. It was not needed for 5a–5e — those groups were neither, they were
**padding**. It is still the right tool for what remains, which is genuinely bandwidth: the 33.8 ms of
`Permute` and 13.1 ms of `Untilize` on the `value` operand, all of it
[11](#candidate-11--absorb-msda-layout-prep)'s.

## Candidate 6 — permute/reshape by reformulation

**Re-scoped to one item after candidate 5 landed.** The entry's premise — that a static reorder of the
Linears' output channels could delete the shuffles — is **dead**, checked against the stage-10 profile
(`2026_09_02_11_49_26`) rather than argued. Permute + ReshapeView was 139.8 ms at stage 05 and is
**61.8 ms** (33.8 + 28.0), 22.0% of a 280.2 ms layer.

### Why the weight reorder is dead

The reorder permutes a Linear's **output channels**. Every surviving shuffle moves the **row axis**:

- `value` and `grid` both need `num_heads` in front of the row axis (`N = bs*heads`). A channel
  permutation cannot move a channel past a row — that is a transpose, and no static weight transform
  expresses it. [5e](#5e-head-major-grid-with-a-tile-transpose) solved it for `grid` by doing the
  transpose in TILE, a *scheduling* change rather than a storage one; `value` cannot follow, because
  reaching a TILE-transposable shape costs a 4×-padded re-layout of 92 MB
  ([stage 10](perf_reports/10-value-head-split-unpadded.md)).
- The remaining reshapes change **row width** — `256 → 32`, `32 → 2`, `16 → 4` — because the op's
  operands are 2 and `head_dim` wide while the Linears emit 256-wide rows. Row width is set by the op
  contract, not by channel order.

So the transform class that worked for [candidate 3](#candidate-3--tile-padding-waste) does not apply
to anything left here.

### The inventory at stage 10

| op | row | ms | what | owner |
|---|--:|---:|---|---|
| Permute | 437 | **19.26** | SCA camera permute, `[num_cams, L, bs, E] → [bs, num_cams, L, E]`, TILE | **[6a](#6a-hoist-the-sca-camera-permute-out-of-the-layer-loop)** |
| | 442 | **13.16** | `value` head permute, ROW_MAJOR | [11](#candidate-11--absorb-msda-layout-prep) route 1 |
| | 394 | 0.71 | the same at TSA | [11](#candidate-11--absorb-msda-layout-prep) route 1 |
| | 489–493 | 0.71 | four permutes **inside `ttnn.scatter_add`** — one call in our code | not model-side |
| Reshape | 440 | **7.42** | `value` `(bs, HW, 256) → (bs, HW*heads, 32)`, TILE | [11](#candidate-11--absorb-msda-layout-prep) route 1 |
| | 457 | **7.37** | `grid` row width 32 → 2 | [11](#candidate-11--absorb-msda-layout-prep) (grid contract) |
| | 464 | 2.60 | `attn` row width 16 → 4 | [11](#candidate-11--absorb-msda-layout-prep) (attn contract) |
| | 444 | 1.77 | `sampling_offsets → (bs, Q, heads, 32)`, TILE | **necessary** — it enables 5e's TILE transpose |
| | 408 | 1.63 | TSA's counterpart of row 457 | [11](#candidate-11--absorb-msda-layout-prep) |
| | 484, 416 | 1.54, 1.04 | SCA and TSA output pack | [11](#candidate-11--absorb-msda-layout-prep) |
| | 448 | 1.04 | `attention_weights → (bs, Q, heads, L*P)` so softmax reduces over `L*P` | **necessary** |
| | 12 more | ~3.6 | rebatch plan and small TSA reshapes, none above 1.2 ms | — |

**Of the 61.8 ms: 19.3 ms is 6a, ~35 ms is the fused op's operand contract and therefore
[11](#candidate-11--absorb-msda-layout-prep)'s, ~2.8 ms is necessary shaping, 0.7 ms is inside
`scatter_add`.** There is no third category left for this entry.

### 6a. Hoist the SCA camera permute out of the layer loop

**The largest model-side item left, and it is a hoist, not a reformulation.**
`TTSpatialCrossAttention.forward` permutes `value` from `[num_cams, L, bs, embed_dims]` to
`[bs, num_cams, L, embed_dims]` at [sca:348-352](../tt/tt_spatial_cross_attention.py#L348-L352):
profiler row 437, **19.26 ms**, the most expensive non-`MSDAOperation` op in the layer.

**`value` does not change across encoder layers.** It is an argument to
`TTBEVFormerEncoder.forward`, passed into every layer unmodified
([tt_encoder.py:505-516](../tt/tt_encoder.py#L505-L516) — `value=value` inside the loop, never
reassigned). So the permute runs **six times per frame on identical data**; five are waste, **~96 ms
per frame**, ~16 ms per layer amortized. Same shape as
[1c](#1c-hoist-index-computation-above-the-layer-loop).

- [ ] Permute in `TTBEVFormerEncoder.forward` above the layer loop and pass `value` already as
      `[bs, num_cams, L, embed_dims]`. Keep the reshape at
      [sca:352](../tt/tt_spatial_cross_attention.py#L352) — it is free, no profiler row.
- [ ] `key` no longer has a use on this path ([5a](#5a-delete-the-key-permute)), so the
      `L = key.shape[1]` read at [sca:336](../tt/tt_spatial_cross_attention.py#L336) has to come off
      `value` instead.
- [ ] **This will measure ~0 on the layer harness**, which pays the permute once either way — the same
      trap [PERF.md](PERF.md#encoder-level-changes) records for 1c. Measure on
      `test_encoder_perf.py` end-to-end wall clock and say in the report that the layer number is
      structurally blind to it.
- [ ] Decide whether the permute belongs in the encoder at all, or whether whatever builds `value`
      upstream should emit `[bs, num_cams, L, embed_dims]` directly. That deletes it rather than
      hoisting it, and it is the only genuine *reformulation* left in this entry.

After 6a there is nothing between this entry and
[11](#candidate-11--absorb-msda-layout-prep), which owns every remaining row above. Its route 1 —
the op accepting TILE input and native layouts — would delete the 13.16 ms permute, the 8.91 ms
untilize behind it and 17.4 ms of contract reshapes together, **~40 ms**, where reordering reached
6.6 ms.

## Candidate 7 — L1 vs DRAM

**Expected small, and the reason is size.** Wormhole L1 is ~1.5 MB per Tensix core and an N150 has 64,
but a large fraction is reserved for kernels, circular buffers and allocator overhead — interleaved L1
across the chip is tens of MB, not hundreds. Against that:

| tensor | rough bytes (bf16) | fits usable L1? |
|---|---:|---|
| SCA `value` (`6 × ~30125 × 256`) | ~92 MB | no |
| TSA `query` (`10000 × 256`) | ~5 MB | maybe, tight |
| SCA rebatched `query` (`2484 × 256`) | ~1.3 MB | maybe |
| `sampling_grids` at SCA | large | no |
| per-level `attn` | small | yes |

A DRAM→L1 move pays only when the tensor is reused enough to amortize the copy, or when the consumer
is bandwidth-bound on that operand and the working set already lives in L1. The fused op reads
`value` / `grid` / `attn` once per call and the Linears read their weights once — no obvious reuse.

- [ ] List each operand's byte size and current `BufferType` from the CSV or the device-op args.
- [ ] Keep only operands that fit **and** are read more than once, or that feed a bandwidth-bound op.
      Price `ttnn.to_memory_config(..., L1)` on those alone.
- [ ] If nothing survives the filter — the expected outcome — record it in
      [DEAD_ENDS](perf_reports/DEAD_ENDS.md) with the size table so it is not re-derived.

Do not hold any other candidate behind this one.

## Candidate 8 — fuse `BinaryNg`

**Closed. Not worth doing.** `BinaryNg` is **2.9 ms at stage 10, 1.0% of the layer**, down from
48.2 ms — and candidate 5 got there by *deleting* the cost, not fusing it. The 48.2 ms was never
arithmetic: 46.6 of it was elementwise work on tensors tile-padded 128×, and
[5b](#5b-sampling-location-math-in-row_major) moved that onto unpadded shapes and folded away the
`* 2 - 1` this entry had queued as its next item.

The inventory the entry asked for, from the stage-10 CSV (`2026_09_02_11_49_26`) — and it is what
closes it:

| rows | ms | what | fusable into |
|---|---:|---|---|
| 456, 472, 476, 480 | 1.51 | SCA grid add, then **3 per-level accumulates** of the fused-op outputs | accumulate-into-output on the op — [12](#candidate-12--one-fused-call-for-all-levels) |
| 452, 453, 403, 404 | 0.57 | `2·ref - 1` in [`_grid_bias`](../tt/tt_ms_deformable_attention.py#L188-L215) | precompute per forward on the [rebatch plan](../tt/tt_spatial_cross_attention.py#L49-L64) |
| 389, 418, 420, 426, 486, 498, 507 | 0.58 | `query + query_pos` and the residual adds | epilogue of the preceding matmul |
| 407 | 0.20 | TSA grid add | — |
| 496 | 0.07 | `slots / count` | — |

**The largest single instance is 0.53 ms.** The biggest group is 1.51 ms, most of it the per-level
accumulate — an *op contract* ask that [12](#candidate-12--one-fused-call-for-all-levels) already
owns, since one multi-level call has nothing to accumulate. The `2·ref - 1` group is the only thing an
hour could remove (**~0.4 ms**); take it as a by-product if 1e/1f rewrite the plan anyway.

### What is worth keeping: the method

1. Take every `BinaryNg` row. If one operand is the previous op's output and the other is a broadcast,
   scalar or residual, it is a fusion candidate.
2. If the producer is a `Matmul` / `Linear`, the binary is an epilogue. ttnn already fuses bias; a
   residual add is still a separate launch.
3. If the producer is another `BinaryNg`, it is a tree — one kernel, or fold the constants away.
4. **Do not fuse across a layout change.** A binary between an untilize and a tilize is not an
   epilogue; the layout ops are the cost.
5. **Check the padded shape before believing the cost is arithmetic.** A 15 ms `add` on `(…, 4, 2)` is
   not an expensive add, it is a `32 × 32` tile holding 8 real values. Read `INPUT_0_*_PAD` before
   proposing a kernel.

Point 5 is worth 46 ms of hindsight and applies to every elementwise row in this model.

## Candidate 9 — trace capture

Trace capture needs shapes static across replays, and `rebatch_len` is the one shape that is not — it
is redecided every frame. 1a and 1c removed the transfers and moved the last readback out of the layer
loop, but neither makes that shape constant. Two things would:
[1b](#1b-bound-max_len-statically), rejected, and
[1e](#1e-an-empirical-high-water-mark-for-max_len), which is not — at the price of re-capturing
whenever the mark grows.

**Re-measurement collapsed what this is worth.** Per-layer gap is **8.9 ms**, not the 218 ms the entry
was written against, so trace capture cannot recover more than that at the layer level. It keeps a
reason to exist at the *encoder* level, where per-forward host work is not hidden behind device time —
encoder wall was 4234.5 ms against 6 × 691 = 4146 ms of steady-state layer time, so ~90 ms sits
outside the layers. Nobody has measured that case. **Treat 9 as low-value, not as blocked
high-value.** Both perf harnesses document the blocker in their module docstrings; update them when it
lifts.

## Candidate 10 — `MSDAOperation` itself

**The largest single cost in the layer and it has not moved since it appeared** — 167.6 ms at
[stage 04](perf_reports/04-fused-msda.md), 167.9 ms at stage 10, where it is now **59.9% of layer
kernel** because everything around it was halved. 143.3 ms of it is the four SCA calls at ~35.9 ms
each.

It is more expensive than the sampling it absorbed — **+45% per sample**:

| | old `GridSample` | new `MSDAOperation` |
|---|---:|---:|
| TSA (1 level) | 16.8 ms | 24.4 ms |
| SCA (4 levels) | 99.2 ms | 143.3 ms |

The op still won, because it deleted 215 ms of tail — but that says the tail was bad, not that the
kernel is good. VADv2 independently measured a floor of `N*Q >= 1024` below which the decomposition
beats it, consistent with real launch/packing overhead.

**Upstream, not model-side:**

- [ ] Standalone microbenchmark of `ttnn.experimental.multi_scale_deformable_attn` at SCA shapes
      (`N=48, Q=2496, P=4, D=32`) against a bare `ttnn.grid_sample` at the same shapes.
- [ ] If the gap reproduces, file it against the op with the numbers.

The ~103 ms of per-level layout prep that stage 04 measured alongside it is no longer this entry's:
candidate 5 took most of it, and what remains is [11](#candidate-11--absorb-msda-layout-prep)'s
~40 ms. This entry owns the op's device time, not the shuffle around it.

## Candidate 11 — absorb MSDA layout prep

**Now the largest model-side item, and no longer gated on 6.** Candidate 5 answered what 6 was
supposed to leave here: every remaining `Permute` and contract `Reshape` row
([the inventory](#the-inventory-at-stage-10)) is this entry's, **~40 ms**:

| what | ms |
|---|---:|
| `value` head permute, ROW_MAJOR (rows 442 + 394) | 13.9 |
| the untilize behind it (row 441) | 8.9 |
| contract reshapes — `grid` 32 → 2, `attn` 16 → 4, `value` 256 → 32, TSA's counterpart (rows 457, 464, 440, 408) | 18.8 |
| output pack, SCA + TSA (rows 484, 416) | 2.6 |

The site is [`_fused_msda_level`](../tt/tt_ms_deformable_attention.py#L36-L70) and the post-loop pack.
Two routes, in the order to try them:

1. **Relax the fused op's input/output contract** so it accepts the native BEVFormer layouts —
   `(bs, H*W, heads, D)`, `(bs, heads, Q, P, 2)`, `(bs, Q, heads, P)` in, `(bs, Q, embed)` out — and
   does the head-batch merge, the `H*W → (H, W)` split and the output pack inside the kernel. One
   launch, no intermediates. **The better landing if the op can take it:** the shuffle is addressing,
   not arithmetic, and the kernel already touches every element.
2. **A specialized MSDA-reshape op** taking the three native tensors and emitting the three contract
   tensors (and the inverse on the way out) in one pass. Fallback if the op cannot grow its contract —
   e.g. because VADv2 depends on today's shapes. Cheaper to land, but it is still a launch and still
   writes the shuffled tensors out, so it only wins if today's permutes/untilizes are launch-bound
   rather than bandwidth-bound.

**Route 1 is worth more than this entry originally argued.**
[Candidate 6](#why-the-weight-reorder-is-dead) establishes that the untilize of the ~92 MB `value`
tensor cannot be deleted by any weight reorder — a matmul emits TILE and the op demands ROW_MAJOR — so
accepting TILE input is the only thing that removes it. And
[candidate 12](#candidate-12--one-fused-call-for-all-levels) wants a contract change to the same op:
**take both upstream in one conversation.**

Either route must keep the op's [hard constraints](#op-contract) (INTERLEAVED, bfloat16,
`D % 16 == 0`, `grid.shape[1] == Q*P`) or move them to the new wrapper. Do not silently drop them.

- [ ] Microbenchmark route 1 vs route 2 vs today at the SCA per-level shapes
      (`N=48, Q=2496, P=4, D=32`, four levels). Gate on PCC **against the current fused path**, not
      against torch — this is a layout change, not a math change.
- [ ] Run candidate 5's noc pass first if the choice between routes is close: it is the thing that
      says whether the group is launch-bound or bandwidth-bound.
- [ ] Check `vadv2` before proposing a signature. If it feeds the op in today's layout, route 2 does
      not break it; route 1 needs a compatible default.

## Candidate 12 — one fused call for all levels

**It may be the largest single item left.** SCA runs **four** `multi_scale_deformable_attn` launches,
one per pyramid level ([msda:122-132](../tt/tt_ms_deformable_attention.py#L122-L132)), because the op
is `num_levels == 1` only. That one constraint costs three things:

| what | why the level count drives it |
|---|---|
| 4 launches instead of 1 | [candidate 10](#candidate-10--msdaoperation-itself) shows real launch/packing overhead — +45% per sample vs bare `grid_sample`, and VADv2's measured `N*Q >= 1024` floor |
| **4× the per-level prep** | [candidate 11](#candidate-11--absorb-msda-layout-prep)'s ~40 ms is *per level*; one level is one prep |
| 3 `ttnn.add`s | combining the per-level outputs — the largest group in [candidate 8](#candidate-8--fuse-binaryng) |

**Why it fell through the cracks:** [candidate 2](#candidate-2--fused-msda) recorded it as a blocker,
shipped the per-level workaround, and nothing inherited the follow-up. 10 owns the op's device time
but scopes itself to the single-level kernel; 11 asks the op to relax its *layout* contract but never
its *level* contract. Three entries, and the item sat between all of them.

**What the op would have to grow.** The reference formulation flattens all levels into one value
tensor and indexes with `level_start_index`; this op takes `(N, H, W, D)`, so the level count is baked
into the operand shape. A multi-level contract means either a list of value tensors plus their shapes,
or the flattened value plus `level_start_index` and `spatial_shapes` — the second is what mmcv does
and what the sampling grid is already normalized for. `attention_weights` needs no change: it is
already softmaxed jointly over `L*P`, so a multi-level kernel just stops decomposing it.

- [ ] Fold into 10's microbenchmark: 4 single-level calls at SCA's per-level shapes against one call
      over the same total sampled points. **That delta is what 12 is worth and it is measurable before
      any op change.**
- [ ] Only if the delta is real: raise the multi-level contract upstream together with 11's
      layout-contract ask. Two contract changes to one op are one negotiation, not two.
- [ ] Check in-tree callers first — `vadv2` runs `num_levels == 1` and must keep working.

## Candidate 13 — dtype and math fidelity

**Unpriced, and the first item here that spends accuracy.** Everything is bfloat16 today:
`DEFAULT_DTYPE` in [model_preprocessing.py](../tt/model_preprocessing.py) sets weights and biases and
every activation follows. The hypothesis is **bfloat16 activations + bfloat8_b weights + HiFi2**.

**One third of that hypothesis is already true.** BEVFormer's matmuls already run HiFi2: no call site
passes `compute_kernel_config`, and with bf16 inputs, no `program_config` and no user core grid,
matmul defaults to `MathFidelity::HiFi2`
([matmul_device_operation.cpp:2798-2799](../../../../ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp#L2798-L2799)).
So "move to HiFi2" is a no-op, not a change. Setting it explicitly is still worth doing — as
documentation, and because the default is conditional — but do not book a win against it.

**And there is a trap in the same code.** Fidelity drops to `LoFi` when `are_inputs_low_precision_df`
holds, which needs **both** operands bfloat8_b/bfloat4_b. bf8_b weights against bf16 activations keeps
HiFi2 — the hypothesis. But if activations are ever lowered too, fidelity silently drops to LoFi in
the same step: two precision changes landing as one, with one PCC number between them. **Pin
`compute_kernel_config` explicitly before touching activation dtype.**

**Where the win would come from — and why it is small.** Not matmul: it is 1.7% of the layer and has
been ~1% since the baseline. The reason to care is bytes — data movement is 32% of the layer at stage
10 and bfloat8_b is half the bytes of bfloat16. But the fused op requires **bfloat16**
([contract](#op-contract)), so `value`, `grid` and `attn` must be bf16 at the op boundary: a bf8_b
`value` buys a cheaper permute and then pays a typecast that reads and writes the whole tensor, which
is roughly what the permute cost. **Expect that to cancel.** The honest scope is weights, plus the
tensors that never reach the op — the FFN pair and `output_proj`.

- [ ] Set `compute_kernel_config` explicitly at all seven matmul sites at today's dtypes. Expect
      **zero** change — that is the point. It pins the conditional default and makes every later
      comparison honest.
- [ ] Flip `DEFAULT_DTYPE` for weights only (bfloat8_b), activations stay bfloat16, measure PCC and
      kernel time. One line, and weights are small, so expect a small number.
- [ ] Per-tensor, not global: list every tensor that never crosses the op's bf16 boundary and price
      bf8_b on those alone. A global flip either hits the `TT_FATAL` or silently inserts typecasts —
      both make the measurement meaningless.
- [ ] Report PCC **per change**, never for the batch, against the [gates](#candidates) — encoder
      0.997, SCA 0.999, plus the abs/rel/high-error columns. Headroom is 0.0026 at the encoder and the
      per-module gates bind first. Do not relax a threshold to make a dtype change pass.

Do not sequence any other candidate behind this one. It is independent of 6/11 and, unlike them, not
correctness-preserving.

---

## Candidate 14 — SFPU sampling geometry

**The largest single item, and it obsoletes [candidate 10](#candidate-10--msdaoperation-itself)'s premise.**
Tracked as [#55231](https://github.com/tenstorrent/tt-metal/issues/55231).

10 reads `MSDAOperation`'s 167.9 ms as a slow kernel and proposes filing it upstream. A profiling
branch measured the inside of that op and it is neither slow sampling nor bandwidth:
`CB-COMPUTE-WAIT-FRONT` was **36.1 ms on a 36.0 ms call** — the compute kernel idle for the entire
op, waiting on a reader deriving per-point coordinates in **soft float at ~140 cycles per operation**
on a dataflow core with no FPU, while the SFPU sat unused.

Moving that arithmetic to the SFPU: **167.6 → 29.5 ms**, sampling kernel untouched. The "+45% per
sample versus bare `grid_sample`" this backlog records was measuring the reader.

Two correctness items ride along: out-of-bounds corners lose their mask when the scalar moves to the
compute kernel (high-error ratio ~0.55 that the PCC gate passed), and `fp32_dest_acc_en` is required
because bf16's ulp at 200 is 1.0, so `floor(px)` on a 200×200 map degrades bilinear to
nearest-neighbour — 200×200 PCC 0.996156 → 0.999823, and the op got *faster*.

## Candidate 15 — axes as addresses, not data

**Tracked as [#55232](https://github.com/tenstorrent/tt-metal/issues/55232)**, with four additive op-contract changes beneath it:
[15a](https://github.com/tenstorrent/tt-metal/issues/55233) grid page fold · [15b](https://github.com/tenstorrent/tt-metal/issues/55234) packed value heads · [15c](https://github.com/tenstorrent/tt-metal/issues/55235) packed attn
runs · [15d](https://github.com/tenstorrent/tt-metal/issues/55236) rank-3 packed grid.

This is a third route past what [candidate 11](#candidate-11--absorb-msda-layout-prep) owns, and it
beat both of 11's. **Stop moving the axis; make it an address.** Widen the operand and let the
reader reach head `h` at a byte offset in the row it already reads, so the head-major tensor is
never produced.

That is the argument against 11's route 2 and against the `nlp_create_qkv_heads`-family op: a fused
head-reshape still has to **write** 92.6 MB and read it back. The permute ran at 14 GB/s because
both its pages were 64 bytes, not because there were four calls — and 11's own note that a hoist
measured as a wash is the same fact from the other side.

Layout plumbing went **60.5 → 15.4 ms**, the op paying under 1.5 ms total for offset reads.

Two rules, both learned expensively:

- **A NoC source offset must be 32-byte aligned.** attn's `(h*L*P + l*P)*2` gives 8/16/24 at
  `P=4, L=4`; the reader must round down, index from the boundary, and clamp at the page end.
- **An offset belongs to the input that packs the axis, never to the call.** Applying it to an
  unpacked operand read past the page end — which did not fail a test, it hung the kernel and took
  the chip off the PCIe bus.

**All figures in 14 and 15 are N150, from the profiling branch
`ctr-ikasic/bev_former_kernel_optimizations`.** That branch is evidence for these proposals, not an
implementation and not a Blackhole result; every number has to be remeasured on the target.

## Ordering

Landed or closed: **1a** (−2344.7 ms wall) · **1c** (−94.6 ms encoder) · **1d** (−56.4 ms encoder) ·
**2** (−191.6 ms kernel) · **3** (−32.7 ms) · **5**, all six sub-items (−176.3 ms, −38.6%) ·
**4** never run, [deleted by 2](perf_reports/04-fused-msda.md) · **8** closed at 2.9 ms ·
**1b** [rejected](perf_reports/DEAD_ENDS.md#3-a-static-bound-on-max_len), though
[its memory argument is gone](#the-memory-half-of-the-rejection-no-longer-holds).

What is left, in order:

1. **[6a](#6a-hoist-the-sca-camera-permute-out-of-the-layer-loop)** — the next thing to do. 19.3 ms
   per layer on a layer-invariant tensor, so five of six executions per frame are waste (~96 ms per
   frame). A 1c-class hoist, effort S, and it **must be measured on the encoder harness** — the layer
   harness is structurally blind to it.
2. **[11](#candidate-11--absorb-msda-layout-prep)** — ~40 ms, and after 6a it owns every remaining
   layout row. Route 1 also carries 12's level contract: one upstream conversation, two asks.
3. **[10](#candidate-10--msdaoperation-itself) + [12](#candidate-12--one-fused-call-for-all-levels)** —
   **167.9 ms, 59.9% of the layer**, and four launches where one would do. One microbenchmark answers
   both: per-sample cost against bare `grid_sample`, and 4 single-level calls against one call over
   the same points. The answer likely belongs upstream rather than in this model.
4. **[1e](#1e-an-empirical-high-water-mark-for-max_len)** — the only route to a static `rebatch_len`
   that costs neither +129 ms nor accuracy, and what makes 9 answerable. Its own gap payoff is
   ≤2.1 ms; **rank it on trace-capturability, not milliseconds.**
5. **[1f](#1f-derive-max_len-where-the-mask-is-produced)** — refactor only, and only as a by-product
   of 1e.
6. **[7](#candidate-7--l1-vs-dram)** — expected small. Reject-with-numbers, or keep the operands that
   actually fit. Do not block anything on it.
7. **[13](#candidate-13--dtype-and-math-fidelity)** — independent of everything above and cheap to
   try, but the first lossy change on the list, so it is ranked late on purpose: land the
   correctness-preserving work first and spend the 0.0026 of PCC headroom knowingly.
8. **[9](#candidate-9--trace-capture)** — needs 1e, and is worth ≤9 ms/layer rather than the 218 ms
   first claimed. Revisit only if an encoder-harness measurement shows per-forward host time the layer
   profile does not — and that harness would have to fix the gap column first.

Parked, not ranked — no device-time case, kept because they are structural:

- [**a statistical `max_len` bound**](#a-statistical-bound-instead-of-a-worst-case-one) — needs a
  dataset sweep and a PCC-vs-percentile curve. Not correctness-preserving, so it cannot be ranked
  against the rest. It needs the same sweep as 1e; run it once for both.
- [**fold the offset normalizer at init**](#the-fold-runs-in-forward-not-in-init) — moves two device
  muls and a per-call `.tolist()` out of `forward`. Zero measured gap; the point is that the first
  forward stops differing from the rest.
- [**the host-fallback residue**](#1g-what-is-still-host-side) — ~3.5 ms across four sites in one host
  block, plus seven sites at exactly 0.0 ms. Revisit if batch size or camera count grows.

### Lessons this backlog got wrong and should not repeat

- **Sequence rewrites before the cleanups inside them.** 4 was ranked first for being cheap; 2 deleted
  it.
- **Grep the tree for prior art before deriving an op contract.** `vadv2` had a working call site, a
  measured shape floor, and the offset-normalizer fold — all three relevant, none referenced here
  until stage 04.
- **A blocker deferred inside a landed entry needs a home, or it disappears.** 2 recorded the
  single-level limitation, shipped the workaround, and no entry inherited it; 10 and 11 each assumed
  the other owned it. It is [12](#candidate-12--one-fused-call-for-all-levels) now, and it may be the
  largest item left. When a candidate lands *around* a constraint, open the entry for the constraint
  in the same commit.
- **Read the op's defaults before proposing to change them.** HiFi2 was proposed as a change; matmul
  has been running HiFi2 all along by a conditional default no call site overrides.
- **Op count is not the metric; padded bytes are.** [5b](#5b-sampling-location-math-in-row_major)
  added five ops and removed 82 ms.
- **Measure the one item you flagged as able to lose.** [5d](#5d-value-head-split-without-the-padding)
  was benchmarked before it was written and came in at a third of its estimate — which is how it
  stayed a 20-minute change instead of an afternoon spent on a hoped-for 20 ms.
