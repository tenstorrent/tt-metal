# Stage: 02-rebatch-plan-hoisted

- source commit: [`d10beef5a04`](https://github.com/tenstorrent/tt-metal/commit/d10beef5a04f7362351b76e73781ef5bdee42a18)
- harness: **encoder wall clock** — the layer harness measures this stage as a no-op, see below
- config: `nuscenes_base`, 100×100, 6 layers, N150, 11 timed iterations after one warm-up
- encoder wall: **4290.9 ms median** (−94.6 ms, **−2.2%**), 4266.0 ms min (−29.7 ms, −0.7%)
- PCC: 16 tests pass across SCA, layer and encoder suites; no numerical change

## Why this stage is measured differently

The win is `num_layers − 1` copies of work that a single layer runs once. The layer harness also
drives `TTSpatialCrossAttention` directly, which builds its own `SCARebatchPlan` when none is
supplied — so it takes the unhoisted path and measures the same 146 ops as stage 01. Profiled on
this tree (`2026_08_27_21_31_29`) it reports **680.4 ms kernel / 46.3 ms gap / 726.7 ms wall**
against stage 01's 683.0 / 44.4 / 727.4 — −0.7 ms of wall, inside the noise. A two-iteration pass
agrees: 38.8 ms of steady-state gap against stage 01's 30.9 ms, which is this metric's ~±8 ms noise
floor. The 6-layer encoder cannot be
Tracy-profiled either (it exceeds the device op buffer, see [PERF.md](../PERF.md#harness)), so what
is left is end-to-end encoder wall time.

**Read this as ~1–2%, not 2.2%** — median and minimum disagree by a factor of three, putting the
effect in the same range as the run-to-run spread.

## What this change was

**The spatial-cross-attention rebatch is resolved once per forward instead of once per layer.**

`bev_mask` and `reference_points_cam` come out of the camera projection, which the encoder already
runs once above the layer loop. Everything the rebatch derives from them is therefore identical for
all six layers, and was being recomputed six times:

| Recomputed per layer | Now |
|---|---|
| `ttnn.to_torch(bev_mask)` — the host readback | once per forward |
| `max_len`, `rebatch_len`, the per-camera `torch.nonzero` loops | once per forward |
| `ttnn.clamp` on the reference points | once per forward |
| **the entire reference-point rebatch** — layout conversion, `embedding`, reshape | once per forward |
| the scatter index upload and its `ttnn.repeat` widening | once per forward |
| the contributor-count reduction and upload | once per forward |

The reference-point rebatch is the surprise: it never touches `query`, so it is fully determined by
the projection. Six identical copies of a gather were being computed and thrown away.

What genuinely remains per layer is **only the query gather** — `query` changes at every layer,
which is the point of the layer.

Carried in an `SCARebatchPlan` built by the encoder and passed down. `TTSpatialCrossAttention`
builds its own when none is supplied, which keeps the module usable standalone — the SCA PCC tests
take that path.

## Where the remaining host work is

Five of six `bev_mask` readbacks are gone, and **the layer loop no longer contains one at all**.
That is the structural result, and it matters more than the 2%: a readback inside the layer loop is
a hard blocker for [candidate 9](../perf_optimization_candidates.md#candidate-9--trace-capture),
where one above the loop is not.

Not sufficient for trace capture on its own: `rebatch_len` still varies per frame, so a captured
trace would be invalid on the next one. That is
[candidate 1b](DEAD_ENDS.md#3-a-static-bound-on-max_len), rejected on its own terms.

## Honest accounting

A ~2% change costing a dataclass and two extra parameters through the layer signature. The
performance case alone does not carry it; it is in the tree because six identical gathers is a
defect regardless of what it measures, and because it moves the last per-layer host readback out of
the layer loop — which every later trace-capture attempt would otherwise have to do first.
