# Stage: 02-rebatch-plan-hoisted

- harness: **encoder wall clock**, not the layer harness — see below
- config: `nuscenes_base`, 100×100, 6 layers, N150, 11 timed iterations after one warm-up
- encoder wall: **4290.9 ms median** (−94.6 ms, **−2.2%**), 4266.0 ms min (−29.7 ms, −0.7%)
- PCC: 16 tests pass across SCA, layer and encoder suites; no numerical change

## Why this stage is measured differently

The win is `num_layers − 1` copies of work that the layer harness runs exactly once, so **the layer
harness structurally cannot show it** — and the 6-layer encoder cannot be Tracy-profiled, because it
exceeds the device op buffer (see [PERF.md](../PERF.md#harness)). What is left is end-to-end wall
time of the encoder forward, which is what this change targets anyway.

That measurement is noisier than the profiler's. **Read this as ~1–2%, not as 2.2%** — the median
and the minimum disagree by a factor of three, which puts the effect in the same range as the
run-to-run spread. It is not in the same class of result as stage 01.

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
a hard blocker for [candidate 5](../perf_optimization_candidates.md#candidate-5--trace-capture),
where one above the loop is not.

It is not sufficient for trace capture on its own. `rebatch_len` still varies per frame, so a
captured trace would be invalid on the next one — that is
[candidate 1b](DEAD_ENDS.md#3-a-static-bound-on-max_len), which is rejected on its own terms.

## Honest accounting

This is a **~2% change that costs a dataclass and two extra parameters** threaded through the layer
signature. The performance case alone does not carry it. It is in the tree for two other reasons:

1. It deletes work that was provably redundant — six identical gathers is a defect regardless of
   what it measures.
2. It moves the last per-layer host readback out of the layer loop, which every later trace-capture
   attempt would otherwise have to do first.

If either of those had been false, the number would not have justified the change.
