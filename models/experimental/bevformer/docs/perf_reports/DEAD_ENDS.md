# Dead ends

Companion to [PERF.md](../PERF.md). Everything here was measured on a Wormhole N150 and is **not**
in the tree, kept so nobody spends a day rediscovering it.

Each entry names the **stage** it was tried at — the row in
[PERF.md's results table](../PERF.md#results), whose `wall` column gives the layer's state at that
moment. That matters: the bottleneck moved as the work progressed, and a negative result here is
evidence about a layer at that wall clock, not a law. Two of these would be worth re-testing after
[candidate 2](../perf_optimization_candidates.md#candidate-2--fused-msda) lands.

| # | what | why it lost |
|---|---|---|
| [1](#1-hoist-the-invariant-reads-out-of-the-host-rebatch-loop) | hoist invariant reads out of the host rebatch loop | −56% and dropped anyway — the loop it optimizes no longer exists |
| [2](#2-ttnngather-for-the-reference-point-rebatch) | `ttnn.gather` for the reference-point rebatch | 800× slower than `ttnn.embedding` at the same shapes |
| [3](#3-a-static-bound-on-max_len) | a static bound on `max_len` | costs more than half of what it unlocks, and the naive bound does not fit in DRAM |

---

## 1. Hoist the invariant reads out of the host rebatch loop

Tried at stage 0 (baseline, 3072.1 ms wall). **Measured −1721 ms, −56%** — and deliberately not in
the tree.

The rebatch loop ran `bs × num_cams` times and on each iteration re-read `query` and
`reference_points_cam`, which are loop-invariant, plus both accumulators it had written the
iteration before, then re-uploaded both. Hoisting the invariant reads and doing one upload after the
loop removed ~30 of ~36 host transfers, at zero risk: the PCC gate came back at 0.999608, identical
to six decimal places.

**Why it was dropped:** it reduces the number of host transfers instead of removing the reason for
having any, and [stage 01](01-sca-rebatch-on-device.md) deletes the loop it optimizes. Reducing a
cost is not the same as removing it, and when both are available at comparable effort the removal
wins. Stage 01 went on to measure −2171.9 ms, −71%, on the same baseline.

Worth recording because **−56% was a genuinely large number to walk away from**, and because it is
the trap this kind of work sets: the profile points at a loop, and optimizing the loop is not the
same as asking why the loop is there.

## 2. `ttnn.gather` for the reference-point rebatch

Tried inside stage 01. **97.59 ms in a single call**, against **0.12 ms** for the `ttnn.embedding`
that replaced it — the fifth most expensive op in the layer, above every matmul.

| | `ttnn.gather(input, dim, index)` | `ttnn.embedding(index, weight)` |
|---|---|---|
| index granularity | one id per **element** | one id per **row** |
| index shape here | `[6, 1, 2496, 8]` — the row id copied across all 8 columns | `[1, 1, 1, 14976]` — one id per output row |
| internals | transposes the gather dim to last, then walks it | direct table lookup |
| **measured** | **97.59 ms** | **0.12 ms** |

The cost is not data movement — the output is under 1 MB either way. `gather` moves the gather
dimension to the last position before walking it, and at `[6, 1, 10000, 8]` that transpose is the
entire 97.59 ms.

**The rule:** when the indexing is over whole rows and the row width is fixed, `embedding` is the
op. Reach for `gather` only when the index genuinely varies *within* a row. An index tensor that
turns out to be the same value repeated across the last dimension is the tell.

It also matters that this was caught by **measurement, not review**. The `gather` version was
correct and passed every PCC gate. A correctness-only workflow ships it.

## 3. A static bound on `max_len`

Investigated at stage 01 (900.2 ms wall) as
[candidate 1b](../perf_optimization_candidates.md#1b-bound-max_len-statically). Full data in
[max_len_static_bound.md](max_len_static_bound.md). Not implemented.

`max_len` is a data-dependent shape, so it forces a host readback of `bev_mask`, which is what keeps
the encoder from being trace-capturable. Pinning it to a constant would remove that.

What the measurements said:

- **The bound would be derivable.** `max_len / num_queries` is 0.247 for the nuScenes rig and 0.215
  for CARLA, identical from 30×30 to 200×200 — it is a camera-FOV property, not a grid property, and
  CAM_BACK alone sets it. A 1 m mounting-translation error moves it under 4%.
- **Cost is exactly linear**, 213–219 µs per row with no fixed overhead, so every row of headroom is
  paid for at full price — the worst possible shape for a safety margin.
- **The naive bound is impossible, not merely slow.** At `max_len = 5120`, deformable attention
  fails to allocate a 2.0 GB DRAM buffer. The feasible range at 100×100 is `(2484, ~4096]`. This is
  the same wall that keeps the `200×200` PCC parametrization deselected.

**Why it lost:** at a sensible bound of 3072 it is **+129 ms of kernel to unlock −218 ms of gap** —
net −89 ms on a 900 ms layer, ~10% — and it buys a failure mode the current code does not have,
since a frame exceeding the bound silently drops queries rather than failing. Candidates 4 and 2 are
worth more, carry no correctness risk, and need no geometric argument.

**Re-test after candidate 2.** The fused deformable-attention op owns the allocation that sets the
memory ceiling and most of the per-row cost. If it moves either, the terms of this trade change.
