# Dead ends

Companion to [PERF.md](../PERF.md). Everything here was measured on a Wormhole N150 and is **not**
in the tree, kept so nobody spends a day rediscovering it.

Each entry names the **stage** it was tried at — the row in
[PERF.md's results table](../PERF.md#results) that gives the layer's state at that moment. The
bottleneck moved as the work progressed, so a negative result here is evidence about a layer at that
wall clock, not a law. Two are worth re-testing after
[candidate 2](../perf_optimization_candidates.md#candidate-2--fused-msda) lands.

| # | what | why it lost |
|---|---|---|
| [1](#1-hoist-the-invariant-reads-out-of-the-host-rebatch-loop) | hoist invariant reads out of the host rebatch loop | −56% and dropped anyway — the loop it optimizes no longer exists |
| [2](#2-ttnngather-for-the-reference-point-rebatch) | `ttnn.gather` for the reference-point rebatch | 800× slower than `ttnn.embedding` at the same shapes |
| [3](#3-a-static-bound-on-max_len) | a static bound on `max_len` | +129 ms of kernel to unlock ~9 ms of gap, and the naive bound does not fit in DRAM |

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
having any, and [stage 01](01-sca-rebatch-on-device.md) deletes the loop it optimizes — measuring
−2171.9 ms on the same baseline.

Recorded because −56% is a large number to walk away from, and because it is the trap: the profile
points at a loop, and optimizing the loop is not the same as asking why the loop is there.

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

Caught by measurement, not review — the `gather` version was correct and passed every PCC gate.

## 3. A static bound on `max_len`

Investigated at stage 01 (727.4 ms wall) as
[candidate 1b](../perf_optimization_candidates.md#1b-bound-max_len-statically). Not implemented.

`max_len` is a data-dependent shape, so it forces a host readback of `bev_mask` — the last thing
keeping the encoder from being trace-capturable. Pinning it to a constant would remove that.

**Why it lost: +129 ms of kernel to unlock ~9 ms of gap**, plus a failure mode the current code does
not have. Candidates 4 and 2 are worth more, carry no correctness risk, and need no geometric
argument.

The trade was originally priced against the 218 ms of per-layer gap reported in stage 01. That
figure [does not reproduce](01-sca-rebatch-on-device.md); steady-state gap is ~9 ms, so the
rejection is decisive rather than marginal.

### The bound is derivable

`max_len / num_queries` is **scale-invariant** — it is the rig's FOV coverage of the BEV disc, not a
property of how finely that disc is sampled:

| rig | 30×30 | 50×50 | 100×100 | 200×200 | ratio |
|---|---:|---:|---:|---:|---:|
| nuScenes | 225 | 621 | 2472 | 9885 | **0.247** |
| CARLA | 193 | 537 | 2146 | 8587 | **0.215** |

**CAM_BACK alone sets it** — 2472 of 10000 queries at nuScenes 100×100, against 1579–1919 for the
other five. It is the wide-FOV unit (809 px focal against 1266 px). Any bound is a bound on one
camera.

It is stable: yaw ±5° and pitch +5° move `max_len` under 1%, and a 1 m mounting-translation error —
far outside calibration tolerance — moves it 3.9%.

Two caveats this study cannot close:

- **Device and host disagree.** Device computes 2484 where host computes 2472 — a bfloat16
  boundary-comparison effect, documented in
  [pcc_drop_after_deterministic_lidar2img.md](../pcc_drop_after_deterministic_lidar2img.md). A
  host-derived bound must cover that +0.5%.
- **Synthetic rigs only.** Real nuScenes `lidar2img` is calibration plus ego motion between lidar
  and camera timestamps. That it varies within the perturbations above is an inference from rig
  geometry, not a measurement. Confirm against real matrices before landing a bound.

### Cost is linear, and there is a hard ceiling

Measured by driving the real SCA path with a synthetic `bev_mask` of controlled density, so
`max_len` is the swept variable. nuScenes 100×100, N150.

| `rebatch_len` | MSDA kernel | vs baseline | µs/row |
|---:|---:|---:|---:|
| 2496 *(today)* | 532.4 ms | 1.00× | 213.3 |
| 2560 | 561.7 ms | 1.06× | 219.4 |
| 2816 | 604.5 ms | 1.14× | 214.7 |
| 3072 | 661.3 ms | 1.24× | 215.3 |
| 4096 | 870.4 ms | 1.64× | 212.5 |
| 5120 | — | — | **out of memory** |

213–219 µs per row across the whole range: **no fixed overhead, no economy of scale.** Every row of
headroom is paid at full price — the worst possible shape for a safety margin.

At 5120 the run dies inside deformable attention:

```
TT_FATAL: Out of Memory: Not enough space to allocate 2013265920 B DRAM buffer
```

The feasible range at this grid is `(2484, ~4096]`. **The naive bound of `num_queries` was never a
slow option — it was never an option.** Same wall the `200×200` PCC parametrization hits
(`max_len` 9885), which is why that test is deselected.

### The failure mode

A frame exceeding the bound **silently drops queries**. Not a crash, not something an existing PCC
gate catches — a quietly worse BEV feature for the queries that fell off the end. Guarding it needs
a runtime assert, and an assert on a device tensor is another sync unless it is debug-gated.

### Re-test after candidate 2

**The memory ceiling is the real finding.** It is why `200×200` cannot run, and it caps any future
`max_len` growth. It belongs to the deformable-attention allocation, so
[candidate 2](../perf_optimization_candidates.md#candidate-2--fused-msda) may move both it and most
of the per-row cost. Land 2 and 4 first; if the fused op lowers either, the same bound costs
proportionally less and the trade improves on its own.

**Reproducing:** both studies are scratch scripts, not committed. Coverage and sensitivity are host
only. The cost curve drives `TTSpatialCrossAttention` with a synthetic `bev_mask` under `tracy`; the
MSDA signpost reports `max_len` as `query.shape[1]`, so the CSV segments by it without extra
instrumentation.
