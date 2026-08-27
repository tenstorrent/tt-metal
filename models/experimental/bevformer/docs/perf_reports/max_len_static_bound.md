# Can `max_len` be bounded statically?

Investigation for [candidate 1b](perf_optimization_candidates.md#1b-bound-max_len-statically). No
code change — this decides whether one is worth making.

## Why the question exists

`max_len` is the number of BEV queries a camera can see, maxed over cameras. It sizes the
spatial-cross-attention rebatch, so it must be a Python int, so `bev_mask` must come back to host.
That readback is the last thing standing between the encoder and trace capture
([candidate 5](perf_optimization_candidates.md#candidate-5--trace-capture)), which is what would
collapse the remaining 218 ms of per-layer host gap.

TTNN has no dynamic shapes and will not grow them for us. The only way out is to pick `max_len`
ahead of time. That is safe only if a bound exists which is **tight enough** not to cost more than
it saves and **large enough** that no frame exceeds it — a frame that does would silently drop
queries.

## Answer

**A static bound is feasible, but the case for it is weak, and it is not the blocker it looked
like.**

- The compute cost is **exactly linear** in the bound — there is no overhead to amortize, so every
  row of headroom is paid for in full.
- The naive bound (`max_len = num_queries`) is not merely expensive, it is **impossible**: it runs
  out of DRAM.
- A sensible bound is ~1.24× today's `max_len`, costing **+129 ms** of kernel time to unlock
  **−218 ms** of gap. Net −89 ms on a 900 ms layer, ~10%, in exchange for a new correctness risk.

Candidates 4 and 2 are worth 115 ms and up to 623 ms respectively, carry no correctness risk, and
need no geometric argument. **1b stays deprioritized.**

## 1. How big is `max_len`, really

Measured on host from the deterministic rigs in
[`camera_rig.py`](../config/encoder_config/camera_rig.py), across configs and grid sizes.

| rig | 30×30 | 50×50 | 100×100 | 200×200 | `max_len / num_queries` |
|---|---:|---:|---:|---:|---:|
| nuScenes | 225 | 621 | 2472 | 9885 | **0.247** |
| CARLA | 193 | 537 | 2146 | 8587 | **0.215** |

**The ratio is scale-invariant.** It does not drift with grid resolution, because it is a property
of the rig's field-of-view coverage of the BEV disc, not of how finely that disc is sampled. That is
what makes a *relative* bound (a fraction of `num_queries`) meaningful in the first place — an
absolute one would have to be re-derived per grid size.

Per camera, at nuScenes 100×100:

| camera | yaw | queries seen | fraction |
|---|---:|---:|---:|
| CAM_FRONT | 0° | 1579 | 0.158 |
| CAM_FRONT_LEFT | 55° | 1913 | 0.191 |
| CAM_FRONT_RIGHT | −55° | 1919 | 0.192 |
| CAM_BACK_LEFT | 110° | 1812 | 0.181 |
| CAM_BACK_RIGHT | −110° | 1799 | 0.180 |
| **CAM_BACK** | 180° | **2472** | **0.247** |

`max_len` is set by **CAM_BACK alone**, and by a wide margin — it is the wide-FOV unit (809 px focal
against 1266 px for the other five). Any bound is really a bound on one camera.

## 2. How much does it move

Same rig, perturbed uniformly, nuScenes 100×100, nominal 2472.

| perturbation | `max_len` | vs nominal |
|---|---:|---:|
| yaw ±0.5–5° | 2471–2475 | ≤ 1.001× |
| pitch +1° | 2478 | 1.002× |
| pitch +5° | 2492 | 1.008× |
| translate x +0.1 m | 2482 | 1.004× |
| translate x +0.5 m | 2520 | 1.019× |
| translate x +1.0 m | 2568 | **1.039×** |
| translate z +0.5 m | 2471 | 1.000× |

**`max_len` is remarkably stable.** A 1 m mounting error — far outside any real calibration
tolerance — moves it under 4%. Realistic perturbation moves it under 0.5%.

Two caveats this study cannot close:

- **Device and host disagree.** The device computes `max_len = 2484` where host computes 2472, a
  boundary-comparison effect in bfloat16 already documented in
  [pcc_drop_after_deterministic_lidar2img.md](pcc_drop_after_deterministic_lidar2img.md). A bound
  derived on host must cover the device's own inflation. Small (+0.5%) but not zero.
- **No real dataset here.** These are synthetic rigs. `lidar2img` on real nuScenes is calibration
  plus the ego motion between lidar and camera timestamps, so it varies per frame by roughly the
  perturbations tabulated above — but that is an inference from the rig geometry, not a measurement
  of the dataset. Anyone landing a bound should confirm against real `lidar2img` matrices.

## 3. What a bound costs

Measured by driving the real SCA path with a synthetic `bev_mask` of controlled density, so
`max_len` is the swept variable. nuScenes 100×100, N150.

| `rebatch_len` | MSDA kernel | vs baseline | µs per row |
|---:|---:|---:|---:|
| 2496 *(today)* | 532.4 ms | 1.00× | 213.3 |
| 2560 | 561.7 ms | 1.06× | 219.4 |
| 2816 | 604.5 ms | 1.14× | 214.7 |
| 3072 | 661.3 ms | 1.24× | 215.3 |
| 4096 | 870.4 ms | 1.64× | 212.5 |
| 5120 | — | — | **out of memory** |

**Cost is linear.** 213–219 µs per row across the whole range: no fixed overhead, no economy of
scale. Every row of headroom is paid for at full price — which is the worst possible shape for a
safety margin.

**And there is a hard ceiling.** At 5120 the run dies inside deformable attention:

```
TT_FATAL: Out of Memory: Not enough space to allocate 2013265920 B DRAM buffer
```

So the feasible range at this grid is `(2484, ~4096]` — a bound above ~0.41 × `num_queries` does not
run at all. The naive worst-case bound of `num_queries` was never a slow option; it was never an
option. This is the same wall the `200×200` PCC parametrization hits (`max_len` 9885), which is why
that test is deselected.

## 4. The trade, stated plainly

Take 3072 — 1.24× today's `max_len`, 20% headroom over the worst perturbation observed (2568), well
inside the memory ceiling.

| | |
|---|---:|
| extra MSDA kernel | **+129 ms** |
| gap removed by trace capture | **−218 ms** |
| net, per layer | **−89 ms** on 900 ms (**−10%**) |

At 2816 (13% headroom over the device's 2484, 9.7% over the worst perturbation) it is +72 ms for a
net −146 ms, ~−16%. Better, and thinner.

That is the whole prize, and it comes with a failure mode the current code does not have: **a frame
that exceeds the bound silently drops queries.** Not a crash, not a PCC failure that any existing
gate would catch — a quietly worse BEV feature for the queries that fell off the end. Buying it
needs a runtime assert, and an assert on a device tensor is another sync unless it is debug-gated.

## 5. What this changes

- **1b is not the blocker it appeared to be.** It gates trace capture, and trace capture is worth
  218 ms — but 1b's own cost eats more than half of that.
- **The memory ceiling is the real finding.** It is why `200×200` cannot run at all, and it caps any
  future `max_len` growth. That belongs to the deformable-attention allocation, so
  [candidate 2](perf_optimization_candidates.md#candidate-2--fused-msda) may move it — worth
  re-measuring this ceiling after 2 lands rather than designing a bound around today's number.
- **Better sequencing:** land 2 and 4 first. If the fused op lowers both MSDA time and its
  allocation, the same bound costs proportionally less and the trade improves on its own.

## Reproducing

Both studies are scratch scripts, not committed:

- coverage and sensitivity — host only, no device
- the cost curve — drives `TTSpatialCrossAttention` with a synthetic `bev_mask`, profiled under
  `tracy`; the MSDA signpost reports `max_len` directly as `query.shape[1]`, so the CSV segments by
  it without extra instrumentation
