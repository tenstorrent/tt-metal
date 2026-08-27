# BEVFormer encoder — performance

Measured results live here. The backlog of things still to try lives in
[perf_optimization_candidates.md](perf_optimization_candidates.md); each landed change moves from
there to the table below and gets its own report under [`perf_reports/`](perf_reports/).

## Harness

| | |
|---|---|
| test | [`tests/perf/test_layer_perf.py`](../tests/perf/test_layer_perf.py) — **one** encoder layer |
| config | `nuscenes_base`, `bev_size=(100, 100)`, `batch_size=1`, 4 levels, 6 cameras |
| device | Wormhole N150 |
| gate | PCC ≥ 0.997 against the torch reference, asserted inside the perf test itself |
| metric | `DEVICE KERNEL DURATION` and `OP TO OP LATENCY`, summed over the signposted region |
| iters | 1 warm-up, `DEVICE_PERF_ITERS = 1` — the gap column carries region-entry cost, see below |

```bash
MESH_DEVICE=N150 python -m tracy -p -r -v --op-support-count 20000 -m pytest \
  models/experimental/bevformer/tests/perf/test_layer_perf.py::test_bevformer_layer_perf -sv
```

CSV lands in `generated/profiler/reports/<timestamp>/ops_perf_results_<timestamp>.csv`; every report
names the one it used. **Keep the CSV** for any number quoted here.

`-p` profiles only the signposted zone, `-r` emits the ops report, `--op-support-count` sizes the
per-device op buffer. **None of these is trace capture** — the gap column is real host dispatch.

**One layer, not six.** The 6-layer encoder harness
([`test_encoder_perf.py`](../tests/perf/test_encoder_perf.py)) still runs and still gates PCC, but
it cannot be profiled: it emits more device ops than Tracy's per-device buffer holds, the device
report comes back truncated, and `process_ops_logs` aborts with `Device data missing: Op N not
present in cpp_device_perf_report.csv`. The layer is the repeated unit anyway — encoder ≈ 6 × layer
plus one point-sampling pass — so it is the right optimization target. Multiply by 6 for an encoder
estimate.

**Wall clock, not traced replay.** `TTSpatialCrossAttention.forward` reads `bev_mask` back to host
and the host result decides the shapes of the ops after it, so the encoder is not trace-capturable
today. The signposted region therefore carries host dispatch, and both columns are reported.

## Results

| # | change | kernel | gap | wall | Δ wall | ops |
|--:|---|---:|---:|---:|---:|---:|
| 0 | [baseline](perf_reports/00-baseline.md) | 655.6 ms | 2416.5 ms | **3072.1 ms** | — | 131 |
| 1 | [SCA rebatch and scatter-back on device](perf_reports/01-sca-rebatch-on-device.md) | 683.0 ms | 44.4 ms | **727.4 ms** | **−2344.7 ms** | 146 |
| 2 | [rebatch plan resolved once per forward](perf_reports/02-rebatch-plan-hoisted.md) | 680.4 ms | 46.3 ms | **726.7 ms** | −0.7 ms | 146 |
| 3 | [constant uploads cached](perf_reports/03-constant-uploads-cached.md) | 681.7 ms | 40.5 ms | **722.2 ms** | −4.5 ms | 121 |

`kernel` = summed `DEVICE KERNEL DURATION`. `gap` = summed `OP TO OP LATENCY`, i.e. device idle
between ops waiting on host dispatch. `wall` = kernel + gap, per layer.

Rows 1–3 were re-measured 2026-08-27 (`2026_08_27_08_56_58`, `2026_08_27_21_31_29`,
`2026_08_27_13_47_55`), each on that stage's `tt/` sources — Python-only diffs, no rebuild. PCC
0.999608 in all three.

**−76.5% of wall clock since the baseline.** Stage 1 traded +27 ms of kernel for −2372 ms of host
gap. **Kernel is now 94% of wall clock**, and 623 ms of it is the two deformable-attention calls.
There is no host-gap lever left at the layer level.

**Kernel is flat across stages 1–3** — 683.0 / 680.4 / 681.7 ms. None of these changes touches what
runs on device inside a layer; they remove dispatches.

**Stage 2 measures as a no-op here, by construction.** It hoists work into the *encoder*, above the
layer loop. This harness drives a layer directly, and `TTSpatialCrossAttention` builds its own
`SCARebatchPlan` when none is supplied — so the harness takes the unhoisted path and emits the same
146 ops as stage 1, at −0.7 ms. Read stage 2's value from the encoder table below.

**Stage 3's −4.5 ms understates it, and the gap column is why.** The op count drop, 146 → 121, is
the reliable signal; see the next section for what the gap actually is.

**Stage 1's gap was corrected from 218.3 ms to 44.4 ms.** The original figure does not reproduce on
its own tree. That CSV is gone, so it cannot be re-audited. Kernel time was consistent throughout.

### The gap column carries region-entry cost

The first ops after `signpost("start")` are charged the host sync, the deallocation and the queue
refill that precede them, because the command queue is empty at region entry. At
`DEVICE_PERF_ITERS = 1` that lands inside the reported gap and it is **large and noisy** — 6.4,
25.7 and 38.1 ms across three runs, sitting on one or two ops right after the signpost.

A diagnostic pass at `DEVICE_PERF_ITERS = 2` separates it. Second iteration, per tree:

| tree | iter 1 gap | iter 2 gap | entry cost | CSV |
|---|---:|---:|---:|---|
| stage 1 | 37.5 ms | 30.9 ms | 6.6 ms | `2026_08_27_20_58_53` |
| stage 2 | 45.1 ms | 38.8 ms | 6.3 ms | `2026_08_27_21_04_33` |
| stage 3 | 16.1 ms | **8.9 ms** | 7.2 ms | `2026_08_27_20_44_43` |

**Steady-state gap at HEAD is 8.9 ms over 121 ops** — host dispatch is otherwise fully hidden, and
kernel is 99% of steady-state wall clock. Two trees with identical op counts (stages 1 and 2) differ
by 7.9 ms of gap, which puts the **noise floor for this metric at ~±8 ms**.

More warm-up iterations do not help — the queue is empty at region entry no matter how many forwards
preceded the signpost. Only a second measured iteration separates it, at the cost of doubling the
ops inside the region, so the harness stays at 1. **Consequence: do not read gap deltas below
~40 ms from the table above.** Raise `DEVICE_PERF_ITERS` to 2 and segment the CSV at its midpoint
when a gap delta is the result being claimed.

### Encoder-level changes

Stage 2's win is per-forward, and the layer harness takes the unhoisted path (see above), so its
effect is only visible end-to-end. Stage 3 appears in both tables — the layer figure is the
per-layer cost, this one the whole forward.

| # | change | encoder wall (6 layers) | Δ | report |
|--:|---|---:|---:|---|
| — | *before stage 2* | 4385.5 ms median | — | — |
| 2 | rebatch plan resolved once per forward | 4290.9 ms median | −94.6 ms (−2.2%) | [02](perf_reports/02-rebatch-plan-hoisted.md) |
| 3 | constant uploads cached | **4234.5 ms** median | −56.4 ms (−1.3%) | [03](perf_reports/03-constant-uploads-cached.md) |

Encoder wall clock is end-to-end host time over 11 iterations, not a profiled figure — noisier than
the layer harness. Stage 2's median and minimum disagree by a factor of three, so read it as
**~1–2%**; stage 3's agree (−56.4 vs −48.3 ms). Together **−151.0 ms, −3.4%**.

Both are **steady-state** figures: the benchmark warms up before timing, and stage 3 in particular
caches across forwards, so the first frame still pays.

## Where the baseline time is

| region | ops | kernel | gap |
|---|---:|---:|---:|
| SCA — deformable attention (`2484 × 30125`) | 72 | 522.0 ms | 0.3 ms |
| TSA — deformable attention (`10000 × 10000`) | 39 | 91.5 ms | 154.5 ms |
| SCA — rebatch / scatter-back (outside MSDA) | 5 | 40.0 ms | **1917.0 ms** |
| TSA — forward, outside MSDA | 3 | 0.2 ms | 267.7 ms |
| MSDA exit | 5 | 0.4 ms | 68.3 ms |
| FFN | 5 | 1.3 ms | 7.6 ms |
| rest | 2 | 0.2 ms | 1.1 ms |

Two independent problems, and they do not compete:

1. **Host round-trips.** A *single* 1.917 s stall sits at the first op after the SCA rebatch loop —
   62% of the layer's wall clock in one gap, produced by ~36 `to_torch`/`from_torch` calls that move
   tensors which never had to leave host memory. This is [candidate 1](perf_optimization_candidates.md#candidate-1--host-round-trips).
2. **Deformable-attention kernel time.** 613 ms of the 656 ms kernel total is the two MSDA calls,
   and none of it is matmul (11 matmuls, 4.7 ms combined). It is concat, reshape, permute and
   grid-sample on ROW_MAJOR tensors, plus tile padding of degenerate dimensions. This is
   [candidate 2](perf_optimization_candidates.md#candidate-2--fused-msda) and
   [candidate 3](perf_optimization_candidates.md#candidate-3--tile-padding-waste).

Problem 1 is closed — candidate 1 is complete. Problem 2 is untouched.

## Where the time is now

`2026_08_27_13_47_55`, HEAD: 121 ops, 681.7 ms kernel.

| Op | inst | ms | % of kernel |
|---|---:|---:|---:|
| ReshapeViewDeviceOperation | 18 | 157.3 | 23.1 |
| GridSampleOperation | 5 | 116.0 | 17.0 |
| ConcatDeviceOperation | 1 | 115.5 | 16.9 |
| PermuteDeviceOperation | 23 | 105.7 | 15.5 |
| BinaryNgDeviceOperation | 18 | 85.8 | 12.6 |
| SliceDeviceOperation | 9 | 29.3 | 4.3 |
| UntilizeWithUnpaddingDeviceOperation | 12 | 28.9 | 4.2 |
| TilizeWithValPaddingDeviceOperation | 4 | 14.3 | 2.1 |
| ScatterDeviceOperation | 1 | 10.5 | 1.5 |
| MatmulDeviceOperation | 11 | 4.7 | 0.7 |
| *13 others* | 20 | 15.5 | 2.3 |

Matmul is still 0.7%. The three largest ops are the ones the baseline named, and the concat went
from 3 instances totalling 113.6 ms to **1 instance at 115.5 ms** — same time, one call, so
[candidate 4](perf_optimization_candidates.md#candidate-4--the-msda-concat) is untouched by stages
01–03 and remains the cheapest kernel win.

## What was tried and rejected

[perf_reports/DEAD_ENDS.md](perf_reports/DEAD_ENDS.md) — measured, not in the tree, with the reason
each one lost. Two of the three are worth re-testing after candidate 2.

## Report format

One file per landed change, `NN-slug.md`, containing: source commit, kernel/gap after, delta from
the previous stage, what the change was and why, and the per-op-code table. Same numbers, same
harness, same signposts every time — so the deltas sum.

Only landed changes get a report. Re-measurements that revise a published figure are corrected in
place, in the report that published it.
