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

**One layer, not six.** The 6-layer encoder harness
([`test_encoder_perf.py`](../tests/perf/test_encoder_perf.py)) still runs and still gates PCC, but
it cannot be profiled: it emits more device ops than Tracy's per-device buffer holds, the device
report comes back truncated, and `process_ops_logs` aborts with `Device data missing: Op N not
present in cpp_device_perf_report.csv`. The layer is the repeated unit anyway — encoder ≈ 6 × layer
plus one point-sampling pass — so it is the right optimization target. Multiply by 6 for an encoder
estimate.

**Wall clock, not traced replay.** `TTSpatialCrossAttention.forward` reads `bev_mask` back to host
and the host result decides the shapes of the ops after it, so the encoder is not trace-capturable
today (candidate 1 is what changes that). Until then the signposted region carries host dispatch,
and **op-to-op latency is a real part of the cost**, not measurement noise. Both columns are
reported for that reason.

## Results

| # | change | kernel | gap | wall | Δ wall |
|--:|---|---:|---:|---:|---:|
| 0 | [**baseline**](perf_reports/00-baseline.md) | 655.6 ms | 2416.5 ms | **3072.1 ms** | — |
| 1 | [SCA rebatch and scatter-back on device](perf_reports/01-sca-rebatch-on-device.md) | 682.0 ms | 218.3 ms | **900.2 ms** | **−2171.9 ms** |

`kernel` = summed `DEVICE KERNEL DURATION`. `gap` = summed `OP TO OP LATENCY`, i.e. the time the
device spent idle between ops waiting on host dispatch. `wall` = kernel + gap, per layer. Every
stage re-measured through the same harness with the same signposts, so the deltas sum.

**−70.7% of wall clock, at an unchanged PCC of 0.999608.** Stage 1 traded +26.4 ms of kernel for
−2198.2 ms of host gap. The balance has flipped: kernel is now 76% of wall clock, and 623 ms of that
is the two deformable-attention calls.

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

## What was tried and rejected

[perf_reports/DEAD_ENDS.md](perf_reports/DEAD_ENDS.md) — measured, not in the tree, with the reason
each one lost. Two of the three are worth re-testing after candidate 2.

## Report format

One file per landed change, `NN-slug.md`, containing: source commit, kernel/gap after, delta from
the previous stage, what the change was and why, and the per-op-code table. Same numbers, same
harness, same signposts every time — so the deltas sum.
