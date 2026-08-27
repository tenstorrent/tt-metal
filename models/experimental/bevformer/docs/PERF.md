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

| # | change | build | kernel | gap | wall | Δ wall |
|--:|---|---|---:|---:|---:|---:|
| 0 | [**baseline**](perf_reports/00-baseline.md) | Debug | 655.6 ms | 2416.5 ms | **3072.1 ms** | — |
| 1 | [SCA rebatch and scatter-back on device](perf_reports/01-sca-rebatch-on-device.md) | Debug | 682.0 ms | 218.3 ms | **900.2 ms** | **−2171.9 ms** |
| — | *stage 1 code, re-measured on Release* | Release | 748.8 ms | 40.8 ms | **789.6 ms** | — |
| 2 | [MSDA through the fused ttnn op](perf_reports/02-fused-msda.md) | Release | 557.6 ms | 37.9 ms | **595.5 ms** | **−194.1 ms** |

`kernel` = summed `DEVICE KERNEL DURATION`. `gap` = summed `OP TO OP LATENCY`, i.e. the time the
device spent idle between ops waiting on host dispatch. `wall` = kernel + gap, per layer.

**Stages 0–1 and stage 2 are not on the same build, so their deltas do not sum across that line.**
Op-to-op latency is host dispatch cost, and a Debug build inflates it ~5×: the same stage-1 code
measures 218.3 ms of gap on Debug and 40.8 ms on Release, at an identical device-op count of 146.
Each stage's Δ is against a re-measurement of the previous stage on its own build; the un-numbered
row above is that re-measurement for stage 1.

**Stage 1: −70.7% of wall clock**, trading +26.4 ms of kernel for −2198.2 ms of host gap.

**Stage 2: −24.6% of wall clock**, all of it kernel — 191.2 ms out of the two deformable-attention
calls, at a PCC of 0.999611 against 0.999608. Kernel is now **94%** of wall clock, so the remaining
host-round-trip work is bounded by 40.8 ms.

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

## Report format

One file per landed change, `NN-slug.md`, containing: source commit, kernel/gap after, delta from
the previous stage, what the change was and why, and the per-op-code table. Same numbers, same
harness, same signposts every time — so the deltas sum.
