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
| — | *stage 1 code, re-measured on Release* | Release | 681.5 ms | 40.8 ms | **722.3 ms** | — |
| 2 | [MSDA through the fused ttnn op](perf_reports/02-fused-msda.md) | Release | 487.4 ms | 37.9 ms | **525.3 ms** | **−197.0 ms** |
| 3 | [camera fold without tiling a batch-of-one](perf_reports/03-camera-fold.md) | Release | 450.6 ms | 33.2 ms | **483.8 ms** | **−41.5 ms** |
| 4 | [flat, tile-clean sampling chain](perf_reports/04-flat-sampling-chain.md) | Release | 310.1 ms | 37.6 ms | **347.7 ms** | **−136.1 ms** |

`kernel` = summed `DEVICE KERNEL DURATION`. `gap` = summed `OP TO OP LATENCY`, i.e. the time the
device spent idle between ops waiting on host dispatch. `wall` = kernel + gap, per layer.

**Stages 0–1 and stages 2–4 are not on the same build, so the deltas do not sum across that line.**
Op-to-op latency is host dispatch cost and a Debug build inflates it ~5×. Device time is unaffected:
the same stage-1 code measures 682.0 ms of kernel on Debug and 681.5 ms on Release, at an identical
146 device ops, while the gap differs 218.3 against 40.8 ms. The un-numbered row is that
re-measurement, and stage 2 onwards is measured against it.

**Stage 1: −70.7% of wall clock**, trading kernel for −2198.2 ms of host gap.

**Stages 2–4: 681.5 → 310.1 ms of kernel, −54.5%**, at a PCC gate held at 0.999611 throughout. All
of it is device time. Kernel is **89%** of wall clock, so what remains of the host-round-trip work
is bounded by 37.6 ms.

`200×200` spatial cross attention runs as of stage 4; it had been failing on an allocation of 2.97
GB for a 23.2 MB tensor. The full `tests/pcc/` suite is 33 passed with nothing deselected.

**MSDAOperation is now 54% of kernel** — 167.6 ms in 5 calls of one experimental op, running at
roughly 1.3% of the DRAM roof. See [04](perf_reports/04-flat-sampling-chain.md) for the estimate.
Nothing in Python reaches it.

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
