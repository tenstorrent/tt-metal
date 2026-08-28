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
| 5 | [hoisted head permute, untilize before the head split](perf_reports/05-hoisted-layout-ops.md) | Release | 263.6 ms | 41.8 ms | **305.4 ms** | **−42.3 ms** |
| 6 | [sampling geometry on the SFPU](perf_reports/06-sfpu-geometry.md) | Release | 125.3 ms | 44.1 ms | **169.4 ms** | **−136.0 ms** |
| 7 | [the grid's point axis folded into its page](perf_reports/07-folded-grid-page.md) | Release | 112.4 ms | 40.3 ms | **152.7 ms** | **−16.7 ms** |
| 8 | [value heads addressed by byte offset](perf_reports/08-packed-value-heads.md) | Release | 91.9 ms | 41.6 ms | **133.4 ms** | **−19.3 ms** |
| 9 | [attn level runs addressed by byte offset](perf_reports/09-packed-attn-runs.md) | Release | 74.0 ms | 56.1 ms | **130.0 ms** | **−3.4 ms** |
| 10 | [a rank-3 grid packing head and level](perf_reports/10-packed-grid.md) | Release | 69.3 ms | 60.7 ms | **130.0 ms** | **0.0 ms** |

`kernel` = summed `DEVICE KERNEL DURATION`. `gap` = summed `OP TO OP LATENCY`, i.e. the time the
device spent idle between ops waiting on host dispatch. `wall` = kernel + gap, per layer.

**The gap column carries a spread the table does not show.** Three runs of identical stage-8 code
measured gaps of 63.3, 41.6 and 34.1 ms while kernel held at 91.9 / 91.9 / 92.3. The whole spread is
one op — the `bev_mask` readback that [candidate 1b](perf_optimization_candidates.md#1b-bound-max_len-statically)
is about — so wall clock here resolves to roughly ±15 ms. Every row from stage 2 to stage 7 is a
single run and is not being re-measured; treat their kernel deltas as sound and their wall deltas as
indicative. Stages 8, 9 and 10 quote the median of three.

**Stages 0–1 and stages 2–4 are not on the same build, so the deltas do not sum across that line.**
Op-to-op latency is host dispatch cost and a Debug build inflates it ~5×. Device time is unaffected:
the same stage-1 code measures 682.0 ms of kernel on Debug and 681.5 ms on Release, at an identical
146 device ops, while the gap differs 218.3 against 40.8 ms. The un-numbered row is that
re-measurement, and stage 2 onwards is measured against it.

**Stage 1: −70.7% of wall clock**, trading kernel for −2198.2 ms of host gap.

**Stages 2–10: 681.5 → 69.3 ms of kernel, −89.8%**, at a PCC gate held at 0.9996 throughout. All of
it is device time. Kernel is **53%** of the median wall clock; most of the rest is one host sync.

`200×200` spatial cross attention runs as of stage 4; it had been failing on an allocation of 2.97
GB for a 23.2 MB tensor. The full `tests/pcc/` suite is 33 passed with nothing deselected.

**MSDAOperation is 27.7 ms, 24.6% of kernel.** Stage 05 read it as 167.8 ms and 78× above its DRAM
roof, and concluded the fused kernel was slow. It was not: the compute kernel was idle for the whole
call, waiting on a reader doing per-point float maths at ~140 cycles an operation on a core with no
FPU. Moving that onto the SFPU collapsed the op to 29.5 ms without touching the sampling kernel —
see [06](perf_reports/06-sfpu-geometry.md).

**Effective bandwidth on a layout op tracks its page size, not its size.** A ROW_MAJOR page is the
last dimension, rounded up to the 32-byte DRAM alignment, so a 2-wide tensor spends 32 bytes per 4
bytes of data. Sorted by GB/s the layout ops sort by page width: 512 B → 38 GB/s, 64 B → 14 GB/s,
4 B → 2 GB/s, against a 288 GB/s roof. Folding the grid's point axis into its page deleted both
4-byte-page ops — see [07](perf_reports/07-folded-grid-page.md).

**`MSDAOperation` is the largest item again** at 29.1 ms, **42% of kernel** — for the first time
since stage 06. Layout plumbing is 15.4 ms, down from 60.5, and no single item in it is above
1.9 ms.

The fix was never the deformable member of the 19 `nlp_*` head-reshape ops: such an op would still
have to *produce* the head-major tensor, 92.6 MB written and read, when the cost is the page it
produces rather than the number of calls. Making the head and the level into addresses instead of
axes deletes the tensor entirely — see [08](perf_reports/08-packed-value-heads.md),
[09](perf_reports/09-packed-attn-runs.md) and [10](perf_reports/10-packed-grid.md). Only the output
(~1.9 ms) is still owed the same treatment. The ~10 ms after that is real movement, not plumbing.

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
