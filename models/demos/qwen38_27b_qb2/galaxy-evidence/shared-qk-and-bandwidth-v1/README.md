# Shared Q/K, physical bandwidth calibration and completed stage profiles

Observed Oct 8, 2026 UTC on 10.228.203.98. Native Metal and checkpoint pins remain
unchanged. All physical measurements use one TP4 replica; no eightfold number is
claimed as physical Galaxy throughput. Fixed BFP8 KV, BF16 activations and FP32
recurrent state; existing BFP4 projection weights are unchanged.

## Shared FP32 Q/K normalization

The old fused recurrence repeats normalization for each of three value heads
and four value-column partitions. The candidate normalizes compact Q/K once,
uses persistent FP32 scratch, removes Q/K repeat-interleave, and reads the shared
vectors from the recurrence. Arithmetic and state precision are unchanged.

| Batch | Old adapter (us) | Shared adapter (us) | Speedup |
|---:|---:|---:|---:|
| 32 | 518.90 | 396.91 | 1.307x |
| 16 | 341.48 | 270.56 | 1.262x |
| 8 | 239.32 | 204.31 | 1.171x |
| 64 | 904.96 | 652.01 | 1.388x |
| 1 | 141.85 | 145.87 | 0.972x |

All five control/shared/control groups passed identical output and state hashes
on four ranks, including 4,096 changing-input updates and alternating scratch
allocations. Control drift stayed below 0.44%. The real-weight layer-0 GDN block
also passed with bit-identical projected output:

| Batch | Old block (us) | Shared block (us) | Speedup |
|---:|---:|---:|---:|
| 32 | 1092.41 | 973.30 | 1.122x |
| 16 | 730.64 | 667.13 | 1.095x |

The block includes projections, convolution, gates, recurrence, output norm and
output projection; it excludes MLP. This is not a full-model speedup. At B32,
119.12 us saved across 48 recurrent layers would suggest about 6.7% model uplift
against the earlier 91.32-ms step if that saving transfers unchanged. The active
matched comparison must establish the actual result.

The opt-in `single_step_shared_qk` policy preallocates scratch before traces.
B1 retains the existing fused path because sharing was slower. B2/B4 also retain
it because they were not in this component qualification. B8/16/32/64 use shared
scratch; B64 component support does not remove the full-model B64 projection guard.
No default or serving deployment was changed.

Sources and raw results: [component](gdn-shared-qk-v1/shared.json),
[real block](gdn-shared-qk-layer-v1/layer.json), and their `queue.json` manifests.

## Physical read calibration

The raw probe reads each unique 1088-byte page once. Four chips execute together;
reported GB/s is bytes per chip divided by synchronized traced call latency,
including dispatch and marker work. It is not a hardware-counter reading or an
attention benchmark. Each timing is the median of five groups of 20 trace replays.

| Bytes per chip/call | 120-worker interleaved | Bank-local tile reads | Bulk, row placement | Bulk, bank-adjacent placement |
|---:|---:|---:|---:|---:|
| 272 MiB | 163.0 | 190.0 | 340.3 | 499.3 |
| 544 MiB | 168.6 | 190.0 | 342.1 | 505.5 |
| 1088 MiB | 170.7 | 190.5 | 343.0 | 508.2 |

All rates are GB/s. Bulk columns use 15 pages/packet and four independent TRID
slots. The best across all variants is 499.5/505.5/508.2 GB/s at these volumes,
97.6-99.3% of the assumed 512-GB/s ceiling. Eight pages/packet already reaches
499.0/505.1/507.9 GB/s; depth eight provides no material gain over depth four.
Depth one falls to 218-220 GB/s. Moving bulk readers from a generic row to the
bank-adjacent cores improves this probe by about 47-48%. Moving tile-at-a-time
readers alone does not improve their roughly 190-GB/s result.

This isolates **packet aggregation plus placement plus in-flight buffering** as
a useful combination. It does not show a 3x attention gain: actual attention
already achieves substantially more bandwidth than this probe's intentionally
simple interleaved control. The best raw result excludes unpacking, softmax,
matmul, cross-core redistribution and arbitrary page-table traversal.

All 11 variants passed full-byte input/output verification at 2,056 pages with
two live allocations and all four ranks, including packet tails. Large timing
volumes verify first/last words of every completed packet and an ordered
aggregate checksum, not every payload byte. Baseline timing drift was <=0.33%.
The [raw receipt](dram-read-probe-v2/probe.json) retains every sample and variant.

Next reader experiment: preserve the model's page-table and logical KV layout,
then measure bank-adjacent bulk reads **with delivery to compute workers and
backpressure**. Include the redistribution cost and compare against the current
attention reader before changing model kernels. Four slots and eight-page
packets are the conservative starting point; no evidence yet supports larger
buffers or a runtime KV-layout migration.

## Stage attribution

All 12 bounded profiles completed: native and single-step at 16K/B16/B32,
32K/B16/B32, 128K/B16 and near256K/B8. Their raw per-op CSVs are losslessly
compressed, alongside source hashes, test receipts and derived summaries. Local
reanalysis reproduced every operation/stage table from those CSVs.

At 32K/B32 on rank 0, recurrence preparation plus recurrence falls from 1,168.5
us/63 device-op rows to 492.0 us/33 rows. The whole GDN layer including MLP falls
from 1,870.5 to 1,196.9 us. The separate attention layer stays about 1,900 us,
including about 1,499 us in SDPA. Packed convolution still costs about 150 us;
RoPE about 109 us. These are sums of eager kernel intervals in two real-weight
layers with synthetic populated caches. Device-op rows are not program counts;
RISC intervals include waits and overlap. This does not complete the full P0
critical-path, compute or collective calibration.

## Full-model capacity comparison completed

Near256K/B8: native **89.983** versus single-step **89.831 output tok/s per TP4**
(-0.169%, effectively flat). Both completed all three measured fresh-prefill
repeats with clean device shutdown. Current single-step prefill was 3,687 input
tok/s and TTFT about 568.60 s for the entire batch. This comparison is distinct
from the new shared-Q/K experiment and is not reference-eval qualification.

## Persistent full-model comparison

Launched at 09:09:38 UTC as `qwen38-gdn-shared-qk-model-v1-20261008.service`.
Eight fresh model processes: single-step control then shared Q/K at 32K/B32,
16K/B32, 128K/B16 and 262016/B8. Each uses one warmup and three measured
fresh-prefill runs, 128 output tokens and identical offered concurrency. The
comparison stops on changed output hashes, precision, source or workload.
350 CPU tests plus 40 subtests passed before launch. The launch manifest and
queue snapshot are evidence of a running experiment, not its success.

The service has a 20-hour bound, each arm a two-hour bound, 128-GiB host-memory
cap, eight-CPU quota and `/tmp/tt-device.lock`. Artifacts have per-file/total
limits and a free-space floor. It survives SSH/session disconnect; automatic
resume after host reboot is not configured. Stop only its named user service
to cancel; preserve checkpoints, native installation and immutable source snapshots.

## Failed attempts retained

Bandwidth v1 stopped in CPU preflight: ten negative tests omitted the existing
`expect_error` fixture's required message argument. No device opened. Corrected
v2 passed 341 tests plus 40 subtests, then the physical probe passed in 42.57 s.
A preceding interrupted SSH staging call created no remote source/service;
a mistaken source-copy path was corrected before the CPU-preflight snapshot.
The failed v1 JUnit receipt is retained here.

Reference GPQA requalification, physical eight-replica scaling and model-level
promotion remain outstanding. Speculative decoding remains opt-in and requires
higher total committed output throughput at matched offered concurrency.
