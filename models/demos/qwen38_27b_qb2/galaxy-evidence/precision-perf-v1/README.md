# BFP4/BFP8 performance cost: estimate and queued measurement

No measured BFP8 throughput result was available when this queue was launched.
The short reference comparison improved numerical error; it did not time
inference. Model-loading durations are not token-throughput measurements.

The explicit bandwidth-only estimate in `bandwidth-estimate.json` uses one
read of padded DRAM-sharded decoder weights plus the unchanged BFP8 head:
**3.925 GB/chip/step for BFP4, 7.113 GB for BFP8**, a 3.188 GB increase. This is
different from the 5.808 GiB resident-memory increase, which includes both
interleaved and DRAM-sharded representations. KV remains BFP8 and recurrent
state remains FP32. At identical effective bandwidth and with only these
weight/KV/state streams, the estimated output-throughput reductions are:

| Context | Active users per TP4 replica | Bandwidth-only output TPS reduction |
|---|---:|---:|
| 16,384 | 16 | 30.1% |
| 32,768 | 16 | 24.7% |
| 131,072 | 8 | 18.9% |
| 262,016 | 4 | 19.3% |

These are **unmeasured model estimates**, not bounds or predictions of the
complete engine. They exclude fixed overhead, HiFi2 arithmetic cost, CCL,
temporary traffic, real reader efficiency and prefill. The lower concurrency
at long contexts reflects the current token-pool scale; no admission result is
implied. In particular, this table does not estimate input-token throughput.

To replace the estimate, `qwen38-precision-perf-v1-20261009.service` was queued
at **Oct 9, 10:07:18 UTC**. It waits for the exact chunked-state job, whose
dependencies are BFP8 GPQA and its completion auditor. It uses the shared
hardware lock and therefore also serializes with the queued container CPU
checks; their relative order after chunked-state is unspecified.

The matched comparison has three policies: BFP4/LoFi, BFP4/HiFi2, BFP8/HiFi2.
All keep the BFP8/HiFi2 head, native recurrence, accurate-full-tile attention,
BFP8 KV and FP32 state. It measures one physical TP4 replica, batch 16, 16K
and 32K prompts, 128 output tokens, one warmup and three timed repeats per
cell. The existing full-model sweep separately reports input-token throughput,
prefill time, decode output TPS/TSU and combined throughput. It checks greedy
repeatability, zero timed trace captures and allocator snapshots. It does not
claim full-Galaxy scaling, benchmark accuracy or production-serving latency.

Native preflight passed 18 tests and verified the frozen imports and two-cell
plan. The source manifest matches the active BFP8 accuracy snapshot exactly;
the controller is separately frozen. PID 1491803, invocation
`122b1077bed2438d9802360ab6277514`. Each policy has a 45-minute pytest bound
and 50-minute process-group bound; the 14-hour outer bound includes dependency
waiting. A failed test or unproven cleanup stops the sequence. It survives
disconnect, not reboot; serving defaults remain unchanged.

Remote roots are `precision-perf-source-v1`, `precision-perf-control-v1` and
`precision-perf-v1` under `/home/ttuser/qwen38-artifacts-20261007`. Launch,
source manifest, controller, preflight tests and initial service state are
preserved here. All numerical throughput entries above are estimates until
completed native sweep receipts are collected.
