# Fusion qualification and compact-GDN recovery, October 10, 2026

At B16/32K/TP4, the latest measured full-model result remains **16.550 TSU**
(60.422 ms), up from 14.871 TSU. Thirty native TSU requires 33.333 ms:
another 27.089 ms / 44.83% reduction in step time. No speculative decoding
or lower-precision policy is included.

## Completed qualification

The flat-preparation/epilogue fusion finished full GPQA at 09:02 UTC:
**177/198 (89.39%)**, passing the unchanged 89.2% threshold. Six requests
reached the 65536-output-token limit and remain incorrect in the denominator.
All 198 samples completed. Duration: 3954.06 seconds (65m54s), concurrency 128
over eight TP4 workers. The GPQA mean decode rate of 18.067 TSU reflects its
variable request/context distribution and must not replace the fixed B16/32K
measurement. Aggregate client output during GPQA was 737.70 tokens/s including
the workload's prefill, scheduling and tail effects.

Eight-replica G0 passed before serving. Every one of its 34 model/policy hashes
was checked against the qualification controller's source manifest, mapping
`effective_precision_override` to
`precision_single_step_flat_prepare_epilogue_bfp8_all.json`. G0's own short
prompt/isolated-concurrent timing is not a B16/32K full-Galaxy throughput test.
The controller stopped its owned serving processes after evaluation; this
receipt does not assert that an endpoint is currently serving.

## Completed prefill-budget experiment

On the prior shared-QK decoder, increasing the internal prefill activation
budget from 32768 to 65536 tokens reduced B16/32K TTFT from approximately
98.66 to **93.09 seconds** (5.6% reduction; input throughput 5322 to 5640
tokens/s). At 16K, TTFT changed from 45.76 to 44.25 seconds. Before/after
controls were stable; generated token hashes match for both contexts.
Decode remained about 14.87 TSU at 32K. The budget change has not undergone
full task qualification or been combined with the new decoder fusion. It is
separate from the scheduler-admission budget discussed in the Shield audit.

## Compact-GDN failure and recovery

The first compact run passed all 26 standalone epilogue cases and stationary
real-weight comparisons at B16/B32/B8/B1. It then failed exact comparisons at
the first changing-input update. Its dependent projection sweep exited before
touching hardware. Both terminal receipts and the clean device teardown were
verified before restarting anything.

The changing-input test allocated the second independent session's persistent
input/state after capturing the first trace. Such allocations can reuse
addresses reserved by that trace's transient scratch. The fix allocates both
sessions and zero sources, warms both complete graphs and reset copies, and
only then captures either trace. Kernel code and equality requirements are
unchanged. The corrected physical test passes exact recurrent-state,
convolution-history and projected-output comparisons on every rank at updates
1/2/4/8/16/32/64 for both B16 and B32. This isolates the failing test to the
allocation/warmup ordering change; no individual conflicting address was
instrumented, so a particular alias is not claimed.

Corrected real-weight block timings:

| Batch | Prior fusion | Compact | Block time reduction | 48-layer saving extrapolation |
|---|---:|---:|---:|---:|
| 16 | 567.18 us | 350.97 us | 38.1% | 10.38 ms |
| 32 | 803.56 us | 463.80 us | 42.3% | 16.31 ms |

These blocks include projection, convolution, recurrence, epilogue and output
projection/reduction, but exclude MLP. B8/B1 retain their native path and
show no meaningful change. The B16 extrapolation applied to 60.422 ms would
give about **19.98 TSU**; it is not a full-model measurement, and no additional
layout/compact-front-end saving may be added again for the same operations.

## Persistent follow-up

- `qwen38-compact-gdn-v2-20261010.service`, initial PID 208588, invocation
  `43f1990817724573a9b130ed61f3ff0e`: corrected epilogue and real-layer gates
  passed; full-model B16/32K and B16/16K before/compact/after sweeps started
  at 18:02 UTC. It uses a new frozen source/output directory and the same
  native installation, with a hardware lock and bounded time/memory.
- `qwen38-projection-sweep-v3-20261010.service`, initial PID 208592, invocation
  `b8a146f6117847b2afba2ccbe1bee1f4`: waits on that exact compact invocation
  and its completed cleanup receipt. The original projection source is unchanged.
- Both survive client disconnection, not reboot. Prior failed runs and
  artifacts remain intact. No firmware, NFS or native-install changes.
- Host preflight: 512 CPU tests passed, 69 subtests passed, one unrelated skip;
  the real-layer hardware entry point collected and then passed physically.

The retained `receipts/` tree contains source manifests, launch commands,
GPQA/G0 results, prefill comparisons, original failure and corrected hardware
results. Queue snapshots are observations, not guarantees of future completion.
Full-model compact performance, full task qualification of compact mode,
30 native TSU, matched full-Galaxy throughput and release gates remain open.
