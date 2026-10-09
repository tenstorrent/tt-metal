# GDN phase attribution, October 9

The physical TP4 diagnostic passed all eight cases and 24 calls, with identical
profiled/control outputs and FP32 state on every rank. The device closed cleanly.
This measures synthetic shared-Q/K recurrence, excluding model preparation,
weights, attention, MLP, serving and trace replay. There is no context-length
dimension in recurrent-state geometry.

## Results

Median eager device-kernel duration across three calls and four ranks:

| Batch | Input buffers | Without added zones (us) | With zones (us) |
| --- | --- | --- | --- |
| 32 | 1 | 234.84 | 239.11 |
| 32 | 2 | 158.95 | 162.14 |
| 16 | 1 | 132.57 | 135.06 |
| 16 | 2 | 91.67 | 93.50 |

The two-buffer setting is already selected by the model candidate. Its benefit
must not be added again to full-model projections. These controls still have the
base device profiler enabled; "without added zones" is not unprofiled serving.

With two buffers at B32, mean accumulated time per core, over all its work items:

| Processor | Region | Mean us |
| --- | --- | --- |
| Reader | Waiting for input CB capacity | 33.19 |
| Reader | DRAM issue and completion | 39.92 |
| Reader | L1 broadcast/zero preparation | 65.88 |
| Unpack | Waiting for inputs | 9.82 |
| Unpack | Delta region | 57.42 |
| Unpack | State-update region | 53.74 |
| Unpack | Output-reduction region | 31.41 |
| Writer | Waiting for computed state | 103.20 |
| Writer | State writes and waiting for computed output | 34.84 |
| Writer | Output packing, write and completion | 18.05 |

There are 120 active cores per rank, averaging 12.8 work items at B32 and 6.4 at
B16. Phase totals include those repeated items. Processor timelines overlap;
do not add reader, compute and writer times together. Compute regions include
unpack/pack synchronization and are not pure arithmetic measurements. The
~10 us input wait versus ~162 us kernel duration points toward the compute
pipeline/synchronization as the next recurrence lead, rather than predominantly
waiting for reader data. It does not establish a complete critical-path model.

NoC utilization, DRAM utilization and congestion fields remain empty. DRAM
issue/wait time does not measure link utilization or prove congestion. L1 reader
preparation is substantial but partly overlapped; removing all of it would not
subtract 66 us from step latency. A direct model-input preparation experiment is
queued separately to target the layout programs outside this recurrence.

## Reproduction and integrity

Run `python analyze.py --output /tmp/gdn-phase-analysis.json` from this directory.
The output includes all per-core aggregates and all-rank kernel distributions.
It validates signposts, exactly 96 rank/call mappings, raw fragment hashes,
matched start/end markers, and the exact expected work-item count for every
phase on each RISC. All 2,105,856 selected events pair correctly; none duplicate.
The initial parser mistakenly added a device ID to an already device-specific
global call ID; the four-rank count assertion caught it before publication.

`device-events/` contains untouched raw CSV rows partitioned by call and device,
including every GDN zone and processor-kernel boundary for the 24 calls. This
keeps each artifact under the repository's 500 KB file limit. The original
246 MB CSV also includes 188,480 upload/tilize/firmware records not used here.
Analysis from the partitions was checked against the complete original: every
kernel, phase and per-core statistic matches exactly.

`event-partitions.json` records selection and original hash. `capture.json`
records the original full CSV and host Tracy binary locations and hashes; those
large originals remain on the Galaxy and in `/private/tmp/qwen38-phase-raw-v2`.
`summary.json` excludes the bulky, reproducible per-core array. Timing-zone source,
JUnit, queue receipts and launch source manifest are retained in this directory
and the preceding `gdn-phase-profile-launch-v2` evidence directory.
