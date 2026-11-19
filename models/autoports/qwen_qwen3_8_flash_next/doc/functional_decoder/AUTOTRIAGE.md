# AUTOTRIAGE

## Diagnosis

The complete resident PLE lookup selects a native gather kernel whose work
scales with every source tile for every output tile. The full packed table
has 6250030 source tiles; its size disables the kernel's small-source bitmap
shortcut. The evidence supports excessive finite work, not a proven deadlock.

## Triage evidence

Captured chip 0 while `probe_ple_decoder.py` remained live under its original
1800-second timeout. No process termination or reset preceded capture.
Files: `triage/ple_full/tt-triage.txt`, `triage/ple_full/summary.txt`,
`logs/ple_full_triage.log`, `logs/ple_full_host_stack.log`.

Running op 26 is `GatherDeviceOperation`, source UINT32 TILE
`[1,1,32,200000960]`, indices UINT32 TILE `[1,1,32,10240]`, on 110 cores.
Previous op 25 is reshape of `[512,20]`. Readers are inside
`gather_reader_single_row_multi_core.cpp` lines 167–181; writers wait for
input CB space at line 64. ARC heartbeats, DRAM, NoC, binary-integrity,
lightweight-assert and CB checks report pass. The host stack waits for
outstanding reads in mesh synchronization. Allocation and upload succeeded
far enough to execute the full-table gather alongside the real decoder.

## Source evidence

Source directory:
`ttnn/cpp/ttnn/operations/data_movement/gather/device/kernels/dataflow/`.

| Resource/work | Producer | Consumer | Count per output tile |
|---|---|---|---:|
| Source-tile CB | writer reads source tiles | reader examines then pops | 6250030 |
| Index-tile CB | reader reads indices | reader scans same tile | 1 |
| Scalar index comparisons | reader | reader | 6250030 × 1024 |
| Output-tile CB | reader pushes after entire scan | writer writes output | 1 |

The reader bitmap is enabled only when `ceil(Wt_input/32) <= 256`, namely
at most 8192 source tiles. Here that condition is false. The writer always
streams all source tiles. Both sides agree on counts; source inspection
does not reveal an unbalanced producer/consumer protocol. There are 320
output tiles distributed across 110 cores, so most cores repeat the scan
three times. Small-table tests did not expose this scaling behavior.

## Downstream effects

Writer CB waits and host synchronization waits follow the reader's scan.
They are not evidence that the table does not fit DRAM or that hardware
needs a reset. A future timeout may leave dispatch unhealthy, which must
be checked separately before another device job.

## Proposed fix

Keep the complete striped table resident and implement a stage-local generic
TTNN data-movement op that directly reads the indexed tile and scalar using
TensorAccessor addressing. Work must scale with requested values, not table
width. Preserve exact UINT32 payloads, BF16 scales, tile-face addressing,
alignment and caller-owned trace buffers. Do not weaken the table domain,
hash semantics or PCC threshold.

## Uncertainty

The second capture (`triage/ple_full_progress/tt-triage.txt`) at approximately
20.5 minutes shows operation 1138, GatherDeviceOperation inside trace 0, with
indices [32,320]. The first capture showed operation 26 with indices
[32,10240]. Thus the process completed prefill and advanced into decode;
the source scan is finite slow work, not evidence of a circular-buffer deadlock.

The original run completed and closed normally at 17:02:40 UTC, before its
1800-second timeout. Exit 1 was the numerical acceptance assertion, not timeout
or a hardware exception. Total command wall time was approximately 1791 seconds;
this is not a warmed kernel latency. No reset or recovery was needed.
`ple_decoder_probe.json` records complete-table prefill PCC 0.99606168 but first
decode PCC 0.99478281, below 0.995. All repeated outputs and states were exact.
The replacement requires small-table parity,
full-domain address tests, complete resident-table reconstruction, watcher,
trace replay and full real-weight decoder acceptance. AutoFix continues.
