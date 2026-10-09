# Remote-delivery placement/buffering sweep, Oct 9 2026 UTC

Completed at 07:06 UTC. All 54 delivery variants passed the bounded byte-equality
checks on four ranks. The timed sweep contains 162 delivery cases (three payload
sizes per variant) plus six before/after read-only controls. Every timed packet
checks first/last words and ordered markers on all four ranks; this does not
claim a full-byte readback of each large timed payload.

| BFP8 tiles per chip | Read-only control GB/s/chip | Best delivered GB/s/chip | Best receiver placement / packet tiles / depth |
|---:|---:|---:|---|
| 262,144 | 499.18 | 275.14 | opposite / 15 / 8 |
| 524,288 | 504.94 | 278.14 | opposite / 15 / 4 |
| 1,048,576 | 507.89 | 279.84 | opposite / 15 / 8 |

The best no-delay cases deliver about 55% of the measured read-only bandwidth.
This improves on the earlier 245-249 GB/s/chip diagnostic but still leaves a
large redistribution/receiver cost. Depth four and eight are almost equal at
15 tiles per packet. Deliberate consumer delays greatly reduce useful rate;
those are backpressure correctness cases, not production speed settings.
Read-only control drift stayed below 0.06% for all three payload sizes.

No production page-table traversal or attention math is included. Timings
include dispatch and receiver work and count useful read bytes once. These
results do not establish a full-attention or model throughput uplift; the mover
remains unpromoted. Inspect the sender/consumer protocol and receiver traffic
before replacing a passing attention path whose useful bandwidth is higher.

[Raw results and source hashes](probe.json), [JUnit](hardware.xml.gz),
[execution log](run.log.gz), [terminal parent queue](queue-terminal.json).

The parent queue completed all three stages and released its processes. Its
overall `passed=false` is the retained evaluation failure (GPQA 170/198 and
Tau3 3/12), not a device failure or a failed delivery test. Head-control G0
loading began automatically at 07:06:18 UTC after that clean terminal state.
