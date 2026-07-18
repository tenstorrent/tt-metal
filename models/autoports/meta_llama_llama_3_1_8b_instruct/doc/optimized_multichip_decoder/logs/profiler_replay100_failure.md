# Rejected 100-replay profiler retry

The targeted retry used the final default with the same replay-only signpost
and 100 trace replays.  The functional test passed and reported 0.267327 ms
profiler-enabled E2E, but the device profiler repeatedly reported:

`Profiler DRAM buffers were full, markers were dropped! ... bufferEndIndex = 12000. Please either decrease the number of ops being profiled or run read device profiler more often.`

Dropped markers invalidate device-time aggregation, so no number from that raw
device report is accepted.  The valid ten-replay profile stays below the
buffer limit.  The 100-replay JUnit/E2E log is retained as
`replay_only_profile_100.xml`; the 3.2-GB invalid Tracy directory was removed
after this compact failure evidence was written.  It is not recoverable from
the worktree.  The retry still informed the gap audit: the corresponding
unprofiled 100-replay run is valid and measures 0.247157 ms.
