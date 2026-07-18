# Replay-only Tracy evidence

This is the first-review QKV32/O8 device profile.  It is retained as
intermediate provenance and is superseded for final numbers by
`../tracy_replay_final/`, which profiles the promoted QKV32/O32 default.  This
capture uses batch 1, logical sequence length 18,
QKV `in0_block_w=32`, and ten warmed trace replays.  The filtered window is
strictly `PERF_MULTI_DECODE_REPLAY` through
`PERF_MULTI_DECODE_REPLAY_END`; eager warm-up and trace construction are
outside it.

The profiler-enabled run reports 0.275304 ms/replay end to end.  The filtered
summary contains 360 operations across ten executions and 2,364.770 us of
summed merged-device time, or 236.477 us/replay.  The same-run gap is therefore
38.827 us/replay.  A targeted non-profiler audit at the same ten-replay count
reports 249.346 us/replay, attributing 25.958 us of the gap to profiler
instrumentation.  Increasing the unprofiled replay count from 10 to 100 lowers
E2E from 249.346 to 247.157 us, quantifying 2.189 us/replay of final-sync
amortization.  The remaining approximately 10.680 us is TTNN trace enqueue and
cross-run profiler/accounting overhead; replay already uses nonblocking
`execute_trace` calls followed by one amortized synchronization, so there is no
decoder-level host boundary left to remove.

At the unchanged 59.9-us theoretical weight/KV roofline, the replay-only
profile reaches 25.33% by device time and 21.76% by profiler-enabled E2E.  The
normal, uninstrumented 1,000-replay then-default QKV32/O8 path is 246.752 us
median.

Artifacts:

- `reports/2026_07_18_13_50_36/ops_perf_results_2026_07_18_13_50_36.csv.gz`:
  losslessly compressed raw ops provenance.  Uncompressed SHA-256:
  `e096e4156ab3fb73c763a31971987c85aa45d2adebd72ef9baf78972f41d1153`.
- `analysis/multi_decode_replay.csv`: advice-enabled filtered op table.
- `analysis/multi_decode_replay_summary.csv`: replay-only category summary.
- `analysis/multi_decode_replay.console.log`: report-generation provenance.
- `../logs/replay_only_profile_qkv32_o8.xml`, `replay_gap_nonprofile_10.xml`, and
  `replay_gap_nonprofile_100.xml`: same-harness E2E provenance.

A 100-replay profiler retry overflowed the fixed profiler DRAM marker buffers,
so its device rows are invalid and are not used.  The exact failure and cleanup
are retained in `../logs/profiler_replay100_failure.md`.
