# Exact-final replay-only Tracy evidence

This is the authoritative device profile for the final default after the
second independent-review repair.  The capture uses TP4 on the real 1x4 mesh,
batch 1, logical sequence length 18, QKV and O `in0_block_w=32`, and ten
warmed trace replays.  The filtered window is strictly
`PERF_MULTI_DECODE_REPLAY` through `PERF_MULTI_DECODE_REPLAY_END`; eager
warm-up and trace construction are outside it.

The machine-retained raw signpost timestamps bracket the ten profiler-enabled
replays from 11,428,587,354 through 11,431,500,008 ns, or 291.265 us/replay.
The filtered summary contains 360 operations and 2,367.64 us of summed
merged-device time, or 236.764 us/replay.  A targeted exact-default unprofiled
run reports 248.833 us at ten replays and 246.850 us at 100 replays.  Thus
profiler plus signpost instrumentation accounts for 42.432 us/replay,
final-sync amortization for 1.983 us/replay, and the remaining approximately
10.086 us is nonblocking TTNN trace enqueue and cross-run accounting overhead.

At the unchanged 59.9-us theoretical weight/KV roofline, the exact-final
profile reaches 25.30% by device time and 20.57% by profiler-enabled signpost
E2E.  The
normal uninstrumented 1,000-replay final default is 246.686 us median over six
independent processes.

Artifacts:

- `reports/2026_07_18_14_22_36/ops_perf_results_2026_07_18_14_22_36.csv.gz`:
  losslessly compressed raw ops provenance.  Uncompressed SHA-256:
  `7a18efa2929df638e6d2e1f95874ca83f23072b78da0c5740024602dec7c1001`.
- `analysis/multi_decode_replay.csv`: advice-enabled filtered op table.
- `analysis/multi_decode_replay_summary.csv` and `.png`: replay-only category
  summary and plot.
- `analysis/multi_decode_replay.console.log`: both report-generation passes,
  including actionable advice and the human-readable table.
- `../logs/replay_gap_final_nonprofile_10.xml` and
  `replay_gap_final_nonprofile_100.xml`: exact-default E2E gap provenance.

The generated `.logs` directory, duplicated device marker CSV, and host Tracy
trace occupied about 482 MB and were removed after preserving all report
inputs and outputs above.  Those scratch copies are not recoverable.
