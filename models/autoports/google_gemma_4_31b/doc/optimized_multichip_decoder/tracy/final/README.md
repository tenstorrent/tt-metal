# Final multichip decoder profiler provenance

## Capture

- Date: 2026-07-14 UTC
- Repo HEAD at capture: `e1a3f724877`
- Hardware: 4x Blackhole P150b, `MeshShape(1, 4)`, firmware 19.9.0
- Tool: `tt-perf-report==1.2.7`
- Workload: one real-shape representative layer for sliding/full attention,
  prefill-128 and a traced decode replay after a 32-token populated cache.
- Source enriched CSV:
  `capture/reports/gemma4_mc_final/2026_07_14_10_33_16/ops_perf_results_gemma4_mc_final_2026_07_14_10_33_16.csv`
- Source SHA-256:
  `aa5e8674b7be30ad491a400b1a66494bf915e112eed614b422d75196b23d8cfd`
- Test result: `profile_tests.xml` (4 passed).

Capture command:

```bash
GEMMA4_MULTICHIP_PROFILE=1 MPLCONFIGDIR=/tmp/mpl \
LD_LIBRARY_PATH=$PWD/build/lib:$LD_LIBRARY_PATH \
python -m tracy -r -p -v --check-exit-code \
  -o models/autoports/google_gemma_4_31b/doc/optimized_multichip_decoder/tracy/final/capture \
  -n gemma4_mc_final -m pytest -vv -s --tb=short \
  models/autoports/google_gemma_4_31b/tests/test_multichip_decoder.py \
  -k test_multichip_profile
```

The outer shell pipeline returned 1 because `tee` tried to open `capture.log`
before Tracy created its parent directory.  This was a console-log plumbing
error, not a profiler/test failure: the command's `--check-exit-code` path ran
all four tests to completion, `profile_tests.xml` records four passes, Tracy
generated the enriched CSV, and every expected signpost is present.  Raw
192 MB device logs and duplicate 16 MB traces were removed after report
generation; the enriched source, filtered CSVs, reports, summaries, hashes,
and test XML remain.

## Reports

Each directory contains an advice-enabled human table (`report.txt`), filtered
CSV (`perf_report.csv`), summary CSV/PNG, command chatter, and hashes:

- `sliding_prefill`: `MC_sliding_attention_PREFILL` to `_END`
- `full_prefill`: `MC_full_attention_PREFILL` to `_END`
- `sliding_decode`: `MC_sliding_attention_DECODE` to `_END`
- `full_decode`: `MC_full_attention_DECODE` to `_END`

The command pattern was:

```bash
tt-perf-report "$SOURCE_CSV" \
  --start-signpost "$START" --end-signpost "$END" --no-color \
  --summary-file "$OUT/summary.csv" > "$OUT/report.txt"
tt-perf-report "$SOURCE_CSV" \
  --start-signpost "$START" --end-signpost "$END" --no-color \
  --csv "$OUT/perf_report.csv" --summary-file "$OUT/summary_csv_mode.csv" \
  > "$OUT/perf_report.console.log"
```

Durations in the filtered CSV are microseconds.  Advice remained enabled for
the human tables.  The decode device-op sums are 436.768 us sliding and
489.673 us full.  Multi-device merged op-to-op gaps are not a critical path:
rows selected from different devices overlap, so accounting uses the device-op
sum and same-workload warmed host median.
