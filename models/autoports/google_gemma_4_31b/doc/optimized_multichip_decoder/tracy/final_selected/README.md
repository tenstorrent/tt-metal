# Final selected-default profiler provenance

- Date: 2026-07-14 UTC
- Repo starting HEAD: `e1a3f724877`; capture used the live Stage 05
  worktree after review remediation.
- Hardware: 4x Blackhole P150b, `MeshShape(1, 4)`, firmware 19.9.0.
- Tool: `tt-perf-report==1.2.7` with advice enabled.
- Workload: real-shape sliding/full layers, prefill-128, and one traced
  decode replay after a 32-token populated cache.
- Tests: `profile_tests.xml`, four passed.
- Enriched source CSV:
  `capture/reports/gemma4_mc_final_selected/2026_07_14_11_48_27/ops_perf_results_gemma4_mc_final_selected_2026_07_14_11_48_27.csv`
- Source SHA-256:
  `4a2240acac53d671b9b6c5363264c6309ff7b9aa59aec1aae40ad1b263410049`.

Capture command:

```bash
GEMMA4_MULTICHIP_PROFILE=1 MPLCONFIGDIR=/tmp/mpl \
LD_LIBRARY_PATH=$PWD/build/lib:$LD_LIBRARY_PATH \
python -m tracy -r -p -v --check-exit-code \
  -o models/autoports/google_gemma_4_31b/doc/optimized_multichip_decoder/tracy/final_selected/capture \
  -n gemma4_mc_final_selected -m pytest -vv -s --tb=short \
  models/autoports/google_gemma_4_31b/tests/test_multichip_decoder.py \
  -k test_multichip_profile \
  --junitxml=models/autoports/google_gemma_4_31b/doc/optimized_multichip_decoder/tracy/final_selected/profile_tests.xml
```

Each of `sliding_prefill`, `full_prefill`, `sliding_decode`, and
`full_decode` contains the advice table, filtered CSV, summary CSV/PNG, and
hashes. Reports used the corresponding `MC_<kind>_<mode>` and `_END`
signposts:

```bash
tt-perf-report "$SOURCE_CSV" --start-signpost "$START" \
  --end-signpost "$END" --no-color --summary-file "$OUT/summary.csv" \
  > "$OUT/report.txt"
tt-perf-report "$SOURCE_CSV" --start-signpost "$START" \
  --end-signpost "$END" --no-color --csv "$OUT/perf_report.csv" \
  --summary-file "$OUT/summary_csv_mode.csv" \
  > "$OUT/perf_report.console.log"
```

The final decode device-op sums are 429.639 us sliding and 484.146 us full.
Both async reductions use BFP8 input pages and BFP8 output; prefill retains
BF16 collectives. Raw 225 MB device logs, 260 MB intermediate host timing
logs, and duplicate 18 MB traces were removed after report generation. The
enriched source CSV, filtered reports, summaries, hashes, capture log, and
test XML remain.
