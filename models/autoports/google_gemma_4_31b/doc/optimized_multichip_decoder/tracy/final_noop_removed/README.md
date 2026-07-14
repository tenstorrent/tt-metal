# Final default profiler provenance after no-op removal

- Date: 2026-07-14 UTC.
- Repo starting HEAD: `e1a3f724877`; capture used the live Stage 05
  worktree after both review remediation passes.
- Hardware: 4x Blackhole P150b, `MeshShape(1, 4)`, firmware 19.9.0.
- Tool: `tt-perf-report==1.2.7` with advice enabled.
- Workload: real-shape sliding/full layers, prefill-128, and one traced
  decode replay after a 32-token populated cache.
- Tests: `profile_tests.xml`, four passed.
- Enriched source CSV:
  `capture/reports/gemma4_mc_final_noop_removed/2026_07_14_12_23_36/ops_perf_results_gemma4_mc_final_noop_removed_2026_07_14_12_23_36.csv`.
- Source SHA-256:
  `f04fda81d3b5ffdae5a63261378a9d73e959746245aa8fe73b99463c529652e6`.

Capture command:

```bash
GEMMA4_MULTICHIP_PROFILE=1 MPLCONFIGDIR=/tmp/mpl \
LD_LIBRARY_PATH=$PWD/build/lib:$LD_LIBRARY_PATH \
python -m tracy -r -p -v --check-exit-code \
  -o models/autoports/google_gemma_4_31b/doc/optimized_multichip_decoder/tracy/final_noop_removed/capture \
  -n gemma4_mc_final_noop_removed -m pytest -vv -s --tb=short \
  models/autoports/google_gemma_4_31b/tests/test_multichip_decoder.py \
  -k test_multichip_profile \
  --junitxml=models/autoports/google_gemma_4_31b/doc/optimized_multichip_decoder/tracy/final_noop_removed/profile_tests.xml
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

The final decode device-op sums are 428.14175 us sliding and 480.92775 us
full. The second review's redundant MLP pre-collective BFP8-to-BFP8 typecast
is absent. The remaining BF16-to-BFP8 cast is the attention partial and the
two BFP8-to-BF16 casts restore the two collective outputs to the declared
layer boundary. Compared with the immediately preceding selected-default
capture, device-op sums improve by 1.49725 us sliding and 3.21825 us full.

Both async reductions use BFP8 input pages and BFP8 output during decode;
prefill retains BF16 collectives. Raw 225 MB device logs, 259 MB intermediate
host timing logs, and duplicate 18 MB traces were removed after report
generation. The enriched source CSV, filtered reports, summaries, hashes,
capture log, and test XML remain.
