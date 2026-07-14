# Final default profiler provenance after fused-family remediation

- Date: 2026-07-14 UTC.
- Repo starting HEAD: `e1a3f724877`; capture used the final live Stage 05
  worktree after the exact fused-AGMM remediation. Production decoder source
  is unchanged from the post-no-op selected default.
- Hardware: 4x Blackhole P150b, `MeshShape(1, 4)`, firmware 19.9.0.
- Tool: `tt-perf-report==1.2.7` with advice enabled.
- Workload: real-shape sliding/full layers, prefill-128, and one traced decode
  replay after a 32-token populated cache.
- Tests: `profile_tests.xml`, four passed.
- Enriched source CSV:
  `capture/reports/gemma4_mc_final_post_fused_review/2026_07_14_13_26_44/ops_perf_results_gemma4_mc_final_post_fused_review_2026_07_14_13_26_44.csv`.
- Source SHA-256:
  `2266ba3301d17ab5eaa9a0f68aaf91ae2b975df75397ec4f5b8ed5a8689fa594`.

Capture command:

```bash
GEMMA4_MULTICHIP_PROFILE=1 MPLCONFIGDIR=/tmp/mpl \
LD_LIBRARY_PATH=$PWD/build/lib:$LD_LIBRARY_PATH \
python -m tracy -r -p -v --check-exit-code \
  -o models/autoports/google_gemma_4_31b/doc/optimized_multichip_decoder/tracy/final_post_fused_review/capture \
  -n gemma4_mc_final_post_fused_review -m pytest -vv -s --tb=short \
  models/autoports/google_gemma_4_31b/tests/test_multichip_decoder.py \
  -k test_multichip_profile \
  --junitxml=models/autoports/google_gemma_4_31b/doc/optimized_multichip_decoder/tracy/final_post_fused_review/profile_tests.xml
```

Reports used the corresponding `MC_<kind>_<mode>` and `_END` signposts:

```bash
tt-perf-report "$SOURCE_CSV" --start-signpost "$START" \
  --end-signpost "$END" --no-color --summary-file "$OUT/summary.csv" \
  > "$OUT/report.txt"
tt-perf-report "$SOURCE_CSV" --start-signpost "$START" \
  --end-signpost "$END" --no-color --csv "$OUT/perf_report.csv"
```

Device-op sums are 1156.52/1422.28 us for sliding/full prefill and
428.30025/481.62025 us for sliding/full decode. The source-current warmed host
medians are 2.4064085/2.205460 ms prefill and 0.463813/0.5166275 ms decode;
decode therefore leaves 35.51275/35.00725 us of traced replay/synchronization
overhead beyond the device rows.

Both decode reductions use BFP8 input pages and BFP8 output, while prefill
uses BF16. The redundant BFP8-to-BFP8 typecast remains absent: each decode
report has only the attention BF16-to-BFP8 conversion and two BFP8-to-BF16
layer-boundary restorations. Raw device/host profiler intermediates and
duplicate traces (about 711 MB) were removed after report generation. The
enriched CSV, four filtered advice/CSV/summary report families, capture log,
source hash, and test XML remain.
