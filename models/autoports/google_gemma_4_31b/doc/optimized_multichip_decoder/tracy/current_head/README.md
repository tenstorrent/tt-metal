# Current-head profiler provenance

- Date: 2026-07-17 UTC.
- Git HEAD: `b68b16df75d121dcbe0128d6fa4cea98f993b870`.
- Hardware: four Blackhole P150b boards, `MeshShape(1, 4)`, firmware 19.9.0.
- Workload: one real sliding layer and one real full-attention layer;
  prefill-128 and one traced decode replay after a 32-token cache fill.
- Tests: `profile_tests.xml`, four passed.
- Tool: `tt-perf-report` 1.2.7 with advice enabled.
- Enriched CSV:
  `reports/gemma4_mc_current_head/2026_07_17_02_17_50/ops_perf_results_gemma4_mc_current_head_2026_07_17_02_17_50.csv`.
- Source SHA-256:
  `0454648836ec291e6ee090a10ff1202f7a65d9bfe0ef0016c7bd0f8cc8fa6002`.

Capture command:

```bash
GEMMA4_MULTICHIP_PROFILE=1 MPLCONFIGDIR=/tmp/mpl_stage05_revalidation \
LD_LIBRARY_PATH=$PWD/build/lib:$LD_LIBRARY_PATH \
python -m tracy -r -p -v --check-exit-code \
  -o models/autoports/google_gemma_4_31b/doc/optimized_multichip_decoder/tracy/current_head \
  -n gemma4_mc_current_head -m pytest -vv -s --tb=short \
  models/autoports/google_gemma_4_31b/tests/test_multichip_decoder.py \
  -k test_multichip_profile \
  --junitxml=models/autoports/google_gemma_4_31b/doc/optimized_multichip_decoder/tracy/current_head/profile_tests.xml
```

The outer console `tee` was opened before Tracy created the output directory,
so it reported a setup-time missing-directory error and made the shell pipeline
return 1.  Tracy itself completed: all four tests passed, the trace was saved,
post-processing finished, and the enriched CSV was generated.  The JUnit file,
CSV, signposts, reports, and hash are the result authorities.

Report commands used the four `MC_<kind>_<mode>` / `_END` signpost pairs:

```bash
tt-perf-report "$SOURCE_CSV" --start-signpost "$START" \
  --end-signpost "$END" --no-color --summary-file "$OUT/summary.csv" \
  > "$OUT/report.txt"
tt-perf-report "$SOURCE_CSV" --start-signpost "$START" \
  --end-signpost "$END" --no-color --csv "$OUT/perf_report.csv" \
  > "$OUT/perf_report.console.log"
```

| Window | Device ops | Device-time sum |
|---|---:|---:|
| sliding prefill | 33 | 1,153.2255 us |
| full prefill | 30 | 1,418.152 us |
| sliding decode | 59 | 427.9625 us |
| full decode | 58 | 481.827 us |

Runtime rows prove BFP8/LoFi attention, BFP4/LoFi packed gate-up/down, BFP8
cache use, and two BFP8 persistent async reductions in decode.  Prefill uses
BF16 collectives.  The same-dtype BFP8 no-op typecast remains absent.

Raw `.tracy`, device-profiler CSV, and Tracy host-op intermediates totaling
about 720 MB were removed after compact artifact verification.  The enriched
source CSV, source hash, four advice-enabled text reports, four filtered CSVs,
summary CSV/PNGs, and JUnit file remain.
