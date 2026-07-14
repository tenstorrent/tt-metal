# Exact fused all-gather + matmul coherent-family provenance

- Date: 2026-07-14 UTC.
- Repo starting HEAD: `e1a3f724877`; capture used the live Stage 05
  worktree after the exact-fused-family remediation.
- Hardware: 4x Blackhole P150b, `MeshShape(1, 4)`, firmware 19.9.0.
- Tool: `tt-perf-report==1.2.7` with advice enabled.
- Tests: `profile_tests.xml`, two passed (sliding layer 0 and full layer 5).
- Enriched source CSV:
  `capture/reports/gemma4_mc_fused_agmm_coherent/2026_07_14_12_58_17/ops_perf_results_gemma4_mc_fused_agmm_coherent_2026_07_14_12_58_17.csv`.
- Source SHA-256:
  `543f5f832d3b41e0bbabe74d1d25277e1677ddb0c0af6ebd10c5a9a9fd2b8cd7`.

Capture command:

```bash
GEMMA4_MULTICHIP_FUSED_AGMM_COHERENT=1 \
GEMMA4_MULTICHIP_FUSED_AGMM_PROFILE=1 MPLCONFIGDIR=/tmp/mpl \
LD_LIBRARY_PATH=$PWD/build/lib:$LD_LIBRARY_PATH \
python -m tracy -r -p -v --check-exit-code \
  -o models/autoports/google_gemma_4_31b/doc/optimized_multichip_decoder/tracy/fused_agmm_coherent/capture \
  -n gemma4_mc_fused_agmm_coherent -m pytest -vv -s --tb=short \
  models/autoports/google_gemma_4_31b/tests/test_multichip_decoder.py \
  -k fused_agmm_coherent_boundary \
  --junitxml=models/autoports/google_gemma_4_31b/doc/optimized_multichip_decoder/tracy/fused_agmm_coherent/profile_tests.xml
```

The real checkpoint weights were repacked from the production local-K/full-N
form to full-K/local-N. The Ring candidate uses one link, dimension 3, four
hard-coded kernel transfers, persistent gathered-input buffers, DRAM
interleaved AG/matmul storage, an 8x6 grid, and a 1x1 output subblock. Its
coherent endpoint keeps H/TP local through fused O, distributed norms and
residuals, fused packed gate/up, fused down, and the next fused QKV. It does
not immediately restore the production replicated-residual contract.

Correctness passed at PCC 0.999751962 sliding and 0.999694822 full. Normal
12-replay warmed medians were 0.384409 -> 0.952576 ms sliding (2.478x slower)
and 0.413468 -> 1.055626 ms full (2.553x slower). The corresponding device-op
sums were 349.77 -> 925.49 us and 384.95 -> 1026.01 us.

Separate fused-projection-plus-immediate-gather compatibility candidates
also lost. Sliding O was 49.21 -> 105.14 us and down was 72.47 -> 221.11 us;
full O was 72.59 -> 188.84 us and down was 72.85 -> 222.30 us. Thus the
rejection does not depend on measuring the lower-movement family only after
an immediate restore to the old boundary.

Reports use these signpost pairs for both `sliding_attention` and
`full_attention`: `BASELINE_SPINE`, `COHERENT_SPINE`, `BASELINE_O`,
`TRAILING_O`, `BASELINE_DOWN`, and `TRAILING_DOWN`, each prefixed with
`AGMM_<kind>_` and terminated by `_END`.

```bash
tt-perf-report "$SOURCE_CSV" --start-signpost "$START" \
  --end-signpost "$END" --no-color --summary-file "$OUT/summary.csv" \
  > "$OUT/report.txt"
tt-perf-report "$SOURCE_CSV" --start-signpost "$START" \
  --end-signpost "$END" --no-color --csv "$OUT/report.csv"
```

Raw device/host profiler intermediates and duplicate traces (about 1.3 GB)
were removed after report generation. The enriched source CSV, all 12
filtered advice/CSV/summary report families, capture log, source hash, and
test XML remain.
