#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Profile the fused decoder's eight perf windows with Tracy, plus the two
# multi-chunk functional baselines, and render
# tt-perf-report tables/CSVs.  One device job at a time; watcher must NOT be set.
set -euo pipefail
cd "$(dirname "$0")/../../../../../.."          # repo root
D=$PWD/models/autoports/meta_models_muse_glimmer_30b/doc/fused_decoder
T=models/autoports/meta_models_muse_glimmer_30b/tests/test_fused_decoder.py
mkdir -p "$D/tracy/sliding" "$D/tracy/full" "$D/logs"

report () {  # $1=ops csv  $2=out prefix  $3=start signpost  $4=end signpost
  tt-perf-report "$1" --start-signpost "$3" --end-signpost "$4" --no-summary --no-advice > "$2.txt"
  tt-perf-report "$1" --start-signpost "$3" --end-signpost "$4" --csv "$2.csv" --no-advice > "$2.console.log"
}

for kind in sliding full; do
  for seq in 8192 16384; do
    tag=prefill; [ "$seq" = 16384 ] && tag=prefill_16384
    MG_PERF_DECODE_ITERS=8 python -m tracy -r -p -v -m pytest "$T::test_perf_prefill[$seq-$kind]" \
        2>&1 | tee "$D/logs/tracy_${tag}_$kind.log"
    cp "$(ls -t generated/profiler/reports/*/ops_perf_results_*.csv | head -1)" "$D/tracy/$kind/${tag}_ops.csv"
    report "$D/tracy/$kind/${tag}_ops.csv" "$D/tracy/$kind/${tag}_perf_report" PERF_PREFILL PERF_PREFILL_END
  done

  MG_PERF_DECODE_ITERS=8 python -m tracy -r -p -v -m pytest "$T::test_perf_decode_traced[2048-$kind]" \
      2>&1 | tee "$D/logs/tracy_decode_$kind.log"
  cp "$(ls -t generated/profiler/reports/*/ops_perf_results_*.csv | head -1)" "$D/tracy/$kind/decode_ops.csv"
  report "$D/tracy/$kind/decode_ops.csv" "$D/tracy/$kind/decode_perf_report" PERF_DECODE PERF_DECODE_END

  MG_PERF_DECODE_ITERS=8 python -m tracy -r -p -v -m pytest "$T::test_perf_decode_traced[131071-$kind]" \
      2>&1 | tee "$D/logs/tracy_decode_131071_$kind.log"
  cp "$(ls -t generated/profiler/reports/*/ops_perf_results_*.csv | head -1)" "$D/tracy/$kind/decode_131071_ops.csv"
  report "$D/tracy/$kind/decode_131071_ops.csv" "$D/tracy/$kind/decode_131071_perf_report" PERF_DECODE PERF_DECODE_END
done

# The 16384 (multi-chunk) before/after needs the *functional* baseline captured
# the same way.  The functional test takes its length from an env var.
FT=models/autoports/meta_models_muse_glimmer_30b/tests/test_functional_decoder.py
for kind in sliding full; do
  MG_PERF_PREFILL_SEQ=16384 python -m tracy -r -p -v -m pytest "$FT::test_perf_prefill[$kind]" \
      2>&1 | tee "$D/logs/tracy_prefill_16384_baseline_$kind.log"
  cp "$(ls -t generated/profiler/reports/*/ops_perf_results_*.csv | head -1)" \
      "$D/tracy/$kind/prefill_16384_baseline_ops.csv"
  report "$D/tracy/$kind/prefill_16384_baseline_ops.csv" "$D/tracy/$kind/prefill_16384_baseline_perf_report" \
      PERF_PREFILL PERF_PREFILL_END
done

echo "=== dropped-marker integrity check (all must be 0) ==="
grep -c "markers were dropped" "$D"/logs/tracy_*.log || true

# The repo's check-large-files pre-commit hook rejects anything over 500 KB, and
# the raw decode captures are ~1 MB.  gunzip -k them to re-run tt-perf-report.
for f in "$D"/tracy/*/decode_ops.csv "$D"/tracy/*/decode_131071_ops.csv \
         "$D"/tracy/*/prefill_16384_ops.csv "$D"/tracy/*/prefill_16384_baseline_ops.csv; do
  [ -f "$f" ] && [ "$(stat -c %s "$f")" -gt 400000 ] && gzip -9 -f "$f"
done
