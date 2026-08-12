#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Profile the optimized decoder's ten perf windows with Tracy and render
# tt-perf-report tables/CSVs, with advice enabled.  One device job at a time;
# TT_METAL_WATCHER must NOT be set for any of this ($tt-device-usage).
#
# The 8192-token prefill and the four decode baselines are the *fused* stage's own
# committed captures under ../fused_decoder/tracy/, so they are not re-taken here;
# the 16384 baselines were re-captured by the fusing stage's chain and are reused
# from there too.
set -euo pipefail
cd "$(dirname "$0")/../../../../../.."          # repo root
D=$PWD/models/autoports/meta_models_muse_glimmer_30b/doc/optimized_decoder
T=models/autoports/meta_models_muse_glimmer_30b/tests/test_optimized_decoder.py
mkdir -p "$D/tracy/sliding" "$D/tracy/full" "$D/logs"

report () {  # $1=ops csv  $2=out prefix  $3=start signpost  $4=end signpost
  # Advice stays enabled ($optimize "Advice Policy"); the rendered table goes to
  # .txt and the --csv run's boilerplate to .console.log, never mixed.
  tt-perf-report "$1" --start-signpost "$3" --end-signpost "$4" --no-summary > "$2.txt"
  tt-perf-report "$1" --start-signpost "$3" --end-signpost "$4" --csv "$2.csv" > "$2.console.log"
}

for kind in sliding full; do
  for seq in 128 8192 16384; do
    tag=prefill
    [ "$seq" = 16384 ] && tag=prefill_16384
    [ "$seq" = 128 ] && tag=prefill_128
    python -m tracy -r -p -v -m pytest "$T::test_perf_prefill[$seq-$kind]" \
        2>&1 | tee "$D/logs/tracy_${tag}_$kind.log"
    cp "$(ls -t generated/profiler/reports/*/ops_perf_results_*.csv | head -1)" "$D/tracy/$kind/${tag}_ops.csv"
    report "$D/tracy/$kind/${tag}_ops.csv" "$D/tracy/$kind/${tag}_perf_report" PERF_PREFILL PERF_PREFILL_END
  done

  # 8 trace replays per decode window, matching both earlier stages exactly: the
  # functional stage established that 32 replays overflow the profiler's DRAM
  # marker buffer and silently under-count.
  MG_PERF_DECODE_ITERS=8 python -m tracy -r -p -v -m pytest "$T::test_perf_decode_traced[2048-$kind]" \
      2>&1 | tee "$D/logs/tracy_decode_$kind.log"
  cp "$(ls -t generated/profiler/reports/*/ops_perf_results_*.csv | head -1)" "$D/tracy/$kind/decode_ops.csv"
  report "$D/tracy/$kind/decode_ops.csv" "$D/tracy/$kind/decode_perf_report" PERF_DECODE PERF_DECODE_END

  MG_PERF_DECODE_ITERS=8 python -m tracy -r -p -v -m pytest "$T::test_perf_decode_traced[131071-$kind]" \
      2>&1 | tee "$D/logs/tracy_decode_131071_$kind.log"
  cp "$(ls -t generated/profiler/reports/*/ops_perf_results_*.csv | head -1)" "$D/tracy/$kind/decode_131071_ops.csv"
  report "$D/tracy/$kind/decode_131071_ops.csv" "$D/tracy/$kind/decode_131071_perf_report" PERF_DECODE PERF_DECODE_END
done

echo "=== dropped-marker integrity check (all must be 0) ==="
grep -c "markers were dropped" "$D"/logs/tracy_*.log || true

# The repo's check-large-files pre-commit hook rejects anything over 500 KB, and
# the raw decode captures are ~1 MB.  gunzip -k them to re-run tt-perf-report.
for f in "$D"/tracy/*/*_ops.csv; do
  [ -f "$f" ] && [ "$(stat -c %s "$f")" -gt 400000 ] && gzip -9 -f "$f"
done
echo "=== done ==="
