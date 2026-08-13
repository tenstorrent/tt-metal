#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Profile the multichip decoder's perf windows with Tracy and render
# tt-perf-report tables/CSVs, advice enabled.  One device job at a time;
# TT_METAL_WATCHER must NOT be set for any of this ($tt-device-usage).
#
# On a mesh capture, tt-perf-report sees all four devices' ops and merges them:
# it prints "Detected data from 4 devices. Merging device data..." and emits one
# row per op *instance*, attributed to the device that ran it, rather than four
# summed rows.  So a Device Time column is already one chip's work -- verified by
# summing it over a decode window and dividing by the 8 replays, which reproduces
# the end-to-end ms/token to within the host gap.  No --device-id is passed.
set -euo pipefail
cd "$(dirname "$0")/../../../../../.."          # repo root
D=$PWD/models/autoports/meta_models_muse_glimmer_30b/doc/multichip_decoder
B=$D/bench/perf_windows.py
mkdir -p "$D/tracy/sliding" "$D/tracy/full" "$D/logs"

report () {  # $1=ops csv  $2=out prefix  $3=start signpost  $4=end signpost
  tt-perf-report "$1" --start-signpost "$3" --end-signpost "$4" --no-summary > "$2.txt"
  tt-perf-report "$1" --start-signpost "$3" --end-signpost "$4" --csv "$2.csv" > "$2.console.log"
}

capture () {  # $1=ops csv destination
  # Pick the newest ops CSV that still has the host+device columns.  Tracy's
  # post-processing writes a *device-only* CSV over the same path a moment later,
  # and that one has no "DEVICE ID" column, so tt-perf-report refuses it -- taking
  # the newest file unconditionally is a race that silently loses one window.
  local src
  for src in $(ls -t generated/profiler/reports/*/ops_perf_results_*.csv); do
    if head -1 "$src" | grep -q "DEVICE ID"; then cp "$src" "$1"; return 0; fi
  done
  echo "no ops CSV with a DEVICE ID column found" >&2
  return 1
}

for kind in sliding full; do
  for seq in 128 8192; do
    tag=prefill_$seq
    python -m tracy -r -p -v "$B" --window prefill --seq-len "$seq" --kind "$kind" \
        2>&1 | tee "$D/logs/tracy_${tag}_$kind.log"
    capture "$D/tracy/$kind/${tag}_ops.csv"
    report "$D/tracy/$kind/${tag}_ops.csv" "$D/tracy/$kind/${tag}_perf_report" PERF_PREFILL PERF_PREFILL_END
  done

  # 8 trace replays per decode window, matching every earlier stage of this model:
  # the functional stage established that 32 replays overflow the profiler's DRAM
  # marker buffer and silently under-count.
  for ctx in 2048 131071; do
    tag=decode_$ctx
    python -m tracy -r -p -v "$B" --window decode --context "$ctx" --kind "$kind" --iters 8 \
        2>&1 | tee "$D/logs/tracy_${tag}_$kind.log"
    capture "$D/tracy/$kind/${tag}_ops.csv"
    report "$D/tracy/$kind/${tag}_ops.csv" "$D/tracy/$kind/${tag}_perf_report" PERF_DECODE PERF_DECODE_END
  done
done

echo "=== dropped-marker integrity check (all must be 0) ==="
grep -c "markers were dropped" "$D"/logs/tracy_*.log || true

# The repo's check-large-files pre-commit hook rejects anything over 500 KB.
for f in "$D"/tracy/*/*_ops.csv; do
  [ -f "$f" ] && [ "$(stat -c %s "$f")" -gt 400000 ] && gzip -9 -f "$f"
done
echo "=== done ==="
