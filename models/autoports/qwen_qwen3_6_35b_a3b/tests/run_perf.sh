#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Warmed prefill / traced-warmed decode performance capture for the Qwen3.6-35B-A3B
# functional decoder. One Tracy run per (mode, layer kind) so each measured window gets its
# own ops CSV, then tt-perf-report over the signposted range.
#
#   ./models/autoports/qwen_qwen3_6_35b_a3b/tests/run_perf.sh [prefill|decode] [linear|full]
#   ./models/autoports/qwen_qwen3_6_35b_a3b/tests/run_perf.sh            # all four
#
# Never set TT_METAL_WATCHER for these runs: watcher and the device profiler contend for the
# same debug resources (the test fails fast if it is set).
set -euo pipefail

cd "${TT_METAL_HOME:?TT_METAL_HOME must be set}"
AUTOPORT=models/autoports/qwen_qwen3_6_35b_a3b
ARTIFACT_DIR="$AUTOPORT/doc/functional_decoder"
TEST=$AUTOPORT/tests/test_perf.py

run_one() {
  local mode=$1 kind=$2
  local sp_start sp_end outdir
  outdir="$ARTIFACT_DIR/tracy/${kind}_${mode}"
  mkdir -p "$outdir"
  if [ "$mode" = "prefill" ]; then
    sp_start=PERF_PREFILL; sp_end=PERF_PREFILL_END
  else
    sp_start=PERF_DECODE; sp_end=PERF_DECODE_END
  fi

  echo "=== tracy: $mode / $kind ==="
  # -n/-r/-p/-v are tracy's own options and must precede "-m pytest".
  # tracy re-splits the forwarded argv on whitespace, so a quoted `-k "a and b"` expression
  # arrives as separate arguments; select the single test by node id instead (no spaces).
  # --op-support-count: the profiler defaults to 1000 ops (tools/tracy/common.py); one warmed
  # prefill of this layer alone issues ~450 (the 32-step delta-rule scan plus 4 MoE chunks), so
  # the default loses device data and post-processing aborts with "Device data missing".
  python -m tracy -r -p -v --op-support-count "${OP_SUPPORT_COUNT:-20000}" \
    -n "qwen36_${kind}_${mode}" -m pytest \
    "${TEST}::test_perf_${mode}[${kind}]" 2>&1 | tee "$outdir/tracy_run.log"

  # tracy launches a WASM GUI server subprocess; never leave it holding a port
  pkill -f "tools/tracy/serve_wasm.py" 2>/dev/null || true

  # newest post-processed ops CSV produced by the run above
  local csv
  csv=$(find generated/profiler -name 'ops_perf_results_*.csv' -newermt '-30 minutes' \
        -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)
  if [ -z "$csv" ]; then
    echo "NO OPS CSV FOUND for $mode/$kind" >&2
    return 1
  fi
  cp "$csv" "$outdir/${mode}_ops.csv"
  echo "ops csv: $csv -> $outdir/${mode}_ops.csv"

  tt-perf-report "$outdir/${mode}_ops.csv" \
    --start-signpost "$sp_start" --end-signpost "$sp_end" \
    --csv "$outdir/${mode}_perf_report.csv" --no-advice \
    > "$outdir/${mode}_perf_report.console.log"

  # human-readable table (never from a --csv run: that prints status boilerplate instead)
  tt-perf-report "$outdir/${mode}_ops.csv" \
    --start-signpost "$sp_start" --end-signpost "$sp_end" \
    --no-summary --no-advice \
    > "$outdir/${mode}_perf_report.txt"

  echo "report: $outdir/${mode}_perf_report.txt"
}

MODES=${1:-"prefill decode"}
KINDS=${2:-"linear full"}
for mode in $MODES; do
  for kind in $KINDS; do
    run_one "$mode" "$kind"
  done
done
