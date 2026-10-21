#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0
#
# One round of the Timestral optimization loop: benchmark, Tracy profile, tt-perf-report summary.
#
#   perf_iteration.sh <iter_dir> [--no-bench] [--full-bench]
#
#   default       predict_block paper (both precisions) plus the long_context and multivariate
#                 guards (accuracy precision) -> bench.json
#   --full-bench  every predict_block shape and precision plus both forecast cases
#   --no-bench    profile and summarize only
#
# Outputs in <iter_dir>: bench.json, ops_perf_results.csv (raw, gitignored), layer_counts.json,
# perf_report.{txt,csv}, perf_report_stacked.{csv,png}, sections/*.csv, summary.{json,md}.
# Never combine with TT_METAL_WATCHER: the device profiler and watcher cannot share a run.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <iter_dir> [--no-bench] [--full-bench]" >&2
    exit 2
fi
iter_dir="$(realpath -m "$1")"
shift
bench="default"
for arg in "$@"; do
    case "$arg" in
        --no-bench) bench="none" ;;
        --full-bench) bench="full" ;;
        *) echo "unknown option $arg" >&2; exit 2 ;;
    esac
done

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../../../../.." && pwd)"
tests="models/experimental/timestral/tests/perf"
cd "$repo"
# shellcheck disable=SC1091
source python_env/bin/activate
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
if [[ -n "${TT_METAL_WATCHER:-}" ]]; then
    echo "unset TT_METAL_WATCHER: watcher and the device profiler cannot share a run" >&2
    exit 2
fi
mkdir -p "$iter_dir"

if [[ "$bench" != "none" ]]; then
    rm -f "$iter_dir/bench.json"
    selection=(-k "paper or (long_context and accuracy) or (multivariate and accuracy)" "$tests/test_forward_perf.py::test_predict_block_perf")
    [[ "$bench" == "full" ]] && selection=("$tests/test_forward_perf.py")
    echo "== benchmark ($bench)"
    TIMESTRAL_PERF_JSON="$iter_dir/bench.json" python -m pytest "${selection[@]}" -p no:cacheprovider -s \
        > "$iter_dir/bench.log" 2>&1 || { tail -40 "$iter_dir/bench.log"; exit 1; }
    grep -E '^\[PERF\]|^  (steady_s|pcc|split|cold_s|speedup|median pcc)' "$iter_dir/bench.log" || true
fi

echo "== tracy profile (reduced layers)"
rm -rf "$iter_dir/tracy"
TIMESTRAL_PROFILE_OUT="$iter_dir" python -m tracy -r -p --no-web-server -o "$iter_dir/tracy" \
    -m pytest "$tests/test_profile_forward.py" -p no:cacheprovider > "$iter_dir/tracy.log" 2>&1 \
    || { tail -40 "$iter_dir/tracy.log"; exit 1; }
ops_csv="$(ls -t "$iter_dir"/tracy/reports/*/ops_perf_results_*.csv | head -1)"
cp "$ops_csv" "$iter_dir/ops_perf_results.csv"

echo "== tt-perf-report summary"
python "$tests/summarize_profile.py" "$iter_dir"
