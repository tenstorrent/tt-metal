#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Weekly config-pollution sweep: empirically discover a catalog of real ops' post-execution CFG
# residue (discover_catalog.py), gate each one's restore-mode fidelity, then sweep every (X, K)
# pair and report escapes (a real op X leaves CFG residue that breaks a real op K's own
# correctness check).
#
#   ./ci.sh --arch blackhole [--report-dir DIR] [--sample-per-test N] [--jobs N] [--timeout SECS]
#
# Exit codes: 0 = no escapes, 1 = at least one escape found, 2 = the sweep itself errored out
# (catalog discovery or pair sweep crashed -- nothing meaningful was tested).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE="$(cd "$HERE/../../.." && pwd)"   # tests/python_tests/cfg_pollution -> tt-llk root

ARCH=""
REPORT_DIR="$HERE/reports"
SAMPLE_PER_TEST=150
JOBS=8
TIMEOUT=90

while [[ $# -gt 0 ]]; do
    case "$1" in
        --arch) ARCH="$2"; shift 2 ;;
        --report-dir) REPORT_DIR="$2"; shift 2 ;;
        --sample-per-test) SAMPLE_PER_TEST="$2"; shift 2 ;;
        --jobs) JOBS="$2"; shift 2 ;;
        --timeout) TIMEOUT="$2"; shift 2 ;;
        *) echo "cfg_pollution/ci.sh: unknown option $1" >&2; exit 4 ;;
    esac
done

if [[ -z "$ARCH" ]]; then
    echo "cfg_pollution/ci.sh: --arch blackhole|wormhole is required" >&2
    exit 4
fi

rm -rf "$REPORT_DIR"
mkdir -p "$REPORT_DIR/catalog"

echo ">> [1/3] catalog discovery"
python3 "$HERE/discover_catalog.py" \
    --worktree "$WORKTREE" --arch "$ARCH" \
    --out-dir "$REPORT_DIR/catalog" --manifest "$REPORT_DIR/manifest.json" \
    --sample-per-test "$SAMPLE_PER_TEST" --jobs "$JOBS" --compile-jobs "$JOBS" \
    --timeout "$TIMEOUT" \
    || { echo ">> catalog discovery did not complete" >&2; exit 2; }

echo ">> [2/3] pair sweep"
python3 "$HERE/pair_sweep.py" \
    --worktree "$WORKTREE" --arch "$ARCH" --manifest "$REPORT_DIR/manifest.json" \
    --out "$REPORT_DIR/findings.jsonl" --jobs "$JOBS" --timeout "$TIMEOUT" --skip-compile \
    || { echo ">> pair sweep did not complete" >&2; exit 2; }

echo ">> [3/3] report"
status=0
python3 "$HERE/report.py" \
    --jsonl "$REPORT_DIR/findings.jsonl" \
    --report-md "$REPORT_DIR/report.md" --junit "$REPORT_DIR/junit.xml" \
    || status=$?

exit "$status"
