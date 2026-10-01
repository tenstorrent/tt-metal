#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Weekly reconfig-escape sweep: empirically discover a catalog of real ops' post-execution CFG
# residue (discover_catalog.py), gate each one's restore-mode fidelity, then sweep every (X, K)
# pair and report escapes (a real op X leaves CFG residue that breaks a real op K's own
# correctness check).
#
#   ./ci.sh --arch blackhole [--report-dir DIR] [--sample-per-test N] [--jobs N] [--timeout SECS]
#           [--splits N --group G]
#
# --splits/--group shard the pair-sweep phase across machines (each machine needs its own
# --report-dir). Catalog discovery is NOT sharded -- every shard rebuilds the full catalog
# itself (the weekly seed makes this deterministic across machines) and then sweeps only its
# own slice of polluters against the full victim set, so no artifact distribution between
# shards is needed and no merge step beyond collecting each shard's report.md/junit.xml.
#
# Exit codes: 0 = no escapes, 1 = at least one escape found, 2 = the sweep itself errored out
# (catalog discovery or pair sweep crashed -- nothing meaningful was tested).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE="$(cd "$HERE/../../.." && pwd)"   # tests/python_tests/reconfig_escape -> tt-llk root

ARCH=""
REPORT_DIR="$HERE/reports"
SAMPLE_PER_TEST=150
JOBS=8
TIMEOUT=90
SPLITS=""
GROUP=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --arch) ARCH="$2"; shift 2 ;;
        --report-dir) REPORT_DIR="$2"; shift 2 ;;
        --sample-per-test) SAMPLE_PER_TEST="$2"; shift 2 ;;
        --jobs) JOBS="$2"; shift 2 ;;
        --timeout) TIMEOUT="$2"; shift 2 ;;
        --splits) SPLITS="$2"; shift 2 ;;
        --group) GROUP="$2"; shift 2 ;;
        *) echo "reconfig_escape/ci.sh: unknown option $1" >&2; exit 4 ;;
    esac
done

if [[ -z "$ARCH" ]]; then
    echo "reconfig_escape/ci.sh: --arch blackhole is required" >&2
    exit 4
fi

rm -rf "$REPORT_DIR"
mkdir -p "$REPORT_DIR/catalog"

SPLIT_ARGS=()
[[ -n "$SPLITS" ]] && SPLIT_ARGS+=(--splits "$SPLITS" --group "${GROUP:-1}")
[[ -z "$SPLITS" ]] || echo ">> splits=${SPLITS} group=${GROUP:-1}"

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
    "${SPLIT_ARGS[@]}" \
    || { echo ">> pair sweep did not complete" >&2; exit 2; }

echo ">> [3/3] report"
status=0
python3 "$HERE/report.py" \
    --jsonl "$REPORT_DIR/findings.jsonl" \
    --report-md "$REPORT_DIR/report.md" --junit "$REPORT_DIR/junit.xml" \
    || status=$?

exit "$status"
