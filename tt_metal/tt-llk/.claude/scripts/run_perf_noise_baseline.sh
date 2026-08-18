#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Run the LLK perf suite N times on ONE machine at ONE commit, so the spread
# between the runs measures pure noise. Feeds perf_noise_analysis.py, which
# turns that spread into a merge-gate threshold.
#
# Differences from the CI runner (tests/run_llk_perf_<arch>.sh), all deliberate:
#   * NO --speed-of-light. The gate threshold is being characterized for the
#     normal (runtime-parameter) build.
#   * NO --splits/--group. One machine runs the whole suite; sharding across
#     machines would mix per-machine noise into the measurement.
#   * NO -x on the consumer. -x aborts mid-sweep and the combined CSV is
#     silently partial, which reads as "point missing from a run" downstream.
#
# Usage: run_perf_noise_baseline.sh <wormhole|blackhole> [iterations]   (default 5)
#
# Environment:
#   OUT_DIR=<path>            where run snapshots go (default: perf_noise_baseline/<arch>-<sha>)
#   RESET_BETWEEN_RUNS=1      tt-smi -r between iterations (more independent runs, slower)
#   SKIP_MAIN_CHECK=1         allow a HEAD that is not origin/main
#   ALLOW_DIRTY=1             allow a dirty working tree
#   PERF_RUN_TYPES=<A,B>      pass --perf-run-types to narrow the sweep (cost knob)
set -euo pipefail

ARCH="${1:?usage: run_perf_noise_baseline.sh <wormhole|blackhole> [iterations]}"
ITERATIONS="${2:-5}"

case "$ARCH" in
wormhole | blackhole) ;;
*)
    echo "error: arch must be 'wormhole' or 'blackhole', got '$ARCH'" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLK_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export LLK_HOME="${LLK_HOME:-$LLK_ROOT}"
PERF_DATA="$LLK_HOME/perf_data"

# --- provenance: a threshold is only meaningful next to the commit it came from
cd "$LLK_ROOT"
SHA="$(git rev-parse HEAD)"
SHORT_SHA="$(git rev-parse --short HEAD)"
# .claude/ is agent tooling (this script included). It is never a build input, so
# carrying it on top of main does not change a measured cycle -- excluding it lets
# the run start without committing the tooling to main first.
NOT_CLAUDE=(':(exclude,glob)**/.claude/**')

if [ -z "${ALLOW_DIRTY:-}" ] && [ -n "$(git status --porcelain -- . "${NOT_CLAUDE[@]}")" ]; then
    echo "error: working tree has changes outside .claude/. Commit/stash first," >&2
    echo "       or set ALLOW_DIRTY=1 if you know they cannot affect perf." >&2
    git status --short -- . "${NOT_CLAUDE[@]}" >&2
    exit 2
fi

if [ -z "${SKIP_MAIN_CHECK:-}" ]; then
    git fetch origin main --quiet
    if [ "$SHA" != "$(git rev-parse origin/main)" ]; then
        if [ -n "$(git diff --name-only origin/main HEAD -- . "${NOT_CLAUDE[@]}")" ]; then
            echo "error: HEAD ($SHORT_SHA) differs from origin/main outside .claude/." >&2
            echo "       A gate threshold must be measured on the code the gate defends." >&2
            echo "       Run 'git checkout main && git pull', or set SKIP_MAIN_CHECK=1." >&2
            exit 2
        fi
        echo "note: HEAD ($SHORT_SHA) differs from origin/main only under .claude/ — continuing."
    fi
fi

OUT_DIR="${OUT_DIR:-$LLK_HOME/perf_noise_baseline/${ARCH}-${SHORT_SHA}}"
mkdir -p "$OUT_DIR"

EXTRA_ARGS=()
if [ -n "${PERF_RUN_TYPES:-}" ]; then
    EXTRA_ARGS+=(--perf-run-types "$PERF_RUN_TYPES")
fi

{
    echo "arch: $ARCH"
    echo "commit: $SHA"
    echo "iterations: $ITERATIONS"
    echo "speed_of_light: off"
    echo "host: $(hostname)"
    echo "perf_run_types: ${PERF_RUN_TYPES:-<test default>}"
} | tee "$OUT_DIR/run_metadata.txt"

cd "$LLK_HOME/tests/python_tests"

for i in $(seq 1 "$ITERATIONS"); do
    snapshot="$OUT_DIR/run_$i"
    if [ -d "$snapshot" ]; then
        echo "== iteration $i/$ITERATIONS: snapshot exists, skipping (resume)"
        continue
    fi

    echo "== iteration $i/$ITERATIONS ($(date -u +%H:%M:%SZ))"
    # A stale perf_data would let a previous iteration's CSV survive a test that
    # failed to report this time, hiding the gap instead of showing it.
    rm -rf "$PERF_DATA"

    CHIP_ARCH="$ARCH" pytest -q --override-ini=log_cli=false \
        --compile-producer -n 10 -m "perf and not accuracy" --timeout=60 \
        "${EXTRA_ARGS[@]}" .
    CHIP_ARCH="$ARCH" pytest -q --override-ini=log_cli=false \
        --compile-consumer -n 15 -m "perf and not accuracy" --timeout=60 \
        "${EXTRA_ARGS[@]}" .

    if [ ! -d "$PERF_DATA" ]; then
        echo "error: iteration $i produced no $PERF_DATA" >&2
        exit 1
    fi
    cp -R "$PERF_DATA" "$snapshot"
    echo "   -> $snapshot ($(find "$snapshot" -name '*.csv' -not -name '*.post.csv' | wc -l | tr -d ' ') test CSVs)"

    if [ -n "${RESET_BETWEEN_RUNS:-}" ] && [ "$i" -lt "$ITERATIONS" ]; then
        tt-smi -r
    fi
done

echo "== analyzing"
RUN_ARGS=()
for i in $(seq 1 "$ITERATIONS"); do
    RUN_ARGS+=(--run "$OUT_DIR/run_$i")
done
python "$SCRIPT_DIR/perf_noise_analysis.py" "${RUN_ARGS[@]}" \
    --report "$OUT_DIR/noise_report.md" \
    --label "arch=$ARCH" --label "commit=$SHA" --label "speed_of_light=off" \
    --label "host=$(hostname)"

echo
echo "report:  $OUT_DIR/noise_report.md"
echo "points:  $OUT_DIR/noise_report.points.csv"
