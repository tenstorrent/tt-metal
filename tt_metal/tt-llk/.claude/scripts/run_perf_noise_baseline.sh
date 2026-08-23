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
#   BUILD_ROOT=<dir>          private build tree (artefacts land in <dir>/tt-llk-build)
#   WIPE_BUILD=1              cold-compile every iteration (see below)
#
# BUILD_ROOT is strongly recommended on a shared machine. The default artefact
# path is the shared /tmp/tt-llk-build that every other user also builds into;
# a run of several hours there can have its ELFs wiped or rebuilt underneath it.
# By default the build tree is NOT wiped between iterations: iteration 1 pays the
# compile, iterations 2..N reuse the ELFs and measure only. That isolates the
# device-side noise and is far cheaper.
#
# WIPE_BUILD=1 rebuilds from cold before every iteration. Slower -- N cold
# compiles instead of one -- but it is what a PR gate actually does: a fresh
# runner compiles, then measures, every time. Use it when the question is "what
# noise would the gate see" rather than "how stable is the silicon". It is also
# the only way to tell a cold-start effect from a one-off glitch: with a warm
# tree, iteration 1 is the sole cold run, so anything cold-specific can only ever
# appear as "run 1 differs".
#
# On this branch --perf-run-types is not yet on main, so the branch differs from
# origin/main in conftest.py and helpers/perf/core.py and the main check trips.
# Those two files only SELECT which run types are measured; they change no kernel
# and no test body, so the cycles still describe main. Set SKIP_MAIN_CHECK=1.
# Drop the override once --perf-run-types lands.
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

# TestConfig.resolve_artefacts_path() reads RUNNER_TEMP, so this is how a run
# claims a build tree of its own.
if [ -n "${BUILD_ROOT:-}" ]; then
    mkdir -p "$BUILD_ROOT"
    export RUNNER_TEMP="$BUILD_ROOT"
fi
BUILD_TREE="${RUNNER_TEMP:+$RUNNER_TEMP/tt-llk-build}"
BUILD_TREE="${BUILD_TREE:-/tmp/tt-llk-build}"

# Never wipe a build tree we do not own: the default is the shared
# /tmp/tt-llk-build that every other user on the machine also builds into.
if [ -n "${WIPE_BUILD:-}" ] && [ -z "${BUILD_ROOT:-}" ] && [ -e "$BUILD_TREE" ] && [ ! -O "$BUILD_TREE" ]; then
    echo "error: WIPE_BUILD=1 refuses to wipe $BUILD_TREE: it belongs to another user." >&2
    echo "       Set BUILD_ROOT=<your-own-dir> to build somewhere private." >&2
    exit 2
fi
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
    echo "build_root: ${BUILD_ROOT:-<shared default>}"
    echo "wipe_build: ${WIPE_BUILD:+yes}${WIPE_BUILD:-no}"
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

    if [ -n "${WIPE_BUILD:-}" ]; then
        echo "   cold compile: wiping $BUILD_TREE"
        rm -rf "$BUILD_TREE"
    fi

    # A handful of tests can time out or fail on any given iteration, and pytest
    # then exits 1. That must NOT abort the baseline: the surviving tests still
    # reported, the snapshot is still usable, and a point missing from one run is
    # something the analysis already reports rather than something to crash on.
    # Aborting here costs every remaining iteration, which is the expensive thing.
    set +e
    CHIP_ARCH="$ARCH" pytest -q --override-ini=log_cli=false \
        --compile-producer -n 10 -m "perf and not accuracy" --timeout=60 \
        "${EXTRA_ARGS[@]}" .
    rc_produce=$?
    CHIP_ARCH="$ARCH" pytest -q --override-ini=log_cli=false \
        --compile-consumer -n 15 -m "perf and not accuracy" --timeout=60 \
        "${EXTRA_ARGS[@]}" .
    rc_measure=$?
    set -e

    if [ "$rc_produce" -ne 0 ] || [ "$rc_measure" -ne 0 ]; then
        echo "   note: iteration $i pytest rc=${rc_produce}/${rc_measure} (some tests failed)"
        echo "run_$i: pytest rc=${rc_produce}/${rc_measure}" >>"$OUT_DIR/run_failures.txt"
    fi

    # An empty perf_data is the real failure: nothing reported, nothing to compare.
    if [ ! -d "$PERF_DATA" ] || [ -z "$(find "$PERF_DATA" -name '*.csv' -print -quit)" ]; then
        echo "error: iteration $i produced no reports in $PERF_DATA" >&2
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
