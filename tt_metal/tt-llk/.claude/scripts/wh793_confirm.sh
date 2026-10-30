#!/usr/bin/env bash
# Is the SFPI 7.93.0 Wormhole LoFi matmul slowdown real, or a code-placement effect?
#
# Both trees hold the SAME source: the stage-2 branch, which carries the TRISC code-shift
# support. Only tt_metal/sfpi-version differs, so a difference can only come from the
# toolchain. Each side measures the LoFi perf_math_matmul tests with the code linked at
# 5 positions. Slower at all 5 means the toolchain; slower at some means placement.
#
# Run it on a reserved Wormhole board. Put --work somewhere local: the container's home
# is near full, so /tmp is the right place.
#
# Usage: wh793_confirm.sh [--work DIR] [--jobs N] [--step all|trees|nodes|measure|pack]
set -uo pipefail

W=/tmp/s2-793
JOBS=15
STEP=all
while [ $# -gt 0 ]; do
    case "$1" in
        --work) W=$2; shift 2 ;;
        --jobs) JOBS=$2; shift 2 ;;
        --step) STEP=$2; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

REPO_URL=https://github.com/tenstorrent/tt-metal
BRANCH=nstojictt/llk-perf-gate-placement
# The two commits are one SFPI bump apart. Only their sfpi-version file is taken.
SFPI_A=3f57b50ff00415889e9e1405df36d80f4d1f4cbe   # pins 7.92.0
SFPI_B=012c19beaae306bb2b5053c69e91892506d4061f   # pins 7.93.0

REPO=$W/repo
say() { echo "[$(date -u +%H:%M:%S)] $*"; }
run() { say "+ $*"; "$@"; }

# --- python ------------------------------------------------------------------
venv() {
    [ -x "$W/venv/bin/pytest" ] && return 0
    say "building the venv"
    python3 -m venv "$W/venv"
    "$W/venv/bin/pip" -q install --upgrade pip
    "$W/venv/bin/pip" -q install -r "$W/t_a/tt_metal/tt-llk/tests/requirements.txt"
}

# --- the trees ---------------------------------------------------------------
# One clone, two worktrees of the same commit. The only edit is the toolchain pin,
# so the generated code is the toolchain's doing and nothing else.
trees() {
    [ -d "$REPO/.git" ] || run git clone -q --filter=blob:none --no-checkout "$REPO_URL" "$REPO"
    run git -C "$REPO" fetch -q --depth=1 origin "$BRANCH"
    local head; head=$(git -C "$REPO" rev-parse FETCH_HEAD)
    say "source commit: $head"
    for pair in "a $SFPI_A" "b $SFPI_B"; do
        set -- $pair
        local tag=$1 sha=$2 t=$W/t_$1
        if [ ! -f "$t/.ready" ]; then
            rm -rf "$t"
            git -C "$REPO" worktree prune
            run git -C "$REPO" worktree add -q --detach "$t" "$head" || return 1
            run git -C "$REPO" fetch -q --depth=1 origin "$sha" || return 1
            git -C "$REPO" show "$sha:tt_metal/sfpi-version" > "$t/tt_metal/sfpi-version" || return 1
            touch "$t/.ready"
        fi
        say "  tree $tag: $(grep -m1 '^sfpi_version=' "$t/tt_metal/sfpi-version")"
        sfpi "$t/tt_metal/tt-llk/tests" || return 1
    done
}

# The download and hash check setup_testing_env.sh does, without the pre-commit hooks.
sfpi() {
    ( cd "$1" || exit 1
      eval "$(./sfpi-info.sh SHELL txz)"
      if [ -f sfpi/sfpi.version ] && [ "$sfpi_version" = "$(cat sfpi/sfpi.version)" ]; then
          say "    sfpi $sfpi_version already here"; exit 0
      fi
      say "    fetching sfpi $sfpi_version"
      tmp=$(mktemp -d)
      curl -sSfL --retry 3 -o "$tmp/$sfpi_filename" "$sfpi_url/$sfpi_filename" || exit 1
      got=$("${sfpi_hashtype}sum" -b "$tmp/$sfpi_filename" | cut -d' ' -f1)
      [ "$got" = "$sfpi_hash" ] || { echo "[ERROR] $sfpi_filename hash $got != $sfpi_hash" >&2; exit 1; }
      rm -rf sfpi; tar -xJf "$tmp/$sfpi_filename" -C .; rm -rf "$tmp"
      echo "$sfpi_version" > sfpi/sfpi.version
      say "    sfpi $sfpi_version installed" )
}

# --- the tests to measure ----------------------------------------------------
# Only the LoFi matmul tests. They hold the cluster the CI artifacts flagged, and the
# HiFi ones stay out so the run fits the board time.
nodes() {
    local t=$W/t_a/tt_metal/tt-llk/tests/python_tests
    ( cd "$t" && PATH="$W/venv/bin:$PATH" CHIP_ARCH=wormhole TT_LLK_DISABLE_ASSERTS=1 \
        pytest --collect-only -q -m "perf and not accuracy" perf_math_matmul.py ) \
        > "$W/collect.txt" 2>&1
    grep -oE '^perf_math_matmul\.py::[^ ]*MathFidelity\.LoFi[^ ]*' "$W/collect.txt" | sort -u > "$W/nodes.txt"
    say "node list: $(wc -l < "$W/nodes.txt") LoFi tests of $(grep -c '^perf_math_matmul\.py::' "$W/collect.txt") collected"
    [ -s "$W/nodes.txt" ] || { echo "[ERROR] no LoFi nodes collected; see $W/collect.txt" >&2; return 1; }
}

# --- measure -----------------------------------------------------------------
measure() {
    for tag in a b; do
        local llk=$W/t_$tag/tt_metal/tt-llk
        say "measuring tree $tag at 5 code positions"
        ( cd "$llk" && PATH="$W/venv/bin:$PATH" CHIP_ARCH=wormhole \
            LLK_PERF_NODE_LIST="$W/nodes.txt" RUNNER_TEMP="$W/tmp-$tag" \
            TT_LLK_DISABLE_ASSERTS=1 LLK_PERF_RUN_TYPES=L1_TO_L1 \
            ./tests/run_llk_perf_placement.sh ) > "$W/measure-$tag.log" 2>&1
        say "  exit $? ; csv: $(find "$llk/tests/python_tests/perf_data/runs" -name '*.csv' 2>/dev/null | wc -l)"
    done
}

pack() {
    rm -f "$W/results.tgz"
    ( cd "$W" && tar czf results.tgz \
        t_a/tt_metal/tt-llk/tests/python_tests/perf_data/runs \
        t_b/tt_metal/tt-llk/tests/python_tests/perf_data/runs \
        nodes.txt measure-a.log measure-b.log 2>/dev/null )
    say "packed $(du -h "$W/results.tgz" | cut -f1) into $W/results.tgz"
}

mkdir -p "$W"
case "$STEP" in
    trees)   trees ;;
    nodes)   venv && nodes ;;
    measure) measure ;;
    pack)    pack ;;
    all)     trees && venv && nodes && measure && pack ;;
    *) echo "unknown step: $STEP" >&2; exit 2 ;;
esac
