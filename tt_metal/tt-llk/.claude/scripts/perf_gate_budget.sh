#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# perf_gate_budget.sh -- measure what a PR/merge perf gate would cost.
#
# Runs the same non-speed-of-light perf sweep three times, differing only in
# which PerfRunTypes are measured, and reports wall time and yield for each:
#
#   full      every run type the tests declare (5 on Quasar)
#   isolates  UNPACK_ISOLATE, MATH_ISOLATE, PACK_ISOLATE
#   l1        L1_TO_L1 only
#
# Cost scales with (variants x selected run types): each run type is a separate
# ELF and a separate device run. The point of the experiment is to find out how
# much of the gate's detection power survives paying only a fraction of that.
#
# Every configuration starts from a cold build tree. Without that, runs 2 and 3
# reuse run 1's cached ELFs (build_elfs short-circuits on .build_complete) and
# report a compile time near zero -- which is not what a gate on a fresh runner
# would pay.
#
# Usage:
#   perf_gate_budget.sh --arch wormhole [options]
#
# Options:
#   --arch ARCH        wormhole | blackhole | quasar          (required)
#   --out DIR          results root (default: ./perf_gate_budget)
#   --markers EXPR     pytest -m expression (default: "perf and not accuracy")
#   --k EXPR           pytest -k filter, applied to both phases (default: none)
#   --tests PATH       test path passed to pytest (default: .)
#   --jobs-compile N   producer parallelism (default: 10, matches CI)
#   --jobs-run N       consumer parallelism (default: 15, matches CI)
#   --timeout SECS     pytest --timeout (default: 60, matches CI)
#   --configs LIST     comma-separated subset of: full,isolates,l1 (default: all)
#   --keep-build       do not wipe the build tree between configs (warm compile;
#                      use only when you want measure-phase time alone)
#   --build-root DIR   private build tree, exported as RUNNER_TEMP so artefacts
#                      land in <DIR>/tt-llk-build. STRONGLY recommended on a
#                      shared machine: this script wipes the build root between
#                      configs, and the default is the shared /tmp/tt-llk-build
#                      that every other user's run also builds into.
#
# Output:
#   <out>/<config>/perf_data/     the reports that config produced
#   <out>/<config>/{compile,run}.log
#   <out>/summary.md              the comparison table
#
# Speed of light is deliberately never passed: CI's perf job uses it, this
# experiment asks about the non-SoL numbers.

set -euo pipefail

ARCH=""
OUT="$(pwd)/perf_gate_budget"
MARKERS="perf and not accuracy"
KFILTER=""
TESTS="."
JOBS_COMPILE=10
JOBS_RUN=15
TIMEOUT=60
CONFIGS="full,isolates,l1"
KEEP_BUILD=0
BUILD_ROOT_OPT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --arch)          ARCH="$2"; shift 2 ;;
    --out)           OUT="$2"; shift 2 ;;
    --markers)       MARKERS="$2"; shift 2 ;;
    --k)             KFILTER="$2"; shift 2 ;;
    --tests)         TESTS="$2"; shift 2 ;;
    --jobs-compile)  JOBS_COMPILE="$2"; shift 2 ;;
    --jobs-run)      JOBS_RUN="$2"; shift 2 ;;
    --timeout)       TIMEOUT="$2"; shift 2 ;;
    --configs)       CONFIGS="$2"; shift 2 ;;
    --keep-build)    KEEP_BUILD=1; shift ;;
    --build-root)    BUILD_ROOT_OPT="$2"; shift 2 ;;
    -h|--help)       sed -n '2,50p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$ARCH" ]] || { echo "--arch is required" >&2; exit 2; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLK_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTEST_DIR="$LLK_ROOT/tests/python_tests"
# TestConfig.resolve_artefacts_path(): $RUNNER_TEMP/tt-llk-build when RUNNER_TEMP
# is set, else /tmp/tt-llk-build. --build-root drives it through that same lever
# so the wipe below can never reach a build tree we do not own.
if [[ -n "$BUILD_ROOT_OPT" ]]; then
  mkdir -p "$BUILD_ROOT_OPT"
  export RUNNER_TEMP="$(cd "$BUILD_ROOT_OPT" && pwd)"
fi
BUILD_ROOT="${RUNNER_TEMP:+$RUNNER_TEMP/tt-llk-build}"
BUILD_ROOT="${BUILD_ROOT:-/tmp/tt-llk-build}"

if [[ "$KEEP_BUILD" -eq 0 && -z "${BUILD_ROOT_OPT}" && -e "$BUILD_ROOT" && ! -O "$BUILD_ROOT" ]]; then
  echo "refusing to wipe $BUILD_ROOT: it belongs to another user." >&2
  echo "pass --build-root <your-own-dir> to build somewhere private." >&2
  exit 2
fi

mkdir -p "$OUT"

# Run-type selection per configuration. An empty value means "pass no
# --perf-run-types at all", i.e. whatever each test declares.
run_types_for() {
  case "$1" in
    full)     echo "" ;;
    isolates) echo "UNPACK_ISOLATE,MATH_ISOLATE,PACK_ISOLATE" ;;
    l1)       echo "L1_TO_L1" ;;
    *) echo "unknown config: $1" >&2; exit 2 ;;
  esac
}

# Seconds elapsed, integer. `date +%s` only -- no bash-5 EPOCHREALTIME
# dependency, and second resolution is far below the noise of a perf sweep.
now() { date +%s; }

fmt_hms() {
  local s=$1
  printf '%d:%02d:%02d' $((s / 3600)) $(((s % 3600) / 60)) $((s % 60))
}

# Timings go to <out>/<config>/timing.env rather than an associative array:
# bash 3.2 (macOS) has no `declare -A`, and a file also means a run killed
# halfway still leaves the configs it finished measurable.
write_timing() {
  cat >"$OUT/$1/timing.env" <<EOF
T_COMPILE=$2
T_RUN=$3
T_TOTAL=$4
RC_COMPILE=$5
RC_RUN=$6
EOF
}

for CONFIG in ${CONFIGS//,/ }; do
  RUN_TYPES="$(run_types_for "$CONFIG")"
  CFG_OUT="$OUT/$CONFIG"
  rm -rf "$CFG_OUT"
  mkdir -p "$CFG_OUT"

  RT_FLAG=()
  [[ -n "$RUN_TYPES" ]] && RT_FLAG=(--perf-run-types "$RUN_TYPES")
  K_FLAG=()
  [[ -n "$KFILTER" ]] && K_FLAG=(-k "$KFILTER")

  echo "=== $CONFIG (run types: ${RUN_TYPES:-<all declared>}) ==="

  if [[ "$KEEP_BUILD" -eq 0 ]]; then
    echo "  wiping $BUILD_ROOT for a cold compile"
    rm -rf "$BUILD_ROOT"
  fi
  # The combiner writes into perf_data/ and overwrites in place, so the previous
  # config's reports must not be sitting there.
  rm -rf "$PYTEST_DIR/perf_data"
  mkdir -p "$PYTEST_DIR/perf_data"

  cd "$PYTEST_DIR"

  t0=$(now)
  set +e
  CHIP_ARCH="$ARCH" pytest -q --override-ini=log_cli=false \
    --compile-producer -n "$JOBS_COMPILE" -m "$MARKERS" --timeout="$TIMEOUT" \
    ${RT_FLAG[@]+"${RT_FLAG[@]}"} ${K_FLAG[@]+"${K_FLAG[@]}"} "$TESTS" >"$CFG_OUT/compile.log" 2>&1
  rc_compile=$?
  set -e
  t1=$(now)

  set +e
  CHIP_ARCH="$ARCH" pytest -q --override-ini=log_cli=false \
    --compile-consumer -n "$JOBS_RUN" -m "$MARKERS" --timeout="$TIMEOUT" \
    ${RT_FLAG[@]+"${RT_FLAG[@]}"} ${K_FLAG[@]+"${K_FLAG[@]}"} "$TESTS" >"$CFG_OUT/run.log" 2>&1
  rc_run=$?
  set -e
  t2=$(now)

  write_timing "$CONFIG" $((t1 - t0)) $((t2 - t1)) $((t2 - t0)) "$rc_compile" "$rc_run"

  cp -r "$PYTEST_DIR/perf_data" "$CFG_OUT/perf_data" 2>/dev/null || true

  echo "  compile $(fmt_hms $((t1 - t0)))  measure $(fmt_hms $((t2 - t1)))  total $(fmt_hms $((t2 - t0)))  rc=$rc_compile/$rc_run"
done

# --- summary -----------------------------------------------------------------
# Yield per config: how many report rows and how many measured (row, metric)
# points came out. Points, not rows, is what a regression check compares.
SUMMARY="$OUT/summary.md"
{
  echo "# Perf gate budget -- $ARCH, non speed-of-light"
  echo
  echo "markers: \`$MARKERS\`${KFILTER:+, -k \`$KFILTER\`}"
  echo "compile -n $JOBS_COMPILE, measure -n $JOBS_RUN, cold build per config: $([[ $KEEP_BUILD -eq 0 ]] && echo yes || echo no)"
  echo
  echo "| config | run types | compile | measure | total | rows | points | modules | rc c/m |"
  echo "|---|---|--:|--:|--:|--:|--:|--:|--:|"
} >"$SUMMARY"

for CONFIG in ${CONFIGS//,/ }; do
  RUN_TYPES="$(run_types_for "$CONFIG")"
  T_COMPILE=; T_RUN=; T_TOTAL=; RC_COMPILE=; RC_RUN=
  # shellcheck source=/dev/null
  [[ -f "$OUT/$CONFIG/timing.env" ]] && . "$OUT/$CONFIG/timing.env"
  read -r ROWS POINTS MODULES < <(
    python3 - "$OUT/$CONFIG/perf_data" <<'PY'
import csv, glob, os, sys

root = sys.argv[1]
rows = points = modules = 0
for path in sorted(glob.glob(os.path.join(root, "*", "*.post.csv"))):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        means = [c for c in (reader.fieldnames or []) if c.startswith("mean(")]
        modules += 1
        for row in reader:
            rows += 1
            # A "point" is one measured metric on one row -- the unit a
            # regression check compares. Fewer run types, fewer mean() columns.
            points += sum(1 for c in means if (row[c] or "").strip())
print(rows, points, modules)
PY
  )
  echo "| $CONFIG | ${RUN_TYPES:-all declared} | $(fmt_hms "${T_COMPILE:-0}") | $(fmt_hms "${T_RUN:-0}") | $(fmt_hms "${T_TOTAL:-0}") | $ROWS | $POINTS | $MODULES | ${RC_COMPILE:-?}/${RC_RUN:-?} |" >>"$SUMMARY"
done

{
  echo
  echo "\`rc c/m\` is the compile/measure pytest exit code. A non-zero measure code"
  echo "means the sweep did not finish, so that row's time is a floor, not a cost."
} >>"$SUMMARY"

echo
cat "$SUMMARY"
echo
echo "(results in $OUT)"
