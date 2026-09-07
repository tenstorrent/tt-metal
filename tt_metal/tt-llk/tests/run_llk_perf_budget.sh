#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Time-budget measurement runner, used by the LLK perf budget workflow
# (.github/workflows/llk-perf-budget.yaml, matrix in
# tests/pipeline_reorg/llk_perf_budget_tests.yaml).
#
# It runs the same two-phase flow as the nightly runners -- producer compiles
# every item to an ELF, consumer loads them and measures on device -- but times
# the two phases and writes a machine-readable summary, so a real CI run can
# answer "what would a non-speed-of-light perf gate cost, per architecture, per
# card, per run-type selection".
#
# Differences from run_llk_perf_{wormhole,blackhole}.sh, all deliberate:
#   * speed of light defaults OFF (a gate compares cycles, not the SoL bound),
#   * the consumer drops -x, because a first failure would truncate the very
#     duration this script exists to measure; the exit codes are reported
#     instead,
#   * the build tree is wiped first, so the compile time is the cold one a
#     fresh gate runner pays and not a short-circuit on .build_complete,
#   * the sweep is one group by default, so one job measures the whole suite.
#
# Usage: run_llk_perf_budget.sh <config> [<group> <n_groups>]
#   config: full | isolates | l1 | isolates_l1
# Env:
#   SPEED_OF_LIGHT  true|false            (default false)
#   BUDGET_OUT      summary json path     (default python_tests/perf_budget/summary.json)
#   COLD_BUILD      true|false            (default true; wipes the build tree)
set -uo pipefail

CONFIG="${1:?usage: run_llk_perf_budget.sh <full|isolates|l1|isolates_l1> [group n_groups]}"
GROUP="${2:-1}"
N_GROUPS="${3:-1}"
SPEED_OF_LIGHT="${SPEED_OF_LIGHT:-false}"
COLD_BUILD="${COLD_BUILD:-true}"

case "$SPEED_OF_LIGHT" in
  true)  SPEED_OF_LIGHT_ARGS=(--speed-of-light) ;;
  false) SPEED_OF_LIGHT_ARGS=() ;;
  *) echo "SPEED_OF_LIGHT must be 'true' or 'false', got '$SPEED_OF_LIGHT'" >&2; exit 2 ;;
esac

# Each run type is a separate ELF and a separate device run, so the selection is
# the cost knob. 'full' passes no flag at all and therefore keeps whatever each
# perf module declares -- the same set the nightly measures.
case "$CONFIG" in
  full)        RUN_TYPES="" ;;
  isolates)    RUN_TYPES="UNPACK_ISOLATE,MATH_ISOLATE,PACK_ISOLATE" ;;
  l1)          RUN_TYPES="L1_TO_L1" ;;
  isolates_l1) RUN_TYPES="L1_TO_L1,UNPACK_ISOLATE,MATH_ISOLATE,PACK_ISOLATE" ;;
  *) echo "unknown config '$CONFIG' (full|isolates|l1|isolates_l1)" >&2; exit 2 ;;
esac

RUN_TYPE_ARGS=()
if [[ -n "$RUN_TYPES" ]]; then
  RUN_TYPE_ARGS=(--perf-run-types "$RUN_TYPES")
fi

# pytest-split is skipped entirely at one group: --splits 1 is a no-op that
# still pulls the plugin into the selection path, and the whole point of this
# measurement is the unsharded suite.
SPLIT_ARGS=()
if [[ "$N_GROUPS" -gt 1 ]]; then
  SPLIT_ARGS=(--splits "$N_GROUPS" --group "$GROUP")
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/python_tests"
mkdir -p perf_data

ARCH="${CHIP_ARCH:-unknown}"
TAG="${CONFIG}-${ARCH}-${GROUP}"
BUDGET_OUT="${BUDGET_OUT:-$PWD/perf_budget/summary.json}"
mkdir -p "$(dirname "$BUDGET_OUT")"

# The harness builds into $RUNNER_TEMP/tt-llk-build under GHA, else
# $TMPDIR/tt-llk-build. Wipe it so the producer pays a cold compile. Guarded on
# CI, because on a shared dev box that path is other people's build tree too.
BUILD_TREE="${RUNNER_TEMP:-${TMPDIR:-/tmp}}/tt-llk-build"
BUILD_TREE="${BUILD_TREE//\/\//\/}"
if [[ "$COLD_BUILD" == "true" ]]; then
  if [[ -n "${RUNNER_TEMP:-}" || "${COLD_BUILD_FORCE:-0}" == "1" ]]; then
    echo "wiping $BUILD_TREE for a cold compile"
    rm -rf "$BUILD_TREE"
  else
    echo "refusing to wipe the shared $BUILD_TREE outside CI;" \
         "set COLD_BUILD_FORCE=1 to override. Compile time will not be cold." >&2
  fi
fi

PYTEST_EXTRA=(-q --override-ini=log_cli=false)
SELECT=(-m "perf and not accuracy" --timeout=60)

echo "=== $TAG: run types: ${RUN_TYPES:-<all declared>}, speed of light: $SPEED_OF_LIGHT," \
     "group $GROUP/$N_GROUPS"

PRODUCER_START=$(date +%s)
pytest "${PYTEST_EXTRA[@]}" "${SPEED_OF_LIGHT_ARGS[@]}" "${RUN_TYPE_ARGS[@]}" \
  --compile-producer -n 10 "${SELECT[@]}" "${SPLIT_ARGS[@]}" \
  --junitxml="pytest-report-${TAG}-compile.xml" .
PRODUCER_RC=$?
PRODUCER_S=$(( $(date +%s) - PRODUCER_START ))
echo "=== $TAG: producer ${PRODUCER_S}s rc=$PRODUCER_RC"

CONSUMER_START=$(date +%s)
pytest "${PYTEST_EXTRA[@]}" "${SPEED_OF_LIGHT_ARGS[@]}" "${RUN_TYPE_ARGS[@]}" \
  --compile-consumer -n 15 "${SELECT[@]}" "${SPLIT_ARGS[@]}" \
  --junitxml="pytest-report-${TAG}-run.xml" .
CONSUMER_RC=$?
CONSUMER_S=$(( $(date +%s) - CONSUMER_START ))
echo "=== $TAG: consumer ${CONSUMER_S}s rc=$CONSUMER_RC"

TOTAL_S=$(( PRODUCER_S + CONSUMER_S ))

hms() { printf '%d:%02d:%02d' $(( $1 / 3600 )) $(( ($1 % 3600) / 60 )) $(( $1 % 60 )); }

# Which card this landed on. The runner name and the host identify the machine,
# which is what makes "the same configuration on two different cards"
# checkable after the fact rather than assumed.
BOARDS="$(ls /dev/tenstorrent 2>/dev/null | tr '\n' ' ' | sed 's/ $//')"

python3 - "$BUDGET_OUT" <<PY
import json, sys

summary = {
    "config": "$CONFIG",
    "arch": "$ARCH",
    "run_types": "${RUN_TYPES:-<all declared>}",
    "speed_of_light": "$SPEED_OF_LIGHT" == "true",
    "cold_build": "$COLD_BUILD" == "true",
    "card": "${BUDGET_CARD:-}",
    "group": "$GROUP",
    "n_groups": "$N_GROUPS",
    "producer_s": $PRODUCER_S,
    "consumer_s": $CONSUMER_S,
    "total_s": $TOTAL_S,
    "producer": "$(hms $PRODUCER_S)",
    "consumer": "$(hms $CONSUMER_S)",
    "total": "$(hms $TOTAL_S)",
    "producer_rc": $PRODUCER_RC,
    "consumer_rc": $CONSUMER_RC,
    "runner": "${RUNNER_NAME:-}",
    "hostname": "$(hostname)",
    "devices": "$BOARDS",
    "commit": "${GITHUB_SHA:-$(git rev-parse HEAD 2>/dev/null || echo unknown)}",
    "run_id": "${GITHUB_RUN_ID:-}",
}
with open(sys.argv[1], "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
PY

# The measurement itself succeeded even when tests failed -- the durations are
# the deliverable and the exit codes travel in the summary. Only a phase that
# died before producing a duration is a failure of this script.
# pytest: 0 clean, 1 tests failed, 2 interrupted, 3 internal, 4 usage, 5 nothing
# collected. Only 0 and 1 leave a duration worth quoting.
if [[ $PRODUCER_RC -gt 1 || $CONSUMER_RC -gt 1 ]]; then
  echo "::error::a pytest phase failed to run (producer rc=$PRODUCER_RC," \
       "consumer rc=$CONSUMER_RC); the timings are not comparable"
  exit 1
fi
if [[ $PRODUCER_RC -ne 0 || $CONSUMER_RC -ne 0 ]]; then
  echo "::warning::tests failed (producer rc=$PRODUCER_RC, consumer rc=$CONSUMER_RC);" \
       "timings are reported but read them with that in mind"
fi
exit 0
