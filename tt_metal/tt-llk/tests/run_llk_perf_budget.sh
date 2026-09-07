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
# Every requested configuration is measured in sequence on THIS card, one cold
# build each. Sequential, not one card per configuration: a card is the scarce
# thing, and measuring all four on the same card is also the better comparison,
# since the differences between them are then not a difference of machine.
#
# Differences from run_llk_perf_{wormhole,blackhole}.sh, all deliberate:
#   * speed of light defaults OFF (a gate compares cycles, not the SoL bound),
#   * the consumer drops -x, because a first failure would truncate the very
#     duration this script exists to measure; the exit codes are reported
#     instead,
#   * the build tree is wiped before each configuration, so every compile time
#     is the cold one a fresh gate runner pays and not a short-circuit on
#     .build_complete,
#   * the sweep is one group by default, so one job measures the whole suite.
#
# Usage: run_llk_perf_budget.sh [<group> <n_groups>]
# Env:
#   CONFIGS         comma-separated: full,isolates,l1,isolates_l1 (all four by default)
#   SPEED_OF_LIGHT  true|false            (default false)
#   BUDGET_DIR      summary directory     (default python_tests/perf_budget)
#   BUDGET_CARD     card label for the report, e.g. 1 or 2
#   COLD_BUILD      true|false            (default true; wipes the build tree)
set -uo pipefail

GROUP="${1:-1}"
N_GROUPS="${2:-1}"
CONFIGS="${CONFIGS:-full,isolates,l1,isolates_l1}"
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
run_types_for() {
  case "$1" in
    full)        echo "" ;;
    isolates)    echo "UNPACK_ISOLATE,MATH_ISOLATE,PACK_ISOLATE" ;;
    l1)          echo "L1_TO_L1" ;;
    isolates_l1) echo "L1_TO_L1,UNPACK_ISOLATE,MATH_ISOLATE,PACK_ISOLATE" ;;
    *) return 1 ;;
  esac
}

IFS=',' read -r -a CONFIG_LIST <<< "$CONFIGS"
if [[ ${#CONFIG_LIST[@]} -eq 0 ]]; then
  echo "CONFIGS is empty" >&2
  exit 2
fi
for config in "${CONFIG_LIST[@]}"; do
  if ! run_types_for "$config" >/dev/null; then
    echo "unknown config '$config' (full|isolates|l1|isolates_l1)" >&2
    exit 2
  fi
done

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
CARD="${BUDGET_CARD:-1}"
BUDGET_DIR="${BUDGET_DIR:-$PWD/perf_budget}"
mkdir -p "$BUDGET_DIR"

# The harness builds into $RUNNER_TEMP/tt-llk-build under GHA, else
# $TMPDIR/tt-llk-build.
BUILD_TREE="${RUNNER_TEMP:-${TMPDIR:-/tmp}}/tt-llk-build"
BUILD_TREE="${BUILD_TREE//\/\//\/}"

# Wipe it so each producer pays a cold compile. Guarded outside CI, because on a
# shared dev box that path is other people's build tree too.
wipe_build_tree() {
  [[ "$COLD_BUILD" == "true" ]] || return 0
  if [[ -n "${RUNNER_TEMP:-}" || "${COLD_BUILD_FORCE:-0}" == "1" ]]; then
    echo "wiping $BUILD_TREE for a cold compile"
    rm -rf "$BUILD_TREE"
  else
    echo "refusing to wipe the shared $BUILD_TREE outside CI;" \
         "set COLD_BUILD_FORCE=1 to override. Compile time will not be cold." >&2
  fi
}

hms() { printf '%d:%02d:%02d' $(( $1 / 3600 )) $(( ($1 % 3600) / 60 )) $(( $1 % 60 )); }

PYTEST_EXTRA=(-q --override-ini=log_cli=false)
SELECT=(-m "perf and not accuracy" --timeout=60)

# Which machine and card this landed on. The runner name and the host identify
# it, which is what makes "the same configuration on two different cards"
# checkable after the fact rather than assumed.
BOARDS="$(ls /dev/tenstorrent 2>/dev/null | tr '\n' ' ' | sed 's/ $//')"
HOST="$(hostname)"
COMMIT="${GITHUB_SHA:-$(git rev-parse HEAD 2>/dev/null || echo unknown)}"

WORST_RC=0

for config in "${CONFIG_LIST[@]}"; do
  RUN_TYPES="$(run_types_for "$config")"
  RUN_TYPE_ARGS=()
  if [[ -n "$RUN_TYPES" ]]; then
    RUN_TYPE_ARGS=(--perf-run-types "$RUN_TYPES")
  fi
  TAG="${config}-${ARCH}-card${CARD}-${GROUP}"

  echo "::group::$TAG"
  echo "=== $TAG: run types: ${RUN_TYPES:-<all declared>}," \
       "speed of light: $SPEED_OF_LIGHT, group $GROUP/$N_GROUPS"
  wipe_build_tree

  PRODUCER_START=$(date +%s)
  pytest "${PYTEST_EXTRA[@]}" "${SPEED_OF_LIGHT_ARGS[@]}" "${RUN_TYPE_ARGS[@]}" \
    --compile-producer -n 10 "${SELECT[@]}" "${SPLIT_ARGS[@]}" \
    --junitxml="pytest-report-${TAG}-compile.xml" .
  PRODUCER_RC=$?
  PRODUCER_S=$(( $(date +%s) - PRODUCER_START ))
  echo "=== $TAG: producer $(hms $PRODUCER_S) rc=$PRODUCER_RC"

  CONSUMER_START=$(date +%s)
  pytest "${PYTEST_EXTRA[@]}" "${SPEED_OF_LIGHT_ARGS[@]}" "${RUN_TYPE_ARGS[@]}" \
    --compile-consumer -n 15 "${SELECT[@]}" "${SPLIT_ARGS[@]}" \
    --junitxml="pytest-report-${TAG}-run.xml" .
  CONSUMER_RC=$?
  CONSUMER_S=$(( $(date +%s) - CONSUMER_START ))
  echo "=== $TAG: consumer $(hms $CONSUMER_S) rc=$CONSUMER_RC"

  TOTAL_S=$(( PRODUCER_S + CONSUMER_S ))
  echo "=== $TAG: total $(hms $TOTAL_S)"
  echo "::endgroup::"

  python3 - "$BUDGET_DIR/summary-${config}.json" <<PY
import json, sys

summary = {
    "config": "$config",
    "arch": "$ARCH",
    "run_types": "${RUN_TYPES:-<all declared>}",
    "speed_of_light": "$SPEED_OF_LIGHT" == "true",
    "cold_build": "$COLD_BUILD" == "true",
    "card": "$CARD",
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
    "hostname": "$HOST",
    "devices": "$BOARDS",
    "commit": "$COMMIT",
    "run_id": "${GITHUB_RUN_ID:-}",
}
with open(sys.argv[1], "w") as f:
    json.dump(summary, f, indent=2)
PY

  # pytest: 0 clean, 1 tests failed, 2 interrupted, 3 internal, 4 usage,
  # 5 nothing collected. Only 0 and 1 leave a duration worth quoting.
  for rc in "$PRODUCER_RC" "$CONSUMER_RC"; do
    [[ $rc -gt $WORST_RC ]] && WORST_RC=$rc
  done
  if [[ $PRODUCER_RC -gt 1 || $CONSUMER_RC -gt 1 ]]; then
    echo "::error::$TAG: a pytest phase failed to run (producer rc=$PRODUCER_RC," \
         "consumer rc=$CONSUMER_RC); its timings are not comparable"
  elif [[ $PRODUCER_RC -ne 0 || $CONSUMER_RC -ne 0 ]]; then
    echo "::warning::$TAG: tests failed (producer rc=$PRODUCER_RC," \
         "consumer rc=$CONSUMER_RC); timings are reported but read them with that in mind"
  fi
done

echo
echo "=== $ARCH card $CARD: what each configuration cost"
python3 - "$BUDGET_DIR" <<'PY'
import glob, json, os, sys

rows = []
for path in sorted(glob.glob(os.path.join(sys.argv[1], "summary-*.json"))):
    with open(path) as f:
        rows.append(json.load(f))
order = {"full": 0, "isolates": 1, "isolates_l1": 2, "l1": 3}
rows.sort(key=lambda r: order.get(r["config"], 9))
width = max([len(r["config"]) for r in rows] + [6])
print(f"{'config'.ljust(width)}  {'compile':>9}  {'measure':>9}  {'total':>9}  rc c/m")
for r in rows:
    print(f"{r['config'].ljust(width)}  {r['producer']:>9}  {r['consumer']:>9}  "
          f"{r['total']:>9}  {r['producer_rc']}/{r['consumer_rc']}")
PY

# The measurement itself succeeded even when tests failed -- the durations are
# the deliverable and the exit codes travel in the summaries. Only a phase that
# died before producing a duration is a failure of this script.
if [[ $WORST_RC -gt 1 ]]; then
  exit 1
fi
exit 0
