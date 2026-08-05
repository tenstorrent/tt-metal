#!/usr/bin/env bash
# Orchestrate the Quasar perf suite one run type at a time, with L1_TO_L1 last.
# For each PerfRunType: run suite → rename CSVs → next type.
#
# Resume example:
#   RUN_ORDER="MATH_ISOLATE PACK_ISOLATE L1_CONGESTION L1_TO_L1" \
#     nohup bash ./run_perf_suite_orchestrator.sh >> perf_suite_orchestrator.stdout 2>&1 &
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=perf_suite_common.sh
source "${SCRIPT_DIR}/perf_suite_common.sh"
SUITE_DIR="$PERF_SUITE_DIR"
RENAME="${SUITE_DIR}/rename_perf_csvs.sh"
ORCH_LOG="${SUITE_DIR}/perf_suite_orchestrator.log"
STDOUT_LOG="${SUITE_DIR}/perf_suite_orchestrator.stdout"

# Isolates / congestion first; L1_TO_L1 last (expected to be healthy).
# Override with: RUN_ORDER="MATH_ISOLATE L1_TO_L1"
if [[ -n "${RUN_ORDER:-}" ]]; then
  # shellcheck disable=SC2206
  RUN_ORDER_ARR=(${RUN_ORDER})
else
  RUN_ORDER_ARR=(
    UNPACK_ISOLATE
    MATH_ISOLATE
    PACK_ISOLATE
    L1_CONGESTION
    L1_TO_L1
  )
fi

log() {
  local line="$*"
  printf '%s\n' "$line" | tee -a "$ORCH_LOG" >>"$STDOUT_LOG"
}

cd "$SUITE_DIR"
# Ignore SIGHUP so Cursor session cancel does not stop the suite.
trap '' HUP

log "================================================================"
log "Orchestrator started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
log "Order: ${RUN_ORDER_ARR[*]}"
log "================================================================"

for run_type in "${RUN_ORDER_ARR[@]}"; do
  log ""
  log "########## BEGIN ${run_type} $(date -u +%Y-%m-%dT%H:%M:%SZ) ##########"

  suite_log="${SUITE_DIR}/perf_suite_${run_type}.run.log"
  : >"$suite_log"

  set +e
  # Avoid pipe/SIGPIPE: suite writes to its own log; we also append to orch logs.
  bash ./run_perf_suite_and_report.sh --run-type "$run_type" >"$suite_log" 2>&1
  suite_rc=$?
  set -e

  # Mirror suite log into orchestrator logs without a live pipe.
  cat "$suite_log" >>"$ORCH_LOG"
  cat "$suite_log" >>"$STDOUT_LOG"

  log "########## SUITE ${run_type} exit=${suite_rc} $(date -u +%Y-%m-%dT%H:%M:%SZ) ##########"

  rename_args=(--llk-root "$LLK_ROOT" --run-type "$run_type")
  for entry in "${PERF_SUITE_TESTS[@]}"; do
    name="${entry#*:}"
    name="${name%.py}"
    rename_args+=(--test-name "$name")
  done

  log "########## RENAME ${run_type} ##########"
  set +e
  bash "$RENAME" "${rename_args[@]}" >>"$ORCH_LOG" 2>&1
  rename_rc=$?
  set -e
  # Also mirror rename section to stdout log
  grep -A1000 "########## RENAME ${run_type} ##########" "$ORCH_LOG" | tail -n 40 >>"$STDOUT_LOG" || true
  log "########## RENAME ${run_type} exit=${rename_rc} ##########"

  log "########## END ${run_type} $(date -u +%Y-%m-%dT%H:%M:%SZ) ##########"
done

log ""
log "Orchestrator finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
log "DONE_ALL_RUN_TYPES"
