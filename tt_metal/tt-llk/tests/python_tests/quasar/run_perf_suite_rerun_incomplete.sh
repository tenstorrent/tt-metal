#!/usr/bin/env bash
# After the main orchestrator finishes, re-run suite entries that never finished
# or had no usable result. Skip PASSED and HANG.
#
# Waits for DONE_ALL_RUN_TYPES in the orchestrator stdout (or --now to skip wait).
set -uo pipefail

LLK_ROOT=/proj_sw/user_dev/ndivnic/tt-metal/tt_metal/tt-llk
SUITE_DIR="${LLK_ROOT}/tests/python_tests/quasar"
RENAME="${LLK_ROOT}/.cursor/skills/quasar-perf-suite-by-run-type/scripts/rename_perf_csvs.sh"
STDOUT_LOG="${SUITE_DIR}/perf_suite_orchestrator.stdout"
ORCH_LOG="${SUITE_DIR}/perf_suite_orchestrator.log"
RERUN_LOG="${SUITE_DIR}/perf_suite_rerun_incomplete.log"

WAIT_FOR_ORCH=1
if [[ "${1:-}" == "--now" ]]; then
  WAIT_FOR_ORCH=0
fi

# suite_id -> test basename (no .py)
declare -A ID_TO_NAME=(
  [02]=perf_eltwise_unary_datacopy_quasar
  [03]=perf_eltwise_binary_broadcast_quasar
  [04]=perf_eltwise_binary_quasar
  [05]=perf_unpack_tilize_quasar
  [06]=perf_unpack_unary_operand_quasar
  [07]=perf_transpose_dest_quasar
  [08]=perf_pack_quasar
  [09]=perf_pack_untilize_quasar
  [10]=perf_unary_broadcast_quasar
  [11]=perf_pack_l1_acc_quasar
  [12]=perf_reduce_quasar
  [13]=perf_eltwise_binary_reuse_dest_quasar
  [14]=perf_unpack_reduce_col_tilizeA_strided_quasar
)

ALL_IDS=(02 03 04 05 06 07 08 09 10 11 12 13 14)
RUN_TYPES=(UNPACK_ISOLATE MATH_ISOLATE PACK_ISOLATE L1_CONGESTION L1_TO_L1)

log() {
  printf '%s\n' "$*" | tee -a "$RERUN_LOG" >>"$STDOUT_LOG"
}

cd "$SUITE_DIR"
trap '' HUP
: >>"$RERUN_LOG"

if [[ $WAIT_FOR_ORCH -eq 1 ]]; then
  log "Rerun-incomplete: waiting for orchestrator completion marker ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
  while ! grep -qx 'DONE_ALL_RUN_TYPES' "$STDOUT_LOG" 2>/dev/null; do
    orch_alive=0
    if pgrep -f '[.]/run_perf_suite_orchestrator\.sh' >/dev/null 2>&1 \
      || pgrep -f 'bash ./run_perf_suite_orchestrator.sh' >/dev/null 2>&1; then
      orch_alive=1
    fi
    if [[ $orch_alive -eq 0 ]]; then
      if grep -qx 'DONE_ALL_RUN_TYPES' "$STDOUT_LOG" 2>/dev/null; then
        break
      fi
      log "Rerun-incomplete: orchestrator process not found; proceeding to scan ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
      break
    fi
    sleep 60
  done
fi

log "================================================================"
log "Rerun-incomplete started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
log "Policy: skip PASSED, HANG, and COMPILE_FAIL; rerun missing / aborted / unusable"
log "================================================================"

is_compile_failure() {
  local out="$1"
  [[ -f "$out" ]] || return 1
  # build_elfs failures raise RuntimeError with the g++ command + stderr.
  if grep -qE 'riscv-tt-elf-g\+\+|Command.s stderr:' "$out" 2>/dev/null; then
    if grep -qiE 'error:|fatal error:|compilation terminated|undefined reference|cannot open output file|assembler messages' "$out" 2>/dev/null; then
      return 0
    fi
  fi
  # Broader: toolchain compile diagnostics in a FAILED session.
  if grep -qE '^FAILED |^=+ FAILURES|Error type: RuntimeError' "$out" 2>/dev/null \
    && grep -qiE 'error: .*\.(cpp|h|hpp|cc):|fatal error:|compilation terminated' "$out" 2>/dev/null; then
    return 0
  fi
  return 1
}

# Return status: PASSED|HANG|COMPILE_FAIL|OTHER|MISSING
entry_status() {
  local run_type="$1" suite_id="$2"
  local report="perf_suite_results_${run_type}.md"
  local out="perf_output_${run_type}_${suite_id}.txt"
  local name="${ID_TO_NAME[$suite_id]}.py"

  if [[ -f "$out" ]] && is_compile_failure "$out"; then
    echo "COMPILE_FAIL"
    return
  fi

  if [[ -f "$report" ]]; then
    local row
    row=$(grep -F "\`${name}\`" "$report" | grep "\`${run_type}\`" | tail -1 || true)
    if [[ -n "$row" ]]; then
      if [[ "$row" == *"**PASSED**"* ]]; then
        echo "PASSED"
        return
      fi
      if [[ "$row" == *"**HANG**"* ]]; then
        echo "HANG"
        return
      fi
      # Reason column may mention compile.
      if [[ "$row" == *[Cc]ompil* || "$row" == *"error:"* || "$row" == *"riscv-tt-elf"* ]]; then
        if [[ -f "$out" ]] && is_compile_failure "$out"; then
          echo "COMPILE_FAIL"
          return
        fi
      fi
      # Present but not passed/hang (FAILED, EXALENS_TIMEOUT, NO_TESTS, ...)
      echo "OTHER"
      return
    fi
  fi

  # No report row: usable only if output shows a clean pass.
  if [[ -f "$out" ]] && grep -qE '^=+ .* passed in' "$out" \
      && ! grep -qE '^FAILED |^=+ FAILURES|Error type:' "$out"; then
    echo "PASSED"
    return
  fi
  echo "MISSING"
}

for run_type in "${RUN_TYPES[@]}"; do
  need_ids=()
  for sid in "${ALL_IDS[@]}"; do
    st=$(entry_status "$run_type" "$sid")
    case "$st" in
      PASSED|HANG|COMPILE_FAIL)
        log "SKIP  ${run_type}/${sid} (${ID_TO_NAME[$sid]}) status=${st}"
        ;;
      *)
        need_ids+=("$sid")
        log "RERUN ${run_type}/${sid} (${ID_TO_NAME[$sid]}) status=${st}"
        ;;
    esac
  done

  if [[ ${#need_ids[@]} -eq 0 ]]; then
    log "########## ${run_type}: nothing to rerun ##########"
    continue
  fi

  ids_csv=$(IFS=,; echo "${need_ids[*]}")
  log "########## BEGIN RERUN ${run_type} ids=${ids_csv} $(date -u +%Y-%m-%dT%H:%M:%SZ) ##########"

  suite_log="${SUITE_DIR}/perf_suite_${run_type}.rerun.log"
  set +e
  bash ./run_perf_suite_and_report.sh \
    --run-type "$run_type" \
    --suite-ids "$ids_csv" \
    --append-report \
    >"$suite_log" 2>&1
  rc=$?
  set -e
  cat "$suite_log" >>"$RERUN_LOG"
  cat "$suite_log" >>"$STDOUT_LOG"
  log "########## SUITE RERUN ${run_type} exit=${rc} ##########"

  rename_args=(--llk-root "$LLK_ROOT" --run-type "$run_type")
  for sid in "${need_ids[@]}"; do
    rename_args+=(--test-name "${ID_TO_NAME[$sid]}")
  done
  log "########## RENAME ${run_type} ##########"
  bash "$RENAME" "${rename_args[@]}" >>"$RERUN_LOG" 2>&1
  log "########## END RERUN ${run_type} $(date -u +%Y-%m-%dT%H:%M:%SZ) ##########"
done

log "Rerun-incomplete finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
log "DONE_RERUN_INCOMPLETE"
