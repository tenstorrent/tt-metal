#!/bin/bash
# Run quasar perf suite per PerfRunType with --speed-of-light, retry exalens
# timeouts, and kill single-test hangs (>5 min after exalens ready).
set -u
cd /proj_sw/user_dev/ndivnic/tt-metal/tt_metal/tt-llk/tests/python_tests/quasar

export LLK_ROOT=/proj_sw/user_dev/ndivnic/tt-metal/tt_metal/tt-llk
export CHIP_ARCH=quasar
export TT_METAL_SIMULATOR="/proj_sw/user_dev/${USER}/tt-umd-simulators/build/emu-quasar-1x3"
export TT_UMD_SIMULATOR_PATH="$TT_METAL_SIMULATOR"
export NNG_SOCKET_ADDR="tcp://tensix-l-01:54948"
export NNG_SOCKET_LOCAL_PORT=5555
# shellcheck source=/dev/null
source "$LLK_ROOT/tests/.venv/bin/activate"

cp -a /proj_sw/user_dev/ndivnic/tt-umd-simulators/emu/quasar-1x3/quasar-1x3_run_dev.instrumented.sh \
  "$TT_METAL_SIMULATOR/quasar-1x3_run_dev.sh"

REPORT="perf_suite_results.md"
STARTED_AT=$(date -u +"%Y-%m-%d %H:%M:%S UTC")

EXALENS_FAIL_SEC=610   # retry after ~10 min exalens startup failure
TEST_HANG_SEC=300      # single test >5 min after exalens ready => hang
POLL_SEC=10
COOLDOWN_SEC=60   # 1 min rest between suite entries (incl. after hang cleanup)
MAX_EXALENS_RETRIES=5

PERF_RUN_TYPES=(
  L1_TO_L1
  UNPACK_ISOLATE
  MATH_ISOLATE
  PACK_ISOLATE
  L1_CONGESTION
)

# Optional: restrict to one PerfRunType and/or a subset of suite ids.
#   ./run_perf_suite_and_report.sh
#   ./run_perf_suite_and_report.sh --run-type L1_TO_L1
#   ./run_perf_suite_and_report.sh --run-type UNPACK_ISOLATE --suite-ids 11,12,13,14
#   ./run_perf_suite_and_report.sh L1_TO_L1
RUN_TYPE_FILTER=""
SUITE_IDS_FILTER=""
APPEND_REPORT=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-type)
      RUN_TYPE_FILTER="$2"
      shift 2
      ;;
    --suite-ids)
      SUITE_IDS_FILTER="$2"
      shift 2
      ;;
    --append-report)
      APPEND_REPORT=1
      shift
      ;;
    --help|-h)
      echo "Usage: $0 [--run-type PerfRunType] [--suite-ids ID,ID,...] [--append-report]"
      echo "  PerfRunType one of: ${PERF_RUN_TYPES[*]}"
      echo "  suite-ids are the numeric ids from the suite map (e.g. 11,12,13,14)"
      exit 0
      ;;
    -*)
      echo "ERROR: unknown option: $1" >&2
      exit 2
      ;;
    *)
      RUN_TYPE_FILTER="$1"
      shift
      ;;
  esac
done

if [[ -n "$RUN_TYPE_FILTER" ]]; then
  valid=0
  for rt in "${PERF_RUN_TYPES[@]}"; do
    if [[ "$rt" == "$RUN_TYPE_FILTER" ]]; then
      valid=1
      break
    fi
  done
  if [[ $valid -ne 1 ]]; then
    echo "ERROR: invalid PerfRunType '${RUN_TYPE_FILTER}'. Expected one of: ${PERF_RUN_TYPES[*]}" >&2
    exit 2
  fi
  PERF_RUN_TYPES=("$RUN_TYPE_FILTER")
  REPORT="perf_suite_results_${RUN_TYPE_FILTER}.md"
fi

# suite_id:test_file  (matmul excluded)
tests=(
  "02:perf_eltwise_unary_datacopy_quasar.py"
  "03:perf_eltwise_binary_broadcast_quasar.py"
  "04:perf_eltwise_binary_quasar.py"
  "05:perf_unpack_tilize_quasar.py"
  "06:perf_unpack_unary_operand_quasar.py"
  "07:perf_transpose_dest_quasar.py"
  "08:perf_pack_quasar.py"
  "09:perf_pack_untilize_quasar.py"
  "10:perf_unary_broadcast_quasar.py"
  "11:perf_pack_l1_acc_quasar.py"
  "12:perf_reduce_quasar.py"
  "13:perf_eltwise_binary_reuse_dest_quasar.py"
  "14:perf_unpack_reduce_col_tilizeA_strided_quasar.py"
)

cleanup_procs() {
  pkill -9 -f 'tt-exalens --port=5556' 2>/dev/null || true
  pkill -9 -f 'quasar-1x3_run_dev' 2>/dev/null || true
  pkill -9 -f "pytest.*perf_" 2>/dev/null || true
  sleep 1
}

extract_reason() {
  local out="$1"
  local status="$2"
  if [[ "$status" == "PASSED" ]]; then
    echo "All collected tests in this file passed."
    return
  fi
  if [[ "$status" == "HANG" ]]; then
    echo "Single test exceeded ${TEST_HANG_SEC}s (hang)."
    return
  fi
  if [[ "$status" == "EXALENS_TIMEOUT" ]]; then
    echo "tt-exalens did not become ready within 600s."
    return
  fi
  if grep -q "FATAL: SSH" "$out" 2>/dev/null; then
    grep -m1 'FATAL:' "$out" | sed 's/.*\(FATAL:.*\)/\1/'
    return
  fi
  if grep -q "did not become ready" "$out" 2>/dev/null; then
    echo "tt-exalens did not become ready within timeout."
    return
  fi
  if grep -q "Error type:" "$out" 2>/dev/null; then
    local etype trace
    etype=$(grep -m1 "Error type:" "$out" | sed 's/.*Error type: //')
    trace=$(grep -A3 "Python Call trace:" "$out" | tail -n +2 | head -3 | tr '\n' ' ' | sed 's/[[:space:]]\+/ /g')
    local msg
    msg=$(grep -E "TypeError|AssertionError|ValueError|RuntimeError|KeyError" "$out" | head -1 | sed 's/^[[:space:]]*//')
    echo "${etype}: ${msg}. Trace: ${trace}"
    return
  fi
  if grep -qE '^FAILED ' "$out" 2>/dev/null; then
    grep -m1 '^FAILED ' "$out"
    return
  fi
  if grep -q "no tests ran" "$out" 2>/dev/null; then
    echo "No tests ran (bring-up/collection aborted)."
    return
  fi
  tail -n 8 "$out" | tr '\n' ' ' | sed 's/[[:space:]]\+/ /g' | cut -c1-350
}

classify_failure() {
  local output_file="$1"
  if grep -q "tt-exalens did not become ready within 600s" "${output_file}" 2>/dev/null; then
    echo "EXALENS_TIMEOUT"
  elif grep -q "Waiting for tt-exalens" "${output_file}" 2>/dev/null && \
       ! grep -q "tt-exalens ready" "${output_file}" 2>/dev/null; then
    echo "EXALENS_TIMEOUT"
  else
    echo "TEST_FAILURE"
  fi
}

# Echo the nodeid of the currently running test (started, not yet finished), or empty.
latest_open_test_line() {
  local output_file="$1"
  local last_info last_nr last_text
  last_info=$(grep -nE '\.py::' "${output_file}" 2>/dev/null | tail -1 || true)
  [[ -z "$last_info" ]] && return 0
  last_nr="${last_info%%:*}"
  last_text="${last_info#*:}"
  # Same-line completion.
  if [[ "$last_text" =~ (PASSED|FAILED|SKIPPED|ERROR|XFAIL|XPASS)$ ]]; then
    return 0
  fi
  # Completion on a later line (e.g. after live-log setup).
  if tail -n +"$((last_nr + 1))" "${output_file}" 2>/dev/null \
       | grep -qE '^(PASSED|FAILED|SKIPPED|ERROR|XFAIL|XPASS)$|(PASSED|FAILED|SKIPPED|ERROR|XFAIL|XPASS)$'; then
    return 0
  fi
  printf '%s\n' "$last_text"
}

write_report() {
  local tmp
  tmp=$(mktemp)
  {
    echo "# Quasar perf suite results"
    echo
    echo "Started: ${STARTED_AT}"
    echo "Updated: $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
    echo "Host: $(hostname)"
    echo "Command pattern: \`pytest -x --run-simulator --port=5556 --timeout=1000 --speed-of-light -k PerfRunType.<PerfRunType> <test> > perf_output_<PerfRunType>_<suite_id>.txt\`"
    echo
    echo "| # | Run type | Test file | Status | Duration | Output | Reason |"
    echo "|---|----------|-----------|--------|----------|--------|--------|"
    if [[ -f "$REPORT.rows" ]]; then
      cat "$REPORT.rows"
    fi
    echo
    echo "## Notes"
    echo
    echo "- \`--speed-of-light\` and \`-k PerfRunType.<PerfRunType>\` applied for each of: ${PERF_RUN_TYPES[*]} (bare PACK_ISOLATE also matches UNPACK_ISOLATE)."
    echo "- \`-x\` stops each file on first failure."
    echo "- tt-exalens 600s timeout: retry up to ${MAX_EXALENS_RETRIES} times. Other failures are not retried."
    echo "- Single-test hang threshold: ${TEST_HANG_SEC}s after tt-exalens is ready (not retried)."
    echo "- Emulator bring-up (tt-exalens / Zebu) is included in duration."
  } > "$tmp"
  mv "$tmp" "$REPORT"
}

# Run one pytest attempt with hang / exalens monitoring. Echoes status token.
run_one_attempt() {
  local test_file="$1"
  local output_file="$2"
  local run_type="$3"

  cleanup_procs
  : > "${output_file}"

  # Use "PerfRunType.<name>" so -k PACK_ISOLATE does not also match UNPACK_ISOLATE
  # (pytest -k is a substring match on the node id).
  pytest -x --run-simulator --port=5556 --timeout=1000 \
    --speed-of-light -k "PerfRunType.${run_type}" "${test_file}" \
    > "${output_file}" 2>&1 &
  local pytest_pid=$!

  local start_ts
  start_ts=$(date +%s)
  local exalens_ready=0
  local prev_exalens_ready=0
  local status="unknown"
  local last_open_line=""
  local open_test_start=0
  local last_announced=""

  while kill -0 "$pytest_pid" 2>/dev/null; do
    sleep "$POLL_SEC"
    local now elapsed
    now=$(date +%s)
    elapsed=$((now - start_ts))

    if [[ -f "${output_file}" ]]; then
      if grep -q "tt-exalens ready" "${output_file}" 2>/dev/null; then
        exalens_ready=1
      fi
    fi

    # When exalens becomes ready, restart the hang clock for the in-flight test
    # so bring-up time is not counted against the 5 min per-test budget.
    if [[ $exalens_ready -eq 1 && $prev_exalens_ready -eq 0 ]]; then
      prev_exalens_ready=1
      if [[ -n "$last_open_line" ]]; then
        open_test_start=$now
      fi
    fi

    local open_line
    open_line=$(latest_open_test_line "${output_file}")
    if [[ -n "$open_line" && "$open_line" != "$last_open_line" ]]; then
      last_open_line="$open_line"
      open_test_start=$now
      if [[ "$open_line" != "$last_announced" ]]; then
        last_announced="$open_line"
        echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] START ${open_line}" >&2
      fi
    elif [[ -z "$open_line" ]]; then
      open_test_start=0
      last_open_line=""
    fi

    local hung=0
    local reason=""

    # Exalens bring-up timeout (retryable).
    if [[ $exalens_ready -eq 0 ]] && grep -q "Waiting for tt-exalens" "${output_file}" 2>/dev/null; then
      if [[ $elapsed -ge $EXALENS_FAIL_SEC ]]; then
        hung=1
        reason="exalens not ready after ${elapsed}s"
        status="EXALENS_TIMEOUT"
      fi
    # Per-test hang: only after exalens is ready (bring-up uses EXALENS_FAIL_SEC).
    elif [[ $open_test_start -gt 0 && $exalens_ready -eq 1 ]]; then
      local test_elapsed=$((now - open_test_start))
      if [[ $test_elapsed -ge $TEST_HANG_SEC ]]; then
        hung=1
        reason="single test running ${test_elapsed}s (>${TEST_HANG_SEC}s)"
        status="HANG"
      fi
    fi

    if [[ $hung -eq 1 ]]; then
      echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] HUNG (${reason}) — killing attempt" >&2
      kill -9 "$pytest_pid" 2>/dev/null || true
      cleanup_procs
      break
    fi

    if [[ $((elapsed % 120)) -lt $POLL_SEC ]]; then
      local open_age=0
      if [[ $open_test_start -gt 0 ]]; then
        open_age=$((now - open_test_start))
      fi
      echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] running... elapsed=${elapsed}s exalens_ready=${exalens_ready} open_test=${open_age}s" >&2
    fi
  done

  if [[ "$status" != "EXALENS_TIMEOUT" && "$status" != "HANG" ]]; then
    wait "$pytest_pid" 2>/dev/null
    local exit_code=$?
    if [[ $exit_code -eq 0 ]]; then
      status="PASSED"
    else
      status="$(classify_failure "${output_file}")"
    fi
  fi

  cleanup_procs
  echo "${status}"
}

run_one_suite_entry() {
  local suite_id="$1"
  local test_file="$2"
  local run_type="$3"
  local row_num="$4"
  local out_file="perf_output_${run_type}_${suite_id}.txt"

  echo ""
  echo "========== [${row_num}] ${run_type} / ${test_file} -> ${out_file} =========="
  date -u +%Y-%m-%dT%H:%M:%SZ

  local attempt=1
  local status="FAILED"
  local start end dur dur_h
  start=$(date +%s)

  while [[ $attempt -le $MAX_EXALENS_RETRIES ]]; do
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Attempt ${attempt}/${MAX_EXALENS_RETRIES}"
    status=$(run_one_attempt "${test_file}" "${out_file}" "${run_type}")

    if [[ "$status" == "PASSED" ]]; then
      echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] PASSED on attempt ${attempt}"
      break
    fi

    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] failed: ${status}"
    if [[ "$status" == "EXALENS_TIMEOUT" ]]; then
      echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Retrying after exalens timeout..."
      attempt=$((attempt + 1))
      if [[ $attempt -le $MAX_EXALENS_RETRIES ]]; then
        sleep "$COOLDOWN_SEC"
      fi
      continue
    fi

    # Any other failure (TEST_FAILURE / HANG / ...): do not retry.
    break
  done

  if [[ "$status" == "EXALENS_TIMEOUT" && "$attempt" -gt $MAX_EXALENS_RETRIES ]]; then
    status="EXALENS_TIMEOUT"
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] gave up after ${MAX_EXALENS_RETRIES} exalens retries"
  fi

  # Refine PASSED/FAILED from log if attempt completed normally.
  if [[ "$status" != "HANG" && "$status" != "EXALENS_TIMEOUT" ]]; then
    if grep -qE '^=+ .* passed in' "$out_file" && ! grep -qE '^FAILED |^=+ FAILURES|Error type:' "$out_file"; then
      status="PASSED"
    elif grep -q "no tests ran" "$out_file"; then
      status="NO_TESTS"
    elif grep -qE '^FAILED |^=+ FAILURES|Error type:' "$out_file"; then
      status="FAILED"
    elif [[ "$status" == "TEST_FAILURE" ]]; then
      status="FAILED"
    fi
  fi

  end=$(date +%s)
  dur=$((end - start))
  dur_h=$(printf '%dm%02ds' $((dur / 60)) $((dur % 60)))
  local reason
  reason=$(extract_reason "$out_file" "$status" | tr '|' '/' | tr '\n' ' ' | sed 's/|/\\|/g' | cut -c1-300)

  printf '| %d | `%s` | `%s` | **%s** | %s | `%s` | %s |\n' \
    "$row_num" "$run_type" "$test_file" "$status" "$dur_h" "$out_file" "$reason" >> "$REPORT.rows"
  write_report
  echo "Result: $status (${dur_h}) — $reason"
  sleep "$COOLDOWN_SEC"
}

: > "$REPORT.rows"
row=0
if [[ $APPEND_REPORT -eq 1 && -f "$REPORT" ]]; then
  # Keep prior markdown table rows (PASSED/HANG) and append new results.
  grep -E '^\| [0-9]+' "$REPORT" > "$REPORT.rows" || true
  if [[ -s "$REPORT.rows" ]]; then
    row=$(awk -F'|' 'END { gsub(/ /, "", $2); print $2+0 }' "$REPORT.rows")
  fi
fi
write_report

suite_id_wanted() {
  local sid="$1"
  [[ -z "$SUITE_IDS_FILTER" ]] && return 0
  local IFS=','
  local id
  # shellcheck disable=SC2086
  for id in $SUITE_IDS_FILTER; do
    if [[ "$id" == "$sid" ]]; then
      return 0
    fi
  done
  return 1
}

for run_type in "${PERF_RUN_TYPES[@]}"; do
  for entry in "${tests[@]}"; do
    suite_id="${entry%%:*}"
    test_file="${entry##*:}"
    if ! suite_id_wanted "$suite_id"; then
      continue
    fi
    row=$((row + 1))
    run_one_suite_entry "$suite_id" "$test_file" "$run_type" "$row"
  done
done

echo "Finished: $(date -u +"%Y-%m-%d %H:%M:%S UTC")" >> "$REPORT"
echo "DONE -> $REPORT"
cat "$REPORT"
