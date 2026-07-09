cd ~/rtp/tt-metal

STAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR=~/rtp/tt-metal/logs
BLOCK_LOG=$LOG_DIR/seamless_block_${STAMP}.log
E2E_LOG=$LOG_DIR/seamless_e2e_${STAMP}.log
DEMO_LOG=$LOG_DIR/seamless_demo_${STAMP}.log
RUN_LOCK=$LOG_DIR/seamless_run.lock
mkdir -p "$LOG_DIR"

# P150 allows one process to hold CHIP_IN_USE at a time. Fail fast if something
# else is already using the device instead of blocking forever in UMD.
check_device_free() {
  local conflicts
  conflicts=$(
    pgrep -af 'seamless_m4t_v2_large|demo_perf_sweep\.py' 2>/dev/null \
      | grep -Ev 'pgrep -af|run_tests\.sh' || true
  )
  if [ -n "$conflicts" ]; then
    echo "ERROR: TT device appears busy — another seamless process is still running:" >&2
    echo "$conflicts" >&2
    echo >&2
    echo "Kill the stale process(es) first, then re-run. Example:" >&2
    echo "  pkill -f 'seamless_m4t_v2_large|demo_perf_sweep'" >&2
    echo "To stop a previous run_tests.sh session (kills its whole process group):" >&2
    echo "  kill -TERM -\$(cat $RUN_LOCK 2>/dev/null)" >&2
    return 1
  fi
}

if ! check_device_free; then
  exit 1
fi

setsid bash -c '
  export MESH_DEVICE=P150
  export PYTHONUNBUFFERED=1
  PCC_DIR=models/experimental/seamless_m4t_v2_large/tests/pcc
  BLOCK_LOG="'"$BLOCK_LOG"'"
  E2E_LOG="'"$E2E_LOG"'"
  DEMO_LOG="'"$DEMO_LOG"'"
  RUN_LOCK="'"$RUN_LOCK"'"
  OVERALL_START=$(date +%s)

  # Only one run_tests.sh session at a time.
  exec 9>"$RUN_LOCK"
  if ! flock -n 9; then
    msg="ERROR: Another seamless test run is already in progress (lock: $RUN_LOCK)"
    echo "$msg" | tee -a "$BLOCK_LOG" "$E2E_LOG" "$DEMO_LOG"
    exit 1
  fi
  echo $$ >&9

  cleanup() {
    local rc=$?
    trap - EXIT INT TERM
    # Kill pytest/python/tee children so killing this shell does not leave orphans.
    jobs -pr | xargs -r kill -TERM 2>/dev/null
    wait 2>/dev/null
    flock -u 9 2>/dev/null || true
    exit "$rc"
  }
  trap cleanup EXIT INT TERM

  format_duration() {
    local s=$1
    printf "%dh %dm %ds" $((s / 3600)) $(((s % 3600) / 60)) $((s % 60))
  }

  run_pytest() {
    local log_file="$1"
    local name="$2"
    local target="$3"
    local filter="${4:-}"
    local start elapsed rc
    start=$(date +%s)
    echo "========== $name (started $(date -Iseconds)) ==========" | tee -a "$log_file"
    if [ -n "$filter" ]; then
      pytest "$target" -k "$filter" -s -v --durations=0 2>&1 | tee -a "$log_file"
    else
      pytest "$target" -s -v --durations=0 2>&1 | tee -a "$log_file"
    fi
    rc=${PIPESTATUS[0]}
    elapsed=$(( $(date +%s) - start ))
    echo "========== $name finished in $(format_duration "$elapsed") (exit $rc) ==========" | tee -a "$log_file"
    echo | tee -a "$log_file"
    return "$rc"
  }

  run_python() {
    local log_file="$1"
    local name="$2"
    local target="$3"
    local start elapsed rc
    start=$(date +%s)
    echo "========== $name (started $(date -Iseconds)) ==========" | tee -a "$log_file"
    python "$target" 2>&1 | tee -a "$log_file"
    rc=${PIPESTATUS[0]}
    elapsed=$(( $(date +%s) - start ))
    echo "========== $name finished in $(format_duration "$elapsed") (exit $rc) ==========" | tee -a "$log_file"
    echo | tee -a "$log_file"
    return "$rc"
  }

  finish_section() {
    local log_file="$1"
    local label="$2"
    local section_start="$3"
    local elapsed=$(( $(date +%s) - section_start ))
    echo "========== $label finished in $(format_duration "$elapsed") (started $(date -Iseconds -d "@$section_start")) ==========" | tee -a "$log_file"
    echo | tee -a "$log_file"
  }

  BLOCK_START=$(date +%s)
  echo "========== seamless block tests (started $(date -Iseconds)) ==========" | tee -a "$BLOCK_LOG"
  echo | tee -a "$BLOCK_LOG"

  run_pytest "$BLOCK_LOG" "text encoder PCC" \
    "$PCC_DIR/test_text_encoder.py"

  run_pytest "$BLOCK_LOG" "speech encoder PCC" \
    "$PCC_DIR/test_speech_encoder.py"

  run_pytest "$BLOCK_LOG" "text decoder PCC" \
    "$PCC_DIR/test_text_decoder.py"

  run_pytest "$BLOCK_LOG" "text-to-unit PCC" \
    "$PCC_DIR/test_text_to_unit.py"

  run_pytest "$BLOCK_LOG" "code hifigan PCC" \
    "$PCC_DIR/test_code_hifigan.py"

  run_pytest "$BLOCK_LOG" "prefill PCC" \
    "$PCC_DIR/test_prefill.py"

  run_pytest "$BLOCK_LOG" "decode PCC" \
    "$PCC_DIR/test_decode.py"

  finish_section "$BLOCK_LOG" "ALL BLOCK TESTS" "$BLOCK_START"

  E2E_START=$(date +%s)
  echo "========== seamless e2e tests (started $(date -Iseconds)) ==========" | tee -a "$E2E_LOG"
  echo | tee -a "$E2E_LOG"

  run_pytest "$E2E_LOG" "logit PCC sweep" \
    "$PCC_DIR/test_seamless_e2e_logit_pcc_sweep.py" "sweep and not sanity"

  run_pytest "$E2E_LOG" "token matching sweep" \
    "$PCC_DIR/test_seamless_e2e_token_matching_sweep.py" "sweep and not sanity"

  run_pytest "$E2E_LOG" "WER sweep" \
    "$PCC_DIR/test_seamless_e2e_wer_sweep.py" "sweep and not sanity"

  finish_section "$E2E_LOG" "ALL E2E TESTS" "$E2E_START"

  DEMO_START=$(date +%s)
  echo "========== seamless demo tests (started $(date -Iseconds)) ==========" | tee -a "$DEMO_LOG"
  echo | tee -a "$DEMO_LOG"

  run_python "$DEMO_LOG" "demo perf sweep" \
    models/experimental/seamless_m4t_v2_large/scripts/demo_perf_sweep.py

  finish_section "$DEMO_LOG" "ALL DEMO TESTS" "$DEMO_START"

  total=$(( $(date +%s) - OVERALL_START ))
  summary="========== FULL RUN finished in $(format_duration "$total") (started $(date -Iseconds -d "@$OVERALL_START")) =========="
  echo "$summary" | tee -a "$BLOCK_LOG" "$E2E_LOG" "$DEMO_LOG"
' < /dev/null > /dev/null 2>&1 &

SESSION_PID=$!
echo "PID: $SESSION_PID (session leader — stop with: kill -TERM -$SESSION_PID)"
echo "Block log: $BLOCK_LOG"
echo "E2E log:   $E2E_LOG"
echo "Demo log:  $DEMO_LOG"
