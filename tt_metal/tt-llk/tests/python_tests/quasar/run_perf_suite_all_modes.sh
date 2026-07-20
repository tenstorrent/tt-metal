#!/usr/bin/env bash
# Run each quasar perf test with ALL PerfRunTypes enabled (no -k filter).
# No pytest --timeout. Hang-monitors open tests; on hang clears local + Zebu.
#
# Outputs: perf_output_all_<suite_id>.txt
# Updates: perf_suite_progress_summary.md after each suite entry
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

SUITE_DIR="$(pwd)"
PROGRESS="${SUITE_DIR}/perf_suite_progress_summary.md"
ORCH_LOG="${SUITE_DIR}/perf_suite_all_modes.log"
STARTED_AT=$(date -u +"%Y-%m-%d %H:%M:%S UTC")

EXALENS_FAIL_SEC=610    # retry after ~10 min exalens startup failure
# Hang = ~5–10 min of no progress after exalens ready (then skip mode + retry).
TEST_HANG_SEC=600       # single open test >10 min after exalens ready => hang
STALL_HANG_SEC=600      # no output growth for 10 min after exalens ready => hang
POLL_SEC=10
COOLDOWN_SEC=60
MAX_EXALENS_RETRIES=5
MAX_HANG_RETRIES=5      # after hang: record PerfRunType skip, retry same test
ARTEFACTS_DIR="${ARTEFACTS_DIR:-/tmp/tt-llk-build}"

SSH_OPTS=(
  -T
  -o StrictHostKeyChecking=no
  -o UserKnownHostsFile=/dev/null
  -o BatchMode=yes
  -o PreferredAuthentications=publickey
  -o ConnectTimeout=15
  -o ConnectionAttempts=1
  -o ServerAliveInterval=5
  -o ServerAliveCountMax=3
)

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

log() {
  # Always to log file + stderr so status=$(run_one_attempt) is not polluted.
  printf '%s\n' "$*" | tee -a "$ORCH_LOG" >&2
}

last_zebu_host() {
  # Most recent instrumented launcher host selection.
  grep -h 'Using SSH_MACHINE_NAME=' emu_*.log 2>/dev/null \
    | tail -1 \
    | sed 's/.*Using SSH_MACHINE_NAME=//' \
    | tr -d '[:space:]'
}

cleanup_zebu() {
  local host
  host="$(last_zebu_host)"
  if [[ -z "$host" ]]; then
    log "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Zebu cleanup: no SSH_MACHINE_NAME in emu_*.log"
    return 0
  fi
  log "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Zebu cleanup on ${host}..."
  # Kill leftover remote emulator / zrun jobs for this user.
  ssh "${SSH_OPTS[@]}" "$host" bash -s <<'EOS' || true
pkill -9 -u "$USER" -f '[z]run' 2>/dev/null || true
pkill -9 -u "$USER" -f 'test_umd_remote' 2>/dev/null || true
pkill -9 -u "$USER" -f 'zcui\.work' 2>/dev/null || true
pkill -9 -u "$USER" -f 'zebu' 2>/dev/null || true
# Drop stale make/test shells from aether verification if still around.
pkill -9 -u "$USER" -f 'verification/emu' 2>/dev/null || true
echo "remote cleanup done on $(hostname)"
EOS
}

cleanup_procs() {
  log "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Cleaning local pytest/exalens/emu launcher..."
  # Kill by PID to avoid matching this orchestrator shell's command text.
  local pid
  for pid in $(ps -eo pid,args | grep -E 'tt-exalens --port=5556|quasar-1x3_run_dev|/pytest .*perf_' | grep -v grep | awk '{print $1}'); do
    [[ "$pid" -eq "$$" || "$pid" -eq "$PPID" ]] && continue
    kill -9 "$pid" 2>/dev/null || true
  done
  sleep 1
  cleanup_zebu
  sleep 2
}

detect_hung_perf_run_type() {
  # Newest build.h under artefacts records the PerfRunType that was building/running.
  local newest
  newest=$(find "${ARTEFACTS_DIR}" -name build.h -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)
  if [[ -z "$newest" || ! -f "$newest" ]]; then
    echo ""
    return 0
  fi
  sed -n 's/.*PERF_RUN_TYPE = PerfRunType::\([A-Za-z0-9_]*\).*/\1/p' "$newest" | head -1
}

record_hang_skip() {
  local test_file="$1"
  local run_type="$2"
  if [[ -z "$run_type" ]]; then
    log "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] WARN: could not detect hung PerfRunType; not recording skip"
    return 1
  fi
  log "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Recording hang skip: ${test_file} / ${run_type}"
  PYTHONPATH="${LLK_ROOT}/tests/python_tests${PYTHONPATH:+:$PYTHONPATH}" \
    python - "$test_file" "$run_type" <<'PY'
import sys
from helpers.perf_hang_skips import add_skip, load_skips
test_file, run_type = sys.argv[1], sys.argv[2]
added = add_skip(test_file, run_type)
print(f"skip {'added' if added else 'already present'}: {test_file} -> {run_type}")
print("current skips:", load_skips())
PY
}

extract_reason() {
  local out="$1"
  local status="$2"
  if [[ "$status" == "PASSED" ]]; then
    echo "All collected tests in this file passed."
    return
  fi
  if [[ "$status" == "HANG" ]]; then
    echo "Single test exceeded hang threshold (open-test or stall)."
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
  if grep -q "PerfSchemaError\|schema contamination" "$out" 2>/dev/null; then
    echo "Perf report schema contamination."
    return
  fi
  if grep -qE '^FAILED |^ERROR ' "$out" 2>/dev/null; then
    grep -m1 -E '^FAILED |^ERROR ' "$out"
    return
  fi
  if grep -q "Error type:" "$out" 2>/dev/null; then
    grep -m1 "Error type:" "$out" | sed 's/.*Error type: //'
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

latest_open_test_line() {
  local output_file="$1"
  local last_info last_nr last_text
  last_info=$(grep -nE '\.py::' "${output_file}" 2>/dev/null | tail -1 || true)
  [[ -z "$last_info" ]] && return 0
  last_nr="${last_info%%:*}"
  last_text="${last_info#*:}"
  if [[ "$last_text" =~ (PASSED|FAILED|SKIPPED|ERROR|XFAIL|XPASS)$ ]]; then
    return 0
  fi
  if tail -n +"$((last_nr + 1))" "${output_file}" 2>/dev/null \
       | grep -qE '^(PASSED|FAILED|SKIPPED|ERROR|XFAIL|XPASS)$|(PASSED|FAILED|SKIPPED|ERROR|XFAIL|XPASS)$'; then
    return 0
  fi
  printf '%s\n' "$last_text"
}

fmt_dur() {
  local sec="$1"
  printf '%dm%02ds' $((sec / 60)) $((sec % 60))
}

update_progress() {
  local suite_id="$1"
  local test_file="$2"
  local status="$3"
  local duration="$4"
  local out_file="$5"
  local reason="$6"
  local now
  now=$(date -u +"%Y-%m-%d %H:%M:%S UTC")

  # Ensure a results table section exists; rewrite the ALL-MODES section.
  python3 - "$PROGRESS" "$suite_id" "$test_file" "$status" "$duration" "$out_file" "$reason" "$now" "$STARTED_AT" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
suite_id, test_file, status, duration, out_file, reason, now, started = sys.argv[2:10]
test_base = test_file.removesuffix(".py")

text = path.read_text() if path.exists() else ""

header = f"""# Quasar perf suite progress summary

Updated: {now}
Host: (all-modes run)
Started: {started}

## Mode: ALL PerfRunTypes (no -k, no pytest --timeout)

Each test file runs L1_TO_L1 + UNPACK_ISOLATE + MATH_ISOLATE + PACK_ISOLATE
in a single parametrization (homogeneous CSV schema). L1_CONGESTION excluded (hangs).
On hang (~10 min no progress): clear local+Zebu, record PerfRunType skip in
perf_hang_skips.json, retry the same test.

Hang policy: open-test >{600}s or stall >{600}s after tt-exalens ready → kill local + Zebu,
record PerfRunType skip, retry.
Exalens 600s timeout → retry up to 5×.

| ID | Test | Status | Duration | Output | Reason |
|----|------|--------|----------|--------|--------|
"""

# Parse existing ALL-MODES table rows if present, else start fresh.
rows = {}
order = []
in_table = False
if "## Mode: ALL PerfRunTypes" in text:
    for line in text.splitlines():
        if line.startswith("| ID | Test |"):
            in_table = True
            continue
        if in_table:
            if not line.startswith("|"):
                break
            if line.startswith("|----"):
                continue
            parts = [p.strip() for p in line.strip("|").split("|")]
            if len(parts) >= 6:
                sid = parts[0]
                rows[sid] = line
                order.append(sid)

row = (
    f"| {suite_id} | `{test_base}` | **{status}** | {duration} | "
    f"`{out_file}` | {reason} |"
)
if suite_id not in rows:
    order.append(suite_id)
rows[suite_id] = row

# Keep known suite order 02..14
preferred = [f"{i:02d}" for i in range(2, 15)]
ordered = [s for s in preferred if s in rows] + [s for s in order if s not in preferred]

body = "\n".join(rows[s] for s in ordered)
path.write_text(header + body + "\n")
PY
}

run_one_attempt() {
  local test_file="$1"
  local output_file="$2"

  cleanup_procs
  : > "${output_file}"

  # No --timeout, no -k: run full file with all PerfRunTypes from PERF_RUN_TYPES_QUASAR.
  pytest -x --run-simulator --port=5556 --speed-of-light \
    "${test_file}" > "${output_file}" 2>&1 &
  local pytest_pid=$!

  local start_ts now elapsed
  start_ts=$(date +%s)
  local exalens_ready=0
  local prev_exalens_ready=0
  local status="unknown"
  local last_open_line=""
  local open_test_start=0
  local last_announced=""
  local last_size=0
  local last_change_ts=$start_ts

  while kill -0 "$pytest_pid" 2>/dev/null; do
    sleep "$POLL_SEC"
    now=$(date +%s)
    elapsed=$((now - start_ts))

    if [[ -f "${output_file}" ]]; then
      if grep -q "tt-exalens ready" "${output_file}" 2>/dev/null; then
        exalens_ready=1
      fi
      local cur_size
      cur_size=$(stat -c%s "${output_file}" 2>/dev/null || echo 0)
      if [[ "$cur_size" -ne "$last_size" ]]; then
        last_size=$cur_size
        last_change_ts=$now
      fi
    fi

    if [[ $exalens_ready -eq 1 && $prev_exalens_ready -eq 0 ]]; then
      prev_exalens_ready=1
      last_change_ts=$now
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
        log "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] START ${open_line}"
      fi
    elif [[ -z "$open_line" ]]; then
      open_test_start=0
      last_open_line=""
    fi

    local hung=0
    local reason=""

    if [[ $exalens_ready -eq 0 ]] && grep -q "Waiting for tt-exalens" "${output_file}" 2>/dev/null; then
      if [[ $elapsed -ge $EXALENS_FAIL_SEC ]]; then
        hung=1
        reason="exalens not ready after ${elapsed}s"
        status="EXALENS_TIMEOUT"
      fi
    elif [[ $exalens_ready -eq 1 ]]; then
      local stall=$((now - last_change_ts))
      if [[ $open_test_start -gt 0 ]]; then
        local test_elapsed=$((now - open_test_start))
        if [[ $test_elapsed -ge $TEST_HANG_SEC ]]; then
          hung=1
          reason="single test running ${test_elapsed}s (>${TEST_HANG_SEC}s)"
          status="HANG"
        fi
      fi
      if [[ $hung -eq 0 && $stall -ge $STALL_HANG_SEC ]]; then
        hung=1
        reason="no output change for ${stall}s"
        status="HANG"
      fi
    fi

    if [[ $hung -eq 1 ]]; then
      log "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] HUNG (${reason}) — killing attempt + Zebu"
      kill -9 "$pytest_pid" 2>/dev/null || true
      cleanup_procs
      break
    fi

    if [[ $((elapsed % 120)) -lt $POLL_SEC ]]; then
      local open_age=0
      if [[ $open_test_start -gt 0 ]]; then
        open_age=$((now - open_test_start))
      fi
      local stall_age=$((now - last_change_ts))
      log "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] running... elapsed=${elapsed}s exalens_ready=${exalens_ready} open_test=${open_age}s stall=${stall_age}s"
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
  local out_file="perf_output_all_${suite_id}.txt"

  log ""
  log "========== [${suite_id}] ALL / ${test_file} -> ${out_file} =========="
  log "$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  local attempt=1
  local hang_attempt=0
  local status="FAILED"
  local start end dur
  local hang_notes=""
  start=$(date +%s)

  while [[ $attempt -le $MAX_EXALENS_RETRIES ]]; do
    log "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Attempt ${attempt}/${MAX_EXALENS_RETRIES} (hang_retries=${hang_attempt}/${MAX_HANG_RETRIES})"
    status=$(run_one_attempt "${test_file}" "${out_file}" | tail -n1 | tr -d '\r' | awk '{$1=$1;print}')

    if [[ "$status" == "PASSED" ]]; then
      log "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] PASSED on attempt ${attempt}"
      break
    fi

    log "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] failed: ${status}"
    if [[ "$status" == "EXALENS_TIMEOUT" ]]; then
      log "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Retrying after exalens timeout..."
      attempt=$((attempt + 1))
      if [[ $attempt -le $MAX_EXALENS_RETRIES ]]; then
        sleep "$COOLDOWN_SEC"
      fi
      continue
    fi

    if [[ "$status" == "HANG" ]]; then
      local hung_rt
      hung_rt=$(detect_hung_perf_run_type)
      log "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Hang on ${test_file}; detected PerfRunType=${hung_rt:-unknown}"
      if record_hang_skip "${test_file}" "${hung_rt}"; then
        hang_notes="${hang_notes} skipped ${hung_rt};"
        hang_attempt=$((hang_attempt + 1))
        if [[ $hang_attempt -le $MAX_HANG_RETRIES ]]; then
          log "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Retrying ${test_file} after hang-skip of ${hung_rt}"
          sleep "$COOLDOWN_SEC"
          # Hang retries do not consume exalens attempt budget.
          continue
        fi
        log "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Exhausted hang retries for ${test_file}"
      fi
    fi
    break
  done

  if [[ "$status" == "EXALENS_TIMEOUT" && "$attempt" -gt $MAX_EXALENS_RETRIES ]]; then
    status="EXALENS_TIMEOUT"
  fi

  end=$(date +%s)
  dur=$((end - start))
  local dur_h
  dur_h=$(fmt_dur "$dur")
  local reason
  reason=$(extract_reason "${out_file}" "${status}")
  if [[ -n "$hang_notes" ]]; then
    reason="${reason} Hang-skips:${hang_notes}"
  fi

  log "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] RESULT ${suite_id} ${test_file} status=${status} duration=${dur_h}"
  update_progress "${suite_id}" "${test_file}" "${status}" "${dur_h}" "${out_file}" "${reason}"
  sleep "$COOLDOWN_SEC"
}

# --- main ---
APPEND_MODE=0
if [[ -n "${SUITE_IDS:-}" || "${APPEND_PROGRESS:-0}" == "1" ]]; then
  APPEND_MODE=1
fi

if [[ $APPEND_MODE -eq 1 ]]; then
  {
    echo ""
    echo "================================================================"
    echo "All-modes suite (append/SUITE_IDS=${SUITE_IDS:-}) started: ${STARTED_AT}"
    echo "pytest: -x --run-simulator --port=5556 --speed-of-light  (no -k, no --timeout)"
    echo "Hang: open-test>${TEST_HANG_SEC}s or stall>${STALL_HANG_SEC}s after exalens ready"
    echo "================================================================"
  } | tee -a "$ORCH_LOG" >&2
else
  : > "$ORCH_LOG"
  log "================================================================"
  log "All-modes suite started: ${STARTED_AT}"
  log "pytest: -x --run-simulator --port=5556 --speed-of-light  (no -k, no --timeout)"
  log "Hang: open-test>${TEST_HANG_SEC}s or stall>${STALL_HANG_SEC}s after exalens ready"
  log "================================================================"

  # Fresh progress table only on full runs.
  cat > "$PROGRESS" <<EOF
# Quasar perf suite progress summary

Updated: ${STARTED_AT}
Started: ${STARTED_AT}

## Mode: ALL PerfRunTypes (no -k, no pytest --timeout)

Each test file runs L1_TO_L1 + UNPACK_ISOLATE + MATH_ISOLATE + PACK_ISOLATE
in a single parametrization (homogeneous CSV schema). L1_CONGESTION excluded (hangs).
On hang (~10 min no progress): clear local+Zebu, record PerfRunType skip in
perf_hang_skips.json, retry the same test.

Hang policy: open-test >${TEST_HANG_SEC}s or stall >${STALL_HANG_SEC}s after tt-exalens ready → kill local + Zebu.
Exalens 600s timeout → retry up to ${MAX_EXALENS_RETRIES}×.

| ID | Test | Status | Duration | Output | Reason |
|----|------|--------|----------|--------|--------|
EOF
fi

trap '' HUP

for entry in "${tests[@]}"; do
  suite_id="${entry%%:*}"
  test_file="${entry#*:}"
  # Optional: START_FROM=04 skips earlier suite ids.
  if [[ -n "${START_FROM:-}" ]]; then
    if [[ "$suite_id" < "$START_FROM" ]]; then
      log "Skipping ${suite_id} ${test_file} (START_FROM=${START_FROM})"
      continue
    fi
  fi
  # Optional: SUITE_IDS=02,03 runs only those ids.
  if [[ -n "${SUITE_IDS:-}" ]]; then
    case ",${SUITE_IDS}," in
      *",${suite_id},"*) ;;
      *)
        log "Skipping ${suite_id} ${test_file} (SUITE_IDS=${SUITE_IDS})"
        continue
        ;;
    esac
  fi
  run_one_suite_entry "$suite_id" "$test_file"
done

log "================================================================"
log "DONE_ALL_MODES $(date -u +%Y-%m-%dT%H:%M:%SZ)"
log "================================================================"
echo "DONE_ALL_MODES" >> "$ORCH_LOG"
