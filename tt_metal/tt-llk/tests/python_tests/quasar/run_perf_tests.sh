#!/usr/bin/env bash
set -uo pipefail

export LLK_ROOT=/proj_sw/user_dev/ndivnic/tt-metal/tt_metal/tt-llk
export CHIP_ARCH=quasar
export TT_METAL_SIMULATOR="/proj_sw/user_dev/${USER}/tt-umd-simulators/build/emu-quasar-1x3"
export TT_UMD_SIMULATOR_PATH="$TT_METAL_SIMULATOR"
export NNG_SOCKET_ADDR="tcp://tensix-l-01:54948"
export NNG_SOCKET_LOCAL_PORT=5555
source "$LLK_ROOT/tests/.venv/bin/activate"

cd /proj_sw/user_dev/ndivnic/tt-metal/tt_metal/tt-llk/tests/python_tests/quasar

EXALENS_FAIL_SEC=610       # retry after ~10 min exalens startup failure
STALL_HANG_SEC=900         # no output change for 15 minutes
POLL_SEC=30
COOLDOWN_SEC=60
MAX_RETRIES=5
L1_ONLY="${PERF_L1_ONLY:-1}"
CURRENT_TEST_FILE=""

patch_l1_only() {
    local file="$1"
    python3 - "$file" <<'PY'
import re, sys
path = sys.argv[1]
with open(path) as f:
    text = f.read()
new_text = re.sub(
    r"run_types=\[\s*\[[^\]]+\]\s*\]",
    "run_types=[[PerfRunType.L1_TO_L1]]",
    text,
    flags=re.DOTALL,
)
with open(path, "w") as f:
    f.write(new_text)
PY
}

restore_test_file() {
    local file="$1"
    if [[ -n "$file" && -f "${file}.bak" ]]; then
        mv "${file}.bak" "${file}"
    fi
}

on_exit() {
    restore_test_file "${CURRENT_TEST_FILE}"
}
trap on_exit EXIT

cleanup_hanging() {
    echo "[$(date '+%H:%M:%S')] Cleaning up hanging processes..." >&2
    pkill -9 quasar-1x3_run_ 2>/dev/null || true
    pkill -9 tt-exalens 2>/dev/null || true
    pkill -9 -f "pytest.*perf_" 2>/dev/null || true
    sleep 2
}

classify_failure() {
    local output_file="$1"
    if grep -q "tt-exalens did not become ready within 600s" "${output_file}" 2>/dev/null; then
        echo "EXALENS_TIMEOUT"
    elif grep -q "Waiting for tt-exalens" "${output_file}" 2>/dev/null && \
         ! grep -q "tt-exalens ready" "${output_file}" 2>/dev/null; then
        echo "EXALENS_TIMEOUT"
    elif grep -q "cannot unpack non-iterable int object" "${output_file}" 2>/dev/null; then
        echo "INPUT_DIMENSIONS"
    elif grep -q "No space left on device" "${output_file}" 2>/dev/null; then
        echo "DISK_FULL"
    else
        echo "TEST_FAILURE"
    fi
}

run_one_attempt() {
    local test_file="$1"
    local output_file="$2"

    cleanup_hanging
    restore_test_file "${test_file}"
    CURRENT_TEST_FILE="${test_file}"

    if [[ "$L1_ONLY" == "1" ]]; then
        cp "${test_file}" "${test_file}.bak"
        patch_l1_only "${test_file}"
    fi

    : > "${output_file}"
    pytest -x --run-simulator --port=5556 --timeout=1000 "${test_file}" > "${output_file}" 2>&1 &
    local pytest_pid=$!

    local start_ts=$(date +%s)
    local last_change_ts=$start_ts
    local last_size=0
    local exalens_ready=0
    local status="unknown"

    while kill -0 "$pytest_pid" 2>/dev/null; do
        sleep "$POLL_SEC"
        local now=$(date +%s)
        local elapsed=$((now - start_ts))

        if [[ -f "${output_file}" ]]; then
            local cur_size
            cur_size=$(stat -c%s "${output_file}" 2>/dev/null || echo 0)
            if [[ "$cur_size" -ne "$last_size" ]]; then
                last_size=$cur_size
                last_change_ts=$now
            fi
            if grep -q "tt-exalens ready" "${output_file}" 2>/dev/null; then
                exalens_ready=1
            fi
        fi

        local stall=$((now - last_change_ts))
        local hung=0
        local reason=""

        if [[ $exalens_ready -eq 0 ]] && grep -q "Waiting for tt-exalens" "${output_file}" 2>/dev/null; then
            if [[ $elapsed -ge $EXALENS_FAIL_SEC ]]; then
                hung=1
                reason="exalens not ready after ${elapsed}s"
            fi
        elif [[ $stall -ge $STALL_HANG_SEC ]]; then
            hung=1
            reason="no output change for ${stall}s"
        fi

        if [[ $hung -eq 1 ]]; then
            echo "[$(date '+%H:%M:%S')] HUNG (${reason}) - killing attempt" >&2
            kill -9 "$pytest_pid" 2>/dev/null || true
            cleanup_hanging
            if [[ $exalens_ready -eq 0 ]] && grep -q "Waiting for tt-exalens" "${output_file}" 2>/dev/null; then
                status="EXALENS_TIMEOUT"
            else
                status="STALL_HANG"
            fi
            break
        fi

        if [[ $((elapsed % 120)) -lt $POLL_SEC ]]; then
            echo "[$(date '+%H:%M:%S')] running... elapsed=${elapsed}s stall=${stall}s exalens_ready=${exalens_ready}" >&2
        fi
    done

    if [[ "$status" != "EXALENS_TIMEOUT" ]]; then
        wait "$pytest_pid" 2>/dev/null
        local exit_code=$?
        if [[ $exit_code -eq 0 ]]; then
            status="PASSED"
        else
            status="$(classify_failure "${output_file}")"
        fi
    fi

    cleanup_hanging
    restore_test_file "${test_file}"
    CURRENT_TEST_FILE=""

    echo "${status}"
}

run_one_test() {
    local test_file="$1"
    local output_file="$2"
    local test_num="$3"

    echo ""
    echo "================================================================"
    echo "[$(date '+%H:%M:%S')] Test ${test_num}: ${test_file} -> ${output_file} (L1_TO_L1 only)"
    echo "================================================================"

    local attempt=1
    local final_status="FAILED"
    while [[ $attempt -le $MAX_RETRIES ]]; do
        echo "[$(date '+%H:%M:%S')] Attempt ${attempt}/${MAX_RETRIES}"
        local status
        status=$(run_one_attempt "${test_file}" "${output_file}")

        if [[ "$status" == "PASSED" ]]; then
            final_status="PASSED"
            echo "[$(date '+%H:%M:%S')] test ${test_num} PASSED on attempt ${attempt}"
            break
        fi

        echo "[$(date '+%H:%M:%S')] test ${test_num} failed: ${status}"
        if [[ "$status" == "EXALENS_TIMEOUT" ]]; then
            echo "[$(date '+%H:%M:%S')] Retrying after exalens timeout..."
            attempt=$((attempt + 1))
            if [[ $attempt -le $MAX_RETRIES ]]; then
                sleep "$COOLDOWN_SEC"
            fi
            continue
        fi

        final_status="FAILED (${status})"
        echo "[$(date '+%H:%M:%S')] test ${test_num} failed during runtime - moving to next test"
        break
    done

    if [[ "$final_status" == "FAILED" && "$status" == "EXALENS_TIMEOUT" ]]; then
        final_status="FAILED (EXALENS_TIMEOUT, ${MAX_RETRIES} attempts)"
        echo "[$(date '+%H:%M:%S')] test ${test_num} gave up after ${MAX_RETRIES} exalens retries"
    fi

    echo "[$(date '+%H:%M:%S')] Waiting ${COOLDOWN_SEC}s before next test..."
    sleep "$COOLDOWN_SEC"

    echo "${test_num}|${test_file}|${output_file}|${final_status}|$(date '+%Y-%m-%d %H:%M:%S')"
}

TESTS=(
    "2|perf_eltwise_unary_datacopy_quasar.py|perf_output02.txt"
    "3|perf_eltwise_binary_broadcast_quasar.py|perf_output03.txt"
    "4|perf_eltwise_binary_quasar.py|perf_output04.txt"
    "5|perf_unpack_tilize_quasar.py|perf_output05.txt"
    "6|perf_unpack_unary_operand_quasar.py|perf_output06.txt"
    "7|perf_transpose_dest_quasar.py|perf_output07.txt"
    "8|perf_pack_quasar.py|perf_output08.txt"
    "9|perf_pack_untilize_quasar.py|perf_output09.txt"
    "10|perf_unary_broadcast_quasar.py|perf_output10.txt"
    "11|perf_pack_l1_acc_quasar.py|perf_output11.txt"
    "12|perf_reduce_quasar.py|perf_output12.txt"
    "13|perf_eltwise_binary_reuse_dest_quasar.py|perf_output13.txt"
    "14|perf_unpack_reduce_col_tilizeA_strided_quasar.py|perf_output14.txt"
)

SUMMARY_FILE="run_perf_summary_l1_rerun.txt"
LOG_FILE="run_perf_tests_l1_rerun.log"
START_FROM="${START_FROM:-1}"

{
    echo "Perf test run (L1_TO_L1 only, excluding matmul) started at $(date)"
    if [[ "$START_FROM" -gt 1 ]]; then
        echo "Resuming from test number ${START_FROM}"
    fi
    for entry in "${TESTS[@]}"; do
        IFS='|' read -r num test_file output_file <<< "$entry"
        if [[ "$num" -lt "$START_FROM" ]]; then
            echo "Skipping test ${num}: ${test_file} (already completed)"
            continue
        fi
        result=$(run_one_test "$test_file" "$output_file" "$num")
        echo "$result" >> "${SUMMARY_FILE}.partial"
        echo "$result"
    done
    echo ""
    echo "All tests complete. Summary in ${SUMMARY_FILE}"
} 2>&1 | tee "${LOG_FILE}"

mv -f "${SUMMARY_FILE}.partial" "${SUMMARY_FILE}" 2>/dev/null || true
