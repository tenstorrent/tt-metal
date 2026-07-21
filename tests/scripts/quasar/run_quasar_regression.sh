#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TT_METAL_HOME="$(cd "$SCRIPT_DIR/../../.." && pwd)"
export TT_METAL_RUNTIME_ROOT="$TT_METAL_HOME"

# Defaults
TESTS_FILE="$SCRIPT_DIR/quasar_regression_tests.yaml"
BUILD=false
FAST_DISPATCH=false
FILTER_CONFIG=""
FILTER_GROUP=""
BUILD_DIR="$TT_METAL_HOME/build"
LOG_DIR="$SCRIPT_DIR/logs"
DRY_RUN=false
BACK2BACK=true
# Retry gtest when ZeBu reports a hostname reservation conflict in the local
# emu_<date>_<time>_.log file (gtest hangs waiting for the device in this case).
# Set QUASAR_EMU_HOSTNAME_MAX_RETRIES=0 to disable retries (still fails fast).
EMU_HOSTNAME_MAX_RETRIES="${QUASAR_EMU_HOSTNAME_MAX_RETRIES:-2}"
if [[ ! "$EMU_HOSTNAME_MAX_RETRIES" =~ ^[0-9]+$ ]]; then
    echo "ERROR: QUASAR_EMU_HOSTNAME_MAX_RETRIES must be a non-negative integer"
    exit 1
fi
EMU_HOSTNAME_RETRY_DELAY="${QUASAR_EMU_HOSTNAME_RETRY_DELAY:-30}"
EMU_MONITOR_POLL_INTERVAL="${QUASAR_EMU_HOSTNAME_POLL_INTERVAL:-2}"
EMU_LOG_GLOB='emu_*_.log'
EMU_HOSTNAME_CONFLICT_REGEX='zServer : ERROR.*Hostname = .* is used by'
SIMULATOR_PID_REGEX='Simulator process spawned with PID: ([0-9]+)'

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Run Quasar emulator regression tests defined in a YAML file, batching entries
marked back2back: true into single gtest invocations when they share the same
binary, config, env, and gtest_repeat settings.

Required environment variables:
  TT_METAL_SIMULATOR_BASE   Base path containing simulator build directories
                           (e.g. the parent of emu-quasar-1x3/, emu-quasar-2x3/,
                           emu-quasar-2x3_DISPATCH/)
                           The script sets TT_METAL_SIMULATOR per test automatically.
                           If TT_METAL_SIMULATOR is already set, the base is
                           derived automatically (one directory up).
  NNG_SOCKET_ADDR          NNG socket address
  NNG_SOCKET_LOCAL_PORT    NNG local port

Options:
  --build                          Run build_metal.sh --build-tests before testing
  --config <1x3|2x3|2x3_DISPATCH>  Only run tests for the specified configuration
  --group <name>                   Only run tests from the specified test group
  --tests <path>                   Path to YAML test file (default: quasar_regression_tests.yaml)
  --build-dir <path>               Path to build directory (default: $BUILD_DIR)
  --log-dir <path>                 Save per-test gtest JSON results to this directory
  --no-back2back                   Run each test in a separate gtest process
  --fast-dispatch                  Do not set TT_METAL_SLOW_DISPATCH_MODE by default (fast dispatch).
                                   Per-test env in the YAML file can still set TT_METAL_SLOW_DISPATCH_MODE.
  --dry-run                        Print commands without executing
  -h, --help                       Show this help message

Environment (optional):
  QUASAR_EMU_HOSTNAME_MAX_RETRIES   Retries after ZeBu hostname conflict (default: 2)
  QUASAR_EMU_HOSTNAME_RETRY_DELAY   Seconds to wait between retries (default: 30)
  QUASAR_EMU_HOSTNAME_POLL_INTERVAL Seconds between emu log polls (default: 2)
EOF
    exit 0
}

need_arg() { if [[ $# -lt 2 || "$2" == --* ]]; then echo "ERROR: $1 requires an argument"; exit 1; fi; }
while [[ $# -gt 0 ]]; do
    case "$1" in
        --build)       BUILD=true; shift ;;
        --config)      need_arg "$1" "${2:-}"; FILTER_CONFIG="$2"; shift 2 ;;
        --group)       need_arg "$1" "${2:-}"; FILTER_GROUP="$2"; shift 2 ;;
        --tests)       need_arg "$1" "${2:-}"; TESTS_FILE="$2"; shift 2 ;;
        --build-dir)   need_arg "$1" "${2:-}"; BUILD_DIR="$2"; shift 2 ;;
        --log-dir)     need_arg "$1" "${2:-}"; LOG_DIR="$2"; shift 2 ;;
        --no-back2back) BACK2BACK=false; shift ;;
        --fast-dispatch) FAST_DISPATCH=true; shift ;;
        --dry-run)     DRY_RUN=true; shift ;;
        -h|--help)     usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

if [[ "$FAST_DISPATCH" == true ]]; then
    unset TT_METAL_SLOW_DISPATCH_MODE
else
    export TT_METAL_SLOW_DISPATCH_MODE="${TT_METAL_SLOW_DISPATCH_MODE:-1}"
fi

if [[ -n "$FILTER_CONFIG" && "$FILTER_CONFIG" != "1x3" && "$FILTER_CONFIG" != "2x3" && "$FILTER_CONFIG" != "2x3_DISPATCH" ]]; then
    echo "ERROR: invalid --config value '$FILTER_CONFIG'. Supported: 1x3, 2x3, 2x3_DISPATCH"
    exit 1
fi

# Derive SIMULATOR_BASE from TT_METAL_SIMULATOR if available
if [[ -z "${TT_METAL_SIMULATOR_BASE:-}" && -n "${TT_METAL_SIMULATOR:-}" ]]; then
    TT_METAL_SIMULATOR_BASE="$(cd "$(dirname "${TT_METAL_SIMULATOR%/}")" && pwd)"
    echo "Derived TT_METAL_SIMULATOR_BASE=$TT_METAL_SIMULATOR_BASE from TT_METAL_SIMULATOR=$TT_METAL_SIMULATOR"
fi

# Validate required environment variables
missing_vars=()
[[ -z "${TT_METAL_SIMULATOR_BASE:-}" ]] && missing_vars+=("TT_METAL_SIMULATOR_BASE (or TT_METAL_SIMULATOR)")
[[ -z "${NNG_SOCKET_ADDR:-}" ]]       && missing_vars+=("NNG_SOCKET_ADDR")
[[ -z "${NNG_SOCKET_LOCAL_PORT:-}" ]] && missing_vars+=("NNG_SOCKET_LOCAL_PORT")

if [[ ${#missing_vars[@]} -gt 0 ]]; then
    echo "ERROR: Required environment variables not set:"
    for v in "${missing_vars[@]}"; do
        echo "  $v"
    done
    echo ""
    echo "Example:"
    echo "  export TT_METAL_SIMULATOR_BASE=/path/to/simulators/build"
    echo "  export NNG_SOCKET_ADDR=tcp://hostname:port"
    echo "  export NNG_SOCKET_LOCAL_PORT=5555"
    exit 1
fi

SIMULATOR_BASE="$TT_METAL_SIMULATOR_BASE"

if [[ ! -f "$TESTS_FILE" ]]; then
    echo "ERROR: Tests file not found: $TESTS_FILE"
    exit 1
fi

if ! command -v yq &>/dev/null; then
    echo "ERROR: yq is required but not found. Install from https://github.com/mikefarah/yq"
    exit 1
fi
if ! yq --version 2>&1 | grep -qi "mikefarah\|version v4"; then
    echo "WARNING: This script requires Mike Farah's yq (v4+)."
    echo "         Detected: $(yq --version 2>&1)"
    echo "         Install from https://github.com/mikefarah/yq"
    exit 1
fi

simulator_path_for_config() {
    local config="$1"
    echo "$SIMULATOR_BASE/emu-quasar-${config}"
}

# Build if requested
if [[ "$BUILD" == true ]]; then
    echo "=== Building tests ==="
    (cd "$TT_METAL_HOME" && ./build_metal.sh -c --build-tests)
    echo ""
fi

if [[ -n "$LOG_DIR" ]]; then
    LOG_DIR="$LOG_DIR/$(date +%Y-%m-%d_%H%M%S)"
    mkdir -p "$LOG_DIR"
    LOG_DIR="$(cd "$LOG_DIR" && pwd)"
    echo "Logs:   $LOG_DIR"
fi

passed=0
failed=0
skipped=0
declare -a results=()

# Load tests from YAML into an array (single-pass read via yq)
VALID_CONFIGS=("1x3" "2x3" "2x3_DISPATCH")

is_valid_config() {
    local cfg="$1"
    for valid in "${VALID_CONFIGS[@]}"; do
        [[ "$cfg" == "$valid" ]] && return 0
    done
    return 1
}

SEP=$'\x1f'   # field separator for test entry records
ENV_SEP='|'  # separator between env KEY=value pairs (yq join() does not interpret \u escapes)
EMPTY_ENV='-'  # TSV placeholder for empty env (bash read collapses consecutive tab fields)
EMPTY_GTEST_REPEAT='-'  # TSV placeholder for empty gtest_repeat (same issue as env)
declare -a test_entries=()
while IFS=$'\t' read -r group filter configs envvars gtest_repeat back2back; do
    [[ -z "$group" ]] && continue
    [[ "$gtest_repeat" == "$EMPTY_GTEST_REPEAT" ]] && gtest_repeat=""
    # Split comma-separated configs and create one entry per config
    IFS=',' read -ra config_list <<< "$configs"
    for config in "${config_list[@]}"; do
        config="$(echo "$config" | xargs)"  # trim whitespace
        if ! is_valid_config "$config"; then
            echo "ERROR: invalid config '$config' for test '$filter' in group '$group'"
            echo "       Supported configs: ${VALID_CONFIGS[*]}"
            exit 1
        fi
        test_entries+=("${group}${SEP}${filter}${SEP}${config}${SEP}${envvars}${SEP}${gtest_repeat}${SEP}${back2back}")
    done
done < <(yq -r 'to_entries[] | .key as $group | .value[] | [$group, .filter, .config, ((.env // {} | to_entries | map(.key + "=" + (.value | tostring)) | join("|") | sub("^$"; "-"))), ((.gtest_repeat // "" | tostring) | sub("^$"; "-")), (.back2back // false | tostring)] | @tsv' "$TESTS_FILE")

total_tests=${#test_entries[@]}

echo "=== Quasar Back-to-Back Tests ==="
echo "File:   $TESTS_FILE"
echo "Tests:  $total_tests total"
if [[ "$BACK2BACK" == true ]]; then
    echo "Mode:   back2back (batch entries with back2back: true)"
else
    echo "Mode:   sequential (--no-back2back)"
fi
[[ -n "$FILTER_CONFIG" ]] && echo "Filter: config=$FILTER_CONFIG"
[[ -n "$FILTER_GROUP" ]]  && echo "Filter: group=$FILTER_GROUP"
if [[ "$EMU_HOSTNAME_MAX_RETRIES" -gt 0 ]]; then
    echo "Emu:    monitor emu_<date>_<time>_.log, retry on hostname conflict (max ${EMU_HOSTNAME_MAX_RETRIES} retries, ${EMU_HOSTNAME_RETRY_DELAY}s delay)"
else
    echo "Emu:    monitor emu_<date>_<time>_.log, fail fast on hostname conflict"
fi
echo ""

fmt_duration() {
    local secs="$1"
    if [[ "$secs" -ge 60 ]]; then
        printf "%dm%02ds" $((secs / 60)) $((secs % 60))
    else
        printf "%ds" $secs
    fi
}

sanitize_name() { echo "$1" | tr -cs 'A-Za-z0-9_.-' '_' | sed 's/^_//;s/_$//'; }

declare -a emu_logs_baseline=()
declare -a simulator_baseline_pids=()

refresh_emu_logs_baseline() {
    emu_logs_baseline=()
    local f
    shopt -s nullglob
    for f in $EMU_LOG_GLOB; do
        emu_logs_baseline+=("$f")
    done
    shopt -u nullglob
}

emu_log_is_new() {
    local candidate="$1" known
    for known in "${emu_logs_baseline[@]}"; do
        [[ "$candidate" == "$known" ]] && return 1
    done
    return 0
}

find_newest_emu_log_since_baseline() {
    local newest="" f mtime newest_mtime=0
    shopt -s nullglob
    for f in $EMU_LOG_GLOB; do
        if emu_log_is_new "$f"; then
            mtime=$(stat -c %Y "$f" 2>/dev/null || echo 0)
            if [[ $mtime -ge $newest_mtime ]]; then
                newest_mtime=$mtime
                newest="$f"
            fi
        fi
    done
    shopt -u nullglob
    echo "$newest"
}

# Read new lines from an emu log; print matches and return 0 on hostname conflict.
poll_emu_log_for_hostname_conflict() {
    local logfile="$1" offset_var="$2"
    local offset="${!offset_var}"
    local line new_size
    local -a matches=()

    [[ -n "$logfile" && -f "$logfile" ]] || return 1

    new_size=$(stat -c %s "$logfile" 2>/dev/null || echo 0)
    if [[ $new_size -le $offset ]]; then
        return 1
    fi

    while IFS= read -r line || [[ -n "$line" ]]; do
        if [[ "$line" =~ $EMU_HOSTNAME_CONFLICT_REGEX ]]; then
            matches+=("$line")
        fi
    done < <(tail -c +$((offset + 1)) "$logfile" 2>/dev/null || true)

    printf -v "$offset_var" '%s' "$new_size"

    if [[ ${#matches[@]} -gt 0 ]]; then
        printf '%s\n' "${matches[@]}"
        return 0
    fi
    return 1
}

# Read new lines from TT_METAL_LOGGER_FILE for the detached simulator PID.
poll_metal_log_for_simulator_pid() {
    local logfile="$1" offset_var="$2" pid_var="$3"
    local offset="${!offset_var}" line new_size

    [[ -n "$logfile" && -f "$logfile" ]] || return 1

    new_size=$(stat -c %s "$logfile" 2>/dev/null || echo 0)
    if [[ $new_size -le $offset ]]; then
        return 1
    fi

    while IFS= read -r line || [[ -n "$line" ]]; do
        if [[ "$line" =~ $SIMULATOR_PID_REGEX ]]; then
            printf -v "$pid_var" '%s' "${BASH_REMATCH[1]}"
        fi
    done < <(tail -c +$((offset + 1)) "$logfile" 2>/dev/null || true)

    printf -v "$offset_var" '%s' "$new_size"
    [[ -n "${!pid_var}" ]]
}

snapshot_simulator_pids() {
    simulator_baseline_pids=()
    local pid
    while IFS= read -r pid; do
        [[ -n "$pid" ]] && simulator_baseline_pids+=("$pid")
    done < <(find_simulator_pids_for_path "$TT_METAL_SIMULATOR")
}

find_simulator_pids_for_path() {
    local sim_path="${1%/}"
    if [[ -z "$sim_path" || ! -d "$sim_path" ]]; then
        return
    fi
    if command -v pgrep &>/dev/null; then
        pgrep -f "${sim_path}/run\\.sh" 2>/dev/null || true
    fi
}

simulator_pid_is_new() {
    local candidate="$1" known
    for known in "${simulator_baseline_pids[@]}"; do
        [[ "$candidate" == "$known" ]] && return 1
    done
    return 0
}

collect_new_simulator_pids() {
    local -a pids=() pid
    while IFS= read -r pid; do
        [[ -z "$pid" ]] && continue
        if simulator_pid_is_new "$pid"; then
            pids+=("$pid")
        fi
    done < <(find_simulator_pids_for_path "$TT_METAL_SIMULATOR")
    printf '%s\n' "${pids[@]}"
}

wait_for_process_exit() {
    local pid="$1" timeout="${2:-10}" i
    [[ -z "$pid" ]] && return 0
    for ((i = 0; i < timeout; i++)); do
        kill -0 "$pid" 2>/dev/null || return 0
        sleep 1
    done
    return 1
}

terminate_process_tree() {
    local pid="$1" child
    [[ -z "$pid" || ! "$pid" =~ ^[0-9]+$ ]] && return

    if command -v pgrep &>/dev/null; then
        while IFS= read -r child; do
            [[ -n "$child" && "$child" != "$pid" ]] && terminate_process_tree "$child"
        done < <(pgrep -P "$pid" 2>/dev/null || true)
    fi

    kill -TERM "$pid" 2>/dev/null || true
    if ! wait_for_process_exit "$pid" 5; then
        kill -KILL "$pid" 2>/dev/null || true
        wait_for_process_exit "$pid" 3 || true
    fi
}

terminate_gtest_process() {
    local pid="$1"
    kill -TERM "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    if ! wait_for_process_exit "$pid" 10; then
        kill -KILL "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
        wait_for_process_exit "$pid" 5 || true
    fi
}

# The simulator is spawned detached (UV_PROCESS_DETACHED), so killing gtest alone
# leaves run.sh running and a retry would launch a second emulator.
cleanup_emu_attempt() {
    local gtest_pid="$1" tracked_sim_pid="${2:-}"
    local -a sim_pids=() pid

    terminate_gtest_process "$gtest_pid"
    wait "$gtest_pid" 2>/dev/null || true

    if [[ -n "$tracked_sim_pid" ]]; then
        sim_pids+=("$tracked_sim_pid")
    fi
    while IFS= read -r pid; do
        [[ -z "$pid" ]] && continue
        local seen=0
        for known in "${sim_pids[@]}"; do
            [[ "$pid" == "$known" ]] && seen=1 && break
        done
        [[ $seen -eq 0 ]] && sim_pids+=("$pid")
    done < <(collect_new_simulator_pids)

    if [[ ${#sim_pids[@]} -gt 0 ]]; then
        echo "  terminating orphaned simulator process(es): ${sim_pids[*]}"
        for pid in "${sim_pids[@]}"; do
            terminate_process_tree "$pid"
        done
    fi

    local i
    for ((i = 0; i < 15; i++)); do
        [[ -z "$(collect_new_simulator_pids)" ]] && break
        sleep 1
    done

    refresh_emu_logs_baseline
}

# Run gtest in the background and poll emu_<date>_<time>_.log for hostname conflicts.
# Sets gtest_emu_hostname_conflict=1 when the emulator log reports a conflict.
# Returns the gtest exit code (1 when terminated due to conflict).
# Tracks active_gtest_pid / active_tracked_sim_pid for INT cleanup (setsid isolates
# the gtest process group from the terminal's Ctrl-C).
run_gtest_with_emu_monitor() {
    local gtest_pid rc=0
    local emu_log="" emu_offset=0 metal_offset=0 conflict_lines=""
    local tracked_sim_pid=""

    gtest_emu_hostname_conflict=0
    gtest_emu_conflict_lines=""
    refresh_emu_logs_baseline
    snapshot_simulator_pids

    if command -v setsid &>/dev/null; then
        setsid "${gtest_cmd_env[@]}" "$binary" --gtest_filter="$combined_filter" "${gtest_repeat_args[@]}" "${gtest_log_args[@]}" &
    else
        "${gtest_cmd_env[@]}" "$binary" --gtest_filter="$combined_filter" "${gtest_repeat_args[@]}" "${gtest_log_args[@]}" &
    fi
    gtest_pid=$!
    active_gtest_pid=$gtest_pid
    active_tracked_sim_pid=""

    while kill -0 "$gtest_pid" 2>/dev/null; do
        if [[ -n "$log_file" ]]; then
            poll_metal_log_for_simulator_pid "$log_file" metal_offset tracked_sim_pid || true
            active_tracked_sim_pid="$tracked_sim_pid"
        fi

        if [[ -z "$emu_log" ]]; then
            emu_log="$(find_newest_emu_log_since_baseline)"
            emu_offset=0
        fi

        if [[ -n "$emu_log" ]]; then
            if conflict_lines="$(poll_emu_log_for_hostname_conflict "$emu_log" emu_offset)"; then
                gtest_emu_hostname_conflict=1
                gtest_emu_conflict_lines="$conflict_lines"
                echo "  EMU HOSTNAME CONFLICT in $emu_log (terminating hung gtest)"
                cleanup_emu_attempt "$gtest_pid" "$tracked_sim_pid"
                active_gtest_pid=""
                active_tracked_sim_pid=""
                break
            fi
        fi

        sleep "$EMU_MONITOR_POLL_INTERVAL"
    done

    if [[ $gtest_emu_hostname_conflict -eq 0 ]]; then
        wait "$gtest_pid" || rc=$?
    else
        rc=1
    fi
    active_gtest_pid=""
    active_tracked_sim_pid=""
    return $rc
}

# Parse gtest --gtest_output=json and record one summary line per test case.
record_gtest_result() {
    local config="$1" group="$2" label="$3" elapsed="$4" rc="$5" json_file="$6"
    shift 6
    local filters=("$@")

    local inv_passed=0 inv_failed=0 inv_skipped=0

    record_one_result() {
        local status="$1" test_label="$2" test_time="$3"
        case "$status" in
            PASS)
                passed=$((passed + 1))
                inv_passed=$((inv_passed + 1))
                results+=("PASS  $test_label  ($test_time)")
                ;;
            FAIL)
                failed=$((failed + 1))
                inv_failed=$((inv_failed + 1))
                results+=("FAIL  $test_label  ($test_time)")
                ;;
            SKIP)
                skipped=$((skipped + 1))
                inv_skipped=$((inv_skipped + 1))
                results+=("SKIP  $test_label  ($test_time)")
                ;;
        esac
    }

    if [[ -f "$json_file" ]]; then
        local count=0
        while IFS=$'\t' read -r classname testname result test_time has_failures; do
            [[ -z "$testname" ]] && continue
            local status
            if [[ "$result" == "SKIPPED" ]]; then
                status=SKIP
            elif [[ "$has_failures" == "true" ]]; then
                status=FAIL
            elif [[ "$result" == "COMPLETED" ]]; then
                status=PASS
            else
                status=FAIL
            fi
            record_one_result "$status" "[$config] $group ${classname}.${testname}" "$test_time"
            count=$((count + 1))
        done < <(yq -r '.testsuites[].testsuite[]? | [.classname, .name, .result, (.time // ""), ((.failures // []) | length > 0 | tostring)] | @tsv' "$json_file")

        if [[ $count -eq 0 ]]; then
            local no_match_status=SKIP
            [[ $rc -ne 0 ]] && no_match_status=FAIL
            if [[ ${#filters[@]} -gt 0 ]]; then
                for filter in "${filters[@]}"; do
                    record_one_result "$no_match_status" "[$config] $group --gtest_filter=$filter" "$(fmt_duration "$elapsed"), no tests ran"
                done
            else
                record_one_result "$no_match_status" "$label" "$(fmt_duration "$elapsed"), no tests ran"
            fi
        fi
    else
        local status=PASS
        [[ $rc -ne 0 ]] && status=FAIL
        if [[ ${#filters[@]} -gt 0 ]]; then
            for filter in "${filters[@]}"; do
                record_one_result "$status" "[$config] $group --gtest_filter=$filter" "$(fmt_duration "$elapsed")"
            done
        else
            record_one_result "$status" "$label" "$(fmt_duration "$elapsed")"
        fi
    fi

    if [[ $rc -ne 0 && $inv_failed -eq 0 ]]; then
        record_one_result FAIL "$label" "$(fmt_duration "$elapsed"), process exited with status $rc"
    fi
    if [[ $inv_failed -gt 0 ]]; then
        echo "  RESULT: ${inv_passed} passed, ${inv_failed} failed, ${inv_skipped} skipped"
    elif [[ $inv_skipped -gt 0 && $inv_passed -eq 0 ]]; then
        echo "  RESULT: ${inv_skipped} skipped"
    else
        echo "  RESULT: ${inv_passed} passed, ${inv_skipped} skipped"
    fi
}

print_summary() {
    echo ""
    echo "========================================="
    if [[ "$BACK2BACK" == true ]]; then
        echo "  BACK-TO-BACK REGRESSION SUMMARY"
    else
        echo "  REGRESSION SUMMARY"
    fi
    echo "========================================="
    if [[ ${#results[@]} -gt 0 ]]; then
        for r in "${results[@]}"; do
            echo "  $r"
        done
    fi
    echo "-----------------------------------------"
    echo "  PASSED:  $passed"
    echo "  FAILED:  $failed"
    echo "  SKIPPED: $skipped"
    local total_elapsed=$((SECONDS - run_start))
    echo "  ELAPSED: $(fmt_duration $total_elapsed)"
    echo "========================================="
}

batch_key() {
    local group="$1" config="$2" envvars="$3" gtest_repeat="$4"
    echo "${group}${SEP}${config}${SEP}${envvars}${SEP}${gtest_repeat}"
}

apply_env_vars() {
    local envvars="$1"
    extra_env_keys=()
    extra_env_pairs=()
    [[ "$envvars" == "$EMPTY_ENV" ]] && envvars=""
    if [[ -n "$envvars" ]]; then
        while IFS= read -r -d "$ENV_SEP" untrimmed_pair || [[ -n "$untrimmed_pair" ]]; do
            pair="${untrimmed_pair%"${untrimmed_pair##*[![:space:]]}"}"
            [[ -z "$pair" ]] && continue
            key="${pair%%=*}"
            extra_env_keys+=("$key")
            extra_env_pairs+=("$pair")
        done <<< "$envvars"
    fi
}

build_gtest_cmd_env() {
    local log_file="${1:-}"
    local has_slow_dispatch_override=false
    for key in "${extra_env_keys[@]}"; do
        if [[ "$key" == "TT_METAL_SLOW_DISPATCH_MODE" ]]; then
            has_slow_dispatch_override=true
            break
        fi
    done

    gtest_cmd_env=(env)
    if [[ "$FAST_DISPATCH" == true && "$has_slow_dispatch_override" == false ]]; then
        gtest_cmd_env+=(-u TT_METAL_SLOW_DISPATCH_MODE)
    fi
    if [[ -n "$log_file" ]]; then
        gtest_cmd_env+=("TT_METAL_LOGGER_FILE=${log_file}")
    fi
    for pair in "${extra_env_pairs[@]}"; do
        gtest_cmd_env+=("$pair")
    done
}

print_run_env() {
    echo "  TT_METAL_SIMULATOR=$TT_METAL_SIMULATOR"
    echo "  NNG_SOCKET_ADDR=$NNG_SOCKET_ADDR"
    echo "  NNG_SOCKET_LOCAL_PORT=$NNG_SOCKET_LOCAL_PORT"
    if [[ -n "${TT_METAL_SLOW_DISPATCH_MODE+x}" ]]; then
        echo "  TT_METAL_SLOW_DISPATCH_MODE=$TT_METAL_SLOW_DISPATCH_MODE"
    fi
    for pair in "${extra_env_pairs[@]}"; do
        echo "  ENV: $pair"
    done
}

combine_gtest_filters() {
    local combined=""
    for filter in "$@"; do
        if [[ -z "$combined" ]]; then
            combined="$filter"
        else
            combined="${combined}:${filter}"
        fi
    done
    echo "$combined"
}

run_gtest_invocation() {
    local group="$1" config="$2" envvars="$3" gtest_repeat="$4" label="$5"
    shift 5
    local filters=("$@")

    local combined_filter
    combined_filter="$(combine_gtest_filters "${filters[@]}")"

    local sim_path binary
    sim_path="$(simulator_path_for_config "$config")"
    binary="$BUILD_DIR/test/tt_metal/$group"

    local gtest_repeat_args=()
    if [[ -n "$gtest_repeat" ]]; then
        gtest_repeat_args+=("--gtest_repeat=$gtest_repeat")
    fi

    echo "--- [$run_num] $label ---"

    if [[ ! -f "$binary" ]]; then
        echo "  SKIP: binary not found: $binary"
        skipped=$((skipped + 1))
        results+=("SKIP  $label  (binary not found)")
        return
    fi

    if [[ ! -d "$sim_path" ]]; then
        echo "  SKIP: simulator path not found: $sim_path"
        skipped=$((skipped + 1))
        results+=("SKIP  $label  (simulator not found)")
        return
    fi

    export TT_METAL_SIMULATOR="$sim_path/"
    apply_env_vars "$envvars"
    print_run_env
    echo "  CMD: $binary --gtest_filter=$combined_filter${gtest_repeat_args[*]:+ ${gtest_repeat_args[*]}}"

    if [[ "$DRY_RUN" == true ]]; then
        results+=("DRY   $label")
        return
    fi

    local gtest_log_args=()
    local json_file=""
    local log_file=""
    local log_stem
    if [[ ${#filters[@]} -eq 1 ]]; then
        log_stem="$(sanitize_name "${filters[0]}")"
    else
        log_stem="back2back_${run_num}"
    fi
    local max_attempts=$((1 + EMU_HOSTNAME_MAX_RETRIES))
    local attempt=1
    local rc=0
    local test_start=$SECONDS

    while [[ $attempt -le $max_attempts ]]; do
        if [[ $attempt -gt 1 ]]; then
            echo "  RETRY: attempt $attempt/$max_attempts after emulator hostname conflict"
        fi

        gtest_log_args=()
        json_file=""
        log_file=""
        if [[ -n "$LOG_DIR" ]]; then
            local log_base="$LOG_DIR/${config}_${group}_${log_stem}"
            local count="${log_base_counts["$log_base"]-0}"
            count=$((count + 1))
            log_base_counts["$log_base"]=$count
            if [[ $count -gt 1 ]]; then
                log_base="${log_base}_${count}"
            fi
            json_file="${log_base}.json"
            log_file="${log_base}.log"
            gtest_log_args=("--gtest_output=json:${json_file}")
            echo "  TT_METAL_LOGGER_FILE=${log_file}"
        fi
        build_gtest_cmd_env "$log_file"

        rc=0
        gtest_emu_hostname_conflict=0
        run_gtest_with_emu_monitor || rc=$?

        if [[ $gtest_emu_hostname_conflict -eq 1 ]]; then
            if [[ $attempt -lt $max_attempts ]]; then
                echo "  EMU HOSTNAME CONFLICT (retrying in ${EMU_HOSTNAME_RETRY_DELAY}s):"
                while IFS= read -r line || [[ -n "$line" ]]; do
                    [[ -z "$line" ]] && continue
                    echo "    $line"
                done <<< "$gtest_emu_conflict_lines"
                sleep "$EMU_HOSTNAME_RETRY_DELAY"
                attempt=$((attempt + 1))
                continue
            fi
            echo "  EMU HOSTNAME CONFLICT (no retries remaining):"
            while IFS= read -r line || [[ -n "$line" ]]; do
                [[ -z "$line" ]] && continue
                echo "    $line"
            done <<< "$gtest_emu_conflict_lines"
            rc=1
        fi
        break
    done

    local elapsed=$((SECONDS - test_start))
    record_gtest_result "$config" "$group" "$label" "$elapsed" "$rc" "$json_file" "${filters[@]}"
    if [[ $rc -ne 0 && -n "$log_file" ]]; then
        echo "  LOG: $log_file"
    fi

    echo ""
}

clear_back2back_batch() {
    batch_filters=()
    batch_group=""
    batch_config=""
    batch_envvars=""
    batch_gtest_repeat=""
    batch_key_current=""
}

flush_back2back_batch() {
    [[ ${#batch_filters[@]} -eq 0 ]] && return

    run_num=$((run_num + 1))
    local filter_summary="${batch_filters[0]}"
    if [[ ${#batch_filters[@]} -gt 1 ]]; then
        filter_summary="back2back(${#batch_filters[@]}): $(combine_gtest_filters "${batch_filters[@]}")"
    fi

    local label="[$batch_config] $batch_group --gtest_filter=$filter_summary"
    if [[ -n "$batch_gtest_repeat" ]]; then
        label="$label --gtest_repeat=$batch_gtest_repeat"
    fi

    # Capture and clear before invoking so an INT mid-run cannot re-enter with
    # the same filters and launch a duplicate gtest/emulator.
    local -a filters_to_run=("${batch_filters[@]}")
    local group="$batch_group" config="$batch_config"
    local envvars="$batch_envvars" gtest_repeat="$batch_gtest_repeat"
    clear_back2back_batch

    run_gtest_invocation "$group" "$config" "$envvars" "$gtest_repeat" "$label" "${filters_to_run[@]}"
}

# Ctrl-C must not flush/re-run a batch: batch_filters used to remain set until
# run_gtest_invocation returned, so the INT trap re-entered flush and started
# duplicate work. gtest is also launched under setsid, so the terminal signal
# never reaches it — explicit cleanup is required to avoid leaking the detached
# simulator (UV_PROCESS_DETACHED).
on_interrupt() {
    trap - INT
    echo ""
    echo "*** Interrupted ***"

    clear_back2back_batch

    if [[ -n "${active_gtest_pid:-}" ]]; then
        echo "  cleaning up active gtest (pid $active_gtest_pid) and simulator..."
        cleanup_emu_attempt "$active_gtest_pid" "${active_tracked_sim_pid:-}"
        active_gtest_pid=""
        active_tracked_sim_pid=""
    fi

    print_summary
    exit 130
}

run_start=$SECONDS
trap on_interrupt INT
run_num=0

declare -A log_base_counts=()
declare -a extra_env_keys=()
declare -a extra_env_pairs=()
declare -a gtest_cmd_env=()
gtest_emu_hostname_conflict=0
gtest_emu_conflict_lines=""
active_gtest_pid=""
active_tracked_sim_pid=""

# Back-to-back batch state
declare -a batch_filters=()
batch_group=""
batch_config=""
batch_envvars=""
batch_gtest_repeat=""
batch_key_current=""

for entry in "${test_entries[@]}"; do
    IFS="$SEP" read -r group filter config envvars gtest_repeat back2back <<< "$entry"

    if [[ -n "$FILTER_CONFIG" && "$config" != "$FILTER_CONFIG" ]]; then
        skipped=$((skipped + 1))
        continue
    fi
    if [[ -n "$FILTER_GROUP" && "$group" != "$FILTER_GROUP" ]]; then
        skipped=$((skipped + 1))
        continue
    fi

    if [[ "$BACK2BACK" == true && "$back2back" == "true" ]]; then
        local_key="$(batch_key "$group" "$config" "$envvars" "$gtest_repeat")"
        if [[ -n "$batch_key_current" && "$local_key" != "$batch_key_current" ]]; then
            flush_back2back_batch
        fi
        batch_key_current="$local_key"
        batch_group="$group"
        batch_config="$config"
        batch_envvars="$envvars"
        batch_gtest_repeat="$gtest_repeat"
        batch_filters+=("$filter")
    else
        flush_back2back_batch
        run_num=$((run_num + 1))
        label="[$config] $group --gtest_filter=$filter"
        if [[ -n "$gtest_repeat" ]]; then
            label="$label --gtest_repeat=$gtest_repeat"
        fi
        run_gtest_invocation "$group" "$config" "$envvars" "$gtest_repeat" "$label" "$filter"
    fi
done

flush_back2back_batch

print_summary

if [[ $failed -gt 0 ]]; then
    exit 1
fi
