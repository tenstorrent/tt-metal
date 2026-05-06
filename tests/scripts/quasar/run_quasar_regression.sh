#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TT_METAL_HOME="$(cd "$SCRIPT_DIR/../../.." && pwd)"
export TT_METAL_RUNTIME_ROOT="$TT_METAL_HOME"
export TT_METAL_SLOW_DISPATCH_MODE="${TT_METAL_SLOW_DISPATCH_MODE:-1}"

# Defaults
TESTS_FILE="$SCRIPT_DIR/quasar_regression_tests.yaml"
BUILD=false
FILTER_CONFIG=""
FILTER_GROUP=""
BUILD_DIR="$TT_METAL_HOME/build"
LOG_DIR="$SCRIPT_DIR/logs"
DRY_RUN=false

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Run Quasar emulator regression tests defined in a YAML file.

Required environment variables:
  TT_METAL_SIMULATOR_BASE   Base path containing simulator build directories
                           (e.g. the parent of emu-quasar-1x3/, emu-quasar-2x3/)
                           The script sets TT_METAL_SIMULATOR per test automatically.
                           If TT_METAL_SIMULATOR is already set, the base is
                           derived automatically (one directory up).
  NNG_SOCKET_ADDR          NNG socket address
  NNG_SOCKET_LOCAL_PORT    NNG local port

Options:
  --build                 Run build_metal.sh --build-tests before testing
  --config <1x3|2x3>      Only run tests for the specified configuration
  --group <name>          Only run tests from the specified test group
  --tests <path>          Path to YAML test file (default: quasar_regression_tests.yaml)
  --build-dir <path>      Path to build directory (default: $BUILD_DIR)
  --log-dir <path>        Save per-test gtest JSON results to this directory
  --dry-run               Print commands without executing
  -h, --help              Show this help message
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
        --dry-run)     DRY_RUN=true; shift ;;
        -h|--help)     usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

if [[ -n "$FILTER_CONFIG" && "$FILTER_CONFIG" != "1x3" && "$FILTER_CONFIG" != "2x3" ]]; then
    echo "ERROR: invalid --config value '$FILTER_CONFIG'. Supported: 1x3, 2x3"
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
VALID_CONFIGS=("1x3" "2x3")

is_valid_config() {
    local cfg="$1"
    for valid in "${VALID_CONFIGS[@]}"; do
        [[ "$cfg" == "$valid" ]] && return 0
    done
    return 1
}

SEP=$'\x1f'
declare -a test_entries=()
while IFS=$'\t' read -r group filter config envvars; do
    [[ -z "$group" ]] && continue
    if ! is_valid_config "$config"; then
        echo "ERROR: invalid config '$config' for test '$filter' in group '$group'"
        echo "       Supported configs: ${VALID_CONFIGS[*]}"
        exit 1
    fi
    test_entries+=("${group}${SEP}${filter}${SEP}${config}${SEP}${envvars}")
done < <(yq -r 'to_entries[] | .key as $group | .value[] | [$group, .filter, .config, (.env // {} | to_entries | map(.key + "=" + (.value | tostring)) | join("\u001f"))] | @tsv' "$TESTS_FILE")

total_tests=${#test_entries[@]}

echo "=== Quasar Tests ==="
echo "File:   $TESTS_FILE"
echo "Tests:  $total_tests total"
[[ -n "$FILTER_CONFIG" ]] && echo "Filter: config=$FILTER_CONFIG"
[[ -n "$FILTER_GROUP" ]]  && echo "Filter: group=$FILTER_GROUP"
echo ""

fmt_duration() {
    local secs="$1"
    if [[ "$secs" -ge 60 ]]; then
        printf "%dm%02ds" $((secs / 60)) $((secs % 60))
    else
        printf "%ds" "$secs"
    fi
}

sanitize_name() { echo "$1" | tr -cs 'A-Za-z0-9_.-' '_' | sed 's/^_//;s/_$//'; }

print_summary() {
    echo ""
    echo "========================================="
    echo "  REGRESSION SUMMARY"
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

run_start=$SECONDS
trap 'echo ""; echo "*** Interrupted ***"; print_summary; exit 130' INT
test_num=0

for entry in "${test_entries[@]}"; do
    IFS="$SEP" read -r group filter config envvars <<< "$entry"

    if [[ -n "$FILTER_CONFIG" && "$config" != "$FILTER_CONFIG" ]]; then
        skipped=$((skipped + 1))
        continue
    fi
    if [[ -n "$FILTER_GROUP" && "$group" != "$FILTER_GROUP" ]]; then
        skipped=$((skipped + 1))
        continue
    fi

    test_num=$((test_num + 1))
    sim_path="$(simulator_path_for_config "$config")"
    binary="$BUILD_DIR/test/tt_metal/$group"
    label="[$config] $group --gtest_filter=$filter"

    echo "--- [$test_num] $label ---"

    if [[ ! -f "$binary" ]]; then
        echo "  SKIP: binary not found: $binary"
        skipped=$((skipped + 1))
        results+=("SKIP  $label  (binary not found)")
        continue
    fi

    if [[ ! -d "$sim_path" ]]; then
        echo "  SKIP: simulator path not found: $sim_path"
        skipped=$((skipped + 1))
        results+=("SKIP  $label  (simulator not found)")
        continue
    fi

    export TT_METAL_SIMULATOR="$sim_path/"

    # Apply per-test env vars
    extra_env_keys=()
    if [[ -n "$envvars" ]]; then
        while IFS= read -r -d "$SEP" pair || [[ -n "$pair" ]]; do
            [[ -z "$pair" ]] && continue
            key="${pair%%=*}"
            export "$pair"
            extra_env_keys+=("$key")
        done <<< "$envvars"
    fi

    echo "  TT_METAL_SIMULATOR=$TT_METAL_SIMULATOR"
    echo "  TT_METAL_SLOW_DISPATCH_MODE=$TT_METAL_SLOW_DISPATCH_MODE"
    echo "  NNG_SOCKET_ADDR=$NNG_SOCKET_ADDR"
    echo "  NNG_SOCKET_LOCAL_PORT=$NNG_SOCKET_LOCAL_PORT"
    for key in "${extra_env_keys[@]}"; do
        echo "  $key=${!key}"
    done
    echo "  CMD: $binary --gtest_filter=$filter"

    if [[ "$DRY_RUN" == true ]]; then
        results+=("DRY   $label")
        continue
    fi

    resolved_name="$(sanitize_name "$filter")"
    matched_tests="$("$binary" --gtest_list_tests --gtest_filter="$filter" 2>/dev/null | grep -v -e '^\s*$' -e '^Running main' || true)"
    if [[ -n "$matched_tests" ]]; then
        full_name=""
        suite=""
        while IFS= read -r line; do
            if [[ "$line" == *. ]]; then
                suite="$line"
            else
                test="$(echo "$line" | xargs)"
                if [[ -n "$full_name" ]]; then
                    full_name="${full_name}__${test}"
                else
                    full_name="${suite}${test}"
                fi
            fi
        done <<< "$matched_tests"
        resolved_name="$(sanitize_name "$full_name")"
    fi

    gtest_log_args=()
    logger_env=()
    if [[ -n "$LOG_DIR" ]]; then
        log_base="$LOG_DIR/${config}_${group}_${resolved_name}"
        gtest_log_args+=("--gtest_output=json:${log_base}.json")
        logger_env=(env "TT_METAL_LOGGER_FILE=${log_base}.log")
        echo "  TT_METAL_LOGGER_FILE=${log_base}.log"
    fi

    test_start=$SECONDS
    rc=0
    "${logger_env[@]}" "$binary" --gtest_filter="$filter" "${gtest_log_args[@]}" || rc=$?

    elapsed=$((SECONDS - test_start))
    if [[ $rc -eq 0 ]]; then
        passed=$((passed + 1))
        results+=("PASS  $label  ($(fmt_duration $elapsed))")
    else
        failed=$((failed + 1))
        results+=("FAIL  $label  ($(fmt_duration $elapsed))")
        [[ -n "$LOG_DIR" ]] && echo "  LOG: ${log_base}.log"
    fi

    # Clean up per-test env vars
    for key in "${extra_env_keys[@]}"; do
        unset "$key"
    done

    echo ""
done

print_summary

if [[ $failed -gt 0 ]]; then
    exit 1
fi
