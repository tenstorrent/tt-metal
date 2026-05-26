#! /usr/bin/env bash

set -x

source scripts/tools_setup_common.sh

set -eo pipefail

usage() {
    cat <<'EOF'
Usage: run_profiler_regressions.sh [options]

Options:
  --post-commit                Post-commit mode: exclude tests marked with @pytest.mark.skip_post_commit (default)
  --full                       Full mode: include tests marked with @pytest.mark.skip_post_commit
  --include-skip-post-commit   Alias for --full (backwards compatible)
  -h, --help                   Show this help
EOF
}

# Default to post-commit behavior to preserve existing CI behavior.
MODE="post_commit" # or "full"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --post-commit)
            MODE="post_commit"
            break
            ;;
        --full)
            MODE="full"
            break
            ;;
        --include-skip-post-commit)
            MODE="full"
            break
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" 1>&2
            usage 1>&2
            exit 2
            ;;
    esac
done

run_mid_run_data_dump() {
    remove_default_log_locations
    echo "Smoke test, checking mid-run device data dump for hangs"
    mkdir -p $PROFILER_ARTIFACTS_DIR
    python -m tracy -v -r -p --sync-host-device --dump-device-data-mid-run -m pytest tests/ttnn/tracy/test_profiler_sync.py::test_mesh_device
    python $PROFILER_SCRIPTS_ROOT/compare_ops_logs.py
}

run_device_profiler_test() {
    remove_default_log_locations
    device_profiler_marker_args=()
    if [[ "$MODE" == "post_commit" ]]; then
        device_profiler_marker_args=(-m "not skip_post_commit")
    fi

    TT_METAL_DEVICE_PROFILER=1 pytest $PROFILER_TEST_SCRIPTS_ROOT/test_device_profiler.py --noconftest --timeout 360 "${device_profiler_marker_args[@]}"
}

run_perf_op_report_test() {
    remove_default_log_locations
    TT_METAL_DEVICE_PROFILER=1 pytest tests/ttnn/tracy/test_perf_op_report.py --noconftest -k "not TestOpSupportCount"
}

run_realtime_profiler_test() {
    remove_default_log_locations
    # Consolidated real-time profiler test suite: callback smoke test, short-zone
    # regression, host/device correlation, cross-reference vs device profiler,
    # sync-accuracy check (and TG cross-reference, which auto-skips off-Galaxy).
    # Each test runs its device-touching work in its own subprocess workload
    # (so the pytest parent never takes the PCIe lock) and exports
    # TT_METAL_DEVICE_PROFILER=1 itself when needed.  Per-test timeouts come
    # from @pytest.mark.timeout decorators on the individual tests.
    pytest tests/ttnn/tracy/test_realtime_profiler.py
}

# Umbrella that runs every individual test in sequence. Kept for callers that
# don't pass a function name (CI invokes individual functions via the matrix).
run_profiling_test() {
    run_mid_run_data_dump
    run_device_profiler_test
    run_perf_op_report_test
    run_realtime_profiler_test
}

main() {
    cd $TT_METAL_HOME

    TTNN_CONFIG_OVERRIDES='{"enable_fast_runtime_mode": false}'

    if [[ -z "$ARCH_NAME" ]]; then
        echo "Must provide ARCH_NAME in environment" 1>&2
        exit 1
    fi

    echo "Make sure this test runs in a build with cmake option ENABLE_TRACY=ON"

    if [[ -z "$DONT_USE_VIRTUAL_ENVIRONMENT" ]]; then
        source python_env/bin/activate
    fi

    # If a function name is provided as last argument, run that function
    if [[ -n "${!#}" ]] && [[ "$(type -t "${!#}")" == "function" ]]; then
        echo "Running function: ${!#}"
        "${!#}"
    else
        # Otherwise run all tests
        run_profiling_test
    fi
}

main "$@"
