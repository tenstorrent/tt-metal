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
            shift
            ;;
        --full)
            MODE="full"
            shift
            ;;
        --include-skip-post-commit)
            MODE="full"
            shift
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
    echo "Smoke test, checking mid-run device data dump for hangs"
    remove_default_log_locations
    mkdir -p $PROFILER_ARTIFACTS_DIR
    python -m tracy -v -r -p --sync-host-device --dump-device-data-mid-run -m pytest tests/ttnn/tracy/test_profiler_sync.py::test_mesh_device
    python $PROFILER_SCRIPTS_ROOT/compare_ops_logs.py
}

run_profiling_test() {

    run_mid_run_data_dump

    device_profiler_marker_args=()
    if [[ "$MODE" == "post_commit" ]]; then
        device_profiler_marker_args=(-m "not skip_post_commit")
    fi

    TT_METAL_DEVICE_PROFILER=1 pytest $PROFILER_TEST_SCRIPTS_ROOT/test_device_profiler.py --noconftest --timeout 360 "${device_profiler_marker_args[@]}"

    TT_METAL_DEVICE_PROFILER=1 pytest tests/ttnn/tracy/test_perf_op_report.py --noconftest -k "not TestOpSupportCount"

    remove_default_log_locations
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

    run_profiling_test
}

main "$@"
