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
  --group <group>              Run specific test group (sync_midrun, basic, trace, dispatch, events, noc_events, fabric, advanced, perf_report)
  -h, --help                   Show this help
EOF
}

# Default to post-commit behavior to preserve existing CI behavior.
MODE="post_commit" # or "full"
GROUP="all" # or specific group name

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
        --group)
            GROUP="$2"
            shift 2
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

run_profiling_test_group_sync_midrun() {
    run_mid_run_data_dump
}

run_profiling_test_group_basic() {
    device_profiler_marker_args=()
    if [[ "$MODE" == "post_commit" ]]; then
        device_profiler_marker_args=(-m "not skip_post_commit")
    fi

    # Basic profiling tests
    TT_METAL_DEVICE_PROFILER=1 pytest $PROFILER_TEST_SCRIPTS_ROOT/test_device_profiler.py::test_multi_op --noconftest --timeout 360 "${device_profiler_marker_args[@]}"
    TT_METAL_DEVICE_PROFILER=1 pytest $PROFILER_TEST_SCRIPTS_ROOT/test_device_profiler.py::test_multi_op_buffer_overflow --noconftest --timeout 360 "${device_profiler_marker_args[@]}"
    TT_METAL_DEVICE_PROFILER=1 pytest $PROFILER_TEST_SCRIPTS_ROOT/test_device_profiler.py::test_custom_cycle_count_slow_dispatch --noconftest --timeout 360 "${device_profiler_marker_args[@]}"
    TT_METAL_DEVICE_PROFILER=1 pytest $PROFILER_TEST_SCRIPTS_ROOT/test_device_profiler.py::test_custom_cycle_count --noconftest --timeout 360 "${device_profiler_marker_args[@]}"
    TT_METAL_DEVICE_PROFILER=1 pytest $PROFILER_TEST_SCRIPTS_ROOT/test_device_profiler.py::test_full_buffer --noconftest --timeout 360 "${device_profiler_marker_args[@]}"
    TT_METAL_DEVICE_PROFILER=1 pytest $PROFILER_TEST_SCRIPTS_ROOT/test_device_profiler.py::test_device_api_debugger_non_dropping --noconftest --timeout 360 "${device_profiler_marker_args[@]}"

    # Performance op reports (excluding TestOpSupportCount)
    TT_METAL_DEVICE_PROFILER=1 pytest tests/ttnn/tracy/test_perf_op_report.py --noconftest -k "not TestOpSupportCount"

    remove_default_log_locations
}

run_profiling_test_group_trace() {
    device_profiler_marker_args=()
    if [[ "$MODE" == "post_commit" ]]; then
        device_profiler_marker_args=(-m "not skip_post_commit")
    fi

    # Trace tests
    TT_METAL_DEVICE_PROFILER=1 pytest $PROFILER_TEST_SCRIPTS_ROOT/test_device_profiler.py::test_trace_run --noconftest --timeout 360 "${device_profiler_marker_args[@]}"
    TT_METAL_DEVICE_PROFILER=1 pytest $PROFILER_TEST_SCRIPTS_ROOT/test_device_profiler.py::test_device_trace_run --noconftest --timeout 360 "${device_profiler_marker_args[@]}"

    remove_default_log_locations
}

run_profiling_test_group_dispatch() {
    device_profiler_marker_args=()
    if [[ "$MODE" == "post_commit" ]]; then
        device_profiler_marker_args=(-m "not skip_post_commit")
    fi

    # Dispatch and NOC tests
    TT_METAL_DEVICE_PROFILER=1 pytest $PROFILER_TEST_SCRIPTS_ROOT/test_device_profiler.py::test_quick_push_on_noc_profiler --noconftest --timeout 360 "${device_profiler_marker_args[@]}"
    TT_METAL_DEVICE_PROFILER=1 pytest $PROFILER_TEST_SCRIPTS_ROOT/test_device_profiler.py::test_dispatch_cores --noconftest --timeout 360 "${device_profiler_marker_args[@]}"
    TT_METAL_DEVICE_PROFILER=1 pytest $PROFILER_TEST_SCRIPTS_ROOT/test_device_profiler.py::test_dispatch_cores_extended_worker --noconftest --timeout 360 "${device_profiler_marker_args[@]}"
    TT_METAL_DEVICE_PROFILER=1 pytest $PROFILER_TEST_SCRIPTS_ROOT/test_device_profiler.py::test_ethernet_dispatch_cores --noconftest --timeout 360 "${device_profiler_marker_args[@]}"

    remove_default_log_locations
}

run_profiling_test_group_events() {
    device_profiler_marker_args=()
    if [[ "$MODE" == "post_commit" ]]; then
        device_profiler_marker_args=(-m "not skip_post_commit")
    fi

    # Sync and event tests
    TT_METAL_DEVICE_PROFILER=1 pytest $PROFILER_TEST_SCRIPTS_ROOT/test_device_profiler.py::test_profiler_host_device_sync --noconftest --timeout 360 "${device_profiler_marker_args[@]}"
    TT_METAL_DEVICE_PROFILER=1 pytest $PROFILER_TEST_SCRIPTS_ROOT/test_device_profiler.py::test_timestamped_events --noconftest --timeout 360 "${device_profiler_marker_args[@]}"

    remove_default_log_locations
}

run_profiling_test_group_noc_events() {
    device_profiler_marker_args=()
    if [[ "$MODE" == "post_commit" ]]; then
        device_profiler_marker_args=(-m "not skip_post_commit")
    fi

    # NOC event profiler tests
    TT_METAL_DEVICE_PROFILER=1 pytest $PROFILER_TEST_SCRIPTS_ROOT/test_device_profiler.py::test_noc_event_profiler_linked_multicast_hang --noconftest --timeout 360 "${device_profiler_marker_args[@]}"
    TT_METAL_DEVICE_PROFILER=1 pytest $PROFILER_TEST_SCRIPTS_ROOT/test_device_profiler.py::test_noc_event_profiler --noconftest --timeout 360 "${device_profiler_marker_args[@]}"

    remove_default_log_locations
}

run_profiling_test_group_fabric() {
    device_profiler_marker_args=()
    if [[ "$MODE" == "post_commit" ]]; then
        device_profiler_marker_args=(-m "not skip_post_commit")
    fi

    # Fabric event profiler tests
    TT_METAL_DEVICE_PROFILER=1 pytest $PROFILER_TEST_SCRIPTS_ROOT/test_device_profiler.py::test_fabric_event_profiler_1d --noconftest --timeout 360 "${device_profiler_marker_args[@]}"
    TT_METAL_DEVICE_PROFILER=1 pytest $PROFILER_TEST_SCRIPTS_ROOT/test_device_profiler.py::test_fabric_event_profiler_fabric_mux --noconftest --timeout 360 "${device_profiler_marker_args[@]}"
    TT_METAL_DEVICE_PROFILER=1 pytest $PROFILER_TEST_SCRIPTS_ROOT/test_device_profiler.py::test_fabric_event_profiler_2d --noconftest --timeout 360 "${device_profiler_marker_args[@]}"

    remove_default_log_locations
}

run_profiling_test_group_advanced() {
    device_profiler_marker_args=()
    if [[ "$MODE" == "post_commit" ]]; then
        device_profiler_marker_args=(-m "not skip_post_commit")
    fi

    # Advanced profiler tests
    TT_METAL_DEVICE_PROFILER=1 pytest $PROFILER_TEST_SCRIPTS_ROOT/test_device_profiler.py::test_sub_device_profiler --noconftest --timeout 360 "${device_profiler_marker_args[@]}"
    TT_METAL_DEVICE_PROFILER=1 pytest $PROFILER_TEST_SCRIPTS_ROOT/test_device_profiler.py::test_get_programs_perf_data --noconftest --timeout 360 "${device_profiler_marker_args[@]}"

    remove_default_log_locations
}

run_profiling_test() {
    case "$GROUP" in
        sync_midrun)
            run_profiling_test_group_sync_midrun
            ;;
        basic)
            run_profiling_test_group_basic
            ;;
        trace)
            run_profiling_test_group_trace
            ;;
        dispatch)
            run_profiling_test_group_dispatch
            ;;
        events)
            run_profiling_test_group_events
            ;;
        noc_events)
            run_profiling_test_group_noc_events
            ;;
        fabric)
            run_profiling_test_group_fabric
            ;;
        advanced)
            run_profiling_test_group_advanced
            ;;
        all)
            # Run all test groups (backward compatibility)
            run_profiling_test_group_sync_midrun
            run_profiling_test_group_basic
            run_profiling_test_group_trace
            run_profiling_test_group_dispatch
            run_profiling_test_group_events
            run_profiling_test_group_noc_events
            run_profiling_test_group_fabric
            run_profiling_test_group_advanced
            ;;
        *)
            echo "Unknown test group: $GROUP" 1>&2
            echo "Valid groups: sync_midrun, basic, trace, dispatch, events, noc_events, fabric, advanced, all" 1>&2
            exit 2
            ;;
    esac
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
