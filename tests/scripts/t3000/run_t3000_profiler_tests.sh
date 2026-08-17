#! /usr/bin/env bash

set -x

source scripts/tools_setup_common.sh

set -eo pipefail

run_mid_run_data_dump() {
    remove_default_log_locations
    echo "Smoke test, checking mid-run device data dump for hangs"
    mkdir -p $PROFILER_ARTIFACTS_DIR
    python -m tracy -v -r -p --sync-host-device --dump-device-data-mid-run -m pytest tests/ttnn/tracy/test_profiler_sync.py::test_mesh_device
    runDate=$(ls $PROFILER_OUTPUT_DIR/)
    cat $PROFILER_OUTPUT_DIR/$runDate/ops_perf_results_$runDate.csv
    python $PROFILER_SCRIPTS_ROOT/compare_ops_logs.py
}

run_ccl_T3000_test() {
    remove_default_log_locations
    mkdir -p $PROFILER_ARTIFACTS_DIR

    python -m tracy -v -r -p -m "pytest tests/nightly/t3000/ccl/test_all_gather.py::test_all_gather[wormhole_b0-fabric_ring-mem_config_input0-mem_config_ag0-sd35_prompt-check-mesh_device0]" | tee $PROFILER_ARTIFACTS_DIR/test_out.log


    if cat $PROFILER_ARTIFACTS_DIR/test_out.log | grep "SKIPPED"
    then
        echo "No verification as test was skipped"
    else
        echo "Verifying test results"
        runDate=$(ls $PROFILER_OUTPUT_DIR/)
        LINE_COUNT=8 #8 devices
        res=$(verify_perf_line_count "$PROFILER_OUTPUT_DIR/$runDate/ops_perf_results_$runDate.csv" "$LINE_COUNT" "AllGatherDeviceOperation")
        echo $res
        python $PROFILER_SCRIPTS_ROOT/compare_ops_logs.py
    fi
}

run_tracy_wasm_gui_http_integration() {
    echo "Tracy WASM web GUI HTTP integration (python -m tracy capture + serve_wasm probe)"
    # Free default Tracy WASM ports in case a prior step left serve_wasm listening.
    if command -v fuser >/dev/null 2>&1; then
        fuser -k 8080/tcp 2>/dev/null || true
        fuser -k 8081/tcp 2>/dev/null || true
    fi
    # Shared CI runners: kill listeners after assertions (see test module docstring).
    export TRACY_WASM_HTTP_TEST_TEARDOWN=1
    TTNN_CONFIG_OVERRIDES='{"enable_fast_runtime_mode": false}' pytest \
        tests/ttnn/tracy/test_tracy_wasm_http_integration.py \
        -v --tb=short
}

run_multi_host_tracy_smoke() {
    remove_default_log_locations
    echo "Multi-host tracy smoke test (2 ranks via tt-run)"
    mkdir -p $PROFILER_ARTIFACTS_DIR

    set +e
    tt-run --bare \
        --mpi-args "--allow-run-as-root" \
        --rank-binding tests/ttnn/distributed/config/t3k_tracy_smoke_rank_bindings.yaml \
        --tracy "-r" \
        pytest tests/ttnn/distributed/test_tracy_multi_host_smoke.py | tee $PROFILER_ARTIFACTS_DIR/test_out.log
    tt_run_status=${PIPESTATUS[0]}
    set -e

    if grep -q "SKIPPED" $PROFILER_ARTIFACTS_DIR/test_out.log; then
        echo "No verification as test was skipped (not a T3K)"
        return 0
    fi

    if [ "$tt_run_status" -ne 0 ]; then
        echo "ERROR: tt-run exited with status ${tt_run_status} (see $PROFILER_ARTIFACTS_DIR/test_out.log)"
        exit 1
    fi

    # tt-run may still exit 0 when pytest fails; treat pytest summary lines as failure.
    if grep -qE 'FAILED tests/|ERROR tests/' $PROFILER_ARTIFACTS_DIR/test_out.log; then
        echo "ERROR: pytest reported FAILED or ERROR (see $PROFILER_ARTIFACTS_DIR/test_out.log)"
        exit 1
    fi

    echo "Verifying multi-host tracy results"
    for rank_dir in $PROFILER_ARTIFACTS_DIR/ttrun/rank*; do
        rank=$(basename $rank_dir)
        if [ ! -f "$rank_dir/.logs/tracy_ops_times.csv" ]; then
            echo "ERROR: Missing tracy_ops_times.csv for $rank"
            exit 1
        fi
        echo "✓ $rank: tracy host reports present"
    done
}

run_device_profiler_test() {
    remove_default_log_locations
    TT_METAL_DEVICE_PROFILER=1 pytest $PROFILER_TEST_SCRIPTS_ROOT/test_device_profiler.py --noconftest --timeout 360
}

run_perf_op_report_test() {
    remove_default_log_locations
    TT_METAL_DEVICE_PROFILER=1 pytest tests/ttnn/tracy/test_perf_op_report.py --noconftest
}

run_process_ops_logs_test() {
    remove_default_log_locations
    pytest tests/ttnn/tracy/test_process_ops_logs.py --noconftest
}

# Umbrella that runs every individual test in sequence. Kept for callers that
# don't pass a function name (CI invokes individual functions via the matrix).
run_profiling_test() {
    run_ccl_T3000_test
    run_mid_run_data_dump
    run_multi_host_tracy_smoke
    run_device_profiler_test
    run_perf_op_report_test
    run_process_ops_logs_test
    run_tracy_wasm_gui_http_integration
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

    # If a function name is provided as first argument, run that function
    if [[ -n "$1" ]] && [[ "$(type -t "$1")" == "function" ]]; then
        echo "Running function: $1"
        "$@"
    else
        # Otherwise run all tests
        run_profiling_test
    fi
}

main "$@"
