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

run_async_tracing_T3000_test() {
    remove_default_log_locations
    #Some tests here do not skip grayskull
    if [ "$ARCH_NAME" == "wormhole_b0" ]; then
        mkdir -p $PROFILER_ARTIFACTS_DIR

        python -m tracy -v -r -p -m "pytest models/demos/vision/classification/resnet50/ttnn_resnet/tests/test_resnet50_performant.py::test_run_resnet50_trace_2cqs_inference[wormhole_b0-16-DataType.BFLOAT8_B-DataType.BFLOAT8_B-MathFidelity.LoFi-device_params0]" | tee $PROFILER_ARTIFACTS_DIR/test_out.log

        if cat $PROFILER_ARTIFACTS_DIR/test_out.log | grep "SKIPPED"
        then
            echo "No verification as test was skipped"
        else
            echo "Verifying test results"
            runDate=$(ls $PROFILER_OUTPUT_DIR/)
            echo $runDate
            LINE_COUNT=2600
            res=$(verify_perf_line_count_floor "$PROFILER_OUTPUT_DIR/$runDate/ops_perf_results_$runDate.csv" "$LINE_COUNT")
            echo $res
            python $PROFILER_SCRIPTS_ROOT/compare_ops_logs.py

            # Compare runtime analysis report with legacy path report
            runtime_report="$PROFILER_OUTPUT_DIR/$runDate/ops_perf_results_$runDate.csv"
            echo "Comparing runtime analysis report with legacy path report..."
            echo "Runtime analysis report: $runtime_report"

            # Regenerate report using legacy path
            echo "Regenerating report using legacy path (--force-legacy-device-logs)..."
            ./tools/tracy/process_ops_logs.py -o $PROFILER_ARTIFACTS_DIR -n "legacy_comparison" --force-legacy-device-logs

            # Find the legacy report (it's in the reports subdirectory)
            legacy_report="$PROFILER_ARTIFACTS_DIR/reports/legacy_comparison/ops_perf_results_legacy_comparison.csv"

            if [ ! -f "$legacy_report" ]; then
                echo "ERROR: Legacy path report not found at $legacy_report"
                exit 1
            fi

            echo "Legacy path report: $legacy_report"

            # Compare the two reports
            if python tools/tracy/compare_full_op_report.py "$runtime_report" "$legacy_report"; then
                echo "✓ Reports are identical - runtime analysis and legacy path produce the same results"
            else
                echo "✗ Reports differ - runtime analysis and legacy path produce different results"
                exit 1
            fi

            rm -rf $PROFILER_ARTIFACTS_DIR/reports/legacy_comparison


            # Testing device only report on the same artifacts
            rm -rf $PROFILER_OUTPUT_DIR/$runDate
            ./tools/tracy/process_ops_logs.py --device-only --date
            echo "Verifying device-only results"
            runDate=$(ls $PROFILER_OUTPUT_DIR/)
            echo $runDate
            LINE_COUNT=1700
            res=$(verify_perf_line_count_floor "$PROFILER_OUTPUT_DIR/$runDate/ops_perf_results_$runDate.csv" "$LINE_COUNT")
            echo $res
            LINE_COUNT=1700
            res=$(verify_perf_line_count_floor "$PROFILER_OUTPUT_DIR/$runDate/per_core_op_to_op_times_$runDate.csv" "$LINE_COUNT")
            echo $res
        fi
    fi
}

run_async_tracing_mid_run_dump_T3000_test() {
    remove_default_log_locations
    #Some tests here do not skip grayskull
    if [ "$ARCH_NAME" == "wormhole_b0" ]; then
        mkdir -p $PROFILER_ARTIFACTS_DIR

        python -m tracy -v -r -p --dump-device-data-mid-run -m pytest models/demos/vision/classification/resnet50/ttnn_resnet/tests/test_resnet50_performant.py::test_run_resnet50_trace_2cqs_inference[wormhole_b0-16-DataType.BFLOAT8_B-DataType.BFLOAT8_B-MathFidelity.LoFi-device_params0] | tee $PROFILER_ARTIFACTS_DIR/test_out.log

        if cat $PROFILER_ARTIFACTS_DIR/test_out.log | grep "SKIPPED"
        then
            echo "No verification as test was skipped"
            return
        fi

        midRunDumpRunDate=$(ls $PROFILER_OUTPUT_DIR/)
        mv $PROFILER_ARTIFACTS_DIR/.logs/cpp_device_perf_report.csv $PROFILER_OUTPUT_DIR/$midRunDumpRunDate/cpp_device_perf_report.csv

        python -m tracy -v -r -p -m pytest models/demos/vision/classification/resnet50/ttnn_resnet/tests/test_resnet50_performant.py::test_run_resnet50_trace_2cqs_inference[wormhole_b0-16-DataType.BFLOAT8_B-DataType.BFLOAT8_B-MathFidelity.LoFi-device_params0]

        nonMidRunDumpRunDate=$(ls $PROFILER_OUTPUT_DIR/ | grep -v $midRunDumpRunDate)
        mv $PROFILER_ARTIFACTS_DIR/.logs/cpp_device_perf_report.csv $PROFILER_OUTPUT_DIR/$nonMidRunDumpRunDate/cpp_device_perf_report.csv

        python $PROFILER_SCRIPTS_ROOT/compare_ops_logs.py $PROFILER_OUTPUT_DIR/$midRunDumpRunDate/ops_perf_results_$midRunDumpRunDate.csv $PROFILER_OUTPUT_DIR/$midRunDumpRunDate/cpp_device_perf_report.csv
        python $PROFILER_SCRIPTS_ROOT/compare_ops_logs.py $PROFILER_OUTPUT_DIR/$nonMidRunDumpRunDate/ops_perf_results_$nonMidRunDumpRunDate.csv $PROFILER_OUTPUT_DIR/$nonMidRunDumpRunDate/cpp_device_perf_report.csv

        python $PROFILER_SCRIPTS_ROOT/compare_ops_logs.py --only-compare-op-ids $PROFILER_OUTPUT_DIR/$midRunDumpRunDate/ops_perf_results_$midRunDumpRunDate.csv $PROFILER_OUTPUT_DIR/$nonMidRunDumpRunDate/cpp_device_perf_report.csv
        python $PROFILER_SCRIPTS_ROOT/compare_ops_logs.py --only-compare-op-ids $PROFILER_OUTPUT_DIR/$nonMidRunDumpRunDate/ops_perf_results_$nonMidRunDumpRunDate.csv $PROFILER_OUTPUT_DIR/$midRunDumpRunDate/cpp_device_perf_report.csv
    fi
}

run_ccl_T3000_test() {
    remove_default_log_locations
    mkdir -p $PROFILER_ARTIFACTS_DIR

    python -m tracy -v -r -p -m "pytest tests/nightly/t3000/ccl/test_minimal_all_gather_async.py::test_ttnn_all_gather[wormhole_b0-fabric_ring-mem_config_input0-mem_config_ag0-sd35_prompt-check-1link-mesh_device0]" | tee $PROFILER_ARTIFACTS_DIR/test_out.log


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

run_trace_only_resnet() {
    remove_default_log_locations
    mkdir -p $PROFILER_ARTIFACTS_DIR

    TT_METAL_DEVICE_PROFILER=1 python -m tracy -v -p --device-trace-profiler -m \
            pytest models/demos/vision/classification/resnet50/ttnn_resnet/tests/test_resnet50_performant.py::test_run_resnet50_trace_2cqs_inference[wormhole_b0-16-DataType.BFLOAT8_B-DataType.BFLOAT8_B-MathFidelity.LoFi-device_params0]
    if cat $PROFILER_ARTIFACTS_DIR/test_out.log | grep "SKIPPED"
    then
        echo "No verification as test was skipped"
    else
        echo "Test ran successfully"
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
    run_async_tracing_T3000_test
    run_async_tracing_mid_run_dump_T3000_test
    run_mid_run_data_dump
    run_trace_only_resnet
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
