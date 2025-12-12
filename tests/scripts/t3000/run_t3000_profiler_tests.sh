#! /usr/bin/env bash

set -x

source scripts/tools_setup_common.sh

set -eo pipefail

run_mid_run_data_dump() {
    echo "Smoke test, checking mid-run device data dump for hangs"
    remove_default_log_locations
    mkdir -p $PROFILER_ARTIFACTS_DIR
    python -m tracy -v -r -p --sync-host-device --dump-device-data-mid-run -m pytest tests/ttnn/tracy/test_profiler_sync.py::test_mesh_device
    runDate=$(ls $PROFILER_OUTPUT_DIR/)
    cat $PROFILER_OUTPUT_DIR/$runDate/ops_perf_results_$runDate.csv
    python $PROFILER_SCRIPTS_ROOT/compare_ops_logs.py
}

run_async_test() {
    #Some tests here do not skip grayskull
    if [ "$ARCH_NAME" == "wormhole_b0" ]; then
        remove_default_log_locations
        mkdir -p $PROFILER_ARTIFACTS_DIR
        python -m tracy -v -r -p -m "pytest -svv models/demos/ttnn_falcon7b/tests/multi_chip/test_falcon_causallm.py::test_falcon_causal_lm[wormhole_b0-20-2-BFLOAT16-L1-falcon_7b-layers_2-decode_batch32]" | tee $PROFILER_ARTIFACTS_DIR/test_out.log

        if cat $PROFILER_ARTIFACTS_DIR/test_out.log | grep "SKIPPED"
        then
            echo "No verification as test was skipped"
        else
            echo "Verifying test results"
            runDate=$(ls $PROFILER_OUTPUT_DIR/)
            LINE_COUNT=1000 # Smoke test to see at least 1000 ops are reported
            res=$(verify_perf_line_count_floor "$PROFILER_OUTPUT_DIR/$runDate/ops_perf_results_$runDate.csv" "$LINE_COUNT")
            echo $res
            python $PROFILER_SCRIPTS_ROOT/compare_ops_logs.py
        fi
    fi
}

run_async_tracing_T3000_test() {
    #Some tests here do not skip grayskull
    if [ "$ARCH_NAME" == "wormhole_b0" ]; then
        remove_default_log_locations
        mkdir -p $PROFILER_ARTIFACTS_DIR

        python -m tracy -v -r -p -m "pytest models/demos/ttnn_resnet/tests/test_resnet50_performant.py::test_run_resnet50_trace_2cqs_inference[wormhole_b0-16-act_dtype0-weight_dtype0-math_fidelity0-device_params0]" | tee $PROFILER_ARTIFACTS_DIR/test_out.log

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
    #Some tests here do not skip grayskull
    if [ "$ARCH_NAME" == "wormhole_b0" ]; then
        remove_default_log_locations
        mkdir -p $PROFILER_ARTIFACTS_DIR

        python -m tracy -v -r -p --dump-device-data-mid-run -m pytest models/demos/ttnn_resnet/tests/test_resnet50_performant.py::test_run_resnet50_trace_2cqs_inference[wormhole_b0-16-act_dtype0-weight_dtype0-math_fidelity0-device_params0] | tee $PROFILER_ARTIFACTS_DIR/test_out.log

        if cat $PROFILER_ARTIFACTS_DIR/test_out.log | grep "SKIPPED"
        then
            echo "No verification as test was skipped"
            return
        fi

        midRunDumpRunDate=$(ls $PROFILER_OUTPUT_DIR/)
        mv $PROFILER_ARTIFACTS_DIR/.logs/cpp_device_perf_report.csv $PROFILER_OUTPUT_DIR/$midRunDumpRunDate/cpp_device_perf_report.csv

        python -m tracy -v -r -p -m pytest models/demos/ttnn_resnet/tests/test_resnet50_performant.py::test_run_resnet50_trace_2cqs_inference[wormhole_b0-16-act_dtype0-weight_dtype0-math_fidelity0-device_params0]

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

run_profiling_test() {
    run_async_test

    run_ccl_T3000_test

    run_async_tracing_T3000_test

    run_async_tracing_mid_run_dump_T3000_test

    run_mid_run_data_dump

    TT_METAL_DEVICE_PROFILER=1 pytest $PROFILER_TEST_SCRIPTS_ROOT/test_device_profiler.py --noconftest --timeout 360

    TT_METAL_DEVICE_PROFILER=1 pytest tests/ttnn/tracy/test_perf_op_report.py --noconftest

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
