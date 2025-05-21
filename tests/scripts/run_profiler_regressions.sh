#! /usr/bin/env bash

set -x

source scripts/tools_setup_common.sh

set -eo pipefail

run_mid_run_tracy_push() {
    echo "Smoke test, checking tracy mid-run device data push for hangs"
    remove_default_log_locations
    mkdir -p $PROFILER_ARTIFACTS_DIR
    python -m tracy -v -r -p --sync-host-device --push-device-data-mid-run -m pytest tests/ttnn/tracy/test_profiler_sync.py::test_all_devices
    runDate=$(ls $PROFILER_OUTPUT_DIR/)
    cat $PROFILER_OUTPUT_DIR/$runDate/ops_perf_results_$runDate.csv
}

run_async_test() {
    #Some tests here do not skip grayskull
    if [ "$ARCH_NAME" == "wormhole_b0" ]; then
        remove_default_log_locations
        mkdir -p $PROFILER_ARTIFACTS_DIR
        ./tt_metal/tools/profiler/profile_this.py -c "pytest -svv models/demos/ttnn_falcon7b/tests/multi_chip/test_falcon_causallm.py::test_falcon_causal_lm[wormhole_b0-20-2-BFLOAT16-L1-falcon_7b-layers_2-decode_batch32]" | tee $PROFILER_ARTIFACTS_DIR/test_out.log

        if cat $PROFILER_ARTIFACTS_DIR/test_out.log | grep "SKIPPED"
        then
            echo "No verification as test was skipped"
        else
            echo "Verifying test results"
            runDate=$(ls $PROFILER_OUTPUT_DIR/)
            LINE_COUNT=1000 # Smoke test to see at least 1000 ops are reported
            res=$(verify_perf_line_count_floor "$PROFILER_OUTPUT_DIR/$runDate/ops_perf_results_$runDate.csv" "$LINE_COUNT")
            echo $res
        fi
    fi
}

run_async_tracing_T3000_test() {
    #Some tests here do not skip grayskull
    if [ "$ARCH_NAME" == "wormhole_b0" ]; then
        remove_default_log_locations
        mkdir -p $PROFILER_ARTIFACTS_DIR

        env WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml ./tt_metal/tools/profiler/profile_this.py -c "pytest models/demos/t3000/resnet50/tests/test_resnet50_performant.py::test_run_resnet50_trace_2cqs_inference[wormhole_b0-16-act_dtype0-weight_dtype0-math_fidelity0-device_params0]" | tee $PROFILER_ARTIFACTS_DIR/test_out.log

        if cat $PROFILER_ARTIFACTS_DIR/test_out.log | grep "SKIPPED"
        then
            echo "No verification as test was skipped"
        else
            echo "Verifying test results"
            runDate=$(ls $PROFILER_OUTPUT_DIR/)
            echo $runDate
            LINE_COUNT=4100 # Smoke test to see at least 4100 ops are reported
            res=$(verify_perf_line_count_floor "$PROFILER_OUTPUT_DIR/$runDate/ops_perf_results_$runDate.csv" "$LINE_COUNT")
            echo $res

            # Testing device only report on the same artifacts
            rm -rf $PROFILER_OUTPUT_DIR/
            ./tt_metal/tools/profiler/process_ops_logs.py --device-only --date
            echo "Verifying device-only results"
            runDate=$(ls $PROFILER_OUTPUT_DIR/)
            echo $runDate
            LINE_COUNT=3600 # Smoke test to see at least 3600 ops are reported
            res=$(verify_perf_line_count_floor "$PROFILER_OUTPUT_DIR/$runDate/ops_perf_results_$runDate.csv" "$LINE_COUNT")
            echo $res

            LINE_COUNT=3600 # Smoke test to see at least 3600 ops are reported
            res=$(verify_perf_line_count_floor "$PROFILER_OUTPUT_DIR/$runDate/per_core_op_to_op_times_$runDate.csv" "$LINE_COUNT")
            echo $res
        fi
    fi
}

run_ccl_T3000_test() {
    remove_default_log_locations
    mkdir -p $PROFILER_ARTIFACTS_DIR

    ./tt_metal/tools/profiler/profile_this.py -c "'pytest tests/ttnn/unit_tests/operations/ccl/test_all_gather.py::test_all_gather_on_t3000_post_commit_for_profiler_regression'" | tee $PROFILER_ARTIFACTS_DIR/test_out.log

    if cat $PROFILER_ARTIFACTS_DIR/test_out.log | grep "SKIPPED"
    then
        echo "No verification as test was skipped"
    else
        echo "Verifying test results"
        runDate=$(ls $PROFILER_OUTPUT_DIR/)
        LINE_COUNT=8 #8 devices
        res=$(verify_perf_line_count "$PROFILER_OUTPUT_DIR/$runDate/ops_perf_results_$runDate.csv" "$LINE_COUNT" "AllGather")
        echo $res
    fi
}

run_async_ccl_T3000_test() {
    remove_default_log_locations
    mkdir -p $PROFILER_ARTIFACTS_DIR

    ./tt_metal/tools/profiler/profile_this.py -c "'pytest tests/ttnn/unit_tests/operations/ccl/test_new_all_gather.py::test_all_gather_sharded_ring'" | tee $PROFILER_ARTIFACTS_DIR/test_out.log

    if cat $PROFILER_ARTIFACTS_DIR/test_out.log | grep "SKIPPED"
    then
        echo "No verification as test was skipped"
    else
        echo "Verifying test results"
        runDate=$(ls $PROFILER_OUTPUT_DIR/)
        LINE_COUNT=128 #8 devices x 16 iterations
        res=$(verify_perf_line_count "$PROFILER_OUTPUT_DIR/$runDate/ops_perf_results_$runDate.csv" "$LINE_COUNT" "AllGatherAsync")
        echo $res
    fi
}

run_profiling_test() {
    run_async_test

    run_ccl_T3000_test

    run_async_ccl_T3000_test

    run_async_tracing_T3000_test

    run_mid_run_tracy_push

    TT_METAL_DEVICE_PROFILER=1 pytest $PROFILER_TEST_SCRIPTS_ROOT/test_device_profiler.py --noconftest --timeout 360

    pytest tests/ttnn/tracy/test_perf_op_report.py --noconftest

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

    export PYTHONPATH=$TT_METAL_HOME

    run_profiling_test
}

main "$@"
