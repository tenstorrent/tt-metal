#! /usr/bin/env bash

source scripts/tools_setup_common.sh

set -eo pipefail

run_additional_T3000_test(){
    remove_default_log_locations
    mkdir -p $PROFILER_ARTIFACTS_DIR

    ./tt_metal/tools/profiler/profile_this.py -c "'pytest tests/ttnn/unit_tests/operations/test_all_gather.py::test_all_gather_on_t3000_post_commit_for_profiler_regression'" | tee $PROFILER_ARTIFACTS_DIR/test_out.log

    if cat $PROFILER_ARTIFACTS_DIR/test_out.log | grep "SKIPPED"
    then
        echo "No verification as test was skipped"
    else
        echo "Verifying test results"
        runDate=$(ls $PROFILER_OUTPUT_DIR/)
        LINE_COUNT=9 #1 header + 8 devices
        res=$(verify_perf_line_count "$PROFILER_OUTPUT_DIR/$runDate/ops_perf_results_$runDate.csv" "$LINE_COUNT")
        echo $res
    fi
}

run_async_mode_T3000_test(){
    #Some tests here do not skip grayskull
    if [ "$ARCH_NAME" == "wormhole_b0" ]; then
        remove_default_log_locations
        mkdir -p $PROFILER_ARTIFACTS_DIR

        ./tt_metal/tools/profiler/profile_this.py -c "pytest -svv models/demos/ttnn_falcon7b/tests/multi_chip/test_falcon_causallm.py::test_falcon_causal_lm[wormhole_b0-True-True-20-2-BFLOAT16-L1-falcon_7b-layers_2-decode_batch32]" | tee $PROFILER_ARTIFACTS_DIR/test_out.log

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

run_profiling_test(){
    if [[ -z "$ARCH_NAME" ]]; then
      echo "Must provide ARCH_NAME in environment" 1>&2
      exit 1
    fi

    echo "Make sure this test runs in a build with cmake option ENABLE_TRACY=ON"

    source python_env/bin/activate
    export PYTHONPATH=$TT_METAL_HOME

    run_additional_T3000_test

    run_async_mode_T3000_test

    TT_METAL_DEVICE_PROFILER=1 pytest $PROFILER_TEST_SCRIPTS_ROOT/test_device_profiler.py

    remove_default_log_locations

    $PROFILER_SCRIPTS_ROOT/profile_this.py -c "pytest tests/tt_eager/python_api_testing/sweep_tests/pytests/tt_dnn/test_matmul.py::test_run_matmul_test[BFLOAT16-input_shapes0]"

    runDate=$(ls $PROFILER_OUTPUT_DIR/)

    CORE_COUNT=7
    res=$(verify_perf_column "$PROFILER_OUTPUT_DIR/$runDate/ops_perf_results_$runDate.csv" "$CORE_COUNT" "1" "1")
    echo $res

    remove_default_log_locations
}

run_profiling_no_reset_test(){
    if [[ -z "$ARCH_NAME" ]]; then
      echo "Must provide ARCH_NAME in environment" 1>&2
      exit 1
    fi

    echo "Make sure this test runs in a build with cmake option ENABLE_TRACY=ON"

    source python_env/bin/activate
    export PYTHONPATH=$TT_METAL_HOME

    TT_METAL_DEVICE_PROFILER=1 pytest $PROFILER_TEST_SCRIPTS_ROOT/test_device_profiler_gs_no_reset.py

    remove_default_log_locations
}

run_post_proc_test(){
    source python_env/bin/activate
    export PYTHONPATH=$TT_METAL_HOME

    pytest $PROFILER_TEST_SCRIPTS_ROOT/test_device_logs.py -vvv
}

cd $TT_METAL_HOME

#
TTNN_CONFIG_OVERRIDES='{"enable_fast_runtime_mode": false}'

if [[ $1 == "PROFILER" ]]; then
    run_profiling_test
elif [[ $1 == "PROFILER_NO_RESET" ]]; then
    run_profiling_no_reset_test
elif [[ $1 == "POST_PROC" ]]; then
    run_post_proc_test
else
    run_profiling_test
    run_post_proc_test
fi
