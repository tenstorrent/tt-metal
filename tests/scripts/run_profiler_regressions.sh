#! /usr/bin/env bash

source scripts/tools_setup_common.sh

set -eo pipefail

run_profiling_test(){
    if [[ -z "$ARCH_NAME" ]]; then
      echo "Must provide ARCH_NAME in environment" 1>&2
      exit 1
    fi

    echo "Make sure this test runs in a build with ENABLE_PROFILER=1"

    source build/python_env/bin/activate
    export PYTHONPATH=$TT_METAL_HOME

    TT_METAL_DEVICE_PROFILER=1 pytest $PROFILER_TEST_SCRIPTS_ROOT/test_device_profiler.py -vvv

    remove_default_log_locations

    $PROFILER_SCRIPTS_ROOT/profile_this.py -c "pytest -svvv $TT_METAL_HOME/tests/tt_eager/python_api_testing/sweep_tests/pytests/tt_dnn/test_composite.py::test_run_eltwise_composite_test[lerp_binary-input_shapes0]"

    runDate=$(ls $PROFILER_OUTPUT_DIR/ops/)

    ls $PROFILER_OUTPUT_DIR/ops/$runDate/ops_perf_results_$runDate.csv
    ls $PROFILER_OUTPUT_DIR/ops/$runDate/ops_perf_results_$runDate.tgz

    remove_default_log_locations
}

run_post_proc_test(){
    source build/python_env/bin/activate
    export PYTHONPATH=$TT_METAL_HOME

    pytest $PROFILER_TEST_SCRIPTS_ROOT/test_device_logs.py -vvv
}

run_tracy_test(){

    if [[ -z "$ARCH_NAME" ]]; then
      echo "Must provide ARCH_NAME in environment" 1>&2
      exit 1
    fi

    echo "Make sure this test runs in a build with ENABLE_PROFILER=1"

    source build/python_env/bin/activate
    export PYTHONPATH=$TT_METAL_HOME

    python -m tracy -r -p -m pytest -svvv $TT_METAL_HOME/tests/tt_eager/python_api_testing/sweep_tests/pytests/tt_dnn/test_composite.py::test_run_eltwise_composite_test[lerp_binary-input_shapes0]

    ls $PROFILER_ARTIFACTS_DIR/.logs/tracy_profile_log_host.csv
    ls $PROFILER_ARTIFACTS_DIR/.logs/tracy_profile_log_host.tracy

    remove_default_log_locations
}

cd $TT_METAL_HOME

if [[ $1 == "PROFILER" ]]; then
    run_profiling_test
elif [[ $1 == "TRACY" ]]; then
    run_tracy_test
elif [[ $1 == "POST_PROC" ]]; then
    run_post_proc_test
else
    run_profiling_test
    run_tracy_test
    run_post_proc_test
fi
