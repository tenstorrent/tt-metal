#! /usr/bin/env bash

set -x

source scripts/tools_setup_common.sh

set -eo pipefail

run_mid_run_data_dump() {
    echo "Smoke test, checking mid-run device data dump for hangs"
    remove_default_log_locations
    mkdir -p $PROFILER_ARTIFACTS_DIR
    python -m tracy -v -r -p --sync-host-device --cpp-post-process --dump-device-data-mid-run -m pytest tests/ttnn/tracy/test_profiler_sync.py::test_all_devices
    runDate=$(ls $PROFILER_OUTPUT_DIR/ | grep -v "reports" | grep -v "ops_perf_results.csv")
    python $PROFILER_SCRIPTS_ROOT/compare_ops_logs.py --python-ops-perf-report $PROFILER_OUTPUT_DIR/$runDate/ops_perf_results_$runDate.csv --cpp-ops-perf-report $PROFILER_OUTPUT_DIR/ops_perf_results.csv
}

run_profiling_test() {

    run_mid_run_data_dump

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

    run_profiling_test
}

main "$@"
