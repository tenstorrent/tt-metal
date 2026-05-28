#! /usr/bin/env bash

set -x

source scripts/tools_setup_common.sh

set -eo pipefail

run_device_profiler_test() {
    remove_default_log_locations
    TT_METAL_DEVICE_PROFILER=1 pytest $PROFILER_TEST_SCRIPTS_ROOT/test_device_profiler.py --noconftest --timeout 360
}

run_realtime_profiler_cross_reference_tg_test() {
    remove_default_log_locations
    # Cross-reference real-time profiler durations against device profiler
    # on a full TG (8x4) mesh. This test was consolidated into the unified
    # real-time profiler test suite (tests/ttnn/tracy/test_realtime_profiler.py)
    # under the name test_cross_reference_tg; the old standalone file
    # test_profiler_cross_reference_TG.py no longer exists.
    TT_METAL_DEVICE_PROFILER=1 pytest tests/ttnn/tracy/test_realtime_profiler.py::test_cross_reference_tg --timeout 2400
}

# Umbrella that runs every individual test in sequence. Kept for callers that
# don't pass a function name (CI invokes individual functions via the matrix).
run_profiling_test() {
    run_device_profiler_test
    run_realtime_profiler_cross_reference_tg_test
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
