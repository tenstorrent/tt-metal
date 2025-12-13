#! /usr/bin/env bash

set -x

source scripts/tools_setup_common.sh

set -eo pipefail

run_profiling_test() {
    TT_METAL_DEVICE_PROFILER=1 pytest $PROFILER_TEST_SCRIPTS_ROOT/test_device_profiler.py --noconftest --timeout 360
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
