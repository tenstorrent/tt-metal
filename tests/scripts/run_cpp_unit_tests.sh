#!/bin/bash

set -eo pipefail

if [[ -z "$TT_METAL_HOME" ]]; then
    echo "Must provide TT_METAL_HOME in environment" 1>&2
    exit 1
fi

if [[ ! -z "$TT_METAL_SLOW_DISPATCH_MODE" ]]; then
    ./build/test/tt_metal/unit_tests
    env python tests/scripts/run_tt_metal.py --dispatch-mode slow
    env python tests/scripts/run_tt_eager.py --dispatch-mode slow
else
    ./build/test/tt_metal/unit_tests_fast_dispatch
    TT_METAL_GTEST_NUM_HW_CQS=2 ./build/test/tt_metal/unit_tests_fast_dispatch_single_chip_multi_queue --gtest_filter=MultiCommandQueueSingleDeviceFixture.*
    if [[ "$ARCH_NAME" == "wormhole_b0" || "$ARCH_NAME" == "blackhole" ]]; then
        TT_METAL_GTEST_ETH_DISPATCH=1 ./build/test/tt_metal/unit_tests_fast_dispatch
    fi
    env python tests/scripts/run_tt_eager.py --dispatch-mode fast
    env python tests/scripts/run_tt_metal.py --dispatch-mode fast
fi

# Tool tests use C++ unit tests so include them here.
./tests/scripts/run_tools_tests.sh
