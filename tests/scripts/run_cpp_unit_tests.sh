#!/bin/bash

set -eo pipefail

if [[ -z "$TT_METAL_HOME" ]]; then
    echo "Must provide TT_METAL_HOME in environment" 1>&2
    exit 1
fi

kernel_path="/tmp/kernels"
mkdir -p $kernel_path
TT_METAL_KERNEL_PATH=$kernel_path ./build/test/tt_metal/unit_tests_api --gtest_filter=CompileProgramWithKernelPathEnvVarFixture.*
rm -rf $kernel_path

./build/test/tt_metal/unit_tests_api
./build/test/tt_metal/unit_tests_debug_tools
./build/test/tt_metal/unit_tests_device
./build/test/tt_metal/unit_tests_dispatch
./build/test/tt_metal/unit_tests_eth
./build/test/tt_metal/unit_tests_llk
./build/test/tt_metal/unit_tests_stl

if [[ ! -z "$TT_METAL_SLOW_DISPATCH_MODE" ]]; then
    env python3 tests/scripts/run_tt_metal.py --dispatch-mode slow
    env python3 tests/scripts/run_tt_eager.py --dispatch-mode slow
else
    TT_METAL_GTEST_NUM_HW_CQS=2 ./build/test/tt_metal/unit_tests_dispatch --gtest_filter=MultiCommandQueue*Fixture.*
    # Enable this on BH after #14613
    if [[ "$ARCH_NAME" == "wormhole_b0" ]]; then
        TT_METAL_GTEST_ETH_DISPATCH=1 ./build/test/tt_metal/unit_tests_dispatch
    fi
    env python3 tests/scripts/run_tt_eager.py --dispatch-mode fast
    env python3 tests/scripts/run_tt_metal.py --dispatch-mode fast
fi

# Tool tests use C++ unit tests so include them here.
./tests/scripts/run_tools_tests.sh
