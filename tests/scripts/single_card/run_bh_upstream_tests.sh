#!/usr/bin/env bash
set -euo pipefail

echo "[upstream-tests] Run BH python upstream tests"
pytest --collect-only tests/ttnn/unit_tests

echo "[upstream-tests] Running BH upstream metal runtime tests"
ARCH_NAME=blackhole TT_METAL_SLOW_DISPATCH_MODE=1 ./tests/scripts/run_cpp_fd2_tests.sh
# I wonder why we can't put these in the validation suite?
./build/test/tt_metal/unit_tests_dispatch --gtest_filter=CommandQueueSingleCardProgramFixture.*
./build/test/tt_metal/unit_tests_dispatch --gtest_filter=CommandQueueProgramFixture.*
./build/test/tt_metal/unit_tests_dispatch --gtest_filter=RandomProgramFixture.*
./build/test/tt_metal/unit_tests_dispatch --gtest_filter=CommandQueueSingleCardBufferFixture.* # Tests EnqueueRead/EnqueueWrite Buffer from DRAM/L1
./build/test/tt_metal/unit_tests_api_blackhole --gtest_filter=*SimpleDram*:*SimpleL1* # Executable is dependent on arch (provided through GitHub CI workflow scripts)
