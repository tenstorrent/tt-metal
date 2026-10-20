#!/bin/bash

# Runs the L1 ECC halt test that halts a Blackhole BRISC for real through the watchdog's software interrupt. A
# halted BRISC stays inside its interrupt handler until the chip is reset, so the chip is always reset afterwards.

if [ "$ARCH_NAME" != "blackhole" ]; then
    exit 0
fi

rc=0
timeout 600 ./build/test/tt_metal/unit_tests_debug_tools --gtest_also_run_disabled_tests \
    --gtest_filter='L1EccHaltTest.DISABLED_WatchdogHalt' || rc=$?
tt-smi -r || rc=$?
exit $rc
