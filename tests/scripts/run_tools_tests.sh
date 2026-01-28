#!/bin/bash

set -eo pipefail

if [ -z "${ARCH_NAME}" ]; then
  echo "Error: ARCH_NAME is not set. Exiting." >&2
  exit 1
fi

if [[ -z "$TT_METAL_SLOW_DISPATCH_MODE" ]] ; then
    # Watcher dump tool testing
    echo "Running watcher dump tool tests..."

    # Now run with all watcher features, expect it to throw.
    TT_METAL_WATCHER_KEEP_ERRORS=1 ./build/test/tt_metal/unit_tests_debug_tools --gtest_filter=WatcherAssertTests/*Brisc
    ./build/tools/watcher_dump -d=0 -w &> tmp.log || { echo "Above failure is expected."; }

    # Verify the error we expect showed up in the program output.
    grep "brisc tripped an assert" tmp.log > /dev/null || { echo "Error: couldn't find expected string in command output:" ; cat tmp.log; exit 1; }
    echo "Watcher dump all data test - Pass"

    # Check that stack dumping is working
    ./build/test/tt_metal/unit_tests_debug_tools --gtest_filter=*TestWatcherRingBufferBrisc
    ./build/tools/watcher_dump -d=0 -w
    grep "brisc highest stack usage:" generated/watcher/watcher.log > /dev/null || { echo "Error: couldn't find stack usage in watcher log after dump." ; exit 1; }
    echo "Watcher stack usage test - Pass"

    # Remove created files.
    rm tmp.log
    rm generated/watcher/watcher.log
    rm generated/watcher/command_queue_dump/*
    echo "Watcher dump tool tests finished..."


    # Clean init testing
    echo "Running clean init tests - FD-on-Tensix"
    echo "First run, no teardown"
    ./build/test/tt_metal/test_clean_init --skip-teardown || { echo "Above failure is expected."; }
    echo "Second run, expect clean init"
    timeout 40 ./build/test/tt_metal/test_clean_init || { echo "Error: second run timed out, clean init (FD-on-Tensix) failed."; exit 1; }
    echo "Clean init tests - FD-on-Tensix passed!"

    if [[ "$ARCH_NAME" == "wormhole_b0" ]]; then
        echo "Running clean init tests - FD-on-Eth"
        echo "First run, no teardown"
        env ./build/test/tt_metal/test_clean_init --skip-teardown || { echo "Above failure is expected."; }
        echo "Second run, expect clean init"
        timeout 40 env ./build/test/tt_metal/test_clean_init || { echo "Error: second run timed out, clean init (FD-on-Eth) failed."; exit 1; }
        echo "Clean init tests - FD-on-Eth passed!"
    fi
fi

# MGD generation tests (tests that generate Mesh Graph Descriptors from cabling descriptors)
echo "Running MGD generation tests..."
./build/test/tools/scaleout/test_cabling_descriptor_mgd_generation
echo "MGD generation tests finished"

# Descriptor merger tests (tests for merging cabling descriptors)
echo "Running descriptor merger tests..."
TT_METAL_LOGGER_LEVEL=Fatal ./build/test/tools/scaleout/test_descriptor_merger
echo "Descriptor merger tests finished"
