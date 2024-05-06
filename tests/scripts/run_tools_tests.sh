#/bin/bash

set -eo pipefail

if [[ -z "$TT_METAL_HOME" ]]; then
    echo "Must provide TT_METAL_HOME in environment" 1>&2
    exit 1
fi

# For now, only test watcher dump tool here.
if [[ -z "$TT_METAL_SLOW_DISPATCH_MODE" ]] ; then
    # Run a test that populates basic fields but not watcher fields
    ./build/test/tt_metal/unit_tests_fast_dispatch --gtest_filter=*PrintHanging

    # Run dump tool w/ minimum data - no error expected.
    ./build/tools/watcher_dump -d=0

    # Verify the kernel we ran shows up in the log.
    grep "tests/tt_metal/tt_metal/test_kernels/misc/print_hang.cpp" generated/watcher/watcher.log > /dev/null || { echo "Error: couldn't find expected string in watcher log after dump." ; exit 1; }
    echo "Watcher dump minimal test - Pass"

    # Now run with all watcher features, expect it to throw.
    ./build/test/tt_metal/unit_tests_fast_dispatch --gtest_filter=*WatcherAssertBrisc
    ./build/tools/watcher_dump -d=0 &> tmp.log || { echo "Above failure is expected."; }

    # Verify the error we expect showed up in the program output.
    grep "brisc tripped an assert" tmp.log > /dev/null || { echo "Error: couldn't find expected string in command output:" ; cat tmp.log; exit 1; }
    echo "Watcher dump all data test - Pass"

    # Remove created files.
    rm tmp.log
    rm generated/watcher/watcher.log
fi
