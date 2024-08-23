#/bin/bash

set -eo pipefail

if [[ -z "$TT_METAL_HOME" ]]; then
    echo "Must provide TT_METAL_HOME in environment" 1>&2
    exit 1
fi

if [[ -z "$TT_METAL_SLOW_DISPATCH_MODE" ]] ; then
    # Watcher dump tool testing
    echo "Running watcher dump tool tests..."

    # Run a test that populates basic fields but not watcher fields
    ./build/test/tt_metal/unit_tests_fast_dispatch --gtest_filter=*PrintHanging

    # Run dump tool w/ minimum data - no error expected.
    ./build/tools/watcher_dump -d=0 -w -c

    # Verify the kernel we ran shows up in the log.
    grep "tests/tt_metal/tt_metal/test_kernels/misc/print_hang.cpp" generated/watcher/watcher.log > /dev/null || { echo "Error: couldn't find expected string in watcher log after dump." ; exit 1; }
    echo "Watcher dump minimal test - Pass"

    # Now run with all watcher features, expect it to throw.
    ./build/test/tt_metal/unit_tests_fast_dispatch --gtest_filter=*WatcherAssertBrisc
    ./build/tools/watcher_dump -d=0 -w &> tmp.log || { echo "Above failure is expected."; }

    # Verify the error we expect showed up in the program output.
    grep "brisc tripped an assert" tmp.log > /dev/null || { echo "Error: couldn't find expected string in command output:" ; cat tmp.log; exit 1; }
    echo "Watcher dump all data test - Pass"

    # Check that stack dumping is working
    ./build/test/tt_metal/unit_tests_fast_dispatch --gtest_filter=*TestWatcherRingBufferBrisc
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
    timeout 10 ./build/test/tt_metal/test_clean_init || { echo "Error: second run timed out, clean init (FD-on-Tensix) failed."; exit 1; }
    echo "Clean init tests - FD-on-Tensix passed!"

    if [[ "$ARCH_NAME" == "wormhole_b0" ]]; then
        echo "Running clean init tests - FD-on-Eth"
        echo "First run, no teardown"
        env WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml ./build/test/tt_metal/test_clean_init --skip-teardown || { echo "Above failure is expected."; }
        echo "Second run, expect clean init"
        timeout 10 env WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml ./build/test/tt_metal/test_clean_init || { echo "Error: second run timed out, clean init (FD-on-Eth) failed."; exit 1; }
        echo "Clean init tests - FD-on-Eth passed!"
    fi
fi
