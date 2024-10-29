#!/bin/bash

set -eo pipefail

if [[ -z "$TT_METAL_HOME" ]]; then
    echo "Must provide TT_METAL_HOME in environment" 1>&2
    exit 1
fi

if [[ -z "$TT_METAL_SLOW_DISPATCH_MODE" ]] ; then
    # Temporary dispatch compile args testing
    check_list="Semaphore Allocated Configure Sysmem"
    rm -rf built
    echo "FD Compile Args Test - 1CQ"

    TT_METAL_NEW=1 ./build/test/tt_metal/unit_tests_debug_tools --gtest_filter=*WatcherRingBufferBrisc | tee log.new
    for i in $check_list; do
        grep $i log.new > $i.new; sort -n -o $i.new{,}
    done
    find . -name "kernel_args.csv" | xargs -I {} cp {} kernel_args_new.csv
    rm -rf built

    ./build/test/tt_metal/unit_tests_debug_tools --gtest_filter=*WatcherRingBufferBrisc | tee log.old
    for i in $check_list; do
        grep $i log.old > $i.old; sort -n -o $i.old{,}
    done
    find . -name "kernel_args.csv" | xargs -I {} cp {} kernel_args_old.csv
    rm -rf built

    for i in $check_list; do
        if diff $i.old $i.new; then
            echo "$i matches."
        else
            echo "FD Compile Args Test - 1CQ FAIL $i mismatch"
            exit 1
        fi
    done
    if diff kernel_args_old.csv kernel_args_new.csv; then
        echo "Kernel Args match."
    else
        echo "FD Compile Args Test - 1CQ FAIL"
        exit 1
    fi
    echo "FD Compile Args Test - 1CQ PASS"

    if [[ "$ARCH_NAME" == "wormhole_b0" ]]; then
        echo "FD Compile Args Test - 2CQ"

        TT_METAL_GTEST_ETH_DISPATCH=1 TT_METAL_GTEST_NUM_HW_CQS=2 TT_METAL_NEW=1 ./build/test/tt_metal/unit_tests_debug_tools --gtest_filter=*WatcherRingBufferBrisc | tee log.new
        for i in $check_list; do
            grep $i log.new > $i.new; sort -n -o $i.new{,}
        done
        find . -name "kernel_args.csv" | xargs -I {} cp {} kernel_args_new.csv

        TT_METAL_GTEST_ETH_DISPATCH=1 TT_METAL_GTEST_NUM_HW_CQS=2 ./build/test/tt_metal/unit_tests_debug_tools --gtest_filter=*WatcherRingBufferBrisc | tee log.old
        for i in $check_list; do
            grep $i log.old > $i.old; sort -n -o $i.old{,}
        done
        find . -name "kernel_args.csv" | xargs -I {} cp {} kernel_args_old.csv

        for i in $check_list; do
            if diff $i.old $i.new; then
                echo "$i matches."
            else
                echo "FD Compile Args Test - 2CQ FAIL $i mismatch"
                exit 1
            fi
        done
        if diff kernel_args_old.csv kernel_args_new.csv; then
            echo "FD Compile Args Test - 2CQ PASS"
        else
            echo "FD Compile Args Test - 2CQ FAIL"
            exit 1
        fi
    fi
    exit 0

    # Watcher dump tool testing
    echo "Running watcher dump tool tests..."

    # Run a test that populates basic fields but not watcher fields
    ./build/test/tt_metal/unit_tests_debug_tools --gtest_filter=*PrintHanging

    # Run dump tool w/ minimum data - no error expected.
    ./build/tools/watcher_dump -d=0 -w -c

    # Verify the kernel we ran shows up in the log.
    grep "tests/tt_metal/tt_metal/test_kernels/misc/print_hang.cpp" generated/watcher/watcher.log > /dev/null || { echo "Error: couldn't find expected string in watcher log after dump." ; exit 1; }
    echo "Watcher dump minimal test - Pass"

    # Now run with all watcher features, expect it to throw.
    ./build/test/tt_metal/unit_tests_debug_tools --gtest_filter=*WatcherAssertBrisc
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
