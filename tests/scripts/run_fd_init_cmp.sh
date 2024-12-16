#!/bin/bash

set -eo pipefail

if [[ -z "$TT_METAL_HOME" ]]; then
    echo "Must provide TT_METAL_HOME in environment" 1>&2
    exit 1
fi

if [[ -z "$TT_METAL_SLOW_DISPATCH_MODE" ]] ; then
    # Temporary dispatch compile args testing
    check_list="Semaphore Allocated Configure Sysmem"
    test_cmd="./build/test/tt_metal/unit_tests_dispatch --gtest_filter=CommandQueueSingleCardBufferFixture.WriteOneTileToDramBank0"
    export TT_METAL_WATCHER=1
    export TT_METAL_WATCHER_NOINLINE=1
    rm -rf built
    echo "FD Compile Args Test - 1CQ"

    TT_METAL_ENABLE_REMOTE_CHIP=1 $test_cmd | tee log.new
    for i in $check_list; do
        grep $i log.new > $i.new; sort -n $i.new -o $i.new
    done
    find . -name "kernel_args.csv" | xargs -I {} wc -l {} | sort -n | tail -1l | awk '{print $2}' | xargs -I {} cp {} kernel_args_new.csv
    rm -rf built

    TT_METAL_ENABLE_REMOTE_CHIP=1 TT_METAL_OLD_FD_INIT=1 $test_cmd | tee log.old
    for i in $check_list; do
        grep $i log.old > $i.old; sort -n $i.old -o $i.old
    done
    find . -name "kernel_args.csv" | xargs -I {} wc -l {} | sort -n | tail -1l | awk '{print $2}' | xargs -I {} cp {} kernel_args_old.csv
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

    test_cmd="./build/test/tt_metal/unit_tests_dispatch --gtest_filter=MultiCommandQueueMultiDeviceBufferFixture.WriteOneTileToDramBank0"
    if [[ "$ARCH_NAME" == "wormhole_b0" ]]; then
        echo "FD Compile Args Test - 2CQ"

        TT_METAL_GTEST_ETH_DISPATCH=1 TT_METAL_GTEST_NUM_HW_CQS=2 $test_cmd | tee log.new
        for i in $check_list; do
            grep $i log.new > $i.new; sort -n $i.new -o $i.new
        done
        find . -name "kernel_args.csv" | xargs -I {} wc -l {} | sort -n | tail -1l | awk '{print $2}' | xargs -I {} cp {} kernel_args_new.csv

        TT_METAL_GTEST_ETH_DISPATCH=1 TT_METAL_GTEST_NUM_HW_CQS=2 TT_METAL_OLD_FD_INIT=1 $test_cmd | tee log.old
        for i in $check_list; do
            grep $i log.old > $i.old; sort -n $i.old -o $i.old
        done
        find . -name "kernel_args.csv" | xargs -I {} wc -l {} | sort -n | tail -1l | awk '{print $2}' | xargs -I {} cp {} kernel_args_old.csv

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
fi
