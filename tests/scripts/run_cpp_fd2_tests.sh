#!/bin/bash

set -eo pipefail

run_test() {
    echo $1
    $1
    echo
};

run_test_with_watcher() {
    echo $1
    TT_METAL_WATCHER=1 TT_METAL_WATCHER_NOINLINE=1 $1
    echo
};
# Unset the variable for these tests
(
    unset TT_METAL_SLOW_DISPATCH_MODE
    #############################################
    # TEST_PREFETCHER TESTS                     #
    #############################################
    echo "Running test_prefetcher with fast dispatch mode..";

    run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_prefetcher"

    echo "Running test_prefetcher with split prefetcher..";

    run_test "env TT_METAL_SPLIT_PREFETCHER=1 ./build/test/tt_metal/perf_microbenchmark/dispatch/test_prefetcher --gtest_filter=*SplitPrefetcherJitCompile*"

    #############################################
    # TEST_DISPATCHER TESTS                     #
    #############################################
    echo "Running test_dispatcher with fast dispatch mode..";

    run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_dispatcher"
)
