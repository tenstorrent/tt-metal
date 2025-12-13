#!/bin/bash

set -eo pipefail

if [[ -z "$TT_METAL_HOME" ]]; then
    echo "Must provide TT_METAL_HOME in environment" 1>&2
    exit 1
fi

if [[ -z "$TT_METAL_SLOW_DISPATCH_MODE" ]]; then
    echo "Must provide TT_METAL_SLOW_DISPATCH_MODE in environment" 1>&2
    exit 1
fi

if [[ -z "$ARCH_NAME" ]]; then
    echo "Must provide ARCH_NAME in environment" 1>&2
    exit 1
fi

export TT_METAL_CLEAR_L1=1

# Not super obvious which test is which during runtime unless you count, so occasionally sprinkle echo statements
# to make it easier to see where we are.

#############################################
# TEST_PREFETCHER TESTS                     #
#############################################
echo "Running test_prefetcher tests now...";

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

run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_prefetcher -t 0 -i 5"  # TrueSmoke Test
run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_prefetcher -t 1 -i 5"  # Smoke Test
run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_prefetcher -t 2 -i 5"  # Random Test
run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_prefetcher -t 2 -i 5 -b" # Random Test, big data
run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_prefetcher -t 3 -i 5"  # PCIE Test
run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_prefetcher -t 4 -i 5"  # Paged DRAM Read Test
run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_prefetcher -t 5 -i 5"  # Paged DRAM Write + Read Test
run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_prefetcher -t 6 -i 5"  # Host Test
run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_prefetcher -t 7 -i 5"  # Packed Read Test
run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_prefetcher -t 8 -i 5"  # Ringbuffer Test
run_test_with_watcher "./build/test/tt_metal/perf_microbenchmark/dispatch/test_prefetcher -t 7 -i 10 -x -mpps" # Packed Read Test w/ max num subcmds
run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_prefetcher -t 1 -i 1000 -rb"  # Smoke Test
run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_prefetcher -t 2 -i 1000 -rb"  # Random Test

echo "split pre/dis tests"
run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_prefetcher -t 0 -i 5 -spre -sdis" # TrueSmoke Test
run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_prefetcher -t 1 -i 5 -spre -sdis" # Smoke Test
run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_prefetcher -t 2 -i 5 -spre -sdis" # Random Test
run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_prefetcher -t 3 -i 5 -spre -sdis" # PCIE Test
run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_prefetcher -t 4 -i 5 -spre -sdis" # Paged DRAM Read Test
run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_prefetcher -t 5 -i 5 -spre -sdis" # Paged DRAM Write + Read Test
run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_prefetcher -t 6 -i 5 -spre -sdis" # Host Test
run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_prefetcher -t 7 -i 5 -spre -sdis" # Packed Read Test
run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_prefetcher -t 8 -i 5 -spre -sdis"  # Ringbuffer Test
run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_prefetcher -t 9 -i 5 -spre" # Raw Copy Test
run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_prefetcher -t 9 -i 5 -spre -sdis" # Raw Copy Test
run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_prefetcher -t 1 -i 1000 -rb -spre -sdis"  # Smoke Test
run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_prefetcher -t 2 -i 1000 -rb -spre -sdis"  # Random Test

echo "exec_buf tests"
run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_prefetcher -t 0 -i 5 -x" # TrueSmoke Test
run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_prefetcher -t 1 -i 5 -x" # Smoke Test
run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_prefetcher -t 2 -i 5 -x" # Random Test
run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_prefetcher -t 3 -i 5 -x" # PCIE Test
run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_prefetcher -t 4 -i 5 -x" # Paged DRAM Read Test
run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_prefetcher -t 5 -i 5 -x" # Paged DRAM Write + Read Test
run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_prefetcher -t 6 -i 5 -x" # Host Test
run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_prefetcher -t 7 -i 5 -x" # Packed Read Test
run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_prefetcher -t 8 -i 5 -x" # Rinbuffer Test

run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_prefetcher -t 1 -i 5 -x -spre" # Smoke Test
run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_prefetcher -t 1 -i 5 -x -spre -sdis" # Smoke Test
run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_prefetcher -t 2 -i 5 -x -spre -sdis" # Random Test
run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_prefetcher -t 6 -i 5 -x -spre -sdis" # Host Test
run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_prefetcher -t 7 -i 5 -x -spre -sdis" # Packed Read Test
run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_prefetcher -t 1 -i 1000 -x -rb -spre -sdis"  # Smoke Test
run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_prefetcher -t 2 -i 1000 -x -rb -spre -sdis"  # Random Test

# Testcase: Paged Write Cmd to DRAM. 256 pages, 256b size.
./build/test/tt_metal/perf_microbenchmark/dispatch/test_prefetcher -t 4 -i 1 -dpgs 256 -dpgr 256
# Testcase: Paged Write Cmd to DRAM. 120 pages, 64b size.
./build/test/tt_metal/perf_microbenchmark/dispatch/test_prefetcher -t 4 -i 1 -dpgs 64 -dpgr 120

#############################################
# TEST_DISPATCHER TESTS                     #
#############################################
echo "Running test_dispatcher with fast dispatch mode (unsetting TT_METAL_SLOW_DISPATCH_MODE)..";

# Unset the variable for these tests
(
    #This is temporary until we refactor test_prefetcher.cpp to use FDMeshCommandQueue
    unset TT_METAL_SLOW_DISPATCH_MODE
    run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_dispatcher"
)
