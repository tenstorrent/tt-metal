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

run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_prefetcher -t 1 -i 5 -x -spre" # Smoke Test
run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_prefetcher -t 1 -i 5 -x -spre -sdis" # Smoke Test
run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_prefetcher -t 2 -i 5 -x -spre -sdis" # Random Test
run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_prefetcher -t 6 -i 5 -x -spre -sdis" # Host Test
run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_prefetcher -t 7 -i 5 -x -spre -sdis" # Packed Read Test
run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_prefetcher -t 1 -i 1000 -x -rb -spre -sdis"  # Smoke Test
run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_prefetcher -t 2 -i 1000 -x -rb -spre -sdis"  # Random Test

run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_prefetcher -t 0 -i 5 -spre -sdis -packetized_en" # TrueSmoke Test with packetized path
run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_prefetcher -t 1 -i 5 -spre -sdis -packetized_en" # Smoke Test with packetized path
run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_prefetcher -t 2 -i 5 -spre -sdis -packetized_en" # Random Test with packetized path
run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_prefetcher -t 6 -i 5 -spre -sdis -packetized_en" # Host Test with packetized path
run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_prefetcher -t 7 -i 5 -spre -sdis -packetized_en" # Packed Read Test
run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_prefetcher -t 1 -i 1000 -rb -spre -sdis -packetized_en"  # Smoke Test
run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_prefetcher -t 2 -i 1000 -rb -spre -sdis -packetized_en"  # Random Test

run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_prefetcher -t 0 -i 5 -x -spre -sdis -packetized_en" # TrueSmoke Test with packetized path+exec
run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_prefetcher -t 1 -i 5 -x -spre -sdis -packetized_en" # Smoke Test with packetized path+exec
run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_prefetcher -t 2 -i 5 -x -spre -sdis -packetized_en" # Random Test with packetized path+exec
run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_prefetcher -t 6 -i 5 -x -spre -sdis -packetized_en" # Host Test with packetized path+exec
run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_prefetcher -t 7 -i 5 -x -spre -sdis -packetized_en" # Packed Read Test+exec
run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_prefetcher -t 1 -i 1000 -x -rb -spre -sdis -packetized_en"  # Smoke Test
run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_prefetcher -t 2 -i 1000 -x -rb -spre -sdis -packetized_en"  # Random Test

# Testcase: Paged Write Cmd to DRAM. 256 pages, 256b size.
./build/test/tt_metal/perf_microbenchmark/dispatch/test_prefetcher -t 4 -i 1 -dpgs 256 -dpgr 256
# Testcase: Paged Write Cmd to DRAM. 120 pages, 64b size.
./build/test/tt_metal/perf_microbenchmark/dispatch/test_prefetcher -t 4 -i 1 -dpgs 64 -dpgr 120

#############################################
# TEST_DISPATCHER TESTS                     #
#############################################
echo "Running test_dispatcher tests now...";

# Linear Write (Unicast)
run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_dispatcher -i 3 -w 5 -t 0 -min 256 -max 256 -wx 0 -wy 1"
run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_dispatcher -i 3 -w 5 -t 0 -min 1024 -max 1024 -wx 0 -wy 1"

# Linear Write (Multicast)
run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_dispatcher -i 3 -w 5 -t 1 -min 256 -max 256"
run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_dispatcher -i 3 -w 5 -t 1 -min 1024 -max 1024"

# Paged Write CMD (L1/DRAM)
# Testcase: 512 page, CQDispatchWritePagedCmd.page_size is 16B, same as dispatch buffer.
run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_dispatcher -i 1 -w 0 -t 2 -min 16 -max 16 -lps 4 -pbs 1 -np 512"
run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_dispatcher -i 1 -w 0 -t 3 -min 16 -max 16 -lps 4 -pbs 1 -np 512"
# Testcase: 256 Pages, Bigger CQDispatchWritePagedCmd.page_size than dispatch buffer page size. Write page size is 2048 Bytes dispatch buffer is 1024 Bytes
run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_dispatcher -i 1 -w 0 -t 2 -min 2048 -max 2048 -lps 10 -pbs 1 -np 128"
run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_dispatcher -i 1 -w 0 -t 3 -min 2048 -max 2048 -lps 10 -pbs 1 -np 128"
# Testcase: 4128 page size (not aligned to 4KB transfer page size)
run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_dispatcher -i 1 -w 0 -t 2 -wx 0 -wy 1 -min 4128 -max 4128 -lps 12 -pbs 1 -np 10 -c"
# Testcase: Arbitrary non-even numbers. This caught some test issues with overflowing start_page one test implementation.
run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_dispatcher -i 1 -w 0 -t 2 -min 16 -max 16 -lps 5 -pbs 275 -np 13"
run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_dispatcher -i 1 -w 0 -t 3 -min 16 -max 16 -lps 5 -pbs 275 -np 13"
# 11.885 GB/s whb0 - DRAM.   Have to reduce number of pages to not exceed 1MB L1 for GS. Also, number of pages per block.
run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_dispatcher -w 10 -t 2 -min 8192 -max 8192 -lps 13 -pbs 2 -np 100 -i 1 -pi 5000 -bs 24"

# Packed Write
run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_dispatcher -i 3 -w 5 -t 4 -min 256 -max 256"
run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_dispatcher -i 3 -w 5 -t 4 -min 1024 -max 1024"

# Packed Write Large
run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_dispatcher -i 1 -w 5 -t 5 -min 1024 -max 1024"
run_test "./build/test/tt_metal/perf_microbenchmark/dispatch/test_dispatcher -i 10 -w 5 -t 5"

if [[ $ARCH_NAME == "wormhole_b0" ]]; then
    #############################################
    # PACKETIZED PATH TESTS - WH only           #
    #############################################
    echo "Running packetized path tests now...";
    # 4 TX -> 4:1 Mux -> 1:4 Demux -> 4 RX
    run_test "./build/test/tt_metal/perf_microbenchmark/routing/test_mux_demux"

    # 16 TX -> 4 x 4:1 Mux -> 4:1 Mux -> 1:4 Demux -> 4 x 1:4 Demux -> 16 RX
    TT_METAL_THREADCOUNT=64 ./build/test/tt_metal/perf_microbenchmark/routing/test_mux_demux_2level
fi
