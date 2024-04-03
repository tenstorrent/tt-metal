#!/usr/bin/bash

# Just a quick script to run bunch of latest passing paged write tests targeting DRAM and L1

SUFFIX=$(date +%Y%m%d%H%M%S)

# Define the test numbers to iterate over
TEST_IDS=(2 3)

# Loop over the modes
for TEST_ID in ${TEST_IDS[@]}; do

    LABEL=""
    if [ $TEST_ID -eq 2 ]; then
        LABEL="DRAM"
    elif [ $TEST_ID -eq 3 ]; then
        LABEL="L1"
    fi

    DIR="${TT_METAL_HOME}/paged_write_tests_sanity_${LABEL}_${SUFFIX}"
    mkdir -p $DIR

    echo "Running tests for ${DIR} now..."

    #################################################################################
    # SANITY TESTS : CQDispatchWritePagedCmd.page_size = 16 B                       #
    #################################################################################

    # Testcase: Simplest case, 1 page, CQDispatchWritePagedCmd.page_size is 16B, same as dispatch buffer.
    TT_METAL_SLOW_DISPATCH_MODE=1 ${TT_METAL_HOME}/build/test/tt_metal/perf_microbenchmark/dispatch/test_dispatcher -i 1 -w 0 -t $TEST_ID -wx 0 -wy 1 -min 16 -max 16 -lps 4 -pbs 1 -np 1 -c |& tee ${DIR}/write_1_page_16b_size_dispatch_buffer_16b_pages.log

    # Testcase: 128 page, CQDispatchWritePagedCmd.page_size is 16B, same as dispatch buffer.
    TT_METAL_SLOW_DISPATCH_MODE=1 ${TT_METAL_HOME}/build/test/tt_metal/perf_microbenchmark/dispatch/test_dispatcher -i 1 -w 0 -t $TEST_ID -wx 0 -wy 1 -min 16 -max 16 -lps 4 -pbs 1 -np 128 -c |& tee ${DIR}/write_128_page_16b_size_dispatch_buffer_16b_pages.log

    # Testcase: 512 page, CQDispatchWritePagedCmd.page_size is 16B, same as dispatch buffer.
    TT_METAL_SLOW_DISPATCH_MODE=1 ${TT_METAL_HOME}/build/test/tt_metal/perf_microbenchmark/dispatch/test_dispatcher -i 1 -w 0 -t $TEST_ID -wx 0 -wy 1 -min 16 -max 16 -lps 4 -pbs 1 -np 512 -c |& tee ${DIR}/write_512_page_16b_size_dispatch_buffer_16b_pages.log


    #################################################################################
    # VARYING PAGE SIZES : CQDispatchWritePagedCmd.page_size                        #
    #################################################################################

    # Matching page sizes (least interesting)
    #########################################

    # How to make dispatch buffer, prefetch buffer not be required to hold all the data (right now prefetch buffer is required to hold everything) - need to break the data into multiple writes (?)

    # Testcase: 140 Pages, Matching CQDispatchWritePagedCmd.page_size and dispatch buffer page size. Both are 2048 Bytes.
    TT_METAL_SLOW_DISPATCH_MODE=1 ${TT_METAL_HOME}/build/test/tt_metal/perf_microbenchmark/dispatch/test_dispatcher -i 1 -w 0 -t $TEST_ID -wx 0 -wy 1 -min 2048 -max 2048 -lps 12 -pbs 1 -np 140 -c |& tee ${DIR}/write_140_page_2048b_size_dispatch_buffer_2048b_pages.log

    # Testcase: 70 Pages, Matching CQDispatchWritePagedCmd.page_size and dispatch buffer page size. Both are 4096 Bytes.
    TT_METAL_SLOW_DISPATCH_MODE=1 ${TT_METAL_HOME}/build/test/tt_metal/perf_microbenchmark/dispatch/test_dispatcher -i 1 -w 0 -t $TEST_ID -wx 0 -wy 1 -min 4096 -max 4096 -lps 12 -pbs 1 -np 70 -c |& tee ${DIR}/write_70_page_4096b_size_dispatch_buffer_4096b_pages.log


    # Paged write page size is smaller than dispatch buffer  (not too interesting, but good to check)
    #################################################################################################

    # Testcase: 256 Pages, Smaller CQDispatchWritePagedCmd.page_size than dispatch buffer page size. Write page size is 16 Bytes dispatch buffer is 32 Bytes
    TT_METAL_SLOW_DISPATCH_MODE=1 ${TT_METAL_HOME}/build/test/tt_metal/perf_microbenchmark/dispatch/test_dispatcher -i 1 -w 0 -t $TEST_ID -wx 0 -wy 1 -min 16 -max 16 -lps 5 -pbs 1 -np 256 -c |& tee ${DIR}/write_256_page_16b_size_dispatch_buffer_32b_pages.log

    # Testcase: 256 Pages, Smaller CQDispatchWritePagedCmd.page_size than dispatch buffer page size. Write page size is 32 Bytes dispatch buffer is 64 Bytes
    TT_METAL_SLOW_DISPATCH_MODE=1 ${TT_METAL_HOME}/build/test/tt_metal/perf_microbenchmark/dispatch/test_dispatcher -i 1 -w 0 -t $TEST_ID -wx 0 -wy 1 -min 32 -max 32 -lps 6 -pbs 1 -np 256 -c |& tee ${DIR}/write_256_page_32b_size_dispatch_buffer_64b_pages.log

    # Testcase: 256 Pages, Smaller CQDispatchWritePagedCmd.page_size than dispatch buffer page size. Write page size is 496 Bytes dispatch buffer is 4096 Bytes
    TT_METAL_SLOW_DISPATCH_MODE=1 ${TT_METAL_HOME}/build/test/tt_metal/perf_microbenchmark/dispatch/test_dispatcher -i 1 -w 0 -t $TEST_ID -wx 0 -wy 1 -min 496 -max 496 -lps 12 -pbs 1 -np 256 -c |& tee ${DIR}/write_256_page_496b_size_dispatch_buffer_4096b_pages.log


    # Paged write page size is larger than dispatch buffer (more interesting)
    #########################################################################

    # Testcase: 256 Pages, Bigger CQDispatchWritePagedCmd.page_size than dispatch buffer page size. Write page size is 32 Bytes dispatch buffer is 16 Bytes
    TT_METAL_SLOW_DISPATCH_MODE=1 ${TT_METAL_HOME}/build/test/tt_metal/perf_microbenchmark/dispatch/test_dispatcher -i 1 -w 0 -t $TEST_ID -wx 0 -wy 1 -min 32 -max 32 -lps 4 -pbs 1 -np 256 -c |& tee ${DIR}/write_256_page_32b_size_dispatch_buffer_16b_pages.log

    # Testcase: 256 Pages, Bigger CQDispatchWritePagedCmd.page_size than dispatch buffer page size. Write page size is 48 Bytes dispatch buffer is 16 Bytes
    TT_METAL_SLOW_DISPATCH_MODE=1 ${TT_METAL_HOME}/build/test/tt_metal/perf_microbenchmark/dispatch/test_dispatcher -i 1 -w 0 -t $TEST_ID -wx 0 -wy 1 -min 48 -max 48 -lps 4 -pbs 1 -np 256 -c |& tee ${DIR}/write_256_page_48b_size_dispatch_buffer_16b_pages.log

    # Testcase: 256 Pages, Bigger CQDispatchWritePagedCmd.page_size than dispatch buffer page size. Write page size is 2048 Bytes dispatch buffer is 16 Bytes
    TT_METAL_SLOW_DISPATCH_MODE=1 ${TT_METAL_HOME}/build/test/tt_metal/perf_microbenchmark/dispatch/test_dispatcher -i 1 -w 0 -t $TEST_ID -wx 0 -wy 1 -min 2048 -max 2048 -lps 4 -pbs 1 -np 128 -c |& tee ${DIR}/write_128_page_2048b_size_dispatch_buffer_16b_pages.log

    # Testcase: 256 Pages, Bigger CQDispatchWritePagedCmd.page_size than dispatch buffer page size. Write page size is 2048 Bytes dispatch buffer is 1024 Bytes
    TT_METAL_SLOW_DISPATCH_MODE=1 ${TT_METAL_HOME}/build/test/tt_metal/perf_microbenchmark/dispatch/test_dispatcher -i 1 -w 0 -t $TEST_ID -wx 0 -wy 1 -min 2048 -max 2048 -lps 10 -pbs 1 -np 128 -c |& tee ${DIR}/write_128_page_2048b_size_dispatch_buffer_1024b_pages.log


    #################################################################################
    # VARYING BASE_ADDR : CQDispatchWritePagedCmd.base_addr                         #
    #################################################################################

    # Testcase: 256 Pages, Bigger CQDispatchWritePagedCmd.page_size than dispatch buffer page size. Write page size is 48 Bytes dispatch buffer is 16 Bytes. Base addr is 513 KB.
    TT_METAL_SLOW_DISPATCH_MODE=1 ${TT_METAL_HOME}/build/test/tt_metal/perf_microbenchmark/dispatch/test_dispatcher -i 1 -w 0 -t $TEST_ID -wx 0 -wy 1 -min 48 -max 48 -lps 4 -pbs 1 -np 256 -min-addr 525312 -max-addr 525312 -c |& tee ${DIR}/write_256_page_48b_size_dispatch_buffer_16b_pages_525312_base.log

    # Testcase: 256 Pages, Bigger CQDispatchWritePagedCmd.page_size than dispatch buffer page size. Write page size is 48 Bytes dispatch buffer is 16 Bytes. Base addr is 515 KB + 16 bytes.
    TT_METAL_SLOW_DISPATCH_MODE=1 ${TT_METAL_HOME}/build/test/tt_metal/perf_microbenchmark/dispatch/test_dispatcher -i 1 -w 0 -t $TEST_ID -wx 0 -wy 1 -min 48 -max 48 -lps 4 -pbs 1 -np 256 -min-addr 527376 -max-addr 527376 -c |& tee ${DIR}/write_256_page_48b_size_dispatch_buffer_16b_pages_527376_base.log

    # Testcase: 256 Pages, Bigger CQDispatchWritePagedCmd.page_size than dispatch buffer page size. Write page size is 48 Bytes dispatch buffer is 16 Bytes. Base addr is 521 KB + 48 bytes.
    TT_METAL_SLOW_DISPATCH_MODE=1 ${TT_METAL_HOME}/build/test/tt_metal/perf_microbenchmark/dispatch/test_dispatcher -i 1 -w 0 -t $TEST_ID -wx 0 -wy 1 -min 48 -max 48 -lps 4 -pbs 1 -np 256 -min-addr 533552 -max-addr 533552 -c |& tee ${DIR}/write_256_page_48b_size_dispatch_buffer_16b_pages_533552_base.log

    #################################################################################
    # MULTIPLE WRITE CMDS using start_page to write to mem without gaps             #
    #################################################################################

    # Testcase: Increase prefetcher buffer size to 6 pages so that 2 paged write cmds (3 pages each) are needed to fill buffer. End at small even page number.
    TT_METAL_SLOW_DISPATCH_MODE=1 ${TT_METAL_HOME}/build/test/tt_metal/perf_microbenchmark/dispatch/test_dispatcher -i 1 -w 0 -t $TEST_ID -wx 0 -wy 1 -min 16 -max 16 -lps 4 -pbs 6 -np 3 -c |& tee ${DIR}/write_3_page_2x_16b_size_dispatch_buffer_16b_pages_pbs6.log

    # Testcase: Increase prefetcher buffer size to 15 pages so that 5 paged write cmds (3 pages each) are needed to fill buffer. End at small even page number.
    TT_METAL_SLOW_DISPATCH_MODE=1 ${TT_METAL_HOME}/build/test/tt_metal/perf_microbenchmark/dispatch/test_dispatcher -i 1 -w 0 -t $TEST_ID -wx 0 -wy 1 -min 16 -max 16 -lps 4 -pbs 15 -np 3 -c |& tee ${DIR}/write_3_page_5x_16b_size_dispatch_buffer_16b_pages_pbs15.log

    # Testcase: Increase prefetcher buffer size to 280 pages so that 20 paged write cmds (14 pages each) are needed to fill buffer. Wrap around and finish writes on lower banks before starting the next.
    TT_METAL_SLOW_DISPATCH_MODE=1 ${TT_METAL_HOME}/build/test/tt_metal/perf_microbenchmark/dispatch/test_dispatcher -i 1 -w 0 -t $TEST_ID -wx 0 -wy 1 -min 16 -max 16 -lps 4 -pbs 280 -np 14 -c |& tee ${DIR}/write_14_page_20x_16b_size_dispatch_buffer_16b_pages_pbs280.log

    # Testcase: Arbitrary non-even numbers. This caught some test issues with overflowing start_page one test implementation.
    TT_METAL_SLOW_DISPATCH_MODE=1 ${TT_METAL_HOME}/build/test/tt_metal/perf_microbenchmark/dispatch/test_dispatcher -i 1 -w 0 -t $TEST_ID -wx 0 -wy 1 -min 16 -max 16 -lps 5 -pbs 275 -np 13 -c |& tee ${DIR}/write_13_page_21x_16b_size_dispatch_buffer_16b_pages_pbs275.log

    #################################################################################
    # MISC                                                                          #
    #################################################################################

    # Multiple warmup iterations and test iterations
    TT_METAL_SLOW_DISPATCH_MODE=1 ${TT_METAL_HOME}/build/test/tt_metal/perf_microbenchmark/dispatch/test_dispatcher -i 10 -w 10 -t $TEST_ID -wx 0 -wy 1 -min 32 -max 32 -lps 4 -pbs 1 -np 256 -c |& tee ${DIR}/write_256_page_32b_size_dispatch_buffer_16b_pages_10warmup_10iter.log

    # Increase prefetch buffer size.
    TT_METAL_SLOW_DISPATCH_MODE=1 ${TT_METAL_HOME}/build/test/tt_metal/perf_microbenchmark/dispatch/test_dispatcher -i 1 -w 0 -t $TEST_ID -wx 0 -wy 1 -min 32 -max 32 -lps 4 -pbs 2 -np 256 -c |& tee ${DIR}/write_256_page_32b_size_dispatch_buffer_16b_pages_pbs2.log
    TT_METAL_SLOW_DISPATCH_MODE=1 ${TT_METAL_HOME}/build/test/tt_metal/perf_microbenchmark/dispatch/test_dispatcher -i 1 -w 0 -t $TEST_ID -wx 0 -wy 1 -min 64 -max 64 -lps 4 -pbs 4 -np 230 -c |& tee ${DIR}/write_230_page_64b_size_dispatch_buffer_16b_pages_pbs4.log

done


###############################################
# PERF                                        #
###############################################


DIR="TT_METAL_SLOW_DISPATCH_MODE=1 ${TT_METAL_HOME}/paged_write_tests_sanity_perf_dram_${SUFFIX}"
mkdir -p $DIR

# 3.845 GB/s whb0
TT_METAL_SLOW_DISPATCH_MODE=1 ${TT_METAL_HOME}/build/test/tt_metal/perf_microbenchmark/dispatch/test_dispatcher -w 10 -t 2 -wx 0 -wy 1 -min 1024 -max 1024 -lps 10 -pbs 2 -np 128 -c -i 1 -pi 10000 & tee ${DIR}/perf_write_128_page_1024b_size_dispatch_buffer_1024b_pages_10000_iter_dram_pbs2.log

# 6.374 GB/s whb0
TT_METAL_SLOW_DISPATCH_MODE=1 ${TT_METAL_HOME}/build/test/tt_metal/perf_microbenchmark/dispatch/test_dispatcher -w 10 -t 2 -wx 0 -wy 1 -min 2048 -max 2048 -lps 11 -pbs 2 -np 128 -c -i 1 -pi 10000 |& tee ${DIR}/perf_write_128_page_2048b_size_dispatch_buffer_2048b_pages_10000_iter_dram_pbs2.log

# 9.600 GB/s whb0
TT_METAL_SLOW_DISPATCH_MODE=1 ${TT_METAL_HOME}/build/test/tt_metal/perf_microbenchmark/dispatch/test_dispatcher -w 10 -t 2 -wx 0 -wy 1 -min 4096 -max 4096 -lps 12 -pbs 2 -np 128 -c -i 1 -pi 10000 |& tee ${DIR}/perf_write_128_page_4096b_size_dispatch_buffer_4096b_pages_10000_iter_dram_pbs2.log

# 11.872 GB/s whb0 - reduced number of pages per block in half otherwise uses 1536 KB L1 (exceeds for GS, WH)
TT_METAL_SLOW_DISPATCH_MODE=1 ${TT_METAL_HOME}/build/test/tt_metal/perf_microbenchmark/dispatch/test_dispatcher -w 10 -t 2 -wx 0 -wy 1 -min 8192 -max 8192 -lps 13 -pbs 2 -np 128 -c -i 1 -pi 5000 -bs 24 |& tee ${DIR}/perf_write_128_page_8192b_size_dispatch_buffer_8192b_pages_10000_iter_dram_pbs2.log
