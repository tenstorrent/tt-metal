#!/bin/bash

  START_ID=-2
  LOG_DIR=/localdev/$USER
  ERROR_FILE=$LOG_DIR/error_log
  STDOUT_FILE=$LOG_DIR/stdout_log
  TIME_LIMIT=100

  run_bw_tests() {
     echo "DRAM READ TEST"
     build_Release/test/tt_metal/perf_microbenchmark/dispatch/test_bw_and_latency -bs 256 -p 4096 -m 3 # read dram

     echo "L1 FAR READ TEST"
     build_Release/test/tt_metal/perf_microbenchmark/dispatch/test_bw_and_latency -bs 256 -p 4096 -m 2 -rx 0 -ry 0 -sx 6 -sy 5 # read far l1

     echo "MCAST TO GRID TEST"
     build_Release/test/tt_metal/perf_microbenchmark/dispatch/test_bw_and_latency -bs 256 -p 4096 -m 6 -rx 0 -ry 0 -sx 0 -sy 1 -tx 12 -ty 9 # mcast to grid
  }

  run_test_in_loop() {

     export ERR_FILE_PATH=$ERROR_FILE

     #The script will add 2 and start from the counter.
     #Script the one after the last successful one.
     echo $START_ID  > $ERROR_FILE

#     while true
#     do
	rm -rf built
        timeout $TIME_LIMIT tt-smi -r 0 || exit 1
#        TT_METAL_DPRINT_CORES=worker TT_METAL_SLOW_DISPATCH_MODE=1 pytest tests/ttnn/unit_tests/benchmarks/test_benchmark.py::test_matmul_2d_host_perf
        TT_METAL_SLOW_DISPATCH_MODE=1 pytest tests/ttnn/unit_tests/benchmarks/test_benchmark.py::test_matmul_2d_host_perf
#     done

  }

  run_test_in_loop > $STDOUT_FILE
