#/bin/bash

# Run this test w/
#   sweep_fw_buffer.sh 2>&1 | tee log
# then run
# filt_pgm_dispatch.pl log
# then paste the results into the H2D & D2H spreadsheet in appropriate columns

function run_one() {
    echo "Running $@"
    build/test/tt_metal/perf_microbenchmark/3_pcie_transfer/test_rw_buffer_${ARCH_NAME} $@
}

function run_set() {
    run_one $@ --page-size 32
    run_one $@ --page-size 64
    run_one $@ --page-size 128
    run_one $@ --page-size 256
    run_one $@ --page-size 512
    run_one $@ --page-size 1024
    run_one $@ --page-size 2048
    run_one $@ --page-size 4096
    run_one $@ --page-size 8192
    run_one $@ --page-size 16384
    run_one $@ --page-size 32768
}

echo "###" read bw 32K buffer
run_set --transfer-size 32768 --skip-write

echo "###" read bw 512M buffer
run_set --transfer-size 536870912 --skip-write

echo "###" write bw 32K buffer
run_set --transfer-size 32768 --skip-read

echo "###" write bw 512M buffer
run_set --transfer-size 536870912 --skip-read

echo "###" read bw 32K buffer remote device
run_set --transfer-size 32768 --skip-write --device 1

echo "###" read bw 512M buffer remote device
run_set --transfer-size 536870912 --skip-write --device 1

echo "###" write bw 32K buffer remote device
run_set --transfer-size 32768 --skip-read --device 1

echo "###" write bw 512M buffer remote device
run_set --transfer-size 536870912 --skip-read --device 1

echo "###" done
