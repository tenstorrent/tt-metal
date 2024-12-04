#/bin/bash

# Run this test w/
#   sweep_bw_and_latency.sh 2>&1 | tee log
# then run
# filt_pgm_dispatch.pl log
# then paste the results into the BW spreadsheet

if [ "$ARCH_NAME" = "grayskull" ]; then
    echo "Configured core range for grayskull"
    max_x=11
    max_y=8
elif [ "$ARCH_NAME" = "wormhole_b0" ]; then
    echo "Configured core range for wormhole_b0"
    max_x=7
    max_y=6
elif [ "$ARCH_NAME" = "blackhole" ]; then
    echo "Configured core range for blackhole"
    max_x=12
    max_y=9
else
    echo "Unknown arch: $ARCH_NAME"
    exit 1
fi

function get_half_way_away_core_x() {
    half_way_away_core_x=$(( ($1 + (($max_x + 1) / 2)) % ($max_x + 1) ))
    echo $half_way_away_core_x
}

function get_half_way_away_core_y() {
    half_way_away_core_y=$(( ($1 + (($max_y + 1) / 2)) % ($max_y + 1) ))
    echo $half_way_away_core_y
}

hx=$(get_half_way_away_core_x 0);
hy=$(get_half_way_away_core_y 0);

function run_one() {
    echo "Running $@"
    build/test/tt_metal/perf_microbenchmark/dispatch/test_bw_and_latency $@
}

function bw_test() {
    run_one -bs 8 -p 16 $@
    run_one -bs 8 -p 32 $@
    run_one -bs 16 -p 64 $@
    run_one -bs 32 -p 128 $@
    run_one -bs 64 -p 256 $@
    run_one -bs 128 -p 512 $@
    run_one -bs 256 -p 1024 $@
    run_one -bs 256 -p 2048 $@
    run_one -bs 256 -p 4096 $@
    run_one -bs 256 -p 8192 $@
    run_one -bs 256 -p 16384 $@
    run_one -bs 256 -p 32768 $@
    run_one -bs 256 -p 65536 $@
}

function latency_test() {
    run_one -bs 8 -p 16 -l $@
}

echo "###" read pcie
bw_test "-m 0"
latency_test "-m 0"

echo "###" read dram
bw_test "-m 1"
latency_test "-m 1"

echo "###" read drams
bw_test "-m 3"
latency_test "-m 3"

echo "###" read l1 adjacent
bw_test "-m 2"
latency_test "-m 2"

echo "###" read l1 far halfway away
bw_test "-m 2  -rx 0 -ry 0 -sx $hx -sy $hy"
latency_test "-m 2 -rx 0 -ry 0 -sx $hx -sy $hy"

echo "###" read local
bw_test "-m 2 -rx 0"
latency_test "-m 2 -rx 0"

echo "###" write l1 far halfway away
bw_test "-m 2 -rx 0 -ry 0 -sx $hx -sy $hy -wr"
latency_test "-m 2 -rx 0 -ry 0 -sx $hx -sy $hy -wr"

echo "###" mcast write to adjacent
bw_test "-m 6 -rx 0 -ry 0 -sx 1 -sy 0 -tx 1 -ty 0"
latency_test "-m 6 -rx 0 -ry 0 -sx 1 -sy 0 -tx 1 -ty 0"

echo "###" mcast write to halfway away
bw_test "-m 6 -rx 0 -ry 0 -sx $hx -sy $hy -tx $hx -ty $hy"
latency_test "-m 6 -rx 0 -ry 0 -sx $hx -sy $hy -tx $hx -ty $hy"

echo "###" mcast write to all
bw_test "-m 6 -rx 0 -ry 0 -sx 0 -sy 1 -tx $max_x -ty $max_y"
latency_test "-m 6 -rx 0 -ry 0 -sx 0 -sy 1 -tx $max_x -ty $max_y"

echo "###" mcast write to all, linked
bw_test "-m 6 -rx 0 -ry 0 -sx 0 -sy 1 -tx $max_x -ty $max_y -link"
latency_test "-m 6 -rx 0 -ry 0 -sx 0 -sy 1 -tx $max_x -ty $max_y -link"

echo "###" done
