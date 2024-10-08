#!/bin/bash

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

function read_from_half_way_away_core() {
    half_way_away_core_x=$(get_half_way_away_core_x $1)
    half_way_away_core_y=$(get_half_way_away_core_y $2)
    echo "./build/test/tt_metal/perf_microbenchmark/dispatch/test_bw_and_latency -m 2 -rx $1 -ry $2 -sx $half_way_away_core_x -sy $half_way_away_core_y"
    ./build/test/tt_metal/perf_microbenchmark/dispatch/test_bw_and_latency -m 2 -rx $1 -ry $2 -sx $half_way_away_core_x -sy $half_way_away_core_y
}

function mcast_write_to_half_way_away_core() {
    half_way_away_core_x=$(get_half_way_away_core_x $1)
    half_way_away_core_y=$(get_half_way_away_core_y $2)
    echo "./build/test/tt_metal/perf_microbenchmark/dispatch/test_bw_and_latency -m 6 -rx $1 -ry $2 -sx $half_way_away_core_x -sy $half_way_away_core_y -tx $half_way_away_core_x -ty $half_way_away_core_y"
    ./build/test/tt_metal/perf_microbenchmark/dispatch/test_bw_and_latency -m 6 -rx $1 -ry $2 -sx $half_way_away_core_x -sy $half_way_away_core_y -tx $half_way_away_core_x -ty $half_way_away_core_y
}

function mcast_write_to_adjacent_core() {
    adj_core_y=$(($2 + 1))
    if [ $adj_core_y -gt $max_y ]; then
        adj_core_y=$(($2 - 1))
    fi
    echo "./build/test/tt_metal/perf_microbenchmark/dispatch/test_bw_and_latency -m 6 -rx $1 -ry $2 -sx $1 -sy $adj_core_y -tx $1 -ty $adj_core_y"
    ./build/test/tt_metal/perf_microbenchmark/dispatch/test_bw_and_latency -m 6 -rx $1 -ry $2 -sx $1 -sy $adj_core_y -tx $1 -ty $adj_core_y
}

function mcast_write_from_core_after_curr_core_to_half_way_away_core() {
    half_way_away_core_x=$(get_half_way_away_core_x $1)
    half_way_away_core_y=$(get_half_way_away_core_y $2)
    mcast_start_y=$(($2 + 1))
    echo "./build/test/tt_metal/perf_microbenchmark/dispatch/test_bw_and_latency -m 6 -rx $1 -ry $2 -sx $1 -sy $mcast_start_y -tx $half_way_away_core_x -ty $half_way_away_core_y"
    ./build/test/tt_metal/perf_microbenchmark/dispatch/test_bw_and_latency -m 6 -rx $1 -ry $2 -sx $1 -sy $mcast_start_y -tx $half_way_away_core_x -ty $half_way_away_core_y
}

for ((x=0; x<=max_x; x++)); do
    for ((y=0; y<=max_y; y++)); do
        read_from_half_way_away_core $x $y
        mcast_write_to_half_way_away_core $x $y
        mcast_write_to_adjacent_core $x $y
        mcast_write_from_core_after_curr_core_to_half_way_away_core $x $y

        if [ $y -eq 0 ]; then
            mcast_start_y=$(($y + 1))
            echo "./build/test/tt_metal/perf_microbenchmark/dispatch/test_bw_and_latency -m 6 -rx $x -ry $y -sx 0 -sy $mcast_start_y -tx $max_x -ty $max_y"
            ./build/test/tt_metal/perf_microbenchmark/dispatch/test_bw_and_latency -m 6 -rx $x -ry $y -sx 0 -sy $mcast_start_y -tx $max_x -ty $max_y
        fi

        if [ $y -eq $max_y ]; then
            mcast_end_y=$(($y - 1))
            echo "./build/test/tt_metal/perf_microbenchmark/dispatch/test_bw_and_latency -m 6 -rx $x -ry $y -sx 0 -sy 0 -tx $max_x -ty $mcast_end_y"
            ./build/test/tt_metal/perf_microbenchmark/dispatch/test_bw_and_latency -m 6 -rx $x -ry $y -sx 0 -sy 0 -tx $max_x -ty $mcast_end_y
        fi
    done
done
