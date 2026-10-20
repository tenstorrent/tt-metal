#!/bin/bash
# Runs each negative-test yaml on the branch build and checks that it fails with the expected message.
# Resets all chips after each case, since a build-time error can leave the fabric initialized.
set -u
here=$(cd "$(dirname "$0")" && pwd)
export TT_METAL_HOME=/home/aagarwal/tt-metal-streams TT_METAL_CLEAR_L1=1 TT_METAL_DEVICE_PROFILER=0 TT_FABRIC_NO_BW_TELEMETRY=1
bin=./build/test/tt_metal/tt_fabric/test_infra/test_tt_fabric
cd "$TT_METAL_HOME"  # the runtime finds its root directory from here
declare -A expect=(
    [num_packets_in_defaults]="a stream pattern can't have num_packets"
    [extra_pattern]="a sender with a stream can't have other patterns"
    [all_to_all]="a sender with a stream can't have other patterns"
    [multi_dir_mcast]="multicast hops must name one direction"
    [benchmark_flow_control]="can't combine benchmark mode with enable_flow_control"
    [benchmark_shared_router]="use the same router"
    [latency_mode]="streams don't support latency_test_mode"
    [segment_ftype]="Unsupported stream segment field 'ftype'"
    [repeat_zero]="'repeat' must be at least 1"
    [write_atomic_val]="sets atomic_inc_val but its ntype sends no atomic increment"
    [empty_stream]="'stream' needs at least one segment"
    [count_zero]="'count' must be at least 1"
    [too_many_segments]="at most 64 are supported"
)
pass=0; fail=0
for name in "${!expect[@]}"; do
    log=$here/$name.log
    timeout 180 "$bin" --test_config "$here/$name.yaml" > "$log" 2>&1
    if grep -qF "${expect[$name]}" "$log"; then
        echo "PASS $name"; pass=$((pass + 1))
    else
        echo "FAIL $name: expected '${expect[$name]}'"; grep -E "TT_FATAL|TT_THROW|executed" "$log" | head -2 | cut -c1-200
        fail=$((fail + 1))
    fi
    tt-smi -r all > /dev/null 2>&1
done
echo "negative tests: $pass passed, $fail failed"
