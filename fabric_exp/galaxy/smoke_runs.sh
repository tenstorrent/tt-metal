#!/bin/bash
# Smoke test of the /data trees on this host (before Galaxy jobs): one short 2D 1-hop row from each tree, run exactly
# as frs_glx.sbatch runs it (TT_METAL_HOME and LD_LIBRARY_PATH into the tree, a fresh kernel cache).
G=/data/aagarwal/frs-glx
Y=/home/aagarwal/tt-metal/fabric_exp/pcs3/compile2d.yaml
(while sleep 60; do echo "heartbeat $(date +%T)"; done) & HB=$!
trap 'kill $HB 2>/dev/null' EXIT
for cfg in "base 1 0" "v3q 2 13129088"; do
    set -- $cfg
    tree=$G/trees/$1; lib=$tree/build_Release; c=/home/aagarwal/.tt-metal-cache-glxsmoke-$1
    rm -rf $c; mkdir -p $c
    echo "=== $1: kernel v$2"
    (cd $tree && env -u TT_METAL_SLOW_DISPATCH_MODE TT_METAL_HOME=$tree \
        LD_LIBRARY_PATH=$lib/tt_metal:$lib/tt_stl:$lib/lib:$lib/tt_metal/third_party/umd/lib \
        TT_METAL_CACHE=$c TT_METAL_CLEAR_L1=1 TT_METAL_DEVICE_PROFILER=0 TT_FABRIC_NO_BW_TELEMETRY=1 \
        TT_FABRIC_KERNEL_VERSION=$2 TT_FABRIC_EXP_FLAGS=$3 \
        timeout 600 ./build/test/tt_metal/tt_fabric/test_infra/test_tt_fabric --test_config $Y 2>&1 \
        | sed 's/\x1b\[[0-9;]*m//g' | grep --line-buffered -aE 'Tests: |FATAL|error|TT_THROW')
    f=$tree/generated/fabric/bandwidth_summary_results_blackhole.csv
    [ -f $f ] && awk -F, 'NR>1{printf "   %s %.1f cycles/packet\n", $1, 1.35e9/$10}' $f
    rm -rf $tree/generated
done
echo ALL_DONE
