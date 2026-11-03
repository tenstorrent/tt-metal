#!/bin/bash
# Smoke test of the /data trees on this host (before Galaxy jobs): one short 2D 1-hop row from each tree, run exactly
# as the Galaxy sbatch files run it (TT_METAL_HOME and LD_LIBRARY_PATH into the tree, a fresh kernel cache).
# Trees: the arguments (default base v3q). base* trees run main's router (kernel v1, flags 0, firmware keepalive); the
# others the v2 slot at flags 13129088 with the study keepalive (1000) and -O3, as frs_glx_rsz.sbatch.
# SAMPLE=1: then a longer row (fabric_exp/pcs3/ew4352.yaml) with the tree's pc_sampler.py, run from the tree's own
# python_env beside the test, as frs_glx_rsz.sbatch runs it (samples: fabric_exp/results/glxsmoke_<tree>_samples.*).
G=/data/aagarwal/frs-glx
S=/home/aagarwal/tt-metal
Y=$S/fabric_exp/pcs3/compile2d.yaml
YS=$S/fabric_exp/pcs3/ew4352.yaml
(while sleep 60; do echo "heartbeat $(date +%T)"; done) & HB=$!
trap 'kill $HB 2>/dev/null' EXIT
rc_all=0
run1() {  # tree yaml log env...
    local tree=$1 yaml=$2 log=$3 lib=$1/build_Release; shift 3
    (cd $tree && env -u TT_METAL_SLOW_DISPATCH_MODE -u TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT TT_METAL_HOME=$tree \
        LD_LIBRARY_PATH=$lib/tt_metal:$lib/tt_stl:$lib/lib:$lib/tt_metal/third_party/umd/lib \
        TT_METAL_CACHE=$c TT_METAL_CLEAR_L1=1 TT_METAL_DEVICE_PROFILER=0 TT_FABRIC_NO_BW_TELEMETRY=1 "$@" \
        timeout 600 ./build/test/tt_metal/tt_fabric/test_infra/test_tt_fabric --test_config $yaml > $log 2>&1)
}
for t in ${@:-base v3q}; do
    case $t in
        base*) env=(TT_FABRIC_KERNEL_VERSION=1 TT_FABRIC_EXP_FLAGS=0) ;;
        *) env=(TT_FABRIC_KERNEL_VERSION=2 TT_FABRIC_EXP_FLAGS=13129088 TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=1000 TT_METAL_FABRIC_OPT_LEVEL=O3) ;;
    esac
    tree=$G/trees/$t; c=/home/aagarwal/.tt-metal-cache-glxsmoke-$t
    rm -rf $c; mkdir -p $c
    log=$S/fabric_exp/results/glxsmoke_${t}_compile.log
    echo "=== $t: ${env[*]}"
    run1 $tree $Y $log "${env[@]}"; rc=$?
    sed 's/\x1b\[[0-9;]*m//g' $log | grep -aE 'Tests: |FATAL|TT_THROW' | head -5
    echo "   exit $rc"; [ $rc = 0 ] || rc_all=1
    f=$tree/generated/fabric/bandwidth_summary_results_blackhole.csv
    [ -f $f ] && awk -F, 'NR>1{printf "   %s %s B %.1f cycles/packet\n", $1, $7, 1.35e9/$10}' $f
    rm -f $f
    if [ "${SAMPLE:-0}" = 1 ]; then
        log=$S/fabric_exp/results/glxsmoke_${t}_sampled.log
        out=$S/fabric_exp/results/glxsmoke_${t}_samples
        rm -f $out.bin $out.json
        run1 $tree $YS $log "${env[@]}" & pid=$!
        timeout 400 $tree/python_env/bin/python $tree/fabric_exp/consolidation/pc_sampler.py $out --start-log $log \
            --start-delay-s 3 --max-s 25 --wait-s 300 --eriscs 0,1 2>&1 | grep -E '^sampler: |Error|Traceback'
        wait $pid; rc=$?
        sed 's/\x1b\[[0-9;]*m//g' $log | grep -aE 'Tests: |FATAL|TT_THROW' | head -5
        echo "   sampled row: exit $rc"; [ $rc = 0 ] || rc_all=1
        [ -f $f ] && awk -F, 'NR>1{printf "   %s %s B %.1f cycles/packet (sampled)\n", $1, $7, 1.35e9/$10}' $f
    fi
    rm -rf $tree/generated
done
[ $rc_all = 0 ] && echo ALL_DONE || { echo "SMOKE FAILED"; exit 1; }
