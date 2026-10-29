#!/bin/bash
# v4se single-ERISC bring-up on Blackhole (2026-10-28; V4_SINGLE_ERISC.md: functional on Blackhole, tuned on Wormhole).
# v4se = v4gi_m with the role loop bodies as sender::pass / receiver::pass and router/router_single.hpp: one ERISC runs
# both passes, receiver first, no command-buffer presets, counter credits on the one TX queue (host: the credit plan
# gives the study router counters in every mode). Per MODE given, fresh caches, depths logged, hang capture armed:
#   aer1  TT_METAL_DISABLE_MULTI_AERISC=1 (router on ERISC 0, CI's single-ERISC mode)
#   fab1  TT_METAL_DISABLE_FABRIC_TWO_ERISC=1 (router on ERISC 1)
# mesh sanity, 1D validation (v4's val1d_m1), ring and Y-torus validation, the 2D 4352 B and 1D benchmarks, then
# fabric_unit_tests --gtest_filter="Fabric1DFixture.*" with the study router (the standalone mux fixture runs in
# mux4_runs.sh; the Tensix fixtures need the fabric's Tensix mux mode, not in v4 yet). A hang stops the job; failures
# do not. Labels s1<mode>_v4se_*.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/cap_lib.sh
source $CON/vc1_capture_lib.sh
[ $# -gt 0 ] || { echo "usage: se4_runs.sh MODE..."; exit 1; }
unset TT_FABRIC_EXP_SLOTS_VC1 TT_FABRIC_EXP_SLOTS TT_FABRIC_EXP_SLOTS_2D TT_FABRIC_EXP_SLOTS_1D TT_FABRIC_EXP_TIER_SLOTS
L="TT_FABRIC_EXP_LOG_SLOTS=1"
declare -A MODE_ENV=([aer1]="TT_METAL_DISABLE_MULTI_AERISC=1" [fab1]="TT_METAL_DISABLE_FABRIC_TWO_ERISC=1")
procs() {  # log: which ERISCs the router binaries were built for
    sed 's/\x1b\[[0-9;]*m//g' $1 | grep -ao 'kernels/fabric_erisc_router_v2/[0-9]*/[a-z_]*erisc' | awk -F/ '{print $NF}' \
        | sort | uniq -c | tr '\n' ' '
}
python3 $CON/swap.py apply fabric_exp/variants/v4se $BK || { echo STOPPED; exit 1; }
stop() { python3 $CON/swap.py restore $BK; echo STOPPED; exit 1; }
for mode in "$@"; do
    [ -n "${MODE_ENV[$mode]+x}" ] || { echo "unknown mode $mode"; stop; }
    E=${MODE_ENV[$mode]}
    echo "##### mode $mode: $E (job at ${SECONDS} s)"
    C=/home/aagarwal/.tt-metal-cache-s1-$mode
    rm -rf "${C:?}"; mkdir -p $C
    run_cap s1${mode}_v4se_13129088_mesh $(suite_yaml mesh) $C 300 $L $E; [ $? = 2 ] && stop
    echo "   router binaries: $(procs $out/s1${mode}_v4se_13129088_mesh.log)"
    run_cap s1${mode}_v4se_13129088_val1d $(suite_yaml val1d_m1) $C 300 $L $E; [ $? = 2 ] && stop
    run_cap s1${mode}_v4se_val_ring_p150 fabric_exp/v4/val_ring_p150.yaml $C 120 $L $E; [ $? = 2 ] && stop
    run_cap s1${mode}_v4se_val_torusy_p150 fabric_exp/v4/val_torus_y_p150.yaml $C 120 $L $E; [ $? = 2 ] && stop
    for s in 4352 1d_m1; do
        run_cap s1${mode}_v4se_13129088_$s $(suite_yaml $s) $C 200 $L $E; [ $? = 2 ] && stop
    done
    echo "=== s1${mode}_v4se_unit1d: fabric_unit_tests Fabric1DFixture.* $E"
    env -u TT_METAL_SLOW_DISPATCH_MODE TT_METAL_HOME=/home/aagarwal/tt-metal TT_FABRIC_NO_BW_TELEMETRY=1 \
        TT_FABRIC_KERNEL_VERSION=2 TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=1000 TT_FABRIC_EXP_FLAGS=13129088 \
        TT_METAL_FABRIC_OPT_LEVEL=O3 TT_METAL_CACHE=$C $E timeout 600 ./build/test/tt_metal/tt_fabric/fabric_unit_tests \
        --gtest_filter="Fabric1DFixture.*" > $out/s1${mode}_v4se_unit1d.log 2>&1
    rc=$?
    echo "exit $rc; $(grep -aE '^\[  PASSED  \]|^\[  FAILED  \] [0-9]+ test|^\[  SKIPPED \] [0-9]+ test' $out/s1${mode}_v4se_unit1d.log | tr '\n' ' ')"
    grep -a '^\[  FAILED  \] [A-Z]' $out/s1${mode}_v4se_unit1d.log | head -8 | sed 's/^/   | /'
    [ $rc = 124 ] && { echo "!!! unit tests timed out in mode $mode"; stop; }
done
python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
rm -f $M
echo ALL_DONE
