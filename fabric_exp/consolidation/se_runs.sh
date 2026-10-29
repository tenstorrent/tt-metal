#!/bin/bash
# Single-ERISC baseline (2026-10-28, after the merge of main): main's router as shipped (kernel v1, flags 0, the firmware
# keepalive, the builder's levels: -O3 VC0, -Os VC1) in each MODE given:
#   two    the default (two ERISCs per router on Blackhole), the reference
#   fab1   TT_METAL_DISABLE_FABRIC_TWO_ERISC=1: the fabric uses one ERISC, the system stays two-ERISC (router on ERISC 1)
#   aer1   TT_METAL_DISABLE_MULTI_AERISC=1: the whole system single-ERISC (router on ERISC 0, shared with base firmware)
# Per mode, fresh caches, depths logged: mesh sanity, 1D sanity (test_fabric_sanity_at_least_2x2_mesh.yaml), the 2D and
# 1D 4352 B benchmarks, the VC1 split (vc1_split.yaml, hang capture armed), then CI's single-ERISC fabric leg
# (fabric_unit_tests --gtest_filter="Fabric1D*Fixture.*", 600 s timeout). A hang stops the job; failures do not.
# Labels se_<mode>_v1_13129088_<suite>, se_<mode>vc1_v1_default_p1_auto, se_<mode>_unit1d.log.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/cap_lib.sh
source $CON/vc1_lib.sh
source $CON/vc1_capture_lib.sh
[ $# -gt 0 ] || { echo "usage: se_runs.sh MODE..."; exit 1; }
MAIN="TT_FABRIC_KERNEL_VERSION=1 TT_FABRIC_EXP_FLAGS=0"
L="TT_FABRIC_EXP_LOG_SLOTS=1"
declare -A MODE_ENV=([two]="" [fab1]="TT_METAL_DISABLE_FABRIC_TWO_ERISC=1" [aer1]="TT_METAL_DISABLE_MULTI_AERISC=1")
procs() {  # log: which ERISCs the router binaries were built for
    sed 's/\x1b\[[0-9;]*m//g' $1 | grep -ao 'kernels/fabric_erisc_router/[0-9]*/[a-z_]*erisc' | awk -F/ '{print $NF}' \
        | sort | uniq -c | tr '\n' ' '
}
for mode in "$@"; do
    [ -n "${MODE_ENV[$mode]+x}" ] || { echo "unknown mode $mode"; exit 1; }
    E=${MODE_ENV[$mode]}
    echo "##### mode $mode: ${E:-default} (job at ${SECONDS} s)"
    C=/home/aagarwal/.tt-metal-cache-se-$mode
    rm -rf "${C:?}"; mkdir -p $C
    KA_ENV= run_cap se_${mode}_v1_13129088_mesh $(suite_yaml mesh) $C 300 $MAIN $L $E; [ $? = 2 ] && { echo STOPPED; exit 1; }
    echo "   router binaries: $(procs $out/se_${mode}_v1_13129088_mesh.log)"
    KA_ENV= run_cap se_${mode}_v1_13129088_val1d $(suite_yaml val1d) $C 300 $MAIN $L $E; [ $? = 2 ] && { echo STOPPED; exit 1; }
    for s in 4352 1d; do
        KA_ENV= run_cap se_${mode}_v1_13129088_$s $(suite_yaml $s) $C 200 $MAIN $L $E; [ $? = 2 ] && { echo STOPPED; exit 1; }
    done
    KA_ENV= VC1CAP_EXTRA_ENV="$MAIN $L $E" run_vc1_cap se_${mode}vc1_v1_default_p1_auto \
        /home/aagarwal/.tt-metal-cache-p1vc1-se-$mode auto || echo "!!! VC1 split failed in mode $mode"
    [ -d $out/se_${mode}vc1_v1_default_p1_auto_hangcap ] && { echo STOPPED; exit 1; }
    echo "   VC1 router binaries: $(procs $out/se_${mode}vc1_v1_default_p1_auto_13129088_vc1.log)"
    echo "=== se_${mode}_unit1d: fabric_unit_tests Fabric1D*Fixture.* ${E:-default}"
    env -u TT_METAL_SLOW_DISPATCH_MODE TT_METAL_HOME=/home/aagarwal/tt-metal TT_FABRIC_NO_BW_TELEMETRY=1 \
        TT_METAL_CACHE=$C $E timeout 600 ./build/test/tt_metal/tt_fabric/fabric_unit_tests \
        --gtest_filter="Fabric1D*Fixture.*" > $out/se_${mode}_unit1d.log 2>&1
    rc=$?
    echo "exit $rc; $(grep -aE '^\[  PASSED  \]|^\[  FAILED  \] [0-9]+ test|^\[  SKIPPED \] [0-9]+ test' $out/se_${mode}_unit1d.log | tr '\n' ' ')"
    grep -a '^\[  FAILED  \] [A-Z]' $out/se_${mode}_unit1d.log | head -8 | sed 's/^/   | /'
    [ $rc = 124 ] && { echo "!!! unit tests timed out in mode $mode"; echo STOPPED; exit 1; }
done
rm -f $M
echo ALL_DONE
