#!/bin/bash
# Stage 0 of the receiver ownership pass (V4_RECEIVER_OWNERSHIP.md §6): the inline-write hang check of V4_BOUNDS.md §5
# (bnd_inline_runs.sh's sequence, copied), on a fixed variant. fabric_unit_tests, one fixture (one fabric instance) per
# run: the unicast inline-write test, then the fused write + flushed atomic-inc test. Same job: main's router as shipped
# (kernel v1, flags 0, firmware keepalive; the tree untouched) first as the control, then VARIANT (swapped in). 1D, then
# 2D. Labels rxoi_<router>_<1d|2d>. On a hang: run_cap's capture (results/<label>_hangcap/), then bnd_ethniu.py's NIU
# counters and ERISC PCs (results/<label>_ethniu.json); the job stops. Usage: rxo0_inline_runs.sh VARIANT
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/cap_lib.sh
[ -n "$out" ] && [ -d "$out" ] || { echo "no results directory"; exit 1; }
V=$1
[ -n "$V" ] && [ -d fabric_exp/variants/$V ] || { echo "usage: rxo0_inline_runs.sh VARIANT"; exit 1; }
F1D="NightlyFabric1DFixture.TestLinearFabricUnicastNocInlineWrite:NightlyFabric1DFixture.TestLinearFabricUnicastNocFusedAtomicInc"
F2D="NightlyFabric2DFixture.TestMeshFabricUnicastNocInlineWrite:NightlyFabric2DFixture.TestMeshFabricUnicastNocFusedAtomicInc"
stop() { python3 $CON/swap.py restore $BK; echo STOPPED; exit 1; }
one() {  # label filter cache [ENV...]
    local l=$1 f=$2 c=$3 rc; shift 3
    run_cap $l "gtest:fabric_unit_tests:$f" $c 300 "$@"; rc=$?
    if [ $rc = 2 ]; then
        echo "=== $l hung: NIU counters and ERISC PCs"
        env -u TT_VISIBLE_DEVICES timeout 240 python_env/bin/python $CON/bnd_ethniu.py $out/${l}_ethniu.json 2>&1 | tail -3
        stop
    fi
    [ $rc = 0 ] || stop
}
CM=/home/aagarwal/.tt-metal-cache-rxoi-main
CV=/home/aagarwal/.tt-metal-cache-rxoi-$V
rm -rf "${CM:?}" "${CV:?}"; mkdir -p $CM $CV
KA_ENV= one rxoi_main_1d "$F1D" $CM TT_FABRIC_KERNEL_VERSION=1 TT_FABRIC_EXP_FLAGS=0
KA_ENV= one rxoi_main_2d "$F2D" $CM TT_FABRIC_KERNEL_VERSION=1 TT_FABRIC_EXP_FLAGS=0
python3 $CON/swap.py apply fabric_exp/variants/$V $BK || stop
one rxoi_${V}_1d "$F1D" $CV
one rxoi_${V}_2d "$F2D" $CV
python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
rm -f $M
echo "ALL_DONE (job at ${SECONDS} s)"
