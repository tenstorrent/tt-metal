#!/bin/bash
# Inline-write check (2026-11-01; V4_BOUNDS.md §B, item 4): an inline-write packet's local delivery is emulated on
# write_cmd_buf (0) of E1's local NoC, the buffer v4 presets for local writes (non-posted); the emulation rewrites its
# NOC_CTRL as posted and v4 does not restore it, so later counted local writes go out posted and a later flush
# (flushed atomic: ncrisc_noc_nonposted_writes_flushed; another inline write: ..._sent) cannot complete (code read).
# fabric_unit_tests, one fixture (one fabric instance) per run: the unicast inline-write test, then the fused write +
# flushed atomic-inc test (gtest runs them in definition order). Same job: main's router as shipped (kernel v1, flags 0,
# firmware keepalive; the tree untouched) first as the control, then v4rxv2i3 (swapped in). 1D, then 2D.
# Labels bndi_<router>_<1d|2d>. On a hang: run_cap's capture (results/<label>_hangcap/), then bnd_ethniu.py's NIU
# counters and ERISC PCs (results/<label>_ethniu.json); the job stops.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/cap_lib.sh
[ -n "$out" ] && [ -d "$out" ] || { echo "no results directory"; exit 1; }
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
CM=/home/aagarwal/.tt-metal-cache-bndi-main
CL=/home/aagarwal/.tt-metal-cache-bndi-lead
rm -rf "${CM:?}" "${CL:?}"; mkdir -p $CM $CL
KA_ENV= one bndi_main_1d "$F1D" $CM TT_FABRIC_KERNEL_VERSION=1 TT_FABRIC_EXP_FLAGS=0
KA_ENV= one bndi_main_2d "$F2D" $CM TT_FABRIC_KERNEL_VERSION=1 TT_FABRIC_EXP_FLAGS=0
python3 $CON/swap.py apply fabric_exp/variants/v4rxv2i3 $BK || stop
one bndi_v4rxv2i3_1d "$F1D" $CL
one bndi_v4rxv2i3_2d "$F2D" $CL
python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
rm -f $M
echo "ALL_DONE (job at ${SECONDS} s)"
