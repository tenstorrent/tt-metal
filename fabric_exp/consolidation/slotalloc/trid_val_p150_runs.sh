#!/bin/bash
# Transaction-ID allocation, P150 validation (TRID_ALLOCATION.md §6.1; broker job on a09u32 after
# host_rebuild_tridalloc.sh, or an lb2 job after lb2_stage.sh full): variants/v4txa11id in the v2 slot with
# TT_FABRIC_EXP_TRID_ALLOC=1 (and TT_FABRIC_EXP_LOG_SLOTS=1, so every run logs its receivers' G / ids / start), -O3,
# study settings, BW telemetry off, hang capture armed. Data-checked runs; a hang or a failure stops the job.
#   trid_val_p150_runs.sh PART      PART a | b | all
#   a  two ERISCs: mesh sanity (mesh), val1d_m1 (val1d), val_ring_p150 (ring), val_torus_y_p150 (torusy),
#      val_torus_xy_p150 on the 2x4 XY torus (txy); one ERISC: mesh sanity and val_ring_p150 (mesh_1e, ring_1e)
#   b  the VC1 split (vc1_alloc.yaml, two ranks, 50 finishes; VC2 with VC1 does not build on v4: not run), the
#      inline-write and fused / flushed atomic-inc gtests in 1D and 2D (inline1d, inline2d), the UDM fixtures (one
#      ERISC, 2D: udm_unit, udm_nightly)
# After each two-ERISC run: the log must show the assigned IDs ("receiver transaction IDs (TT_FABRIC_EXP_TRID_ALLOC)")
# and no uniform-rule line, so the switch took effect. The host logs them for E1's build only (risc 1), so one-ERISC
# runs (*_1e, UDM) have no such lines and are not checked; the same library and switch are proven by the others. Labels tridv_<run> (results/<label>.log; vc1: <label>_13129088_vc1.*).
# Fresh caches /home/aagarwal/.tt-metal-cache-tridv-<two|one|vc1|gt>. Runs are not started past TRID_BUDGET s
# (default 1350) of job time; the job then says where it stopped (rerun the rest with a part).
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/cap_lib.sh
source $CON/vc1_lib.sh
source $CON/vc1_capture_lib.sh
PART=${1:-}
case $PART in a|b|all) ;; *) echo "usage: trid_val_p150_runs.sh a|b|all"; exit 1 ;; esac
[ -n "$out" ] && [ -d "$out" ] || { echo "no results directory"; exit 1; }
unset TT_FABRIC_EXP_SLOTS_VC1 TT_FABRIC_EXP_SLOTS_VC2 TT_FABRIC_EXP_SLOTS TT_FABRIC_EXP_SLOTS_2D TT_FABRIC_EXP_SLOTS_1D \
    TT_FABRIC_EXP_TIER_SLOTS TT_FABRIC_EXP_DYN_SLOTS TT_FABRIC_EXP_DYN_SLOTS_POLICY
grep -aq TT_FABRIC_EXP_TRID_ALLOC build_Release/tt_metal/libtt_metal.so ||
    { echo "!!! the host library has no TT_FABRIC_EXP_TRID_ALLOC (host_rebuild_tridalloc.sh first)"; exit 1; }
V=v4txa11id
TRID=(TT_FABRIC_EXP_TRID_ALLOC=1 TT_FABRIC_EXP_LOG_SLOTS=1)
E1=TT_METAL_DISABLE_FABRIC_TWO_ERISC=1
SAN=tests/tt_metal/tt_fabric/test_infra/test_yamls
C=/home/aagarwal/.tt-metal-cache-tridv
for k in two one vc1 gt; do rm -rf "${C:?}-$k"; mkdir -p $C-$k; done
stop() { python3 $CON/swap.py restore $BK; echo STOPPED; exit 1; }
# ids LOG: the switch took effect (assigned IDs logged, no uniform-rule line)
ids() {
    local n u
    n=$(sed 's/\x1b\[[0-9;]*m//g' "$1" | grep -ac "receiver transaction IDs (TT_FABRIC_EXP_TRID_ALLOC)")
    u=$(sed 's/\x1b\[[0-9;]*m//g' "$1" | grep -ac "receiver transaction IDs (uniform rule)")
    echo "   transaction IDs: $n assigned lines, $u uniform-rule lines; distinct:"
    sed 's/\x1b\[[0-9;]*m//g' "$1" | grep -aoE "RX_CH_[0-9] G [0-9]+ ids [0-9]+ start [0-9]+" | sort | uniq -c | head -8
    [ "$n" -gt 0 ] && [ "$u" = 0 ] || { echo "!!! the switch did not take effect"; return 1; }
}
budget() { [ $SECONDS -le ${TRID_BUDGET:-1350} ] || { echo "=== budget: not starting $1 or later at ${SECONDS} s"; python3 $CON/swap.py restore $BK; echo "BUDGET_STOP at $1"; exit 1; }; }
r() {  # run yaml cache timeout [ENV...]   (NOIDS=1: a one-ERISC run, no ID lines to check)
    local run=$1 y=$2 c=$3 t=$4 rc; shift 4
    budget $run
    run_cap tridv_$run "$y" $c $t "${TRID[@]}" "$@"; rc=$?
    [ $rc = 0 ] || { echo "!!! tridv_$run: rc $rc"; stop; }
    [ -n "${NOIDS:-}" ] || ids $out/tridv_$run.log || stop
}
python3 $CON/swap.py apply fabric_exp/variants/$V $BK || stop
if [ $PART != b ]; then
    echo "##### part a: two ERISCs (job at ${SECONDS} s)"
    r mesh $SAN/test_fabric_mesh_sanity_common.yaml $C-two 300
    r val1d fabric_exp/v4/val1d_m1.yaml $C-two 300
    r ring fabric_exp/v4/val_ring_p150.yaml $C-two 120
    r torusy fabric_exp/v4/val_torus_y_p150.yaml $C-two 120
    r txy fabric_exp/torus/val_torus_xy_p150.yaml $C-two 900
    echo "##### part a: one ERISC (job at ${SECONDS} s)"
    NOIDS=1 r mesh_1e $SAN/test_fabric_mesh_sanity_common.yaml $C-one 300 $E1
    NOIDS=1 r ring_1e fabric_exp/v4/val_ring_p150.yaml $C-one 120 $E1
fi
if [ $PART != a ]; then
    echo "##### part b: VC1 split (job at ${SECONDS} s)"
    budget vc1
    VC1CAP_EXTRA_ENV="${TRID[*]}" VC1CAP_SPLIT_YAML=fabric_exp/vc1/vc1_alloc.yaml VC1CAP_SPLIT_FINISHES=50 \
        VC1CAP_SPLIT_TIMEOUT=300 run_vc1_cap tridv_vc1 $C-vc1 O3 || stop
    ids $out/tridv_vc1_13129088_vc1.log || stop
    echo "##### part b: inline write, fused / flushed atomic (job at ${SECONDS} s)"
    r inline1d "gtest:fabric_unit_tests:NightlyFabric1DFixture.TestLinearFabricUnicastNocInlineWrite:NightlyFabric1DFixture.TestLinearFabricUnicastNocFusedAtomicInc" $C-gt 300
    r inline2d "gtest:fabric_unit_tests:NightlyFabric2DFixture.TestMeshFabricUnicastNocInlineWrite:NightlyFabric2DFixture.TestMeshFabricUnicastNocFusedAtomicInc" $C-gt 300
    echo "##### part b: UDM (job at ${SECONDS} s)"
    NOIDS=1 r udm_unit "gtest:Fabric2DUDMModeFixture.*" $C-gt 600
    NOIDS=1 r udm_nightly "gtest:NightlyFabric2DUDMModeFixture.*" $C-gt 900
fi
python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
rm -f $M
echo "ALL_DONE part $PART (job at ${SECONDS} s)"
