#!/bin/bash
# v4sc functional bring-up (2026-10-28): stream-register credits on one TX queue (TT_FABRIC_EXP_STREAM_CREDITS=1, main's
# credit plan there; variants/v4sc, the mux-mode analysis in RESULTS §98). With counter credits v4sc loads the same
# images as v4tm (250 mock configs); with stream credits every logged single-ERISC config builds (130). One ERISC
# (TT_METAL_DISABLE_FABRIC_TWO_ERISC=1; mux mode forces it), stream credits, fresh caches, hang capture armed: mesh
# sanity, val1d_m1, ring and Y-torus validation, val_mux, the VC1 split, then fabric_unit_tests on Fabric1DFixture and
# the Tensix fixtures. A hang or a unit-test timeout stops the job; failures do not. Labels scv_v4sc_*.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/cap_lib.sh
source $CON/vc1_lib.sh
source $CON/vc1_capture_lib.sh
[ -n "$out" ] && [ -d "$out" ] || { echo "no results directory"; exit 1; }
unset TT_FABRIC_EXP_SLOTS_VC1 TT_FABRIC_EXP_SLOTS TT_FABRIC_EXP_SLOTS_2D TT_FABRIC_EXP_SLOTS_1D TT_FABRIC_EXP_TIER_SLOTS
L="TT_FABRIC_EXP_LOG_SLOTS=1"
E="TT_METAL_DISABLE_FABRIC_TWO_ERISC=1"
S="TT_FABRIC_EXP_STREAM_CREDITS=1"
STUDY="TT_FABRIC_KERNEL_VERSION=2 TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=1000 TT_FABRIC_EXP_FLAGS=13129088 TT_METAL_FABRIC_OPT_LEVEL=O3"
python3 $CON/swap.py apply fabric_exp/variants/v4sc $BK || { echo STOPPED; exit 1; }
stop() { python3 $CON/swap.py restore $BK; echo STOPPED; exit 1; }
C=/home/aagarwal/.tt-metal-cache-scv
rm -rf "${C:?}"; mkdir -p $C
run_cap scv_v4sc_13129088_mesh $(suite_yaml mesh) $C 300 $L $E $S; [ $? = 2 ] && stop
grep -aoE '"VC0_USES_COUNTER_CREDITS",[01]' $(find $C -path '*fabric_erisc_router_v2*' -name named_ct_arg_map_generated.h | head -1) \
    | sed 's/^/   credit plan: /'
run_cap scv_v4sc_13129088_val1d $(suite_yaml val1d_m1) $C 300 $L $E $S; [ $? = 2 ] && stop
run_cap scv_v4sc_val_ring_p150 fabric_exp/v4/val_ring_p150.yaml $C 120 $L $E $S; [ $? = 2 ] && stop
run_cap scv_v4sc_val_torusy_p150 fabric_exp/v4/val_torus_y_p150.yaml $C 120 $L $E $S; [ $? = 2 ] && stop
run_cap scv_v4sc_val_mux fabric_exp/v4/val_mux.yaml $C 300 $L $S; [ $? = 2 ] && stop
VC1CAP_EXTRA_ENV="$L $E $S" run_vc1_cap scvvc1_v4sc_default_p1_o3 /home/aagarwal/.tt-metal-cache-scv-vc1 O3 \
    || echo "!!! VC1 split failed"
[ -d $out/scvvc1_v4sc_default_p1_o3_hangcap ] && stop
echo "=== scv_v4sc_unit: fabric_unit_tests Fabric1DFixture + Tensix fixtures, stream credits, one ERISC"
env -u TT_METAL_SLOW_DISPATCH_MODE TT_METAL_HOME=/home/aagarwal/tt-metal TT_FABRIC_NO_BW_TELEMETRY=1 TT_METAL_CACHE=$C \
    $STUDY $E $S timeout 600 ./build/test/tt_metal/tt_fabric/fabric_unit_tests \
    --gtest_filter="Fabric1DFixture.*:Fabric1DTensixFixture.*:NightlyFabric1DTensixFixture.*:NightlyFabric2DTensixFixture.*" \
    > $out/scv_v4sc_unit.log 2>&1
rc=$?
echo "exit $rc; $(grep -aE '^\[  PASSED  \]|^\[  FAILED  \] [0-9]+ test|^\[  SKIPPED \] [0-9]+ test' $out/scv_v4sc_unit.log | tr '\n' ' ')"
grep -a '^\[  FAILED  \] [A-Z]' $out/scv_v4sc_unit.log | head -8 | sed 's/^/   | /'
[ $rc = 124 ] && { echo "!!! unit tests timed out"; stop; }
python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
rm -f $M
echo ALL_DONE
