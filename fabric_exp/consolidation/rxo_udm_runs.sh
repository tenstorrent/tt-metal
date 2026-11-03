#!/bin/bash
# udm_runs.sh's UDM fixtures for any variant (V4_RECEIVER_OWNERSHIP.md; the UDM relay moved out of VC0's downstreams in
# v4rxo4). fabric_unit_tests Fabric2DUDMModeFixture.* and NightlyFabric2DUDMModeFixture.* (one ERISC, 2D), the variant
# in the v2 slot, hang capture armed; a hang stops the job, test failures do not. Usage: rxo_udm_runs.sh VARIANT.
# Logs fabric_exp/results/udm_<variant>_{unit,nightly}.log.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/cap_lib.sh
[ -n "$out" ] && [ -d "$out" ] || { echo "no results directory"; exit 1; }
unset TT_FABRIC_EXP_SLOTS_VC1 TT_FABRIC_EXP_SLOTS TT_FABRIC_EXP_SLOTS_2D TT_FABRIC_EXP_SLOTS_1D TT_FABRIC_EXP_TIER_SLOTS
r=$1
[ -n "$r" ] && [ -d fabric_exp/variants/$r ] || { echo "usage: rxo_udm_runs.sh VARIANT"; exit 1; }
c=/home/aagarwal/.tt-metal-cache-udm-$r
rm -rf "${c:?}"; mkdir -p $c
python3 $CON/swap.py apply fabric_exp/variants/$r $BK || { echo STOPPED; exit 1; }
rc=0
run_cap udm_${r}_unit "gtest:Fabric2DUDMModeFixture.*" $c 600; [ $? = 2 ] && rc=2
[ $rc = 2 ] || { run_cap udm_${r}_nightly "gtest:NightlyFabric2DUDMModeFixture.*" $c 900; [ $? = 2 ] && rc=2; }
python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
[ $rc = 2 ] && { echo STOPPED; exit 1; }
rm -f $M
echo ALL_DONE
