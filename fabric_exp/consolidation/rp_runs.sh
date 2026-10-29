#!/bin/bash
# Credits by ring position, bring-up (2026-10-28; V4_RING_CREDITS.md, variants/v4rp = v4tm + ring-position credits on
# routers whose only VC is VC0 with first-level acks: rings and tori). Offline: every other config loads v4tm's image
# (250 mock configs); ring and torus configs build in both ERISC modes and with the mux.
#   1. v4rp validation, hang capture armed (a hang stops the job): two ERISCs: ring and Y-torus validation, mesh sanity,
#      val1d_m1, val_mux (mux mode, one ERISC forced); one ERISC (TT_METAL_DISABLE_FABRIC_TWO_ERISC=1): ring and Y-torus
#      validation. Labels rpv_<mode>_v4rp_*.
#   2. Throughput, same job, v4tm then v4rp, two passes: two ERISCs exp_ring (ring) and exp_torus_y (torusy); one ERISC
#      exp_ring_se (rse); mux mode exp_mux (mux). One cache per variant. Labels rp<pass>_{tm,rp}_13129088_<suite>;
#      compare with same_job_compare.py rpa_tm rpa_rp rpb_tm rpb_rp.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/cap_lib.sh
[ -n "$out" ] && [ -d "$out" ] || { echo "no results directory"; exit 1; }
unset TT_FABRIC_EXP_SLOTS_VC1 TT_FABRIC_EXP_SLOTS TT_FABRIC_EXP_SLOTS_2D TT_FABRIC_EXP_SLOTS_1D TT_FABRIC_EXP_TIER_SLOTS
L="TT_FABRIC_EXP_LOG_SLOTS=1"
E="TT_METAL_DISABLE_FABRIC_TWO_ERISC=1"
stop() { python3 $CON/swap.py restore $BK; echo STOPPED; exit 1; }
CT=/home/aagarwal/.tt-metal-cache-rp-tm
CR=/home/aagarwal/.tt-metal-cache-rp-rp
rm -rf "${CT:?}" "${CR:?}"; mkdir -p $CT $CR

echo "##### 1. v4rp validation (job at ${SECONDS} s)"
python3 $CON/swap.py apply fabric_exp/variants/v4rp $BK || stop
run_cap rpv_two_v4rp_val_ring_p150 fabric_exp/v4/val_ring_p150.yaml $CR 120 $L; [ $? = 2 ] && stop
run_cap rpv_two_v4rp_val_torusy_p150 fabric_exp/v4/val_torus_y_p150.yaml $CR 120 $L; [ $? = 2 ] && stop
run_cap rpv_two_v4rp_13129088_mesh $(suite_yaml mesh) $CR 300 $L; [ $? = 2 ] && stop
run_cap rpv_two_v4rp_13129088_val1d $(suite_yaml val1d_m1) $CR 300 $L; [ $? = 2 ] && stop
run_cap rpv_two_v4rp_val_mux fabric_exp/v4/val_mux.yaml $CR 300 $L; [ $? = 2 ] && stop
run_cap rpv_one_v4rp_val_ring_p150 fabric_exp/v4/val_ring_p150.yaml $CR 120 $L $E; [ $? = 2 ] && stop
run_cap rpv_one_v4rp_val_torusy_p150 fabric_exp/v4/val_torus_y_p150.yaml $CR 120 $L $E; [ $? = 2 ] && stop
python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }

suites() {  # pass arm cache
    local p=$1 a=$2 c=$3
    run_cap rp${p}_${a}_13129088_ring fabric_exp/v4/exp_ring.yaml $c 200 $L; [ $? = 2 ] && return 2
    run_cap rp${p}_${a}_13129088_torusy fabric_exp/v4/exp_torus_y.yaml $c 200 $L; [ $? = 2 ] && return 2
    run_cap rp${p}_${a}_13129088_rse fabric_exp/v4/exp_ring_se.yaml $c 300 $L $E; [ $? = 2 ] && return 2
    run_cap rp${p}_${a}_13129088_mux fabric_exp/v4/exp_mux.yaml $c 400 $L; [ $? = 2 ] && return 2
    return 0
}
for p in a b; do
    echo "##### 2. throughput, pass $p (job at ${SECONDS} s)"
    python3 $CON/swap.py apply fabric_exp/variants/v4tm $BK || stop
    suites $p tm $CT || stop
    python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
    python3 $CON/swap.py apply fabric_exp/variants/v4rp $BK || stop
    suites $p rp $CR || stop
    python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
done
rm -f $M
echo ALL_DONE
