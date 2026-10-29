#!/bin/bash
# Credits by ring position, sender rework 2 (2026-10-29; V4_RING_CREDITS.md; job 135's attribution of v4rq). variants/
# v4rr = v4rq with the sender's per-packet state in registers, replacing state rather than adding it: the completion
# count as taken is the worker's `completed` (unused under first-level acks), the ack count as taken and the newest
# run's channel sit with the worker's loop locals; a send compares its channel with the newest run's and touches memory
# only on a change (start_run, out of line, queueing the finished run); the worker's send reads the two credit words in
# its post-send load group (the visit reads them only on passes without a send). Offline: non-ring configs load v4tm's
# images; sender::run 438/49/53 (v4tm 401/36/46, v4rq 405/41/50).
#   1. v4rr validation, hang capture armed (a hang stops the job): two ERISCs: ring and Y-torus validation, mesh sanity,
#      val1d_m1, val_mux (mux mode, one ERISC forced); one ERISC (TT_METAL_DISABLE_FABRIC_TWO_ERISC=1): ring and Y-torus
#      validation. Labels rrv_<mode>_v4rr_*.
#   2. Throughput, same job, v4tm then v4rr, two passes: two ERISCs exp_ring (ring) and exp_torus_y (torusy); one ERISC
#      exp_ring_se (rse); mux mode exp_mux (mux). One cache per variant. Labels rr<pass>_{tm,rr}_13129088_<suite>;
#      compare with same_job_compare.py rra_tm rra_rr rrb_tm rrb_rr.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/cap_lib.sh
[ -n "$out" ] && [ -d "$out" ] || { echo "no results directory"; exit 1; }
unset TT_FABRIC_EXP_SLOTS_VC1 TT_FABRIC_EXP_SLOTS TT_FABRIC_EXP_SLOTS_2D TT_FABRIC_EXP_SLOTS_1D TT_FABRIC_EXP_TIER_SLOTS
L="TT_FABRIC_EXP_LOG_SLOTS=1"
E="TT_METAL_DISABLE_FABRIC_TWO_ERISC=1"
stop() { python3 $CON/swap.py restore $BK; echo STOPPED; exit 1; }
CT=/home/aagarwal/.tt-metal-cache-rr-tm
CR=/home/aagarwal/.tt-metal-cache-rr-rr
rm -rf "${CT:?}" "${CR:?}"; mkdir -p $CT $CR

echo "##### 1. v4rr validation (job at ${SECONDS} s)"
python3 $CON/swap.py apply fabric_exp/variants/v4rr $BK || stop
run_cap rrv_two_v4rr_val_ring_p150 fabric_exp/v4/val_ring_p150.yaml $CR 120 $L; [ $? = 2 ] && stop
run_cap rrv_two_v4rr_val_torusy_p150 fabric_exp/v4/val_torus_y_p150.yaml $CR 120 $L; [ $? = 2 ] && stop
run_cap rrv_two_v4rr_13129088_mesh $(suite_yaml mesh) $CR 300 $L; [ $? = 2 ] && stop
run_cap rrv_two_v4rr_13129088_val1d $(suite_yaml val1d_m1) $CR 300 $L; [ $? = 2 ] && stop
run_cap rrv_two_v4rr_val_mux fabric_exp/v4/val_mux.yaml $CR 300 $L; [ $? = 2 ] && stop
run_cap rrv_one_v4rr_val_ring_p150 fabric_exp/v4/val_ring_p150.yaml $CR 120 $L $E; [ $? = 2 ] && stop
run_cap rrv_one_v4rr_val_torusy_p150 fabric_exp/v4/val_torus_y_p150.yaml $CR 120 $L $E; [ $? = 2 ] && stop
python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }

suites() {  # pass arm cache
    local p=$1 a=$2 c=$3
    run_cap rr${p}_${a}_13129088_ring fabric_exp/v4/exp_ring.yaml $c 200 $L; [ $? = 2 ] && return 2
    run_cap rr${p}_${a}_13129088_torusy fabric_exp/v4/exp_torus_y.yaml $c 200 $L; [ $? = 2 ] && return 2
    run_cap rr${p}_${a}_13129088_rse fabric_exp/v4/exp_ring_se.yaml $c 300 $L $E; [ $? = 2 ] && return 2
    run_cap rr${p}_${a}_13129088_mux fabric_exp/v4/exp_mux.yaml $c 400 $L; [ $? = 2 ] && return 2
    return 0
}
for p in a b; do
    echo "##### 2. throughput, pass $p (job at ${SECONDS} s)"
    python3 $CON/swap.py apply fabric_exp/variants/v4tm $BK || stop
    suites $p tm $CT || stop
    python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
    python3 $CON/swap.py apply fabric_exp/variants/v4rr $BK || stop
    suites $p rr $CR || stop
    python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
done
rm -f $M
echo ALL_DONE
