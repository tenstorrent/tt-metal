#!/bin/bash
# Credit reads placed in the send (2026-10-29; RESULTS §98, v4rr's attribution: its single-flow gain was the sender's
# credit read inside the send, not the protocol). variants/v4rs = v4tm with, on first-level-ack VCs, each channel's ack
# counter read in the send's post-send load group with its completion counter (the visit uses it when it sent, and reads
# the counter itself only on passes without a send); the protocol is v4tm's (per-channel counts). Offline: only ring
# and torus sender images change (receivers and every other config load v4tm's images); sender::run 410/38/47 (v4tm
# 401/36/46).
#   1. v4rs validation, hang capture armed (a hang stops the job): two ERISCs: ring and Y-torus validation, mesh sanity,
#      val1d_m1, val_mux (mux mode, one ERISC forced); one ERISC (TT_METAL_DISABLE_FABRIC_TWO_ERISC=1): ring and Y-torus
#      validation. Labels rsv_<mode>_v4rs_*.
#   2. Throughput, same job, v4tm, v4rs, v4rr per pass, two passes: two ERISCs exp_ring (ring) and exp_torus_y (torusy);
#      one ERISC exp_ring_se (rse); mux mode exp_mux (mux). One cache per variant. Labels
#      rs<pass>_{tm,rs,rr}_13129088_<suite>; compare with same_job_compare.py rsa_tm rsa_rs rsa_rr rsb_tm rsb_rs rsb_rr.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/cap_lib.sh
[ -n "$out" ] && [ -d "$out" ] || { echo "no results directory"; exit 1; }
unset TT_FABRIC_EXP_SLOTS_VC1 TT_FABRIC_EXP_SLOTS TT_FABRIC_EXP_SLOTS_2D TT_FABRIC_EXP_SLOTS_1D TT_FABRIC_EXP_TIER_SLOTS
L="TT_FABRIC_EXP_LOG_SLOTS=1"
E="TT_METAL_DISABLE_FABRIC_TWO_ERISC=1"
stop() { python3 $CON/swap.py restore $BK; echo STOPPED; exit 1; }
CT=/home/aagarwal/.tt-metal-cache-rs-tm
CS=/home/aagarwal/.tt-metal-cache-rs-rs
CR=/home/aagarwal/.tt-metal-cache-rs-rr
rm -rf "${CT:?}" "${CS:?}" "${CR:?}"; mkdir -p $CT $CS $CR

echo "##### 1. v4rs validation (job at ${SECONDS} s)"
python3 $CON/swap.py apply fabric_exp/variants/v4rs $BK || stop
run_cap rsv_two_v4rs_val_ring_p150 fabric_exp/v4/val_ring_p150.yaml $CS 120 $L; [ $? = 2 ] && stop
run_cap rsv_two_v4rs_val_torusy_p150 fabric_exp/v4/val_torus_y_p150.yaml $CS 120 $L; [ $? = 2 ] && stop
run_cap rsv_two_v4rs_13129088_mesh $(suite_yaml mesh) $CS 300 $L; [ $? = 2 ] && stop
run_cap rsv_two_v4rs_13129088_val1d $(suite_yaml val1d_m1) $CS 300 $L; [ $? = 2 ] && stop
run_cap rsv_two_v4rs_val_mux fabric_exp/v4/val_mux.yaml $CS 300 $L; [ $? = 2 ] && stop
run_cap rsv_one_v4rs_val_ring_p150 fabric_exp/v4/val_ring_p150.yaml $CS 120 $L $E; [ $? = 2 ] && stop
run_cap rsv_one_v4rs_val_torusy_p150 fabric_exp/v4/val_torus_y_p150.yaml $CS 120 $L $E; [ $? = 2 ] && stop
python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }

suites() {  # pass arm cache
    local p=$1 a=$2 c=$3
    run_cap rs${p}_${a}_13129088_ring fabric_exp/v4/exp_ring.yaml $c 200 $L; [ $? = 2 ] && return 2
    run_cap rs${p}_${a}_13129088_torusy fabric_exp/v4/exp_torus_y.yaml $c 200 $L; [ $? = 2 ] && return 2
    run_cap rs${p}_${a}_13129088_rse fabric_exp/v4/exp_ring_se.yaml $c 300 $L $E; [ $? = 2 ] && return 2
    run_cap rs${p}_${a}_13129088_mux fabric_exp/v4/exp_mux.yaml $c 400 $L; [ $? = 2 ] && return 2
    return 0
}
for p in a b; do
    echo "##### 2. throughput, pass $p (job at ${SECONDS} s)"
    python3 $CON/swap.py apply fabric_exp/variants/v4tm $BK || stop
    suites $p tm $CT || stop
    python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
    python3 $CON/swap.py apply fabric_exp/variants/v4rs $BK || stop
    suites $p rs $CS || stop
    python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
    python3 $CON/swap.py apply fabric_exp/variants/v4rr $BK || stop
    suites $p rr $CR || stop
    python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
done
rm -f $M
echo ALL_DONE
