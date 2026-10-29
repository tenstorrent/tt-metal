#!/bin/bash
# Credit transport on one ERISC, A/B (2026-10-28; RESULTS §98 mux-mode analysis): v4sc with counter credits (the
# default; the same images as v4tm) against stream-register credits (TT_FABRIC_EXP_STREAM_CREDITS=1, main's plan on one
# TX queue), main's router as the reference, in the order main, counters, stream per pass. One ERISC
# (TT_METAL_DISABLE_FABRIC_TWO_ERISC=1; mux mode forces it), one cache per arm, hang capture armed.
#   pass a: exp_ring_se.yaml (rse), exp_1d_m1 (1d_m1), exp_2d_4352 (4352), exp_mux.yaml (mux), the VC1 split
#   pass b: rse and mux again (the rows that moved, for the repeat)
# Labels sc<pass>_{v1,v4c,v4s}_13129088_<suite>, sc<pass>vc1_{v1,v4c,v4s}_default_p1_{auto,o3}; compare with
# same_job_compare.py sca_v4c sca_v4s scb_v4c scb_v4s (and sca_v1 as the control).
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/cap_lib.sh
source $CON/vc1_lib.sh
source $CON/vc1_capture_lib.sh
[ -n "$out" ] && [ -d "$out" ] || { echo "no results directory"; exit 1; }
unset TT_FABRIC_EXP_SLOTS_VC1 TT_FABRIC_EXP_SLOTS TT_FABRIC_EXP_SLOTS_2D TT_FABRIC_EXP_SLOTS_1D TT_FABRIC_EXP_TIER_SLOTS
L="TT_FABRIC_EXP_LOG_SLOTS=1"
E="TT_METAL_DISABLE_FABRIC_TWO_ERISC=1"
MAIN="TT_FABRIC_KERNEL_VERSION=1 TT_FABRIC_EXP_FLAGS=0"
S="TT_FABRIC_EXP_STREAM_CREDITS=1"
C1=/home/aagarwal/.tt-metal-cache-scp-v1
CC=/home/aagarwal/.tt-metal-cache-scp-v4c
CS=/home/aagarwal/.tt-metal-cache-scp-v4s
rm -rf "${C1:?}" "${CC:?}" "${CS:?}"; mkdir -p $C1 $CC $CS
stop() { python3 $CON/swap.py restore $BK; echo STOPPED; exit 1; }
arm() {  # pass arm cache vc1_level env...: the pass's suites for one arm (KA_ENV as the caller sets it)
    local p=$1 a=$2 c=$3 o=$4 s; shift 4
    local suites="rse 1d_m1 4352"
    [ $p = b ] && suites="rse"
    for s in $suites; do
        local y=$(suite_yaml $s); [ $s = rse ] && y=fabric_exp/v4/exp_ring_se.yaml
        run_cap sc${p}_${a}_13129088_$s $y $c 400 $L $E "$@"; [ $? = 2 ] && return 2
    done
    run_cap sc${p}_${a}_13129088_mux fabric_exp/v4/exp_mux.yaml $c 400 $L "$@"; [ $? = 2 ] && return 2
    if [ $p = a ]; then
        VC1CAP_EXTRA_ENV="$L $E $*" run_vc1_cap sc${p}vc1_${a}_default_p1_$o $c-vc1 $o || echo "!!! VC1 split failed ($a)"
        [ -d $out/sc${p}vc1_${a}_default_p1_${o}_hangcap ] && return 2
    fi
    return 0
}
for p in a b; do
    echo "##### pass $p (job at ${SECONDS} s)"
    KA_ENV= arm $p v1 $C1 auto $MAIN || { echo STOPPED; exit 1; }
    python3 $CON/swap.py apply fabric_exp/variants/v4sc $BK || stop
    arm $p v4c $CC O3 || stop
    arm $p v4s $CS O3 $S || stop
    python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
done
rm -f $M
echo ALL_DONE
