#!/bin/bash
# Ring-position credits, attribution of v4rr on mixed traffic (2026-10-29; job 136: one-ERISC full-ring multicast
# 500.6 -> 574.1 cycles/packet, every router's sender interleaving its worker's packets with forwarded ones). The
# full-ring multicast row on one ERISC (TT_METAL_DISABLE_FABRIC_TWO_ERISC=1), per variant (v4tm, then v4rr), each
# swapped in for its own run, a fresh cache each: fabric_exp/pcse/fullring_compile.yaml (the row, short: compiles the
# routers), then fabric_exp/pcse/fullring.yaml (15M packets of 1024 B per sender) under pc_sampler.py (ERISC 1, where
# the router runs). Outputs results/pcsf_{tm,rr}_samples.{bin,json}; per source line with pcs_lines.py PREFIX CACHE
# CHIP/CHAN/ERISC.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/cap_lib.sh
[ -n "$out" ] && [ -d "$out" ] || { echo "no results directory"; exit 1; }
unset TT_FABRIC_EXP_SLOTS_VC1 TT_FABRIC_EXP_SLOTS TT_FABRIC_EXP_SLOTS_2D TT_FABRIC_EXP_SLOTS_1D TT_FABRIC_EXP_TIER_SLOTS
L="TT_FABRIC_EXP_LOG_SLOTS=1 TT_METAL_DISABLE_FABRIC_TWO_ERISC=1"
stop() { python3 $CON/swap.py restore $BK; echo STOPPED; exit 1; }
sample() {  # arm variant cache
    local a=$1 v=$2 c=$3
    python3 $CON/swap.py apply fabric_exp/variants/$v $BK || return 2
    rm -rf "${c:?}"; mkdir -p $c
    run_cap pcsf_${a}_compile fabric_exp/pcse/fullring_compile.yaml $c 300 $L; [ $? = 2 ] && return 2
    rm -f $out/pcsf_${a}_full.log $out/pcsf_${a}_samples.bin $out/pcsf_${a}_samples.json
    ( run_cap pcsf_${a}_full fabric_exp/pcse/fullring.yaml $c 300 $L ) & local rp=$!
    timeout 300 python_env/bin/python $CON/pc_sampler.py $out/pcsf_${a}_samples --start-log $out/pcsf_${a}_full.log \
        --start-delay-s 2 --max-s 60 --eriscs 1 2>&1 | grep --line-buffered -E '^sampler: |Error|error|Traceback'
    wait $rp; local rc=$?
    awk -F, 'NR>1 {printf "   rate: %s %s B %.1f cycles/packet\n", $1, $7, 1.35e9/$10}' $out/pcsf_${a}_full.csv 2>/dev/null
    python3 $CON/swap.py restore $BK || return 2
    return $rc
}
sample tm v4tm /home/aagarwal/.tt-metal-cache-pcsf-tm; [ $? = 2 ] && stop
sample rr v4rr /home/aagarwal/.tt-metal-cache-pcsf-rr; [ $? = 2 ] && stop
rm -f $M
echo ALL_DONE
