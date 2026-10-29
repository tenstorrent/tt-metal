#!/bin/bash
# Ring-position credits, attribution (2026-10-29; RESULTS §98 "Credits by ring position, first build"): on two ERISCs
# the ring 1-hop write row went 150.7 -> 193.1 cycles/packet with v4rp (job 132), the sender loop growing 401 -> 457
# instructions. PC sampling of both ERISCs of every router while the row runs, per variant (v4tm, then v4rp), each
# swapped in for its own run, a fresh cache each: fabric_exp/pcse/ring2_compile.yaml (the row, short: compiles the
# routers), then fabric_exp/pcse/ring2_1hop.yaml (60M packets of 4352 B) under pc_sampler.py (both ERISCs).
# Outputs results/pcsr_{tm,rp}_samples.{bin,json}; per source line with pcs_lines.py PREFIX CACHE CHIP/CHAN/ERISC.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/cap_lib.sh
[ -n "$out" ] && [ -d "$out" ] || { echo "no results directory"; exit 1; }
unset TT_FABRIC_EXP_SLOTS_VC1 TT_FABRIC_EXP_SLOTS TT_FABRIC_EXP_SLOTS_2D TT_FABRIC_EXP_SLOTS_1D TT_FABRIC_EXP_TIER_SLOTS
L="TT_FABRIC_EXP_LOG_SLOTS=1"
stop() { python3 $CON/swap.py restore $BK; echo STOPPED; exit 1; }
sample() {  # arm variant cache
    local a=$1 v=$2 c=$3
    python3 $CON/swap.py apply fabric_exp/variants/$v $BK || return 2
    rm -rf "${c:?}"; mkdir -p $c
    run_cap pcsr_${a}_compile fabric_exp/pcse/ring2_compile.yaml $c 300 $L; [ $? = 2 ] && return 2
    rm -f $out/pcsr_${a}_1hop.log $out/pcsr_${a}_samples.bin $out/pcsr_${a}_samples.json
    ( run_cap pcsr_${a}_1hop fabric_exp/pcse/ring2_1hop.yaml $c 300 $L ) & local rp=$!
    timeout 300 python_env/bin/python $CON/pc_sampler.py $out/pcsr_${a}_samples --start-log $out/pcsr_${a}_1hop.log \
        --start-delay-s 2 --max-s 60 --eriscs 0,1 2>&1 | grep --line-buffered -E '^sampler: |Error|error|Traceback'
    wait $rp; local rc=$?
    awk -F, 'NR>1 {printf "   rate: %s %s B %.1f cycles/packet\n", $1, $7, 1.35e9/$10}' $out/pcsr_${a}_1hop.csv 2>/dev/null
    python3 $CON/swap.py restore $BK || return 2
    return $rc
}
sample tm v4tm /home/aagarwal/.tt-metal-cache-pcsr-tm; [ $? = 2 ] && stop
sample rp v4rp /home/aagarwal/.tt-metal-cache-pcsr-rp; [ $? = 2 ] && stop
rm -f $M
echo ALL_DONE
