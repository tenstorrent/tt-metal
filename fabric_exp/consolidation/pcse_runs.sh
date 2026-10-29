#!/bin/bash
# Single-ERISC ring analysis, PC sampling (2026-10-28; RESULTS §98): on one ERISC v4 (counter credits) is +19
# cycles/packet against main on the ring 1-hop bidirectional row, while stream-register credits are slower still (job
# 130). Per router (main's: kernel v1, flags 0, firmware keepalive; then v4tm), TT_METAL_DISABLE_FABRIC_TWO_ERISC=1 (the
# router on ERISC 1), a fresh cache: fabric_exp/pcse/compile.yaml (the row, short: compiles the routers and logs their
# compile commands), then fabric_exp/pcse/bidir.yaml (the row, 40M packets of 1024 B, ~11 s) while pc_sampler.py samples
# every chip's ERISC 1 PCs (safe: no halt, no measured perturbation; RESULTS §98 phase/cost trial). Outputs
# results/pcse_{v1,v4}_samples.{bin,json}; report with pcs_report.py PREFIX CACHE. Labels pcse_{v1,v4}_{compile,bidir}.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/cap_lib.sh
[ -n "$out" ] && [ -d "$out" ] || { echo "no results directory"; exit 1; }
unset TT_FABRIC_EXP_SLOTS_VC1 TT_FABRIC_EXP_SLOTS TT_FABRIC_EXP_SLOTS_2D TT_FABRIC_EXP_SLOTS_1D TT_FABRIC_EXP_TIER_SLOTS
L="TT_FABRIC_EXP_LOG_SLOTS=1"
E="TT_METAL_DISABLE_FABRIC_TWO_ERISC=1"
MAIN="TT_FABRIC_KERNEL_VERSION=1 TT_FABRIC_EXP_FLAGS=0"
stop() { python3 $CON/swap.py restore $BK; echo STOPPED; exit 1; }
sample() {  # router cache env...: compile row, then the long row under the sampler
    local r=$1 c=$2; shift 2
    rm -rf "${c:?}"; mkdir -p $c
    run_cap pcse_${r}_compile fabric_exp/pcse/compile.yaml $c 300 $L $E "$@"; [ $? = 2 ] && return 2
    rm -f $out/pcse_${r}_bidir.log $out/pcse_${r}_samples.bin $out/pcse_${r}_samples.json
    ( run_cap pcse_${r}_bidir fabric_exp/pcse/bidir.yaml $c 300 $L $E "$@" ) & local rp=$!
    timeout 300 python_env/bin/python $CON/pc_sampler.py $out/pcse_${r}_samples --start-log $out/pcse_${r}_bidir.log \
        --start-delay-s 2 --max-s 60 --eriscs 1 2>&1 | grep --line-buffered -E '^sampler: |Error|error|Traceback'
    wait $rp; local rc=$?
    awk -F, 'NR>1 {printf "   rate: %s %s B %.1f cycles/packet\n", $1, $7, 1.35e9/$10}' $out/pcse_${r}_bidir.csv 2>/dev/null
    return $rc
}
KA_ENV= sample v1 /home/aagarwal/.tt-metal-cache-pcse-v1 $MAIN; [ $? = 2 ] && { echo STOPPED; exit 1; }
python3 $CON/swap.py apply fabric_exp/variants/v4tm $BK || stop
sample v4 /home/aagarwal/.tt-metal-cache-pcse-v4; [ $? = 2 ] && stop
python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
rm -f $M
echo ALL_DONE
