#!/bin/bash
# VC1 allocation, round 4 (2026-10-28): a test of alloc_model.py's allocations (predictions written beforehand to
# consolidation/vcd_predictions.txt), the best all-power-of-2 split, and main's router on the same rows.
# Per allocation, one fresh cache each: vc1/vc1_alloc.yaml (sanity data checked, intra-mesh rows; hang capture armed),
# vc1/vc1_inter.yaml and vc1/vc1_mix.yaml (run_vc1, 180 s timeout; skipped after a failure).
#   1. Main's router as shipped (kernel v1, flags 0, the firmware keepalive, the builder's level = -Os), the tree as is
#      (v4gi's swap replaces headers main's router includes): main's allocation, and 25,8,16,4,4 (every channel a power
#      of 2 but the worker, as main's own allocation has it). Labels vcd{,i,m}_v1_<allocation>_p1_auto.
#   2. v4gi -O3: the model's allocations for VC0:VC1 weights 1:1 (18,8,16,6,5), 3:2 (18,9,16,5,5), 1:2 (19,5,16,8,7);
#      the power-of-2 split 25,8,16,4,4; job 111's 19,9,16,5,4 as the control. Labels vcd{,i,m}_v4gi_<allocation>_p1_o3.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/vc1_lib.sh
source $CON/vc1_capture_lib.sh
export TT_FABRIC_EXP_LOG_SLOTS=1
declare -A SPLIT=([main]="" [pow2]=25,8,16,4,4 [eq]=18,8,16,6,5 [w32]=18,9,16,5,5 [w12]=19,5,16,8,7 [a9]=19,9,16,5,4)
INTER=1
MIX=1
slots() {  # log
    sed 's/\x1b\[[0-9;]*m//g' $1 | grep -a 'sender slots' | grep -v '{4, 0, 0, 0}\|{0, 0, 0}' \
        | sed 's/.*\(VC[0-9] sender slots [^(]*\).*/   \1/' | sort -u
}
one() {  # router-tag allocation level
    local r=$1 a=$2 o=$3 c=/home/aagarwal/.tt-metal-cache-p1vc1-vcd lo=$(echo $3 | tr A-Z a-z)
    if [ -n "${SPLIT[$a]}" ]; then export TT_FABRIC_EXP_SLOTS_VC1=${SPLIT[$a]}; else unset TT_FABRIC_EXP_SLOTS_VC1; fi
    echo "##### $r $a: TT_FABRIC_EXP_SLOTS_VC1=${TT_FABRIC_EXP_SLOTS_VC1:-unset}, $o (job at ${SECONDS} s)"
    VC1CAP_SPLIT_YAML=fabric_exp/vc1/vc1_alloc.yaml VC1CAP_SPLIT_FINISHES=50 run_vc1_cap vcd_${r}_${a}_p1_${lo} $c $o \
        || { python3 $CON/swap.py restore $BK; echo STOPPED; exit 1; }
    slots $out/vcd_${r}_${a}_p1_${lo}_13129088_vc1.log
    if [ $INTER = 1 ]; then
        VC1_YAML=fabric_exp/vc1/vc1_inter.yaml VC1_TIMEOUT=180 run_vc1 vcdi_${r}_${a}_p1_${lo} $c $o \
            || { INTER=0; echo "!!! inter-mesh rows failed: skipped from here"; }
    fi
    if [ $MIX = 1 ]; then
        VC1_YAML=fabric_exp/vc1/vc1_mix.yaml VC1_TIMEOUT=180 run_vc1 vcdm_${r}_${a}_p1_${lo} $c $o \
            || { MIX=0; echo "!!! mixed-load rows failed: skipped from here"; }
    fi
}
for a in main pow2; do
    KA_ENV= VC1CAP_EXTRA_ENV="TT_FABRIC_KERNEL_VERSION=1 TT_FABRIC_EXP_FLAGS=0" one v1 $a auto
done
python3 $CON/swap.py apply fabric_exp/variants/v4gi $BK || { echo STOPPED; exit 1; }
for a in eq w32 w12 pow2 a9; do one v4gi $a O3; done
python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
rm -f $M
echo ALL_DONE
