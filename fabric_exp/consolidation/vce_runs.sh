#!/bin/bash
# The builder's study default slots, end to end (2026-10-28): v4gi -O3 with no slot switch set, logging the depths
# every router gets (TT_FABRIC_EXP_LOG_SLOTS=1).
#   1. VC1 split (vc1/vc1_alloc.yaml with hang capture, vc1/vc1_inter.yaml): expect the new VC1 default 19,9,16,5,4 and
#      the rows of job 111/112's 19,9,16,5,4 (labels vce_v4gi_default_p1_o3, vcei_v4gi_default_p1_o3);
#   2. the 2D 4352 B suite and the 1D suite: expect unchanged depths (2D 17/16/16, 1D 26/25/32) and job 093's v4gi
#      rows (v4gi_13129088_4352, v4gi_13129088_1d_m1) (labels vce_v4gi_13129088_4352, vce_v4gi_13129088_1d_m1).
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/cap_lib.sh
source $CON/vc1_lib.sh
source $CON/vc1_capture_lib.sh
unset TT_FABRIC_EXP_SLOTS_VC1 TT_FABRIC_EXP_SLOTS TT_FABRIC_EXP_SLOTS_2D TT_FABRIC_EXP_SLOTS_1D TT_FABRIC_EXP_TIER_SLOTS
export TT_FABRIC_EXP_LOG_SLOTS=1
EXTRA_ENV="TT_FABRIC_EXP_LOG_SLOTS=1"
slots() {  # log
    sed 's/\x1b\[[0-9;]*m//g' $1 | grep -a 'sender slots' | grep -v '{4, 0, 0, 0}\|{0, 0, 0}' \
        | sed 's/.*\(VC[0-9] sender slots [^(]*\).*/   \1/' | sort | uniq -c
}
python3 $CON/swap.py apply fabric_exp/variants/v4gi $BK || { echo STOPPED; exit 1; }
c=/home/aagarwal/.tt-metal-cache-p1vc1-vce
VC1CAP_SPLIT_YAML=fabric_exp/vc1/vc1_alloc.yaml VC1CAP_SPLIT_FINISHES=50 run_vc1_cap vce_v4gi_default_p1_o3 $c O3 \
    || { python3 $CON/swap.py restore $BK; echo STOPPED; exit 1; }
slots $out/vce_v4gi_default_p1_o3_13129088_vc1.log
VC1_YAML=fabric_exp/vc1/vc1_inter.yaml VC1_TIMEOUT=180 run_vc1 vcei_v4gi_default_p1_o3 $c O3 || echo "!!! inter rows failed"
C=/home/aagarwal/.tt-metal-cache-vce
rm -rf "${C:?}"; mkdir -p $C
for s in 4352 1d_m1; do
    run_cap vce_v4gi_13129088_$s $(suite_yaml $s) $C 200; [ $? = 2 ] && { python3 $CON/swap.py restore $BK; echo STOPPED; exit 1; }
    slots fabric_exp/results/vce_v4gi_13129088_$s.log
done
python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
rm -f $M
echo ALL_DONE
