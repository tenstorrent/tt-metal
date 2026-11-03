#!/bin/bash
# Sender consolidation C1b (V4_SENDER_CONSOLIDATION.md; 2026-11-03): main's channel-trimming gtests (fabric_unit_tests
# ChannelTrimming*: capture bitfields, override profiles, the 11 capture-then-replay scenarios) on one variant, two
# ERISCs then one, as variant_val_runs.sh's trim / trim_1e. txc_trim_val_runs.sh VARIANT. Labels txv_<variant>_<two|one>.
# Hang capture armed; a hang stops the job; failures are listed at the end.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/cap_lib.sh
V=$1
[ -n "$V" ] && [ -d fabric_exp/variants/$V ] || { echo "usage: txc_trim_val_runs.sh VARIANT"; exit 1; }
[ -n "$out" ] && [ -d "$out" ] || { echo "no results directory"; exit 1; }
unset TT_FABRIC_EXP_SLOTS_VC1 TT_FABRIC_EXP_SLOTS TT_FABRIC_EXP_SLOTS_2D TT_FABRIC_EXP_SLOTS_1D TT_FABRIC_EXP_TIER_SLOTS
C=/home/aagarwal/.tt-metal-cache-txv-$V
rm -rf "${C:?}-two" "${C:?}-one"; mkdir -p $C-two $C-one
stop() { python3 $CON/swap.py restore $BK; echo STOPPED; exit 1; }
FAILED=""
python3 $CON/swap.py apply fabric_exp/variants/$V $BK || stop
run_cap txv_${V}_two "gtest:ChannelTrimming*" $C-two 1300
case $? in 2) stop ;; 1) FAILED="$FAILED two" ;; esac
run_cap txv_${V}_one "gtest:ChannelTrimming*" $C-one 1300 TT_METAL_DISABLE_FABRIC_TWO_ERISC=1
case $? in 2) stop ;; 1) FAILED="$FAILED one" ;; esac
python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
for m in two one; do
    echo "$m: $(grep -aE '^\[  (PASSED|FAILED|SKIPPED) +\] [0-9]+ test' $out/txv_${V}_$m.log | tr '\n' ' ')"
done
rm -f $M
echo "failed:${FAILED:- none}"
echo "ALL_DONE (job at ${SECONDS} s)"
