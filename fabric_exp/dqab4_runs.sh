#!/bin/bash
# §41 A/B on the user's v4 (scratchpad ab4.py; v4 files restored at the end).
cd /home/aagarwal/tt-metal
AB=/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/ab4.py
tt-smi -r all > /dev/null 2>&1
for v in ${VARIANTS:-noinline nodefer nofusedpre}; do
    python3 $AB $v || exit 0
  for y in ${YAMLS:-mix}; do
    case $y in mix) yf=fabric_exp/streams/p3_mixes.yaml;; small) yf=fabric_exp/exp_2d_trim_small.yaml;; lat) yf=fabric_exp/exp_latency.yaml;; 4352) yf=fabric_exp/exp_2d_4352.yaml;; esac
    l=dqab_${v}_15226240_$y
    env -u TT_METAL_SLOW_DISPATCH_MODE timeout 900 fabric_exp/run_exp.sh $l $yf \
        TT_FABRIC_NO_BW_TELEMETRY=1 TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=1000 TT_FABRIC_EXP_FLAGS=15226240 > /dev/null 2>&1
    s=$?; r=$(grep -oE 'Tests: .*executed, [0-9]+ failed' fabric_exp/results/$l.log)
    echo "$l: ${r:-NO RESULT (exit $s)}"
    if [ $s -ne 0 ] || ! echo "$r" | grep -q ", 0 failed"; then pkill -x test_tt_fabric; sleep 2; tt-smi -r all > /dev/null 2>&1; break 2; fi
  done
done
python3 $AB v4
echo ALL_DONE
