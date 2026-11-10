#!/bin/bash
# usage: run_sim_fm11.sh <tag> <NL>   (inside ttsim-mega container)
TAG=$1; NLV=$2
cd /tmp/ttsim-work && source simenv.sh bh $TAG
export SIM_SLOW_DISPATCH=1 TT_METAL_SLOW_DISPATCH_MODE=1
export TT_METAL_DEVICE_PROFILER=1 FM_KD=/work/fused_mlp/kernels FM_NL=$NLV MK_SMIX_US=${3:-67} MK_SAR_US=${4:-18} MK_MIXN=${5:-1} FM_ITERS=2
s=$(date +%s)
$PY /work/sim_mk1.py > $TT_METAL_LOGS_PATH/run.log 2>&1; rc=$?
echo "rc=$rc wall=$(( $(date +%s)-s ))s"
grep -E "\[(fm11|mk1)\]" $TT_METAL_LOGS_PATH/run.log | grep -vE "noc x|gu/bank|bal dn"
CSV=$(find $TT_METAL_PROFILER_DIR /home/container_app_user/tt-metal/generated -name profile_log_device.csv -newer /tmp/ttsim-work/simenv.sh 2>/dev/null | head -1); echo "csv=$CSV"
python3 /tmp/ttsim-work/profparse.py "$CSV" | tail -4
