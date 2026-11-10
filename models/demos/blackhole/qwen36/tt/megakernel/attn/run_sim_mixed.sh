#!/bin/bash
# usage (inside ttsim-mega): run_sim_mixed.sh <tag> <NL> <AMASK> [ctx]   -- sim, slow dispatch (large-tensor uploads)
TAG=$1; NLV=$2; AM=$3; CTXV=${4:-4096}
cd /tmp/ttsim-work && source simenv.sh bh $TAG
export SIM_SLOW_DISPATCH=1 TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_DEVICE_PROFILER=1
export MKA_KD=/work/wt-mixed/models/demos/blackhole/qwen36/tt/megakernel/attn/kernels
export FM_NL=$NLV MK_AMASK=$AM MK_SGDN_US=64 MK_SAR_US=18 MKA_SWO_US=10 MKA_CTX=$CTXV MKA_KVDT=bfp8 MKA_NBB=1 MKA_BUDGET=16384 FM_ITERS=2 MK_DEFINES=MKA_GUPRE=1
s=$(date +%s)
$PY /work/wt-mixed/models/demos/blackhole/qwen36/tt/megakernel/attn/bench_mka2.py > $TT_METAL_LOGS_PATH/run.log 2>&1; rc=$?
echo "rc=$rc wall=$(( $(date +%s)-s ))s"
grep -E "\[(mk1|mka2|fm11)\] iter|TT_FATAL|TT_THROW|Error" $TT_METAL_LOGS_PATH/run.log | grep -v "Werror" | head -20
CSV=$(find $TT_METAL_PROFILER_DIR /home/container_app_user/tt-metal/generated -name profile_log_device.csv -newer /tmp/ttsim-work/simenv.sh 2>/dev/null | head -1); echo csv=$CSV
[ -n "$CSV" ] && python3 /work/wt-mixed/models/demos/blackhole/qwen36/tt/megakernel/attn/mka2_layers.py "$CSV" 1350 | head -8
