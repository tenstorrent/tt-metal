#!/bin/bash
# usage: run_mka1.sh <tag> [env assignments...]   (inside ttsim-mega)
TAG=$1; shift
cd /tmp/ttsim-work && source simenv.sh bh $TAG >/dev/null
export SIM_SLOW_DISPATCH=0; unset TT_METAL_SLOW_DISPATCH_MODE
export TT_METAL_DEVICE_PROFILER=1
for kv in "$@"; do export "$kv"; done
s=$(date +%s)
nice -n 10 $PY /work/mk-attn/sim_mka1.py > $TT_METAL_LOGS_PATH/run.log 2>&1; rc=$?
echo "rc=$rc wall=$(( $(date +%s)-s ))s"
grep -E "\[mka1\]|TT_THROW|TT_FATAL|error:|Error" $TT_METAL_LOGS_PATH/run.log | head -30
CSV=$(find $TT_METAL_PROFILER_DIR -name profile_log_device.csv 2>/dev/null | head -1); echo "csv=$CSV"
[ -n "$CSV" ] && python3 /work/mk-attn/mka_prof.py "$CSV" | tail -12
