#!/bin/bash
# usage (inside the d30u20 dev pod): run_si.sh <tag> [env assignments...]   -- silicon, chip 0, JIT kernels only
TAG=$1; shift
TM=/home/container_app_user/tt-metal
export TT_METAL_HOME=$TM PYTHONPATH=$TM:$TM/ttnn ARCH_NAME=blackhole
export TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_DIR=/tmp/mka/prof/$TAG MKA_KD=/tmp/mka/kernels
unset TT_METAL_SIMULATOR MESH_DEVICE TT_VISIBLE_DEVICES
for kv in "$@"; do export "$kv"; done
mkdir -p $TT_METAL_PROFILER_DIR; cd /tmp/mka
s=$(date +%s); $TM/python_env/bin/python /tmp/mka/sim_mka1.py > /tmp/mka/$TAG.log 2>&1; rc=$?
echo "rc=$rc wall=$(( $(date +%s)-s ))s"; grep -E "\[mka1\]|TT_THROW|TT_FATAL|Error" /tmp/mka/$TAG.log | head -20
CSV=$(find $TT_METAL_PROFILER_DIR $TM/generated -name profile_log_device.csv -newer /tmp/mka/sim_mka1.py 2>/dev/null | head -1); echo "csv=$CSV"
[ -n "$CSV" ] && $TM/python_env/bin/python /tmp/mka/mka_prof.py "$CSV" | tail -14
