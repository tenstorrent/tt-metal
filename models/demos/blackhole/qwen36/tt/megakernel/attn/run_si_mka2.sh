#!/bin/bash
# usage (inside the d30u20 dev pod): run_si_mka2.sh <tag> [env...]   -- silicon chip 0, JIT kernels only, 600 s timeout
TAG=$1; shift
MK2=${MK2:-/tmp/mk2}
TM=/home/container_app_user/tt-metal
export TT_METAL_HOME=$TM PYTHONPATH=$TM:$TM/ttnn ARCH_NAME=blackhole TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_DIR=$MK2/prof/$TAG
export MKA_KD=$MK2/megakernel/attn/kernels MK_MIXN=${MK_MIXN:-1}
unset TT_METAL_SIMULATOR MESH_DEVICE TT_VISIBLE_DEVICES
for kv in "$@"; do export "$kv"; done
mkdir -p $TT_METAL_PROFILER_DIR; cd $MK2
s=$(date +%s); timeout ${MK_TIMEOUT:-600} $TM/python_env/bin/python $MK2/megakernel/attn/bench_mka2.py > $MK2/$TAG.log 2>&1; rc=$?
echo "rc=$rc wall=$(( $(date +%s)-s ))s"; grep -E "\[mka2\]|\[mk1\] iter|\[fm11\] iter|TT_THROW|TT_FATAL|Error|Timeout" $MK2/$TAG.log | head -24
CSV=$(find $TT_METAL_PROFILER_DIR -name profile_log_device.csv 2>/dev/null | head -1)
[ -n "$CSV" ] && $TM/python_env/bin/python $MK2/megakernel/attn/mka_prof.py "$CSV" | tail -24
[ $rc = 124 ] && echo "TIMEOUT: hang" && $TM/python_env/bin/python -c "print('see tt-triage')"
