#!/usr/bin/env bash
# longfull.sh <tag> <case -k> <cache> <start> [extra env assignments...]  -> saves outputs under $S/lf/<tag>_*
S=${PUSH_EXP_DIR:-/tmp/push_exp}
mkdir -p $S/lf
cd /home/pjosipovic/tt-metal && source python_env/bin/activate && unset TT_METAL_ENV TT_METAL_CCACHE_KERNEL_SUPPORT
tag=$1 case=$2 cache=$3 start=$4; shift 4
env MESH_DEVICE=TG TT_METAL_SHM_TRACKING_DISABLED=1 LOGURU_LEVEL=WARNING LONGFULL_META=1 LONGFULL_CACHE=$cache LONGFULL_START=$start LONGFULL_SAVE=$S/lf/$tag "$@" \
  timeout 1800 scripts/run_safe_pytest.sh models/demos/deepseek_v3_d_p/tests/sparse_mla/push_experiment/exp_long_full.py -k "$case" -s > $S/lf/$tag.log 2>&1
grep -oE "LONGFULL .*" $S/lf/$tag.log; grep SAFE_PYTEST_RESULT $S/lf/$tag.log
