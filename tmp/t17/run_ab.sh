#!/bin/bash
# blx03 broker job: single-chip conv3d blocking A/B. Kernels/JIT from the main tree (warm cache), models/ + test from t17.
M=/home/smarton/fasth3/tt-metal; T=/home/smarton/fasth3/t17; LOG=/home/smarton/fasth3/out/t17_ab.log
source $M/python_env/bin/activate
export TT_METAL_HOME=$M PYTHONPATH=$T:$M/ttnn:$M/tools TT_METAL_CACHE=/var/tmp/fasth3/cache/tt-metal-cache HF_HUB_OFFLINE=1
cd $T
echo "[t17ab] commit=$(git rev-parse --short HEAD) clock: $(python /home/smarton/tray-stress/hostfmax.py 1150 | tail -1)" | tee $LOG
python -u -m pytest -sv --timeout=600 tmp/t17/test_blk_ab.py 2>&1 | tee -a $LOG
rc=${PIPESTATUS[0]}
python /home/smarton/tray-stress/hostfmax.py 0 >/dev/null 2>&1
echo "T17AB_EXIT=$rc" | tee -a $LOG
exit $rc
