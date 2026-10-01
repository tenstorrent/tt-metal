#!/bin/bash
# blx03 broker job for #70: LTX_FUSE_GATE_ON_DEVICE fold check + LTX_FUSE_NORM_ADALN block A/B on a 2x4 submesh.
# hostfmax.py 1150 is not a drop guard; safety is full mesh then create_submesh(2,4), one job at a time, no 4x8.
# models/tt_dit comes from t51 @80882b291c2 (src/, ahead of the tree on PYTHONPATH; models is a namespace package),
# everything else (ttnn build, models/common) from blx03's tree, which is not touched.
M=/home/smarton/fasth3/tt-metal; D=/home/smarton/fasth3/t70; LOG=$D/run70.log
source $M/python_env/bin/activate
export TT_METAL_HOME=$M PYTHONPATH=$D/src:$M:$M/ttnn:$M/tools HF_HUB_OFFLINE=1
export TT_METAL_CACHE=/var/tmp/fasth3/cache/tt-metal-cache
# cwd is $D, not $M: python -m puts cwd first on sys.path, and $M ahead of src/ shadowed t51's models/tt_dit (job 035).
cd $D
echo "[t70] tree=$(git rev-parse --short HEAD) src=$(cat $D/src/REV) clock: $(python /home/smarton/tray-stress/hostfmax.py 1150 | tail -1)" | tee $LOG
timeout 1500 python -m pytest -p conftest -c $M/pytest.ini --rootdir=$M -sv --timeout=1440 $D/test_fold_gate_device_check.py 2>&1 | tee -a $LOG
rc=${PIPESTATUS[0]}
python /home/smarton/tray-stress/hostfmax.py 0 >/dev/null 2>&1
echo "T70_EXIT=$rc" | tee -a $LOG
exit $rc
