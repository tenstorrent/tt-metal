#!/bin/bash
# blx03 broker job for #115: time ONE exact_s2_res conv3d blocking (arg "64,64,3,8,8") in halo mode on the 2x4
# submesh of the full mesh (the test opens (4,8), then create_submesh(2,4)). Bisects the job 484 hang.
# Python from the staged overlay $S (stage115.sh); C++ build and kernels from $B (no conv3d guard in it).
BLK=${1:?blocking}; TAG=${BLK//,/_}
BASE=/home/smarton/fasth3/tt-metal; B=${B:-/home/smarton/fasth3/t48}; V=/var/tmp/fasth3/t115; S=$V/src
LOG=$V/run115_$TAG.log
source $BASE/python_env/bin/activate
export TT_METAL_HOME=$B PYTHONPATH=$S:$B/ttnn:$B/tools HF_HUB_OFFLINE=1
export SWEEP_OUT_DIR=$V/results/$TAG SWEEP_ONLY_BLOCKINGS=$BLK SWEEP_MAX_SECONDS=300
cd $S
echo "[t115] blocking=$BLK build=$(git -C $B rev-parse --short HEAD) src=$(cat $S/REV) clock: $(python /home/smarton/tray-stress/hostfmax.py 1150 | tail -1)" | tee $LOG
test -f $B/ttnn/ttnn/_ttnn.so || { echo "[t115] no build at $B" | tee -a $LOG; exit 4; }
TT_METAL_CACHE=/var/tmp/fasth3/cache/tt-metal-cache \
  timeout 500 python -m pytest -c $S/pytest.ini --rootdir=$S -sv --timeout=480 \
  "models/tt_dit/tests/models/ltx/bruteforce_conv3d_sweep_ltx.py::test_bruteforce_sweep_ltx25_544p_145f_halo" \
  -k "exact_s2_res" 2>&1 | tee -a $LOG
rc=${PIPESTATUS[0]}
python /home/smarton/tray-stress/hostfmax.py 0 >/dev/null 2>&1
echo "T115_EXIT=$rc" | tee -a $LOG
exit $rc
