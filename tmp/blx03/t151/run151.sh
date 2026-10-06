#!/bin/bash
# #151 broker job on blx03: check the t149 vol2col_rm straddle guard on the exact_s2_res conv3d
# (C_in=C_out=512, k=3, halo mode) on the 2x4 submesh of the full mesh (the test opens (4,8), then
# create_submesh(2,4)). Three blockings, the passing one first so a guard miss cannot hide it:
#   (64,128,6,8,2)  96 patches, aligned       -> should run
#   (64,128,5,4,4)  80 patches, unaligned >64 -> should hit the TT_FATAL (hung in job 273 without it)
#   (64,32,5,4,4)   80 patches, unaligned >64 -> should hit the TT_FATAL
# C++ from $B (t48 + guard, build151.sh). Python from the t115 overlay (bc134f7c656), which has no
# Python-side straddle filter, so the C++ guard is what rejects the blockings.
BASE=/home/smarton/fasth3/tt-metal; B=/home/smarton/fasth3/t48; V=/var/tmp/fasth3/t151; S=/var/tmp/fasth3/t115/src
LOG=$V/run151.log
source $BASE/python_env/bin/activate
export TT_METAL_HOME=$B PYTHONPATH=$S:$B/ttnn:$B/tools HF_HUB_OFFLINE=1
export SWEEP_OUT_DIR=$V/results SWEEP_ONLY_BLOCKINGS="64,128,6,8,2;64,128,5,4,4;64,32,5,4,4" SWEEP_MAX_SECONDS=300
cd $S
echo "[t151] build=$(git -C $B log -1 --format='%h %s') src=$(cat $S/REV)" | tee $LOG
test -f $B/ttnn/ttnn/_ttnn.so || { echo "[t151] no build at $B" | tee -a $LOG; exit 4; }
grep -q 'vol2col_rm CB chunks straddle' $B/ttnn/cpp/ttnn/operations/experimental/conv3d/device/conv3d_program_factory.cpp \
  || { echo "[t151] guard not in $B" | tee -a $LOG; exit 5; }
TT_METAL_CACHE=/var/tmp/fasth3/cache/tt-metal-cache \
  timeout 500 python -m pytest -c $S/pytest.ini --rootdir=$S -sv --timeout=480 \
  "models/tt_dit/tests/models/ltx/bruteforce_conv3d_sweep_ltx.py::test_bruteforce_sweep_ltx25_544p_145f_halo" \
  -k "exact_s2_res" 2>&1 | tee -a $LOG
rc=${PIPESTATUS[0]}
echo "T151_EXIT=$rc" | tee -a $LOG
exit $rc
