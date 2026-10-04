#!/bin/bash
# blx03 broker job for #114: halo-mode conv3d blocking sweep of ONE LTX-2.5 decoder conv (arg: s2_res, s3_res,
# s4_res, s1_up, s3_chg) on the 2x4 submesh of the full mesh (the test opens (4,8), then create_submesh(2,4)).
# Writes $V/results/<layer>_*.json: table blocking in halo and pre-padded mode, sweep top 20, best-vs-table
# output check. Python from the staged overlay $S (stage100.sh); C++ build and kernels from $B.
L=${1:?layer}
BASE=/home/smarton/fasth3/tt-metal; B=${B:-/home/smarton/fasth3/t48}; V=/var/tmp/fasth3/t114; S=$V/src
LOG=$V/run114_$L.log
# Test ids: exact_<layer> for the exact-shard list, bare <layer> for pad mode (whose -k also matches exact_<layer>).
case $L in exact_*) K=$L ;; *) K="$L and not exact" ;; esac
source $BASE/python_env/bin/activate
export TT_METAL_HOME=$B PYTHONPATH=$S:$B/ttnn:$B/tools HF_HUB_OFFLINE=1
export SWEEP_OUT_DIR=$V/results SWEEP_MAX_SECONDS=${SWEEP_MAX_SECONDS:-720}
cd $S
echo "[t114] layer=$L build=$(git -C $B rev-parse --short HEAD) src=$(cat $S/REV)" | tee $LOG
test -f $B/ttnn/ttnn/_ttnn.so || { echo "[t114] no build at $B" | tee -a $LOG; exit 4; }
TT_METAL_CACHE=/var/tmp/fasth3/cache/tt-metal-cache \
  timeout 1100 python -m pytest -c $S/pytest.ini --rootdir=$S -sv --timeout=1060 \
  "models/tt_dit/tests/models/ltx/bruteforce_conv3d_sweep_ltx.py::test_bruteforce_sweep_ltx25_544p_145f_halo" \
  -k "$K" 2>&1 | tee -a $LOG
rc=${PIPESTATUS[0]}
echo "T114_EXIT=$rc" | tee -a $LOG
exit $rc
