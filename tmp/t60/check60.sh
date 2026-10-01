#!/bin/bash
# blx03 broker job for #60: device unit checks of logical_w (op level and LTX conv level, fold on vs off must be
# bit-identical). Full mesh opened, then create_submesh(2,4). Usage (via submit.sh): bash .../check60.sh
BASE=/home/smarton/fasth3/tt-metal; W=/home/smarton/fasth3/t60; V=/var/tmp/fasth3/t60; LOG=$V/check60.log
mkdir -p $V
source $BASE/python_env/bin/activate
export TT_METAL_HOME=$W PYTHONPATH=$W:$W/ttnn:$W/tools HF_HUB_OFFLINE=1 TT_METAL_CACHE=$V/jit
cd $W
echo "[t60] check tree=$(git rev-parse --short HEAD)" | tee $LOG
timeout 840 python -m pytest -sv --timeout=800 \
  tests/nightly/tg/ccl/test_neighbor_pad_async.py::test_np_bh_logical_w_2d_submesh \
  models/tt_dit/tests/models/ltx/test_vae_ltx.py::test_ltx_conv3d_fold_w_mask 2>&1 | tee -a $LOG
rc=${PIPESTATUS[0]}
echo "T60_CHECK_EXIT=$rc" | tee -a $LOG
exit $rc
