#!/bin/bash
# blx03 broker job for #87: traced vs eager conv VAE decode (LTX_VIDEO_VAE_TRACE), 2x4 submesh of the full 4x8
# mesh (the test opens (4,8) and calls create_submesh(2,4)), 544x960/145f, real 2.5 latent, fused YUV output,
# LTX_TIME_STAGES=1 (VAE_DECODE_SPLIT lines), 1 warmup + 3 eager + capture + 3 traced replays, one process.
# Runs the t48 build (a613d669eef; t87 adds Python only) with this branch's models/ from an overlay under
# /var/tmp/fasth3/t87/src (stage it with stage87.sh). Usage on blx03:
#   cd ~/fasth3/tt-metal && tmp/blx03/submit.sh 1200 bash /var/tmp/fasth3/t87/src/tmp/blx03/run87.sh
BASE=/home/smarton/fasth3/tt-metal; B=/home/smarton/fasth3/t48; V=/var/tmp/fasth3/t87; S=$V/src
LOG=$V/run87.log
source $BASE/python_env/bin/activate
export TT_METAL_HOME=$B PYTHONPATH=$S:$B/ttnn:$B/tools HF_HUB_OFFLINE=1
export TT_METAL_CACHE=/var/tmp/fasth3/cache/tt-metal-cache
export LTX_FUSE_YUV_OUTPUT=1 LTX_TIME_STAGES=1 AB_OUT_DIR=$V AB_LATENT=/home/smarton/fasth3/out/t37/s2reuse0/lat.gen0.pt
export LTX_TRACE_REGION=${LTX_TRACE_REGION:-500000000}
cd $S
echo "[t87] build=$(git -C $B rev-parse --short HEAD) src=$(cat $S/REV) clock: $(python /home/smarton/tray-stress/hostfmax.py 1150 | tail -1)" | tee $LOG
test -f "$AB_LATENT" || { echo "[t87] $AB_LATENT missing" | tee -a $LOG; exit 3; }
test -f $B/ttnn/ttnn/_ttnn.so || { echo "[t87] no t48 build" | tee -a $LOG; exit 4; }
timeout 900 python -m pytest -c $S/pytest.ini --rootdir=$S -sv --timeout=860 \
  models/tt_dit/tests/models/ltx/test_vae_ltx_trace_ab.py 2>&1 | tee -a $LOG
rc=${PIPESTATUS[0]}
python /home/smarton/tray-stress/hostfmax.py 0 >/dev/null 2>&1
echo "T87_EXIT=$rc" | tee -a $LOG
exit $rc
