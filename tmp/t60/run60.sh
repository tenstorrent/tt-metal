#!/bin/bash
# blx03 broker job for #60: one arm of the conv VAE decode fold A/B on a 2x4 submesh (full mesh opened, then
# create_submesh(2,4)), 544x960/145f, real 2.5 latent, fused YUV output, 1 warmup + 3 timed decodes.
# Arms: t<LTX_VAE_FOLD_TIME_PAD>w<LTX_VAE_FOLD_W_MASK>, e.g. t1w0 (#44 only) and t1w1 (#44 + #60). One arm per job.
# Runs the t60 worktree (~/fasth3/t60, built by blx03_setup60.sh) with the shared tree's venv; kernels JIT into
# their own cache under /var/tmp. hostfmax.py 1150 matches #44's runs; it is not a drop guard.
# Usage (on blx03, via submit.sh): bash /home/smarton/fasth3/t60/tmp/t60/run60.sh <t0w0|t1w0|t1w1|t0w1>
ARM=${1:?arm t<0|1>w<0|1>}
[[ $ARM =~ ^t([01])w([01])$ ]] || { echo "bad arm $ARM"; exit 2; }
TPAD=${BASH_REMATCH[1]} WMASK=${BASH_REMATCH[2]}
BASE=/home/smarton/fasth3/tt-metal; W=/home/smarton/fasth3/t60; V=/var/tmp/fasth3/t60; LOG=$V/run60_$ARM.log
mkdir -p $V
source $BASE/python_env/bin/activate
export TT_METAL_HOME=$W PYTHONPATH=$W:$W/ttnn:$W/tools HF_HUB_OFFLINE=1
export LTX_FUSE_YUV_OUTPUT=1 LTX_VAE_FOLD_TIME_PAD=$TPAD LTX_VAE_FOLD_W_MASK=$WMASK
export AB_OUT_DIR=$V AB_LATENT=/home/smarton/fasth3/out/t37/s2reuse0/lat.gen0.pt
export TT_METAL_CACHE=$V/jit
cd $W
echo "[t60] arm=$ARM tree=$(git rev-parse --short HEAD) clock: $(python /home/smarton/tray-stress/hostfmax.py 1150 | tail -1)" | tee $LOG
test -f "$AB_LATENT" || echo "[t60] WARNING: $AB_LATENT missing, the harness falls back to a random latent" | tee -a $LOG
timeout 900 python -m pytest -c $W/pytest.ini --rootdir=$W -sv --timeout=840 \
  models/tt_dit/tests/models/ltx/test_vae_ltx_fold_time_pad_ab.py 2>&1 | tee -a $LOG
rc=${PIPESTATUS[0]}
python /home/smarton/tray-stress/hostfmax.py 0 >/dev/null 2>&1
echo "T60_EXIT[$ARM]=$rc" | tee -a $LOG
exit $rc
