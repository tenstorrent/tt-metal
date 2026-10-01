#!/bin/bash
# blx03 broker job for #44: one arm of the LTX_VAE_FOLD_TIME_PAD conv VAE decode A/B on a 2x4 submesh.
# Usage (on blx03, via submit.sh): bash ~/fasth3/t44/run44.sh <0|1>. One arm per job.
FOLD=${1:?fold 0 or 1}
M=/home/smarton/fasth3/tt-metal; D=/home/smarton/fasth3/t44; V=/var/tmp/fasth3/t44; LOG=$D/run44_fold$FOLD.log
source $M/python_env/bin/activate
export TT_METAL_HOME=$M PYTHONPATH=$M:$M/ttnn:$M/tools HF_HUB_OFFLINE=1
# Production decode runs the fused on-device YUV output; match it (same as #43).
export LTX_FUSE_YUV_OUTPUT=1 LTX_VAE_FOLD_TIME_PAD=$FOLD
# blx03's tree (t36) plus the #44 fold, loaded in place of the tree's vae_ltx.py; the tree is not touched.
export AB_VAE_LTX_OVERLAY=$D/vae_ltx_t36_fold.py AB_OUT_DIR=$V AB_LATENT=/home/smarton/fasth3/out/t37/s2reuse0/lat.gen0.pt
export TT_METAL_CACHE=/var/tmp/fasth3/cache/tt-metal-cache
mkdir -p $V
cd $M
echo "[t44] fold=$FOLD tree=$(git rev-parse --short HEAD) clock: $(python /home/smarton/tray-stress/hostfmax.py 1150 | tail -1)" | tee $LOG
timeout 900 python -m pytest -p conftest -c $M/pytest.ini --rootdir=$M -sv --timeout=840 $D/test_vae_ltx_fold_time_pad_ab.py 2>&1 | tee -a $LOG
rc=${PIPESTATUS[0]}
python /home/smarton/tray-stress/hostfmax.py 0 >/dev/null 2>&1
echo "T44_EXIT=$rc" | tee -a $LOG
exit $rc
