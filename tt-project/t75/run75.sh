#!/bin/bash
# blx03 broker job for #75: conv VAE decode on the t48 tip with all default folds stacked, 2x4 submesh of the
# full 4x8 mesh, 544x960/145f (per-chip shard of 1080p/145f on 4x8), real 2.5 latent, fused YUV output,
# 1 warmup + 3 timed decodes per arm. Same harness and settings as #46/#65 (run60.sh).
# Arm "def": tree defaults (fold-time-pad, W-mask fold, #58 unpatch) -> yuv_t1w1.pt
# Arm "ref": all three off (LTX_VAE_FOLD_TIME_PAD=0, LTX_VAE_FOLD_W_MASK=0, pre-#58 unpatch overlay) -> yuv_t0w0.pt
# Usage (on blx03, via submit.sh): bash /home/smarton/fasth3/t75/run75.sh
BASE=/home/smarton/fasth3/tt-metal; W=/home/smarton/fasth3/t48; D=/home/smarton/fasth3/t75; V=/var/tmp/fasth3/t75
LOG=$D/run75.log
mkdir -p $V
source $BASE/python_env/bin/activate
export TT_METAL_HOME=$W PYTHONPATH=$W:$W/ttnn:$W/tools HF_HUB_OFFLINE=1 TT_METAL_CACHE=$V/jit
export LTX_FUSE_YUV_OUTPUT=1 AB_OUT_DIR=$V AB_LATENT=/home/smarton/fasth3/out/t37/s2reuse0/lat.gen0.pt
cd $W
echo "[t75] tree=$(git rev-parse --short HEAD) clock: $(python /home/smarton/tray-stress/hostfmax.py 1150 | tail -1)" | tee $LOG
test -f "$AB_LATENT" || { echo "[t75] $AB_LATENT missing" | tee -a $LOG; exit 3; }
T=models/tt_dit/tests/models/ltx/test_vae_ltx_fold_time_pad_ab.py
echo "[t75] arm=def" | tee -a $LOG
timeout 600 python -m pytest -c $W/pytest.ini --rootdir=$W -sv --timeout=560 $T 2>&1 | tee -a $LOG
rc1=${PIPESTATUS[0]}
echo "T75_EXIT[def]=$rc1" | tee -a $LOG
rc2=99
if [ $rc1 = 0 ]; then
  echo "[t75] arm=ref" | tee -a $LOG
  LTX_VAE_FOLD_TIME_PAD=0 LTX_VAE_FOLD_W_MASK=0 AB_VAE_LTX_OVERLAY=$D/vae_ltx_pre58.py \
    timeout 600 python -m pytest -c $W/pytest.ini --rootdir=$W -sv --timeout=560 $T 2>&1 | tee -a $LOG
  rc2=${PIPESTATUS[0]}
  echo "T75_EXIT[ref]=$rc2" | tee -a $LOG
fi
python /home/smarton/tray-stress/hostfmax.py 0 >/dev/null 2>&1
python - $V 2>&1 <<'PY' | tee -a $LOG
import sys, torch
V = sys.argv[1]
a, b = torch.load(f"{V}/yuv_t1w1.pt"), torch.load(f"{V}/yuv_t0w0.pt")
same = a.shape == b.shape and torch.equal(a, b)
d = (a.int() - b.int()).abs().max().item() if a.shape == b.shape else "shape"
print(f"T75_CMP def_vs_ref identical={same} max_abs_diff={d} shape={tuple(a.shape)} dtype={a.dtype}")
PY
[ $rc1 = 0 ] && [ $rc2 = 0 ]; rc=$?
echo "T75_EXIT=$rc" | tee -a $LOG
exit $rc
