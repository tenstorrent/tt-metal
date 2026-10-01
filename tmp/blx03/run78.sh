#!/bin/bash
# blx03 broker job for #78: conv VAE decode A/B of LTX_VAE_HALO_ONLY, 2x4 submesh of the full 4x8 mesh (the
# test opens (4,8) and calls create_submesh(2,4)), 544x960/145f (per-chip shard of 1080p/145f on 4x8), real
# 2.5 latent, fused YUV output, 1 warmup + 3 timed decodes per arm, tree defaults otherwise.
# Arm "h0": LTX_VAE_HALO_ONLY unset (neighbor_pad full-pad copy)          -> yuv_h0.pt
# Arm "h1": LTX_VAE_HALO_ONLY=1 (neighbor_pad_halo + conv3d halo_buffer)  -> yuv_h1.pt
# Needs the t78 build (tmp/blx03/setup78.sh). Usage on blx03:
#   cd ~/fasth3/tt-metal && tmp/blx03/submit.sh 1500 bash /home/smarton/fasth3/t78/tmp/blx03/run78.sh
BASE=/home/smarton/fasth3/tt-metal; W=/home/smarton/fasth3/t78; V=/var/tmp/fasth3/t78
LOG=$V/run78.log
mkdir -p $V
source $BASE/python_env/bin/activate
export TT_METAL_HOME=$W PYTHONPATH=$W:$W/ttnn:$W/tools HF_HUB_OFFLINE=1 TT_METAL_CACHE=$V/jit
export LTX_FUSE_YUV_OUTPUT=1 AB_OUT_DIR=$V AB_LATENT=/home/smarton/fasth3/out/t37/s2reuse0/lat.gen0.pt
cd $W
echo "[t78] tree=$(git rev-parse --short HEAD) clock: $(python /home/smarton/tray-stress/hostfmax.py 1150 | tail -1)" | tee $LOG
test -f "$AB_LATENT" || { echo "[t78] $AB_LATENT missing" | tee -a $LOG; exit 3; }
test -f $W/ttnn/ttnn/_ttnn.so || { echo "[t78] no t78 build; run setup78.sh" | tee -a $LOG; exit 4; }
T=models/tt_dit/tests/models/ltx/test_vae_ltx_halo_only_ab.py
echo "[t78] arm=h0" | tee -a $LOG
LTX_VAE_HALO_ONLY=0 timeout 600 python -m pytest -c $W/pytest.ini --rootdir=$W -sv --timeout=560 $T 2>&1 | tee -a $LOG
rc1=${PIPESTATUS[0]}
echo "T78_EXIT[h0]=$rc1" | tee -a $LOG
rc2=99
if [ $rc1 = 0 ]; then
  echo "[t78] arm=h1" | tee -a $LOG
  LTX_VAE_HALO_ONLY=1 timeout 600 python -m pytest -c $W/pytest.ini --rootdir=$W -sv --timeout=560 $T 2>&1 | tee -a $LOG
  rc2=${PIPESTATUS[0]}
  echo "T78_EXIT[h1]=$rc2" | tee -a $LOG
fi
python /home/smarton/tray-stress/hostfmax.py 0 >/dev/null 2>&1
python - $V 2>&1 <<'PY' | tee -a $LOG
import sys, torch
V = sys.argv[1]
a, b = torch.load(f"{V}/yuv_h1.pt"), torch.load(f"{V}/yuv_h0.pt")
same = a.shape == b.shape and torch.equal(a, b)
d = (a.int() - b.int()).abs().max().item() if a.shape == b.shape else "shape"
print(f"T78_CMP h1_vs_h0 identical={same} max_abs_diff={d} shape={tuple(a.shape)} dtype={a.dtype}")
PY
[ $rc1 = 0 ] && [ $rc2 = 0 ]; rc=$?
echo "T78_EXIT=$rc" | tee -a $LOG
exit $rc
