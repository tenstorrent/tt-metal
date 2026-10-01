#!/bin/bash
# blx03 broker job for #76: conv VAE decode A/B of the batched neighbor_pad local copy, 2x4 submesh of the
# full 4x8 mesh (the test opens (4,8) and calls create_submesh(2,4)), 544x960/145f (per-chip shard of
# 1080p/145f on 4x8), real 2.5 latent, fused YUV output, 1 warmup + 3 timed decodes per arm, tree defaults.
# Arm "b0":   TT_NEIGHBOR_PAD_LOCAL_BATCH unset (per-stick copy)   -> yuv_b0.pt
# Arm "b128": TT_NEIGHBOR_PAD_LOCAL_BATCH=128 (one batch per row)  -> yuv_b128.pt
# Needs the t76 build (tmp/blx03/setup76.sh). Usage on blx03:
#   cd ~/fasth3/tt-metal && tmp/blx03/submit.sh 1500 bash /home/smarton/fasth3/t76/tmp/blx03/run76.sh
BASE=/home/smarton/fasth3/tt-metal; W=/home/smarton/fasth3/t76; V=/var/tmp/fasth3/t76
LOG=$V/run76.log
mkdir -p $V
source $BASE/python_env/bin/activate
export TT_METAL_HOME=$W PYTHONPATH=$W:$W/ttnn:$W/tools HF_HUB_OFFLINE=1 TT_METAL_CACHE=$V/jit
export LTX_FUSE_YUV_OUTPUT=1 AB_OUT_DIR=$V AB_LATENT=/home/smarton/fasth3/out/t37/s2reuse0/lat.gen0.pt
cd $W
echo "[t76] tree=$(git rev-parse --short HEAD) clock: $(python /home/smarton/tray-stress/hostfmax.py 1150 | tail -1)" | tee $LOG
test -f "$AB_LATENT" || { echo "[t76] $AB_LATENT missing" | tee -a $LOG; exit 3; }
test -f $W/ttnn/ttnn/_ttnn.so || { echo "[t76] no t76 build; run setup76.sh" | tee -a $LOG; exit 4; }
T=models/tt_dit/tests/models/ltx/test_vae_ltx_fold_time_pad_ab.py
echo "[t76] arm=b0" | tee -a $LOG
AB_ARM=b0 timeout 600 python -m pytest -c $W/pytest.ini --rootdir=$W -sv --timeout=560 $T 2>&1 | tee -a $LOG
rc1=${PIPESTATUS[0]}
echo "T76_EXIT[b0]=$rc1" | tee -a $LOG
rc2=99
if [ $rc1 = 0 ]; then
  echo "[t76] arm=b128" | tee -a $LOG
  AB_ARM=b128 TT_NEIGHBOR_PAD_LOCAL_BATCH=128 \
    timeout 600 python -m pytest -c $W/pytest.ini --rootdir=$W -sv --timeout=560 $T 2>&1 | tee -a $LOG
  rc2=${PIPESTATUS[0]}
  echo "T76_EXIT[b128]=$rc2" | tee -a $LOG
fi
python /home/smarton/tray-stress/hostfmax.py 0 >/dev/null 2>&1
python - $V 2>&1 <<'PY' | tee -a $LOG
import sys, torch
V = sys.argv[1]
a, b = torch.load(f"{V}/yuv_b128.pt"), torch.load(f"{V}/yuv_b0.pt")
same = a.shape == b.shape and torch.equal(a, b)
d = (a.int() - b.int()).abs().max().item() if a.shape == b.shape else "shape"
print(f"T76_CMP b128_vs_b0 identical={same} max_abs_diff={d} shape={tuple(a.shape)} dtype={a.dtype}")
PY
[ $rc1 = 0 ] && [ $rc2 = 0 ]; rc=$?
echo "T76_EXIT=$rc" | tee -a $LOG
exit $rc
