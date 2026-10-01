#!/bin/bash
# blx03 broker job for #82: A/B of the BH SFPU constant hoists (PR 58179 + 58217, header-only) on a 2x4
# submesh of the full 4x8 mesh. Mode "vae": conv VAE decode 544x960/145f, 1 warmup + 3 timed, YUV out.
# Mode "block": traced LTX AV block, 2x4 Linear sp1/tp0, 10x34x60. Arm "base" runs the t48 runtime
# root, arm "hoist" the t82rt overlay (setup82.sh); separate JIT caches so nothing is shared.
# Usage on blx03: cd ~/fasth3/tt-metal && tmp/blx03/submit.sh 1500 bash /home/smarton/fasth3/t48/tmp/t82/run82.sh vae
MODE=$1
BASE=/home/smarton/fasth3/tt-metal; W=/home/smarton/fasth3/t48; V=/var/tmp/fasth3/t82
LOG=$V/run82_$MODE.log
source $BASE/python_env/bin/activate
export TT_METAL_HOME=$W PYTHONPATH=$W:$W/ttnn:$W/tools HF_HUB_OFFLINE=1
export LTX_FUSE_YUV_OUTPUT=1 AB_LATENT=/home/smarton/fasth3/out/t37/s2reuse0/lat.gen0.pt
cd $W
case $MODE in
  vae) T=models/tt_dit/tests/models/ltx/test_vae_ltx_fold_time_pad_ab.py ;;
  block) T=tmp/t82/test_block_sfpu_ab.py ;;
  *) echo "mode vae|block"; exit 2 ;;
esac
echo "[t82] mode=$MODE tree=$(git rev-parse --short HEAD) clock: $(python /home/smarton/tray-stress/hostfmax.py 1150 | tail -1)" | tee $LOG
test -f "$AB_LATENT" || { echo "[t82] $AB_LATENT missing" | tee -a $LOG; exit 3; }
test -f /home/smarton/fasth3/t82rt/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h || { echo "[t82] no overlay" | tee -a $LOG; exit 4; }
rc=0
for arm in base hoist; do
  root=$W; [ $arm = hoist ] && root=/home/smarton/fasth3/t82rt
  mkdir -p $V/$MODE/$arm
  echo "[t82] arm=$arm root=$root" | tee -a $LOG
  TT_METAL_RUNTIME_ROOT=$root TT_METAL_CACHE=$V/jit_$arm AB_OUT_DIR=$V/$MODE/$arm AB_ARM=$arm \
    timeout 650 python -m pytest -c $W/pytest.ini --rootdir=$W -sv --timeout=620 $T 2>&1 | tee -a $LOG
  r=${PIPESTATUS[0]}
  echo "T82_EXIT[$MODE/$arm]=$r" | tee -a $LOG
  [ $r = 0 ] || { rc=$r; break; }
done
python /home/smarton/tray-stress/hostfmax.py 0 >/dev/null 2>&1
[ $rc = 0 ] && python - $V/$MODE 2>&1 <<'PY' | tee -a $LOG
import glob, sys, torch
d = sys.argv[1]
a, b = (torch.load(glob.glob(f"{d}/{arm}/*.pt")[0]) for arm in ("base", "hoist"))
if not isinstance(a, dict):
    a, b = {"out": a}, {"out": b}
ok = True
for k in a:
    x, y = a[k], b[k]
    same = x.shape == y.shape and torch.equal(x, y)
    diff = (x.float() - y.float()).abs().max().item() if x.shape == y.shape else "shape"
    ok &= same
    print(f"T82_CMP {k} identical={same} max_abs_diff={diff} shape={tuple(x.shape)} dtype={x.dtype}")
print(f"T82_IDENTICAL={ok}")
PY
echo "T82_EXIT=$rc" | tee -a $LOG
exit $rc
