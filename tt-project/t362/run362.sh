#!/bin/bash
# t362 one broker job on blx01 (4x8): rgb_to_yuv patch-input unit test, then the 1088x1920/145f decode A/B with
# the real LTX-2.5 conv VAE on diffvae/latents/seed0-4, each arm (LTX_VAE_FUSE_UNPATCH=0, then 1, both with
# LTX_FUSE_YUV_OUTPUT=1) in its own pytest process; then md5/PCC/PSNR per seed.
if [ -z "$INNER" ]; then
  INNER=1 setsid bash "$0" "$@" & PG=$!
  trap 'kill -TERM -- -$PG 2>/dev/null; sleep 5; kill -KILL -- -$PG 2>/dev/null' EXIT
  trap 'exit 143' TERM; trap 'exit 130' INT
  wait $PG; exit $?
fi
F=/var/tmp/fasth3; D=$F/t362; B=$D/b; OUT=$D/out
use=$(df --output=pcent / | tail -1 | tr -dc 0-9); [ "$use" -le 70 ] || { echo "[t362] / at $use%"; exit 5; }
gb=$(timeout 120 du -sxBG $F | cut -f1 | tr -dc 0-9); [ "${gb:-0}" -le 150 ] || { echo "[t362] $F ${gb}G"; exit 5; }
cd $B; source $F/t48/python_env/bin/activate
export TT_METAL_HOME=$B PYTHONPATH=$B:$B/ttnn:$B/tools HF_HUB_OFFLINE=1 TT_METAL_CACHE=$D/jit AB_OUT_DIR=$OUT/yuv
export VAE_CKPT=$F/models/ltx-2.5/vae/ltx-2.5-video-vae-conv-bf16.safetensors AB_LATENT_DIR=$F/diffvae/latents
export LTX_FUSE_YUV_OUTPUT=1 LTX_PIN_CORES=0 PYTHONDONTWRITEBYTECODE=1
for f in $VAE_CKPT $AB_LATENT_DIR/seed4.pt $B/ttnn/ttnn/_ttnn.so; do [ -r $f ] || { echo "[t362] missing $f"; exit 3; }; done
mkdir -p $OUT; R=$OUT/run.log
echo "[t362] host=$(hostname) commit=$(git rev-parse HEAD) $(date -u '+%F %T')" | tee $R
T0=$(date +%s)
python -u -m pytest -sv --timeout=150 models/tt_dit/tests/unit/test_rgb_to_yuv_patch_input.py > $OUT/unit.log 2>&1; r=$?
echo "[t362] unit rc=$r wall=$(( $(date +%s) - T0 )) s" | tee -a $R
grep -E 'PASSED|FAILED|Error' $OUT/unit.log | tee -a $R
rc=0
for arm in 0 1; do
  T0=$(date +%s)
  LTX_VAE_FUSE_UNPATCH=$arm python -u -m pytest -sv --timeout=200 \
    models/tt_dit/tests/models/ltx/test_vae_ltx_fuse_unpatch_ab.py > $OUT/arm$arm.log 2>&1; r=$?
  echo "[t362] arm=$arm rc=$r wall=$(( $(date +%s) - T0 )) s" | tee -a $R
  grep -E '^AB |PASSED|FAILED|Error' $OUT/arm$arm.log | tee -a $R
  [ $r -ne 0 ] && { rc=$r; break; }
done
if [ $rc -eq 0 ]; then
  python - $OUT/yuv <<'PY' 2>&1 | tee -a $R
import sys, torch
d = sys.argv[1]
for s in range(5):
    a = torch.load(f"{d}/yuv_r0_s{s}.pt").double().flatten(); b = torch.load(f"{d}/yuv_r1_s{s}.pt").double().flatten()
    same = torch.equal(a, b); pcc = torch.corrcoef(torch.stack([a, b]))[0, 1].item() if not same else 1.0
    mse = (a - b).pow(2).mean().item()
    psnr = float("inf") if mse == 0 else 10 * torch.log10(torch.tensor(255.0 ** 2 / mse)).item()
    print(f"[t362] seed={s} identical={same} pcc={pcc:.6f} psnr={psnr:.2f} maxabs={(a - b).abs().max().item():.4g}")
PY
fi
rm -rf $OUT/yuv
echo "T362_EXIT=$rc" | tee -a $R
exit $rc
