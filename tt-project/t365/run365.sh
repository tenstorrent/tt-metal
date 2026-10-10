#!/bin/bash
# t365 one broker job on blx01 (4x8): 1088x1920/145f conv VAE decode A/B with the real LTX-2.5 conv VAE on
# diffvae/latents/seed0-4. Arm y0: LTX_FUSE_YUV_OUTPUT=0 (old default: unpatch to BCTHW + fast_device_to_host_yuv).
# Arm y1: LTX_FUSE_YUV_OUTPUT=1 (new default: fused YUV path, conv_out unpatch fused into rgb_to_yuv). Each arm in
# its own pytest process; then md5/PCC/PSNR per seed.
if [ -z "$INNER" ]; then
  INNER=1 setsid bash "$0" "$@" & PG=$!
  trap 'kill -TERM -- -$PG 2>/dev/null; sleep 5; kill -KILL -- -$PG 2>/dev/null' EXIT
  trap 'exit 143' TERM; trap 'exit 130' INT
  wait $PG; exit $?
fi
F=/var/tmp/fasth3; D=$F/t365; B=$D/b; OUT=$D/out
use=$(df --output=pcent / | tail -1 | tr -dc 0-9); [ "$use" -le 70 ] || { echo "[t365] / at $use%"; exit 5; }
gb=$(timeout 120 du -sxBG $F | cut -f1 | tr -dc 0-9); [ "${gb:-0}" -le 150 ] || { echo "[t365] $F ${gb}G"; exit 5; }
cd $B; source $F/t48/python_env/bin/activate
export TT_METAL_HOME=$B PYTHONPATH=$B:$B/ttnn:$B/tools HF_HUB_OFFLINE=1 TT_METAL_CACHE=$D/jit AB_OUT_DIR=$OUT/yuv
export VAE_CKPT=$F/models/ltx-2.5/vae/ltx-2.5-video-vae-conv-bf16.safetensors AB_LATENT_DIR=$F/diffvae/latents
export LTX_VAE_FUSE_UNPATCH=1 LTX_PIN_CORES=0 PYTHONDONTWRITEBYTECODE=1
for f in $VAE_CKPT $AB_LATENT_DIR/seed4.pt $B/ttnn/ttnn/_ttnn.so; do [ -r $f ] || { echo "[t365] missing $f"; exit 3; }; done
mkdir -p $OUT; R=$OUT/run.log
echo "[t365] host=$(hostname) commit=$(git rev-parse HEAD) $(date -u '+%F %T')" | tee $R
rc=0
for arm in 0 1; do
  T0=$(date +%s)
  LTX_FUSE_YUV_OUTPUT=$arm python -u -m pytest -sv --timeout=150 \
    models/tt_dit/tests/models/ltx/test_vae_ltx_fuse_unpatch_ab.py > $OUT/arm$arm.log 2>&1; r=$?
  echo "[t365] arm=y$arm rc=$r wall=$(( $(date +%s) - T0 )) s" | tee -a $R
  grep -E '^AB |PASSED|FAILED|Error' $OUT/arm$arm.log | tee -a $R
  [ $r -ne 0 ] && { rc=$r; break; }
done
if [ $rc -eq 0 ]; then
  python - $OUT/yuv <<'PY' 2>&1 | tee -a $R
import sys, torch
d = sys.argv[1]
for s in range(5):
    a = torch.load(f"{d}/yuv_y0r1_s{s}.pt").double().flatten(); b = torch.load(f"{d}/yuv_y1r1_s{s}.pt").double().flatten()
    same = torch.equal(a, b); pcc = torch.corrcoef(torch.stack([a, b]))[0, 1].item() if not same else 1.0
    mse = (a - b).pow(2).mean().item()
    psnr = float("inf") if mse == 0 else 10 * torch.log10(torch.tensor(255.0 ** 2 / mse)).item()
    print(f"[t365] seed={s} identical={same} pcc={pcc:.6f} psnr={psnr:.2f} maxabs={(a - b).abs().max().item():.4g}")
PY
fi
rm -rf $OUT/yuv
echo "T365_EXIT=$rc" | tee -a $R
exit $rc
