#!/bin/bash
# t335 broker job (blx01, full 4x8): conv VAE decode of the 5 saved LTX-2.5 latents, one python process per arm
# so the ttnn program cache cannot carry kernels across arms. Arms: default (LTX_VAE_CONV_FIDELITY unset = HiFi4),
# HiFi2, LoFi. Usage (broker -e env.yaml -t 600): run335.sh <tag> <arm> [<arm>...]
# Code: blx01 lean t48 build (bf7db12a149) + Python overlay of ttp/t48-ltx25-integrated f6547442b30; the C++
# differs only by a conv3d host guard and the DiffVAE-only neighborhood SDPA, so conv3d kernels are the same.
if [ -z "$INNER" ]; then
  INNER=1 setsid bash "$0" "$@" & PG=$!
  trap 'kill -TERM -- -$PG 2>/dev/null; sleep 5; kill -KILL -- -$PG 2>/dev/null' EXIT
  trap 'exit 143' TERM; trap 'exit 130' INT
  wait $PG; exit $?
fi
set -o pipefail
TAG=${1:?tag}; shift; F=/var/tmp/fasth3; D=$F/t335; W=$F/t48; O=$D/ov; OUT=$D/out_$TAG
# blx01 caps: / at most 70% used, project data under /var/tmp/fasth3 at most 150G.
use=$(df --output=pcent / | tail -1 | tr -dc 0-9); [ "$use" -le 70 ] || { echo "df / $use% > 70%"; exit 5; }
gb=$(timeout 120 du -sxBG $F | cut -f1 | tr -dc 0-9); [ "${gb:-0}" -le 150 ] || { echo "$F ${gb}G > 150G"; exit 5; }
mkdir -p $OUT $D/tmp
export HOME=$F/home XDG_CACHE_HOME=$F/home/.cache TMPDIR=$D/tmp TORCH_HOME=$F/home/.cache/torch
source $W/python_env/bin/activate
export TT_METAL_HOME=$W PYTHONPATH=$O:$W:$W/ttnn:$W/tools HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export TT_METAL_CACHE=$F/cache/tt-metal-cache
export VAE_CKPT=${VAE_CKPT:-$F/models/ltx-2.5/vae/ltx-2.5-video-vae-conv-bf16.safetensors}
L=$OUT/run.log
{ echo "[t335] tag=$TAG arms=$* host=$(hostname) build=$(git -C $W rev-parse --short=11 HEAD 2>/dev/null) py=$(cat $O/OVERLAY_COMMIT) $(date -u '+%F %T') UTC"
  env | grep -E '^(LTX_|TT_|VAE_|PYTHONPATH)' | sort; } | tee $L
[ "$(cat $O/OVERLAY_COMMIT)" = f6547442b30 ] || { echo "wrong overlay" | tee -a $L; exit 3; }
for f in $W/ttnn/ttnn/_ttnn.so $VAE_CKPT $D/latents/seed0.pt $D/dec335.py; do
  [ -r $f ] || { echo "missing $f" | tee -a $L; exit 3; }
done
rc=0
for A in "$@"; do
  case $A in default) unset LTX_VAE_CONV_FIDELITY;; LoFi|HiFi2|HiFi3|HiFi4) export LTX_VAE_CONV_FIDELITY=$A;; *) echo "bad arm $A"; exit 2;; esac
  T0=$(date +%s)
  timeout ${ARM_S:-280} python -u $D/dec335.py $D/latents $OUT/$A 2>&1 | tee -a $L
  r=${PIPESTATUS[0]}
  echo "[t335] arm=$A rc=$r process wall $(( $(date +%s) - T0 )) s" | tee -a $L
  [ $r = 0 ] || { rc=$r; break; }
done
echo "T335_EXIT=$rc" | tee -a $L
exit $rc
