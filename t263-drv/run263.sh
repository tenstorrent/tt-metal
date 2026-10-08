#!/bin/bash
# t263 job (blx01 broker): build $F/t263/b (6a23f8dfe10 C++) checked out @REV = + DIFFVAE_NA_BF8 + bf8 mask-stride fix (python/kernel only).
# ONE python process per arm, SEEDS=0, no host-noise decodes (ablated arms are garbage). Args: "name:K=V name2:..." OUT [PROFILE_ARMS] [SCORE_ARMS: HOST_SEEDS=0,1]
set -o pipefail
F=/var/tmp/fasth3; T=$F/t263; B=$F/t263/b; ARMLIST=$1; O=$2; PARMS=" $3 "; SARMS=" $4 "; mkdir -p $O; L=$O/run.log
mkdir -p $F/tmp
export HOME=$F/home XDG_CACHE_HOME=$F/home/.cache TMPDIR=$F/tmp TORCH_HOME=$F/home/.cache/torch HF_HOME=$F/home/.cache/huggingface
source $F/t48/python_env/bin/activate
export TT_METAL_HOME=$B PYTHONPATH=$B:$B/ttnn:$B/tools HF_HUB_OFFLINE=1 TT_METAL_CACHE=$F/cache/tt-metal-cache
for v in $(env | sed -n 's/^\(DIFFVAE_[A-Z0-9_]*\)=.*/\1/p'); do unset $v; done
export DIFFVAE_CHECKPOINT=/mnt/MLPerf/huggingface/hub/models--Lightricks--LTX-2.5/snapshots/28dac7acdc1f78a70e98687db261a949754f8941/vae/ltx-2.5-video-vae-bf16.safetensors
export SEEDS=0 HOST_SEEDS=
head=$(git -C $B rev-parse --short=11 HEAD)
REV=51ecd3ac3e2
echo "[t263] host=$(hostname) $(date -u '+%F %T') UTC build=$head arms=$ARMLIST" | tee -a $L
[ "$head" = $REV ] || { echo "[t263] build tree moved off $REV"; echo "T263_EXIT=9" | tee -a $L; exit 9; }
cd $O
rc=0
for a in $ARMLIST; do
  name=${a%%:*}; envs=${a#*:}; [ "$envs" = "$a" ] && envs=
  case "$PARMS" in *" $name "*) P=1 ;; *) P=0 ;; esac
  case "$SARMS" in *" $name "*) HS=0,1 ;; *) HS= ;; esac
  T0=$(date +%s)
  ARM=$name ARM_ENV=$envs PROFILE=$P HOST_SEEDS=$HS timeout 300 python -u $T/drv/decode261.py $F/diffvae/latents $O 2>&1 | tee -a $L
  r=${PIPESTATUS[0]}
  echo "[t263] arm=$name rc=$r process wall $(($(date +%s) - T0)) s" | tee -a $L
  [ $r = 0 ] || rc=$r
done
echo "T263_EXIT=$rc" | tee -a $L
exit $rc
