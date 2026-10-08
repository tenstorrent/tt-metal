#!/bin/bash
# t253 job (blx01 broker): C++ from t238/b @34a571c5f47 (no C++ change since, except the NA program-hash fix,
# moot with one arm per process); python models/ overlay $T/src/<SRC> (a t253 commit). ONE process per arm.
# Args: SRC "name:K=V,K=V name2:..." OUT [PROFILE_ARMS]
set -o pipefail
F=/var/tmp/fasth3; T=$F/t253; B=$F/t238/b; SRC=$1; ARMLIST=$2; O=$3; PARMS=" $4 "; mkdir -p $O; L=$O/run.log
mkdir -p $F/tmp
export HOME=$F/home XDG_CACHE_HOME=$F/home/.cache TMPDIR=$F/tmp TORCH_HOME=$F/home/.cache/torch HF_HOME=$F/home/.cache/huggingface
source $F/t48/python_env/bin/activate
export TT_METAL_HOME=$B PYTHONPATH=$T/src/$SRC:$B:$B/ttnn:$B/tools HF_HUB_OFFLINE=1 TT_METAL_CACHE=$F/cache/tt-metal-cache
for v in $(env | sed -n 's/^\(DIFFVAE_[A-Z0-9_]*\)=.*/\1/p'); do unset $v; done
export DIFFVAE_CHECKPOINT=/mnt/MLPerf/huggingface/hub/models--Lightricks--LTX-2.5/snapshots/28dac7acdc1f78a70e98687db261a949754f8941/vae/ltx-2.5-video-vae-bf16.safetensors
export SEEDS=0,1 HOST_SEEDS=0,1
head=$(git -C $B rev-parse --short=11 HEAD)
echo "[t253] host=$(hostname) $(date -u '+%F %T') UTC build=$head src=$SRC arms=$ARMLIST" | tee -a $L
[ "$head" = 34a571c5f47 ] || { echo "[t253] build tree moved off 34a571c5f47"; echo "T253_EXIT=9" | tee -a $L; exit 9; }
[ -d $T/src/$SRC/models/tt_dit ] || { echo "T253_EXIT=8" | tee -a $L; exit 8; }
cd $O
rc=0
for a in $ARMLIST; do
  name=${a%%:*}; envs=${a#*:}; [ "$envs" = "$a" ] && envs=
  case "$PARMS" in *" $name "*) P=1 ;; *) P=0 ;; esac
  T0=$(date +%s)
  ARM=$name ARM_ENV=$envs PROFILE=$P timeout 200 python -u $T/drv/decode253.py $F/diffvae/latents $O 2>&1 | tee -a $L
  r=${PIPESTATUS[0]}
  echo "[t253] arm=$name rc=$r process wall $(($(date +%s) - T0)) s" | tee -a $L
  [ $r = 0 ] || rc=$r
done
echo "T253_EXIT=$rc" | tee -a $L
exit $rc
