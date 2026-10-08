#!/bin/bash
# t241 job (blx01 broker): t48 + traced decode with noise as input, arms def (eager) then traced (DIFFVAE_TRACED=1)
# each its own process: warm-up, (capture), 2 timed seeds x2, host-noise seeds 0,1. t240 job 882 took 209 s.
set -o pipefail
F=/var/tmp/fasth3; T=$F/t241; B=$F/t238/b; O=$T/out; mkdir -p $O; L=$O/run.log
mkdir -p $F/tmp
export HOME=$F/home XDG_CACHE_HOME=$F/home/.cache TMPDIR=$F/tmp TORCH_HOME=$F/home/.cache/torch HF_HOME=$F/home/.cache/huggingface
source $F/t48/python_env/bin/activate
export TT_METAL_HOME=$B PYTHONPATH=$B:$B/ttnn:$B/tools HF_HUB_OFFLINE=1 TT_METAL_CACHE=$F/cache/tt-metal-cache
export DIFFVAE_CHECKPOINT=/mnt/MLPerf/huggingface/hub/models--Lightricks--LTX-2.5/snapshots/28dac7acdc1f78a70e98687db261a949754f8941/vae/ltx-2.5-video-vae-bf16.safetensors
unset DIFFVAE_TRACED DIFFVAE_S5_2D DIFFVAE_NA_KEY_PHASE DIFFVAE_GNA_STRIDE DIFFVAE_NA_BRICK DIFFVAE_S5_LEAN DIFFVAE_NA_GATHER_REPHASE
export SEEDS=0,1 HOST_SEEDS=0,1
cd $B
echo "[t241] host=$(hostname) $(date -u '+%F %T') UTC build=$(git -C $B rev-parse --short=11 HEAD)" | tee -a $L
cd $O
rc=0
for arm in def traced; do
  T0=$(date +%s)
  tmo=150
  [ $arm = traced ] && export DIFFVAE_TRACED=1
  ARM=$arm timeout $tmo python -u $T/drv/decode241.py $F/diffvae/latents $O 2>&1 | tee -a $L
  r=${PIPESTATUS[0]}
  echo "[t241] arm=$arm rc=$r process wall $(($(date +%s) - T0)) s" | tee -a $L
  [ $r = 0 ] || rc=$r
done
echo "T241_EXIT=$rc" | tee -a $L
exit $rc
