#!/bin/bash
# t238 job (blx01 broker): t48 head + DIFFVAE_S5_LEAN, decode arms def (no stage-5 knobs) then lean
# (DIFFVAE_S5_LEAN=1), each its own process: warm-up, 2 timed seeds (device noise), deep profile, host-noise seeds 0,1.
# t235's single default arm took 147 s.
set -o pipefail
F=/var/tmp/fasth3; T=$F/t238; B=$T/b; O=$T/out; mkdir -p $O; L=$O/run.log
mkdir -p $F/tmp
export HOME=$F/home XDG_CACHE_HOME=$F/home/.cache TMPDIR=$F/tmp TORCH_HOME=$F/home/.cache/torch HF_HOME=$F/home/.cache/huggingface
source $F/t48/python_env/bin/activate
export TT_METAL_HOME=$B PYTHONPATH=$B:$B/ttnn:$B/tools HF_HUB_OFFLINE=1 TT_METAL_CACHE=$F/cache/tt-metal-cache
export DIFFVAE_CHECKPOINT=/mnt/MLPerf/huggingface/hub/models--Lightricks--LTX-2.5/snapshots/28dac7acdc1f78a70e98687db261a949754f8941/vae/ltx-2.5-video-vae-bf16.safetensors
unset DIFFVAE_S5_2D DIFFVAE_NA_KEY_PHASE DIFFVAE_GNA_STRIDE DIFFVAE_NA_BRICK DIFFVAE_S5_LEAN
export SEEDS=0,1 HOST_SEEDS=0,1
cd $B
echo "[t238] host=$(hostname) $(date -u '+%F %T') UTC build=$(git -C $B rev-parse --short=11 HEAD)" | tee -a $L
cd $O
rc=0
for arm in def lean; do
  T0=$(date +%s)
  # the first arm may JIT-compile on a new build tree; the second runs warm
  [ $arm = def ] && tmo=320 || tmo=250
  [ $arm = lean ] && export DIFFVAE_S5_LEAN=1
  ARM=$arm timeout $tmo python -u $T/drv/decode238.py $F/diffvae/latents $O 2>&1 | tee -a $L
  r=${PIPESTATUS[0]}
  echo "[t238] arm=$arm rc=$r process wall $(($(date +%s) - T0)) s" | tee -a $L
  [ $r = 0 ] || rc=$r
done
echo "T238_EXIT=$rc" | tee -a $L
exit $rc
