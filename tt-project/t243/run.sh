#!/bin/bash
# t242 job (blx01 broker): t48 @94958f7acee + DIFFVAE_S5_PACKED_LANES (b8c2b3403f8), decode arms def (shipped defaults) then packed
# (DIFFVAE_S5_PACKED_LANES=1), each its own process: warm-up, 2 timed seeds, deep profile, host-noise seeds 0,1.
# Reuses the t238 Release build tree (python-only change). t240 job 882 took 209 s.
set -o pipefail
F=/var/tmp/fasth3; T=$F/t242; B=$F/t238/b; O=$T/out; mkdir -p $O; L=$O/run.log
mkdir -p $F/tmp
export HOME=$F/home XDG_CACHE_HOME=$F/home/.cache TMPDIR=$F/tmp TORCH_HOME=$F/home/.cache/torch HF_HOME=$F/home/.cache/huggingface
source $F/t48/python_env/bin/activate
export TT_METAL_HOME=$B PYTHONPATH=$B:$B/ttnn:$B/tools HF_HUB_OFFLINE=1 TT_METAL_CACHE=$F/cache/tt-metal-cache
export DIFFVAE_CHECKPOINT=/mnt/MLPerf/huggingface/hub/models--Lightricks--LTX-2.5/snapshots/28dac7acdc1f78a70e98687db261a949754f8941/vae/ltx-2.5-video-vae-bf16.safetensors
unset DIFFVAE_S5_2D DIFFVAE_NA_KEY_PHASE DIFFVAE_GNA_STRIDE DIFFVAE_NA_BRICK DIFFVAE_S5_LEAN DIFFVAE_NA_GATHER_REPHASE DIFFVAE_S5_PACKED_LANES DIFFVAE_TRACED DIFFVAE_NA_APPROX_EXP DIFFVAE_NA_FIDELITY
export SEEDS=0,1 HOST_SEEDS=0,1
cd $B
echo "[t242] host=$(hostname) $(date -u '+%F %T') UTC build=$(git -C $B rev-parse --short=11 HEAD)" | tee -a $L
cd $O
rc=0
for arm in def packed; do
  T0=$(date +%s)
  tmo=150
  [ $arm = packed ] && export DIFFVAE_S5_PACKED_LANES=1
  ARM=$arm timeout $tmo python -u $T/drv/decode242.py $F/diffvae/latents $O 2>&1 | tee -a $L
  r=${PIPESTATUS[0]}
  echo "[t242] arm=$arm rc=$r process wall $(($(date +%s) - T0)) s" | tee -a $L
  [ $r = 0 ] || rc=$r
done
echo "T242_EXIT=$rc" | tee -a $L
exit $rc
