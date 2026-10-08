#!/bin/bash
# t244 job (blx01 broker): t48 @34a571c5f47, ONE process, NA compute-config arms def, approx (DIFFVAE_NA_APPROX_EXP=1),
# lofi (DIFFVAE_NA_FIDELITY=lofi), each warm-up + 2 timed seeds + host-noise seeds 0,1; then a def re-time.
# Reuses the t238 Release build tree (python-only change). t242 job 886: ~105 s per arm with a fresh process each.
set -o pipefail
F=/var/tmp/fasth3; T=$F/t244; B=$F/t238/b; O=$T/out; mkdir -p $O; L=$O/run.log
mkdir -p $F/tmp
export HOME=$F/home XDG_CACHE_HOME=$F/home/.cache TMPDIR=$F/tmp TORCH_HOME=$F/home/.cache/torch HF_HOME=$F/home/.cache/huggingface
source $F/t48/python_env/bin/activate
export TT_METAL_HOME=$B PYTHONPATH=$B:$B/ttnn:$B/tools HF_HUB_OFFLINE=1 TT_METAL_CACHE=$F/cache/tt-metal-cache
export DIFFVAE_CHECKPOINT=/mnt/MLPerf/huggingface/hub/models--Lightricks--LTX-2.5/snapshots/28dac7acdc1f78a70e98687db261a949754f8941/vae/ltx-2.5-video-vae-bf16.safetensors
unset DIFFVAE_S5_2D DIFFVAE_NA_KEY_PHASE DIFFVAE_GNA_STRIDE DIFFVAE_NA_BRICK DIFFVAE_S5_LEAN DIFFVAE_NA_GATHER_REPHASE DIFFVAE_S5_PACKED_LANES DIFFVAE_TRACED DIFFVAE_NA_APPROX_EXP DIFFVAE_NA_FIDELITY
export SEEDS=0,1 HOST_SEEDS=0,1
cd $B
echo "[t244] host=$(hostname) $(date -u '+%F %T') UTC build=$(git -C $B rev-parse --short=11 HEAD)" | tee -a $L
cd $O
T0=$(date +%s)
timeout 310 python -u $T/drv/decode244.py $F/diffvae/latents $O 2>&1 | tee -a $L
rc=${PIPESTATUS[0]}
echo "[t244] rc=$rc process wall $(($(date +%s) - T0)) s" | tee -a $L
echo "T244_EXIT=$rc" | tee -a $L
exit $rc
