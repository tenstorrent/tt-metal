#!/bin/bash
# t249 job (blx01 broker): t238/b @34a571c5f47 + approx exp (= t48 @946a37952bd), ONE python process per arm (the NA program
# hash ignores compute_kernel_config, so arms must not share a process). ARMLIST (arg 1) e.g. "def approx lofi";
# OUT (arg 2) the output dir. Each process: load, warm-up, 2 timed seeds, host-noise seeds 0,1 (~100 s).
set -o pipefail
F=/var/tmp/fasth3; T=$F/t249; B=$F/t238/b; ARMLIST=$1; O=$2; mkdir -p $O; L=$O/run.log
mkdir -p $F/tmp
export HOME=$F/home XDG_CACHE_HOME=$F/home/.cache TMPDIR=$F/tmp TORCH_HOME=$F/home/.cache/torch HF_HOME=$F/home/.cache/huggingface
source $F/t48/python_env/bin/activate
export TT_METAL_HOME=$B PYTHONPATH=$B:$B/ttnn:$B/tools HF_HUB_OFFLINE=1 TT_METAL_CACHE=$F/cache/tt-metal-cache
export DIFFVAE_CHECKPOINT=/mnt/MLPerf/huggingface/hub/models--Lightricks--LTX-2.5/snapshots/28dac7acdc1f78a70e98687db261a949754f8941/vae/ltx-2.5-video-vae-bf16.safetensors
unset DIFFVAE_S5_2D DIFFVAE_NA_KEY_PHASE DIFFVAE_GNA_STRIDE DIFFVAE_NA_BRICK DIFFVAE_S5_LEAN DIFFVAE_NA_GATHER_REPHASE DIFFVAE_S5_PACKED_LANES DIFFVAE_TRACED DIFFVAE_NA_APPROX_EXP DIFFVAE_NA_FIDELITY DIFFVAE_NA_CHUNK_BRICKS DIFFVAE_NA_KV_CHUNK_TILES
export SEEDS=0,1 HOST_SEEDS=0,1
head=$(git -C $B rev-parse --short=11 HEAD)
echo "[t249] host=$(hostname) $(date -u '+%F %T') UTC build=$head arms=$ARMLIST" | tee -a $L
[ "$head" = 34a571c5f47 ] || { echo "[t249] build tree moved off 34a571c5f47"; echo "T249_EXIT=9" | tee -a $L; exit 9; }
cd $O
rc=0
for arm in $ARMLIST; do
  T0=$(date +%s)
  ARMS=$arm timeout 170 python -u $T/drv/decode249.py $F/diffvae/latents $O 2>&1 | tee -a $L
  r=${PIPESTATUS[0]}
  echo "[t249] arm=$arm rc=$r process wall $(($(date +%s) - T0)) s" | tee -a $L
  [ $r = 0 ] || rc=$r
done
echo "T249_EXIT=$rc" | tee -a $L
exit $rc
