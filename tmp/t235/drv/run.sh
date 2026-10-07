#!/bin/bash
# t235 job (blx01 broker): the t232 build with the t235 python overlay (key phase on by default), decode with NO
# stage-5 env knobs set: warm-up, 2 timed seeds (device noise), deep profile, host-noise seeds 0,1 (compared to
# #232's kp1 yuvs afterwards). One process; #232's kp1 arm took 147 s.
set -o pipefail
F=/var/tmp/fasth3; T=$F/t235; B=$F/t232/b; OV=$T/ov; O=$T/out; mkdir -p $O; L=$O/run.log
mkdir -p $F/tmp
export HOME=$F/home XDG_CACHE_HOME=$F/home/.cache TMPDIR=$F/tmp TORCH_HOME=$F/home/.cache/torch HF_HOME=$F/home/.cache/huggingface
source $F/t48/python_env/bin/activate
export TT_METAL_HOME=$B PYTHONPATH=$OV:$B/ttnn:$B/tools HF_HUB_OFFLINE=1 TT_METAL_CACHE=$F/cache/tt-metal-cache
export DIFFVAE_CHECKPOINT=/mnt/MLPerf/huggingface/hub/models--Lightricks--LTX-2.5/snapshots/28dac7acdc1f78a70e98687db261a949754f8941/vae/ltx-2.5-video-vae-bf16.safetensors
unset DIFFVAE_S5_2D DIFFVAE_NA_KEY_PHASE DIFFVAE_GNA_STRIDE DIFFVAE_NA_BRICK
export SEEDS=0,1 HOST_SEEDS=0,1
cd $OV
echo "[t235] host=$(hostname) $(date -u '+%F %T') UTC build=$(git -C $B rev-parse --short=11 HEAD) overlay=$(md5sum models/tt_dit/layers/neighborhood_attention*.py | cut -c1-8 | tr '\n' ' ')" | tee -a $L
python -c 'from models.tt_dit.layers.neighborhood_attention_plan import key_phase_applies as k; print("[t235] key_phase_applies(1080p 2-D) =", k((145,272,480),(11,11,11),60,8,68,4))' 2>&1 | tail -1 | tee -a $L
cd $O
T0=$(date +%s)
ARM=def timeout 220 python -u $T/drv/decode235.py $F/diffvae/latents $O 2>&1 | tee -a $L
r=${PIPESTATUS[0]}
echo "[t235] arm=def rc=$r process wall $(( $(date +%s) - T0 )) s" | tee -a $L
echo "T235_EXIT=$r" | tee -a $L
exit $r
