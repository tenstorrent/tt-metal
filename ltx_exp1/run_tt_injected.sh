#!/bin/bash
# Experiment 1, ttnn side: served shape, seed 10, default prompt, reference noise injected.
# usage: run_tt_injected.sh <cfg>   cfg in: bf16 fp32 bf16_refemb fp32_refemb   (fp32 = LTX_EULER_FP32, refemb = reference connector embeddings)
cfg=$1; REF=/home/rsalman/tt-metal/ltx_exp1/ref_153f25_seed10
cd /home/rsalman/tt-metal
source /home/rsalman/tt-metal/python_env/bin/activate
export TT_METAL_HOME=/home/rsalman/tt-metal PYTHONPATH=/home/rsalman/tt-metal
export LTX_CHECKPOINT=/home/rsalman/.cache/huggingface/hub/models--Lightricks--LTX-2.3/snapshots/5948be4ced3a4493d1f836df64378ff136ddb770/ltx-2.3-22b-distilled-1.1.safetensors
export TT_DIT_CACHE_DIR=/home/rsalman/tt-metal/tt_dit_cache
export LTX_TRACED=0 NO_PROMPT=1 RUN_WARMUP=0 RUN_VBENCH=0 RUN_CLIP=0 HEIGHT=1088 WIDTH=1920 SEED=10 NUM_FRAMES=153 FPS=25
export LTX_INJECT_NOISE_DIR=$REF LTX_DUMP_LATENT_DIR=/home/rsalman/tt-metal/ltx_exp1/tt_$cfg
export LTX_DUMP_AUDIO_LATENT=/home/rsalman/tt-metal/ltx_exp1/tt_$cfg/audio_latent_final.pt
unset LTX_EULER_FP32 LTX_EMBEDS_OVERRIDE
case $cfg in *fp32*) export LTX_EULER_FP32=1;; esac
case $cfg in *refemb*) export LTX_EMBEDS_OVERRIDE=$REF/embeds.pt;; esac
T="models/tt_dit/tests/models/ltx/test_pipeline_ltx_distilled.py"; SEL="test_pipeline_distilled and 4x8sp1tp0nl2_ring_is_fsdp0 and not i2v"
mkdir -p ltx_exp1/tt_$cfg
echo "=== [$(date +%T)] tt injected $cfg ==="
python -m pytest "$T" -k "$SEL" -s --timeout 3600 > ltx_exp1/tt_$cfg/run.log 2>&1; echo "$cfg exit=$?"
[ -f ltx_av_fast_1920x1088_0.mp4 ] && mv ltx_av_fast_1920x1088_0.mp4 ltx_exp1/tt_$cfg/out.mp4
grep -h "Injecting\|Embeds override\|LTX_INJECT" ltx_exp1/tt_$cfg/run.log | head -4
