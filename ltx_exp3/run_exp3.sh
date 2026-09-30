#!/bin/bash
# Experiment 3: bf16 Euler step (current) vs fp32-accumulate Euler step (reference style), same prompt/seed.
cd /home/rsalman/tt-metal
source /home/rsalman/tt-metal/python_env/bin/activate
export TT_METAL_HOME=/home/rsalman/tt-metal PYTHONPATH=/home/rsalman/tt-metal
export LTX_CHECKPOINT=/home/rsalman/.cache/huggingface/hub/models--Lightricks--LTX-2.3/snapshots/5948be4ced3a4493d1f836df64378ff136ddb770/ltx-2.3-22b-distilled-1.1.safetensors
export TT_DIT_CACHE_DIR=/home/rsalman/tt-metal/tt_dit_cache
export LTX_TRACED=0 NO_PROMPT=1 RUN_WARMUP=0 RUN_VBENCH=0 RUN_CLIP=0 NUM_FRAMES=145 FPS=24 HEIGHT=1088 WIDTH=1920 SEED=10
T="models/tt_dit/tests/models/ltx/test_pipeline_ltx_distilled.py"; SEL="test_pipeline_distilled and 4x8sp1tp0nl2_ring_is_fsdp0 and not i2v"
for mode in bf16 fp32; do
  if [ $mode = fp32 ]; then export LTX_EULER_FP32=1; else unset LTX_EULER_FP32; fi
  echo "=== [$(date +%T)] exp3 $mode Euler step ==="
  LTX_DUMP_AUDIO_LATENT=/home/rsalman/tt-metal/ltx_exp3/audio_latent_$mode.pt python -m pytest "$T" -k "$SEL" -s --timeout 3600 > ltx_exp3/run_$mode.log 2>&1; echo "$mode exit=$?"
  [ -f ltx_av_fast_1920x1088_0.mp4 ] && mv ltx_av_fast_1920x1088_0.mp4 ltx_exp3/euler_${mode}_145f24_guitar_seed10.mp4
  grep -h "Euler step:\|latent\[s2" ltx_exp3/run_$mode.log | grep -v DEBUG | tail -3
done
echo "=== [$(date +%T)] done ==="
