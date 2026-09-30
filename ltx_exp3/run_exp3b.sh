#!/bin/bash
# Experiment 3 continued: fp32 Euler at 145f/24 (pairs with the finished bf16 run), then bf16+fp32 at the served 153f/25 shape.
cd /home/rsalman/tt-metal
source /home/rsalman/tt-metal/python_env/bin/activate
export TT_METAL_HOME=/home/rsalman/tt-metal PYTHONPATH=/home/rsalman/tt-metal
export LTX_CHECKPOINT=/home/rsalman/.cache/huggingface/hub/models--Lightricks--LTX-2.3/snapshots/5948be4ced3a4493d1f836df64378ff136ddb770/ltx-2.3-22b-distilled-1.1.safetensors
export TT_DIT_CACHE_DIR=/home/rsalman/tt-metal/tt_dit_cache
export LTX_TRACED=0 NO_PROMPT=1 RUN_WARMUP=0 RUN_VBENCH=0 RUN_CLIP=0 HEIGHT=1088 WIDTH=1920 SEED=10
T="models/tt_dit/tests/models/ltx/test_pipeline_ltx_distilled.py"; SEL="test_pipeline_distilled and 4x8sp1tp0nl2_ring_is_fsdp0 and not i2v"
run() { # mode frames fps tag
  if [ $1 = fp32 ]; then export LTX_EULER_FP32=1; else unset LTX_EULER_FP32; fi
  export NUM_FRAMES=$2 FPS=$3
  echo "=== [$(date +%T)] exp3 $1 Euler step @ ${2}f/${3}fps ==="
  LTX_DUMP_AUDIO_LATENT=/home/rsalman/tt-metal/ltx_exp3/audio_latent_$1_$4.pt python -m pytest "$T" -k "$SEL" -s --timeout 3600 > ltx_exp3/run_$1_$4.log 2>&1; echo "$1 $4 exit=$?"
  [ -f ltx_av_fast_1920x1088_0.mp4 ] && mv ltx_av_fast_1920x1088_0.mp4 ltx_exp3/euler_$1_$4_seed10.mp4
}
run fp32 145 24 145f24
run bf16 153 25 153f25
run fp32 153 25 153f25
echo "=== [$(date +%T)] all done ==="
