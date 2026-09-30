#!/bin/bash
# Determinism floor: repeat the bf16 Euler run at 153f/25 with identical config.
cd /home/rsalman/tt-metal
source /home/rsalman/tt-metal/python_env/bin/activate
export TT_METAL_HOME=/home/rsalman/tt-metal PYTHONPATH=/home/rsalman/tt-metal
export LTX_CHECKPOINT=/home/rsalman/.cache/huggingface/hub/models--Lightricks--LTX-2.3/snapshots/5948be4ced3a4493d1f836df64378ff136ddb770/ltx-2.3-22b-distilled-1.1.safetensors
export TT_DIT_CACHE_DIR=/home/rsalman/tt-metal/tt_dit_cache
export LTX_TRACED=0 NO_PROMPT=1 RUN_WARMUP=0 RUN_VBENCH=0 RUN_CLIP=0 HEIGHT=1088 WIDTH=1920 SEED=10 NUM_FRAMES=153 FPS=25
unset LTX_EULER_FP32
T="models/tt_dit/tests/models/ltx/test_pipeline_ltx_distilled.py"; SEL="test_pipeline_distilled and 4x8sp1tp0nl2_ring_is_fsdp0 and not i2v"
echo "=== [$(date +%T)] exp3 bf16 REPEAT @ 153f/25fps ==="
LTX_DUMP_AUDIO_LATENT=/home/rsalman/tt-metal/ltx_exp3/audio_latent_bf16rep_153f25.pt python -m pytest "$T" -k "$SEL" -s --timeout 3600 > ltx_exp3/run_bf16rep_153f25.log 2>&1; echo "bf16 repeat exit=$?"
[ -f ltx_av_fast_1920x1088_0.mp4 ] && mv ltx_av_fast_1920x1088_0.mp4 ltx_exp3/euler_bf16rep_153f25_seed10.mp4
echo "=== [$(date +%T)] done ==="
