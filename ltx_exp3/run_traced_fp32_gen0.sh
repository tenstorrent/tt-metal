#!/bin/bash
# After run_discriminate.sh: traced fp32 again, now with per-gen latent dumps, so gen 0's audio latent can be
# compared to the untraced fp32 latent and decoded on CPU (torch oracle) against the gen-0 mp4 audio.
cd /home/rsalman/tt-metal
while pgrep -f "run_discriminate.sh|run_exp3_traced_fp32_nowarm.sh" >/dev/null; do sleep 10; done
source python_env/bin/activate
export TT_METAL_HOME=/home/rsalman/tt-metal PYTHONPATH=/home/rsalman/tt-metal
export LTX_CHECKPOINT=/home/rsalman/.cache/huggingface/hub/models--Lightricks--LTX-2.3/snapshots/5948be4ced3a4493d1f836df64378ff136ddb770/ltx-2.3-22b-distilled-1.1.safetensors
export TT_DIT_CACHE_DIR=/home/rsalman/tt-metal/tt_dit_cache
export LTX_TRACED=1 LTX_EULER_FP32=1 NO_PROMPT=1 RUN_WARMUP=0 RUN_VBENCH=0 RUN_CLIP=0 HEIGHT=1088 WIDTH=1920 SEED=10 NUM_FRAMES=153 FPS=25
T="models/tt_dit/tests/models/ltx/test_pipeline_ltx_distilled.py"; SEL="test_pipeline_distilled and 4x8sp1tp0nl2_ring_is_fsdp0 and not i2v"
rm -f ltx_exp3/audio_latent_fp32traced3_153f25*.pt
echo "=== [$(date +%T)] traced fp32, per-gen dumps ==="
LTX_DUMP_AUDIO_LATENT=/home/rsalman/tt-metal/ltx_exp3/audio_latent_fp32traced3_153f25.pt python -m pytest "$T" -k "$SEL" -s --timeout 3600 > ltx_exp3/run_fp32traced3_153f25.log 2>&1; echo "fp32traced3 exit=$?"
[ -f ltx_av_fast_1920x1088_0.mp4 ] && mv ltx_av_fast_1920x1088_0.mp4 ltx_exp3/euler_fp32traced3_153f25_seed10.mp4
ls ltx_exp3/audio_latent_fp32traced3_153f25*.pt
python ltx_exp3/analyze.py 153f25 fp32 fp32traced3
echo "=== [$(date +%T)] done ==="
