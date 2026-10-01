#!/bin/bash
# Verify the zero-allocation fp32 Euler step UNDER TRACING at the served shape. Same prompt/seed as exp3,
# so the result must match the untraced fp32 run (ltx_exp3/euler_fp32_153f25_seed10.mp4) closely.
# Requires the device to be free (stop the media server first).
cd /home/rsalman/tt-metal
holders=$(for d in /dev/tenstorrent/*; do fuser "$d" 2>/dev/null; done | tr -s ' \n' ' '); [ -n "$holders" ] && { echo "device busy: $holders"; exit 1; }
source python_env/bin/activate
export TT_METAL_HOME=/home/rsalman/tt-metal PYTHONPATH=/home/rsalman/tt-metal
export LTX_CHECKPOINT=/home/rsalman/.cache/huggingface/hub/models--Lightricks--LTX-2.3/snapshots/5948be4ced3a4493d1f836df64378ff136ddb770/ltx-2.3-22b-distilled-1.1.safetensors
export TT_DIT_CACHE_DIR=/home/rsalman/tt-metal/tt_dit_cache
export LTX_TRACED=1 LTX_EULER_FP32=1 NO_PROMPT=1 RUN_WARMUP=1 RUN_VBENCH=0 RUN_CLIP=0 HEIGHT=1088 WIDTH=1920 SEED=10 NUM_FRAMES=153 FPS=25
T="models/tt_dit/tests/models/ltx/test_pipeline_ltx_distilled.py"; SEL="test_pipeline_distilled and 4x8sp1tp0nl2_ring_is_fsdp0 and not i2v"
echo "=== [$(date +%T)] traced fp32 Euler @ 153f/25 ==="
LTX_DUMP_AUDIO_LATENT=/home/rsalman/tt-metal/ltx_exp3/audio_latent_fp32traced_153f25.pt python -m pytest "$T" -k "$SEL" -s --timeout 3600 > ltx_exp3/run_fp32traced_153f25.log 2>&1; echo "exit=$?"
[ -f ltx_av_fast_1920x1088_0.mp4 ] && mv ltx_av_fast_1920x1088_0.mp4 ltx_exp3/euler_fp32traced_153f25_seed10.mp4
grep -h "Euler step:\|latent\[s2\|unsafe" ltx_exp3/run_fp32traced_153f25.log | grep -v DEBUG | tail -4
python ltx_exp3/analyze.py 153f25 fp32 fp32traced
