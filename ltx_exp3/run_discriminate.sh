#!/bin/bash
# After the in-flight traced fp32 run: (1) traced bf16 (flag off) and (2) untraced fp32 with today's binary.
# Tells apart "fp32 scratch changes the allocation layout and breaks the audio decode trace" from
# "today's binary / cache breaks audio regardless of the flag".
cd /home/rsalman/tt-metal
while pgrep -f run_exp3_traced_fp32_nowarm.sh >/dev/null; do sleep 10; done
source python_env/bin/activate
export TT_METAL_HOME=/home/rsalman/tt-metal PYTHONPATH=/home/rsalman/tt-metal
export LTX_CHECKPOINT=/home/rsalman/.cache/huggingface/hub/models--Lightricks--LTX-2.3/snapshots/5948be4ced3a4493d1f836df64378ff136ddb770/ltx-2.3-22b-distilled-1.1.safetensors
export TT_DIT_CACHE_DIR=/home/rsalman/tt-metal/tt_dit_cache
export NO_PROMPT=1 RUN_WARMUP=0 RUN_VBENCH=0 RUN_CLIP=0 HEIGHT=1088 WIDTH=1920 SEED=10 NUM_FRAMES=153 FPS=25
T="models/tt_dit/tests/models/ltx/test_pipeline_ltx_distilled.py"; SEL="test_pipeline_distilled and 4x8sp1tp0nl2_ring_is_fsdp0 and not i2v"
run() { # tag traced fp32
  export LTX_TRACED=$2; if [ "$3" = 1 ]; then export LTX_EULER_FP32=1; else unset LTX_EULER_FP32; fi
  echo "=== [$(date +%T)] $1 (traced=$2 fp32=$3) ==="
  LTX_DUMP_AUDIO_LATENT=/home/rsalman/tt-metal/ltx_exp3/audio_latent_$1_153f25.pt python -m pytest "$T" -k "$SEL" -s --timeout 3600 > ltx_exp3/run_$1_153f25.log 2>&1; echo "$1 exit=$?"
  [ -f ltx_av_fast_1920x1088_0.mp4 ] && mv ltx_av_fast_1920x1088_0.mp4 ltx_exp3/euler_$1_153f25_seed10.mp4
  grep -h "latent\[s2/audio\]" ltx_exp3/run_$1_153f25.log | cut -c90-200
}
run bf16traced 1 0
run fp32today 0 1
echo "=== [$(date +%T)] done ==="
