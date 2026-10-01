#!/bin/bash
cd /home/rsalman/tt-metal
source python_env/bin/activate
export TT_METAL_HOME=/home/rsalman/tt-metal PYTHONPATH=/home/rsalman/tt-metal
export LTX_CHECKPOINT=/home/rsalman/.cache/huggingface/hub/models--Lightricks--LTX-2.3/snapshots/5948be4ced3a4493d1f836df64378ff136ddb770/ltx-2.3-22b-distilled-1.1.safetensors
sum() { (cd "$1" 2>/dev/null && find . -type f | sort | xargs md5sum | md5sum | cut -c1-12); }
echo "=== [$(date +%T)] E1: canary @153f/25 vs server cache (fffc5dfcc5bb) ==="
TT_DIT_CACHE_DIR=/home/rsalman/tt-metal/tt_dit_cache python -m pytest models/tt_dit/tests/models/ltx/test_tmp_quality_153.py -k test_ltx_audio_weight_cache_matches_regeneration -s --timeout 1800 > ltx_exp3/e1_canary153.log 2>&1; echo "E1 exit=$?"
grep -hE "files differ|passed|failed" ltx_exp3/e1_canary153.log | grep -v DEBUG | tail -3 | cut -c90-260
echo "=== [$(date +%T)] E2: pytest gen @153f/25, fresh audio_dec in the clone cache ==="
export TT_DIT_CACHE_DIR=/home/rsalman/tt-metal/tt_dit_cache_pytest
export LTX_TRACED=0 LTX_EULER_FP32=1 NO_PROMPT=1 RUN_WARMUP=0 RUN_VBENCH=0 RUN_CLIP=0 HEIGHT=1088 WIDTH=1920 SEED=10 NUM_FRAMES=153 FPS=25
T="models/tt_dit/tests/models/ltx/test_pipeline_ltx_distilled.py"; SEL="test_pipeline_distilled and 4x8sp1tp0nl2_ring_is_fsdp0 and not i2v"
LTX_DUMP_AUDIO_LATENT=/home/rsalman/tt-metal/ltx_exp3/audio_latent_fp32clone_153f25.pt python -m pytest "$T" -k "$SEL" -s --timeout 3600 > ltx_exp3/run_fp32clone_153f25.log 2>&1; echo "E2 exit=$?"
[ -f ltx_av_fast_1920x1088_0.mp4 ] && mv ltx_av_fast_1920x1088_0.mp4 ltx_exp3/euler_fp32clone_153f25_seed10.mp4
echo "E2 regenerated audio_dec: $(sum tt_dit_cache_pytest/ltx-2.3-22b-distilled-1.1/audio_dec_cin55f0111e)   server: $(sum tt_dit_cache/ltx-2.3-22b-distilled-1.1/audio_dec_cin55f0111e)   (145f canary layout was d8de07dce4fd)"
python ltx_exp3/oracle_vs_mp4.py ltx_exp3/audio_latent_fp32clone_153f25.pt ltx_exp3/euler_fp32clone_153f25_seed10.mp4 2>&1 | grep PCC
echo "=== [$(date +%T)] done ==="
