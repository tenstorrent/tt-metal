#!/bin/bash
# After the gen-0 run. Hypothesis: prepared mel-decoder weights depend on the consumer's audio-decoder config
# (server vs pytest harness) but share one cache key. Test: pytest run against ~/.cache/tt-dit (audio_dec absent ->
# regenerated in the pytest layout) must give good audio, while the same run against the server's copy gave crackle.
cd /home/rsalman/tt-metal
while pgrep -f run_traced_fp32_gen0.sh >/dev/null; do sleep 10; done
source python_env/bin/activate
export TT_METAL_HOME=/home/rsalman/tt-metal PYTHONPATH=/home/rsalman/tt-metal
export LTX_CHECKPOINT=/home/rsalman/.cache/huggingface/hub/models--Lightricks--LTX-2.3/snapshots/5948be4ced3a4493d1f836df64378ff136ddb770/ltx-2.3-22b-distilled-1.1.safetensors
export TT_DIT_CACHE_DIR=/home/rsalman/.cache/tt-dit
export LTX_TRACED=0 LTX_EULER_FP32=1 NO_PROMPT=1 RUN_WARMUP=0 RUN_VBENCH=0 RUN_CLIP=0 HEIGHT=1088 WIDTH=1920 SEED=10 NUM_FRAMES=153 FPS=25
T="models/tt_dit/tests/models/ltx/test_pipeline_ltx_distilled.py"; SEL="test_pipeline_distilled and 4x8sp1tp0nl2_ring_is_fsdp0 and not i2v"
echo "=== [$(date +%T)] untraced fp32, cache=~/.cache/tt-dit (regenerates audio_dec) ==="
LTX_DUMP_AUDIO_LATENT=/home/rsalman/tt-metal/ltx_exp3/audio_latent_fp32defcache_153f25.pt python -m pytest "$T" -k "$SEL" -s --timeout 3600 > ltx_exp3/run_fp32defcache_153f25.log 2>&1; echo "exit=$?"
[ -f ltx_av_fast_1920x1088_0.mp4 ] && mv ltx_av_fast_1920x1088_0.mp4 ltx_exp3/euler_fp32defcache_153f25_seed10.mp4
echo "regenerated audio_dec checksum: $(cd ~/.cache/tt-dit/ltx-2.3-22b-distilled-1.1/audio_dec_cin55f0111e 2>/dev/null && find . -type f | sort | xargs md5sum | md5sum | cut -c1-12)  (pytest layout seen before: d8de07dce4fd, server layout: fffc5dfcc5bb)"
python ltx_exp3/oracle_vs_mp4.py ltx_exp3/audio_latent_fp32defcache_153f25.pt ltx_exp3/euler_fp32defcache_153f25_seed10.mp4 2>&1 | grep PCC
echo "=== [$(date +%T)] e2e PSNR test with the same cache ==="
python -m pytest models/tt_dit/tests/models/ltx/test_audio_ltx.py -k "test_audio_decode_e2e_psnr and 4x8sp1tp0nl2_ring_is_fsdp0" -s --timeout 3600 > ltx_exp3/e2e_psnr_today.log 2>&1; echo "e2e exit=$?"
grep -E "PSNR|passed|failed" ltx_exp3/e2e_psnr_today.log | grep -v DEBUG | tail -3
echo "=== [$(date +%T)] done ==="
