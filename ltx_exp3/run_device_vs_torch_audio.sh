#!/bin/bash
# After the gen-0 run: device audio decode vs torch on today's binary (eager e2e PSNR test at 145f, then the
# girl-fixture decode test). If the Oct 1 rebuild broke the audio decode these fail regardless of cache/trace.
cd /home/rsalman/tt-metal
while pgrep -f run_traced_fp32_gen0.sh >/dev/null; do sleep 10; done
source python_env/bin/activate
export TT_METAL_HOME=/home/rsalman/tt-metal PYTHONPATH=/home/rsalman/tt-metal
export LTX_CHECKPOINT=/home/rsalman/.cache/huggingface/hub/models--Lightricks--LTX-2.3/snapshots/5948be4ced3a4493d1f836df64378ff136ddb770/ltx-2.3-22b-distilled-1.1.safetensors
export TT_DIT_CACHE_DIR=/home/rsalman/tt-metal/tt_dit_cache
echo "=== [$(date +%T)] test_audio_decode_e2e_psnr (eager, device vs torch) ==="
python -m pytest models/tt_dit/tests/models/ltx/test_audio_ltx.py -k "test_audio_decode_e2e_psnr and 4x8sp1tp0nl2_ring_is_fsdp0" -s --timeout 3600 > ltx_exp3/e2e_psnr_today.log 2>&1; echo "e2e exit=$?"
grep -E "PSNR|passed|failed|AssertionError" ltx_exp3/e2e_psnr_today.log | grep -v DEBUG | tail -4
echo "=== [$(date +%T)] test_audio_decode_girl ==="
python -m pytest models/tt_dit/tests/models/ltx/test_pipeline_ltx_distilled.py -k "test_audio_decode_girl and 4x8sp1tp0nl2_ring_is_fsdp0" -s --timeout 3600 > ltx_exp3/girl_today.log 2>&1; echo "girl exit=$?"
grep -E "AUDIO_GIRL|passed|failed|AssertionError" ltx_exp3/girl_today.log | grep -v DEBUG | tail -4
echo "=== [$(date +%T)] done ==="
