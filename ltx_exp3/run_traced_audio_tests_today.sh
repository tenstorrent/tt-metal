#!/bin/bash
# After the audio suite: the tests that exercise the TRACED audio decode on today's binary.
cd /home/rsalman/tt-metal
while pgrep -f "run_discriminate.sh|run_traced_fp32_gen0.sh|run_audio_suite_today.sh" >/dev/null; do sleep 10; done
source python_env/bin/activate
export TT_METAL_HOME=/home/rsalman/tt-metal PYTHONPATH=/home/rsalman/tt-metal
export LTX_CHECKPOINT=/home/rsalman/.cache/huggingface/hub/models--Lightricks--LTX-2.3/snapshots/5948be4ced3a4493d1f836df64378ff136ddb770/ltx-2.3-22b-distilled-1.1.safetensors
export TT_DIT_CACHE_DIR=/home/rsalman/tt-metal/tt_dit_cache
echo "=== [$(date +%T)] 6s quality test (traced ring, audio vs torch oracle) ==="
python -m pytest models/tt_dit/tests/models/ltx/test_pipeline_ltx_distilled_quality_4x8.py -k test_ltx_6s_audio_matches_torch_reference -s --timeout 3600 > ltx_exp3/quality_6s_today.log 2>&1; echo "quality exit=$?"
grep -E "PSNR|PCC|passed|failed|AssertionError" ltx_exp3/quality_6s_today.log | grep -v DEBUG | tail -6
echo "=== [$(date +%T)] test_audio_decode_girl, traced variants ==="
LTX_TRACED=1 python -m pytest models/tt_dit/tests/models/ltx/test_audio_ltx.py -k "test_audio_decode_girl and 4x8sp1tp0nl2_ring_is_fsdp0" -s --timeout 3600 > ltx_exp3/girl_traced_today.log 2>&1; echo "girl exit=$?"
grep -E "AUDIO_GIRL|passed|failed|AssertionError" ltx_exp3/girl_traced_today.log | grep -v DEBUG | tail -6
echo "=== [$(date +%T)] done ==="
