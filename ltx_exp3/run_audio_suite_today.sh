#!/bin/bash
# After the queued runs: the audio reference suite that passed on 2026-09-30 (Sep 28 binary), now on today's binary.
cd /home/rsalman/tt-metal
while pgrep -f "run_discriminate.sh|run_traced_fp32_gen0.sh|run_exp3_traced_fp32_nowarm.sh" >/dev/null; do sleep 10; done
source python_env/bin/activate
export TT_METAL_HOME=/home/rsalman/tt-metal PYTHONPATH=/home/rsalman/tt-metal
export LTX_CHECKPOINT=/home/rsalman/.cache/huggingface/hub/models--Lightricks--LTX-2.3/snapshots/5948be4ced3a4493d1f836df64378ff136ddb770/ltx-2.3-22b-distilled-1.1.safetensors
export TT_DIT_CACHE_DIR=/home/rsalman/tt-metal/tt_dit_cache
echo "=== [$(date +%T)] audio suite on today's binary ==="
python -m pytest models/tt_dit/tests/models/ltx/test_audio_ltx.py -k "(test_stage_a_audio_decoder or test_stage_b_vocoder or test_stage_c_vocoder_with_bwe or test_audio_decode_e2e_psnr) and 4x8sp1tp0nl2_ring_is_fsdp0" -s --timeout 3600 > ltx_exp3/audio_suite_today.log 2>&1; echo "suite exit=$?"
grep -E "PASSED|FAILED|passed|failed|PSNR|pcc=|PCC" ltx_exp3/audio_suite_today.log | grep -v DEBUG | tail -12
echo "=== [$(date +%T)] done ==="
