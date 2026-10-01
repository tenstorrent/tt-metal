#!/bin/bash
# Corrected E1 (the first attempt's sed missed the one-line constants and ran at 145f): canary at 153f/25 vs server cache.
cd /home/rsalman/tt-metal
while pgrep -f run_shape_hypothesis.sh >/dev/null; do sleep 10; done
source python_env/bin/activate
export TT_METAL_HOME=/home/rsalman/tt-metal PYTHONPATH=/home/rsalman/tt-metal
export LTX_CHECKPOINT=/home/rsalman/.cache/huggingface/hub/models--Lightricks--LTX-2.3/snapshots/5948be4ced3a4493d1f836df64378ff136ddb770/ltx-2.3-22b-distilled-1.1.safetensors
echo "=== [$(date +%T)] E1-fixed: canary @153f/25 vs server cache (fffc5dfcc5bb) ==="
TT_DIT_CACHE_DIR=/home/rsalman/tt-metal/tt_dit_cache python -m pytest models/tt_dit/tests/models/ltx/test_tmp_quality_153.py -k test_ltx_audio_weight_cache_matches_regeneration -s --timeout 1800 > ltx_exp3/e1_canary153_fixed.log 2>&1; echo "E1-fixed exit=$?"
grep -hE "files differ|passed|failed" ltx_exp3/e1_canary153_fixed.log | grep -v DEBUG | tail -3 | cut -c90-260
grep -h "num_frames\|audio_N\|aN=" ltx_exp3/e1_canary153_fixed.log | grep -v DEBUG | head -2 | cut -c90-200
echo "=== [$(date +%T)] done ==="
