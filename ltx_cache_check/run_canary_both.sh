#!/bin/bash
# Stale-cache canary against both caches: regenerate audio prepared weights with the CURRENT binary and compare.
cd /home/rsalman/tt-metal
source python_env/bin/activate
export TT_METAL_HOME=/home/rsalman/tt-metal PYTHONPATH=/home/rsalman/tt-metal
export LTX_CHECKPOINT=/home/rsalman/.cache/huggingface/hub/models--Lightricks--LTX-2.3/snapshots/5948be4ced3a4493d1f836df64378ff136ddb770/ltx-2.3-22b-distilled-1.1.safetensors
T=models/tt_dit/tests/models/ltx/test_pipeline_ltx_distilled_quality_4x8.py
for c in /home/rsalman/tt-metal/tt_dit_cache /home/rsalman/.cache/tt-dit; do
  tag=$(basename $c); echo "=== [$(date +%T)] canary vs $c ==="
  TT_DIT_CACHE_DIR=$c python -m pytest $T -k test_ltx_audio_weight_cache_matches_regeneration -s --timeout 1800 > ltx_cache_check/canary_$tag.log 2>&1; echo "$tag exit=$?"
  grep -hE "audio_(dec|voc).*(match|differ|files)|passed|failed|AssertionError" ltx_cache_check/canary_$tag.log | grep -v DEBUG | tail -4
done
echo "=== [$(date +%T)] done ==="
