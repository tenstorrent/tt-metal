#!/usr/bin/env bash
# Profile a warm eager speech frame (neg-LM + diffusion + post + pos-LM) on Blackhole.
# Tracy start/stop signposts wrap frame N (VV_PROFILE_SPEECH_FRAME). Eager only
# (VV_TRACE_SEGMENT=0) so the op stream is per-op, not a fused trace replay.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
source python_env/bin/activate
export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) ARCH_NAME=blackhole
export VV_TRACE_SEGMENT=${VV_TRACE_SEGMENT:-0}
export VV_PROFILE_SPEECH_FRAME=${VV_PROFILE_SPEECH_FRAME:-2}
export VV_PROFILE_SPEECH_FRAME_EXIT=${VV_PROFILE_SPEECH_FRAME_EXIT:-1}

DEMO=${VV_DEMO:-1p_CH2EN}
MAX_NEW_TOKENS=${VV_MAX_NEW_TOKENS:-32}
OUT=${VV_OUT:-models/experimental/vibevoice/lm/speech_frame_baseline.txt}

python -m tracy -v -r -p --op-support-count 100000 \
  models/experimental/vibevoice/demo_ttnn.py --demo "$DEMO" --max_new_tokens "$MAX_NEW_TOKENS"

CSV=$(ls -td generated/profiler/reports/*/ops_perf_results_*.csv | head -1)
echo "CSV: $CSV"
tt-perf-report "$CSV" --start-signpost start --end-signpost stop | tee "$OUT"
echo "wrote $OUT"
