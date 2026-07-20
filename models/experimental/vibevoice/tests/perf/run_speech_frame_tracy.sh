#!/usr/bin/env bash
# Profile one warm eager speech frame (neg LM + diffusion + post + pos LM).
#
# Usage (from tt-metal root):
#   bash models/experimental/vibevoice/tests/perf/run_speech_frame_tracy.sh
#
# Env:
#   VV_PROFILE_SPEECH_FRAME=2       # 1-based diffusion frame (2 = warm; neg LM runs)
#   VV_PROFILE_SPEECH_FRAME_EXIT=1  # stop generate after that frame
#   VV_TRACE_SEGMENT=0              # required — eager op stream (not fused trace)
#   VV_SF_MAX_NEW_TOKENS=32         # enough AR steps to reach frame 2
#
# Output:
#   models/experimental/vibevoice/lm/speech_frame_exp0.txt
set -euo pipefail
cd "$(dirname "$0")/../../../../.."
source python_env/bin/activate
export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd)
export ARCH_NAME="${ARCH_NAME:-blackhole}"
export VV_TRACE_SEGMENT=0
export VV_PROFILE_SPEECH_FRAME="${VV_PROFILE_SPEECH_FRAME:-2}"
export VV_PROFILE_SPEECH_FRAME_EXIT=1
export VV_DEBUG="${VV_DEBUG:-1}"
MAX_TOK="${VV_SF_MAX_NEW_TOKENS:-32}"
DEMO="${VV_SF_DEMO:-1p_CH2EN}"

echo "Profiling speech frame ${VV_PROFILE_SPEECH_FRAME} (eager), demo=${DEMO}, max_new_tokens=${MAX_TOK}"
python -m tracy -v -r -p --op-support-count 100000 \
  models/experimental/vibevoice/demo_ttnn.py \
  --demo "$DEMO" --max_new_tokens "$MAX_TOK"

CSV=$(ls -td generated/profiler/reports/*/ops_perf_results_*.csv | head -1)
OUT=models/experimental/vibevoice/lm/speech_frame_exp0.txt
tt-perf-report "$CSV" --start-signpost start --end-signpost stop > "$OUT"
echo "Wrote $OUT from $CSV"
head -60 "$OUT"
