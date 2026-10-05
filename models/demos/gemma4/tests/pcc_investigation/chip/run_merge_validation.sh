#!/bin/bash
# Re-measure accuracy, PCC and perf of the checked-out Gemma 4 code (QB2 1x4), e.g. after merging main.
#   1. accuracy gate      test_optimizer_gemma4_pcc (prefill the prompt, decode 129 positions)
#   2. token accuracy     tt_token_accuracy.py (prefill 512 book tokens, 499 teacher-forced decode steps)
#   3. per-layer PCC      tt_per_layer_pcc.py (prefill accumulated / isolated, decode 480-511 after a 480 prefill)
#   4. perf               test_optimizer_gemma4_perf (TT_PERF_OSL_TOKENS=256)
# Usage: run_merge_validation.sh <label>    logs: $GEMMA4_PCC_DATA/logs/validate-<label>-*.log
set -u
LABEL=$1
REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
HERE=$(cd "$(dirname "$0")" && pwd)
DATA=${GEMMA4_PCC_DATA:-$HOME/benchmark-data/gemma4-pcc}
LOGS=$DATA/logs; mkdir -p "$LOGS"
cd "$REPO"
export HF_HUB_OFFLINE=1 HF_MODEL=$HOME/benchmark-data/gemma-4-26B-A4B-it HF_MODEL_ID=google/gemma-4-26B-A4B-it MESH_DEVICE=P150x4
echo "tree $(git log -1 --format='%h %s') label $LABEL start $(date -u +%T)"

timeout 3000 python_env/bin/python -m pytest models/demos/gemma4/tests/test_optimizer_gemma4_pcc.py::test_optimizer_gemma4_pcc \
  -x -q --no-header -s > "$LOGS/validate-$LABEL-gate.log" 2>&1
echo "gate rc=$? $(grep -h '^ACCURACY ' "$LOGS/validate-$LABEL-gate.log" | cut -c1-200)"

timeout 3000 python_env/bin/python "$HERE/tt_token_accuracy.py" "$LABEL" > "$LOGS/validate-$LABEL-tokacc.log" 2>&1
echo "tokacc rc=$? $(grep -h '^TOKACC ' "$LOGS/validate-$LABEL-tokacc.log")"

rm -rf "$DATA/tt_cache_perlayer_$LABEL"
timeout 3000 python_env/bin/python "$HERE/tt_per_layer_pcc.py" "$LABEL" > "$LOGS/validate-$LABEL-perlayer.log" 2>&1
echo "perlayer rc=$? $(grep -c '^PREFILL_\|^DECODE_' "$LOGS/validate-$LABEL-perlayer.log") rows"
rm -rf "$DATA/tt_cache_perlayer_$LABEL"

TT_PERF_OSL_TOKENS=256 timeout 3600 python_env/bin/python -m pytest models/demos/gemma4/tests/test_optimizer_gemma4_perf.py::test_optimizer_gemma4_perf \
  -x -q -s > "$LOGS/validate-$LABEL-perf.log" 2>&1
echo "perf rc=$? $(grep -hE '^PERF |^TRACE_STAGE_MS\[decode\]' "$LOGS/validate-$LABEL-perf.log" | tr '\n' ' ')"
echo "done $(date -u +%T)"
