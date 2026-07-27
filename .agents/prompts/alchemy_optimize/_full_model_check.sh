#!/usr/bin/env bash
# Full-model correctness+perf gate for the optimize pipeline.
# Runs the prettified model's full-model test (main.py) which prints "PCC: <x>"
# (correlation vs CPU golden) and traced TPS. This IS the "check the full model
# at every stage" invariant. Exit 0 pass, 2 critical (PCC below bar / run failed),
# 3 infra error. Baseline (pre-optimize): PCC=0.999743, trace TPS ~730.
set -o pipefail
GD="${MODEL_DIR:-/home/mvasiljevic/project-alchemy/ttnn-models/openai/gpt-oss-20b/model/graph_0}"
THRESHOLD="${PCC_THRESHOLD:-0.98}"
[ -f "$GD/main.py" ] || { echo "no main.py at $GD"; exit 3; }
find "$GD" -name "*.pyc" -delete 2>/dev/null
find "$GD" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
# shellcheck disable=SC1091
source /home/mvasiljevic/_env.sh >/dev/null 2>&1 || { echo "cannot source _env.sh"; exit 3; }
cd "$GD" || exit 3
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$GD"
OUT="$(timeout 900 python3 -m pytest main.py -s 2>&1)"
PCC="$(echo "$OUT" | grep -oE '^PCC: [0-9.]+' | tail -1 | awk '{print $2}')"
TPS="$(echo "$OUT" | grep -oE 'trace\): [0-9.]+s, TPS: [0-9.]+' | tail -1)"
echo "===== FULL-MODEL CHECK ====="
echo "FULL_MODEL_PCC=${PCC:-<none>}   ${TPS:-<no-trace-tps>}   (bar: PCC>=$THRESHOLD)"
if [ -z "$PCC" ]; then
  echo "FAIL: no PCC produced (model run failed). Tail:"; echo "$OUT" | grep -viE 'nanobind|leaked' | tail -25
  exit 2
fi
if awk -v p="$PCC" -v t="$THRESHOLD" 'BEGIN{exit !(p+0 >= t+0)}'; then
  echo "PASS: full-model PCC=$PCC >= $THRESHOLD"; exit 0
else
  echo "FAIL(critical): full-model PCC=$PCC < $THRESHOLD — optimization regressed correctness"; exit 2
fi
