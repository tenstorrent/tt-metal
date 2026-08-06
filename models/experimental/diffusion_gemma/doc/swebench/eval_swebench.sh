#!/usr/bin/env bash
# Grade a mini-swe-agent SWE-Bench Verified run with the official harness (swebench 4.1.0).
#
# usage: eval_swebench.sh <run-dir> <run-id> [max-workers]
# env:   DG_SWEBENCH_VENV (default /home/ttuser/zni/venvs/swebench)
#
# Writes <model>.<run-id>.json next to the working directory. Read resolved_instances
# together with empty_patch_instances: an empty patch means the agent died before
# submitting, which is NOT the same failure as a submitted-but-wrong patch.
set -uo pipefail
RUN="${1:?run dir}"
RUNID="${2:?run id}"
MW="${3:-12}"
export HF_HOME="${HF_HOME:-/home/ttuser/zni/benchmarks/hfcache}"
source "${DG_SWEBENCH_VENV:-/home/ttuser/zni/venvs/swebench}/bin/activate"

PREDS="$RUN/preds.json"
[ -f "$PREDS" ] || { echo "no preds.json in $RUN"; exit 1; }

echo "=== $(date -Is) grading $PREDS ==="
python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Verified \
  --split test \
  --predictions_path "$PREDS" \
  --max_workers "$MW" \
  --run_id "$RUNID" \
  --cache_level env
echo "=== $(date -Is) grading done exit=$? ==="
