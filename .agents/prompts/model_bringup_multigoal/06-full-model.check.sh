#!/usr/bin/env bash
# Runner-side gate for the full-model stage: free-running TT generation must
# exist and must not be mechanically degenerate (doubled tokens, single-token
# collapse). Scoped to this run's model so stale artifacts from another model
# can neither pass nor fail it. Exit 0 pass, 1 advisory, 2 critical, 3 error.
if [ -n "${MODEL_DIR:-}" ]; then
  scope_args=(--model-dir "$MODEL_DIR")
elif [ -n "${HF_MODEL:-}" ]; then
  scope_args=(--hf-model "$HF_MODEL")
else
  echo "Neither MODEL_DIR nor HF_MODEL is set; cannot scope the check to the target model." >&2
  exit 3
fi
python models/common/readiness_check/check_degenerate_output.py \
  "${scope_args[@]}" --missing-artifacts critical --scope autoregressive || exit $?

python .agents/scripts/check_context_contract.py \
  --model-dir "${MODEL_DIR:-}" --hf-model "${HF_MODEL:-}" \
  --stage full-model --require-contract
