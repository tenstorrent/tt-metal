#!/usr/bin/env bash
# Runner-side gate for the vLLM integration stage: served qualitative outputs
# (greedy and sampled) must exist and must not be mechanically degenerate. A
# serving path that doubles tokens or collapses to one token is a serving bug,
# not a model-quality limitation. Scoped to this run's model.
# Exit 0 pass, 1 advisory, 2 critical, 3 error.
if [ -n "${MODEL_DIR:-}" ]; then
  scope_args=(--model-dir "$MODEL_DIR")
elif [ -n "${HF_MODEL:-}" ]; then
  scope_args=(--hf-model "$HF_MODEL")
else
  echo "Neither MODEL_DIR nor HF_MODEL is set; cannot scope the check to the target model." >&2
  exit 3
fi
python models/common/readiness_check/check_degenerate_output.py \
  "${scope_args[@]}" --missing-artifacts critical --scope all || exit $?

python .agents/scripts/check_context_contract.py \
  --model-dir "${MODEL_DIR:-}" --hf-model "${HF_MODEL:-}" \
  --stage vllm --require-contract
