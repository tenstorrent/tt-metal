#!/usr/bin/env bash
# Runner-side gate for stage 07 (optimized-full-model): optimization must not
# have removed or degraded free-running TT generation evidence. Scoped to this
# run's model. Exit 0 pass, 1 advisory, 2 critical, 3 error.
if [ -n "${MODEL_DIR:-}" ]; then
  scope_args=(--model-dir "$MODEL_DIR")
elif [ -n "${HF_MODEL:-}" ]; then
  scope_args=(--hf-model "$HF_MODEL")
else
  echo "Neither MODEL_DIR nor HF_MODEL is set; cannot scope the check to the target model." >&2
  exit 3
fi
python models/common/readiness_check/check_degenerate_output.py \
  "${scope_args[@]}" --missing-artifacts critical --scope autoregressive
