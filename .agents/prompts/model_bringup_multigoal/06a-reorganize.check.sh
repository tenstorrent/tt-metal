#!/usr/bin/env bash
# Runner-side gate for the reorganize stage: ensure reorganize evidence exists
# and free-running generation remains non-degenerate for the target model.
# Exit 0 pass, 1 advisory, 2 critical, 3 error.
if [ -n "${MODEL_DIR:-}" ]; then
  scope_args=(--model-dir "$MODEL_DIR")
  readme_path="$MODEL_DIR/doc/reorganize/README.md"
elif [ -n "${HF_MODEL:-}" ]; then
  scope_args=(--hf-model "$HF_MODEL")
  readme_path=""
else
  echo "Neither MODEL_DIR nor HF_MODEL is set; cannot scope the check to the target model." >&2
  exit 3
fi

if [ -n "$readme_path" ] && [ ! -f "$readme_path" ]; then
  echo "Missing expected reorganize evidence file: $readme_path" >&2
  exit 1
fi

python models/common/readiness_check/check_degenerate_output.py \
  "${scope_args[@]}" --missing-artifacts critical --scope autoregressive
