#!/usr/bin/env bash
# Re-run ONLY gpqa_diamond_cot_zeroshot, against the same live server, with the
# same settings the other evals used. Nothing else is re-run.
set -uo pipefail
TTI=/home/raahem/tt-inference-server
OUT=$TTI/workflow_logs/reports_output/release/Qwen3-Coder-30B-A3B-Instruct_p300x2_release/eval_id_qwen3-coder-30b-a3b-autoport_Qwen3-Coder-30B-A3B-Instruct_p300x2

# The token lives in TTI's .env; run.py loads it via handle_secrets(), but we are
# invoking lm_eval directly so we export it ourselves.
export $(grep '^HF_TOKEN=' $TTI/.env)

# Fail loudly NOW if the token is not actually usable for this dataset, rather
# than letting lm-eval reproduce the original gated error 3 seconds in.
$TTI/.workflow_venvs/.venv_evals_common/bin/python - <<'PY' || exit 9
import os, sys
from huggingface_hub import whoami, HfApi
tok = os.environ.get("HF_TOKEN")
assert tok, "HF_TOKEN not exported"
print("whoami:", whoami(token=tok)["name"])
info = HfApi().dataset_info("Idavidrein/gpqa", token=tok)
print("gpqa accessible, gated =", getattr(info, "gated", None))
PY
echo "=== preflight OK, launching lm_eval at $(date) ==="

$TTI/.workflow_venvs/.venv_evals_common/bin/lm_eval \
  --tasks gpqa_diamond_cot_zeroshot \
  --model local-completions \
  --model_args model=Qwen/Qwen3-Coder-30B-A3B-Instruct,base_url=http://127.0.0.1:8100/v1/completions,tokenizer_backend=huggingface,max_length=262144,timeout=7200,num_concurrent=32 \
  --gen_kwargs max_gen_toks=4096,do_sample=false,stream=false,seed=42 \
  --output_path "$OUT" \
  --seed 42 \
  --num_fewshot 0 \
  --batch_size 1 \
  --log_samples \
  --show_config \
  --apply_chat_template \
  --trust_remote_code \
  --confirm_run_unsafe_code
echo "GPQA_EXIT_CODE=$?"
echo "=== finished at $(date) ==="
