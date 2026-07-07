#!/bin/bash
# ACE-Step v1.5 — LM planner (acestep-5Hz-lm-1.7B) PCC + batch>1 benchmark.
# Primary metric = lm_pcc (MIN TT-vs-HF last_hidden_state PCC across seq lengths, HIGHER is better).
# The 28-layer Qwen3 LM has massive activations (absmax ~205) that bf16 mis-represents -> baseline
# ~0.58. Goal: raise toward the 0.97 gate WITHOUT overfitting (metric is the MIN across seqs so a fix
# must generalize) + add batch>1 support (batch_pcc: batch-2 forward vs two batch-1 forwards).
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1

if [ -f python_env/bin/activate ]; then
  # shellcheck disable=SC1091
  source python_env/bin/activate
fi
export TT_METAL_HOME="$(pwd)"
if [ -z "${ACESTEP_PIPELINE_DIR:-}" ] && [ -f /local/ttuser/gtobar/acestep_pipeline/vae/diffusion_pytorch_model.safetensors ]; then
  export ACESTEP_PIPELINE_DIR=/local/ttuser/gtobar/acestep_pipeline
fi

ACE=models/experimental/acestep

# --- fast syntax pre-check (<1s) ---
python -c "import ast,glob,sys
bad=0
for f in glob.glob('$ACE/**/*.py', recursive=True):
    try: ast.parse(open(f).read())
    except SyntaxError as e:
        print(f'SYNTAX ERROR {f}: {e}'); bad=1
sys.exit(bad)" || { echo 'METRIC lm_pcc=0'; echo 'PRECHECK_FAILED syntax'; exit 0; }

# --- primary: LM planner PCC + batch check ---
LOG=$(mktemp)
timeout 550 python "$(dirname "$0")/measure_lm.py" >"$LOG" 2>&1 || true
grep -E "^LM |^METRIC " "$LOG" || { echo "RUN FAILED — tail:"; tail -30 "$LOG"; echo 'METRIC lm_pcc=0'; }
rm -f "$LOG"
