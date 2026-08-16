#!/usr/bin/env bash
# The $qualitative-check shared suite against this stage's release server.
#
#   qualitative.sh
#
# Two arms, as every stage from full-model on has run them:
#
#   chat    -- doc/vllm_integration/bench/qualitative_vllm.py.  The verdict arm.
#              It posts the **pinned token ids** the full-model stage rendered
#              with the checkpoint's own chat template, so the release server,
#              the previous serving stage, the standalone TT model and the HF
#              control all ran the identical input.  Because it posts ids to
#              /v1/completions it is also unaffected by this stage's reasoning
#              parser, which only acts on /v1/chat/completions -- so the
#              comparison against the earlier stages stays like-for-like.
#   runner  -- the readiness runner's own raw-completion arm, kept as labelled
#              continuation stress coverage, not as a quality verdict.
#
# Then the shared degenerate-output gate over both.
set -uo pipefail

REPO=/home/ttuser/dev/muse-glimmer/tt-metal
PYENV=/home/ttuser/dev/muse-glimmer/muse-glimmer_pyenv
MODEL_DIR=models/autoports/meta_models_muse_glimmer_30b
DOC=$REPO/$MODEL_DIR/doc/tti_release
VDOC=$REPO/$MODEL_DIR/doc/vllm_integration
PORT=${PORT:-8000}
URL=http://localhost:$PORT

unset VIRTUAL_ENV VIRTUAL_ENV_PROMPT PYTHONPATH PYTHONHOME
# shellcheck disable=SC1091
source "$PYENV/bin/activate"
export TT_METAL_HOME="$REPO"
export PYTHONPATH="$REPO"
cd "$REPO"

mkdir -p "$DOC/qualitative" "$DOC/logs"

echo "=== chat arm (verdict) ==="
python "$VDOC/bench/qualitative_vllm.py" --server-url "$URL" --out-dir "$DOC/qualitative" \
  2>&1 | tee "$DOC/logs/qualitative_chat.log"
echo "  exit=${PIPESTATUS[0]}"

echo "=== chat arm comparison vs HF control and the previous stage ==="
python "$VDOC/bench/qualitative_vllm.py" --compare --out-dir "$DOC/qualitative" \
  2>&1 | tee "$DOC/logs/qualitative_compare.log"
echo "  exit=${PIPESTATUS[0]}"

echo "=== runner raw-completion arm (labelled continuation coverage) ==="
python -m models.common.readiness_check.run_vllm_server \
  --stages qualitative --server-url "$URL" \
  --model-dir "$MODEL_DIR" --hf-model meta-models/Muse-Glimmer-30B \
  --output-dir "$DOC/qualitative_runner" \
  2>&1 | tee "$DOC/logs/qualitative_runner.log"
echo "  exit=${PIPESTATUS[0]}"

echo "=== shared degenerate-output gate over both arms ==="
python models/common/readiness_check/check_degenerate_output.py \
  "$DOC/qualitative" "$DOC/qualitative_runner" \
  --scope all --missing-artifacts critical \
  --json "$DOC/qualitative/degenerate_check.json" \
  2>&1 | tee "$DOC/logs/degenerate_check.log"
echo "  exit=${PIPESTATUS[0]}"
