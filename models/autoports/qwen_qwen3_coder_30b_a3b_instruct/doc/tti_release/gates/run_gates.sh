#!/usr/bin/env bash
# Stage-10 gates, materialised from origin/agentic-research/hous/multigoal-claude
# into a scratch root so that no .agents/ directory is ever created in the
# tt-metal worktree. `models` in the scratch root is a symlink to the real tree,
# so the relative MODEL_DIR the checks expect resolves to the real model.
#
# Materialise the root first (nothing here creates it):
#   R=$GATES_ROOT/root
#   mkdir -p "$R/.agents/scripts" "$R/.agents/prompts/model_bringup_multigoal"
#   git -C /home/raahem/tt-metal show \
#     origin/agentic-research/hous/multigoal-claude:.agents/scripts/check_context_contract.py \
#     > "$R/.agents/scripts/check_context_contract.py"
#   git -C /home/raahem/tt-metal show \
#     origin/agentic-research/hous/multigoal-claude:.agents/prompts/model_bringup_multigoal/10-tti-release.check.sh \
#     > "$R/.agents/prompts/model_bringup_multigoal/10-tti-release.check.sh"
#   ln -s /home/raahem/tt-metal/models "$R/models"
S=${GATES_ROOT:-/tmp/claude-1001/-home-raahem-tt-metal/ad4804b7-3765-440c-8fb1-dec6c8fa23f6/scratchpad/gates2}
cd "$S/root" || exit 3
export MODEL_DIR=models/autoports/qwen_qwen3_coder_30b_a3b_instruct
export HF_MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct

echo "===================================================================="
echo "GATE 1: .agents/prompts/model_bringup_multigoal/10-tti-release.check.sh"
echo "  cwd=$PWD  MODEL_DIR=$MODEL_DIR  HF_MODEL=$HF_MODEL"
echo "===================================================================="
bash .agents/prompts/model_bringup_multigoal/10-tti-release.check.sh
g1=$?
echo "GATE 1 EXIT: $g1"

echo
echo "===================================================================="
echo "GATE 2: .agents/scripts/check_context_contract.py --stage tti-release --require-contract"
echo "===================================================================="
python3 .agents/scripts/check_context_contract.py \
  --model-dir "$MODEL_DIR" --hf-model "$HF_MODEL" \
  --stage tti-release --require-contract
g2=$?
echo "GATE 2 EXIT: $g2"

echo
echo "===================================================================="
echo "GATE 3 (stage-local): probes/check_published_figures.py"
echo "===================================================================="
( cd /home/raahem/tt-metal/models/autoports/qwen_qwen3_coder_30b_a3b_instruct/doc/tti_release \
  && python3 probes/check_published_figures.py )
g3=$?
echo "GATE 3 EXIT: $g3"

echo
echo "SUMMARY: stage_check=$g1 context_contract=$g2 published_figures=$g3"
