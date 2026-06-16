#!/bin/bash
# Lean direct launch of ttnn-implementer on Refinement 7 — bypasses
# run_refinements.py's baseline + post-golden (the perf-tail wedge risk).
CLONE=/localdev/dnijemcevic/2026_06_12/1332_dnijemcevic_flash_causal/clones/flash_attention_run1/tt-metal
LOGDIR="$CLONE/.eval/r7_direct"
cd "$CLONE" || exit 1
unset PYTHON_ENV_DIR
export TT_METAL_HOME="$CLONE"
export PYTHONPATH="$CLONE"
export TT_METAL_CACHE="$CLONE/built"
export TT_METAL_ENV=dev
export CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1
source python_env/bin/activate
echo "START $(date)" > "$LOGDIR/status.txt"
claude -p --dangerously-skip-permissions --output-format json --max-turns 300 \
    --agent ttnn-implementer "$(cat "$LOGDIR/prompt.txt")" \
    > "$LOGDIR/implementer_output.json" 2> "$LOGDIR/implementer_stderr.log"
echo "R7_DIRECT_EXIT=$? $(date)" >> "$LOGDIR/status.txt"
