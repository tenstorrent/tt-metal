#!/bin/bash
# TDD Pipeline Pre-Commit Hook — Blocks commits if TDD gate not passed
#
# When a TDD pipeline is active for an operation, this hook ensures that
# commits affecting that operation's files only go through if the current
# stage's test has passed (indicated by a .tdd_gate_passed marker file).
#
# Exit codes:
#   0 - All gates passed (or no TDD pipelines active for staged files)
#   1 - Gate check failed — commit blocked
#
# Installed by: install_hooks.sh
# Marker: # TDD_PIPELINE_GATE

STAGED_OPS=$(git diff --cached --name-only | grep -oP 'ttnn/ttnn/operations/[^/]+' | sort -u)

blocked=0
for op_dir in $STAGED_OPS; do
    STATE="$op_dir/.tdd_state.json"
    [[ -f "$STATE" ]] || continue

    GATE="$op_dir/.tdd_gate_passed"
    if [[ ! -f "$GATE" ]]; then
        stage=$(python3 -c "
import json, sys
try:
    s = json.load(open('$STATE'))
    idx = s['current_stage_index']
    if idx < len(s['stages']):
        print(s['stages'][idx]['name'])
    else:
        print('COMPLETE')
except Exception as e:
    print(f'ERROR: {e}', file=sys.stderr)
    print('UNKNOWN')
" 2>/dev/null)

        echo "BLOCKED: TDD gate not passed for $op_dir (stage: $stage)"
        echo "Run: python3 .claude/scripts/tdd-pipeline/tdd_orchestrator.py test --op-path $op_dir"
        blocked=1
    fi
done

if [[ $blocked -eq 1 ]]; then
    echo ""
    echo "To bypass (emergency): git commit --no-verify"
    exit 1
fi

exit 0
