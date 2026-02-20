#!/bin/bash
# Install TDD Pipeline Pre-Commit Hook
#
# Appends the TDD gate check to .git/hooks/pre-commit.
# Idempotent — checks for marker comment before adding.
#
# Usage: bash .claude/scripts/tdd-pipeline/install_hooks.sh

set -e

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)"
if [[ -z "$REPO_ROOT" ]]; then
    echo "ERROR: Not in a git repository." >&2
    exit 1
fi

HOOK_FILE="$REPO_ROOT/.git/hooks/pre-commit"
MARKER="# TDD_PIPELINE_GATE"
GATE_SCRIPT="$REPO_ROOT/.claude/scripts/tdd-pipeline/pre_commit_test_gate.sh"

# Check if already installed
if [[ -f "$HOOK_FILE" ]] && grep -q "$MARKER" "$HOOK_FILE"; then
    echo "TDD pipeline hook already installed in $HOOK_FILE"
    exit 0
fi

# Create hook file if it doesn't exist
if [[ ! -f "$HOOK_FILE" ]]; then
    echo "#!/bin/bash" > "$HOOK_FILE"
    chmod +x "$HOOK_FILE"
    echo "Created $HOOK_FILE"
fi

# Ensure it's executable
chmod +x "$HOOK_FILE"

# Append the gate check
cat >> "$HOOK_FILE" << EOF

$MARKER
# TDD Pipeline: Block commits if stage gate not passed
bash "$GATE_SCRIPT"
tdd_gate_result=\$?
if [[ \$tdd_gate_result -ne 0 ]]; then
    exit \$tdd_gate_result
fi
EOF

echo "TDD pipeline hook installed in $HOOK_FILE"
