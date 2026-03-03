#!/bin/bash
# enable_breadcrumbs.sh — Register the SubagentStart logging hook in settings.local.json
#
# Usage: .claude/scripts/logging/enable_breadcrumbs.sh
#
# Does two things:
# 1. Creates the .claude/active_logging signal file
# 2. Adds the SubagentStart hook to .claude/settings.local.json (if not already present)
#
# To disable: rm -f .claude/active_logging
# (The hook stays registered but does nothing without the signal file)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
SETTINGS="$REPO_ROOT/.claude/settings.local.json"
SIGNAL_FILE="$REPO_ROOT/.claude/active_logging"

# 1. Create signal file
touch "$SIGNAL_FILE"
echo "Created $SIGNAL_FILE"

# 2. Add hook to settings.local.json
if [[ ! -f "$SETTINGS" ]]; then
    cat > "$SETTINGS" <<'EOF'
{
  "hooks": {
    "SubagentStart": [
      {
        "hooks": [
          {
            "type": "command",
            "command": ".claude/scripts/logging/inject-logging-context.sh"
          }
        ]
      }
    ]
  }
}
EOF
    echo "Created $SETTINGS with SubagentStart hook"
else
    # Check if hook already registered
    if jq -e '.hooks.SubagentStart' "$SETTINGS" > /dev/null 2>&1; then
        echo "SubagentStart hook already registered in $SETTINGS"
    else
        # Merge the hook into existing settings
        jq '.hooks = (.hooks // {}) + {
          "SubagentStart": [
            {
              "hooks": [
                {
                  "type": "command",
                  "command": ".claude/scripts/logging/inject-logging-context.sh"
                }
              ]
            }
          ]
        }' "$SETTINGS" > "$SETTINGS.tmp" && mv "$SETTINGS.tmp" "$SETTINGS"
        echo "Added SubagentStart hook to $SETTINGS"
    fi
fi

echo "Breadcrumbs enabled. All subagents will now receive logging instructions."
