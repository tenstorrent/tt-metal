#!/bin/bash
# enable_breadcrumbs.sh — Register the SubagentStart logging hook in settings.local.json
#
# Usage: .claude/scripts/logging/enable_breadcrumbs.sh
#
# Adds the SubagentStart hook to .claude/settings.local.json (if not already present).
# Breadcrumbs are always enabled — this script only needs to run once to register the hook.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
SETTINGS="$REPO_ROOT/.claude/settings.local.json"

# Add hook to settings.local.json
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
