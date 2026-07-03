#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROMPT_TEMPLATE="$SCRIPT_DIR/AUTODEBUG_PROMPT.md"
AGENT="${AUTODEBUG_AGENT:-codex}"
CODEX_MODEL="${AUTODEBUG_CODEX_MODEL:-gpt-5.5}"
CLAUDE_MODEL="${AUTODEBUG_CLAUDE_MODEL:-opus}"
EFFORT="${AUTODEBUG_EFFORT:-xhigh}"
RUN_DIR="$(pwd -P)"

usage() {
    cat <<'USAGE'
Usage:
  .agents/scripts/autodebug.sh [options] [focus-path] <problem...>

Run a fresh AutoDebug investigation in the current directory. The agent should
write ./AUTODEBUG.md. After the run finishes, read that report and act on it.

Options:
  --agent codex|claude     Agent CLI to run. Default: codex.
  --model MODEL            Override the model for the selected agent.
                           Defaults: codex=gpt-5.5, claude=opus.
  --effort LEVEL           Reasoning/thinking effort. Default: xhigh.
  --prompt-template PATH   Prompt template to render. Default:
                           .agents/scripts/AUTODEBUG_PROMPT.md.
  --help                   Show this help.

Examples:
  .agents/scripts/autodebug.sh models/demos/foo "decode diverges after token 128"
  .agents/scripts/autodebug.sh --agent claude "why does this test hang?"
USAGE
}

die() {
    echo "autodebug.sh: $*" >&2
    exit 2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h)
            usage
            exit 0
            ;;
        --agent)
            [[ $# -ge 2 ]] || die "--agent requires codex or claude"
            AGENT="$2"
            shift 2
            ;;
        --agent=*)
            AGENT="${1#*=}"
            shift
            ;;
        --model)
            [[ $# -ge 2 ]] || die "--model requires a value"
            CODEX_MODEL="$2"
            CLAUDE_MODEL="$2"
            shift 2
            ;;
        --model=*)
            CODEX_MODEL="${1#*=}"
            CLAUDE_MODEL="${1#*=}"
            shift
            ;;
        --effort|--thinking|--thinking-level)
            [[ $# -ge 2 ]] || die "$1 requires a value"
            EFFORT="$2"
            shift 2
            ;;
        --effort=*|--thinking=*|--thinking-level=*)
            EFFORT="${1#*=}"
            shift
            ;;
        --prompt-template)
            [[ $# -ge 2 ]] || die "--prompt-template requires a path"
            PROMPT_TEMPLATE="$2"
            shift 2
            ;;
        --prompt-template=*)
            PROMPT_TEMPLATE="${1#*=}"
            shift
            ;;
        --)
            shift
            break
            ;;
        -*)
            die "unknown option: $1"
            ;;
        *)
            break
            ;;
    esac
done

[[ $# -gt 0 ]] || die "provide a problem description"
[[ -f "$PROMPT_TEMPLATE" ]] || die "prompt template not found: $PROMPT_TEMPLATE"

FOCUS_PATH=""
if [[ $# -gt 1 && -e "$1" ]]; then
    FOCUS_PATH="$1"
    shift
fi
[[ $# -gt 0 ]] || die "when focus-path is provided, add a problem description"

PROBLEM="$*"
PROMPT_FILE="$(mktemp "${TMPDIR:-/tmp}/autodebug-prompt.XXXXXX")"
trap 'rm -f "$PROMPT_FILE"' EXIT

python3 - "$PROMPT_TEMPLATE" "$FOCUS_PATH" "$PROBLEM" >"$PROMPT_FILE" <<'PY'
from pathlib import Path
import sys

template_path = Path(sys.argv[1])
focus_path = sys.argv[2]
problem = sys.argv[3]

template = template_path.read_text(encoding="utf-8")
focus_section = f"Focus path: `{focus_path}`\n\n" if focus_path else ""
rendered = template.replace("{{FOCUS_PATH_SECTION}}", focus_section)
rendered = rendered.replace("{{PROBLEM}}", problem)

missing = [token for token in ("{{FOCUS_PATH_SECTION}}", "{{PROBLEM}}") if token in rendered]
if missing:
    raise SystemExit(f"unrendered prompt placeholder(s): {', '.join(missing)}")

print(rendered.strip())
PY

case "${AGENT,,}" in
    codex)
        command -v codex >/dev/null 2>&1 || die "codex executable not found"
        exec codex --ask-for-approval never exec \
            --model "$CODEX_MODEL" \
            -c "model_reasoning_effort=$EFFORT" \
            --sandbox workspace-write \
            --skip-git-repo-check \
            --color never \
            --cd "$RUN_DIR" \
            - <"$PROMPT_FILE"
        ;;
    claude)
        command -v claude >/dev/null 2>&1 || die "claude executable not found"
        exec claude -p \
            --output-format text \
            --model "$CLAUDE_MODEL" \
            --effort "$EFFORT" \
            --permission-mode auto \
            <"$PROMPT_FILE"
        ;;
    *)
        die "--agent must be codex or claude, got: $AGENT"
        ;;
esac
