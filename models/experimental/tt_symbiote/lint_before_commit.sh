#!/usr/bin/env bash
# Run Python linter/formatter (same as pre-commit) before committing.
# Usage (from repo root):
#   ./models/experimental/tt_symbiote/lint_before_commit.sh
#     → runs on staged .py files, or changed .py files under tt_symbiote
#   ./models/experimental/tt_symbiote/lint_before_commit.sh path/to/file.py ...
#     → runs on the given file(s)
set -e
command -v pre-commit >/dev/null 2>&1 || { echo "pre-commit not found. Install with: pip install pre-commit"; exit 1; }
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
if [ $# -gt 0 ]; then
    pre-commit run --files "$@"
else
    FILES="$(git diff --cached --name-only --diff-filter=ACM 2>/dev/null | grep '\.py$' || true)"
    if [ -z "$FILES" ]; then
        FILES="$(git diff --name-only HEAD -- 'models/experimental/tt_symbiote/' 2>/dev/null | grep '\.py$' || true)"
    fi
    if [ -z "$FILES" ]; then
        echo "No Python files to check. Stage files or run: $0 path/to/file.py"
        exit 0
    fi
    echo "Running pre-commit (black, autoflake, etc.) on:"
    echo "$FILES"
    echo "$FILES" | xargs -r pre-commit run --files
fi
