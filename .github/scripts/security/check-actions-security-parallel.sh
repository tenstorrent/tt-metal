#!/usr/bin/env bash
# Parallel wrapper for GitHub Actions Security Linting
# Runs per-file checks in parallel using xargs, then aggregate checks once.
#
# Usage: ./check-actions-security-parallel.sh [OPTIONS] [FILE...]
#   -h, --help    Show help message
#   -j N          Number of parallel jobs (default: 0 = all available CPUs)
#   --strict      Exit with error code if any issues found
#   --precommit   Pre-commit mode: skip if no files provided (don't scan all)
#
# If no FILEs are provided, scans all .yml/.yaml files in .github/workflows and .github/actions.
# In --precommit mode, if no FILEs are provided, shows "Skipped" and exits.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
GITHUB_DIR="$REPO_ROOT/.github"
MAIN_SCRIPT="$SCRIPT_DIR/check-actions-security.sh"

PARALLEL_JOBS=0
STRICT_MODE=false
PRECOMMIT_MODE=false
FILES=()

usage() {
    cat <<'EOF'
Usage: ./check-actions-security-parallel.sh [OPTIONS] [FILE...]

Options:
  -h, --help    Show this help message and exit
  -j N          Number of parallel jobs (default: 0 = all available CPUs)
  --strict      Exit with error code if any issues found
  --precommit   Pre-commit mode: skip if no files provided (don't scan all)

This wrapper runs security checks in parallel for better performance.
Per-file checks run in parallel using xargs -P, then aggregate checks run once.

If no FILEs are provided, scans all .yml/.yaml files in .github/workflows and .github/actions.
In --precommit mode, if no FILEs are provided, shows "Skipped" and exits.

For check descriptions, run: ./check-actions-security.sh --help
EOF
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            ;;
        -j)
            if [[ -z "${2:-}" ]]; then
                echo "Error: -j requires a number argument" >&2
                exit 1
            fi
            PARALLEL_JOBS="$2"
            shift 2
            ;;
        --strict)
            STRICT_MODE=true
            shift
            ;;
        --precommit)
            PRECOMMIT_MODE=true
            shift
            ;;
        *)
            FILES+=("$1")
            shift
            ;;
    esac
done

# Create temp file for results, clean up on exit
RESULTS_FILE=$(mktemp)
trap 'rm -f "$RESULTS_FILE"' EXIT

# Handle no files case
if [[ ${#FILES[@]} -eq 0 ]]; then
    if [[ "$PRECOMMIT_MODE" == "true" ]]; then
        # In pre-commit mode with no files, show skipped message
        echo "Skipped (no .github workflow files in commit)"
        echo ""
        exit 0
    else
        # Default mode: scan all workflow files
        while IFS= read -r -d '' f; do
            FILES+=("$f")
        done < <(find "$GITHUB_DIR/workflows" "$GITHUB_DIR/actions" \
            \( -name "*.yml" -o -name "*.yaml" \) -print0 2>/dev/null)
    fi
fi

# Exit early if still no files to check
if [[ ${#FILES[@]} -eq 0 ]]; then
    echo "No workflow files found to check."
    echo ""
    exit 0
fi

# Phase 1: Run per-file checks in parallel (skip aggregates 5,6)
# Use -P0 for all available CPUs, -n 5 to batch 5 files per invocation
printf '%s\0' "${FILES[@]}" | \
    xargs -0 -P "$PARALLEL_JOBS" -n 5 \
    "$MAIN_SCRIPT" --skip-aggregate --machine-output \
    >> "$RESULTS_FILE" 2>&1 || true

# Phase 2: Run aggregate checks once with all files
"$MAIN_SCRIPT" -c 5,6 --machine-output "${FILES[@]}" \
    >> "$RESULTS_FILE" 2>&1 || true

# Phase 3: Format and display results with deduplicated examples
STRICT_ARG=""
if [[ "$STRICT_MODE" == "true" ]]; then
    STRICT_ARG="--strict"
fi

exec "$MAIN_SCRIPT" --format-results "$RESULTS_FILE" $STRICT_ARG
