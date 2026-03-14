#!/usr/bin/env bash
# find_changed_files.sh - Determine which file categories changed relative to
# a base ref (default: origin/main).  Exports boolean env vars for each
# category.
#
# Ported from: .github/actions/find-changed-files/action.yml
#              .github/scripts/utils/find-changed-files.sh

set -euo pipefail
shopt -s extglob

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "$SCRIPT_DIR/../lib/common.sh"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

BASE_REF="origin/main"

usage() {
    cat >&2 <<EOF
Usage: $(basename "$0") [OPTIONS]

Determine which categories of files changed on the current branch compared
to a base ref.  Results are printed as KEY=VALUE lines and exported as
environment variables when sourced.

Options:
  --base-ref REF   Base ref for git merge-base [default: $BASE_REF]
  -h, --help       Show this help

Exported variables:
  CMAKE_CHANGED, CLANG_TIDY_CONFIG_CHANGED, TT_METALIUM_CHANGED,
  TT_NN_CHANGED, TT_METALIUM_TESTS_CHANGED, TT_NN_TESTS_CHANGED,
  TT_METALIUM_OR_TT_NN_TESTS_CHANGED, TT_TRAIN_CHANGED, TOOLS_CHANGED,
  SUBMODULE_CHANGED, ANY_CODE_CHANGED, DOCS_CHANGED, MODEL_CHARTS_CHANGED,
  MODELS_CHANGED, BUILD_WORKFLOWS_CHANGED
EOF
    exit 1
}

# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case "$1" in
        --base-ref) BASE_REF="$2"; shift 2 ;;
        -h|--help)  usage ;;
        *) log_fatal "Unknown argument: $1" ;;
    esac
done

# ---------------------------------------------------------------------------
# Find changed files
# ---------------------------------------------------------------------------

MERGE_BASE=$(git merge-base "$BASE_REF" HEAD)
CHANGED_FILES=$(git diff --name-only --diff-filter=ACMRT "${MERGE_BASE}..HEAD")

CMAKE_CHANGED=false
CLANG_TIDY_CONFIG_CHANGED=false
TT_METALIUM_CHANGED=false
TT_NN_CHANGED=false
TT_METALIUM_TESTS_CHANGED=false
TT_NN_TESTS_CHANGED=false
TT_METALIUM_OR_TT_NN_TESTS_CHANGED=false
TT_TRAIN_CHANGED=false
TOOLS_CHANGED=false
ANY_CODE_CHANGED=false
DOCS_CHANGED=false
MODEL_CHARTS_CHANGED=false
MODELS_CHANGED=false
BUILD_WORKFLOWS_CHANGED=false
SUBMODULE_CHANGED=false

while IFS= read -r FILE; do
    [[ -z "$FILE" ]] && continue
    case "$FILE" in
        CMakeLists.txt|**/CMakeLists.txt|**/*.cmake)
            CMAKE_CHANGED=true ;;
        tt_metal/sfpi-info.sh|tt_metal/sfpi-version)
            CMAKE_CHANGED=true ;;
        .clang-tidy|**/.clang-tidy)
            CLANG_TIDY_CONFIG_CHANGED=true ;;
        tt_stl/**/*.@(h|hpp|c|cpp))
            TT_METALIUM_CHANGED=true; ANY_CODE_CHANGED=true ;;
        tt_metal/**/*.@(h|hpp|c|cpp|cc|py))
            TT_METALIUM_CHANGED=true; ANY_CODE_CHANGED=true ;;
        ttnn/**/*.@(h|hpp|c|cpp|py))
            TT_NN_CHANGED=true; ANY_CODE_CHANGED=true ;;
        tests/tt_metal/**/*.@(h|hpp|c|cpp|py))
            TT_METALIUM_TESTS_CHANGED=true; ANY_CODE_CHANGED=true ;;
        tests/ttnn/**/*.@(h|hpp|c|cpp|py))
            TT_NN_TESTS_CHANGED=true; ANY_CODE_CHANGED=true ;;
        tt-train/**/*.@(h|hpp|c|cpp|py))
            TT_TRAIN_CHANGED=true; ANY_CODE_CHANGED=true ;;
        tools/**/*.@(h|hpp|c|cpp|py))
            TOOLS_CHANGED=true; ANY_CODE_CHANGED=true ;;
        docs/**|**/*.rst|**/*.md)
            DOCS_CHANGED=true
            if [[ "$FILE" == "README.md" || "$FILE" == "models/README.md" ]]; then
                MODEL_CHARTS_CHANGED=true
            fi
            ;;
        models/**)
            MODELS_CHANGED=true; ANY_CODE_CHANGED=true ;;
        .github/workflows/build-artifact.yaml|.github/workflows/build-docker-artifact.yaml|.github/workflows/ttsim.yaml|.github/workflows/fabric-cpu-only-tests-impl.yaml)
            BUILD_WORKFLOWS_CHANGED=true; ANY_CODE_CHANGED=true ;;
    esac
done <<< "$CHANGED_FILES"

# Submodule changes
SUBMODULE_PATHS=$(git config --file .gitmodules --get-regexp path 2>/dev/null | awk '{print $2}' || true)
for submodule_path in $SUBMODULE_PATHS; do
    if echo "$CHANGED_FILES" | grep -q "^$submodule_path"; then
        SUBMODULE_CHANGED=true
        break
    fi
done

if [[ "$SUBMODULE_CHANGED" == "true" ]]; then
    TT_METALIUM_CHANGED=true
    TT_NN_CHANGED=true
    TT_METALIUM_TESTS_CHANGED=true
    TT_NN_TESTS_CHANGED=true
    TT_TRAIN_CHANGED=true
    TOOLS_CHANGED=true
    ANY_CODE_CHANGED=true
    CMAKE_CHANGED=true
fi

if [[ "$TT_METALIUM_TESTS_CHANGED" == "true" || "$TT_NN_TESTS_CHANGED" == "true" ]]; then
    TT_METALIUM_OR_TT_NN_TESTS_CHANGED=true
fi

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

declare -A changes=(
    [CMAKE_CHANGED]=$CMAKE_CHANGED
    [CLANG_TIDY_CONFIG_CHANGED]=$CLANG_TIDY_CONFIG_CHANGED
    [TT_METALIUM_CHANGED]=$TT_METALIUM_CHANGED
    [TT_NN_CHANGED]=$TT_NN_CHANGED
    [TT_METALIUM_TESTS_CHANGED]=$TT_METALIUM_TESTS_CHANGED
    [TT_NN_TESTS_CHANGED]=$TT_NN_TESTS_CHANGED
    [TT_METALIUM_OR_TT_NN_TESTS_CHANGED]=$TT_METALIUM_OR_TT_NN_TESTS_CHANGED
    [TT_TRAIN_CHANGED]=$TT_TRAIN_CHANGED
    [TOOLS_CHANGED]=$TOOLS_CHANGED
    [SUBMODULE_CHANGED]=$SUBMODULE_CHANGED
    [ANY_CODE_CHANGED]=$ANY_CODE_CHANGED
    [DOCS_CHANGED]=$DOCS_CHANGED
    [MODEL_CHARTS_CHANGED]=$MODEL_CHARTS_CHANGED
    [MODELS_CHANGED]=$MODELS_CHANGED
    [BUILD_WORKFLOWS_CHANGED]=$BUILD_WORKFLOWS_CHANGED
)

for var in "${!changes[@]}"; do
    export "$var=${changes[$var]}"
    echo "${var}=${changes[$var]}"
done
