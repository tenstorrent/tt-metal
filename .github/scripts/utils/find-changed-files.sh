#!/usr/bin/env bash
set -euo pipefail

# Determine the merge-base between main and the current branch
MERGE_BASE=$(git merge-base origin/main HEAD)

# Get the list of files changed since the merge-base, ignoring changes on main
CHANGED_FILES=$(git diff --name-only --diff-filter=ACMRT "${MERGE_BASE}..HEAD")

# Check for specific file patterns
CMAKE_CHANGED=false
CLANG_TIDY_CONFIG_CHANGED=false
TTMETALIUM_CHANGED=false
TTNN_CHANGED=false
TTMETALIUM_OR_TTNN_TESTS_CHANGED=false
TTTRAIN_CHANGED=false
ANY_CODE_CHANGED=false
DOCS_CHANGED=false

while IFS= read -r FILE; do
    case "$FILE" in
        CMakeLists.txt|**/CMakeLists.txt|**/*.cmake)
            CMAKE_CHANGED=true
            ;;
	tt_metal/sfpi-version.sh)
	    # Read in by a cmake file
            CMAKE_CHANGED=true
            ;;
        .clang-tidy|**/.clang-tidy)
            CLANG_TIDY_CONFIG_CHANGED=true
            ;;
        tt_stl/**/*.h|tt_stl/**/*.hpp|tt_stl/**/*.c|tt_stl/**/*.cpp)
            # TT-STL is so small; not going to be so fine grained; just treat it as a TT-Metalium change
            TTMETALIUM_CHANGED=true
            ANY_CODE_CHANGED=true
            ;;
        tt_metal/**/*.h|tt_metal/**/*.hpp|tt_metal/**/*.c|tt_metal/**/*.cpp)
            TTMETALIUM_CHANGED=true
            ANY_CODE_CHANGED=true
            ;;
        ttnn/**/*.h|ttnn/**/*.hpp|ttnn/**/*.c|ttnn/**/*.cpp)
            TTNN_CHANGED=true
            ANY_CODE_CHANGED=true
            ;;
        tests/**/*.h|tests/**/*.hpp|tests/**/*.c|tests/**/*.cpp)
            TTMETALIUM_OR_TTNN_TESTS_CHANGED=true
            ANY_CODE_CHANGED=true
            ;;
        tt-train/**/*.h|tt-train/**/*.hpp|tt-train/**/*.c|tt-train/**/*.cpp)
            TTTRAIN_CHANGED=true
            ANY_CODE_CHANGED=true
            ;;
        docs/**|**/*.rst|**/*.md)
            DOCS_CHANGED=true
            ;;
    esac
done <<< "$CHANGED_FILES"

# FIXME: Can we do this better?
SUBMODULE_PATHS=$(git config --file .gitmodules --get-regexp path | awk '{print $2}')
SUBMODULE_CHANGED=false
for submodule_path in $SUBMODULE_PATHS; do
    if echo "$CHANGED_FILES" | grep -q "^$submodule_path"; then
        SUBMODULE_CHANGED=true
        break
    fi
done
if [[ "$SUBMODULE_CHANGED" = true ]]; then
    # Treat any submodule change as a change to everything; not going to manage dependency trees for this
    TTMETALIUM_CHANGED=true
    TTNN_CHANGED=true
    TTMETALIUM_OR_TTNN_TESTS_CHANGED=true
    TTTRAIN_CHANGED=true
    ANY_CODE_CHANGED=true
fi

declare -A changes=(
    [cmake-changed]=$CMAKE_CHANGED
    [clang-tidy-config-changed]=$CLANG_TIDY_CONFIG_CHANGED
    [tt-metalium-changed]=$TTMETALIUM_CHANGED
    [tt-nn-changed]=$TTNN_CHANGED
    [tt-metalium-or-tt-nn-tests-changed]=$TTMETALIUM_OR_TTNN_TESTS_CHANGED
    [tt-train-changed]=$TTTRAIN_CHANGED
    [submodule-changed]=$SUBMODULE_CHANGED
    [any-code-changed]=$ANY_CODE_CHANGED
    [docs-changed]=$DOCS_CHANGED
)

for var in "${!changes[@]}"; do
    echo "$var=${changes[$var]}"
    if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
        # Output results in GitHub Actions format when run in GHA
        echo "$var=${changes[$var]}" >> "$GITHUB_OUTPUT"
    fi
done
