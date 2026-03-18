#!/usr/bin/env bash
set -euo pipefail
shopt -s extglob

# Determine the merge-base between main and the current branch
MERGE_BASE=$(git merge-base origin/main HEAD)

# Get the list of files changed since the merge-base, ignoring changes on main
CHANGED_FILES=$(git diff --name-only --diff-filter=ACMRT "${MERGE_BASE}..HEAD")

# Check for specific file patterns
CMAKE_CHANGED=false
CLANG_TIDY_CONFIG_CHANGED=false
TTMETALIUM_CHANGED=false
TTNN_CHANGED=false
TTMETALIUM_TESTS_CHANGED=false
TTNN_TESTS_CHANGED=false
TTMETALIUM_OR_TTNN_TESTS_CHANGED=false
TTTRAIN_CHANGED=false
TOOLS_CHANGED=false
ANY_CODE_CHANGED=false
DOCS_CHANGED=false
MODEL_CHARTS_CHANGED=false
MODELS_CHANGED=false
BUILD_WORKFLOWS_CHANGED=false

# LLK-specific change detection
LLK_ENGINE_CHANGED=false
LLK_QUASAR_CHANGED=false
LLK_TESTS_CHANGED=false
LLK_PERF_CHANGED=false
LLK_CI_CHANGED=false

while IFS= read -r FILE; do
    case "$FILE" in
        CMakeLists.txt|**/CMakeLists.txt|**/*.cmake)
            CMAKE_CHANGED=true
            ;;
        tt_metal/sfpi-info.sh)
            # Read in by a cmake file
            CMAKE_CHANGED=true
            ;;
        tt_metal/sfpi-version)
            # Read in by a cmake file
            CMAKE_CHANGED=true
            ;;
        .clang-tidy|**/.clang-tidy)
            CLANG_TIDY_CONFIG_CHANGED=true
            ;;
        tt_stl/**/*.@(h|hpp|c|cpp))
            # TT-STL is so small; not going to be so fine grained; just treat it as a TT-Metalium change
            TTMETALIUM_CHANGED=true
            ANY_CODE_CHANGED=true
            ;;
        tt_metal/**/*.@(h|hpp|c|cpp|cc|py))
            TTMETALIUM_CHANGED=true
            ANY_CODE_CHANGED=true
            ;;
        ttnn/**/*.@(h|hpp|c|cpp|py))
            TTNN_CHANGED=true
            ANY_CODE_CHANGED=true
            ;;
        tests/tt_metal/**/*.@(h|hpp|c|cpp|py))
            TTMETALIUM_TESTS_CHANGED=true
            ANY_CODE_CHANGED=true
            ;;
        tests/ttnn/**/*.@(h|hpp|c|cpp|py))
            TTNN_TESTS_CHANGED=true
            ANY_CODE_CHANGED=true
            ;;
        tt-train/**/*.@(h|hpp|c|cpp|py))
            TTTRAIN_CHANGED=true
            ANY_CODE_CHANGED=true
            ;;
        tools/**/*.@(h|hpp|c|cpp|py))
            TOOLS_CHANGED=true
            ANY_CODE_CHANGED=true
            ;;
        docs/**|**/*.rst|**/*.md)
            DOCS_CHANGED=true
            if [[ "$FILE" == "README.md" || "$FILE" == "models/README.md" ]]; then
               MODEL_CHARTS_CHANGED=true
            fi
            ;;
        models/**)
            MODELS_CHANGED=true
            ANY_CODE_CHANGED=true
            ;;
        .github/workflows/build-artifact.yaml|.github/workflows/build-docker-artifact.yaml|.github/workflows/ttsim.yaml|.github/workflows/fabric-cpu-only-tests-impl.yaml)
            BUILD_WORKFLOWS_CHANGED=true
            ANY_CODE_CHANGED=true
            ;;
        # LLK-specific change detection (detailed patterns before generic submodule)
        tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/**|tt_metal/third_party/tt_llk/tt_llk_blackhole/**|tt_metal/third_party/tt_llk/common/**)
            LLK_ENGINE_CHANGED=true
            ;;
        tt_metal/third_party/tt_llk/tt_llk_quasar/**)
            LLK_QUASAR_CHANGED=true
            ;;
        # Perf pattern must come BEFORE generic tests pattern since perf files are under tests/
        tt_metal/third_party/tt_llk/tests/**/perf/**|tt_metal/third_party/tt_llk/tests/**/*perf*)
            LLK_PERF_CHANGED=true
            LLK_TESTS_CHANGED=true
            ;;
        tt_metal/third_party/tt_llk/tests/**)
            LLK_TESTS_CHANGED=true
            ;;
        # LLK CI/infrastructure changes
        .github/workflows/llk-*.yaml|.github/scripts/llk-*.sh)
            LLK_CI_CHANGED=true
            ;;
    esac
done <<< "$CHANGED_FILES"

# FIXME: Can we do this better?
SUBMODULE_PATHS=$(git config --file .gitmodules --get-regexp path | awk '{print $2}')
SUBMODULE_CHANGED=false
for submodule_path in $SUBMODULE_PATHS; do
    # Skip LLK submodule - it has its own fine-grained change detection above
    if [[ "$submodule_path" == "tt_metal/third_party/tt_llk" ]]; then
        continue
    fi
    if echo "$CHANGED_FILES" | grep -q "^$submodule_path"; then
        SUBMODULE_CHANGED=true
        break
    fi
done
if [[ "$SUBMODULE_CHANGED" = true ]]; then
    # Treat any submodule change as a change to everything; not going to manage dependency trees for this
    TTMETALIUM_CHANGED=true
    TTNN_CHANGED=true
    TTMETALIUM_TESTS_CHANGED=true
    TTNN_TESTS_CHANGED=true
    TTTRAIN_CHANGED=true
    # TODO: Well, this could likely just depend on the UMD submodule changing...
    # Something to make more efficient in future.
    TOOLS_CHANGED=true
    ANY_CODE_CHANGED=true
    # Issue: https://github.com/tenstorrent/tt-metal/issues/31344
    CMAKE_CHANGED=true
fi

# LLK engine changes should also trigger metal tests since LLK headers are used in metal
if [[ "$LLK_ENGINE_CHANGED" = true ]]; then
    TTMETALIUM_CHANGED=true
    ANY_CODE_CHANGED=true
fi

# LLK CI/infrastructure changes should trigger LLK tests
if [[ "$LLK_CI_CHANGED" = true ]]; then
    LLK_TESTS_CHANGED=true
fi

# Derive combined tests-changed flag from isolated flags
if [[ "$TTMETALIUM_TESTS_CHANGED" = true || "$TTNN_TESTS_CHANGED" = true ]]; then
    TTMETALIUM_OR_TTNN_TESTS_CHANGED=true
else
    TTMETALIUM_OR_TTNN_TESTS_CHANGED=false
fi

declare -A changes=(
    [cmake-changed]=$CMAKE_CHANGED
    [clang-tidy-config-changed]=$CLANG_TIDY_CONFIG_CHANGED
    [tt-metalium-changed]=$TTMETALIUM_CHANGED
    [tt-nn-changed]=$TTNN_CHANGED
    [tt-metalium-tests-changed]=$TTMETALIUM_TESTS_CHANGED
    [tt-nn-tests-changed]=$TTNN_TESTS_CHANGED
    [tt-metalium-or-tt-nn-tests-changed]=$TTMETALIUM_OR_TTNN_TESTS_CHANGED
    [tt-train-changed]=$TTTRAIN_CHANGED
    [tools-changed]=$TOOLS_CHANGED
    [submodule-changed]=$SUBMODULE_CHANGED
    [any-code-changed]=$ANY_CODE_CHANGED
    [docs-changed]=$DOCS_CHANGED
    [model-charts-changed]=$MODEL_CHARTS_CHANGED
    [models-changed]=$MODELS_CHANGED
    [build-workflows-changed]=$BUILD_WORKFLOWS_CHANGED
    [llk-engine-changed]=$LLK_ENGINE_CHANGED
    [llk-quasar-changed]=$LLK_QUASAR_CHANGED
    [llk-tests-changed]=$LLK_TESTS_CHANGED
    [llk-perf-changed]=$LLK_PERF_CHANGED
    [llk-ci-changed]=$LLK_CI_CHANGED
)

for var in "${!changes[@]}"; do
    echo "$var=${changes[$var]}"
    if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
        # Output results in GitHub Actions format when run in GHA
        echo "$var=${changes[$var]}" >> "$GITHUB_OUTPUT"
    fi
done
