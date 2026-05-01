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
LLK_WORMHOLE_CHANGED=false
LLK_BLACKHOLE_CHANGED=false
LLK_COMMON_CHANGED=false
LLK_SFPI_CHANGED=false
LLK_QUASAR_CHANGED=false
LLK_TESTS_CHANGED=false
LLK_PERF_CHANGED=false
LLK_CI_CHANGED=false


while IFS= read -r FILE; do
    case "$FILE" in
        CMakeLists.txt|**/CMakeLists.txt|**/*.cmake)
            CMAKE_CHANGED=true
            ;;
        tt_metal/sfpi-info.sh|tt_metal/sfpi-version)
            # Read in by a cmake file; also pins the SFPI compiler used to build LLK
            # device kernels, so any change must re-run LLK tests on all archs.
            CMAKE_CHANGED=true
            LLK_SFPI_CHANGED=true
            ;;
        .clang-tidy|**/.clang-tidy)
            CLANG_TIDY_CONFIG_CHANGED=true
            ;;
        tt_stl/**/*.@(h|hpp|c|cpp))
            # TT-STL is so small; not going to be so fine grained; just treat it as a TT-Metalium change
            TTMETALIUM_CHANGED=true
            ANY_CODE_CHANGED=true
            ;;
        # LLK-specific patterns — must come before the generic tt_metal/** catch-all.
        tt_metal/tt-llk/.github/**|tt_metal/tt-llk/tests/requirements.txt)
            LLK_CI_CHANGED=true
            ;;
        tt_metal/tt-llk/tt_llk_wormhole_b0/**|tt_metal/hw/ckernels/wormhole_b0/**)
            LLK_WORMHOLE_CHANGED=true
            ;;
        tt_metal/tt-llk/tt_llk_blackhole/**|tt_metal/hw/ckernels/blackhole/**)
            LLK_BLACKHOLE_CHANGED=true
            ;;
        tt_metal/tt-llk/common/**)
            LLK_COMMON_CHANGED=true
            ;;
        tt_metal/tt-llk/tt_llk_quasar/**|tt_metal/tt-llk/tests/sources/quasar/**|tt_metal/tt-llk/tests/python_tests/quasar/**|tt_metal/hw/ckernels/quasar/**)
            LLK_QUASAR_CHANGED=true
            ;;
        tt_metal/tt-llk/tests/**/perf/**|tt_metal/tt-llk/tests/**/*perf*)
            LLK_PERF_CHANGED=true
            ;;
        tt_metal/tt-llk/tests/**)
            LLK_TESTS_CHANGED=true
            ;;
        .github/workflows/llk-*.yaml|.github/scripts/llk-*.sh)
            LLK_CI_CHANGED=true
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
        tt_metal/python_env/requirements*.txt|tt_metal/python_env/create_venv.sh)
            # Runtime dependency changes can alter behavior of tests/tooling
            # without touching C++/Python source directly.
            ANY_CODE_CHANGED=true
            ;;
        tools/triage/requirements.txt)
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
    esac
done <<< "$CHANGED_FILES"

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

# LLK engine changes imply Metalium may be affected (LLK is compiled into device kernels)
if [[ "$LLK_WORMHOLE_CHANGED" = true || "$LLK_BLACKHOLE_CHANGED" = true || "$LLK_COMMON_CHANGED" = true || "$LLK_SFPI_CHANGED" = true ]]; then
    TTMETALIUM_CHANGED=true
    ANY_CODE_CHANGED=true
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
    [llk-wormhole-changed]=$LLK_WORMHOLE_CHANGED
    [llk-blackhole-changed]=$LLK_BLACKHOLE_CHANGED
    [llk-common-changed]=$LLK_COMMON_CHANGED
    [llk-sfpi-changed]=$LLK_SFPI_CHANGED
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
