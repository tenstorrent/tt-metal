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

# Fine-grained TTNN operation family tracking for change-aware test routing.
# Each flag maps to a test group in tests/pipeline_reorg/ttnn-tests.yaml via ops_filter.
# When "all" is set, every TTNN test group should run (core dependency changed).
declare -A TTNN_OPS_CHANGED=()

while IFS= read -r FILE; do
    case "$FILE" in
        CMakeLists.txt|**/CMakeLists.txt|**/*.cmake)
            CMAKE_CHANGED=true
            # Build system change → all TTNN ops need testing
            TTNN_OPS_CHANGED[all]=1
            ;;
        tt_metal/sfpi-info.sh)
            # Read in by a cmake file
            CMAKE_CHANGED=true
            TTNN_OPS_CHANGED[all]=1
            ;;
        tt_metal/sfpi-version)
            # Read in by a cmake file
            CMAKE_CHANGED=true
            TTNN_OPS_CHANGED[all]=1
            ;;
        .clang-tidy|**/.clang-tidy)
            CLANG_TIDY_CONFIG_CHANGED=true
            ;;
        tt_stl/**/*.@(h|hpp|c|cpp))
            # TT-STL is so small; not going to be so fine grained; just treat it as a TT-Metalium change
            TTMETALIUM_CHANGED=true
            ANY_CODE_CHANGED=true
            # Core dependency → all TTNN ops need testing
            TTNN_OPS_CHANGED[all]=1
            ;;
        tt_metal/**/*.@(h|hpp|c|cpp|cc|py))
            TTMETALIUM_CHANGED=true
            ANY_CODE_CHANGED=true
            # Core dependency → all TTNN ops need testing
            TTNN_OPS_CHANGED[all]=1
            ;;

        # =====================================================================
        # Fine-grained TTNN operation detection
        # Order matters: more specific patterns must come before general ttnn/**
        # =====================================================================

        # C++ source: ttnn/cpp/ttnn/operations/<family>/...
        ttnn/cpp/ttnn/operations/eltwise/**/*.@(h|hpp|c|cpp))
            TTNN_CHANGED=true
            ANY_CODE_CHANGED=true
            TTNN_OPS_CHANGED[eltwise]=1
            ;;
        ttnn/cpp/ttnn/operations/conv/**/*.@(h|hpp|c|cpp))
            TTNN_CHANGED=true
            ANY_CODE_CHANGED=true
            TTNN_OPS_CHANGED[conv]=1
            ;;
        ttnn/cpp/ttnn/operations/pool/**/*.@(h|hpp|c|cpp))
            TTNN_CHANGED=true
            ANY_CODE_CHANGED=true
            TTNN_OPS_CHANGED[pool]=1
            ;;
        ttnn/cpp/ttnn/operations/matmul/**/*.@(h|hpp|c|cpp))
            TTNN_CHANGED=true
            ANY_CODE_CHANGED=true
            TTNN_OPS_CHANGED[matmul]=1
            ;;
        ttnn/cpp/ttnn/operations/normalization/**/*.@(h|hpp|c|cpp))
            # normalization source maps to "fused" test group
            TTNN_CHANGED=true
            ANY_CODE_CHANGED=true
            TTNN_OPS_CHANGED[fused]=1
            ;;
        ttnn/cpp/ttnn/operations/transformer/sdpa*/**/*.@(h|hpp|c|cpp))
            TTNN_CHANGED=true
            ANY_CODE_CHANGED=true
            TTNN_OPS_CHANGED[sdpa]=1
            ;;
        ttnn/cpp/ttnn/operations/transformer/**/*.@(h|hpp|c|cpp))
            TTNN_CHANGED=true
            ANY_CODE_CHANGED=true
            TTNN_OPS_CHANGED[transformers]=1
            ;;
        ttnn/cpp/ttnn/operations/reduction/**/*.@(h|hpp|c|cpp)|ttnn/cpp/ttnn/operations/debug/**/*.@(h|hpp|c|cpp))
            # reduction + debug source maps to "reduce" test group
            TTNN_CHANGED=true
            ANY_CODE_CHANGED=true
            TTNN_OPS_CHANGED[reduce]=1
            ;;
        ttnn/cpp/ttnn/operations/**/*.@(h|hpp|c|cpp))
            # Any other operations subdir is a core/cross-cutting change → run all
            TTNN_CHANGED=true
            ANY_CODE_CHANGED=true
            TTNN_OPS_CHANGED[all]=1
            ;;

        # Python source: ttnn/ttnn/operations/<module>.py
        ttnn/ttnn/operations/@(binary|unary|ternary|activations|comparison|binary_backward|unary_backward|ternary_backward|binary_complex|unary_complex|complex_unary_backward).py)
            TTNN_CHANGED=true
            ANY_CODE_CHANGED=true
            TTNN_OPS_CHANGED[eltwise]=1
            ;;
        ttnn/ttnn/operations/conv2d.py)
            TTNN_CHANGED=true
            ANY_CODE_CHANGED=true
            TTNN_OPS_CHANGED[conv]=1
            ;;
        ttnn/ttnn/operations/pool.py)
            TTNN_CHANGED=true
            ANY_CODE_CHANGED=true
            TTNN_OPS_CHANGED[pool]=1
            ;;
        ttnn/ttnn/operations/matmul.py)
            TTNN_CHANGED=true
            ANY_CODE_CHANGED=true
            TTNN_OPS_CHANGED[matmul]=1
            ;;
        ttnn/ttnn/operations/normalization.py)
            TTNN_CHANGED=true
            ANY_CODE_CHANGED=true
            TTNN_OPS_CHANGED[fused]=1
            ;;
        ttnn/ttnn/operations/transformer.py)
            TTNN_CHANGED=true
            ANY_CODE_CHANGED=true
            TTNN_OPS_CHANGED[transformers]=1
            TTNN_OPS_CHANGED[sdpa]=1
            ;;
        ttnn/ttnn/operations/reduction.py)
            TTNN_CHANGED=true
            ANY_CODE_CHANGED=true
            TTNN_OPS_CHANGED[reduce]=1
            ;;
        ttnn/ttnn/operations/*.py)
            # Other operation Python files → core change
            TTNN_CHANGED=true
            ANY_CODE_CHANGED=true
            TTNN_OPS_CHANGED[all]=1
            ;;

        # TTNN example usage files
        ttnn/ttnn/examples/**/*.py)
            TTNN_CHANGED=true
            ANY_CODE_CHANGED=true
            TTNN_OPS_CHANGED[example]=1
            ;;

        # Core TTNN source (non-operations) → all ops affected
        ttnn/**/*.@(h|hpp|c|cpp|py))
            TTNN_CHANGED=true
            ANY_CODE_CHANGED=true
            TTNN_OPS_CHANGED[all]=1
            ;;

        # =====================================================================
        # Fine-grained TTNN test file detection
        # =====================================================================
        tests/ttnn/unit_tests/operations/eltwise/**/*.py)
            TTNN_TESTS_CHANGED=true
            ANY_CODE_CHANGED=true
            TTNN_OPS_CHANGED[eltwise]=1
            ;;
        tests/ttnn/unit_tests/operations/conv/**/*.py)
            TTNN_TESTS_CHANGED=true
            ANY_CODE_CHANGED=true
            TTNN_OPS_CHANGED[conv]=1
            ;;
        tests/ttnn/unit_tests/operations/pool/**/*.py)
            TTNN_TESTS_CHANGED=true
            ANY_CODE_CHANGED=true
            TTNN_OPS_CHANGED[pool]=1
            ;;
        tests/ttnn/unit_tests/operations/matmul/**/*.py)
            TTNN_TESTS_CHANGED=true
            ANY_CODE_CHANGED=true
            TTNN_OPS_CHANGED[matmul]=1
            ;;
        tests/ttnn/unit_tests/operations/fused/**/*.py)
            TTNN_TESTS_CHANGED=true
            ANY_CODE_CHANGED=true
            TTNN_OPS_CHANGED[fused]=1
            ;;
        tests/ttnn/unit_tests/operations/transformers/**/*.py)
            TTNN_TESTS_CHANGED=true
            ANY_CODE_CHANGED=true
            TTNN_OPS_CHANGED[transformers]=1
            ;;
        tests/ttnn/unit_tests/operations/sdpa/**/*.py)
            TTNN_TESTS_CHANGED=true
            ANY_CODE_CHANGED=true
            TTNN_OPS_CHANGED[sdpa]=1
            ;;
        tests/ttnn/unit_tests/operations/reduce/**/*.py|tests/ttnn/unit_tests/operations/debug/**/*.py)
            TTNN_TESTS_CHANGED=true
            ANY_CODE_CHANGED=true
            TTNN_OPS_CHANGED[reduce]=1
            ;;
        tests/ttnn/unit_tests/@(base_functionality|tensor|benchmarks)/**/*.py)
            TTNN_TESTS_CHANGED=true
            ANY_CODE_CHANGED=true
            TTNN_OPS_CHANGED[core]=1
            ;;
        tests/scripts/*ttnn*)
            # Example test scripts
            TTNN_TESTS_CHANGED=true
            ANY_CODE_CHANGED=true
            TTNN_OPS_CHANGED[example]=1
            ;;
        tests/ttnn/**/*.@(h|hpp|c|cpp|py))
            # Any other ttnn test change → run all
            TTNN_TESTS_CHANGED=true
            ANY_CODE_CHANGED=true
            TTNN_OPS_CHANGED[all]=1
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
    TTMETALIUM_TESTS_CHANGED=true
    TTNN_TESTS_CHANGED=true
    TTTRAIN_CHANGED=true
    # TODO: Well, this could likely just depend on the UMD submodule changing...
    # Something to make more efficient in future.
    TOOLS_CHANGED=true
    ANY_CODE_CHANGED=true
    # Issue: https://github.com/tenstorrent/tt-metal/issues/31344
    CMAKE_CHANGED=true
    TTNN_OPS_CHANGED[all]=1
fi

# Derive combined tests-changed flag from isolated flags
if [[ "$TTMETALIUM_TESTS_CHANGED" = true || "$TTNN_TESTS_CHANGED" = true ]]; then
    TTMETALIUM_OR_TTNN_TESTS_CHANGED=true
else
    TTMETALIUM_OR_TTNN_TESTS_CHANGED=false
fi

# Build the ttnn-changed-ops output: comma-separated list of changed op families.
# "all" means every TTNN test group should run (core dependency changed).
# Empty means no TTNN ops changed.
if [[ ${TTNN_OPS_CHANGED[all]+_} ]]; then
    TTNN_CHANGED_OPS="all"
else
    TTNN_CHANGED_OPS=$(IFS=,; echo "${!TTNN_OPS_CHANGED[*]}")
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
    [ttnn-changed-ops]=$TTNN_CHANGED_OPS
)

for var in "${!changes[@]}"; do
    echo "$var=${changes[$var]}"
    if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
        # Output results in GitHub Actions format when run in GHA
        echo "$var=${changes[$var]}" >> "$GITHUB_OUTPUT"
    fi
done
