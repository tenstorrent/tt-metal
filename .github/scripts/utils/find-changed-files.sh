#!/usr/bin/env bash
set -euo pipefail
shopt -s extglob

# Determine the merge-base between main and the current branch
MERGE_BASE=$(git merge-base origin/main HEAD)

# Get the list of files changed since the merge-base, ignoring changes on main
#
# D (deleted) is included. Removing a source or header breaks a build exactly as
# thoroughly as editing one, but without D every flag below read false for a
# deletion-only PR -- so the artifact build itself was skipped and every gate
# keyed off it went with it. Deleting a file listed in a *.cmake was partly
# covered by the accompanying cmake edit, but headers are not listed in cmake at
# all and were invisible outright. The original filter (ACMRT, #19018) gave no
# reason for the omission.
CHANGED_FILES=$(git diff --name-only --diff-filter=ACDMRT "${MERGE_BASE}..HEAD")

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
LLK_UNIT_TESTS_CHANGED=false
LLK_PERF_CHANGED=false
LLK_CI_CHANGED=false
WORKFLOWS_CHANGED=false


while IFS= read -r FILE; do
    case "$FILE" in
        CMakeLists.txt|**/CMakeLists.txt|**/*.cmake|*.cmake.in|**/*.cmake.in|CMakePresets.json)
            CMAKE_CHANGED=true
            ANY_CODE_CHANGED=true
            ;;
        tt_metal/sfpi-info.sh|tt_metal/sfpi-version)
            # Read in by a cmake file; also pins the SFPI compiler used to build LLK
            # device kernels, so any change must re-run LLK tests on all archs.
            CMAKE_CHANGED=true
            LLK_SFPI_CHANGED=true
            ANY_CODE_CHANGED=true
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
        # Shared Python test harness (helpers/ and conftest.py) — imported by ALL arch-specific
        # test suites including quasar. A break here causes quasar collection to fail even if no
        # quasar-specific file changed, so treat it as a quasar change.
        tt_metal/tt-llk/tests/python_tests/helpers/**|tt_metal/tt-llk/tests/python_tests/conftest.py)
            LLK_QUASAR_CHANGED=true
            LLK_TESTS_CHANGED=true
            ;;
        tt_metal/tt-llk/tests/python_tests/fuser/**)
            LLK_QUASAR_CHANGED=true
            LLK_TESTS_CHANGED=true
            ;;
        tt_metal/tt-llk/tests/**)
            LLK_TESTS_CHANGED=true
            ;;
        .github/workflows/llk-*.yaml|.github/workflows/build-quasar-perf.yml|.github/scripts/llk-*.sh|tests/pipeline_reorg/llk_unit_tests.yaml|tests/pipeline_reorg/llk_merge_gate_tests.yaml)
            LLK_CI_CHANGED=true
            ;;
        # `*(*/)` (zero or more directory components), not `**/`: only extglob
        # is enabled here, not globstar, so in case-pattern matching the literal
        # '/' in `tt_metal/**/*.ext` REQUIRES at least one subdirectory and
        # silently misses files sitting directly in tt_metal/. That is not
        # hypothetical -- tt_metal/hal.cpp is in TT_METAL_SOURCES
        # (tt_metal/sources.cmake), and while unmatched here a PR touching only
        # it set neither tt-metalium-changed nor any-code-changed, so the
        # artifact build itself was skipped. Same idiom as the clang-tidy scan
        # below. Note `tt_metal/*.ext` would NOT be the fix: in a case pattern
        # `*` matches '/' too, so that form silently matches every depth.
        tt_metal/*(*/)*.@(h|hpp|inl|c|cpp|cc|py))
            TTMETALIUM_CHANGED=true
            ANY_CODE_CHANGED=true
            ;;
        # LLK unit-test sources (built into the unit_tests_llk gtest binary). Mirror the
        # llk-tests-changed pattern but for the in-tree gtest unit tests rather than the
        # LLK engine submodule's pytest suite. Must come before the generic
        # tests/tt_metal/**/*.{h,hpp,c,cpp,py} catch-all so the narrower flag is set; we
        # also raise the broader TTMETALIUM_TESTS_CHANGED here so existing test gates
        # (e.g. runtime-smoke-tests) keep firing for these changes.
        tests/tt_metal/tt_metal/llk/**)
            LLK_UNIT_TESTS_CHANGED=true
            TTMETALIUM_TESTS_CHANGED=true
            ANY_CODE_CHANGED=true
            ;;
        ttnn/**/*.@(h|hpp|inl|c|cpp|py))
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
        tt-train/**/*.@(h|hpp|inl|c|cpp|py))
            TTTRAIN_CHANGED=true
            ANY_CODE_CHANGED=true
            ;;
        tools/**/*.@(h|hpp|inl|c|cpp|py))
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
        .github/workflows/build-artifact.yaml|.github/workflows/build-docker-artifact.yaml)
            BUILD_WORKFLOWS_CHANGED=true
            ANY_CODE_CHANGED=true
            ;;
        # Any other workflow change runs the standard PR gate. More specific workflow
        # patterns above (e.g. llk-*.yaml, build-artifact.yaml) match first and keep
        # their targeted behavior; this catch-all ensures a workflow-only PR never
        # silently skips CI. Fanned out to the full gate below (same as submodule).
        .github/workflows/*.yaml|.github/workflows/*.yml)
            WORKFLOWS_CHANGED=true
            ;;
    esac
done <<< "$CHANGED_FILES"

# --- run-clang-tidy inputs: raw snapshot + dedicated re-scans ---------------
# Snapshot the language-agnostic per-file flags NOW, before the blanket
# submodule/workflow "treat as everything changed" fallback below mutates
# them. run-clang-tidy must reflect only changes that can plausibly alter
# clang-tidy results; deriving it from the post-fallback values would force
# a full code-analysis rescan for every workflow-only PR, including
# standalone workflows with zero relevance to what gets built or scanned.
RAW_CMAKE_CHANGED=$CMAKE_CHANGED
RAW_CLANG_TIDY_CONFIG_CHANGED=$CLANG_TIDY_CONFIG_CHANGED
RAW_LLK_WORMHOLE_CHANGED=$LLK_WORMHOLE_CHANGED
RAW_LLK_BLACKHOLE_CHANGED=$LLK_BLACKHOLE_CHANGED
RAW_LLK_COMMON_CHANGED=$LLK_COMMON_CHANGED
RAW_LLK_SFPI_CHANGED=$LLK_SFPI_CHANGED

# clang-tidy is a C/C++-only linter, so the "did relevant source change?"
# half of run-clang-tidy gets its own dedicated scan for C/C++ extensions
# only. The shared flags above (tt-metalium-changed, tools-changed, ...) are
# deliberately NOT reused here: their patterns also match .py files (and
# e.g. tools/triage/requirements.txt) because other jobs legitimately fire
# on Python changes — but a Python-only PR cannot affect clang-tidy output.
# Each directory needs BOTH a root-level (dir/*.ext) and a nested
# (dir/**/*.ext) alternative: this script only enables extglob, not
# globstar, so in case-pattern matching `dir/**/*.ext` requires at least
# one subdirectory component and would miss files directly in `dir/`.
# tests/scale_out and tests/tt_eager are included because the clang-tidy
# CMake preset inherits TT_METAL_BUILD_TESTS=TRUE and TTNN_BUILD_TESTS=TRUE,
# and tests/CMakeLists.txt builds those trees under exactly those flags.
#
# .fbs is included alongside the C/C++ extensions: FlatBuffer schemas
# (tt_metal/impl/flatbuffer/*.fbs, tt_metal/api/.../serialized_descriptors/*.fbs,
# ttnn/core/tensor/flatbuffer/*.fbs, tt-train/.../serialization/*.fbs) generate
# *_generated.h headers via GENERATE_FBS_HEADER (cmake/flatbuffers.cmake) that
# are compiled straight into their target's sources with no SKIP_LINTING or
# HeaderFilterRegex exclusion — clang-tidy genuinely analyzes them, so an
# .fbs-only PR must not be skipped.
#
# .proto is deliberately NOT included, despite generating C++ the same way:
# GENERATE_PROTO_FILES (cmake/protobuf.cmake) sets SKIP_LINTING on the
# generated .pb.cc, drops a no-op .clang-tidy into the generated directory so
# the .pb.h is never scanned via #include either, and the repo's top-level
# .clang-tidy HeaderFilterRegex separately excludes '.pb.h$' outright. Same
# reasoning excludes .capnp: tt_metal/impl/CMakeLists.txt sets SKIP_LINTING
# on every capnp-generated RPC source/header. Neither can affect clang-tidy
# output no matter how the schema changes.
CPP_SOURCE_FOR_CLANG_TIDY_CHANGED=false
# Explicit, human-auditable list of workflow files whose changes affect the
# clang-tidy scan itself (its definition, implementation, caller/inputs, or
# the docker image it runs inside). Deliberately a literal list rather than
# any computed call-graph traversal. NOT listed on purpose:
# .github/workflows/check-harbor.yaml — it only health-checks the Harbor
# registry cache and cannot affect what gets built or scanned.
CLANG_TIDY_KEY_WORKFLOW_CHANGED=false
while IFS= read -r FILE; do
    case "$FILE" in
        @(tt_metal|ttnn|tests/tt_metal|tests/ttnn|tests/scale_out|tests/tt_eager|tt-train|tools|tt_stl)/*(*/)*.@(h|hpp|c|cpp|cc|inl|tpp|fbs))
            CPP_SOURCE_FOR_CLANG_TIDY_CHANGED=true
            ;;
        # tt_metal/llrt/hal's codegen.py/codegen.sh regenerate dev_msgs.hpp,
        # fabric_telemetry.hpp, and realtime_profiler_msgs.hpp (+ *_impl.hpp),
        # which ARE linted (no SKIP_LINTING in tt_metal/llrt/hal/CMakeLists.txt).
        # Their primary inputs (tt_metal/hw/inc/hostdev/*.h) already match the
        # pattern above, but the generator scripts themselves don't live under
        # tt_metal/**/*.h|.cpp — a change to the generator logic alone, with no
        # input header touched, would otherwise regenerate different linted
        # headers undetected.
        tt_metal/llrt/hal/codegen/codegen.py|tt_metal/llrt/hal/codegen/codegen.sh)
            CPP_SOURCE_FOR_CLANG_TIDY_CHANGED=true
            ;;
        .github/workflows/code-analysis.yaml|\
        .github/workflows/clang-tidy-reusable.yaml|\
        .github/workflows/pr-gate.yaml|\
        .github/workflows/build-docker-artifact.yaml|\
        .github/workflows/resolve-docker-pull-refs.yaml)
            CLANG_TIDY_KEY_WORKFLOW_CHANGED=true
            ;;
    esac
done <<< "$CHANGED_FILES"
# ----------------------------------------------------------------------------

# --- blaze-relevant-changed: does this touch what tt-blaze consumes? --------
# Gates the blaze-merge-gate job, which dispatches tt-blaze CI and waits on a
# BH Loudbox that tt-blaze owns. That is the one merge-gate job outside this
# repo's runner pool and time budget, so it is worth firing narrowly.
#
# Its own scan rather than new branches in the main loop above: that loop is
# one `case` per file with `;;`, so a branch added there would STEAL files
# from whichever flag currently claims them (a `tt_metal/fabric/**` branch
# above the generic tt_metal catch-all would stop fabric changes setting
# tt-metalium-changed). Same reasoning as the clang-tidy rescan above.
#
# What tt-blaze actually uses from this repo: Metalium runtime + fabric, the
# LLK engine its JIT-compiled kernels are built from, and TTNN's tensor /
# dtype / mesh-distribution layer. It does NOT use TTNN ops -- it runs its own
# kernels through ttnn.generic_op -- so ttnn/cpp/ttnn/operations/** is
# deliberately absent below and op-only PRs skip the job. Audited against the
# blaze tree: no conventional TTNN op is called anywhere in it. The one
# exception is ttnn.experimental.disaggregation, which is why that single
# experimental subtree IS listed.
# tests/**, models/**, tt-train/**, docs and examples are likewise absent:
# nothing tt-blaze links against or JIT-compiles.
BLAZE_RELEVANT_CHANGED=false
while IFS= read -r FILE; do
    case "$FILE" in
        # LLK engine + device-kernel headers: compiled into blaze kernels.
        tt_stl/**|tt_metal/tt-llk/**|tt_metal/hw/**)
            BLAZE_RELEVANT_CHANGED=true
            ;;
        # Fabric: blaze's multi-device sockets and CCL ride on it.
        tt_metal/fabric/**)
            BLAZE_RELEVANT_CHANGED=true
            ;;
        # Metalium runtime proper. tt_metal/test, programming_examples,
        # python_env and third_party are excluded by omission.
        tt_metal/@(impl|llrt|jit_build|api|detail|common|hostdevcommon|distributed|core_descriptors|soc_descriptors)/**)
            BLAZE_RELEVANT_CHANGED=true
            ;;
        # Runtime sources sitting directly in tt_metal/ with no subdirectory --
        # today just hal.cpp, a TT_METAL_SOURCES file and squarely core runtime.
        # The directory allowlist above cannot reach them, and `tt_metal/*.ext`
        # would not be the fix: `*` matches '/' in a case pattern, so that form
        # would also pull in tt_metal/test and programming_examples, which are
        # excluded on purpose. `!(*/*)` is the depth-0-only form.
        tt_metal/!(*/*).@(h|hpp|inl|c|cpp|cc|py))
            BLAZE_RELEVANT_CHANGED=true
            ;;
        # Pins the SFPI compiler that builds every device kernel.
        tt_metal/sfpi-info.sh|tt_metal/sfpi-version)
            BLAZE_RELEVANT_CHANGED=true
            ;;
        # Core TTNN: tensor, dtypes, mesh distribution, graph capture, and the
        # kernel API blaze's generic_op path goes through.
        ttnn/api/**|ttnn/core/**|ttnn/cpp/ttnn/@(kernel|kernel_lib|graph)/**)
            BLAZE_RELEVANT_CHANGED=true
            ;;
        # The Python package blaze imports (bindings, tensor helpers).
        ttnn/ttnn/**)
            BLAZE_RELEVANT_CHANGED=true
            ;;
        # The single experimental op tree blaze calls into.
        ttnn/cpp/ttnn/experimental/disaggregation/**)
            BLAZE_RELEVANT_CHANGED=true
            ;;
        # Build definition, and the workflows that define this gate and produce
        # the artifacts it hands tt-blaze. Listed explicitly so a workflow-only
        # PR does not ride the blanket fallback below into occupying a Loudbox
        # for an unrelated workflow edit.
        CMakeLists.txt|**/CMakeLists.txt|**/*.cmake|*.cmake.in|**/*.cmake.in|CMakePresets.json)
            BLAZE_RELEVANT_CHANGED=true
            ;;
        # merge-gate.yaml is here despite being the shared gate every team
        # edits: the blaze job's needs/if and the with: block passing
        # metal-run-id, metal-ref and blaze-ref all live there, so an edit can
        # break or mis-target this gate without blaze-merge-gate.yaml being
        # touched. Accepted knowingly -- that file changes rarely enough that
        # the occasional unrelated Loudbox run is cheaper than a silent miss.
        .github/workflows/blaze-merge-gate.yaml|\
        .github/workflows/merge-gate.yaml|\
        .github/workflows/build-artifact.yaml)
            BLAZE_RELEVANT_CHANGED=true
            ;;
    esac
done <<< "$CHANGED_FILES"
# ----------------------------------------------------------------------------

SUBMODULE_PATHS=$(git config --file .gitmodules --get-regexp path | awk '{print $2}')
SUBMODULE_CHANGED=false
for submodule_path in $SUBMODULE_PATHS; do
    if echo "$CHANGED_FILES" | grep -q "^$submodule_path"; then
        SUBMODULE_CHANGED=true
        break
    fi
done

# run-clang-tidy: should the code-analysis (clang-tidy) job run? Computed
# from the dedicated C/C++-only scan and the RAW pre-fallback flags above
# (plus SUBMODULE_CHANGED, which the fallback never mutates, and the explicit
# key-workflow list) — must be decided before the blanket fallback below
# forces the shared flags true. Submodules stay included to match the repo's
# conservative posture: a submodule bump can change header behavior
# clang-tidy would flag. LLK flags are included because LLK sources are
# compiled into Metalium device kernels.
RUN_CLANG_TIDY=false
if [[ "$CPP_SOURCE_FOR_CLANG_TIDY_CHANGED" = true || \
      "$RAW_CMAKE_CHANGED" = true || "$RAW_CLANG_TIDY_CONFIG_CHANGED" = true || \
      "$RAW_LLK_WORMHOLE_CHANGED" = true || "$RAW_LLK_BLACKHOLE_CHANGED" = true || \
      "$RAW_LLK_COMMON_CHANGED" = true || "$RAW_LLK_SFPI_CHANGED" = true || \
      "$SUBMODULE_CHANGED" = true || "$CLANG_TIDY_KEY_WORKFLOW_CHANGED" = true ]]; then
    RUN_CLANG_TIDY=true
fi

# A submodule bump (UMD, tt-llk, ...) changes what blaze links against and
# JIT-compiles, so it counts. Folded in HERE, before the blanket fallback
# below, for the same reason run-clang-tidy is: that fallback also forces
# every shared flag true on any workflow-only PR, which would put this job on
# a Loudbox for workflow edits that cannot affect tt-blaze. The explicit
# workflow allowlist in the scan above is the intended path for that case.
if [[ "$SUBMODULE_CHANGED" = true ]]; then
    BLAZE_RELEVANT_CHANGED=true
fi

if [[ "$SUBMODULE_CHANGED" = true || "$WORKFLOWS_CHANGED" = true ]]; then
    # Treat any submodule or workflow change as a change to everything; not going to manage dependency trees for this.
    # For workflows this guarantees a workflow-only PR runs the full standard gate (build + smoke + examples + code-analysis)
    # rather than silently skipping, matching every other PR.
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
    [run-clang-tidy]=$RUN_CLANG_TIDY
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
    [llk-unit-tests-changed]=$LLK_UNIT_TESTS_CHANGED
    [llk-perf-changed]=$LLK_PERF_CHANGED
    [llk-ci-changed]=$LLK_CI_CHANGED
    [blaze-relevant-changed]=$BLAZE_RELEVANT_CHANGED
)

for var in "${!changes[@]}"; do
    echo "$var=${changes[$var]}"
    if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
        # Output results in GitHub Actions format when run in GHA
        echo "$var=${changes[$var]}" >> "$GITHUB_OUTPUT"
    fi
done
