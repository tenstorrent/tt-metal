#!/usr/bin/env bash
#SBATCH --job-name=clang-tidy-reusable
#SBATCH --partition=build
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=/weka/ci/logs/%x/%j/%a.log
#SBATCH --error=/weka/ci/logs/%x/%j/%a.err
#
# GHA source: .github/workflows/clang-tidy-reusable.yaml
# Reusable clang-tidy analysis. Supports full and incremental scans with
# configurable targets. Called by code-analysis.sh or directly.
#
# Environment overrides:
#   DO_FULL_SCAN    - "true" for full scan, "false" for incremental (default: true)
#   MERGE_BASE      - Git commit SHA for incremental diff base
#   PREREQ_TARGETS  - Space-separated CMake targets to build first
#   SCAN_TARGETS    - Space-separated source directories/targets to analyze
#   PLATFORM        - Build platform (default: "Ubuntu 24.04")

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"
source "${SCRIPT_DIR}/lib/docker.sh"
source "${SCRIPT_DIR}/lib/setup_job.sh"
source "${SCRIPT_DIR}/lib/cleanup.sh"
source "${SCRIPT_DIR}/workflows/_helpers/resolve_docker_image.sh"

parse_common_args "$@"
resolve_workflow_docker_image ci-build

setup_job
trap 'cleanup_job $?' EXIT

DO_FULL_SCAN="${DO_FULL_SCAN:-true}"
MERGE_BASE="${MERGE_BASE:-}"
PREREQ_TARGETS="${PREREQ_TARGETS:-}"
SCAN_TARGETS="${SCAN_TARGETS:-}"

log_info "Running clang-tidy (full=${DO_FULL_SCAN}, prereqs='${PREREQ_TARGETS}', targets='${SCAN_TARGETS}')"

# Configure + generate compilation database
BUILD_CMD="cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON && \
    cmake --build build --target all_generated_files"

# Build prerequisite targets if any
if [[ -n "${PREREQ_TARGETS}" ]]; then
    BUILD_CMD="${BUILD_CMD} && cmake --build build --target ${PREREQ_TARGETS}"
fi

if [[ "${DO_FULL_SCAN}" == "true" ]]; then
    # Full: analyze everything
    TIDY_CMD="${BUILD_CMD} && \
        mkdir -p generated/test_reports && \
        run-clang-tidy -p build 2>&1 | tee generated/test_reports/clang-tidy.log"
elif [[ -n "${SCAN_TARGETS}" ]]; then
    # Incremental with explicit targets (e.g. "tt_metal", "ttnn")
    TIDY_CMD="${BUILD_CMD} && \
        mkdir -p generated/test_reports && \
        run-clang-tidy -p build \
            -header-filter='(${SCAN_TARGETS// /|})/' \
            '(${SCAN_TARGETS// /|})/' \
            2>&1 | tee generated/test_reports/clang-tidy.log"
elif [[ -n "${MERGE_BASE}" ]]; then
    # Incremental with diff against merge-base
    TIDY_CMD="${BUILD_CMD} && \
        mkdir -p generated/test_reports && \
        CHANGED_FILES=\$(git diff --name-only ${MERGE_BASE} HEAD -- '*.cpp' '*.hpp' '*.h' '*.cc') && \
        if [ -n \"\$CHANGED_FILES\" ]; then \
            echo \"\$CHANGED_FILES\" | xargs run-clang-tidy -p build 2>&1 | tee generated/test_reports/clang-tidy.log; \
        else \
            echo 'No C++ files changed, skipping clang-tidy'; \
        fi"
else
    # Fallback: full scan
    TIDY_CMD="${BUILD_CMD} && \
        mkdir -p generated/test_reports && \
        run-clang-tidy -p build 2>&1 | tee generated/test_reports/clang-tidy.log"
fi

docker_run "$DOCKER_IMAGE" "${TIDY_CMD}"

log_info "Clang-tidy analysis complete"
