#!/usr/bin/env bash
#SBATCH --job-name=code-analysis
#SBATCH --partition=build
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=/weka/ci/logs/%x/%j/%a.log
#SBATCH --error=/weka/ci/logs/%x/%j/%a.err
#
# GHA source: .github/workflows/code-analysis.yaml
# Runs clang-tidy (full or incremental) and optionally the Clang Static
# Analyzer via CodeChecker. No device access required — runs on build partition.
#
# Environment overrides:
#   DO_FULL_SCAN          - Set to "true" for full scan (default: auto-detect)
#   MERGE_BASE            - Git merge-base for incremental scans
#   SCAN_TARGETS          - Space-separated CMake targets to scan (default: all)
#   PREREQ_TARGETS        - Targets to build before scanning
#   ENABLE_STATIC_ANALYSIS - Set to "true" to enable CodeChecker CSA
#   DEBUG_FILE_FILTER     - Analyze only specific files (e.g. "path/to/file.cpp")
#   PLATFORM              - Platform string (default: "Ubuntu 24.04")

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
SCAN_TARGETS="${SCAN_TARGETS:-}"
PREREQ_TARGETS="${PREREQ_TARGETS:-}"
ENABLE_STATIC_ANALYSIS="${ENABLE_STATIC_ANALYSIS:-false}"
DEBUG_FILE_FILTER="${DEBUG_FILE_FILTER:-}"

log_info "Running code analysis (full_scan=${DO_FULL_SCAN}, static_analysis=${ENABLE_STATIC_ANALYSIS})"

# --- Clang-tidy ---
log_info "Running clang-tidy analysis"

TIDY_CMD="mkdir -p generated/test_reports"

if [[ "${DO_FULL_SCAN}" == "true" ]]; then
    TIDY_CMD="${TIDY_CMD} && \
        cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON && \
        cmake --build build --target all_generated_files && \
        run-clang-tidy -p build 2>&1 | tee generated/test_reports/clang-tidy-full.log"
else
    # Build prerequisites if specified
    if [[ -n "${PREREQ_TARGETS}" ]]; then
        TIDY_CMD="${TIDY_CMD} && \
            cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON && \
            cmake --build build --target ${PREREQ_TARGETS}"
    fi

    # Incremental: only scan changed files
    TIDY_CMD="${TIDY_CMD} && \
        cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON && \
        cmake --build build --target all_generated_files"

    if [[ -n "${SCAN_TARGETS}" ]]; then
        TIDY_CMD="${TIDY_CMD} && \
            run-clang-tidy -p build '${SCAN_TARGETS}' 2>&1 | tee generated/test_reports/clang-tidy.log"
    elif [[ -n "${MERGE_BASE}" ]]; then
        TIDY_CMD="${TIDY_CMD} && \
            git diff --name-only ${MERGE_BASE} HEAD -- '*.cpp' '*.hpp' '*.h' | \
            xargs -r run-clang-tidy -p build 2>&1 | tee generated/test_reports/clang-tidy.log"
    else
        TIDY_CMD="${TIDY_CMD} && \
            run-clang-tidy -p build 2>&1 | tee generated/test_reports/clang-tidy.log"
    fi
fi

docker_run "$DOCKER_IMAGE" "${TIDY_CMD}"

# --- Clang Static Analyzer (CodeChecker) ---
if [[ "${ENABLE_STATIC_ANALYSIS}" == "true" ]]; then
    log_info "Running Clang Static Analyzer via CodeChecker"

    CSA_CMD="\
        cmake --preset clang-static-analyzer && \
        cmake --build .build/clang-static-analyzer --target all_generated_files && \
        update-alternatives --install /usr/bin/clang clang /usr/bin/clang-20 100 && \
        update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-20 100 && \
        uv pip install codechecker==6.27.1 && \
        mkdir -p .build/codechecker-reports"

    ANALYZE_ARGS="CodeChecker analyze \
        .build/clang-static-analyzer/compile_commands.json \
        --output .build/codechecker-reports \
        --jobs \$(nproc) \
        --config .codechecker.json"

    if [[ -n "${DEBUG_FILE_FILTER}" ]]; then
        ANALYZE_ARGS="${ANALYZE_ARGS} --file '${DEBUG_FILE_FILTER}'"
    fi

    CSA_CMD="${CSA_CMD} && ${ANALYZE_ARGS} || echo 'Analysis completed with warnings'"

    # Generate JSON + HTML reports
    CSA_CMD="${CSA_CMD} && \
        CodeChecker parse .build/codechecker-reports --export json --output .build/codechecker_issues.json || true && \
        mkdir -p .build/codechecker_html && \
        CodeChecker parse .build/codechecker-reports --export html --output .build/codechecker_html || true"

    docker_run "$DOCKER_IMAGE" "${CSA_CMD}"
fi

log_info "Code analysis complete"
