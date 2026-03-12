#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
# Parallel wrapper for GitHub Actions Security Linting
# Runs per-file checks in parallel batches, then aggregate checks once.
#
# Usage: ./check-actions-security-parallel.sh [OPTIONS] [FILE...]
#   -h, --help    Show help message
#   -j N          Number of parallel jobs (default: 0 = all available CPUs)
#   --strict      Exit with error code if any issues found
#
# If no FILEs are provided, scans all .yml/.yaml files in .github/workflows and .github/actions.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
GITHUB_DIR="${REPO_ROOT}/.github"
MAIN_SCRIPT="${SCRIPT_DIR}/check-actions-security.sh"

PARALLEL_JOBS=0
STRICT_MODE=false
FILES=()

usage() {
    cat <<'EOF'
Usage: ./check-actions-security-parallel.sh [OPTIONS] [FILE...]

Options:
  -h, --help    Show this help message and exit
  -j N          Number of parallel jobs (default: 0 = all available CPUs)
  --strict      Exit with error code if any issues found

This wrapper runs security checks in parallel for better performance.
Per-file checks run in background batches, then aggregate checks run once.

If no FILEs are provided, scans all .yml/.yaml files in .github/workflows and .github/actions.

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
        *)
            FILES+=("$1")
            shift
            ;;
    esac
done

# Create temp files for results, clean up on exit
RESULTS_FILE=$(mktemp)
RESULTS_DIR=$(mktemp -d)
trap 'rm -f "${RESULTS_FILE}"; rm -rf "${RESULTS_DIR}"' EXIT

# Default to finding all files if none provided
if [[ ${#FILES[@]} -eq 0 ]]; then
    while IFS= LC_ALL=C read -r -d '' f; do
        FILES+=("${f}")
    done < <(find "${GITHUB_DIR}/workflows" "${GITHUB_DIR}/actions" \
        \( -name "*.yml" -o -name "*.yaml" \) -print0 2>/dev/null || true)
fi

if [[ "${PARALLEL_JOBS}" -eq 0 ]]; then
    if command -v getconf >/dev/null 2>&1; then
        PARALLEL_JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)"
    else
        PARALLEL_JOBS=1
    fi
    [[ "${PARALLEL_JOBS}" =~ ^[0-9]+$ ]] || PARALLEL_JOBS=1
    [[ "${PARALLEL_JOBS}" -gt 0 ]] || PARALLEL_JOBS=1
fi

# Exit early if no files to check
if [[ ${#FILES[@]} -eq 0 ]]; then
    printf '%s\n' "No workflow files found to check."
    printf '\n'
    exit 0
fi

worker_pids=()
worker_outputs=()

flush_workers() {
    local pid
    local output_file

    for pid in "${worker_pids[@]}"; do
        wait "${pid}" || true
    done

    for output_file in "${worker_outputs[@]}"; do
        [[ -f "${output_file}" ]] || continue
        cat "${output_file}" >> "${RESULTS_FILE}"
    done

    worker_pids=()
    worker_outputs=()
}

launch_batch() {
    local output_file="${RESULTS_DIR}/worker-${#worker_outputs[@]}-${RANDOM}.txt"
    "${MAIN_SCRIPT}" --skip-aggregate --machine-output "$@" > "${output_file}" 2>&1 || true &
    worker_pids+=("$!")
    worker_outputs+=("${output_file}")
}

# Phase 1: Run per-file checks in parallel (skip aggregates 5,6)
# Batch 5 files per invocation to reduce shell startup overhead.
batch_files=()
for file in "${FILES[@]}"; do
    batch_files+=("${file}")

    if [[ ${#batch_files[@]} -eq 5 ]]; then
        launch_batch "${batch_files[@]}"
        batch_files=()

        if [[ ${#worker_pids[@]} -ge ${PARALLEL_JOBS} ]]; then
            flush_workers
        fi
    fi
done

if [[ ${#batch_files[@]} -gt 0 ]]; then
    launch_batch "${batch_files[@]}"
fi

flush_workers

# Phase 2: Run aggregate checks once with all files
"${MAIN_SCRIPT}" -c 5,6 --machine-output "${FILES[@]}" \
    >> "${RESULTS_FILE}" 2>&1 || true

# Phase 3: Format and display results with deduplicated examples
format_args=("--format-results" "${RESULTS_FILE}")
if [[ "${STRICT_MODE}" == "true" ]]; then
    format_args+=("--strict")
fi

exec "${MAIN_SCRIPT}" "${format_args[@]}"
