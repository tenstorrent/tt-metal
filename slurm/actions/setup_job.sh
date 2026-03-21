#!/usr/bin/env bash
# setup_job.sh - CLI wrapper around lib/setup_job.sh setup_job()
# Usage: setup_job.sh [OPTIONS]
#
# Equivalent to .github/actions/setup-job/action.yml

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"
source_lib setup_job

# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Perform common job setup: download artifacts, install wheel, configure env.

Options:
  --build-artifact NAME   Fetch and extract the named build artifact
  --wheel-artifact NAME   Fetch and install the named Python wheel artifact
  --enable-watcher        Enable TT Metal watcher (TT_METAL_WATCHER=1)
  --enable-asserts        Enable lightweight kernel asserts
  --enable-llk-asserts    Enable LLK asserts (TT_METAL_LLK_ASSERTS=1)
  --enable-ccache         Enable kernel ccache (requires CCACHE_REMOTE_STORAGE)
  --workspace PATH        Working directory (default: /work)
  -h, --help              Show this help message

Environment:
  PIPELINE_ID             Pipeline identifier (auto-generated if unset)
  CCACHE_REMOTE_STORAGE   Redis URL for kernel ccache (required with --enable-ccache)
EOF
    exit "${1:-0}"
}

# ---------------------------------------------------------------------------
# Argument parsing — translate CLI flags to env vars consumed by setup_job()
# ---------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-artifact)     export BUILD_ARTIFACT=1; export BUILD_ARTIFACT_NAME="$2"; shift 2 ;;
        --wheel-artifact)     export INSTALL_WHEEL=1;  export WHEEL_ARTIFACT_NAME="$2"; shift 2 ;;
        --enable-watcher)     export ENABLE_WATCHER=1;         shift ;;
        --enable-asserts)     export ENABLE_LIGHTWEIGHT_ASSERTS=1; shift ;;
        --enable-llk-asserts) export ENABLE_LLK_ASSERTS=1;    shift ;;
        --enable-ccache)      export ENABLE_KERNEL_CCACHE=1;    shift ;;
        --workspace)          export JOB_WORKSPACE="$2";        shift 2 ;;
        -h|--help)            usage 0 ;;
        *)                    log_error "Unknown option: $1"; usage 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

setup_job
