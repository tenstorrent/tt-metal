#!/usr/bin/env bash
# download_artifact.sh - Fetch artifacts from Weka shared storage.
# Equivalent to .github/actions/download-artifact-with-retry/action.yml
#
# Usage: download_artifact.sh --name NAME [--dest DIR] [--max-retries N]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"
source_lib artifacts
source_lib retry

ARTIFACT_NAME=""
DEST_DIR="."
MAX_RETRIES=3

while [[ $# -gt 0 ]]; do
    case "$1" in
        --name)        ARTIFACT_NAME="$2"; shift 2 ;;
        --dest)        DEST_DIR="$2"; shift 2 ;;
        --max-retries) MAX_RETRIES="$2"; shift 2 ;;
        *)             log_warn "Unknown option: $1"; shift ;;
    esac
done

require_env PIPELINE_ID

if [[ -z "${ARTIFACT_NAME}" ]]; then
    log_fatal "Artifact name required (--name)"
fi

ARTIFACT_DIR="$(get_artifact_dir "${PIPELINE_ID}")/${ARTIFACT_NAME}"

fetch_with_retry() {
    if [[ ! -d "${ARTIFACT_DIR}" ]]; then
        log_error "Artifact directory not found: ${ARTIFACT_DIR}"
        return 1
    fi

    mkdir -p "${DEST_DIR}"
    cp -r "${ARTIFACT_DIR}"/. "${DEST_DIR}/"

    # Integrity check for .deb files
    for deb in "${DEST_DIR}"/*.deb; do
        [[ -f "${deb}" ]] || continue
        if ! dpkg-deb --info "${deb}" >/dev/null 2>&1; then
            log_error "Integrity check failed for: ${deb}"
            return 1
        fi
        log_info "Integrity check passed: $(basename "${deb}")"
    done

    return 0
}

retry_command "fetch_with_retry" 120 5 "${MAX_RETRIES}"

log_info "Artifact '${ARTIFACT_NAME}' downloaded to ${DEST_DIR}"
