#!/usr/bin/env bash
#SBATCH --job-name=release-verify-tag
#SBATCH --partition=build
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=/weka/ci/logs/%x/%j/%a.log
#SBATCH --error=/weka/ci/logs/%x/%j/%a.err
#
# Verify or create a git tag for the release.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"
source_config env

require_env PIPELINE_ID
require_env RELEASE_VERSION

TAG_NAME="v${RELEASE_VERSION}"

log_info "=== Release tag verification starting ==="
log_info "  Version: ${RELEASE_VERSION}"
log_info "  Tag:     ${TAG_NAME}"
log_info "  SHA:     ${GIT_SHA}"

cd "${REPO_ROOT}"

if git rev-parse "${TAG_NAME}" >/dev/null 2>&1; then
    EXISTING_SHA="$(git rev-parse "${TAG_NAME}^{commit}")"
    if [[ "${EXISTING_SHA}" == "${GIT_SHA}" ]]; then
        log_info "Tag ${TAG_NAME} already exists and points to ${GIT_SHA}"
    else
        log_fatal "Tag ${TAG_NAME} exists but points to ${EXISTING_SHA}, expected ${GIT_SHA}"
    fi
else
    log_info "Tag ${TAG_NAME} does not exist"

    if [[ "${CREATE_TAG:-0}" == "1" ]]; then
        log_info "Creating tag ${TAG_NAME} at ${GIT_SHA}"
        git tag -a "${TAG_NAME}" "${GIT_SHA}" \
            -m "Release ${RELEASE_VERSION} (pipeline ${PIPELINE_ID})"
        git push origin "${TAG_NAME}"
        log_info "Tag ${TAG_NAME} created and pushed"
    else
        log_info "CREATE_TAG not set; skipping tag creation (set CREATE_TAG=1 to create)"
    fi
fi

log_info "=== Release tag verification complete ==="
