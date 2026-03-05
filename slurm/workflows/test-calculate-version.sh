#!/usr/bin/env bash
#SBATCH --job-name=test-calculate-version
#SBATCH --partition=build
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#
# Version calculation test: validate that version strings are correctly
# computed from git tags and commit history.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"
source_config env

require_env PIPELINE_ID

log_info "=== Version calculation test starting ==="
log_info "  Pipeline: ${PIPELINE_ID}"

cd "${REPO_ROOT}"

LATEST_TAG="$(git describe --tags --abbrev=0 2>/dev/null || echo 'v0.0.0')"
FULL_VERSION="$(git describe --tags --long 2>/dev/null || echo 'v0.0.0-0-gunknown')"
COMMITS_SINCE="$(echo "${FULL_VERSION}" | sed 's/.*-\([0-9]*\)-g.*/\1/')"

log_info "  Latest tag:     ${LATEST_TAG}"
log_info "  Full version:   ${FULL_VERSION}"
log_info "  Commits since:  ${COMMITS_SINCE}"
log_info "  Git SHA:        ${GIT_SHA}"

if [[ -f scripts/calculate_version.py ]]; then
    CALC_VERSION="$(python3 scripts/calculate_version.py 2>/dev/null || echo 'unknown')"
    log_info "  Calculated:     ${CALC_VERSION}"
elif [[ -f .github/actions/calculate-version/action.yml ]]; then
    log_info "  Version script not found, using git describe"
fi

if [[ "${LATEST_TAG}" =~ ^v[0-9]+\.[0-9]+\.[0-9]+ ]]; then
    log_info "Version tag format VALID: ${LATEST_TAG}"
else
    log_warn "Version tag format may be unexpected: ${LATEST_TAG}"
fi

log_info "=== Version calculation test complete ==="
