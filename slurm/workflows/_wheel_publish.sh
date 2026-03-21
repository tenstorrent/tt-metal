#!/usr/bin/env bash
#SBATCH --job-name=wheels-publish
#SBATCH --partition=build
#SBATCH --time=00:30:00
#
# Publish tested wheels to internal PyPI / artifact store.
#
# Submitted by build-and-test-wheels.sh with --dependency=afterok on the
# wheel test job — only runs after tests pass.
#
# Environment:
#   PIPELINE_ID       (required)
#   PYTHON_VERSION    Python version matching the built wheel (default: 3.10)
#   PYPI_REPOSITORY   Internal PyPI URL (default: from env.sh)
#   PYPI_TOKEN        Authentication token for PyPI upload

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"
source_lib artifacts
source_config env

require_env PIPELINE_ID

PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
PYPI_REPOSITORY="${PYPI_REPOSITORY:-${INTERNAL_PYPI_URL:-}}"

log_info "=== Wheel publish starting ==="
log_info "  Pipeline: ${PIPELINE_ID}"
log_info "  Python:   ${PYTHON_VERSION}"

WHEEL_DIR="${ARTIFACT_BASE}/${PIPELINE_ID}/build/tt_metal_wheels"

if [[ ! -d "${WHEEL_DIR}" ]]; then
    log_error "Wheel directory not found: ${WHEEL_DIR}"
    exit 1
fi

WHEEL_COUNT=$(find "${WHEEL_DIR}" -name "*.whl" | wc -l)
if [[ "${WHEEL_COUNT}" -eq 0 ]]; then
    log_error "No .whl files found in ${WHEEL_DIR}"
    exit 1
fi

log_info "Found ${WHEEL_COUNT} wheel(s) to publish"

if [[ -z "${PYPI_REPOSITORY}" ]]; then
    log_warn "No PYPI_REPOSITORY configured — copying wheels to release staging"
    RELEASE_DIR="${ARTIFACT_BASE}/${PIPELINE_ID}/release/wheels"
    mkdir -p "${RELEASE_DIR}"
    cp "${WHEEL_DIR}"/*.whl "${RELEASE_DIR}/"
    log_info "Wheels staged to ${RELEASE_DIR}"
else
    for whl in "${WHEEL_DIR}"/*.whl; do
        [[ -f "${whl}" ]] || continue
        log_info "Publishing: $(basename "${whl}")"
        twine upload \
            --repository-url "${PYPI_REPOSITORY}" \
            ${PYPI_TOKEN:+--username __token__ --password "${PYPI_TOKEN}"} \
            --non-interactive \
            "${whl}"
    done
    log_info "All wheels published to ${PYPI_REPOSITORY}"
fi

log_info "=== Wheel publish complete ==="
