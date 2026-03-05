#!/usr/bin/env bash
#SBATCH --job-name=release-cleanup
#SBATCH --partition=build
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=/weka/ci/logs/%x/%j/%a.log
#SBATCH --error=/weka/ci/logs/%x/%j/%a.err
#
# Clean up release artifacts and stale pipeline data.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"
source_lib artifacts
source_lib docker
source_config env

require_env PIPELINE_ID

STALE_DAYS="${STALE_ARTIFACT_DAYS:-7}"

log_info "=== Release cleanup starting ==="
log_info "  Pipeline: ${PIPELINE_ID}"
log_info "  Stale threshold: ${STALE_DAYS} days"

cleanup_artifacts "${PIPELINE_ID}"

log_info "Cleaning up stale artifacts older than ${STALE_DAYS} days"
cleanup_old_artifacts "${STALE_DAYS}"

docker_cleanup

log_info "=== Release cleanup complete ==="
