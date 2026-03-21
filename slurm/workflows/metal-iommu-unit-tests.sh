#!/usr/bin/env bash
#SBATCH --job-name=metal-iommu-unit-tests
#SBATCH --partition=wh-n150
#SBATCH --constraint=viommu
#SBATCH --time=01:00:00
#
# GHA source: IOMMU-specific tests from various workflows
# Requires viommu-capable nodes (Slurm constraint=viommu).
# Runs both gtest IOMMU tests and pytest IOMMU tests.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"
source "${SCRIPT_DIR}/lib/docker.sh"
source "${SCRIPT_DIR}/lib/artifacts.sh"
source "${SCRIPT_DIR}/lib/setup_job.sh"
source "${SCRIPT_DIR}/lib/cleanup.sh"
source "${SCRIPT_DIR}/workflows/_helpers/resolve_docker_image.sh"

parse_common_args "$@"
resolve_workflow_docker_image dev

export BUILD_ARTIFACT=1
setup_job
trap 'cleanup_job $?' EXIT

log_info "Running IOMMU unit tests (viommu constraint)"

export DOCKER_EXTRA_ENV="GTEST_OUTPUT=xml:generated/test_reports/"

docker_run "$DOCKER_IMAGE" "\
    mkdir -p generated/test_reports && \
    ./build/test/tt_metal/unit_tests --gtest_filter='*IOMMU*' \
        --gtest_output=xml:generated/test_reports/iommu_unit_tests.xml && \
    pytest tests/tt_metal/test_iommu.py \
        -x --timeout=600 \
        --junit-xml=generated/test_reports/iommu_pytest.xml
"
