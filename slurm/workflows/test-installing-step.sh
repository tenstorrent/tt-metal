#!/usr/bin/env bash
#SBATCH --job-name=test-installing-step
#SBATCH --partition=wh-n150
#SBATCH --time=00:30:00
#
# Install validation: verify build artifact extraction and wheel installation
# work correctly on hardware nodes.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"
source_lib artifacts
source_lib docker
source_lib setup_job
source_lib cleanup
source_config env

require_env PIPELINE_ID

log_info "=== Install validation starting ==="
log_info "  Pipeline: ${PIPELINE_ID}"

BUILD_ARTIFACT=1 INSTALL_WHEEL=1 setup_job

IMAGE="$(resolve_image "${DOCKER_IMAGE:-}")"
docker_login
docker_pull_with_retry "${IMAGE}"

trap 'cleanup_job --exit-code $?' EXIT

docker_run "${IMAGE}" "
cd \${TT_METAL_HOME}

echo 'Verifying build artifacts...'
test -d build/lib || { echo 'FAIL: build/lib missing'; exit 1; }
test -d build/bin || { echo 'FAIL: build/bin missing'; exit 1; }
echo 'Build artifacts OK'

echo 'Verifying Python imports...'
python -c 'import ttnn; print(f\"ttnn: {ttnn.__version__}\")' || \
    { echo 'FAIL: cannot import ttnn'; exit 1; }
python -c 'import tt_lib; print(\"tt_lib: OK\")' 2>/dev/null || \
    echo 'WARN: tt_lib import failed (may be expected)'
echo 'Python imports OK'

echo 'Verifying device access...'
python -c '
import ttnn
device = ttnn.open_device(0)
print(f\"Device opened: {device}\")
ttnn.close_device(device)
print(\"Device test PASSED\")
' || { echo 'FAIL: device test failed'; exit 1; }
"

log_info "=== Install validation complete ==="
