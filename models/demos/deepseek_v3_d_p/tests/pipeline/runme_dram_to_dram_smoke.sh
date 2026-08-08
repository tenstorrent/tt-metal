#!/usr/bin/env bash
# Single-galaxy launcher for test_dram_to_dram_smoke.py.
#
# Same 2-rank-on-1-galaxy setup as runme_cross_mesh_smoke.sh — rank 0 = tray 1
# (mesh_id=0, 2x4 submesh), rank 1 = tray 2 (mesh_id=1, 2x4 submesh). Intermesh
# fabric over the inter-tray backplane.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TT_METAL_ROOT="$(cd "${SCRIPT_DIR}/../../../../../" && pwd)"

cd "${TT_METAL_ROOT}"

export TT_METAL_HOME="${TT_METAL_HOME:-${TT_METAL_ROOT}}"
export TT_METAL_SLOW_DISPATCH_MODE=1
export PYTHONPATH="${TT_METAL_ROOT}:${PYTHONPATH:-}"

RANK_BINDING="${TT_METAL_ROOT}/models/demos/deepseek_v3_d_p/tests/pipeline/dual_tray_2x4_rank_bindings.yaml"
TEST_FILE="${TT_METAL_ROOT}/models/demos/deepseek_v3_d_p/tests/pipeline/test_dram_to_dram_smoke.py"

for f in "${RANK_BINDING}" "${TEST_FILE}"; do
    [[ -f "$f" ]] || { echo "ERROR: missing file: $f" >&2; exit 1; }
done

echo "=== DRAM→DRAM cross-mesh smoke (1 galaxy, 2x4 submeshes) ==="
echo "  TT_METAL_HOME = ${TT_METAL_HOME}"
echo "  rank-binding  = ${RANK_BINDING}"
echo "  test          = ${TEST_FILE}"
echo

tt-run \
    --bare \
    --rank-binding "${RANK_BINDING}" \
    --mpi-args "--oversubscribe" \
    python -m pytest "${TEST_FILE}" -svv --no-header -k 1galaxy
