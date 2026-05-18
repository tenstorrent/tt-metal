#!/usr/bin/env bash
# Launcher for test_cross_mesh_socket_smoke.py.
#
# Runs 2 MPI ranks on this single SLURM-allocated host, each on its own
# 2x4 submesh slice of one Blackhole galaxy, with cross-mesh fabric between
# them. The rank-binding YAML pins:
#   rank 0 -> mesh_id=0, chips 4,7,10,14,15,18,25,30
#   rank 1 -> mesh_id=1, chips 0,2,3,6,8,21,28,31
# MGD: bh_galaxy_dual_2x4_intermesh.textproto.
#
# Use --bare so tt-run doesn't force its `--mca btl self,tcp ...` defaults
# (those broke intra-host comms in earlier runs).
# Use --oversubscribe so OpenMPI ignores SLURM_TASKS_PER_NODE=1 and packs
# both ranks onto the one allocated host.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TT_METAL_ROOT="$(cd "${SCRIPT_DIR}/../../../../../" && pwd)"

cd "${TT_METAL_ROOT}"

export TT_METAL_HOME="${TT_METAL_HOME:-${TT_METAL_ROOT}}"
export TT_METAL_SLOW_DISPATCH_MODE=1
export PYTHONPATH="${TT_METAL_ROOT}:${PYTHONPATH:-}"

RANK_BINDING="${TT_METAL_ROOT}/models/demos/deepseek_v3_d_p/tests/pipeline/dual_tray_2x4_rank_bindings.yaml"
TEST_FILE="${TT_METAL_ROOT}/models/demos/deepseek_v3_d_p/tests/pipeline/test_cross_mesh_socket_smoke.py"

for f in "${RANK_BINDING}" "${TEST_FILE}"; do
    [[ -f "$f" ]] || { echo "ERROR: missing file: $f" >&2; exit 1; }
done

echo "=== Cross-mesh socket smoke ==="
echo "  TT_METAL_HOME = ${TT_METAL_HOME}"
echo "  rank-binding  = ${RANK_BINDING}"
echo "  test          = ${TEST_FILE}"
echo

tt-run \
    --bare \
    --rank-binding "${RANK_BINDING}" \
    --mpi-args "--oversubscribe" \
    python -m pytest "${TEST_FILE}" -svv --no-header -k 1galaxy
