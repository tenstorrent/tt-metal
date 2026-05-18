#!/usr/bin/env bash
# Launcher for test_cross_mesh_socket_smoke.py on 2 galaxies (2 hosts).
#
# Same Python test as the single-galaxy variant — only the rank-binding YAML
# and the host distribution change. Each rank takes a full galaxy as its
# (4, 8) mesh; intermesh fabric runs over the inter-host network.
#
# Before running, set:
#   HOST_A         hostname of galaxy A (gets rank 0, mesh_id=0)
#   HOST_B         hostname of galaxy B (gets rank 1, mesh_id=1)
#   TCP_INTERFACE  NIC name carrying inter-host fabric traffic (e.g. ens5f0np0)
#
# IMPORTANT:
#   - Both hosts must have an up-to-date tt-metal build with the
#     MEM_ERISC_KERNEL_CONFIG_SIZE bump (commit jjovicic/socket-experiment).
#   - The test parametrize includes (4, 8); make sure the version of
#     test_cross_mesh_socket_smoke.py on both hosts has that entry.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TT_METAL_ROOT="$(cd "${SCRIPT_DIR}/../../../../../" && pwd)"

cd "${TT_METAL_ROOT}"

export TT_METAL_HOME="${TT_METAL_HOME:-${TT_METAL_ROOT}}"
export TT_METAL_SLOW_DISPATCH_MODE=1
export PYTHONPATH="${TT_METAL_ROOT}:${PYTHONPATH:-}"

# >>> Fill these in for your cluster <<<
: "${HOST_A:?Set HOST_A to the hostname of galaxy A}"
: "${HOST_B:?Set HOST_B to the hostname of galaxy B}"
TCP_INTERFACE="${TT_TCP_INTERFACE:-ens5f0np0}"

HOSTSP="${HOST_A}:1,${HOST_B}:1"

RANK_BINDING="${TT_METAL_ROOT}/models/demos/deepseek_v3_d_p/tests/pipeline/dual_galaxy_rank_bindings.yaml"
TEST_FILE="${TT_METAL_ROOT}/models/demos/deepseek_v3_d_p/tests/pipeline/test_cross_mesh_socket_smoke.py"

for f in "${RANK_BINDING}" "${TEST_FILE}"; do
    [[ -f "$f" ]] || { echo "ERROR: missing file: $f" >&2; exit 1; }
done

echo "=== Cross-mesh socket smoke (2 galaxies) ==="
echo "  HOST_A         = ${HOST_A}"
echo "  HOST_B         = ${HOST_B}"
echo "  TCP_INTERFACE  = ${TCP_INTERFACE}"
echo "  rank-binding   = ${RANK_BINDING}"
echo "  test           = ${TEST_FILE}"
echo

tt-run \
    --tcp-interface "${TCP_INTERFACE}" \
    --rank-binding "${RANK_BINDING}" \
    --mpi-args "--host ${HOSTSP} --tag-output --allow-run-as-root --mca btl tcp,self --mca btl_tcp_if_include ${TCP_INTERFACE}" \
    python -m pytest "${TEST_FILE}" -svv --no-header -k 2galaxy
