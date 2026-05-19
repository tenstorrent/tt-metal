#!/usr/bin/env bash
# 2-galaxy launcher for test_dram_to_dram_smoke.py.
#
# rank 0 = galaxy A (full 4x8 mesh, mesh_id=0), rank 1 = galaxy B (4x8, mesh_id=1).
# Inter-galaxy fabric over the inter-host network (TT_TCP_INTERFACE).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TT_METAL_ROOT="$(cd "${SCRIPT_DIR}/../../../../../" && pwd)"

cd "${TT_METAL_ROOT}"

export TT_METAL_HOME="${TT_METAL_HOME:-${TT_METAL_ROOT}}"
export TT_METAL_SLOW_DISPATCH_MODE=1
export PYTHONPATH="${TT_METAL_ROOT}:${PYTHONPATH:-}"

: "${HOST_A:?Set HOST_A to the hostname of galaxy A}"
: "${HOST_B:?Set HOST_B to the hostname of galaxy B}"
TCP_INTERFACE="${TT_TCP_INTERFACE:-ens5f0np0}"
K_FILTER="${K_FILTER:-2galaxy}"

HOSTSP="${HOST_A}:1,${HOST_B}:1"

RANK_BINDING="${TT_METAL_ROOT}/models/demos/deepseek_v3_d_p/tests/pipeline/dual_galaxy_rank_bindings.yaml"
TEST_FILE="${TT_METAL_ROOT}/models/demos/deepseek_v3_d_p/tests/pipeline/test_dram_to_dram_smoke.py"

for f in "${RANK_BINDING}" "${TEST_FILE}"; do
    [[ -f "$f" ]] || { echo "ERROR: missing file: $f" >&2; exit 1; }
done

echo "=== DRAM→DRAM cross-mesh smoke (2 galaxies) ==="
echo "  HOST_A         = ${HOST_A}"
echo "  HOST_B         = ${HOST_B}"
echo "  TCP_INTERFACE  = ${TCP_INTERFACE}"
echo "  rank-binding   = ${RANK_BINDING}"
echo "  test           = ${TEST_FILE}"
echo

# Forward TT_METAL_DPRINT_CORES to the remote rank via `mpirun -x VAR` so kernel
# DPRINTs from both ranks reach this host's stdout.
DPRINT_FWD=""
if [[ -n "${TT_METAL_DPRINT_CORES:-}" ]]; then
    DPRINT_FWD="-x TT_METAL_DPRINT_CORES"
fi

tt-run \
    --tcp-interface "${TCP_INTERFACE}" \
    --rank-binding "${RANK_BINDING}" \
    --mpi-args "--host ${HOSTSP} --tag-output --allow-run-as-root --mca btl tcp,self --mca btl_tcp_if_include ${TCP_INTERFACE} ${DPRINT_FWD}" \
    python -m pytest "${TEST_FILE}" -svv --no-header -k "${K_FILTER}"
