#!/bin/bash
# Run the single-host all-reduce test independently on 4 galaxies via tt-run + SLURM.
#
# Usage (from a SLURM allocation with 4 galaxy nodes):
#   bash run_4galaxy.sh                                              # multi-galaxy all-reduce (default)
#   TEST_SCRIPT=test_single_host_all_reduce.py bash run_4galaxy.sh --fabric-config 2D  # single-host test on all galaxies
#   bash run_4galaxy.sh --per-chip-shape 1 1 64 2048                # custom tensor shape

#SBATCH --job-name=all_reduce_4galaxy
#SBATCH --output=all_reduce_4galaxy_%j.out
#SBATCH --error=all_reduce_4galaxy_%j.err
#SBATCH --nodes=4

set -euo pipefail

if [ -z "${TT_METAL_HOME:-}" ]; then
    TT_METAL_HOME="/data/${USER}/tt-metal"
fi
export TT_METAL_HOME
export PYTHONPATH="${TT_METAL_HOME}"
source "${TT_METAL_HOME}/python_env/bin/activate"
export LD_LIBRARY_PATH="/opt/openmpi-v5.0.7-ulfm/lib:${LD_LIBRARY_PATH:-}"
export TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS=120000

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Reset all galaxies before running
srun tt-smi -glx_reset

# Host order determines rank assignment: rank 0 = first host, rank 1 = second, etc.
# This must match the rank→mesh_id mapping in rank_bindings.yaml and the physical cabling.
HOSTS="bh-glx-c01u08,bh-glx-c01u02,bh-glx-c02u02,bh-glx-c02u08"

PYTHON_SCRIPT="${SCRIPT_DIR}/${TEST_SCRIPT:-test_multi_galaxy_all_reduce.py}"

tt-run --verbose \
    --mpi-args "--host ${HOSTS} --map-by ppr:1:node" \
    --rank-binding "${SCRIPT_DIR}/configs/rank_bindings.yaml" \
    python "${PYTHON_SCRIPT}" "$@"
