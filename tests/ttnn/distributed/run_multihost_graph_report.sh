#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Launch the multi-host TTNN graph-report test on 2 Blackhole "loudbox" hosts.
#
# Submit with slurm (run from the repo root):
#     sbatch tests/ttnn/distributed/run_multihost_graph_report.sh
#
# Or run interactively inside an existing 2-host allocation:
#     bash tests/ttnn/distributed/run_multihost_graph_report.sh
#
# The test manages its own graph capture and asserts on the merged db.sqlite, so we
# deliberately do NOT set enable_graph_report here (that would make the autouse
# ttnn_graph_report fixture open a competing capture).

#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --partition=bh_lb_B67
#SBATCH --job-name=mh_graph_report
#SBATCH --output=mh_graph_report_%j.out
#SBATCH --error=mh_graph_report_%j.err
#SBATCH --time=00:30:00

set -eo pipefail

export TT_METAL_HOME="$(pwd)"
export PYTHONPATH="$(pwd)"
export ARCH_NAME=blackhole

source python_env/bin/activate

RANK_BINDING="tests/ttnn/distributed/config/bh_lbx2_1x16_rank_bindings.yaml"
TEST="tests/ttnn/distributed/test_multihost_graph_report.py::test_multihost_graph_report"

# Build an OpenMPI hostfile from the slurm allocation (1 rank per node) when running
# under slurm; otherwise rely on the caller's MPI environment.
MPI_ARGS=""
if [[ -n "${SLURM_JOB_NODELIST:-}" ]]; then
    HOSTFILE="$(mktemp /tmp/mh_graph_report_hostfile.XXXXXX)"
    scontrol show hostnames "$SLURM_JOB_NODELIST" | awk '{print $1" slots=1"}' > "$HOSTFILE"
    echo "[mh-graph-report] allocated hosts:"
    cat "$HOSTFILE"
    trap 'rm -f "$HOSTFILE"' EXIT
    MPI_ARGS="--hostfile $HOSTFILE"
fi

tt-run \
  --rank-binding "$RANK_BINDING" \
  --mpi-args "$MPI_ARGS" \
  bash -c "source ${TT_METAL_HOME}/python_env/bin/activate && cd ${TT_METAL_HOME} && \
           pytest --disable-warnings -svv ${TEST}"

echo "[mh-graph-report] done. Merged report: generated/ttnn/reports/test_multihost_graph_report/db.sqlite"
