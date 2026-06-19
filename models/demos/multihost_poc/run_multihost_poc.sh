#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Minimal 2-host Blackhole proof-of-concept for the TTNN report feature.
# Submit with:  sbatch models/demos/multihost_poc/run_multihost_poc.sh
# (run it from the repo root, i.e. /data/ctr-smountenay/tt-metal)
#
# This grabs 2 Blackhole "loudbox" nodes (the smallest 2-host config), launches the
# pytest test on both of them with tt-run (which wraps mpirun), and lets the autouse
# ttnn_graph_report fixture write a single merged multi-host report on rank 0.

#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --partition=bh_lb_B67          # grabs 2 of this partition's 4 idle nodes; bh_lb_B45 also has 4 idle
#SBATCH --job-name=multihost_poc
#SBATCH --output=multihost_poc_%j.out
#SBATCH --error=multihost_poc_%j.err
#SBATCH --time=00:30:00

set -eo pipefail

# --- repo / environment -----------------------------------------------------
export TT_METAL_HOME="$(pwd)"
export PYTHONPATH="$(pwd)"
export ARCH_NAME=blackhole

source python_env/bin/activate

# --- the TTNN report feature we are actually testing -------------------------
# tt-run forwards TTNN_* env vars to every rank, so each host captures its own
# graph and rank 0 merges them into one report (generated/ttnn/reports/...).
export TTNN_CONFIG_OVERRIDES='{
    "enable_fast_runtime_mode": false,
    "enable_logging": true,
    "report_name": "multihost_poc",
    "enable_graph_report": true,
    "enable_detailed_buffer_report": true,
    "enable_detailed_tensor_report": true,
    "enable_comparison_mode": false
}'

# --- build an OpenMPI hostfile from the slurm allocation (1 rank per node) ---
HOSTFILE="$(mktemp /tmp/multihost_poc_hostfile.XXXXXX)"
scontrol show hostnames "$SLURM_JOB_NODELIST" | awk '{print $1" slots=1"}' > "$HOSTFILE"
echo "[multihost-poc] allocated hosts:"
cat "$HOSTFILE"
trap 'rm -f "$HOSTFILE"' EXIT

RANK_BINDING="tests/tt_metal/distributed/config/bh_lbx2_1x16_rank_bindings.yaml"
TEST="models/demos/multihost_poc/test_multihost_poc.py::test_multihost_poc"

# --- launch the 2-host run --------------------------------------------------
tt-run \
  --rank-binding "$RANK_BINDING" \
  --mpi-args "--hostfile $HOSTFILE" \
  bash -c "source ${TT_METAL_HOME}/python_env/bin/activate && cd ${TT_METAL_HOME} && \
           pytest --disable-warnings -svv ${TEST}"

echo "[multihost-poc] done. Look for the merged report under: generated/ttnn/reports/multihost_poc_*"
