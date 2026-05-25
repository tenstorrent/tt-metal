#!/usr/bin/env bash
# Login node: submit T3K multiprocess (tt-run) sweep on Galaxy compute.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/lib/nkapre-slurm-b-aisle.sh
source "${REPO}/scripts/lib/nkapre-slurm-b-aisle.sh"

export TT_METAL_HOME="${TT_METAL_HOME:-${REPO}}"
export CRAQ_SIM="${CRAQ_SIM:-/data/rsong/craq-sim}"
STAMP="${RESULTS_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
export RESULTS_STAMP="${STAMP}"
export RESULTS_DIR="${REPO}/craq-parity-results/mp-run-${STAMP}"
export PER_CMD_TIMEOUT="${PER_CMD_TIMEOUT:-900}"
export MP_SKIP_ON_SIM="${MP_SKIP_ON_SIM:-0}"
export TTSIM_USE_DAEMON="${TTSIM_USE_DAEMON:-auto}"

host="$(hostname -s 2>/dev/null || hostname)"
if [[ "$host" == bh-glx-* ]]; then
    echo "ERROR: run from login node." >&2
    exit 1
fi

mkdir -p "${REPO}/craq-parity-results"
chmod +x "${REPO}/scripts/slurm-nkapre-mp-ttsim-job.sh" "${REPO}/scripts/run-nkapre-mp-ttsim-sweep.sh"

if [[ -n "${NODE:-}" ]]; then
    validate_nkapre_node "$NODE"
    PARTITION="${PARTITION:-$(partition_for_node "$NODE")}"
else
    NODE="${NODE:-bh-glx-110-c03u08}"
    validate_nkapre_node "$NODE"
    PARTITION="${PARTITION:-$(partition_for_node "$NODE")}"
fi

job_id=$(sbatch \
    --partition="${PARTITION}" \
    --nodelist="${NODE}" \
    --time=03:00:00 \
    --export=ALL,TT_METAL_HOME,CRAQ_SIM,RESULTS_DIR,RESULTS_STAMP,PER_CMD_TIMEOUT,MP_SKIP_ON_SIM,TTSIM_USE_DAEMON \
    "${REPO}/scripts/slurm-nkapre-mp-ttsim-job.sh" | awk '{print $4}')

echo "Submitted multiprocess (tt-run) job ${job_id} (node=${NODE}, partition=${PARTITION})"
echo "Suites: 10 tt-run T3K mp tests (WH ttsim + t3k mock cluster)"
echo "Prior run (6u mock): 1 PASS / 9 FAIL — retest with t3k_cluster_desc.yaml"
echo "RESULTS_DIR=${RESULTS_DIR}"
echo "Monitor: squeue -j ${job_id}"
echo "Tail:    tail -f ${RESULTS_DIR}/run.log"
