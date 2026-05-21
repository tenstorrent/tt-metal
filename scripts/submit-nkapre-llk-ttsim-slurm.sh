#!/usr/bin/env bash
# Login-node ONLY: submit LLK craq-sim sweeps to bh-glx Galaxy compute.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NODE="${NODE:-bh-glx-b06u08}"
PARTITION="${PARTITION:-bh_sc5_B2B9_D12}"
STAMP="${RESULTS_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
export RESULTS_STAMP="${STAMP}"
export RESULTS_DIR="${RESULTS_DIR:-${REPO}/craq-parity-results/llk-run-${STAMP}}"

host="$(hostname -s 2>/dev/null || hostname)"
if [[ "$host" == bh-glx-* ]]; then
    echo "ERROR: already on compute node ($host). Run ./scripts/slurm-nkapre-llk-ttsim-job.sh directly." >&2
    exit 1
fi

mkdir -p "${REPO}/craq-parity-results"
chmod +x "${REPO}/scripts/slurm-nkapre-llk-ttsim-job.sh"

job_id=$(sbatch \
    --partition="${PARTITION}" \
    --nodelist="${NODE}" \
    --cpus-per-task=1 \
    --export=RESULTS_DIR,RESULTS_STAMP,TT_METAL_HOME,CRAQ_SIM,HOME,PATH,USER \
    "${REPO}/scripts/slurm-nkapre-llk-ttsim-job.sh" | awk '{print $4}')

echo "Submitted LLK job ${job_id} -> ${NODE}"
echo "Monitor: squeue -j ${job_id}"
echo "Tail: tail -f ${RESULTS_DIR}/run.log"
echo "      tail -f ${REPO}/craq-parity-results/llk-latest/run.log"
