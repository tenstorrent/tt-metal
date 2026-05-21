#!/usr/bin/env bash
# Login-node ONLY: submit a quick LLK smoke job to bh-glx Galaxy compute.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NODE="${NODE:-bh-glx-b07u02}"
PARTITION="${PARTITION:-bh_sc5_B2B9_D12}"

host="$(hostname -s 2>/dev/null || hostname)"
if [[ "$host" == bh-glx-* ]]; then
    echo "ERROR: already on compute node ($host). Run ./scripts/slurm-llk-smoke-ttsim-job.sh directly." >&2
    exit 1
fi

mkdir -p "${REPO}/craq-parity-results"
chmod +x "${REPO}/scripts/slurm-llk-smoke-ttsim-job.sh" "${REPO}/scripts/run-llk-smoke-ttsim.sh"

job_id=$(sbatch \
    --partition="${PARTITION}" \
    --nodelist="${NODE}" \
    --cpus-per-task=1 \
    --export=TT_METAL_HOME,CRAQ_SIM,HOME,PATH,USER \
    "${REPO}/scripts/slurm-llk-smoke-ttsim-job.sh" | awk '{print $4}')

echo "Submitted LLK smoke job ${job_id} -> ${NODE}"
echo "Monitor: squeue -j ${job_id}"
echo "Tail: tail -f ${REPO}/craq-parity-results/llk-smoke-${job_id}.out"
echo "      tail -f ${REPO}/craq-parity-results/llk-smoke-latest/summary.txt"
