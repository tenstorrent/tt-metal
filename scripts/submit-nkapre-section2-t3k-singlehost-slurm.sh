#!/usr/bin/env bash
# Login node: submit Section 2 T3000 single-host ttsim sweep to bh-glx Galaxy compute.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NODE="${NODE:-bh-glx-b06u08}"
PARTITION="${PARTITION:-bh_sc5_B2B9_D12}"
STAMP="${RESULTS_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
export RESULTS_STAMP="${STAMP}"
# Always default to a fresh section2 dir unless caller explicitly exported RESULTS_DIR before invoking this script.
if [[ -z "${RESULTS_DIR+x}" ]]; then
    export RESULTS_DIR="${REPO}/craq-parity-results/section2-t3k-${STAMP}"
fi

host="$(hostname -s 2>/dev/null || hostname)"
if [[ "$host" == bh-glx-* ]]; then
    echo "ERROR: already on compute node ($host). Run scripts/slurm-nkapre-section2-t3k-singlehost-job.sh directly." >&2
    exit 1
fi

mkdir -p "${REPO}/craq-parity-results"
chmod +x "${REPO}/scripts/slurm-nkapre-section2-t3k-singlehost-job.sh"

job_id=$(sbatch \
    --partition="${PARTITION}" \
    --nodelist="${NODE}" \
    --export=ALL,RESULTS_DIR,RESULTS_STAMP \
    "${REPO}/scripts/slurm-nkapre-section2-t3k-singlehost-job.sh" | awk '{print $4}')

echo "Submitted Section 2 T3000 single-host job ${job_id}"
echo "RESULTS_DIR=${RESULTS_DIR}"
echo "Monitor: squeue -j ${job_id}"
echo "Tail:    tail -f ${RESULTS_DIR}/run.log"
echo "Slurm:   ${REPO}/craq-parity-results/slurm-section2-t3k-${job_id}.out"
