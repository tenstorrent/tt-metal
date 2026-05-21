#!/usr/bin/env bash
# Login-node ONLY: submit full nkapre parity ttsim sweep to bh-glx Galaxy compute.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NODE="${NODE:-bh-glx-b06u08}"
PARTITION="${PARTITION:-bh_sc5_B2B9_D12}"
MODE="${1:-sbatch}"
STAMP="${RESULTS_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
export RESULTS_STAMP="${STAMP}"
export RESULTS_DIR="${RESULTS_DIR:-${REPO}/craq-parity-results/run-${STAMP}}"

host="$(hostname -s 2>/dev/null || hostname)"
if [[ "$host" == bh-glx-* ]]; then
    echo "ERROR: already on compute node ($host). Run ./scripts/slurm-nkapre-parity-ttsim-job.sh directly." >&2
    exit 1
fi

mkdir -p "${REPO}/craq-parity-results"
chmod +x "${REPO}/scripts/slurm-nkapre-parity-ttsim-job.sh"

echo "Idle ${PARTITION} nodes:"
sinfo -N -p "${PARTITION}" -t idle -o "%N %T %C %e" 2>/dev/null || true
echo ""
echo "Submitting full nkapre parity ttsim sweep to ${NODE}"
echo "RESULTS_DIR=${RESULTS_DIR}"
echo ""
echo "Tail live output (after job starts):"
echo "  tail -f ${RESULTS_DIR}/run.log"
echo "  # or: tail -f ${REPO}/craq-parity-results/latest/run.log"

case "$MODE" in
    srun)
        exec srun \
            --partition="${PARTITION}" \
            --nodelist="${NODE}" \
            --nodes=1 \
            --ntasks=1 \
            --cpus-per-task=1 \
            --time=08:00:00 \
            --export=ALL,RESULTS_DIR,RESULTS_STAMP \
            bash "${REPO}/scripts/slurm-nkapre-parity-ttsim-job.sh"
        ;;
    sbatch)
        job_id=$(sbatch \
            --partition="${PARTITION}" \
            --nodelist="${NODE}" \
            --export=ALL,RESULTS_DIR,RESULTS_STAMP \
            "${REPO}/scripts/slurm-nkapre-parity-ttsim-job.sh" | awk '{print $4}')
        echo "Submitted job ${job_id}"
        echo "Monitor: squeue -j ${job_id}"
        echo "Slurm log: ${REPO}/craq-parity-results/slurm-${job_id}.out"
        ;;
    *)
        echo "usage: $(basename "$0") [sbatch|srun]" >&2
        exit 2
        ;;
esac
