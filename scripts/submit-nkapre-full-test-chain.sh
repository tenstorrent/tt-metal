#!/usr/bin/env bash
# Submit full parity test chain on bh-glx-b06u08 (serial via Slurm dependencies).
# Uses latest craq-sim multichip lib copied at job start.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NODE="${NODE:-bh-glx-b06u08}"
PARTITION="${PARTITION:-bh_sc5_B2B9_D12}"
STAMP="${RESULTS_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
export RESULTS_STAMP="${STAMP}"
export CRAQ_SIM="${CRAQ_SIM:-/data/rsong/craq-sim}"
export TT_METAL_HOME="${TT_METAL_HOME:-$REPO}"
export PER_CMD_TIMEOUT="${PER_CMD_TIMEOUT:-3600}"

host="$(hostname -s 2>/dev/null || hostname)"
if [[ "$host" == bh-glx-* ]]; then
    echo "ERROR: run from login node, not compute." >&2
    exit 1
fi

# Optional: wait for these jobs before starting the chain (comma-separated).
WAIT_AFTER="${1:-11604}"

DEP="$WAIT_AFTER"
submit() {
    local name="$1"
    local script="$2"
    local results_subdir="$3"
    shift 3
    export RESULTS_DIR="${REPO}/craq-parity-results/${results_subdir}-${STAMP}"
    mkdir -p "${REPO}/craq-parity-results"
    chmod +x "$script"
    local dep_arg=""
    if [ -n "$DEP" ]; then
        dep_arg="--dependency=afterany:${DEP}"
    fi
    local job_id
    job_id=$(sbatch ${dep_arg} \
        --partition="${PARTITION}" \
        --nodelist="${NODE}" \
        --export=ALL,RESULTS_DIR,RESULTS_STAMP,CRAQ_SIM,TT_METAL_HOME,PER_CMD_TIMEOUT \
        "$@" \
        "$script" | awk '{print $4}')
    echo "  ${name}: job ${job_id} -> ${RESULTS_DIR}"
    DEP="$job_id"
}

echo "=== Full parity chain (stamp=${STAMP}) ==="
echo "craq-sim: $(git -C "$CRAQ_SIM" log -1 --oneline)"
echo "Waiting after job(s): ${WAIT_AFTER}"
echo ""

# Re-order pending jobs with dependencies on 11604
if squeue -j 11605 -h 2>/dev/null | grep -q .; then
    scancel 11605 2>/dev/null || true
    submit "fabric-ttsim" "${REPO}/scripts/slurm-nkapre-fabric-ttsim-job.sh" "fabric" \
        --time=08:00:00
fi
if squeue -j 11608 -h 2>/dev/null | grep -q .; then
    scancel 11608 2>/dev/null || true
    submit "craq-smoke" "${REPO}/craq-parity-results/run-craq-sim-smoke-compare.sh" "craq-smoke" \
        --time=02:00:00
fi

submit "section2-rerun" "${REPO}/scripts/slurm-nkapre-section2-t3k-singlehost-job.sh" "section2-t3k-v2" \
    --time=12:00:00
submit "section1-cpp" "${REPO}/scripts/slurm-nkapre-section1-cpp-job.sh" "section1-cpp" \
    --time=02:00:00
submit "section1-pytest" "${REPO}/scripts/slurm-nkapre-section1-pytest-job.sh" "section1-pytest" \
    --time=04:00:00
submit "p300-smoke" "${REPO}/craq-parity-results/run-p300-smoke-test.sh" "p300-smoke" \
    --time=02:00:00 --job-name=p300-smoke
submit "section3-galaxy" "${REPO}/scripts/slurm-nkapre-section3-galaxy-job.sh" "section3-galaxy" \
    --time=06:00:00
submit "llk" "${REPO}/scripts/slurm-nkapre-llk-ttsim-job.sh" "llk-run" \
    --time=04:00:00
submit "multiprocess" "${REPO}/scripts/slurm-nkapre-mp-ttsim-job.sh" "mp-run" \
    --time=02:00:00

echo ""
echo "Final job in chain: ${DEP}"
echo "Monitor: squeue -u $(whoami)"
echo "Status doc: ${REPO}/craq-parity-results/PARITY_STATUS.md"
echo "Manifest: ${REPO}/craq-parity-results/full-chain-${STAMP}.txt"
{
    echo "stamp=${STAMP}"
    echo "wait_after=${WAIT_AFTER}"
    echo "final_job=${DEP}"
    echo "craq_sim=$(git -C "$CRAQ_SIM" log -1 --oneline)"
    date -u -Iseconds
} >"${REPO}/craq-parity-results/full-chain-${STAMP}.txt"
