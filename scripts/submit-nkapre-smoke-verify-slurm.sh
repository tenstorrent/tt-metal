#!/usr/bin/env bash
# Login node: submit a small post-rebase smoke set (not full parity chain).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NODE="${NODE:-bh-glx-b06u08}"
PARTITION="${PARTITION:-bh_sc5_B2B9_D12}"
STAMP="${RESULTS_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"

export TT_METAL_HOME="${TT_METAL_HOME:-$REPO}"
export CRAQ_SIM="${CRAQ_SIM:-/data/rsong/craq-sim}"
export RESULTS_STAMP="${STAMP}"

host="$(hostname -s 2>/dev/null || hostname)"
if [[ "$host" == bh-glx-* ]]; then
    echo "ERROR: run from login node, not compute." >&2
    exit 1
fi

mkdir -p "${REPO}/craq-parity-results"
chmod +x \
    "${REPO}/craq-parity-results/run-craq-sim-smoke-compare.sh" \
    "${REPO}/craq-parity-results/run-ttnn-sec1-quick.sh" \
    "${REPO}/scripts/slurm-nkapre-sec1-quick-job.sh" \
    "${REPO}/scripts/slurm-nkapre-section2-quick-verify-job.sh"

submit() {
    local name="$1"
    local script="$2"
    local dep="${3:-}"
    local extra_export="${4:-}"
    local dep_arg=()
    if [[ -n "$dep" ]]; then
        dep_arg=(--dependency="afterok:${dep}")
    fi
    local job_id
    job_id=$(sbatch \
        --partition="${PARTITION}" \
        --nodelist="${NODE}" \
        "${dep_arg[@]}" \
        --export=ALL,TT_METAL_HOME,CRAQ_SIM,RESULTS_STAMP${extra_export} \
        "$script" | awk '{print $4}')
    echo "  ${name}: job ${job_id}"
    echo "$job_id"
}

echo "=== Post-rebase smoke verify (stamp=${STAMP}) ==="
echo "branch: $(git -C "$REPO" log -1 --oneline)"
echo "craq-sim: $(git -C "$CRAQ_SIM" log -1 --oneline)"
echo "node: ${NODE} (sequential chain)"
echo ""

j1=$(submit "craq-smoke (fabric/eth)" "${REPO}/craq-parity-results/run-craq-sim-smoke-compare.sh" "" "")
export RESULTS_DIR="${REPO}/craq-parity-results/section1-quick-${STAMP}"
j2=$(submit "section1-quick (mcq+region)" "${REPO}/scripts/slurm-nkapre-sec1-quick-job.sh" "$j1" "")
export RESULTS_DIR="${REPO}/craq-parity-results/section2-quick-${STAMP}"
export PER_TEST_TIMEOUT="${PER_TEST_TIMEOUT:-600}"
j3=$(submit "section2-quick (5 short T3K)" "${REPO}/scripts/slurm-nkapre-section2-quick-verify-job.sh" "$j2" ",RESULTS_DIR,PER_TEST_TIMEOUT")

echo ""
echo "Chain: ${j1} -> ${j2} -> ${j3}"
echo "Monitor: squeue -u $(whoami)"
