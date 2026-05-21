#!/usr/bin/env bash
#SBATCH --job-name=llk-smoke
#SBATCH --partition=bh_sc5_B2B9_D12
#SBATCH --nodelist=bh-glx-b07u02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:45:00
#SBATCH --output=/data/rsong/tt-metal2/craq-parity-results/llk-smoke-%j.out
#SBATCH --error=/data/rsong/tt-metal2/craq-parity-results/llk-smoke-%j.err

set -euo pipefail
REPO="/data/rsong/tt-metal2"
# shellcheck source=scripts/lib/require-bh-glx-compute.sh
source "${REPO}/scripts/lib/require-bh-glx-compute.sh"
require_bh_glx_compute
exec > >(tee -a "$REPO/craq-parity-results/llk-smoke-${SLURM_JOB_ID:-local}.out") 2>&1
chmod +x "$REPO/scripts/run-llk-smoke-ttsim.sh"
exec "$REPO/scripts/run-llk-smoke-ttsim.sh"
