#!/usr/bin/env bash
#SBATCH --job-name=sec1-quick
#SBATCH --partition=bh_sc5_B2B9_D12
#SBATCH --nodelist=bh-glx-b06u08
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=/data/rsong/tt-metal2/craq-parity-results/slurm-sec1-quick-%j.out
#SBATCH --error=/data/rsong/tt-metal2/craq-parity-results/slurm-sec1-quick-%j.err

set -euo pipefail

REPO="${TT_METAL_HOME:-${SLURM_SUBMIT_DIR:-/data/rsong/tt-metal2}}"
# shellcheck source=scripts/lib/require-bh-glx-compute.sh
source "${REPO}/scripts/lib/require-bh-glx-compute.sh"
require_bh_glx_compute

export TT_METAL_HOME="$REPO"
chmod +x "${REPO}/craq-parity-results/run-ttnn-sec1-quick.sh"
exec "${REPO}/craq-parity-results/run-ttnn-sec1-quick.sh"
