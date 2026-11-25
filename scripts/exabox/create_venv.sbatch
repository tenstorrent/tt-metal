#!/bin/bash
#SBATCH --job-name=ttmetal-venv
#SBATCH --partition=cpu_only
#SBATCH --nodes=1
#SBATCH --exclude=prx-k8s-slurm-agent-03,prx-k8s-slurm-agent-04
#SBATCH --time=00:40:00
#SBATCH --output=/data/dstoiljkovic/logs/%x-%j.out
# Create tt-metal python_env on a cpu_only node. HOME is redirected to /data so uv, its Python and its cache
# do not land on the (nearly full) node-local disk and the env is usable from the reserved node.
set -eo pipefail
echo "node=$(hostname) start=$(date -u +%FT%TZ)"
free_gb=$(df --output=avail -BG /tmp | tail -1 | tr -dc 0-9); echo "free /tmp: ${free_gb} GB"
[ "$free_gb" -ge 5 ] || { echo "ABORT: low /tmp on $(hostname)"; exit 3; }
export HOME=/data/dstoiljkovic/buildhome TMPDIR=/data/dstoiljkovic/tmp
mkdir -p "$HOME" "$TMPDIR"; export PATH="$HOME/.local/bin:$PATH"
cd /data/dstoiljkovic/tt-metal
./create_venv.sh
echo "VENV_DONE $(date -u +%FT%TZ)"
