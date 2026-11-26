#!/bin/bash
#SBATCH --job-name=ttmetal-venv
#SBATCH --nodes=1
#SBATCH --time=00:40:00
# Create tt-metal python_env on a build node. HOME is redirected to $XRUN_DATA so uv, its Python and its cache
# do not land on the (nearly full) node-local disk and the env is usable from the reserved node.
# Submit from the repo: . scripts/exabox/env.sh && sbatch -p "$XRUN_BUILD_PARTITION" -o "$XRUN_LOGS/%x-%j.out" scripts/exabox/create_venv.sbatch
set -eo pipefail
. "$XRUN_REPO/scripts/exabox/env.sh"
echo "node=$(hostname) start=$(date -u +%FT%TZ)"
free_gb=$(df --output=avail -BG /tmp | tail -1 | tr -dc 0-9); echo "free /tmp: ${free_gb} GB"
[ "$free_gb" -ge 5 ] || { echo "ABORT: low /tmp on $(hostname)"; exit 3; }
export HOME="$XRUN_DATA/buildhome" TMPDIR="$XRUN_DATA/tmp"
mkdir -p "$HOME" "$TMPDIR"; export PATH="$HOME/.local/bin:$PATH"
cd "$XRUN_REPO"
./create_venv.sh
echo "VENV_DONE $(date -u +%FT%TZ)"
