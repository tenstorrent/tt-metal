#!/usr/bin/env bash
# Run a command inside the tt-metal dev container with env + python_env sourced.
# Usage: tools/mm_sweep/run_in_container.sh <command...>
set -euo pipefail
CONTAINER="${CONTAINER:-silly_joliot}"
docker exec "$CONTAINER" bash -lc '
cd /home/cglagovich/tt-metal
export ARCH_NAME=wormhole_b0
export TT_METAL_HOME=/home/cglagovich/tt-metal
export PYTHONPATH=/home/cglagovich/tt-metal
source python_env/bin/activate
'"$*"
