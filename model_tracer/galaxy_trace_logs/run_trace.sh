#!/bin/bash
# Helper to run a single model trace under the right env
set -eo pipefail
cd /data/stevenlee/tt-metal
source python_env/bin/activate
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)
export ARCH_NAME=wormhole_b0
exec "$@"
