#!/bin/bash
# Run a MiniMax-H3 pytest on a 4x8 Blackhole Galaxy with the environment the model tests expect.
#
#   models/tt_dit/models/transformers/minimax_h3/scripts/run_h3_test.sh <pytest args...>
#
# Real-weights pipeline tests additionally need MINIMAX_H3_MODEL_PATH (diffusers checkpoint directory)
# and, optionally, TT_DIT_CACHE_DIR (weight cache). Device tests should go through
# scripts/run_safe_pytest.sh (dispatch-timeout + triage wrapper); pass SAFE=1 to use it here.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../../../../../.." && pwd)
cd "$REPO"
export TT_METAL_HOME="$REPO"
export PYTHONPATH="$REPO"
export TT_MESH_GRAPH_DESC_PATH="${TT_MESH_GRAPH_DESC_PATH:-$REPO/tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_torus_xy_graph_descriptor.textproto}"
if [[ -n "${TT_DIT_CACHE_DIR:-}" ]]; then mkdir -p "$TT_DIT_CACHE_DIR"; fi
if [[ "${SAFE:-0}" == "1" ]]; then
    exec ./scripts/run_safe_pytest.sh "$@"
fi
exec ./python_env/bin/python -m pytest "$@"
