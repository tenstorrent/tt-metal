#!/bin/bash
# Run an H3 model test on the 4x8 galaxy under the safe-pytest harness (hang detection + triage).
# Usage: ./run_h3_safe.sh <log-name> <pytest args...>
cd "$(dirname "$0")"
LOG=/tmp/claude-4015/-data-cglagovich-tt-metal/7bf353e5-b3ae-491a-987f-f299bc56b26a/scratchpad/$1.log
shift
# Hang recovery (tt-smi -r) can leave a torus eth link untrained; the galaxy reset retrains it.
tt-smi -glx_reset > /dev/null 2>&1
export TT_MESH_GRAPH_DESC_PATH="$(pwd)/tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_torus_xy_graph_descriptor.textproto"
./scripts/run_safe_pytest.sh "$@" > "$LOG" 2>&1
grep -E "passed|failed|PASSED|FAILED|SAFE_PYTEST_RESULT" "$LOG" | tail -6
