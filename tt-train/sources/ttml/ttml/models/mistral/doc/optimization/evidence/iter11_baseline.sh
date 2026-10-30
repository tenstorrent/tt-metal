#!/bin/bash
# Iteration 11 baseline: SDPA kernel outputs and wall times with the pre-iteration kernels (one chip).
set -u
cd /home/ttuser/agobeljic/tt-metal/tt-train
export TT_METAL_HOME=/home/ttuser/agobeljic/tt-metal TT_METAL_RUNTIME_ROOT=/home/ttuser/agobeljic/tt-metal
unset TT_MESH_GRAPH_DESC_PATH
rm -f /dev/shm/sm_segment.* 2>/dev/null
O=/home/ttuser/agobeljic/tt-metal/tt-train/sources/ttml/ttml/models/mistral/doc/optimization/evidence
SCR=/tmp/claude-1000/-home-ttuser-agobeljic-tt-metal/0cbc71c0-30d2-4e45-a0c0-f0a2063e85a7/scratchpad
echo "=== sdpa baseline $(date -u +%FT%TZ)"
timeout 1200 ../python_env/bin/python $O/iter11_sdpa_check.py --save $SCR/iter11_sdpa_baseline.npz 2>&1 | grep -v "^\s*$" | grep -E "case|pcc|saved|Error|error" ; echo "rc=${PIPESTATUS[0]}"
echo "=== done $(date -u +%FT%TZ)"
