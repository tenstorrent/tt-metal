#!/bin/bash
set -u
O=/home/ttuser/agobeljic/tt-metal/tt-train/sources/ttml/ttml/models/mistral/doc/optimization/evidence
until grep -q "fused shapes sweep finished" $O/iter46_fused_shapes_sweep.log; do sleep 10; done
cd /home/ttuser/agobeljic/tt-metal/tt-train
export TT_METAL_HOME=/home/ttuser/agobeljic/tt-metal TT_METAL_RUNTIME_ROOT=/home/ttuser/agobeljic/tt-metal
unset TT_MESH_GRAPH_DESC_PATH; rm -f /dev/shm/sm_segment.* 2>/dev/null
timeout 900 ../python_env/bin/python $O/iter46_attn_configs_abab.py > $O/iter46_attn_configs_abab.log 2>&1; echo "rc=$?"
grep -E "^(qkv|out)" $O/iter46_attn_configs_abab.log
echo "=== abab finished"
