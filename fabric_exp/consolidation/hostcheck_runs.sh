#!/bin/bash
# Host check on bh-lb-120-a09u32 (read-only, no kernels): the UMD cluster descriptor (topo_dump.py; does chips {0..3} and
# {4..7} form the VC1 split's two 2x2 meshes?) and every Ethernet core's training status and speed (read_eth_speed.py).
cd /home/aagarwal/tt-metal
echo "=== topology"
timeout 300 python_env/bin/python fabric_exp/consolidation/topo_dump.py fabric_exp/a09u32_cluster_desc.yaml 2>&1 | grep -v -E '^\s*$' | tail -40
echo "=== ethernet links"
timeout 300 python_env/bin/python -I fabric_exp/read_eth_speed.py 8 2>&1 | tail -60
echo ALL_DONE
