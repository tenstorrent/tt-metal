#!/bin/bash
# /opt/test_env.sh

echo "=== Environment Variables on $(hostname) ==="
echo "ARCH_NAME: $ARCH_NAME"
echo "TT_METAL_HOME: $TT_METAL_HOME" 
echo "PYTHONPATH: $PYTHONPATH"
echo "TT_METAL_ENV: $TT_METAL_ENV"
echo "TT_METAL_MESH_ID: $TT_METAL_MESH_ID"
echo "TT_METAL_HOST_RANK_ID: $TT_METAL_HOST_RANK_ID"
echo "Current working directory: $(pwd)"
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo "============================================="
tt-smi -r
/home/asaigal/tt-metal/build_Release/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="*TestUnicastRaw*"
