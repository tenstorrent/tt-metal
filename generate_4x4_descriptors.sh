#!/bin/bash
# Script to generate 4x4 dual mesh cluster descriptors from actual hardware
# This should be run on a system with the proper hardware configuration

set -e

REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
OUTPUT_DIR="${REPO_ROOT}/tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors"

echo "=========================================="
echo "Generating 4x4 Dual Mesh Cluster Descriptors"
echo "=========================================="
echo ""

# Check if we're in a distributed environment
if [ ! -z "${TTNN_DEVICE_MESH_RANK}" ]; then
    echo "Running in distributed mode (Rank: ${TTNN_DEVICE_MESH_RANK})"
    echo "Output directory: ${OUTPUT_DIR}"
    python3 "${REPO_ROOT}/generate_4x4_dual_mesh_descriptors.py"
else
    echo "Running in single-process mode - you need to run this with tt-run!"
    echo ""
    echo "Usage:"
    echo "  tt-run --rank-binding tests/tt_metal/distributed/config/galaxy_4x4_strict_connection_rank_bindings.yaml \\"
    echo "         bash generate_4x4_descriptors.sh"
    echo ""
    exit 1
fi

echo ""
echo "✓ Cluster descriptor generation complete!"
