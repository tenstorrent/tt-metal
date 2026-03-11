#!/bin/bash
# Check device IDs visible on each host

echo "=== Checking device IDs on each host ==="

tt-run \
   --tcp-interface enp194s0f0np0 \
   --rank-binding $TT_METAL_HOME/tests/tt_metal/distributed/config/dual_bh_lb_rank_bindings.yaml \
   --mpi-args "--host bh-lb-02,bh-lb-03" \
   bash -c 'echo "Host: $(hostname), Rank: $TT_MESH_HOST_RANK"; python3 -c "import ttnn; print(\"Devices:\", ttnn.get_pcie_device_ids())"'
