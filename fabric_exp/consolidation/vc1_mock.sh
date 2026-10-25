#!/bin/bash
# VC1 split control-plane check on a mock of this host (CPU only, no device): same MGD, rank bindings and yaml as
# vc1_runs.sh, with --mock-cluster-rank-binding (fabric_exp/vc1/a08u24_mock_mapping.yaml). Reproduces whether the
# topology mapper can place the two 2x2 meshes from the cluster descriptor alone.
cd /home/aagarwal/tt-metal
env TT_METAL_HOME=/home/aagarwal/tt-metal TT_FABRIC_KERNEL_VERSION=2 TT_METAL_CACHE=/home/aagarwal/.tt-metal-cache-vc1-mock \
    timeout 300 python_env/bin/python ttnn/ttnn/distributed/ttrun.py \
    --mock-cluster-rank-binding fabric_exp/vc1/a08u24_mock_mapping.yaml \
    --rank-binding fabric_exp/vc1/p150x8_2x2_split_rank_bindings.yaml --mpi-args "--oversubscribe" \
    ./build/test/tt_metal/tt_fabric/test_infra/test_tt_fabric --test_config $PWD/fabric_exp/vc1/vc1_split.yaml \
    > fabric_exp/results/vc1mock.log 2>&1
echo "exit $?"
strings fabric_exp/results/vc1mock.log | sed 's/\x1b\[[0-9;]*m//g' | grep -aE 'Physical groupings|TT_FATAL|placement|Running Test|mock|Mock' | head -20
