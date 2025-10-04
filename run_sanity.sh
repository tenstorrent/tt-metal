#!/bin/bash

# This is a convenience bash script until we upgrade `tt-run` to support top-lvl config
# Source this file to set up MPI environment variables for 4x4 BH quietbox tests
# Usage: source setup_mpi_env.sh

export MPI_HOSTS="10.140.20.237,10.140.20.239,10.140.20.238,10.140.20.240"
export RANKFILE="tests/scale_out/4x_bh_quietbox/rankfile/4x4.txt"
export RANK_BINDING="tests/scale_out/4x_bh_quietbox/rank_bindings/4x4.yaml"
export MPI_COMMON_ARGS="--allow-run-as-root --tag-output --host $MPI_HOSTS --map-by rankfile:file=$RANKFILE --mca btl self,tcp --mca btl_tcp_if_include enp10s0f1np1"

## INSTRUCTIONS

# Distributed Reset:
# Since distributed/MPI reset introduces too much reset skew, we need to manually reset the system to ensure all links come up correctly
# We can do this by running `tt-smi -r` in synchronized tmux panes on each of the four machines

# Simplified command examples:

# If cluster_validation is failing, you need to repeat reset on all machines to ensure all links come up correctly
# If successful, it will report: "[1,0]<stdout>: All connections match between FSD and GSD (64 connections)"

tt-run --mpi-args "$MPI_COMMON_ARGS" --rank-binding $RANK_BINDING bash -c './build/tools/scaleout/run_cluster_validation --factory-descriptor-path tests/scale_out/4x_bh_quietbox/factory_system_descriptors/factory_system_descriptor_4x_bh_quietbox.textproto'
tt-run --mpi-args "$MPI_COMMON_ARGS" --rank-binding $RANK_BINDING bash -c 'TT_METAL_CLEAR_L1=1 ./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric --test_config ${TT_METAL_HOME}/tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_fabric_stability.yaml'

################ HANG ISSUE #1 #####################
#  - name: "MeshDynamicLooped"
#    fabric_setup:
#      topology: Mesh
#      routing_type: Dynamic
#
#    top_level_iterations: 1
#
#    parametrization_params:
#      ftype: [unicast, mcast]
#      ntype: [unicast_write, atomic_inc, unicast_scatter_write] ---------> Add `fused_atomic_inc` to the list and it causes ahang
#      num_links: [1, 2, 3, 4]
#
#    defaults:
#      size: 32
#      num_packets: 128
#
#    patterns:
#      - type: all_to_all
#        iterations: 1


################ HANG ISSUE #2 #####################
# Rerun with `/tests/tt_metal/tt_metal/perf_microbenchmark/routing/bh_qb_4X4_1D_ring_sanity.yaml`
