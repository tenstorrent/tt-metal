#!/bin/bash

source python_env/bin/activate
export TT_METAL_HOME=/home/local-syseng/tt-metal
# tt-run --rank-binding tests/tt_metal/distributed/config/quad_galaxy_rank_bindings.yaml --mpi-args "--host bh-glx-c01u02,bh-glx-c01u08,bh-glx-c02u02,bh-glx-c02u08 --mca btl_tcp_if_include ens5f0np0 --tag-output" ./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric --test_config tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_fabric_stability.yaml
tt-run --rank-binding tests/tt_metal/distributed/config/quad_galaxy_rank_bindings.yaml --mpi-args "--host bh-glx-c01u02,bh-glx-c01u08,bh-glx-c02u02,bh-glx-c02u08 --mca btl_tcp_if_include ens5f0np0 --tag-output" ./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric --test_config tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_fabric_deadlock_stability_bh_6U_galaxy.yaml
tt-run --rank-binding tests/tt_metal/distributed/config/quad_galaxy_rank_bindings.yaml --mpi-args "--host bh-glx-c01u02,bh-glx-c01u08,bh-glx-c02u02,bh-glx-c02u08 --mca btl_tcp_if_include ens5f0np0 --tag-output" ./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric --test_config tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_dual_galaxy_fabric_2d_sanity.yaml
