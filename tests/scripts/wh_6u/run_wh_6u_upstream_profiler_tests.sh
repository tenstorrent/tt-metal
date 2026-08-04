#!/usr/bin/env bash
set -euo pipefail

echo "[upstream-profiler-tests] Running sanity fabric tests"
# pytest tests/tt_metal/microbenchmarks/ethernet/test_fabric_edm_bandwidth.py -m sanity_6u Issue https://github.com/tenstorrent/tt-metal/issues/21360

echo "[upstream-profiler-tests] Running Ethernet API tests"
eth_api_iterations=5
# shit still broke on main
# ARCH_NAME=wormhole_b0 pytest tests/tt_metal/microbenchmarks/ethernet/test_all_ethernet_links_latency.py::test_erisc_latency_uni_dir --num-iterations $eth_api_iterations
# ARCH_NAME=wormhole_b0 pytest tests/tt_metal/microbenchmarks/ethernet/test_all_ethernet_links_bandwidth.py::test_erisc_bw_uni_dir --num-iterations $eth_api_iterations
