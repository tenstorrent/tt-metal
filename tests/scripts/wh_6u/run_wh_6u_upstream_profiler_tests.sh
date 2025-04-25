#!/usr/bin/env bash
set -euo pipefail

echo "[upstream-profiler-tests] Running sanity fabric tests"
pytest tests/tt_metal/microbenchmarks/ethernet/test_fabric_edm_bandwidth.py -m sanity_6u

echo "[upstream-profiler-tests] Running Ethernet API tests"
eth_api_iterations=5
# ARCH_NAME=wormhole_b0 pytest tests/tt_metal/microbenchmarks/ethernet/test_all_ethernet_links_latency.py::test_erisc_latency_uni_dir --num-iterations $eth_api_iterations key_id error
# ARCH_NAME=wormhole_b0 pytest tests/tt_metal/microbenchmarks/ethernet/test_all_ethernet_links_bandwidth.py::test_erisc_bw_uni_dir --num-iterations $eth_api_iterations key_id error
