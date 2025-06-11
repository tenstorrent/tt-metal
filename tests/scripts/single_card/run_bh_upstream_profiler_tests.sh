#!/usr/bin/env bash
set -euo pipefail

echo "[upstream-profiler-tests] Run BH python upstream tests"
pytest --collect-only tests/ttnn/unit_tests

echo "[upstream-profiler-tests] Running BH ethernet profiler tests"
eth_api_iterations=5
ARCH_NAME=blackhole pytest tests/tt_metal/microbenchmarks/ethernet/test_all_ethernet_links_latency.py::test_erisc_latency_uni_dir --num-iterations $eth_api_iterations
ARCH_NAME=blackhole pytest tests/tt_metal/microbenchmarks/ethernet/test_all_ethernet_links_bandwidth.py::test_erisc_bw_uni_dir --num-iterations $eth_api_iterations
