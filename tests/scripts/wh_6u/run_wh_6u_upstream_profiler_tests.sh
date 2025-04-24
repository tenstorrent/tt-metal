#!/usr/bin/env bash
set -euo pipefail

echo "[upstream-profiler-tests] Running sanity fabric tests"
pytest tests/tt_metal/microbenchmarks/ethernet/test_fabric_edm_bandwidth.py -m sanity_6u
