#!/bin/bash

set -x

pytest tests/scripts/test_moreh_microbenchmark.py::test_dram_read_all_core | tee test_dram_read_all_core.log
pytest tests/scripts/test_moreh_microbenchmark.py::test_pcie_transfer | tee test_pcie_transfer.log
pytest tests/scripts/test_moreh_microbenchmark.py::test_noc_rtor | tee test_noc_rtor.log
pytest tests/scripts/test_moreh_microbenchmark.py::test_noc_adjacent | tee test_noc_adjacent.log
pytest tests/tt_metal/microbenchmarks/ethernet/test_all_ethernet_links_bandwidth.py::test_erisc_bw_uni_dir --num-iterations 10 | tee bw.log
pytest tests/tt_metal/microbenchmarks/ethernet/test_all_ethernet_links_latency.py::test_erisc_latency_uni_dir --num-iterations 10 | tee latency.log
