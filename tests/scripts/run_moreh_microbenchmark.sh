#!/bin/bash
set -x
set -eo pipefail

run_profiling_test() {
  if [[ -z "$ARCH_NAME" ]]; then
    echo "Must provide ARCH_NAME in environment" 1>&2
    exit 1
  fi

  if [[ "$TT_METAL_DEVICE_PROFILER" != 1 ]]; then
    echo "Must set TT_METAL_DEVICE_PROFILER to 1 to run microbenchmarks" 1>&2
    exit 1
  fi

  echo "Make sure this test runs in a build with cmake option ENABLE_TRACY=ON"

  export TT_METAL_CLEAR_L1=1

  pytest --capture=tee-sys $TT_METAL_HOME/tests/scripts/test_moreh_microbenchmark.py::test_pcie_h2d_dram --timeout=720
  pytest --capture=tee-sys $TT_METAL_HOME/tests/scripts/test_moreh_microbenchmark.py::test_pcie_d2h_dram --timeout=720
  pytest --capture=tee-sys $TT_METAL_HOME/tests/scripts/test_moreh_microbenchmark.py::test_pcie_h2d_l1 -k $ARCH_NAME
  pytest --capture=tee-sys $TT_METAL_HOME/tests/scripts/test_moreh_microbenchmark.py::test_pcie_d2h_l1 -k $ARCH_NAME
  # pytest --capture=tee-sys $TT_METAL_HOME/tests/scripts/test_moreh_microbenchmark.py::test_noc -k $ARCH_NAME
  pytest --capture=tee-sys $TT_METAL_HOME/tests/scripts/test_moreh_microbenchmark.py::test_matmul_dram -k $ARCH_NAME # how to set r and c for this
  pytest --capture=tee-sys $TT_METAL_HOME/tests/scripts/test_moreh_microbenchmark.py::test_matmul_l1 -k $ARCH_NAME # how to set r and c for this

  pytest --capture=tee-sys $TT_METAL_HOME/tests/scripts/test_moreh_microbenchmark.py::test_matmul_single_core_sharded -k $ARCH_NAME
  pytest --capture=tee-sys $TT_METAL_HOME/tests/scripts/test_moreh_microbenchmark.py::test_dram_read_all_core -k $ARCH_NAME
  pytest --capture=tee-sys $TT_METAL_HOME/tests/scripts/test_moreh_microbenchmark.py::test_dram_read_remote_cb_sync -k $ARCH_NAME
}

run_profiling_test
