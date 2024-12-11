#!/bin/bash
set -x
set -eo pipefail

run_profiling_test() {
  if [[ -z "$ARCH_NAME" ]]; then
    echo "Must provide ARCH_NAME in environment" 1>&2
    exit 1
  fi

  if [[ -z "$TT_METAL_HOME" ]]; then
    echo "Must provide TT_METAL_HOME in environment" 1>&2
    exit 1
  fi

  if [[ "$TT_METAL_DEVICE_PROFILER" != 1 ]]; then
    echo "Must set TT_METAL_DEVICE_PROFILER to 1 to run microbenchmarks" 1>&2
    exit 1
  fi

  echo "Make sure this test runs in a build with cmake option ENABLE_TRACY=ON"

  source python_env/bin/activate
  export PYTHONPATH=$TT_METAL_HOME
  export TT_METAL_CLEAR_L1=1

  pytest --capture=tee-sys $TT_METAL_HOME/tests/ttnn/unit_tests/operations/ccl/perf/ccl_gtest_trace.py::test_cpp_unit_test

}

run_profiling_test
