#!/bin/bash
set -eo pipefail

run_tgg_tests() {

  echo "LOG_METAL: running run_tgg_unit_tests"

  TT_METAL_ENABLE_REMOTE_CHIP=1 ./build/test/tt_metal/unit_tests_dispatch --gtest_filter="CommandQueueSingleCard*Fixture.*"
  TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/unit_tests_device --gtest_filter="GalaxyFixture.*:TGGFixture.*"
  ./build/test/tt_metal/unit_tests_device --gtest_filter="GalaxyFixture.*:TGGFixture.*"
  pytest -s tests/ttnn/distributed/test_mesh_device_TGG.py
}

main() {
  if [[ -z "$TT_METAL_HOME" ]]; then
    echo "Must provide TT_METAL_HOME in environment" 1>&2
    exit 1
  fi

  if [[ -z "$ARCH_NAME" ]]; then
    echo "Must provide ARCH_NAME in environment" 1>&2
    exit 1
  fi

  # Run all tests
  cd $TT_METAL_HOME
  export PYTHONPATH=$TT_METAL_HOME

  run_tgg_tests
}

main "$@"
