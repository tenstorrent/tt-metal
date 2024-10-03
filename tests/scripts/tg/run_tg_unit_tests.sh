#!/bin/bash
set -eo pipefail

run_tg_tests() {

  echo "LOG_METAL: running run_tg_unit_tests"

  TT_METAL_ENABLE_REMOTE_CHIP=1 ./build/test/tt_metal/unit_tests_fast_dispatch --gtest_filter="CommandQueueSingleCardFixture.*"
  ./build/test/ttnn/galaxy_unit_tests_ttnn
  TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/unit_tests_galaxy --gtest_filter="GalaxyFixture.*:TGFixture.*"
  ./build/test/tt_metal/unit_tests_galaxy --gtest_filter="GalaxyFixture.*:TGFixture.*"
  TT_METAL_GTEST_NUM_HW_CQS=2 ./build/test/tt_metal/unit_tests_fast_dispatch_single_chip_multi_queue --gtest_filter="MultiCommandQueueMultiDeviceFixture.*"

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

  run_tg_tests
}

main "$@"
