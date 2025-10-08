#!/bin/bash
#28447 - TGG tests should be killed
set -eo pipefail

run_tgg_tests() {

  echo "LOG_METAL: running run_tgg_unit_tests"

  TT_METAL_ENABLE_REMOTE_CHIP=1 ./build/test/tt_metal/unit_tests_dispatch --gtest_filter="CommandQueueSingleCard*Fixture.*"
  pytest -s tests/ttnn/distributed/test_mesh_device_TGG.py
}

run_tgg_multiprocess_tests() {
  local mpi_args="--allow-run-as-root --tag-output"

  echo "LOG_METAL: running run_tgg_multiprocess_tests (dual 4x4 mesh tests)"

  # Test dual 4x4 meshes on Galaxy (8x4 system split into two 4x4 meshes)
  # This requires a 32-device Galaxy system
  tt-run --mpi-args "--allow-run-as-root --tag-output" --rank-binding tests/tt_metal/distributed/config/galaxy_4x4_multiprocess_rank_bindings.yaml ./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric \--test_config tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_galaxy_4x4.yaml
}

run_tgg_cycle_detection_test() {
  local mpi_args="--allow-run-as-root --tag-output"

  echo "LOG_METAL: running run_tgg_cycle_detection_test (tests FIXED cycle detection)"
  echo "LOG_METAL: This test uses bidirectional traffic to validate the fix filters false positives"
  echo "LOG_METAL: Expected result: No cycles detected (bidirectional flows use independent resources)"

  # Run the bidirectional inter-mesh traffic test
  # With the fix, this should NOT report false positive cycles
  # Enable debug logging to see cycle detection filtering in action
  TT_METAL_LOGGER_LEVEL=DEBUG tt-run --mpi-args "$mpi_args" \
    --rank-binding tests/tt_metal/distributed/config/galaxy_4x4_multiprocess_rank_bindings.yaml \
    ./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric \
    --test_config tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_galaxy_4x4.yaml \
    --gtest_filter="*BidirectionalInterMeshTraffic*"

  echo "LOG_METAL: Cycle detection test complete."
  echo "LOG_METAL: Check debug logs for 'Filtering out false positive' messages."
  echo "LOG_METAL: If cycles were still detected, the fix may need refinement."
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
  run_tgg_multiprocess_tests
}

main "$@"
