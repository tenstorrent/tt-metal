#!/bin/bash
set -eo pipefail

run_tg_all_to_all_dispatch_perf_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running all to all dispatch perf tests"

  pytest -n auto -k TG tests/ttnn/multidevice_perf_tests/test_all_to_all_dispatch_perf.py; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: all_to_all_dispatch_perf_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_tg_op_tests() {
  # Run ccl performance tests
  run_tg_all_to_all_dispatch_perf_tests

}

fail=0
main() {
  # For CI pipeline - source func commands but don't execute tests if not invoked directly
  if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    echo "Script is being sourced, not executing main function"
    return 0
  fi

  if [[ -z "$TT_METAL_HOME" ]]; then
    echo "Must provide TT_METAL_HOME in environment" 1>&2
    exit 1
  fi

  # Run all tests
  cd $TT_METAL_HOME
  export PYTHONPATH=$TT_METAL_HOME

  run_tg_op_tests

  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

main "$@"
