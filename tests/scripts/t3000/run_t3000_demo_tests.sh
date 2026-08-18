#!/bin/bash
set -eo pipefail

TT_CACHE_HOME=/mnt/MLPerf/huggingface/tt_cache



run_t3000_dit_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)
  test_name=${FUNCNAME[1]}
  test_cmd=$1

  echo "LOG_METAL: Running ${test_name}"

  NO_PROMPT=1 pytest ${test_cmd} --timeout 1200 ; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: ${test_name} $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_qwenimage_tests() {
  run_t3000_dit_tests "models/tt_dit/tests/models/qwenimage/test_pipeline_qwenimage.py -k 2x4"
}

run_t3000_tests() {


  # Run qwenimage tests
  run_t3000_qwenimage_tests

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

  if [[ -z "$ARCH_NAME" ]]; then
    echo "Must provide ARCH_NAME in environment" 1>&2
    exit 1
  fi

  # Run all tests
  cd $TT_METAL_HOME
  export PYTHONPATH=$TT_METAL_HOME

  run_t3000_tests

  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

main "$@"
