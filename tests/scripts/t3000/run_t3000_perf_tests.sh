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

  pytest ${test_cmd} ; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: ${test_name} $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_qwenimage_tests() {
  run_t3000_dit_tests "models/tt_dit/tests/models/qwenimage/test_performance_qwenimage.py -k 2x4 --timeout 720"
}

fail=0
main() {
  # For CI pipeline - source func commands but don't execute tests if not invoked directly
  if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    echo "Script is being sourced, not executing main function"
    return 0
  fi

  # Parse the arguments
  while [[ $# -gt 0 ]]; do
    case $1 in
      --pipeline-type)
        pipeline_type=$2
        shift
        ;;
      *)
        echo "Unknown option: $1"
        exit 1
        ;;
    esac
    shift
  done

  if [[ -z "$TT_METAL_HOME" ]]; then
    echo "Must provide TT_METAL_HOME in environment" 1>&2
    exit 1
  fi

  if [[ -z "$ARCH_NAME" ]]; then
    echo "Must provide ARCH_NAME in environment" 1>&2
    exit 1
  fi

  if [[ -z "$pipeline_type" ]]; then
    echo "--pipeline-type cannot be empty" 1>&2
    exit 1
  fi

  # Run all tests
  cd $TT_METAL_HOME
  export PYTHONPATH=$TT_METAL_HOME

  # No aggregate pipeline_type remains: model_perf_t3000 only ever ran
  # run_t3000_resnet50_tests, which moved to the tiered Models CI. CI sources this
  # script and calls the run_t3000_* functions individually via t3k_perf_tests.yaml,
  # so main() is reachable only by direct execution.
  echo "$pipeline_type is invalid: no aggregate pipeline types are defined." 1>&2
  echo "Source this script and call a run_t3000_* function directly, e.g.:" 1>&2
  echo "  source ${BASH_SOURCE[0]} && run_t3000_stable_diffusion_35_large_tests" 1>&2
  exit 1
}

main "$@"
