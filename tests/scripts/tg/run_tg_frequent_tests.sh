#!/bin/bash

run_tg_tests() {

  if [[ "$1" == "unit" ]]; then
    echo "LOG_METAL: running unit/distributed run_tg_frequent_tests"
    ## ERISC IRAM is always on for WH; these tests mix fabric and non-fabric CCL and rely on consistent jit/build behavior.
    pytest tests/ttnn/distributed/test_data_parallel_example_TG.py --timeout=900 ; fail+=$?
    pytest tests/ttnn/distributed/test_multidevice_TG.py --timeout=900 ; fail+=$?
    pytest tests/ttnn/unit_tests/base_functionality/test_multi_device_trace_TG.py --timeout=900 ; fail+=$?


  else
    echo "LOG_METAL: Unknown model type: $1"
    return 1
  fi

  if [[ $fail -ne 0 ]]; then
    echo "LOG_METAL: run_tg_frequent_tests failed"
    exit 1
  fi

}

main() {
  if [[ -z "$TT_METAL_HOME" ]]; then
    echo "Must provide TT_METAL_HOME in environment" 1>&2
    exit 1
  fi

  # Parse the arguments
  while [[ $# -gt 0 ]]; do
    case $1 in
      --model)
        model=$2
        shift
        ;;
      *)
        echo "Unknown option: $1"
        exit 1
        ;;
    esac
    shift
  done

  # Run all tests
  cd $TT_METAL_HOME
  export PYTHONPATH=$TT_METAL_HOME

  run_tg_tests "$model"
}

main "$@"
