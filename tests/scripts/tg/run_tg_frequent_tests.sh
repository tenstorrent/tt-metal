#!/bin/bash

run_tg_llama3_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_tg_llama3_tests"

  # Llama3.3-70B
  llama70b=/mnt/MLPerf/tt_dnn-models/llama/Llama3.3-70B-Instruct/

  # Run all Llama3 tests for 8B, 1B, and 3B weights
  # for llama_dir in "$llama1b" "$llama3b" "$llama8b" "$llama11b" "$llama70b"; do
  for llama_dir in "$llama70b"; do
    LLAMA_DIR=$llama_dir FAKE_DEVICE=TG pytest --timeout 1800 -n auto models/demos/llama3_subdevices/tests/test_llama_model_nd.py --timeout=1800 ; fail+=$?
    LLAMA_DIR=$llama_dir FAKE_DEVICE=TG pytest --timeout 1800 -n auto models/demos/llama3_subdevices/tests/test_llama_model.py -k full --timeout=1800 ; fail+=$?
    echo "LOG_METAL: Llama3 tests for $llama_dir completed"
  done

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_tg_llama3_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_tg_tests() {

  if [[ "$1" == "llama3" ]]; then
    echo "LOG_METAL: running Llama3 run_tg_frequent_tests"
    run_tg_llama3_tests

  elif [[ "$1" == "resnet50" ]]; then
    echo "LOG_METAL: running resnet50 run_tg_frequent_tests"
    pytest -n auto models/demos/tg/resnet50/tests/test_resnet50_performant.py ; fail+=$?

  elif [[ "$1" == "unit" ]]; then
    echo "LOG_METAL: running unit/distributed run_tg_frequent_tests"
    pytest -n auto tests/ttnn/distributed/test_data_parallel_example_TG.py --timeout=900 ; fail+=$?
    pytest -n auto tests/ttnn/distributed/test_multidevice_TG.py --timeout=900 ; fail+=$?
    pytest -n auto tests/ttnn/unit_tests/test_multi_device_trace_TG.py --timeout=900 ; fail+=$?
    pytest -n auto tests/ttnn/unit_tests/operations/ccl/test_all_gather_TG_post_commit.py --timeout=300 ; fail+=$?

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

  if [[ -z "$ARCH_NAME" ]]; then
    echo "Must provide ARCH_NAME in environment" 1>&2
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
