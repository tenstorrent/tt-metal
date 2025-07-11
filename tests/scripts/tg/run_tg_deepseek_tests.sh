#!/bin/bash

run_tg_deepseek_unit_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_tg_deepseek_unit_tests"

  # Run tests
  export HF_MODEL="/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528"
  pytest -n auto models/demos/deepseek_v3/tests/test_embedding_1d.py --timeout 30; fail+=$?;
  pytest -n auto models/demos/deepseek_v3/tests/test_expert.py --timeout 30; fail+=$?;
  pytest -n auto models/demos/deepseek_v3/tests/test_mla_1d.py --timeout 30; fail+=$?;
  pytest -n auto models/demos/deepseek_v3/tests/test_mlp_1d.py --timeout 30; fail+=$?;
  pytest -n auto models/demos/deepseek_v3/tests/test_rms_norm.py --timeout 30; fail+=$?;
  unset HF_MODEL

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_tg_deepseek_unit_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_tg_deepseek_tests() {
  if [[ "$1" == "deepseek" ]]; then
    run_tg_deepseek_unit_tests
  else
    echo "LOG_METAL: Unknown model type: $1"
    return 1
  fi
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

  run_tg_deepseek_tests "$model"

  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

main "$@"
