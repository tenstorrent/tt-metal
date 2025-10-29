#!/bin/bash
# File: tests/scripts/tg/run_tg_deepseek_v3_demo_tests.sh
set -eo pipefail

fail=0

run_tg_deepseek_v3_tests() {
  local start_time end_time duration
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_tg_deepseek_v3_tests"

  pytest -n auto \
    models/demos/deepseek_v3/demo/test_demo_exact_word_match.py \
    --timeout 2700
  (( fail += $? ))

  end_time=$(date +%s)
  duration=$(( end_time - start_time ))
  echo "LOG_METAL: run_tg_deepseek_v3_tests took ${duration} seconds to complete"

  if [[ $fail -ne 0 ]]; then
    echo "LOG_METAL: run_tg_deepseek_v3_tests failed"
    exit 1
  fi
}

run_tg_deepseek_tests() {
  if [[ "$1" == "deepseek_v3" ]]; then
    run_tg_deepseek_v3_tests
  else
    echo "LOG_METAL: Unknown model type: $1 (expected: deepseek_v3)"
    return 1
  fi
}

main() {
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

  local model=""
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

  if [[ -z "$model" || "$model" == "None" ]]; then
    model="deepseek_v3"
  fi

  cd "$TT_METAL_HOME"
  export PYTHONPATH="$TT_METAL_HOME"

  run_tg_deepseek_tests "$model"

  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

main "$@"
