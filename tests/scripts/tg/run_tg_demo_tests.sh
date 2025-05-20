#!/bin/bash

run_tg_llama3_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_tg_llama3_tests"

  # Llama3.3-70B
  llama70b=/mnt/MLPerf/tt_dnn-models/llama/Llama3.3-70B-Instruct/

  # Run all Llama3 tests for 1B, 3B, 8B, 11B and 70B weights
  # for llama_dir in "$llama1b" "$llama3b" "$llama8b" "$llama11b" "$llama70b"; do
  for llama_dir in "$llama70b"; do
    LLAMA_DIR=$llama_dir TT_METAL_ENABLE_ERISC_IRAM=1 FAKE_DEVICE=TG pytest -n auto models/demos/llama3_subdevices/demo/demo_decode.py -k "full" --timeout 1000; fail+=$?;
    LLAMA_DIR=$llama_dir TT_METAL_ENABLE_ERISC_IRAM=1 FAKE_DEVICE=TG pytest -n auto models/demos/llama3_subdevices/demo/text_demo.py -k "repeat" --timeout 1000; fail+=$?;
    LLAMA_DIR=$llama_dir TT_METAL_ENABLE_ERISC_IRAM=1 FAKE_DEVICE=TG pytest -n auto models/demos/llama3_subdevices/demo/text_demo.py -k "long" --timeout 1000; fail+=$?;

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

run_tg_falcon7b_tests() {
  fail=0

  # Falcon7B demo (perf verification for 128/1024/2048 seq lens and output token verification)
  pytest -n auto --disable-warnings -q -s --input-method=json --input-path='models/demos/tg/falcon7b/input_data_tg.json' models/demos/tg/falcon7b/demo_tg.py --timeout=1500 ; fail+=$?

  if [[ $fail -ne 0 ]]; then
    echo "LOG_METAL: run_tg_falcon7b_demo_tests failed"
    exit 1
  fi
}

run_tg_demo_tests() {

  if [[ "$1" == "falcon7b" ]]; then
    run_tg_falcon7b_tests
  elif  [[ "$1" == "llama3" ]]; then
    run_tg_llama3_tests
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

  run_tg_demo_tests "$model"

  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

main "$@"
