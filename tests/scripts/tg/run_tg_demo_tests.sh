#!/bin/bash

run_tg_llama3_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_tg_llama3_tests"

  # Llama3.3-70B
  llama70b=/mnt/MLPerf/tt_dnn-models/llama/Llama3.3-70B-Instruct/

  for llama_dir in "$llama70b"; do
    LLAMA_DIR=$llama_dir FAKE_DEVICE=TG pytest -n auto models/demos/llama3_subdevices/demo/demo_decode.py -k "full" --timeout 1000; fail+=$?;
    LLAMA_DIR=$llama_dir FAKE_DEVICE=TG pytest -n auto models/demos/llama3_subdevices/demo/text_demo.py -k "repeat" --timeout 1000; fail+=$?;
    LLAMA_DIR=$llama_dir FAKE_DEVICE=TG pytest -n auto models/demos/llama3_subdevices/demo/text_demo.py -k "long-context-batch32" --timeout 1000; fail+=$?;
    LLAMA_DIR=$llama_dir FAKE_DEVICE=TG pytest -n auto models/demos/llama3_subdevices/demo/text_demo.py -k "long-context-32k" --timeout 1000; fail+=$?;
    LLAMA_DIR=$llama_dir FAKE_DEVICE=TG pytest -n auto models/demos/llama3_subdevices/demo/text_demo.py -k "pcc-80L" --timeout 1000; fail+=$?;

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

run_tg_llama3_8b_dp_tests() {
  fail=0

  echo "LOG_METAL: Running run_tg_llama3_8b_dp_tests"

  llama8b=/mnt/MLPerf/tt_dnn-models/llama/Meta-Llama-3.1-8B-Instruct/
  LLAMA_DIR=$llama8b MESH_DEVICE=TG pytest models/tt_transformers/demo/simple_text_demo.py --timeout 1000; fail+=$?
  echo "LOG_METAL: Llama3 8B tests for $llama8b completed"

  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_tg_llama3_70b_dp_tests() {
  fail=0

  echo "LOG_METAL: Running run_tg_llama3_70b_dp_tests"

  llama70b=/mnt/MLPerf/tt_dnn-models/llama/Llama3.3-70B-Instruct/
  LLAMA_DIR=$llama70b MESH_DEVICE=TG pytest models/tt_transformers/demo/simple_text_demo.py --timeout 1000; fail+=$?
  echo "LOG_METAL: Llama3 70B tests for $llama70b completed"

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
  elif [[ "$1" == "llama3" ]]; then
    run_tg_llama3_tests
  elif [[ "$1" == "llama3_8b_dp" ]]; then
    run_tg_llama3_8b_dp_tests
  elif [[ "$1" == "llama3_70b_dp" ]]; then
    run_tg_llama3_70b_dp_tests
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
