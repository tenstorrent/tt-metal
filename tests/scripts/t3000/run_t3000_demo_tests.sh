#!/bin/bash
set -eo pipefail

run_t3000_falcon40b_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_falcon40b_tests"

  # Falcon40B prefill 60 layer end to end with 10 loops; we need 8x8 grid size
  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto models/demos/t3000/falcon40b/tests/ci/test_falcon_end_to_end_60_layer_t3000_prefill_10_loops.py --timeout=720 ; fail+=$?

  # Falcon40B end to end demo (prefill + decode)
  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto models/demos/t3000/falcon40b/tests/test_demo.py ; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_falcon40b_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_llama3_70b_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_llama3_70b_tests"

  LLAMA_DIR=/mnt/MLPerf/tt_dnn-models/llama/Meta-Llama-3.1-70B-Instruct/ WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto models/demos/llama3/demo/demo.py --timeout 600; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_llama3_70b_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_llama3_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_llama3_tests"

  wh_arch_yaml=wormhole_b0_80_arch_eth_dispatch.yaml
  # Llama3.1-8B
  llama8b=/mnt/MLPerf/tt_dnn-models/llama/Meta-Llama-3.1-8B-Instruct/
  # Llama3.2-1B
  llama1b=/mnt/MLPerf/tt_dnn-models/llama/Llama3.2-1B-Instruct/
  # Llama3.2-3B
  llama3b=/mnt/MLPerf/tt_dnn-models/llama/Llama3.2-3B-Instruct/
  # Llama3.2-11B
  llama11b=/mnt/MLPerf/tt_dnn-models/llama/Llama3.2-11B-Vision-Instruct/

  # Run all Llama3 tests for 8B, 1B, and 3B weights
  for llama_dir in "$llama1b" "$llama3b" "$llama8b" "$llama11b"; do
    LLAMA_DIR=$llama_dir WH_ARCH_YAML=$wh_arch_yaml pytest -n auto models/demos/llama3/demo/demo.py --timeout 600; fail+=$?
    echo "LOG_METAL: Llama3 tests for $llama_dir completed"
  done

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_llama3_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_llama3_vision_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_llama3_vision_tests"

  wh_arch_yaml=wormhole_b0_80_arch_eth_dispatch.yaml
  # Llama3.2-11B
  llama11b=/mnt/MLPerf/tt_dnn-models/llama/Llama3.2-11B-Vision-Instruct/
  n300=N300
  t3k=T3K

  # Install Vision-specific packages
  pip install -r models/demos/llama3/requirements.txt

  for fake_device in "$n300" "$t3k"; do
    FAKE_DEVICE=$fake_device LLAMA_DIR=$llama11b WH_ARCH_YAML=$wh_arch_yaml pytest -n auto models/demos/llama3/demo/multimodal_demo_chat.py -k "tt and 1" --timeout 600; fail+=$?
    echo "LOG_METAL: Llama3 vision tests for $fake_device completed"
  done

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_llama3_vision_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_falcon7b_tests(){
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_falcon7b_tests"

  # Falcon7B demo (perf verification for 128/1024/2048 seq lens and output token verification)
  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto --disable-warnings -q -s --input-method=json --input-path='models/demos/t3000/falcon7b/input_data_t3000.json' models/demos/t3000/falcon7b/demo_t3000.py ; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_falcon7b_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_mixtral_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_mixtral8x7b_tests"

  # mixtral8x7b 8 chip demo test - 100 token generation with general weights (env flags set inside the test)
  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto models/demos/t3000/mixtral8x7b/demo/demo.py --timeout=720 ; fail+=$?
  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto models/demos/t3000/mixtral8x7b/demo/demo_with_prefill.py --timeout=720 ; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_mixtral_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_tests() {

  # Run llama3 smaller tests (1B, 3B, 8B, 11B)
  run_t3000_llama3_tests

  # Run llama3_70b tests
  run_t3000_llama3_70b_tests

  # Run falcon40b tests
  run_t3000_falcon40b_tests

  # Run falcon7b tests
  run_t3000_falcon7b_tests

  # Run mixtral tests
  run_t3000_mixtral_tests
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
