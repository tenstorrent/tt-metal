#!/bin/bash

run_t3000_falcon7b_perplexity_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_falcon7b_perplexity_tests"

  # Falcon7B perplexity tests
  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto models/demos/falcon7b_common/tests/perplexity/test_perplexity_falcon.py --timeout=1500 ; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_falcon7b_perplexity_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_falcon40b_perplexity_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_falcon40b_perplexity_tests"

  # Falcon40B perplexity tests
  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto models/demos/t3000/falcon40b/tests/test_perplexity_falcon.py --timeout=2100 ; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_falcon40b_perplexity_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_llama70b_perplexity_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_llama70b_perplexity_tests"

  # Llama-70B perplexity tests
  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto models/demos/t3000/llama2_70b/demo/eval_t3000.py --timeout=7200 ; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_llama70b_perplexity_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_mixtral8x7b_perplexity_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_mixtral8x7b_perplexity_tests"

  # Mixtral8x7B perplexity tests
  # WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto models/demos/t3000/mixtral8x7b/tests/test_mixtral_perplexity.py --timeout=3600 ; fail+=$?
  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto models/demos/t3000/mixtral8x7b/tests/test_mixtral_topk.py --timeout=3600 ; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_mixtral8x7b_perplexity_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_llama3_perplexity_tests_single_card() {
  # Split long set of tests into two groups
  # This one runs all the N150 and N300 tests
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_llama3_perplexity_tests_single_card"

  llama1b=/mnt/MLPerf/tt_dnn-models/llama/Llama3.2-1B-Instruct/
  llama3b=/mnt/MLPerf/tt_dnn-models/llama/Llama3.2-3B-Instruct/
  llama8b=/mnt/MLPerf/tt_dnn-models/llama/Meta-Llama-3.1-8B-Instruct/
  llama11b=/mnt/MLPerf/tt_dnn-models/llama/Llama3.2-11B-Vision-Instruct/

  for FAKE_DEVICE in N150 N300; do
    for LLAMA_DIR in "$llama1b" "$llama3b" "$llama8b" "$llama11b"; do
      FAKE_DEVICE=FAKE_DEVICE LLAMA_DIR=$LLAMA_DIR WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto models/demos/llama3/tests/test_llama_accuracy.py --timeout=3600 ; fail+=$?
    done
  done

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_llama3_perplexity_tests_single_card $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_llama3_perplexity_tests_t3000() {
  # Split long set of tests into two groups
  # This one runs all the T3K tests
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_llama3_perplexity_tests_t3000"

  llama1b=/mnt/MLPerf/tt_dnn-models/llama/Llama3.2-1B-Instruct/
  llama3b=/mnt/MLPerf/tt_dnn-models/llama/Llama3.2-3B-Instruct/
  llama8b=/mnt/MLPerf/tt_dnn-models/llama/Meta-Llama-3.1-8B-Instruct/
  llama11b=/mnt/MLPerf/tt_dnn-models/llama/Llama3.2-11B-Vision-Instruct/
  llama70b=/mnt/MLPerf/tt_dnn-models/llama/Meta-Llama-3.1-70B-Instruct/

  for FAKE_DEVICE in T3K; do
    for LLAMA_DIR in "$llama1b" "$llama3b" "$llama8b" "$llama11b" "$llama70b"; do
      FAKE_DEVICE=FAKE_DEVICE LLAMA_DIR=$LLAMA_DIR WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto models/demos/llama3/tests/test_llama_accuracy.py --timeout=3600 ; fail+=$?
    done
  done

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_llama3_perplexity_tests_t3000 $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_tests() {
  # Run Falcon-7B perplexity tests
  run_t3000_falcon7b_perplexity_tests

  # Run Falcon-40B perplexity tests
  run_t3000_falcon40b_perplexity_tests

  # Run Llama-70B perplexity tests
  run_t3000_llama70b_perplexity_tests

  # Run Mixtral8x7B perplexity tests
  run_t3000_mixtral8x7b_perplexity_tests

  # Run llama3 perplexity tests
  run_t3000_llama3_perplexity_tests_single_card
  run_t3000_llama3_perplexity_tests_t3000
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

  run_t3000_perplexity_tests

  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

main "$@"
