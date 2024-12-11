#!/bin/bash
set -eo pipefail

run_tg_llama3-small_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_tg_llama3-small_tests"

  wh_arch_yaml=wormhole_b0_80_arch_eth_dispatch.yaml
  # Llama3.2-1B
  llama1b=/mnt/MLPerf/tt_dnn-models/llama/Llama3.2-1B-Instruct/
  # Llama3.2-3B
  llama3b=/mnt/MLPerf/tt_dnn-models/llama/Llama3.2-3B-Instruct/
  # Llama3.1-8B
  llama8b=/mnt/MLPerf/tt_dnn-models/llama/Meta-Llama-3.1-8B-Instruct/

  # Run all Llama3 tests for 1B, 3B and 8B weights
  for llama_dir in "$llama1b" "$llama3b" "$llama8b"; do
    LLAMA_DIR=$llama_dir FAKE_DEVICE=TG pytest -n auto models/demos/llama3/tests/test_llama_attention.py ; fail+=$?
    LLAMA_DIR=$llama_dir FAKE_DEVICE=TG pytest -n auto models/demos/llama3/tests/test_llama_attention_prefill.py ; fail+=$?
    LLAMA_DIR=$llama_dir FAKE_DEVICE=TG pytest -n auto models/demos/llama3/tests/test_llama_embedding.py ; fail+=$?
    LLAMA_DIR=$llama_dir FAKE_DEVICE=TG pytest -n auto models/demos/llama3/tests/test_llama_mlp.py ; fail+=$?
    LLAMA_DIR=$llama_dir FAKE_DEVICE=TG pytest -n auto models/demos/llama3/tests/test_llama_rms_norm.py ; fail+=$?
    LLAMA_DIR=$llama_dir FAKE_DEVICE=TG pytest -n auto models/demos/llama3/tests/test_llama_decoder.py ; fail+=$?
    LLAMA_DIR=$llama_dir FAKE_DEVICE=TG pytest -n auto models/demos/llama3/tests/test_llama_decoder_prefill.py ; fail+=$?
    echo "LOG_METAL: Llama3 tests for $llama_dir completed"
  done

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_tg_llama3-small_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_tg_llama3.2-11b_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_tg_llama3.2-11b_tests"

  # Llama3.2-11B weights
  llama11b=/mnt/MLPerf/tt_dnn-models/llama/Llama3.2-11B-Vision-Instruct/

  LLAMA_DIR=$llama11b FAKE_DEVICE=TG pytest -n auto models/demos/llama3/tests/test_llama_attention.py ; fail+=$?
  LLAMA_DIR=$llama11b FAKE_DEVICE=TG pytest -n auto models/demos/llama3/tests/test_llama_attention_prefill.py ; fail+=$?
  LLAMA_DIR=$llama11b FAKE_DEVICE=TG pytest -n auto models/demos/llama3/tests/test_llama_embedding.py ; fail+=$?
  LLAMA_DIR=$llama11b FAKE_DEVICE=TG pytest -n auto models/demos/llama3/tests/test_llama_mlp.py ; fail+=$?
  LLAMA_DIR=$llama11b FAKE_DEVICE=TG pytest -n auto models/demos/llama3/tests/test_llama_rms_norm.py ; fail+=$?
  LLAMA_DIR=$llama11b FAKE_DEVICE=TG pytest -n auto models/demos/llama3/tests/test_llama_decoder.py ; fail+=$?
  LLAMA_DIR=$llama11b FAKE_DEVICE=TG pytest -n auto models/demos/llama3/tests/test_llama_decoder_prefill.py ; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_tg_llama3.2-11b_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_tg_llama3.1-70b_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_tg_llama3.1-70b_tests"

  # Llama3.1-70B weights
  llama70b=/mnt/MLPerf/tt_dnn-models/llama/Llama3.1-70B-Instruct/

  LLAMA_DIR=$llama70b FAKE_DEVICE=TG pytest -n auto models/demos/llama3/tests/test_llama_attention.py ; fail+=$?
  LLAMA_DIR=$llama70b FAKE_DEVICE=TG pytest -n auto models/demos/llama3/tests/test_llama_attention_prefill.py ; fail+=$?
  LLAMA_DIR=$llama70b FAKE_DEVICE=TG pytest -n auto models/demos/llama3/tests/test_llama_embedding.py ; fail+=$?
  LLAMA_DIR=$llama70b FAKE_DEVICE=TG pytest -n auto models/demos/llama3/tests/test_llama_mlp.py ; fail+=$?
  LLAMA_DIR=$llama70b FAKE_DEVICE=TG pytest -n auto models/demos/llama3/tests/test_llama_rms_norm.py ; fail+=$?
  LLAMA_DIR=$llama70b FAKE_DEVICE=TG pytest -n auto models/demos/llama3/tests/test_llama_decoder.py ; fail+=$?
  LLAMA_DIR=$llama70b FAKE_DEVICE=TG pytest -n auto models/demos/llama3/tests/test_llama_decoder_prefill.py ; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_tg_llama3.1-70b_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_tg_tests() {

  echo "LOG_METAL: running run_tg_unit_tests"

  TT_METAL_ENABLE_REMOTE_CHIP=1 ./build/test/tt_metal/unit_tests_dispatch --gtest_filter="CommandQueueSingleCard*Fixture.*"
  ./build/test/ttnn/galaxy_unit_tests_ttnn
  TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/unit_tests_device --gtest_filter="GalaxyFixture.*:TGFixture.*"
  ./build/test/tt_metal/unit_tests_device --gtest_filter="GalaxyFixture.*:TGFixture.*"
  TT_METAL_GTEST_NUM_HW_CQS=2 ./build/test/tt_metal/unit_tests_dispatch --gtest_filter="MultiCommandQueueMultiDevice*Fixture.*"

  run_tg_llama3.1-70b_tests
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

  # Run all tests
  cd $TT_METAL_HOME
  export PYTHONPATH=$TT_METAL_HOME

  run_tg_tests
}

main "$@"
