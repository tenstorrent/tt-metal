#!/bin/bash

run_tg_llama3_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_tg_llama3_tests"

  # Llama3.1-70B
  llama70b=/mnt/MLPerf/tt_dnn-models/llama/Llama3.1-70B-Instruct/
  # Llama3.1-8B
  # llama8b=/mnt/MLPerf/tt_dnn-models/llama/Meta-Llama-3.1-8B-Instruct/
  # Llama3.2-1B
  # llama1b=/mnt/MLPerf/tt_dnn-models/llama/Llama3.2-1B-Instruct/
  # Llama3.2-3B
  # llama3b=/mnt/MLPerf/tt_dnn-models/llama/Llama3.2-3B-Instruct/
  # Llama3.2-11B
  # llama11b=/mnt/MLPerf/tt_dnn-models/llama/Llama3.2-11B-Vision-Instruct/

  # Run all Llama3 tests for 8B, 1B, and 3B weights
  for llama_dir in "$llama70b"; do
    LLAMA_DIR=$llama_dir FAKE_DEVICE=TG pytest -n auto models/demos/llama3/tests/test_llama_model.py -k full ; fail+=$?
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
  # Add tests here
  echo "LOG_METAL: running run_tg_frequent_tests"

  pytest -n auto tests/ttnn/distributed/test_data_parallel_example_TG.py --timeout=900 ; fail+=$?
  pytest -n auto tests/ttnn/distributed/test_multidevice_TG.py --timeout=900 ; fail+=$?
  pytest -n auto tests/ttnn/unit_tests/test_multi_device_trace_TG.py --timeout=900 ; fail+=$?
  pytest -n auto models/demos/tg/llama3_70b/tests/test_llama_mlp_galaxy.py --timeout=300 ; fail+=$?
  pytest -n auto models/demos/tg/llama3_70b/tests/test_llama_attention_galaxy.py --timeout=480 ; fail+=$?
  pytest -n auto models/demos/tg/llama3_70b/tests/test_llama_decoder_galaxy.py --timeout=600 ; fail+=$?
  pytest -n auto models/demos/tg/llama3_70b/tests/test_llama_model_galaxy_ci.py --timeout=800 ; fail+=$?
  pytest -n auto models/demos/tg/resnet50/tests/test_resnet50_performant.py ; fail+=$?
  pytest -n auto tests/ttnn/unit_tests/operations/ccl/test_all_gather_TG_post_commit.py --timeout=300 ; fail+=$?

  if [[ $fail -ne 0 ]]; then
    echo "LOG_METAL: run_tg_frequent_tests failed"
    exit 1
  fi

  # Run llama3 tests
  run_tg_llama3_tests
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
