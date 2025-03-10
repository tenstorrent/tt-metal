#!/bin/bash

run_tg_llm_tests() {

  echo "LOG_METAL: Running run_t3000_llama2_70b_tests"
  pytest -n auto models/demos/t3000/llama2_70b/tests/test_llama_perf_decode.py -m "model_perf_t3000" --timeout=600 ; fail+=$?

  # Merge all the generated reports
  env python3 models/perf/merge_perf_results.py; fail+=$?

  if [[ $fail -ne 0 ]]; then
    echo "LOG_METAL: run_tg_model_perf_tests failed"
    exit 1
  fi
}

run_tg_cnn_tests() {

  echo "LOG_METAL: Running run_tg_resnet50_tests"
  env pytest -n auto models/demos/tg/resnet50/tests/test_perf_e2e_resnet50.py -m "model_perf_tg" ; fail+=$?

  # Merge all the generated reports
  env python3 models/perf/merge_perf_results.py; fail+=$?

  if [[ $fail -ne 0 ]]; then
    echo "LOG_METAL: run_tg_model_perf_tests failed"
    exit 1
  fi
}


run_llama_tg_perf_unit_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_llama_tg_perf_unit_tests"

  pytest models/tt_transformers/tests/test_ccl_async_perf_TG_llama.py --timeout 900; fail+=$?
  pytest models/tt_transformers/tests/test_ring_matmul_perf_TG_llama.py --timeout 900; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_llama_tg_perf_unit_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

main() {
  # Parse the arguments
  while [[ $# -gt 0 ]]; do
    case $1 in
      --pipeline-type)
        pipeline_type=$2
        shift
        ;;
      *)
        echo "Unknown option: $1"
        exit 1
        ;;
    esac
    shift
  done

  if [[ -z "$TT_METAL_HOME" ]]; then
    echo "Must provide TT_METAL_HOME in environment" 1>&2
    exit 1
  fi

  if [[ -z "$ARCH_NAME" ]]; then
    echo "Must provide ARCH_NAME in environment" 1>&2
    exit 1
  fi

  if [[ -z "$pipeline_type" ]]; then
    echo "--pipeline-type cannot be empty" 1>&2
    exit 1
  fi

  # Run all tests
  cd $TT_METAL_HOME
  export PYTHONPATH=$TT_METAL_HOME

  if [[ "$pipeline_type" == "llm_model_perf_tg_device" ]]; then
    run_tg_llm_tests
  elif [[ "$pipeline_type" == "cnn_model_perf_tg_device" ]]; then
    run_tg_cnn_tests
  elif [[ "$pipeline_type" == "llama_tg_perf_unit_tests" ]]; then
    run_llama_tg_perf_unit_tests
  else
    echo "$pipeline_type is invalid (supported: [cnn_model_perf_tg_device, cnn_model_perf_tg_device])" 2>&1
    exit 1
  fi
}

main "$@"
