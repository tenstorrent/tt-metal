#!/bin/bash

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

run_tg_llama_70b_model_perf_tests() {

  # Llama3.3-70B
  llama70b=/mnt/MLPerf/tt_dnn-models/llama/Llama3.3-70B-Instruct/

  echo "LOG_METAL: Running run_tg_llama_70b_model_perf_tests"

  # Run non-overlapped dispatch perf test
  TT_METAL_KERNELS_EARLY_RETURN=1 TT_METAL_ENABLE_ERISC_IRAM=1 FAKE_DEVICE=TG  RING_6U=1 LLAMA_DIR=$llama70b pytest -n auto models/demos/llama3_subdevices/tests/test_decoder_device_perf.py::test_llama_TG_perf_device_non_overlapped_dispatch --timeout=600 ; fail+=$?

  # Run kernel and op to op latency test
  TT_METAL_ENABLE_ERISC_IRAM=1 FAKE_DEVICE=TG  RING_6U=1 LLAMA_DIR=$llama70b pytest -n auto models/demos/llama3_subdevices/tests/test_decoder_device_perf.py::test_llama_TG_perf_device --timeout=600 ; fail+=$?

  if [[ $fail -ne 0 ]]; then
    echo "LOG_METAL: run_tg_llama_70b_model_perf_tests failed"
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

  if [[ "$pipeline_type" == "cnn_model_perf_tg_device" ]]; then
    run_tg_cnn_tests
  elif [[ "$pipeline_type" == "tg_llama_model_perf_tg_device" ]]; then
    run_tg_llama_70b_model_perf_tests
  else
    echo "$pipeline_type is invalid (supported: [cnn_model_perf_tg_device, tg_llama_model_perf_tg_device])" 2>&1
    exit 1
  fi
}

main "$@"
