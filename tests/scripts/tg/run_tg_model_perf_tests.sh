#/bin/bash

run_tg_llm_tests() {
  # Merge all the generated reports
  env python models/perf/merge_perf_results.py; fail+=$?

  if [[ $fail -ne 0 ]]; then
    echo "LOG_METAL: run_tg_model_perf_tests failed"
    exit 1
  fi
}

run_tg_cnn_tests() {
  # Merge all the generated reports
  env python models/perf/merge_perf_results.py; fail+=$?

  if [[ $fail -ne 0 ]]; then
    echo "LOG_METAL: run_tg_model_perf_tests failed"
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
  else
    echo "$pipeline_type is invalid (supported: [cnn_model_perf_tg_device, cnn_model_perf_tg_device])" 2>&1
    exit 1
  fi
}

main "$@"
