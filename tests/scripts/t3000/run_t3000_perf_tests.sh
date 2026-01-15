#!/bin/bash
set -eo pipefail

TT_CACHE_HOME=/mnt/MLPerf/huggingface/tt_cache

run_t3000_falcon7b_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_falcon7b_tests"

  pytest -n auto models/demos/falcon7b_common/tests -m "model_perf_t3000" ; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_falcon7b_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_mistral7b_perf_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_mistral7b_perf_tests"

  hf_model=mistralai/Mistral-7B-Instruct-v0.3
  tt_cache_path=$TT_CACHE_HOME/$hf_model
  TT_CACHE_PATH=$tt_cache_path HF_MODEL=$hf_model pytest models/tt_transformers/demo/simple_text_demo.py -k "performance and batch-1" ; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_mistral7b_perf_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_falcon40b_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_falcon40b_tests"

  pytest -n auto models/demos/t3000/falcon40b/tests/test_perf_falcon.py -m "model_perf_t3000" --timeout=600 ; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_falcon40b_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_resnet50_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_resnet50_tests"

  pytest models/demos/ttnn_resnet/tests/test_perf_e2e_resnet50.py -m "model_perf_t3000" ; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_resnet50_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_sentence_bert_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_sentence_bert_tests"

  pytest models/demos/t3000/sentence_bert/tests/test_sentence_bert_e2e_performant.py -m "model_perf_t3000" ; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_sentence_bert_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_dit_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)
  test_name=${FUNCNAME[1]}
  test_cmd=$1

  echo "LOG_METAL: Running ${test_name}"

  pytest ${test_cmd} ; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: ${test_name} $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_gemma3_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  HF_MODEL=/mnt/MLPerf/tt_dnn-models/google/gemma-3-27b-it pytest models/demos/gemma3/tests/test_perf_vision_cross_attention_transformer.py ; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: Gemma3 27B ViT test completed"
  echo "LOG_METAL: run_t3000_gemma3_tests $duration seconds to complete"

  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_gemma3_tests_op_to_op() {
  HF_MODEL=/mnt/MLPerf/tt_dnn-models/google/gemma-3-27b-it pytest models/demos/gemma3/tests/test_vision_cross_attention_transformer_perf_ops.py::test_op_to_op_perf_gemma_vision
}

run_t3000_wan22_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_wan22_tests"

  export TT_DIT_CACHE_DIR="/tmp/TT_DIT_CACHE"
  pytest models/experimental/tt_dit/tests/models/wan2_2/test_performance_wan.py -k "2x4sp0tp1 and resolution_480p"; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_wan22_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_mochi_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_mochi_tests"

  export TT_DIT_CACHE_DIR="/tmp/TT_DIT_CACHE"
  pytest models/experimental/tt_dit/tests/models/mochi/test_performance_mochi.py -k "2x4sp0tp1" --timeout 1800; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_mochi_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_stable_diffusion_35_large_tests() {
  run_t3000_dit_tests "models/experimental/tt_dit/tests/models/sd35/test_performance_sd35.py -k 2x4cfg1sp0tp1"
}

run_t3000_flux1_tests() {
  run_t3000_dit_tests "models/experimental/tt_dit/tests/models/flux1/test_performance_flux1.py -k wh_2x4sp0tp1"
}

run_t3000_motif_tests() {
  run_t3000_dit_tests "models/experimental/tt_dit/tests/models/motif/test_performance_motif.py"
}

run_t3000_qwenimage_tests() {
  run_t3000_dit_tests "models/experimental/tt_dit/tests/models/qwenimage/test_performance_qwenimage.py -k 2x4"
}

run_t3000_model_perf_tests() {
  # Run model performance tests
  run_t3000_sentence_bert_tests
}

fail=0
main() {
  # For CI pipeline - source func commands but don't execute tests if not invoked directly
  if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    echo "Script is being sourced, not executing main function"
    return 0
  fi

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

  if [[ "$pipeline_type" == "model_perf_t3000" ]]; then
    run_t3000_model_perf_tests
  else
    echo "$pipeline_type is invalid (supported: [ccl_perf_t3000_device, model_perf_t3000])" 2>&1
    exit 1
  fi

  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

main "$@"
