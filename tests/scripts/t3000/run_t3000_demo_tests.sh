#!/bin/bash
set -eo pipefail

TT_CACHE_HOME=/mnt/MLPerf/huggingface/tt_cache



run_t3000_resnet50_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_resnet50_tests"

  # resnet50 8 chip demo test - 100 token generation with general weights (env flags set inside the test)
  pytest models/demos/vision/classification/resnet50/ttnn_resnet/tests/test_demo.py --timeout=720 ; fail+=$?

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

  # Sentence BERT demo test
  pytest models/demos/t3000/sentence_bert/demo/demo.py --timeout=600 ; fail+=$?

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

  NO_PROMPT=1 pytest ${test_cmd} --timeout 1200 ; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: ${test_name} $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_sd35large_tests() {
  run_t3000_dit_tests "models/tt_dit/tests/models/sd35/test_pipeline_sd35.py -k 2x4cfg1sp0tp1"
}

run_t3000_flux1_tests() {
  run_t3000_dit_tests "models/tt_dit/tests/models/flux1/test_pipeline_flux1.py -k 2x4sp0tp1-dev"
}

run_t3000_motif_tests() {
  run_t3000_dit_tests "models/tt_dit/tests/models/motif/test_pipeline_motif.py -k 2x4cfg0sp0tp1"
}

run_t3000_qwenimage_tests() {
  run_t3000_dit_tests "models/tt_dit/tests/models/qwenimage/test_pipeline_qwenimage.py -k 2x4"
}




run_t3000_wan22_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_wan22_tests"

  export TT_DIT_CACHE_DIR="/tmp/TT_DIT_CACHE"
  NO_PROMPT=1 pytest models/tt_dit/tests/models/wan2_2/test_pipeline_wan.py -k "2x4sp0tp1 and resolution_480p" --timeout 1500; fail+=$?

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
  pytest models/tt_dit/tests/models/mochi/test_pipeline_mochi.py -k "dit_2x4sp0tp1_vae_1x8sp0tp1" --timeout 1500; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_mochi_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}



run_t3000_tests() {


  # Run resnet50 tests
  run_t3000_resnet50_tests

  # Run sentence bert tests
  run_t3000_sentence_bert_tests


  # Run sd35_large tests
  run_t3000_sd35large_tests

  # Run flux1 tests
  run_t3000_flux1_tests

  # Run motif tests
  run_t3000_motif_tests

  # Run qwenimage tests
  run_t3000_qwenimage_tests


  # Run Wan2.2 tests
  run_t3000_wan22_tests

  # Run mochi tests
  run_t3000_mochi_tests

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
