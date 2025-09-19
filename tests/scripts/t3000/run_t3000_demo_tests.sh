#!/bin/bash
set -eo pipefail

run_t3000_falcon40b_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_falcon40b_tests"

  # Falcon40B prefill 60 layer end to end with 10 loops; we need 8x8 grid size
  pytest -n auto models/demos/t3000/falcon40b/tests/ci/test_falcon_end_to_end_60_layer_t3000_prefill_10_loops.py --timeout=720 ; fail+=$?

  # Falcon40B end to end demo (prefill + decode)
  pytest -n auto models/demos/t3000/falcon40b/tests/test_demo.py ; fail+=$?

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

  LLAMA_DIR=/mnt/MLPerf/tt_dnn-models/llama/Llama3.1-70B-Instruct pytest -n auto models/tt_transformers/demo/simple_text_demo.py --timeout 1800 -k "not performance-ci-stress-1"; fail+=$?


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

  # Llama3.1-8B
  llama8b=/mnt/MLPerf/tt_dnn-models/llama/Meta-Llama-3.1-8B-Instruct
  # Llama3.2-1B
  llama1b=/mnt/MLPerf/tt_dnn-models/llama/Llama3.2-1B-Instruct
  # Llama3.2-3B
  llama3b=/mnt/MLPerf/tt_dnn-models/llama/Llama3.2-3B-Instruct
  # Llama3.2-11B
  llama11b=/mnt/MLPerf/tt_dnn-models/llama/Llama3.2-11B-Vision-Instruct

  # Run all Llama3 tests for 8B, 1B, and 3B weights
  for llama_dir in "$llama1b" "$llama3b" "$llama8b" "$llama11b"; do
    LLAMA_DIR=$llama_dir pytest -n auto models/tt_transformers/demo/simple_text_demo.py --timeout 600 -k "not performance-ci-stress-1"; fail+=$?
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

run_t3000_qwen25_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  export PYTEST_ADDOPTS="--tb=short"
  export HF_HOME=/mnt/MLPerf/huggingface

  echo "LOG_METAL: Running run_t3000_qwen25_tests"
  qwen25_7b=Qwen/Qwen2.5-7B-Instruct
  tt_cache_7b=$HF_HOME/tt_cache/Qwen--Qwen2.5-7B-Instruct
  qwen25_72b=Qwen/Qwen2.5-72B-Instruct
  tt_cache_72b=$HF_HOME/tt_cache/Qwen--Qwen2.5-72B-Instruct
  qwen25_coder_32b=Qwen/Qwen2.5-Coder-32B
  tt_cache_coder_32b=$HF_HOME/tt_cache/Qwen--Qwen2.5-Coder-32B

  MESH_DEVICE=N300 HF_MODEL=$qwen25_7b TT_CACHE_PATH=$tt_cache_7b pytest models/tt_transformers/demo/simple_text_demo.py -k "not performance-ci-stress-1" --timeout 600 || fail+=$?
  HF_MODEL=$qwen25_72b TT_CACHE_PATH=$tt_cache_72b pytest models/tt_transformers/demo/simple_text_demo.py -k "not performance-ci-stress-1" --timeout 1800 || fail+=$?
  pip install -r models/tt_transformers/requirements.txt
  HF_MODEL=$qwen25_coder_32b TT_CACHE_PATH=$tt_cache_coder_32b pytest models/tt_transformers/demo/simple_text_demo.py -k "not performance-ci-stress-1" --timeout 1800 || fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_qwen25_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_qwen25_vl_tests() {
  fail=0

  # install qwen25_vl requirements
  pip install -r models/demos/qwen25_vl/requirements.txt

  # export PYTEST_ADDOPTS for concise pytest output
  export PYTEST_ADDOPTS="--tb=short"
  export HF_HOME=/mnt/MLPerf/huggingface

  # Qwen2.5-VL-32B
  qwen25_vl_32b=Qwen/Qwen2.5-VL-32B-Instruct
  tt_cache_32b=$HF_HOME/tt_cache/Qwen--Qwen2.5-VL-32B-Instruct
  MESH_DEVICE=T3K HF_MODEL=$qwen25_vl_32b TT_CACHE_PATH=$tt_cache_32b pytest models/demos/qwen25_vl/demo/demo.py --timeout 600 || fail=1

  # Qwen2.5-VL-72B
  qwen25_vl_72b=Qwen/Qwen2.5-VL-72B-Instruct
  tt_cache_72b=$HF_HOME/tt_cache/Qwen--Qwen2.5-VL-72B-Instruct
  MESH_DEVICE=T3K HF_MODEL=$qwen25_vl_72b TT_CACHE_PATH=$tt_cache_72b pytest models/demos/qwen25_vl/demo/demo.py --timeout 900 || fail=1

  echo "LOG_METAL: Tests for Qwen2.5-VL-32B and Qwen2.5-VL-72B on T3K completed"

  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_qwen3_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Warning: updating transformers version. Make sure this is the last-run test."
  echo "LOG_METAL: Remove this when https://github.com/tenstorrent/tt-metal/pull/22608 merges."
  pip install -r models/tt_transformers/requirements.txt

  echo "LOG_METAL: Running run_t3000_qwen3_tests"
  qwen32b=/mnt/MLPerf/tt_dnn-models/qwen/Qwen3-32B

  HF_MODEL=$qwen32b pytest models/tt_transformers/demo/simple_text_demo.py --timeout 1800 || fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_qwen3_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_llama3_vision_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_llama3_vision_tests"

  # Llama3.2-11B
  llama11b=/mnt/MLPerf/tt_dnn-models/llama/Llama3.2-11B-Vision-Instruct
  n300=N300
  t3k=T3K

  for mesh_device in "$n300" "$t3k"; do
    MESH_DEVICE=$mesh_device LLAMA_DIR=$llama11b \
    pytest -n auto models/tt_transformers/demo/simple_vision_demo.py -k "not batch1-notrace" --timeout 900; fail+=$?
    echo "LOG_METAL: Llama3 vision tests for $mesh_device completed"
  done

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_llama3_vision_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_llama3_90b_vision_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_llama3_90b_vision_tests"

  # Llama3.2-90B
  llama90b=/mnt/MLPerf/tt_dnn-models/llama/Llama3.2-90B-Vision-Instruct
  mesh_device=T3K

  MESH_DEVICE=$mesh_device LLAMA_DIR=$llama90b pytest -n auto models/tt_transformers/demo/simple_vision_demo.py -k "batch1-notrace" --timeout 1200; fail+=$?
  echo "LOG_METAL: Llama3.2-90B vision tests for $mesh_device completed"

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_llama3_90b_vision_tests $duration seconds to complete"
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
  pytest -n auto --disable-warnings -q -s --input-method=json --input-path='models/demos/t3000/falcon7b/input_data_t3000.json' models/demos/t3000/falcon7b/demo_t3000.py ; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_falcon7b_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_mistral_tests() {

  echo "LOG_METAL: Running run_t3000_mistral_demo_tests"

  tt_cache_path="/mnt/MLPerf/tt_dnn-models/Mistral/TT_CACHE/Mistral-7B-Instruct-v0.3"
  hf_model="/mnt/MLPerf/tt_dnn-models/Mistral/hub/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/e0bc86c23ce5aae1db576c8cca6f06f1f73af2db"
  TT_CACHE_PATH=$tt_cache_path HF_MODEL=$hf_model pytest models/tt_transformers/demo/simple_text_demo.py --timeout 10800 -k "not performance-ci-stress-1"

}

run_t3000_mixtral_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_tt-transformer_mixtral8x7b_tests"

  # mixtral8x7b 8 chip demo test - 100 token generation with general weights (env flags set inside the test)
  # pytest -n auto models/demos/t3000/mixtral8x7b/demo/demo.py --timeout=720 ; fail+=$?
  # pytest -n auto models/demos/t3000/mixtral8x7b/demo/demo_with_prefill.py --timeout=720 ; fail+=$?
  mixtral8x7=/mnt/MLPerf/huggingface/hub/models--mistralai--Mixtral-8x7B-v0.1/snapshots/fc7ac94680e38d7348cfa806e51218e6273104b0

  CI=true HF_MODEL=$mixtral8x7 pytest -n auto models/tt_transformers/demo/simple_text_demo.py -k "not performance-ci-stress-1" --timeout=3600 ; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_mixtral_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_resnet50_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_resnet50_tests"

  # resnet50 8 chip demo test - 100 token generation with general weights (env flags set inside the test)
  pytest -n auto models/demos/t3000/resnet50/demo/demo.py --timeout=720 ; fail+=$?

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
  pytest -n auto models/demos/t3000/sentence_bert/demo/demo.py --timeout=600 ; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_sentence_bert_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}


run_t3000_sd35large_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_sd35large_tests"

  #Cache path
  NO_PROMPT=1 pytest -n auto models/experimental/tt_dit/tests/models/test_pipeline_sd35.py -k "2x4cfg1sp0tp1" --timeout 600 ; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_sd35large_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_llama3_load_checkpoints_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_load_checkpoints_tests"

  # Llama3.1-70B weights
  llama70b=/mnt/MLPerf/tt_dnn-models/llama/Llama3.1-70B-Instruct/original_weights/
  # Llama3.2-90B weights
  llama90b=/mnt/MLPerf/tt_dnn-models/llama/Llama3.2-90B-Vision-Instruct

  for llama_dir in "$llama70b" "$llama90b"; do
    LLAMA_DIR=$llama_dir pytest -n auto models/tt_transformers/tests/test_load_checkpoints.py --timeout=1800; fail+=$?
    echo "LOG_METAL: Llama3 load checkpoints tests for $llama_dir completed"
  done

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_load_checkpoints_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_gemma3_tests() {
  # Record the start time
  start_time=$(date +%s)
  HF_MODEL=/mnt/MLPerf/tt_dnn-models/google/gemma-3-27b-it pytest models/demos/gemma3/demo/text_demo.py -k "performance and ci-1"
  echo "LOG_METAL: Gemma3 27B tests completed (text only)"
  HF_MODEL=/mnt/MLPerf/tt_dnn-models/google/gemma-3-27b-it pytest models/demos/gemma3/demo/vision_demo.py -k "performance and batch1-multi-image-trace"
  echo "LOG_METAL: Gemma3 27B tests completed (text and vision)"
  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_gemma3_tests $duration seconds to complete"
}

run_t3000_tests() {
  # Run llama3 load checkpoints tests
  run_t3000_llama3_load_checkpoints_tests

  # Run llama3 smaller tests (1B, 3B, 8B, 11B)
  run_t3000_llama3_tests

  # Run llama3 vision tests
  run_t3000_llama3_vision_tests

  # Run llama3_90b vision tests
  run_t3000_llama3_90b_vision_tests

  # Run llama3_70b tests
  run_t3000_llama3_70b_tests

  # Run falcon40b tests
  run_t3000_falcon40b_tests

  # Run falcon7b tests
  run_t3000_falcon7b_tests

  # Run mistral tests
  run_t3000_mistral_tests

  # Run mixtral tests
  run_t3000_mixtral_tests

  # Run resnet50 tests
  run_t3000_resnet50_tests

  # Run sentence bert tests
  run_t3000_sentence_bert_tests

  # Run qwen25 tests
  run_t3000_qwen25_tests

  # Run qwen3 tests
  run_t3000_qwen3_tests

  # Run sd35_large tests
  run_t3000_sd35large_tests

  # Run gemma3 tests
  run_t3000_gemma3_tests
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
