#!/bin/bash
set -eo pipefail

TT_CACHE_HOME=/mnt/MLPerf/huggingface/tt_cache

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

  llama70b=meta-llama/Llama-3.1-70B-Instruct
  tt_cache_llama70b=$TT_CACHE_HOME/$llama70b

  HF_MODEL=$llama70b TT_CACHE_PATH=$tt_cache_llama70b pytest -n auto models/tt_transformers/demo/simple_text_demo.py --timeout 1800 -k "not performance-ci-stress-1"; fail+=$?


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
  llama8b=meta-llama/Llama-3.1-8B-Instruct
  # Llama3.2-1B
  llama1b=meta-llama/Llama-3.2-1B-Instruct
  # Llama3.2-3B
  llama3b=meta-llama/Llama-3.2-3B-Instruct
  # Llama3.2-11B
  llama11b=meta-llama/Llama-3.2-11B-Vision-Instruct

  # Run all Llama3 tests for 8B, 1B, and 3B weights
  for hf_model in "$llama1b" "$llama3b" "$llama8b" "$llama11b"; do
    tt_cache=$TT_CACHE_HOME/$hf_model
    HF_MODEL=$hf_model TT_CACHE_PATH=$tt_cache pytest -n auto models/tt_transformers/demo/simple_text_demo.py --timeout 600 -k "not performance-ci-stress-1"; fail+=$?
    echo "LOG_METAL: Llama3 tests for $hf_model completed"
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

  echo "LOG_METAL: Running run_t3000_qwen25_tests"
  qwen25_7b=Qwen/Qwen2.5-7B-Instruct
  tt_cache_7b=$TT_CACHE_HOME/$qwen25_7b
  qwen25_72b=Qwen/Qwen2.5-72B-Instruct
  tt_cache_72b=$TT_CACHE_HOME/$qwen25_72b
  qwen25_coder_32b=Qwen/Qwen2.5-Coder-32B
  tt_cache_coder_32b=$TT_CACHE_HOME/$qwen25_coder_32b

  MESH_DEVICE=N300 HF_MODEL=$qwen25_7b TT_CACHE_PATH=$tt_cache_7b pytest models/tt_transformers/demo/simple_text_demo.py -k "not performance-ci-stress-1" --timeout 600 || fail+=$?
  HF_MODEL=$qwen25_72b TT_CACHE_PATH=$tt_cache_72b pytest models/tt_transformers/demo/simple_text_demo.py -k "not performance-ci-stress-1" --timeout 1800 || fail+=$?
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
  uv pip install -r models/demos/qwen25_vl/requirements.txt

  # export PYTEST_ADDOPTS for concise pytest output
  export PYTEST_ADDOPTS="--tb=short"

  # Qwen2.5-VL-32B
  qwen25_vl_32b=Qwen/Qwen2.5-VL-32B-Instruct
  tt_cache_32b=$TT_CACHE_HOME/$qwen25_vl_32b
  MESH_DEVICE=T3K HF_MODEL=$qwen25_vl_32b TT_CACHE_PATH=$tt_cache_32b pytest models/demos/qwen25_vl/demo/demo.py --timeout 600 || fail=1

  # Qwen2.5-VL-72B
  qwen25_vl_72b=Qwen/Qwen2.5-VL-72B-Instruct
  tt_cache_72b=$TT_CACHE_HOME/$qwen25_vl_72b
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

  echo "LOG_METAL: Running run_t3000_qwen3_tests"
  qwen32b=Qwen/Qwen3-32B
  tt_cache_qwen32b=$TT_CACHE_HOME/$qwen32b

  # Run Qwen3.32B with max_seq_len 32k
  HF_MODEL=$qwen32b TT_CACHE_PATH=$tt_cache_qwen32b pytest models/tt_transformers/demo/simple_text_demo.py --max_seq_len 32768 --timeout 1800 || fail+=$?

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
  llama11b=meta-llama/Llama-3.2-11B-Vision-Instruct
  tt_cache_llama11b=$TT_CACHE_HOME/$llama11b
  n300=N300
  t3k=T3K

  for mesh_device in "$n300" "$t3k"; do
    MESH_DEVICE=$mesh_device HF_MODEL=$llama11b TT_CACHE_PATH=$tt_cache_llama11b \
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

  export PYTEST_ADDOPTS="--tb=short"

  # Llama3.2-90B
  llama90b=meta-llama/Llama-3.2-90B-Vision-Instruct
  tt_cache_llama90b=$TT_CACHE_HOME/$llama90b
  mesh_device=T3K

  MESH_DEVICE=$mesh_device HF_MODEL=$llama90b TT_CACHE_PATH=$tt_cache_llama90b pytest -n auto models/tt_transformers/demo/simple_vision_demo.py -k "batch1-trace" --timeout 1000 || fail=1
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

  hf_model="mistralai/Mistral-7B-Instruct-v0.3"
  tt_cache_path=$TT_CACHE_HOME/$hf_model
  TT_CACHE_PATH=$tt_cache_path HF_MODEL=$hf_model pytest models/tt_transformers/demo/simple_text_demo.py --timeout 10800 -k "not performance-ci-stress-1"
  # test max_seq_len overrides
  TT_CACHE_PATH=$tt_cache_path HF_MODEL=$hf_model pytest models/tt_transformers/demo/simple_text_demo.py --timeout 120 -k "ci-long-context-16k" --max_seq_len=16384

}

run_t3000_mixtral_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_tt-transformer_mixtral8x7b_tests"

  # mixtral8x7b 8 chip demo test - 100 token generation with general weights (env flags set inside the test)
  # pytest -n auto models/demos/t3000/mixtral8x7b/demo/demo.py --timeout=720 ; fail+=$?
  # pytest -n auto models/demos/t3000/mixtral8x7b/demo/demo_with_prefill.py --timeout=720 ; fail+=$?
  mixtral8x7=mistralai/Mixtral-8x7B-v0.1
  tt_cache_path=$TT_CACHE_HOME/$mixtral8x7

  CI=true TT_CACHE_PATH=$tt_cache_path HF_MODEL=$mixtral8x7 pytest models/tt_transformers/demo/simple_text_demo.py -k "not performance-ci-stress-1" --timeout=3600 ; fail+=$?

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
  pytest -n auto models/demos/ttnn_resnet/tests/test_demo.py --timeout=720 ; fail+=$?

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

run_t3000_dit_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)
  test_name=${FUNCNAME[1]}
  test_cmd=$1

  echo "LOG_METAL: Running ${test_name}"

  NO_PROMPT=1 pytest -n auto ${test_cmd} --timeout 1200 ; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: ${test_name} $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_sd35large_tests() {
  run_t3000_dit_tests "models/experimental/tt_dit/tests/models/sd35/test_pipeline_sd35.py -k 2x4cfg1sp0tp1"
}

run_t3000_flux1_tests() {
  run_t3000_dit_tests "models/experimental/tt_dit/tests/models/flux1/test_pipeline_flux1.py -k 2x4sp0tp1-dev"
}

run_t3000_motif_tests() {
  run_t3000_dit_tests "models/experimental/tt_dit/tests/models/motif/test_pipeline_motif.py -k 2x4cfg0sp0tp1"
}

run_t3000_qwenimage_tests() {
  run_t3000_dit_tests "models/experimental/tt_dit/tests/models/qwenimage/test_pipeline_qwenimage.py -k 2x4"
}


run_t3000_gemma3_tests() {
  # Record the start time
  start_time=$(date +%s)
  gemma27b=google/gemma-3-27b-it
  tt_cache_gemma27b=$TT_CACHE_HOME/$gemma27b

  HF_MODEL=$gemma27b TT_CACHE_PATH=$tt_cache_gemma27b pytest models/demos/gemma3/demo/text_demo.py -k "performance and ci-1"
  echo "LOG_METAL: Gemma3 27B tests completed (text only)"
  HF_MODEL=$gemma27b TT_CACHE_PATH=$tt_cache_gemma27b pytest models/demos/gemma3/demo/vision_demo.py -k "performance and batch1-multi-image-trace"
  echo "LOG_METAL: Gemma3 27B tests completed (text and vision)"
  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_gemma3_tests $duration seconds to complete"
}

run_t3000_whisper_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_whisper_tests"

  pytest -n auto models/demos/whisper/demo/demo.py::test_demo_for_conditional_generation --timeout=600 ; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_whisper_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_wan22_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_wan22_tests"

  export TT_DIT_CACHE_DIR="/tmp/TT_DIT_CACHE"
  pytest -n auto models/experimental/tt_dit/tests/models/wan2_2/test_pipeline_wan.py -k "2x4sp0tp1 and resolution_480p" --timeout 1500; fail+=$?

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
  pytest -n auto models/experimental/tt_dit/tests/models/mochi/test_pipeline_mochi.py -k "dit_2x4sp0tp1_vae_1x8sp0tp1" --timeout 1500; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_mochi_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_gpt_oss_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  # Install gpt-oss requirements
  uv pip install -r models/demos/gpt_oss/requirements.txt

  # Test GPT-OSS 20B model
  HF_MODEL=openai/gpt-oss-20b TT_CACHE_PATH=$TT_CACHE_HOME/openai--gpt-oss-20b pytest models/demos/gpt_oss/demo/text_demo.py -k "1x8"
  echo "LOG_METAL: GPT-OSS 20B tests completed"

  # Test GPT-OSS 120B model
  HF_MODEL=openai/gpt-oss-120b TT_CACHE_PATH=$TT_CACHE_HOME/openai--gpt-oss-120b pytest models/demos/gpt_oss/demo/text_demo.py -k "1x8"
  echo "LOG_METAL: GPT-OSS 120B tests completed"

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_gpt_oss_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_tests() {
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

  # Run flux1 tests
  run_t3000_flux1_tests

  # Run motif tests
  run_t3000_motif_tests

  # Run qwenimage tests
  run_t3000_qwenimage_tests

  # Run gemma3 tests
  run_t3000_gemma3_tests

  # Run whisper tests
  run_t3000_whisper_tests

  # Run Wan2.2 tests
  run_t3000_wan22_tests

  # Run mochi tests
  run_t3000_mochi_tests

  # Run gpt-oss tests
  run_t3000_gpt_oss_tests
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
