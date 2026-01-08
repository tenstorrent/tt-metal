#!/bin/bash

TT_CACHE_HOME=/mnt/MLPerf/huggingface/tt_cache

run_t3000_falcon40b_perplexity_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_falcon40b_perplexity_tests"

  # Falcon40B perplexity tests
  pytest -n auto models/demos/t3000/falcon40b/tests/test_perplexity_falcon.py --timeout=2100 ; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_falcon40b_perplexity_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_llama70b_perplexity_tests() {
  # TODO: rewrite test to use HF reference: the code need to be rewritten completely to use common classes and functions instead of custom copies working only with Meta reference

  echo "LOG_METAL: Checking number of devices"
  python3 -c "import ttnn; print('Number of devices:', ttnn.get_num_devices())"

  fail=0
  # Record the start time
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_llama70b_perplexity_tests"

  # Llama-70B perplexity tests
  pytest -n auto models/demos/t3000/llama2_70b/demo/eval_t3000.py --timeout=7200 ; fail+=$?

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
  # pytest -n auto models/demos/t3000/mixtral8x7b/tests/test_mixtral_perplexity.py --timeout=3600 ; fail+=$?
  pytest -n auto models/demos/t3000/mixtral8x7b/tests/test_mixtral_topk.py --timeout=3600 ; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_mixtral8x7b_perplexity_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_llama3_perplexity_tests_single_card() {

  echo "LOG_METAL: Checking number of devices"
  python3 -c "import ttnn; print('Number of devices:', ttnn.get_num_devices())"

  # Split long set of tests into two groups
  # This one runs all the N150 and N300 tests spoofed on a T3k
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_llama3_perplexity_tests_single_card"

  llama1b=meta-llama/Llama-3.2-1B-Instruct
  llama3b=meta-llama/Llama-3.2-3B-Instruct
  llama8b=meta-llama/Llama-3.1-8B-Instruct
  llama11b=meta-llama/Llama-3.2-11B-Vision-Instruct

  for MESH_DEVICE in N150 N300; do
    for hf_model in "$llama1b" "$llama3b" "$llama8b"; do
      tt_cache=$TT_CACHE_HOME/$hf_model
      MESH_DEVICE=$MESH_DEVICE HF_MODEL=$hf_model TT_CACHE_PATH=$tt_cache pytest -n auto models/tt_transformers/demo/simple_text_demo.py -k ci-token-matching --timeout=4600 ; fail+=$?
    done
  done

  # 11B test does not run on N150
  tt_cache_llama11b=$TT_CACHE_HOME/$llama11b
  MESH_DEVICE=N300 HF_MODEL=$llama11b TT_CACHE_PATH=$tt_cache_llama11b pytest -n auto models/tt_transformers/demo/simple_text_demo.py -k ci-token-matching --timeout=4600 ; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_llama3_perplexity_tests_single_card $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_mistral_perplexity_tests() {
  # This one runs all the T3K tests

  echo "LOG_METAL: Running run_t3000_mistral_perplexity_tests"

  tt_cache_path="/mnt/MLPerf/tt_dnn-models/Mistral/TT_CACHE/Mistral-7B-Instruct-v0.3"
  hf_model="mistralai/Mistral-7B-Instruct-v0.3"
  TT_CACHE_PATH=$tt_cache_path HF_MODEL=$hf_model pytest models/tt_transformers/demo/simple_text_demo.py -k ci-token-matching --timeout=3600

}

run_t3000_llama3_perplexity_tests_t3000() {

  echo "LOG_METAL: Checking number of devices"
  python3 -c "import ttnn; print('Number of devices:', ttnn.get_num_devices())"

  # Split long set of tests into two groups
  # This one runs all the T3K tests
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_llama3_perplexity_tests_t3000"

  llama1b=meta-llama/Llama-3.2-1B-Instruct
  llama3b=meta-llama/Llama-3.2-3B-Instruct
  llama8b=meta-llama/Llama-3.1-8B-Instruct
  llama11b=meta-llama/Llama-3.2-11B-Vision-Instruct
  llama70b=meta-llama/Llama-3.1-70B-Instruct
  llama90b=meta-llama/Llama-3.2-90B-Vision-Instruct

  for MESH_DEVICE in T3K; do
    for hf_model in "$llama1b" "$llama3b" "$llama8b" "$llama11b"; do
      tt_cache=$TT_CACHE_HOME/$hf_model
      MESH_DEVICE=$MESH_DEVICE HF_MODEL=$hf_model TT_CACHE_PATH=$tt_cache pytest -n auto models/tt_transformers/demo/simple_text_demo.py -k ci-token-matching --timeout=3600 ; fail+=$?
    done

    # 70B and 90B tests has the same configuration between `-k "attention-accuracy"` and `-k "attention-performance"` so we only run one of them
    for hf_model in "$llama70b" "$llama90b"; do
      tt_cache=$TT_CACHE_HOME/$hf_model
      MESH_DEVICE=$MESH_DEVICE HF_MODEL=$hf_model TT_CACHE_PATH=$tt_cache pytest -n auto models/tt_transformers/demo/simple_text_demo.py -k "performance and ci-token-matching" --timeout=3600 ; fail+=$?
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

run_t3000_qwen25_perplexity_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  export PYTEST_ADDOPTS="--tb=short"

  echo "LOG_METAL: Running run_t3000_qwen25_perplexity_tests"
  qwen72b=Qwen/Qwen2.5-72B-Instruct
  tt_cache_72b=$TT_CACHE_HOME/$qwen72b

  HF_MODEL=$qwen72b TT_CACHE_PATH=$tt_cache_72b pytest -n auto models/tt_transformers/demo/simple_text_demo.py -k ci-token-matching --timeout 3600; fail+=$?
  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_qwen25_perplexity_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_qwen3_perplexity_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Warning: updating transformers version. Make sure this is the last-run test."
  echo "LOG_METAL: Remove this when https://github.com/tenstorrent/tt-metal/pull/22608 merges."

  echo "LOG_METAL: Running run_t3000_qwen3_perplexity_tests"
  qwen32b=Qwen/Qwen3-32B
  tt_cache_qwen32b=$TT_CACHE_HOME/$qwen32b

  HF_MODEL=$qwen32b TT_CACHE_PATH=$tt_cache_qwen32b pytest -n auto models/tt_transformers/demo/simple_text_demo.py -k ci-token-matching --timeout 3600; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_qwen3_perplexity_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_gemma3_accuracy_tests() {
  # Record the start time
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_gemma3_accuracy_tests"
  gemma3_27b=google/gemma-3-27b-it
  tt_cache_gemma3_27b=$TT_CACHE_HOME/$gemma3_27b

  HF_MODEL=$gemma3_27b TT_CACHE_PATH=$tt_cache_gemma3_27b pytest models/demos/gemma3/demo/text_demo.py -k "ci-token-matching"

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_gemma3_accuracy_tests $duration seconds to complete"
}


run_t3000_tests() {
  # Run Qwen2.5 perplexity tests
  run_t3000_qwen25_perplexity_tests

  # Run Qwen3 perplexity tests
  run_t3000_qwen3_perplexity_tests

  # Run Falcon-40B perplexity tests
  run_t3000_falcon40b_perplexity_tests

  # Run Llama-70B perplexity tests
  run_t3000_llama70b_perplexity_tests

  # Run mistral perplexity tests
  run_t3000_mistral_perplexity_tests

  # Run Mixtral8x7B perplexity tests
  run_t3000_mixtral8x7b_perplexity_tests

  # Run llama3 perplexity tests
  run_t3000_llama3_perplexity_tests_single_card
  run_t3000_llama3_perplexity_tests_t3000

  # Run gemma3 accuracy tests
  run_t3000_gemma3_accuracy_tests
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
