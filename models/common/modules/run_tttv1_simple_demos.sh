#!/bin/bash
# run_transformer_simple_demos.sh

# --- 1. Environment Setup ---
set -o pipefail
export HF_HOME=/proj_sw/user_dev/huggingface
export TT_METAL_HOME=${TT_METAL_HOME:-$(pwd)}
export PYTHONPATH=$TT_METAL_HOME:$PYTHONPATH
export ARCH_NAME=${ARCH_NAME:-"wormhole_b0"}
export TT_CACHE_HOME=/localdev/gwang/.cache
export CI=true

# Track failures
FAILED_TESTS=()
record_failure() {
    local test_info="$1"
    echo "LOG_METAL: TEST FAILED: $test_info"
    FAILED_TESTS+=("$test_info")
}

# --- 2. Local Function Definitions (Extracted from source scripts with modifications) ---

# --- Single Card Demos (copied from run_single_card_demo_tests.sh with modifications) ---

# [INFO] this is not run in CI: .github/workflows/single-card-demo-tests-impl.yaml
# run_mistral7b_func() {
#   mistral7b=mistralai/Mistral-7B-Instruct-v0.3
#   mistral_cache=$TT_CACHE_HOME/$mistral7b
#   HF_MODEL=$mistral7b TT_CACHE_PATH=$mistral_cache pytest models/tt_transformers/demo/simple_text_demo.py -k "performance and ci-token-matching" --timeout 1200; fail+=$?
# }

run_qwen7b_func() {
  qwen7b=Qwen/Qwen2-7B-Instruct
  qwen_cache=$TT_CACHE_HOME/$qwen7b
  CI=true MESH_DEVICE=N300 HF_MODEL=$qwen7b TT_CACHE_PATH=$qwen_cache pytest models/tt_transformers/demo/simple_text_demo.py -k performance-ci-1 || record_failure "qwen7b_n300"
}

run_ds_r1_qwen_func() {
  ds_r1_qwen_14b=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
  ds_r1_qwen_14b_cache=$TT_CACHE_HOME/$ds_r1_qwen_14b
  CI=true MESH_DEVICE=N300 HF_MODEL=$ds_r1_qwen_14b TT_CACHE_PATH=$ds_r1_qwen_14b_cache pytest models/tt_transformers/demo/simple_text_demo.py -k performance-ci-1 || record_failure "ds_r1_qwen_14b_n300"
}

run_phi4_func() { # --> just running accuracy and ci-token-matching tests
  phi4=microsoft/phi-4
  phi4_cache=$TT_CACHE_HOME/$phi4
  CI=true MESH_DEVICE=N300 HF_MODEL=$phi4 TT_CACHE_PATH=$phi4_cache pytest models/tt_transformers/demo/simple_text_demo.py -k "accuracy and ci-token-matching" || record_failure "phi4_n300"
}

# [INFO] just running ci-token-matching tests; already covered in run_llama3_perf() below
# run_llama3_func() {
#   fail=0
#   llama1b=meta-llama/Llama-3.2-1B-Instruct
#   llama3b=meta-llama/Llama-3.2-3B-Instruct
#   llama8b=meta-llama/Llama-3.1-8B-Instruct
#   llama11b=meta-llama/Llama-3.2-11B-Vision-Instruct

#   for hf_model in "$llama1b" "$llama3b" "$llama8b" "$llama11b"; do
#     cache_path=$TT_CACHE_HOME/$hf_model
#     HF_MODEL=$hf_model TT_CACHE_PATH=$cache_path MESH_DEVICE=N300 pytest models/tt_transformers/demo/simple_text_demo.py -k ci-token-matching  --timeout 420 || fail=1
#     echo "LOG_METAL: Llama3 accuracy tests for $hf_model completed"
#   done

#   if [[ $fail -ne 0 ]]; then exit 1; fi
# }

run_mistral7b_perf() {
  mistral7b=mistralai/Mistral-7B-Instruct-v0.3
  mistral_cache=$TT_CACHE_HOME/$mistral7b
  CI=true MESH_DEVICE=N150 HF_MODEL=$mistral7b TT_CACHE_PATH=$mistral_cache pytest models/tt_transformers/demo/simple_text_demo.py -k "not performance-ci-stress-1" || record_failure "mistral7b_n150"
  CI=true MESH_DEVICE=N300 HF_MODEL=$mistral7b TT_CACHE_PATH=$mistral_cache pytest models/tt_transformers/demo/simple_text_demo.py -k "not performance-ci-stress-1" || record_failure "mistral7b_n300"
}

run_llama3_perf() {
  llama1b=meta-llama/Llama-3.2-1B-Instruct
  llama3b=meta-llama/Llama-3.2-3B-Instruct
  llama8b=meta-llama/Llama-3.1-8B-Instruct
  llama11b=meta-llama/Llama-3.2-11B-Vision-Instruct

  for hf_model in "$llama1b" "$llama3b" "$llama8b"; do
    cache_path=$TT_CACHE_HOME/$hf_model
    CI=true MESH_DEVICE=N150 HF_MODEL=$hf_model TT_CACHE_PATH=$cache_path pytest models/tt_transformers/demo/simple_text_demo.py -k "not performance-ci-stress-1" || record_failure "llama3_${hf_model}_n150"
    echo "LOG_METAL: Llama3 tests for $hf_model completed on N150"
  done
  for hf_model in "$llama1b" "$llama3b" "$llama8b" "$llama11b"; do
    cache_path=$TT_CACHE_HOME/$hf_model
    CI=true MESH_DEVICE=N300 HF_MODEL=$hf_model TT_CACHE_PATH=$cache_path pytest models/tt_transformers/demo/simple_text_demo.py -k "not performance-ci-stress-1" || record_failure "llama3_${hf_model}_n300"
    echo "LOG_METAL: Llama3 tests for $hf_model completed"
  done
}

# --- T3000 Demos ---

run_t3000_llama3_70b_tests() {
  llama70b=meta-llama/Llama-3.3-70B-Instruct
  tt_cache_llama70b=$TT_CACHE_HOME/llama/3.3-70b
  CI=true MESH_DEVICE=T3K HF_MODEL=$llama70b TT_CACHE_PATH=$tt_cache_llama70b pytest models/tt_transformers/demo/simple_text_demo.py -k "not performance-ci-stress-1" || record_failure "llama3_70b_t3k"
}

run_t3000_llama3_tests() {
  llama8b=meta-llama/Llama-3.1-8B-Instruct
  llama1b=meta-llama/Llama-3.2-1B-Instruct
  llama3b=meta-llama/Llama-3.2-3B-Instruct
  llama11b=meta-llama/Llama-3.2-11B-Vision-Instruct

  for hf_model in "$llama1b" "$llama3b" "$llama8b" "$llama11b"; do
    tt_cache=$TT_CACHE_HOME/$hf_model
    CI=true MESH_DEVICE=T3K HF_MODEL=$hf_model TT_CACHE_PATH=$tt_cache pytest models/tt_transformers/demo/simple_text_demo.py -k "not performance-ci-stress-1" || record_failure "llama3_${hf_model}_t3k"
    echo "LOG_METAL: Llama3 tests for $hf_model completed"
  done
}

run_t3000_qwen25_tests() {
  export PYTEST_ADDOPTS="--tb=short"
  qwen25_7b=Qwen/Qwen2.5-7B-Instruct
  tt_cache_7b=$TT_CACHE_HOME/qwen/2.5-7B
  qwen25_72b=Qwen/Qwen2.5-72B-Instruct
  tt_cache_72b=$TT_CACHE_HOME/qwen/2.5-72B
  qwen25_coder_32b=Qwen/Qwen2.5-Coder-32B
  tt_cache_coder_32b=$TT_CACHE_HOME/$qwen25_coder_32b

  CI=true MESH_DEVICE=N300 HF_MODEL=$qwen25_7b TT_CACHE_PATH=$tt_cache_7b pytest models/tt_transformers/demo/simple_text_demo.py -k "not performance-ci-stress-1" || record_failure "qwen25_7b_n300"
  CI=true MESH_DEVICE=T3K HF_MODEL=$qwen25_72b TT_CACHE_PATH=$tt_cache_72b pytest models/tt_transformers/demo/simple_text_demo.py -k "not performance-ci-stress-1" || record_failure "qwen25_72b_t3k"
  CI=true MESH_DEVICE=T3K HF_MODEL=$qwen25_coder_32b TT_CACHE_PATH=$tt_cache_coder_32b pytest models/tt_transformers/demo/simple_text_demo.py -k "not performance-ci-stress-1" || record_failure "qwen25_coder_32b_t3k"
}

run_t3000_qwen3_tests() {
  qwen32b=Qwen/Qwen3-32B
  tt_cache_qwen32b=$TT_CACHE_HOME/qwen/3-32b
  CI=true MESH_DEVICE=T3K HF_MODEL=$qwen32b TT_CACHE_PATH=$tt_cache_qwen32b pytest models/tt_transformers/demo/simple_text_demo.py -k "not performance-ci-stress-1" || record_failure "qwen3_32b_t3k"
}

run_t3000_llama3_vision_tests() {
  llama11b=meta-llama/Llama-3.2-11B-Vision-Instruct
  tt_cache_llama11b=$TT_CACHE_HOME/$llama11b
  for mesh_device in "N300" "T3K"; do
    CI=true MESH_DEVICE=$mesh_device HF_MODEL=$llama11b TT_CACHE_PATH=$tt_cache_llama11b \
    pytest models/tt_transformers/demo/simple_vision_demo.py --timeout=900 -k "not batch1-notrace" || record_failure "llama3_vision_11b_${mesh_device}"
    echo "LOG_METAL: Llama3 vision tests for $mesh_device completed"
  done
}

run_t3000_llama3_90b_vision_tests() {
  export PYTEST_ADDOPTS="--tb=short"
  llama90b=meta-llama/Llama-3.2-90B-Vision-Instruct
  tt_cache_llama90b=$TT_CACHE_HOME/llama/3.2-90b
  CI=true MESH_DEVICE=T3K HF_MODEL=$llama90b TT_CACHE_PATH=$tt_cache_llama90b pytest models/tt_transformers/demo/simple_vision_demo.py --timeout=900 -k "batch1-trace" || record_failure "llama3_vision_90b_t3k"
  echo "LOG_METAL: Llama3.2-90B vision tests for T3K completed"
}

run_t3000_mistral_tests() {
  hf_model="mistralai/Mistral-7B-Instruct-v0.3"
  tt_cache_path=$TT_CACHE_HOME/$hf_model
  CI=true MESH_DEVICE=T3K TT_CACHE_PATH=$tt_cache_path HF_MODEL=$hf_model pytest models/tt_transformers/demo/simple_text_demo.py -k "not performance-ci-stress-1" || record_failure "mistral7b_t3k"
}

run_t3000_mixtral_tests() {
  mixtral8x7=mistralai/Mixtral-8x7B-v0.1
  tt_cache_path=$TT_CACHE_HOME/$mixtral8x7
  CI=true MESH_DEVICE=T3K TT_CACHE_PATH=$tt_cache_path HF_MODEL=$mixtral8x7 pytest models/tt_transformers/demo/simple_text_demo.py -k "not performance-ci-stress-1" || record_failure "mixtral8x7b_t3k"
}

# --- 3. Unified Runner Functions ---

run_all_simple_text_demos() {
    echo "LOG_METAL: Starting all simple_text_demo.py based tests..."
    run_llama3_perf
    run_mistral7b_perf
    run_qwen7b_func
    run_ds_r1_qwen_func
    run_phi4_func
    run_t3000_llama3_tests
    run_t3000_llama3_70b_tests
    run_t3000_qwen25_tests
    run_t3000_qwen3_tests
    run_t3000_mistral_tests
    run_t3000_mixtral_tests
}

run_all_simple_vision_demos() {
    echo "LOG_METAL: Starting all simple_vision_demo.py based tests..."
    run_t3000_llama3_vision_tests
    run_t3000_llama3_90b_vision_tests
}

# --- 4. Main Entry ---
main() {
    if [[ -z "$1" ]]; then
        echo "Usage: $0 [text|vision|all]"
        exit 1
    fi

    case "$1" in
        text)       run_all_simple_text_demos ;;
        vision)     run_all_simple_vision_demos ;;
        all)
            run_all_simple_text_demos
            run_all_simple_vision_demos
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac

    echo "------------------------------------------------------------"
    if [ ${#FAILED_TESTS[@]} -eq 0 ]; then
        echo "LOG_METAL: ALL TESTS PASSED SUCCESSFULLY!"
        exit 0
    else
        echo "LOG_METAL: SOME TESTS FAILED!"
        echo "LOG_METAL: Summary of failures:"
        for failed_test in "${FAILED_TESTS[@]}"; do
            echo "  - $failed_test"
        done
        exit 1
    fi
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
