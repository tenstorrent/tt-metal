#!/bin/bash
# run_transformer_simple_demos.sh

# --- 1. Environment Setup ---
set -o pipefail
export HF_HOME=${HOME}/.cache/huggingface
export TT_METAL_HOME=${TT_METAL_HOME:-$(pwd)}
export PYTHONPATH=$TT_METAL_HOME:$PYTHONPATH
export ARCH_NAME=${ARCH_NAME:-"wormhole_b0"}
export TT_CACHE_HOME=${HOME}/.cache/tt_models_cache
export CI=true

# Track failures
FAILED_TESTS=()
record_failure() {
    local test_info="$1"
    echo "LOG_METAL: TEST FAILED: $test_info"
    FAILED_TESTS+=("$test_info")
}

# --- 2. Local Function Definitions (Extracted from source scripts with modifications) ---
# models/tt_transformers/demo/simple_text_demo.py:878: CI only runs Llama3 70b DP = 4, TP = 8 or Llama3 8b DP = 4/16/32, TP = 8/2/1 on TG

run_tg_llama3_70b_demo() {
    echo "LOG_METAL: Starting llama3_70b_demo.py..."
    MESH_DEVICE=TG HF_MODEL=meta-llama/Llama-3.3-70B-Instruct TT_CACHE_PATH=$TT_CACHE_HOME/llama/3.3-70b pytest models/tt_transformers/demo/simple_text_demo.py --timeout 1000
}

run_tg_llama3_8b_demo() {
    echo "LOG_METAL: Starting llama3_8b_demo.py..."
    MESH_DEVICE=TG HF_MODEL=meta-llama/Llama-3.1-8B-Instruct TT_CACHE_PATH=$TT_CACHE_HOME/llama/3.1-8b pytest models/tt_transformers/demo/simple_text_demo.py --timeout 1000
}

# --- 3. Unified Runner Functions ---

run_all_simple_text_demos() {
    echo "LOG_METAL: Starting all simple_text_demo.py based tests..."
    run_tg_llama3_70b_demo
    run_tg_llama3_8b_demo
}

# --- 4. Main Entry ---
main() {
    if [[ -z "$1" ]]; then
        echo "Usage: $0 [text|all]"
        exit 1
    fi

    case "$1" in
        text)       run_all_simple_text_demos ;;
        all)        run_all_simple_text_demos ;;
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
