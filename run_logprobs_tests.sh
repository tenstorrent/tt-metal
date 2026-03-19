#!/bin/bash
# Run all logprobs-related tests for tt-metal and gpt-oss-120b
# Usage: bash run_logprobs_tests.sh [--all | --sampling-module | --gpt-oss | --topk-only]
#
# Options:
#   --all              Run all tests (default)
#   --sampling-module  Run only sampling module tests (models/common/tests/test_sampling.py)
#   --gpt-oss          Run only gpt-oss-120b sampling tests
#   --topk-only        Run only the new top-K logprobs tests (skip old backward-compat tests)
#
# Environment variables:
#   HF_MODEL         Path to gpt-oss-120b weights (default: /localdev/gpt-oss-120b)
#   TT_CACHE_PATH    Path to TT cache (default: /localdev/divanovic/tt-metal-cache)

set -e

cd "$(dirname "$0")"

export HF_MODEL=${HF_MODEL:-/localdev/gpt-oss-120b}
export TT_CACHE_PATH=${TT_CACHE_PATH:-/localdev/divanovic/tt-metal-cache}

MODE="${1:---all}"

run_sampling_module() {
    echo "=== Running sampling module logprobs tests (TG only) ==="
    echo "  File: models/common/tests/test_sampling.py"
    echo "  Filter: TG Galaxy top-K tests only"
    pytest models/common/tests/test_sampling.py -v --timeout 600 \
        -k "test_top_k or test_per_user or test_transfer_logprobs or test_set_log_probs_mode"
}

run_gpt_oss() {
    echo "=== Running gpt-oss-120b sampling tests ==="
    echo "  File: models/demos/gpt_oss/tests/unit/test_sampling.py"
    echo "  HF_MODEL=$HF_MODEL"
    if [ "$MODE" = "--topk-only" ]; then
        echo "  Filter: top-K logprobs test only"
        pytest models/demos/gpt_oss/tests/unit/test_sampling.py -v --timeout 600 \
            -k "test_gpt_oss_topk_logprobs"
    else
        pytest models/demos/gpt_oss/tests/unit/test_sampling.py -v --timeout 600 \
            -k "test_gpt_oss_topk_logprobs"
    fi
}

case "$MODE" in
    --sampling-module)
        run_sampling_module
        ;;
    --gpt-oss)
        run_gpt_oss
        ;;
    --topk-only)
        run_sampling_module
        echo ""
        run_gpt_oss
        ;;
    --all|*)
        run_sampling_module
        echo ""
        run_gpt_oss
        ;;
esac

echo ""
echo "All logprobs tests passed!"
