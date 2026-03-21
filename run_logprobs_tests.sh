#!/bin/bash
# Run all logprobs-related tests for tt-metal and gpt-oss-120b
# Usage: bash run_logprobs_tests.sh [--all | --sampling-module | --gpt-oss | --demo | --demo-logprobs]
#
# Options:
#   --all              Run all tests (default)
#   --sampling-module  Run only sampling module tests (models/common/tests/test_sampling.py)
#   --gpt-oss          Run only gpt-oss-120b sampling tests
#   --demo             Run batch128 demo without logprobs (baseline)
#   --demo-logprobs    Run batch128 demo with logprobs (top-5)
#   --demo-all         Run both demo variants (with and without logprobs)
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
        -k "test_top_k or test_per_user or test_set_log_probs_mode"
}

run_gpt_oss() {
    echo "=== Running gpt-oss-120b sampling tests ==="
    echo "  File: models/demos/gpt_oss/tests/unit/test_sampling.py"
    echo "  HF_MODEL=$HF_MODEL"
    pytest models/demos/gpt_oss/tests/unit/test_sampling.py -v --timeout 600 \
        -k "test_gpt_oss_topk_logprobs"
}

run_demo() {
    echo "=== Running gpt-oss-120b batch128 demo (no logprobs) ==="
    echo "  HF_MODEL=$HF_MODEL"
    pytest models/demos/gpt_oss/demo/text_demo.py::test_gpt_oss_demo[batch128-mesh_4x8] \
        -v -s --timeout 1200
}

run_demo_logprobs() {
    echo "=== Running gpt-oss-120b batch128 demo (with logprobs top-5) ==="
    echo "  HF_MODEL=$HF_MODEL"
    pytest models/demos/gpt_oss/demo/text_demo.py::test_gpt_oss_demo[batch128_logprobs-mesh_4x8] \
        -v -s --timeout 1200
}

case "$MODE" in
    --sampling-module)
        run_sampling_module
        ;;
    --gpt-oss)
        run_gpt_oss
        ;;
    --demo)
        run_demo
        ;;
    --demo-logprobs)
        run_demo_logprobs
        ;;
    --demo-all)
        run_demo
        echo ""
        run_demo_logprobs
        ;;
    --all|*)
        run_sampling_module
        echo ""
        run_gpt_oss
        echo ""
        run_demo
        echo ""
        run_demo_logprobs
        ;;
esac

echo ""
echo "All logprobs tests passed!"
