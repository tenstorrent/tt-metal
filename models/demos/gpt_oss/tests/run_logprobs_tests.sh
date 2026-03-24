#!/bin/bash
# Run all logprobs-related tests for tt-metal and gpt-oss-120b
# Usage: bash run_logprobs_tests.sh [--all | --sampling-module | --gpt-oss | --demo | --demo-logprobs | --demo-compare]
#
# Options:
#   --all              Run all tests (default)
#   --sampling-module  Run only sampling module tests (models/common/tests/test_sampling.py)
#   --gpt-oss          Run only gpt-oss-120b sampling tests
#   --demo             Run batch128 demo without logprobs (baseline)
#   --demo-logprobs    Run batch128 demo with logprobs (top-5)
#   --demo-compare     Run both demos, save output to files, print perf diff
#
# Environment variables:
#   HF_MODEL         Path to gpt-oss-120b weights (default: /localdev/gpt-oss-120b)
#   TT_CACHE_PATH    Path to TT cache (default: /localdev/gpt-oss-120b)

set -e

# Navigate to tt-metal repo root (script lives in models/demos/gpt_oss/tests/)
cd "$(dirname "$0")/../../../.."

export HF_MODEL=${HF_MODEL:-/localdev/gpt-oss-120b}
export TT_CACHE_PATH=${TT_CACHE_PATH:-/localdev/gpt-oss-120b}

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

run_demo_compare() {
    OUTDIR="generated/logprobs_comparison"
    mkdir -p "$OUTDIR"

    echo "=== Running batch128 WITHOUT logprobs ==="
    echo "  Output: $OUTDIR/demo_no_logprobs.log"
    pytest models/demos/gpt_oss/demo/text_demo.py -k "4x8 and batch128 and not logprobs" \
        -v -s --timeout 1200 2>&1 | tee "$OUTDIR/demo_no_logprobs.log"

    echo ""
    echo "=== Running batch128 WITH logprobs (top-5) ==="
    echo "  Output: $OUTDIR/demo_with_logprobs.log"
    pytest models/demos/gpt_oss/demo/text_demo.py -k "4x8 and batch128 and logprobs" \
        -v -s --timeout 1200 2>&1 | tee "$OUTDIR/demo_with_logprobs.log"

    echo ""
    echo "=== Comparison files saved to $OUTDIR ==="
    echo ""
    echo "=== Perf comparison ==="
    echo "--- Without logprobs ---"
    grep -E "decode_t/s|TTFT|decode speed|Performance" "$OUTDIR/demo_no_logprobs.log" || true
    echo "--- With logprobs ---"
    grep -E "decode_t/s|TTFT|decode speed|Performance" "$OUTDIR/demo_with_logprobs.log" || true
}

case "$MODE" in
    --sampling-module)
        run_sampling_module
        ;;
    --gpt-oss-sampling-tests)
        run_gpt_oss
        ;;
    --demo-compare)
        run_demo_compare
        ;;
    --all|*)
        run_sampling_module
        echo ""
        run_gpt_oss
        echo ""
        run_demo_compare
        ;;
esac

echo ""
echo "All logprobs tests passed!"
