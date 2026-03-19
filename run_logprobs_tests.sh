#!/bin/bash
# Run all logprobs-related unit tests for tt-metal
# Usage: bash run_logprobs_tests.sh

set -e

echo "=== Running sampling module logprobs tests ==="
pytest models/common/tests/test_sampling.py -v --timeout 600

echo ""
echo "=== Running gpt-oss-120b sampling tests ==="
TT_CACHE_PATH=${TT_CACHE_PATH:-/localdev/divanovic/tt-metal-cache} \
HF_MODEL=${HF_MODEL:-/localdev/gpt-oss-120b} \
pytest models/demos/gpt_oss/tests/unit/test_sampling.py -v --timeout 600

echo ""
echo "All logprobs tests passed!"
