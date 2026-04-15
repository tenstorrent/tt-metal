#!/usr/bin/env bash
# Run all llama3_70b_galaxy unit tests affected by the HF-weights fixes.
# Prerequisites: source set_metal_flags.sh and activate your Python env first.

set -euo pipefail

PYTEST="pytest --timeout=900 -v"
TESTS="models/demos/llama3_70b_galaxy/tests/unit_tests"

echo "=== RMS Norm ==="
$PYTEST $TESTS/test_llama_rms_norm.py

echo "=== MLP (decode) ==="
$PYTEST $TESTS/test_llama_mlp.py

echo "=== MLP (prefill) ==="
$PYTEST $TESTS/test_llama_mlp_prefill.py

echo "=== Attention (decode) ==="
$PYTEST $TESTS/test_llama_attention.py

echo "=== Attention (prefill) ==="
$PYTEST $TESTS/test_llama_attention_prefill.py

echo "=== Decoder (prefill) ==="
$PYTEST $TESTS/test_llama_decoder_prefill.py

echo "=== Full model (prefill) ==="
$PYTEST $TESTS/test_llama_model_prefill.py

echo "=== All tests passed ==="
