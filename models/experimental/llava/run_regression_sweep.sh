#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Regression sweep for all LLaVA tests. Run from tt-metal root.
#
# Usage:
#   bash models/demos/llava/run_regression_sweep.sh
#
# This script runs all component-level tests (vision encoders, projectors,
# token handlers, vision pipeline, end-to-end) in sequence, using the
# safe pytest runner for device serialization. Tests that require a
# healthy n300 are skipped automatically when the device is unavailable.
#
# Prerequisites:
#   - tt-metal built (or pre-installed wheel)
#   - MESH_DEVICE env var set (N300 or N150)
#   - HF token for gated models (optional — CLIP is public)

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TT_METAL_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "============================================"
echo "LLaVA Regression Sweep"
echo "============================================"
echo "MESH_DEVICE: ${MESH_DEVICE:-N300}"
echo

PASS=0
FAIL=0
SKIP=0
FAILED_TESTS=()

run_test() {
    local name="$1"
    local test_path="$2"
    echo "--- Running: $name ---"
    if MESH_DEVICE=${MESH_DEVICE:-N300} scripts/run_safe_pytest.sh "$test_path" -v 2>&1 | tail -5; then
        echo "  PASS: $name"
        PASS=$((PASS + 1))
    else
        exit_code=$?
        if [ $exit_code -eq 2 ]; then
            echo "  SKIP (device hang): $name"
            SKIP=$((SKIP + 1))
        else
            echo "  FAIL: $name"
            FAIL=$((FAIL + 1))
            FAILED_TESTS+=("$name")
        fi
    fi
    echo
}

# CLIP + SigLIP vision encoder tests (component)
run_test "CLIP encoder PCC" "models/demos/llava/tests/test_vision_encoders.py::test_clip_encoder_pcc"
run_test "CLIP encoder shape" "models/demos/llava/tests/test_vision_encoders.py::test_clip_encoder_output_shape"
run_test "CLIP encoder determinism" "models/demos/llava/tests/test_vision_encoders.py::test_clip_encoder_deterministic"
run_test "CLIP encoder single-chip" "models/demos/llava/tests/test_vision_encoders.py::test_clip_encoder_single_chip"
run_test "SigLIP encoder PCC" "models/demos/llava/tests/test_vision_encoders.py::test_siglip_encoder_pcc"
run_test "SigLIP encoder shape" "models/demos/llava/tests/test_vision_encoders.py::test_siglip_encoder_output_shape"
run_test "SigLIP encoder determinism" "models/demos/llava/tests/test_vision_encoders.py::test_siglip_encoder_deterministic"
run_test "SigLIP encoder single-chip" "models/demos/llava/tests/test_vision_encoders.py::test_siglip_encoder_single_chip"

# Projector tests
run_test "MLP projector PCC" "models/demos/llava/tests/test_projectors.py::test_mlp_projector_pcc"
run_test "MLP projector shape" "models/demos/llava/tests/test_projectors.py::test_mlp_projector_output_shape"
run_test "MLP projector determinism" "models/demos/llava/tests/test_projectors.py::test_mlp_projector_deterministic"
run_test "MLP projector single-chip" "models/demos/llava/tests/test_projectors.py::test_mlp_projector_single_chip"

# Token handler tests
run_test "Token interleaver shape" "models/demos/llava/tests/test_token_handlers.py::test_interleaver_shape_single_image"

# Vision pipeline tests
run_test "Vision pipeline CLIP PCC" "models/demos/llava/tests/test_vision_pipeline.py::test_vision_pipeline_clip_pcc"
run_test "Vision pipeline end-to-end PCC" "models/demos/llava/tests/test_vision_pipeline.py::test_vision_pipeline_end_to_end_pcc"

# Qwen1-MoE converter self-test (CPU-only, no device needed)
echo "--- Running: Qwen1-MoE checkpoint converter self-test ---"
if python models/demos/llava/tt/qwen1_moe_checkpoint.py 2>&1 | tail -3; then
    echo "  PASS: Qwen1-MoE converter"
    PASS=$((PASS + 1))
else
    echo "  FAIL: Qwen1-MoE converter"
    FAIL=$((FAIL + 1))
    FAILED_TESTS+=("Qwen1-MoE converter")
fi
echo

# MoE-LLaVA integration and device PCC gates. The device tests require a
# healthy mesh and handle the n150/n300 fabric setup in the test fixture.
run_test "MoE-LLaVA integration (CPU)" "models/demos/llava/tests/test_moe_llava.py"

echo "============================================"
echo "Regression Sweep Summary"
echo "============================================"
echo "  PASS: $PASS"
echo "  FAIL: $FAIL"
echo "  SKIP: $SKIP"
if [ $FAIL -gt 0 ]; then
    echo "  Failed tests:"
    for t in "${FAILED_TESTS[@]}"; do
        echo "    - $t"
    done
fi
echo "============================================"

exit $FAIL
