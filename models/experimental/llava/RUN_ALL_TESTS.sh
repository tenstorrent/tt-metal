#!/bin/bash
# LLaVA All Component Tests Runner
# Runs component, end-to-end, and MoE-LLaVA integration/PCC tests.
#
# Usage:
#   bash models/demos/llava/RUN_ALL_TESTS.sh
#   SKIP_E2E=1 bash models/demos/llava/RUN_ALL_TESTS.sh  # Skip slow E2E tests

echo "============================================"
echo "LLaVA All Component Tests"
echo "============================================"
echo ""

TT_METAL_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../../.." && pwd )"
RESTORE_CPP=false
RESTORE_TT_METAL=false

# Cleanup function: restore hidden directories if they were moved
cleanup() {
    if [ "$RESTORE_CPP" = true ] && [ -d "$TT_METAL_ROOT/ttnn/cpp_backup" ]; then
        mv "$TT_METAL_ROOT/ttnn/cpp_backup" "$TT_METAL_ROOT/ttnn/cpp"
        echo "✅ Restored ttnn/cpp (source tree kernels)"
    fi
    if [ "$RESTORE_TT_METAL" = true ] && [ -d "$TT_METAL_ROOT/tt_metal_backup" ]; then
        mv "$TT_METAL_ROOT/tt_metal_backup" "$TT_METAL_ROOT/tt_metal"
        echo "✅ Restored tt_metal (source tree kernels)"
    fi
}
trap cleanup EXIT

# Auto-detect ttnn installation
if [ -f "$TT_METAL_ROOT/ttnn/ttnn/_ttnn.cpython-310-x86_64-linux-gnu.so" ]; then
    export TT_METAL_HOME="$TT_METAL_ROOT"
    export PYTHONPATH="$TT_METAL_ROOT/ttnn:$TT_METAL_ROOT:$PYTHONPATH"
    echo "ttnn source: Built from source tree"
else
    export TT_METAL_HOME="/opt/venv/lib/python3.10/site-packages/ttnn"
    export PYTHONPATH="$TT_METAL_ROOT:$PYTHONPATH"
    echo "ttnn source: Pre-installed wheel (/opt/venv)"
    if [ -d "$TT_METAL_ROOT/ttnn/cpp" ]; then
        mv "$TT_METAL_ROOT/ttnn/cpp" "$TT_METAL_ROOT/ttnn/cpp_backup"
        RESTORE_CPP=true
        echo "⚠️  Temporarily hid ttnn/cpp (incompatible with installed wheel)"
    fi
    if [ -d "$TT_METAL_ROOT/tt_metal" ]; then
        mv "$TT_METAL_ROOT/tt_metal" "$TT_METAL_ROOT/tt_metal_backup"
        RESTORE_TT_METAL=true
        echo "⚠️  Temporarily hid tt_metal (incompatible with installed wheel)"
    fi
fi

echo ""
echo "Environment:"
echo "  TT_METAL_HOME: $TT_METAL_HOME"
echo ""

set -e
cd "$TT_METAL_ROOT"

# Device reset
echo "Resetting device..."
tt-smi -r --no_reinit 2>/dev/null || true
sleep 2
echo "✅ Device reset complete"
echo ""

PASS_COUNT=0
FAIL_COUNT=0

run_test() {
    local test_name=$1
    local test_path=$2
    local mesh=$3

    echo "--------------------------------------------"
    echo "Running: $test_name ($mesh)"
    echo "--------------------------------------------"
    export MESH_DEVICE=$mesh
    set +e
    scripts/run_safe_pytest.sh "$test_path" -v -s 2>&1 | tail -5
    local exit_code=$?
    set -e

    if [ $exit_code -eq 0 ]; then
        echo "✅ $test_name PASSED"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        echo "❌ $test_name FAILED (exit code: $exit_code)"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    echo ""
}

# === Vision Encoder Tests (4 tests) ===
echo "=== Vision Encoder Tests (Week 1) ==="
echo ""

run_test "CLIP PCC (n300)" \
  "models/demos/llava/tests/test_vision_encoders.py::test_clip_encoder_pcc" "N300"

run_test "CLIP Shape (n300)" \
  "models/demos/llava/tests/test_vision_encoders.py::test_clip_encoder_output_shape" "N300"

run_test "CLIP Determinism (n300)" \
  "models/demos/llava/tests/test_vision_encoders.py::test_clip_encoder_deterministic" "N300"

run_test "CLIP Single Chip (n150)" \
  "models/demos/llava/tests/test_vision_encoders.py::test_clip_encoder_single_chip" "N150"

# === Projector Tests (4 tests) ===
echo "=== Projector Tests (Week 2) ==="
echo ""

run_test "MLP PCC (n300)" \
  "models/demos/llava/tests/test_projectors.py::test_mlp_projector_pcc" "N300"

run_test "MLP Shape (n300)" \
  "models/demos/llava/tests/test_projectors.py::test_mlp_projector_output_shape" "N300"

run_test "MLP Determinism (n300)" \
  "models/demos/llava/tests/test_projectors.py::test_mlp_projector_deterministic" "N300"

run_test "MLP Single Chip (n150)" \
  "models/demos/llava/tests/test_projectors.py::test_mlp_projector_single_chip" "N150"

# === Token Handler Tests (7 tests) ===
echo "=== Token Handler Tests (Week 3) ==="
echo ""

run_test "Interleaver Shape (n300)" \
  "models/demos/llava/tests/test_token_handlers.py::test_interleaver_shape_single_image" "N300"

run_test "Interleaver Correctness (n300)" \
  "models/demos/llava/tests/test_token_handlers.py::test_interleaver_correctness" "N300"

run_test "Interleaver vs PyTorch (n300)" \
  "models/demos/llava/tests/test_token_handlers.py::test_interleaver_matches_pytorch" "N300"

run_test "Interleaver Determinism (n300)" \
  "models/demos/llava/tests/test_token_handlers.py::test_interleaver_deterministic" "N300"

run_test "Interleaver Multi-Image (n300)" \
  "models/demos/llava/tests/test_token_handlers.py::test_interleaver_multi_image" "N300"

run_test "Interleaver Batch Padding (n300)" \
  "models/demos/llava/tests/test_token_handlers.py::test_interleaver_batch_padding" "N300"

run_test "Interleaver Single Chip (n150)" \
  "models/demos/llava/tests/test_token_handlers.py::test_interleaver_single_chip" "N150"

# === Vision Pipeline Tests (5 tests) ===
echo "=== Vision Pipeline Tests (Week 4) ==="
echo ""

run_test "Pipeline CLIP PCC (n300)" \
  "models/demos/llava/tests/test_vision_pipeline.py::test_vision_pipeline_clip_pcc" "N300"

run_test "Pipeline Projector PCC (n300)" \
  "models/demos/llava/tests/test_vision_pipeline.py::test_vision_pipeline_projector_pcc" "N300"

run_test "Pipeline E2E PCC (n300)" \
  "models/demos/llava/tests/test_vision_pipeline.py::test_vision_pipeline_end_to_end_pcc" "N300"

run_test "Pipeline Shapes (n300)" \
  "models/demos/llava/tests/test_vision_pipeline.py::test_vision_pipeline_shapes" "N300"

run_test "Pipeline Single Chip (n150)" \
  "models/demos/llava/tests/test_vision_pipeline.py::test_vision_pipeline_single_chip" "N150"

# === End-to-End Tests (7 tests, slow — downloads LLaVA model) ===
if [ "${SKIP_E2E:-0}" != "1" ]; then
echo "=== End-to-End Tests (Week 4, slow) ==="
echo ""

run_test "E2E CLIP PCC (n300)" \
  "models/demos/llava/tests/test_llava_end_to_end.py::test_e2e_clip_pcc" "N300"

run_test "E2E Projector PCC (n300)" \
  "models/demos/llava/tests/test_llava_end_to_end.py::test_e2e_projector_pcc" "N300"

run_test "E2E Interleaver PCC (n300)" \
  "models/demos/llava/tests/test_llava_end_to_end.py::test_e2e_interleaver_pcc" "N300"

run_test "E2E Full Pipeline PCC (n300)" \
  "models/demos/llava/tests/test_llava_end_to_end.py::test_e2e_full_pipeline_pcc" "N300"

run_test "E2E LLM Generation (n300)" \
  "models/demos/llava/tests/test_llava_end_to_end.py::test_e2e_llm_generation" "N300"

run_test "E2E Shapes (n300)" \
  "models/demos/llava/tests/test_llava_end_to_end.py::test_e2e_shapes" "N300"

run_test "E2E Single Chip (n150)" \
  "models/demos/llava/tests/test_llava_end_to_end.py::test_e2e_single_chip" "N150"
else
echo "=== End-to-End Tests SKIPPED (SKIP_E2E=1) ==="
echo ""
fi

# === MoE-LLaVA Tests ===
echo "=== MoE-LLaVA Integration and PCC Tests ==="
echo ""

run_test "MoE-LLaVA Integration and PCC (n300)" \
  "models/demos/llava/tests/test_moe_llava.py" "N300"

# === Summary ===
echo "============================================"
echo "Test Suite Complete"
echo "============================================"
echo ""
echo "Summary: $PASS_COUNT passed, $FAIL_COUNT failed (out of $((PASS_COUNT + FAIL_COUNT)))"
echo ""
if [ $FAIL_COUNT -eq 0 ]; then
    echo "🎉 All tests passed!"
else
    echo "⚠️  Some tests failed. Review output above."
fi
