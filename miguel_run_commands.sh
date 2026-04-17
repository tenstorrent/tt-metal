#!/usr/bin/env bash
# Run commands for mtairum/bspm_unit_tests branch — BH Galaxy
#
# Prerequisites:
#   source python_env/bin/activate   (or adjust PYTHON below)
#
# All tests require an 8-device 4×2 mesh and will pytest.skip() automatically
# if the mesh is smaller.

set -euo pipefail   # stop on first failure, treat unset vars as errors

export TT_METAL_SLOW_DISPATCH_MODE=1   # exposes full 13×10 grid on BH Galaxy

PYTHON=python_env/bin/python
PYTEST="$PYTHON -m pytest"
TESTS=models/demos/deepseek_v3_b1/tests/unit_tests

# ─── Step 1: BSPM loading + TensorCache (test_prepare_weights.py) ─────────────
#
# Subtests (all PASSING: 2026-04-17):
#   test_prepare_moe_routed_experts_bspm_output_types_4x2
#     — prepare_moe_routed_experts_bspm returns CompressedTensor per expert,
#       correct shapes, DRAM-contiguous, writes tiles.bin to TensorCache.
#   test_prepare_moe_routed_experts_bspm_tile_assignment_4x2
#     — BFP4/BFP2/zero counts preserved through DRAM-shuffle permutation;
#       catches silent Bug 24 reshape mismatch.
#   test_prepare_moe_routed_experts_bspm_footprint_4x2
#     — compact tiles.bin disk footprint is smaller than uniform BFP4 baseline.
#   test_prepare_moe_layer_bspm_cache_roundtrip_4x2
#     — TensorCache roundtrip: cold miss writes tiles.bin; warm hit reloads;
#       code distributions must match between miss and hit.
#   test_prepare_routed_expert_weights_bspm_fallback_4x2
#     — fallback path: missing .bspm file → uniform bfloat4_b ttnn.Tensor (not CompressedTensor).
# PASSING: 2026-04-17
$PYTEST $TESTS/test_prepare_weights.py \
  -k "bspm" -v \
  || { echo "FAILED: test_prepare_weights.py BSPM suite"; exit 1; }

# ─── Step 1: TensorCache roundtrip inside DRAM kernel test ────────────────────

# End-to-end cache roundtrip for DRAMStreamingMatmulCompressed:
# miss writes compact tiles.bin; hit reloads; matmul PCC(miss, hit) ≥ 0.99.
# PASSING: 2026-04-17
$PYTEST $TESTS/test_dram_matmul_custom_compressed.py::test_dram_matmul_bspm_cache_roundtrip -v \
  || { echo "FAILED: test_dram_matmul_bspm_cache_roundtrip"; exit 1; }

echo "All Step 1 BSPM tests passed."
