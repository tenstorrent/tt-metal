# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SDPA precision recipes (ttnn.SDPAPrecision) against an FP64 reference: the sweeps.

The fast subset runs in the ttnn sanity sdpa group (tests/ttnn/unit_tests/operations/sdpa/test_sdpa_recipes.py).
Recipes and their numerics: tech_reports/FlashAttention/SDPAPrecisionRecipes.md.
"""

import pytest
import torch
import ttnn

from tests.ttnn.unit_tests.operations.sdpa.sdpa_recipe_test_utils import (
    L2_PCT_BOUND,
    OP_SELECTED_SHAPES,
    SHAPES,
    VARIANTS,
    blackhole_only,
    check_accuracy,
    check_attn_mask,
    check_joint,
    check_legacy_arguments,
    check_op_selected_blocking,
    l2_pct,
    program_config,
    randn,
    reference,
    sdpa,
    stored,
    to_device,
)

pytestmark = blackhole_only


@pytest.mark.parametrize("shape", SHAPES.values(), ids=SHAPES.keys())
@pytest.mark.parametrize("variant", VARIANTS)
def test_sdpa_recipe_accuracy(device, variant, shape):
    check_accuracy(device, variant, shape)


# Long K with small logits: a BF16 running state would swamp (the legacy kernel: about 3.9%); STANDARD's FP32
# state does not (about 1.7%).
LONG_K_BOUND = {**L2_PCT_BOUND, "standard": 2.2, "fast_bf16": 2.6, "fast_bfp8": 2.7}


@pytest.mark.parametrize("variant", VARIANTS)
def test_sdpa_recipe_long_k(device, variant):
    q, k, v = randn(1, 1, 256, 128, seed=4), randn(1, 1, 32768, 128, seed=5), randn(1, 1, 32768, 128, seed=6)
    q, k = (q * 0.25).bfloat16(), (k * 0.25).bfloat16()
    actual = ttnn.to_torch(sdpa(device, variant, q, k, v, 256, 512))
    assert l2_pct(actual, reference(q, k, v)) < LONG_K_BOUND[variant]


# Rising row maxima: on alternating pairs of tile rows, a later K chunk holds two groups of keys scoring about
# 28 and 25 above the first chunk's maximum (natural-log units of the scaled scores), far past the
# reference-max headroom. Every recipe must rescale those rows; FAST's fused chunks must redo their row
# groups next to kept ones. Without the redo the fast exp saturates both groups to the same P and the 16:1 weight
# ratio collapses. One channel carries the spike with exactly representable values, so K's rounding does not
# move it.
# With scale=0.25 the spike is 2.75 instead of 8: the same scores, about 27.5 and 24.75 above the first chunk's
# maximum, reach the recipes' thresholds (theta and tau are in units of the scaled scores) through the scale.
@pytest.mark.parametrize("scale, spike", [(None, 8.0), (0.25, 2.75)], ids=["default_scale", "scale_0.25"])
@pytest.mark.parametrize("variant", VARIANTS)
def test_sdpa_recipe_rising_max(device, variant, scale, spike):
    sq, sk, d = 512, 4096, 128
    q, k, v = randn(1, 1, sq, d, seed=7), (randn(1, 1, sk, d, seed=8) * 0.25).bfloat16(), randn(1, 1, sk, d, seed=9)
    spiked = (torch.arange(sq) // 64) % 2 == 0
    q[0, 0, :, 0] = torch.where(spiked, spike, 0.0).bfloat16()
    k[0, 0, 1100:1108] = 0
    k[0, 0, 1100:1104, 0], k[0, 0, 1104:1108, 0] = 40.0, 36.0
    actual = ttnn.to_torch(sdpa(device, variant, q, k, v, 256, 512, scale=scale))
    assert l2_pct(actual, reference(q, k, v, scale=scale)) < L2_PCT_BOUND[variant]


@pytest.mark.parametrize("mask_kind", ["random", "key_padding"])
@pytest.mark.parametrize("variant", VARIANTS)
def test_sdpa_recipe_attn_mask(device, variant, mask_kind):
    check_attn_mask(device, variant, mask_kind)


@pytest.mark.parametrize("variant", VARIANTS)
def test_joint_sdpa_recipe(device, variant):
    check_joint(device, variant)


@pytest.mark.parametrize("shape", OP_SELECTED_SHAPES.values(), ids=OP_SELECTED_SHAPES.keys())
@pytest.mark.parametrize("variant", ["standard", "balanced", "accurate", "fast_bfp8"])
def test_sdpa_recipe_op_selected_blocking(device, variant, shape):
    """Chunk sizes left to the op (no program_config, or zero chunk sizes)."""
    check_op_selected_blocking(device, variant, shape)


# ---------------------------------------------------------------------------------------------------------------
# Arguments legacy SDPA callers pass (check_legacy_arguments): scale, packed inputs, compute config, L1, many heads.
# ---------------------------------------------------------------------------------------------------------------
# scale 1.0 (Gemma-style) with Q scaled by 1/8: the same scaled logits as the default scale at D64. Unscaled, the
# logits (std 8) hit a FAST bug that predates the scale argument (Q96 with a one-tile-wide QK subblock, K160 or
# K32: rel-L2 in the thousands, also at the default scale with Q x8); see stream-b notes.
@pytest.mark.parametrize("scale, q_multiplier", [(1.0, 0.125), (0.3, 1.0), (0.02, 1.0)], ids=["1.0", "0.3", "0.02"])
@pytest.mark.parametrize("variant", VARIANTS)
def test_sdpa_recipe_scale(device, variant, scale, q_multiplier):
    """The exp folds the scale in; an attn_mask is pre-scaled by 1/scale (exact for 0 and -inf)."""
    check_legacy_arguments(device, variant, scale=scale, mask=scale == 0.3, q_multiplier=q_multiplier)


@pytest.mark.parametrize(
    "q_dtype, kv_dtype",
    [
        (ttnn.bfloat16, ttnn.bfloat8_b),
        (ttnn.bfloat8_b, ttnn.bfloat8_b),
        (ttnn.bfloat16, ttnn.bfloat4_b),
        (ttnn.bfloat4_b, ttnn.bfloat4_b),
    ],
    ids=["kv_bfp8", "qkv_bfp8", "kv_bfp4", "qkv_bfp4"],
)
@pytest.mark.parametrize("variant", ["standard", "balanced", "accurate"])
def test_sdpa_recipe_packed_inputs(device, variant, q_dtype, kv_dtype):
    """BFP8/BFP4 K/V unpack into the recipe's source registers; BFP8/BFP4 Q is widened to BF16 first and the
    output returned in Q's dtype. FP64 reference on the stored values."""
    check_legacy_arguments(device, variant, q_dtype=q_dtype, kv_dtype=kv_dtype, mask=True)


@pytest.mark.parametrize("variant", VARIANTS)
def test_sdpa_recipe_ignores_compute_kernel_config(device, variant):
    """An FP32-dest HiFi4 compute_kernel_config with math_approx_mode and exp_approx_mode False: bit-identical."""
    plain = ttnn.to_torch(check_legacy_arguments(device, variant, mask=True))
    configured = ttnn.to_torch(check_legacy_arguments(device, variant, mask=True, compute_kernel_config=True))
    assert torch.equal(plain, configured)


@pytest.mark.parametrize("variant", VARIANTS)
def test_sdpa_recipe_l1_inputs_and_output(device, variant):
    """L1-interleaved Q/K/V, mask and output: bit-identical to DRAM."""
    dram = ttnn.to_torch(check_legacy_arguments(device, variant, mask=True))
    l1 = ttnn.to_torch(check_legacy_arguments(device, variant, mask=True, memory_config=ttnn.L1_MEMORY_CONFIG))
    assert torch.equal(dram, l1)


# b, nh, nkv, sq, sk, d, q_chunk, k_chunk (0: op-selected), grid (None: the device grid).
MANY_HEADS = {
    "gqa_2x2_grid": ((2, 6, 2, 288, 640, 64, 96, 160), (2, 2)),
    "op_selected_192_heads": ((8, 24, 8, 256, 300, 64, 0, 0), None),
}


@pytest.mark.parametrize("case", MANY_HEADS)
@pytest.mark.parametrize("variant", ["standard", "balanced", "accurate", "fast_bfp8"])
def test_sdpa_recipe_more_heads_than_cores(device, variant, case):
    """More batch/heads than cores: the Q chunks of every head split evenly over the grid."""
    shape, grid = MANY_HEADS[case]
    check_legacy_arguments(device, variant, shape=shape, grid=grid, mask=True)


@pytest.mark.parametrize("variant", ["standard", "balanced", "accurate"])
def test_joint_sdpa_recipe_legacy_arguments(device, variant):
    """Joint attention with BFP8 inputs (outputs in BFP8), a custom scale and an ignored compute config."""
    q, k, v = randn(1, 2, 1000, 128, seed=37), randn(1, 2, 1000, 128, seed=38), randn(1, 2, 1000, 128, seed=39)
    jq, jk, jv = randn(1, 2, 77, 128, seed=40), randn(1, 2, 77, 128, seed=41), randn(1, 2, 77, 128, seed=42)
    q, k, v, jq, jk, jv = (stored(x, ttnn.bfloat8_b) for x in (q, k, v, jq, jk, jv))
    config = program_config(device, 256, 512)
    config.exp_approx_mode = False
    out, joint_out = ttnn.transformer.joint_scaled_dot_product_attention(
        *(to_device(device, x, ttnn.bfloat8_b) for x in (q, k, v, jq, jk, jv)),
        joint_strategy="rear",
        program_config=config,
        scale=0.0625,  # FP32-exact: the joint binding takes scale with noconvert
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True
        ),
        precision=VARIANTS[variant][0],
    )
    assert out.dtype == ttnn.bfloat8_b and joint_out.dtype == ttnn.bfloat8_b
    expected = reference(torch.cat([q, jq], 2), torch.cat([k, jk], 2), torch.cat([v, jv], 2), scale=0.0625)
    actual = torch.cat([ttnn.to_torch(out), ttnn.to_torch(joint_out)], 2)
    output_rounding = 1.5 * l2_pct(stored(expected.bfloat16(), ttnn.bfloat8_b), expected)
    assert l2_pct(actual, expected) < L2_PCT_BOUND[variant] + output_rounding
