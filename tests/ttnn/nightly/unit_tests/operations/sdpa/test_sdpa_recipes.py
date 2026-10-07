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
    check_op_selected_blocking,
    l2_pct,
    randn,
    reference,
    sdpa,
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
@pytest.mark.parametrize("variant", VARIANTS)
def test_sdpa_recipe_rising_max(device, variant):
    sq, sk, d = 512, 4096, 128
    q, k, v = randn(1, 1, sq, d, seed=7), (randn(1, 1, sk, d, seed=8) * 0.25).bfloat16(), randn(1, 1, sk, d, seed=9)
    spiked = (torch.arange(sq) // 64) % 2 == 0
    q[0, 0, :, 0] = torch.where(spiked, 8.0, 0.0).bfloat16()
    k[0, 0, 1100:1108] = 0
    k[0, 0, 1100:1104, 0], k[0, 0, 1104:1108, 0] = 40.0, 36.0
    actual = ttnn.to_torch(sdpa(device, variant, q, k, v, 256, 512))
    assert l2_pct(actual, reference(q, k, v)) < L2_PCT_BOUND[variant]


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
