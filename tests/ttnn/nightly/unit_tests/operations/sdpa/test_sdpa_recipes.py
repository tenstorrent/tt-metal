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


# Odd Q chunks (a single-row tail group) at D64 with logits about 8x the unit-variance case (Q scaled by 8): the
# reference max moves in middle K chunks, so the unfused chunk (QK subblock width 1: K160, K32; or every chunk with
# an attn_mask) rescales the tail group's O and l from their own row. Q96/K128 is the QK/PV width 4/2 build that
# used to stop the SFPI compiler. FAST's LoFi QK error grows with the logits (about 5.6% here at every chunk size,
# vs 2.9% unscaled); a wrong tail row gives 50% to 1e10%.
ODD_Q_LARGE_LOGIT_BOUND = {"standard": 5.0, "fast_bf16": 7.0, "fast_bfp8": 7.0}


@pytest.mark.parametrize("q_chunk, k_chunk", [(96, 160), (96, 32), (160, 160), (96, 128)])
@pytest.mark.parametrize("variant", ODD_Q_LARGE_LOGIT_BOUND)
def test_sdpa_recipe_odd_q_large_logits(device, variant, q_chunk, k_chunk):
    q, k, v = randn(1, 2, 288, 64, seed=1), randn(1, 2, 640, 64, seed=2), randn(1, 2, 640, 64, seed=3)
    q = (q * 8).bfloat16()
    actual = ttnn.to_torch(sdpa(device, variant, q, k, v, q_chunk, k_chunk))
    assert l2_pct(actual, reference(q, k, v)) < ODD_Q_LARGE_LOGIT_BOUND[variant]


# Odd Q chunks with an attn_mask that hides every key of a row's first K chunks (a causal sliding window, masked
# with -2^100 so such rows start from a finite maximum): the reference max jumps in a middle chunk.
@pytest.mark.parametrize("q_chunk, k_chunk", [(96, 160), (96, 128), (160, 96)])
@pytest.mark.parametrize("variant", ["standard", "fast_bf16"])
def test_sdpa_recipe_odd_q_masked_first_chunk(device, variant, q_chunk, k_chunk):
    s, window = 864, 300
    q, k, v = randn(1, 1, s, 64, seed=33), randn(1, 1, s, 64, seed=34), randn(1, 1, s, 64, seed=35)
    row, col = torch.arange(s)[:, None], torch.arange(s)[None, :]
    mask = torch.zeros(1, 1, s, s)
    mask[..., (col > row) | (col < row - window)] = -(2.0**100)
    mask = mask.bfloat16()
    actual = ttnn.to_torch(sdpa(device, variant, q, k, v, q_chunk, k_chunk, mask))
    assert l2_pct(actual, reference(q, k, v, mask)) < L2_PCT_BOUND[variant]
