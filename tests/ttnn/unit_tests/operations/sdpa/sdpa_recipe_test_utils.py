# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the SDPA precision recipe tests (ttnn.SDPAPrecision) against an FP64 reference.

The fast subset is tests/ttnn/unit_tests/operations/sdpa/test_sdpa_recipes.py (ttnn sanity); the sweeps are
tests/ttnn/nightly/unit_tests/operations/sdpa/test_sdpa_recipes.py. Recipes and their numerics:
tech_reports/FlashAttention/SDPAPrecisionRecipes.md.
"""

import math
import os

import pytest
import torch
import ttnn

from models.common.utility_functions import is_blackhole

blackhole_only = pytest.mark.skipif(
    not is_blackhole() or os.environ.get("TT_METAL_SIMULATOR") is not None,
    reason="SDPA precision recipes run on Blackhole hardware (the simulator disables SFPLOADMACRO)",
)

# Recipe and K/V storage. FAST inputs go through prepare_sdpa_input.
VARIANTS = {
    "standard": (ttnn.SDPAPrecision.STANDARD, ttnn.bfloat16),
    "balanced": (ttnn.SDPAPrecision.BALANCED, ttnn.bfloat16),
    "accurate": (ttnn.SDPAPrecision.ACCURATE, ttnn.bfloat16),
    "fast_bf16": (ttnn.SDPAPrecision.FAST, ttnn.bfloat16),
    "fast_bfp8": (ttnn.SDPAPrecision.FAST, ttnn.bfloat8_b),
    "fast_bfp4": (ttnn.SDPAPrecision.FAST, ttnn.bfloat4_b),
}
# Relative L2 error bound (%) vs FP64 attention on the BF16 inputs, for normally distributed inputs and
# K up to a few thousand.
L2_PCT_BOUND = {
    "standard": 3.5,
    "balanced": 0.8,
    "accurate": 0.6,
    "fast_bf16": 4.2,
    "fast_bfp8": 4.4,
    "fast_bfp4": 23.0,
}


def reference(q, k, v, mask=None):
    q, k, v = q.double(), k.double(), v.double()
    rep = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
    scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    if mask is not None:
        scores = scores + mask.double()
    return torch.softmax(scores, -1) @ v


def l2_pct(actual, expected):
    actual, expected = actual.double(), expected.double()
    assert torch.isfinite(actual).all()
    return 100 * ((actual - expected).norm() / expected.norm()).item()


def to_device(device, x, dtype=ttnn.bfloat16):
    return ttnn.from_torch(
        x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def inputs_for(device, variant, q, k, v):
    precision, kv_dtype = VARIANTS[variant]
    tq, tk, tv = (to_device(device, x) for x in (q, k, v))
    if precision == ttnn.SDPAPrecision.FAST:
        tq = ttnn.transformer.prepare_sdpa_input(tq, is_query=True)
        tk, tv = (ttnn.transformer.prepare_sdpa_input(x, is_query=False, dtype=kv_dtype) for x in (tk, tv))
    return tq, tk, tv


def program_config(device, q_chunk, k_chunk):
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=q_chunk,
        k_chunk_size=k_chunk,
    )


def randn(*shape, seed):
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed)).bfloat16()


def sdpa(device, variant, q, k, v, q_chunk, k_chunk, mask=None):
    return ttnn.transformer.scaled_dot_product_attention(
        *inputs_for(device, variant, q, k, v),
        is_causal=False,
        attn_mask=None if mask is None else to_device(device, mask),
        program_config=program_config(device, q_chunk, k_chunk),
        precision=VARIANTS[variant][0],
    )


# b, nh, nkv, sq, sk, d, q_chunk, k_chunk. Chunks need not divide the sequence lengths, which need not be
# tile multiples.
SHAPES = {
    "q256_k512_d128": (1, 2, 2, 512, 2048, 128, 256, 512),
    "q224_k384_d128": (1, 2, 2, 448, 1536, 128, 224, 384),
    "odd_q96_k160_d64": (1, 2, 2, 288, 640, 64, 96, 160),
    "subtile_tails": (1, 2, 2, 1000, 1500, 128, 256, 512),
    "gqa_batch2": (2, 8, 2, 256, 1024, 128, 128, 256),
    "q128_k256_d256": (1, 1, 1, 256, 1024, 256, 128, 256),
    "q32_k32_d32": (1, 1, 1, 96, 160, 32, 32, 32),
}


def check_accuracy(device, variant, shape):
    b, nh, nkv, sq, sk, d, q_chunk, k_chunk = shape
    q, k, v = randn(b, nh, sq, d, seed=1), randn(b, nkv, sk, d, seed=2), randn(b, nkv, sk, d, seed=3)
    actual = ttnn.to_torch(sdpa(device, variant, q, k, v, q_chunk, k_chunk))
    assert l2_pct(actual, reference(q, k, v)) < L2_PCT_BOUND[variant]


def check_attn_mask(device, variant, mask_kind):
    q, k, v = randn(1, 2, 512, 128, seed=7), randn(1, 2, 1500, 128, seed=8), randn(1, 2, 1500, 128, seed=9)
    generator = torch.Generator().manual_seed(10)
    if mask_kind == "random":
        mask = torch.randn(1, 1, 512, 1500, generator=generator)
        mask[torch.rand(mask.shape, generator=generator) < 0.2] = -math.inf
    else:
        mask = torch.zeros(1, 1, 512, 1500)
        mask[..., 1100:] = -math.inf
    mask = mask.bfloat16()
    actual = ttnn.to_torch(sdpa(device, variant, q, k, v, 256, 512, mask))
    assert l2_pct(actual, reference(q, k, v, mask)) < L2_PCT_BOUND[variant]


def check_joint(device, variant):
    q, k, v = randn(1, 2, 1000, 128, seed=11), randn(1, 2, 1000, 128, seed=12), randn(1, 2, 1000, 128, seed=13)
    jq, jk, jv = randn(1, 2, 77, 128, seed=14), randn(1, 2, 77, 128, seed=15), randn(1, 2, 77, 128, seed=16)
    tq, tk, tv = inputs_for(device, variant, q, k, v)
    tjq, tjk, tjv = inputs_for(device, variant, jq, jk, jv)
    out, joint_out = ttnn.transformer.joint_scaled_dot_product_attention(
        tq,
        tk,
        tv,
        tjq,
        tjk,
        tjv,
        joint_strategy="rear",
        program_config=program_config(device, 256, 512),
        precision=VARIANTS[variant][0],
    )
    expected = reference(torch.cat([q, jq], 2), torch.cat([k, jk], 2), torch.cat([v, jv], 2))
    actual = torch.cat([ttnn.to_torch(out), ttnn.to_torch(joint_out)], 2)
    assert l2_pct(actual, expected) < L2_PCT_BOUND[variant]


# b, nh, sq, sk, d, joint rows (0 = plain SDPA).
OP_SELECTED_SHAPES = {
    "self_attention": (1, 10, 4096, 4096, 128, 0),
    "short_k_cross": (1, 8, 4864, 256, 128, 0),
    "d256": (1, 8, 1024, 1024, 256, 0),
    "joint": (1, 4, 1000, 1000, 128, 77),
}


def check_op_selected_blocking(device, variant, shape):
    """Chunk sizes left to the op (no program_config, or zero chunk sizes)."""
    b, nh, sq, sk, d, joint = shape
    q, k, v = randn(b, nh, sq, d, seed=27), randn(b, nh, sk, d, seed=28), randn(b, nh, sk, d, seed=29)
    precision = VARIANTS[variant][0]
    if not joint:
        out = ttnn.transformer.scaled_dot_product_attention(
            *inputs_for(device, variant, q, k, v), is_causal=False, precision=precision
        )
        actual, expected = ttnn.to_torch(out), reference(q, k, v)
    else:
        jq, jk, jv = randn(b, nh, joint, d, seed=30), randn(b, nh, joint, d, seed=31), randn(b, nh, joint, d, seed=32)
        out, joint_out = ttnn.transformer.joint_scaled_dot_product_attention(
            *inputs_for(device, variant, q, k, v),
            *inputs_for(device, variant, jq, jk, jv),
            joint_strategy="rear",
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=device.compute_with_storage_grid_size()
            ),
            precision=precision,
        )
        actual = torch.cat([ttnn.to_torch(out), ttnn.to_torch(joint_out)], 2)
        expected = reference(torch.cat([q, jq], 2), torch.cat([k, jk], 2), torch.cat([v, jv], 2))
    assert l2_pct(actual, expected) < L2_PCT_BOUND[variant]
