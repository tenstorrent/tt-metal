# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SDPA precision recipes (ttnn.SDPAPrecision) against an FP64 reference.

Recipes and their numerics: tech_reports/FlashAttention/SDPAPrecisionRecipes.md.
"""

import math
import os

import pytest
import torch
import ttnn

from models.common.utility_functions import is_blackhole

pytestmark = pytest.mark.skipif(
    not is_blackhole() or os.environ.get("TT_METAL_SIMULATOR") is not None,
    reason="SDPA precision recipes run on Blackhole hardware (the simulator disables SFPLOADMACRO)",
)

# Recipe and K/V storage. LOW_PRECISION inputs go through prepare_sdpa_input.
VARIANTS = {
    "fast": (ttnn.SDPAPrecision.FAST, ttnn.bfloat16),
    "compensated": (ttnn.SDPAPrecision.COMPENSATED, ttnn.bfloat16),
    "balanced": (ttnn.SDPAPrecision.BALANCED, ttnn.bfloat16),
    "accurate": (ttnn.SDPAPrecision.ACCURATE, ttnn.bfloat16),
    "low_precision_bf16": (ttnn.SDPAPrecision.LOW_PRECISION, ttnn.bfloat16),
    "low_precision_bfp8": (ttnn.SDPAPrecision.LOW_PRECISION, ttnn.bfloat8_b),
    "low_precision_bfp4": (ttnn.SDPAPrecision.LOW_PRECISION, ttnn.bfloat4_b),
}
# Relative L2 error bound (%) vs FP64 attention on the BF16 inputs, for normally distributed inputs and
# K up to a few thousand.
L2_PCT_BOUND = {
    "fast": 3.5,
    "compensated": 3.5,
    "balanced": 0.8,
    "accurate": 0.6,
    "low_precision_bf16": 4.2,
    "low_precision_bfp8": 4.4,
    "low_precision_bfp4": 23.0,
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
    if precision == ttnn.SDPAPrecision.LOW_PRECISION:
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


@pytest.mark.parametrize("shape", SHAPES.values(), ids=SHAPES.keys())
@pytest.mark.parametrize("variant", VARIANTS)
def test_sdpa_recipe_accuracy(device, variant, shape):
    b, nh, nkv, sq, sk, d, q_chunk, k_chunk = shape
    q, k, v = randn(b, nh, sq, d, seed=1), randn(b, nkv, sk, d, seed=2), randn(b, nkv, sk, d, seed=3)
    actual = ttnn.to_torch(sdpa(device, variant, q, k, v, q_chunk, k_chunk))
    assert l2_pct(actual, reference(q, k, v)) < L2_PCT_BOUND[variant]


# Long K with small logits: FAST's BF16 running state swamps (about 3.9%), COMPENSATED's FP32 state does not
# (about 1.7%). The bounds encode that ordering.
LONG_K_BOUND = {**L2_PCT_BOUND, "fast": 5.0, "compensated": 2.2, "low_precision_bf16": 2.6, "low_precision_bfp8": 2.7}


@pytest.mark.parametrize("variant", VARIANTS)
def test_sdpa_recipe_long_k(device, variant):
    q, k, v = randn(1, 1, 256, 128, seed=4), randn(1, 1, 32768, 128, seed=5), randn(1, 1, 32768, 128, seed=6)
    q, k = (q * 0.25).bfloat16(), (k * 0.25).bfloat16()
    actual = ttnn.to_torch(sdpa(device, variant, q, k, v, 256, 512))
    assert l2_pct(actual, reference(q, k, v)) < LONG_K_BOUND[variant]


@pytest.mark.parametrize("mask_kind", ["random", "key_padding"])
@pytest.mark.parametrize("variant", VARIANTS)
def test_sdpa_recipe_attn_mask(device, variant, mask_kind):
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


@pytest.mark.parametrize("variant", VARIANTS)
def test_joint_sdpa_recipe(device, variant):
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


@pytest.mark.parametrize("k_length", [512, 1536, 32768])
def test_sdpa_fast_matches_legacy(device, k_length):
    """FAST is the legacy streaming kernel: bit-identical to precision=None at the same chunks."""
    q, k, v = randn(1, 5, 768, 128, seed=17), randn(1, 5, k_length, 128, seed=18), randn(1, 5, k_length, 128, seed=19)
    tensors = [to_device(device, x) for x in (q, k, v)]
    cfg = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(5, 2), q_chunk_size=256, k_chunk_size=512)
    legacy = ttnn.transformer.scaled_dot_product_attention(*tensors, is_causal=False, program_config=cfg)
    fast = ttnn.transformer.scaled_dot_product_attention(
        *tensors, is_causal=False, program_config=cfg, precision=ttnn.SDPAPrecision.FAST
    )
    assert torch.equal(ttnn.to_torch(fast), ttnn.to_torch(legacy))


@pytest.mark.parametrize("variant", ["fast", "compensated", "accurate", "low_precision_bfp8"])
@pytest.mark.parametrize("device_params", [{"trace_region_size": 4194304}], indirect=True)
def test_sdpa_recipe_program_cache_and_trace(device, variant):
    device.enable_program_cache()
    q, k, v = randn(1, 2, 768, 128, seed=20), randn(1, 2, 1536, 128, seed=21), randn(1, 2, 1536, 128, seed=22)
    inputs = inputs_for(device, variant, q, k, v)
    cfg = program_config(device, 256, 512)
    run = lambda tensors: ttnn.transformer.scaled_dot_product_attention(
        *tensors, is_causal=False, program_config=cfg, precision=VARIANTS[variant][0]
    )
    first = ttnn.to_torch(run(inputs))
    entries = device.num_program_cache_entries()
    # A cache hit with new buffers must read the new addresses: negated V negates the output (up to rounding,
    # which is not sign-symmetric), while a stale address would return the first output.
    negated = inputs_for(device, variant, q, k, -v)
    assert l2_pct(ttnn.to_torch(run(negated)), -first) < 1.0
    assert device.num_program_cache_entries() == entries
    trace = ttnn.begin_trace_capture(device, cq_id=0)
    traced = run(inputs)
    ttnn.end_trace_capture(device, trace, cq_id=0)
    try:
        for _ in range(2):
            ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
            assert torch.equal(ttnn.to_torch(traced), first)
    finally:
        ttnn.release_trace(device, trace)


@pytest.mark.parametrize(
    "invalid",
    [
        "causal",
        "sliding_window",
        "attention_sink",
        "compute_kernel_config",
        "exp_approx_mode_false",
        "sub_core_grids",
        "non_default_scale",
        "packed_kv_for_fast",
        "padded_head_dim",
        "l1_output",
        "mask_shape",
        "fp32_mask_for_bf16_recipe",
    ],
)
def test_sdpa_recipe_rejects_unsupported(expect_error, device, invalid):
    q, k, v = randn(1, 1, 256, 128, seed=23), randn(1, 1, 512, 128, seed=24), randn(1, 1, 512, 128, seed=25)
    if invalid == "padded_head_dim":
        q, k, v = (x[..., :80].contiguous() for x in (q, k, v))
    tensors = [to_device(device, x) for x in (q, k, v)]
    kwargs = dict(is_causal=False, precision=ttnn.SDPAPrecision.ACCURATE)
    cfg = dict(compute_with_storage_grid_size=(1, 1), q_chunk_size=256, k_chunk_size=512)
    if invalid == "causal":
        kwargs["is_causal"] = True
    elif invalid == "sliding_window":
        kwargs["sliding_window_size"] = 256
    elif invalid == "attention_sink":
        kwargs["attention_sink"] = tensors[0]
    elif invalid == "compute_kernel_config":
        kwargs["compute_kernel_config"] = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4)
    elif invalid == "exp_approx_mode_false":
        cfg["exp_approx_mode"] = False
    elif invalid == "sub_core_grids":
        cfg["sub_core_grids"] = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    elif invalid == "non_default_scale":
        kwargs["scale"] = 0.5
    elif invalid == "packed_kv_for_fast":
        kwargs["precision"] = ttnn.SDPAPrecision.FAST
        tensors[1] = to_device(device, k, ttnn.bfloat8_b)
    elif invalid == "l1_output":
        kwargs["memory_config"] = ttnn.L1_MEMORY_CONFIG
    elif invalid == "mask_shape":
        kwargs["attn_mask"] = to_device(device, torch.zeros(1, 1, 256, 256))
    elif invalid == "fp32_mask_for_bf16_recipe":
        kwargs["precision"] = ttnn.SDPAPrecision.COMPENSATED
        kwargs["attn_mask"] = to_device(device, torch.zeros(1, 1, 256, 512), ttnn.float32)
    kwargs["program_config"] = ttnn.SDPAProgramConfig(**cfg)
    before = device.num_program_cache_entries()
    with expect_error(RuntimeError, "SDPA|recipe|precision"):
        ttnn.transformer.scaled_dot_product_attention(*tensors, **kwargs)
    assert device.num_program_cache_entries() == before, "Unsupported arguments must fail before dispatch"


def round_significand(values, bits):
    values = values.double()
    step = torch.ldexp(torch.ones_like(values), torch.frexp(values.abs())[1] - bits)
    return ((values / step).round() * step).float()


def round_block(values, bits, *, ties_even):
    groups = values.double().reshape(-1, 16)
    step = torch.ldexp(torch.ones_like(groups[:, :1]), torch.frexp(groups.abs().amax(-1, keepdim=True))[1] - bits)
    scaled = groups.abs() / step
    integer = scaled.round() if ties_even else (scaled + 0.5).floor()
    return (integer.clamp_max(2**bits - 1) * step * groups.sign()).float().reshape(values.shape)


@pytest.mark.parametrize(
    "is_query, dtype",
    [(True, ttnn.bfloat16), (False, ttnn.bfloat16), (False, ttnn.bfloat8_b), (False, ttnn.bfloat4_b)],
    ids=["q", "kv_bf16", "kv_bfp8", "kv_bfp4"],
)
@pytest.mark.parametrize("distribution", ["normal", "ties"])
@pytest.mark.parametrize("length", [256, 17])
def test_prepare_sdpa_input(device, is_query, dtype, distribution, length):
    shape = (1, 1, length, 128)
    if distribution == "normal":
        host = randn(*shape, seed=26)
    else:
        # Every BF16 mantissa, both signs, a spread of exponents and group maxima: ties and saturation.
        index = torch.arange(length * 128).reshape(-1, 16)
        values = ((index * 17) % 256).float() / 128
        values[:, 0] = 1.75
        values = torch.ldexp(values, (index[:, :1] // 16 % 33 - 16).int())
        values *= torch.where(index % 2 == 0, 1.0, -1.0)
        host = values.reshape(shape).bfloat16()
    if dtype == ttnn.bfloat4_b:
        expected = round_block(host, 3, ties_even=True)
    else:
        expected = round_significand(host, 7 if is_query else 5)
        if dtype == ttnn.bfloat8_b:
            # BFP8 packing rounds shared-exponent ties away from zero; RNE5 values are exact at its ingress.
            expected = round_block(expected, 7, ties_even=False)
    source = ttnn.from_torch(host, device=device, layout=ttnn.TILE_LAYOUT, pad_value=float("nan"))
    output = ttnn.transformer.prepare_sdpa_input(source, is_query=is_query, dtype=dtype)
    assert output.dtype == dtype
    assert torch.equal(ttnn.to_torch(output).float(), expected)
    assert torch.equal(ttnn.to_torch(source), host)
    if length % 32:
        padding = ttnn.to_torch(ttnn.reshape(output, output.padded_shape))[..., length:, :]
        assert torch.count_nonzero(padding) == 0
