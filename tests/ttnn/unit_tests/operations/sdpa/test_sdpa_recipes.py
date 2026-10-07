# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SDPA precision recipes (ttnn.SDPAPrecision): the fast subset that runs in the ttnn sanity sdpa group.

Every recipe once, plus masks, joint attention, op-selected blocking, program cache and trace, rejected arguments
and prepare_sdpa_input. The sweeps (all shapes, long K, rising maxima, every recipe for masks / joint / blocking)
are in tests/ttnn/nightly/unit_tests/operations/sdpa/test_sdpa_recipes.py.
"""

import pytest
import torch
import ttnn

from tests.ttnn.unit_tests.operations.sdpa.sdpa_recipe_test_utils import (
    OP_SELECTED_SHAPES,
    SHAPES,
    VARIANTS,
    blackhole_only,
    check_accuracy,
    check_attn_mask,
    check_joint,
    check_op_selected_blocking,
    inputs_for,
    l2_pct,
    program_config,
    randn,
    to_device,
)

pytestmark = blackhole_only


@pytest.mark.parametrize("variant", VARIANTS)
def test_sdpa_recipe_accuracy(device, variant):
    check_accuracy(device, variant, SHAPES["subtile_tails"])


@pytest.mark.parametrize("variant, mask_kind", [("standard", "key_padding"), ("fast_bfp8", "random")])
def test_sdpa_recipe_attn_mask(device, variant, mask_kind):
    check_attn_mask(device, variant, mask_kind)


def test_joint_sdpa_recipe(device):
    check_joint(device, "standard")


def test_sdpa_recipe_op_selected_blocking(device):
    check_op_selected_blocking(device, "standard", OP_SELECTED_SHAPES["short_k_cross"])


@pytest.mark.parametrize("variant", ["standard", "accurate", "fast_bfp8"])
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
        "packed_kv_for_bf16_recipe",
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
    elif invalid == "packed_kv_for_bf16_recipe":
        kwargs["precision"] = ttnn.SDPAPrecision.STANDARD
        tensors[1] = to_device(device, k, ttnn.bfloat8_b)
    elif invalid == "l1_output":
        kwargs["memory_config"] = ttnn.L1_MEMORY_CONFIG
    elif invalid == "mask_shape":
        kwargs["attn_mask"] = to_device(device, torch.zeros(1, 1, 256, 256))
    elif invalid == "fp32_mask_for_bf16_recipe":
        kwargs["precision"] = ttnn.SDPAPrecision.STANDARD
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
