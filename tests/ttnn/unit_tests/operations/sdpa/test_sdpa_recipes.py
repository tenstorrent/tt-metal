# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SDPA precision recipes (ttnn.SDPAPrecision): the fast subset that runs in the ttnn sanity sdpa group.

Every recipe once, plus masks, causal / sliding-window / chunked / windowed key ranges, paged K/V, MLA, attention
sinks, concatenated-heads output, joint attention, op-selected blocking, program cache and trace, precision routing,
rejected arguments and prepare_sdpa_input. The sweeps (all shapes, long K, rising maxima, every recipe for masks / joint / blocking)
are in tests/ttnn/nightly/unit_tests/operations/sdpa/test_sdpa_recipes.py.
"""

import pytest
import torch
import ttnn

from tests.ttnn.unit_tests.operations.sdpa.sdpa_recipe_test_utils import (
    CAUSAL_SHAPES,
    L2_PCT_BOUND,
    OP_SELECTED_SHAPES,
    SHAPES,
    VARIANTS,
    recipe_hardware,
    check_accuracy,
    check_attn_mask,
    check_cache_hit_rebinds,
    check_chunked,
    check_chunked_trace,
    check_concat_heads,
    check_joint,
    check_key_range,
    check_legacy_arguments,
    check_mixed_kv,
    check_mla,
    check_op_selected_blocking,
    check_routing,
    check_sink,
    check_windowed,
    fp32_dest_config,
    MLA_SHAPES,
    inputs_for,
    l2_pct,
    program_config,
    randn,
    reference,
    to_device,
)

pytestmark = recipe_hardware


@pytest.mark.parametrize("variant", VARIANTS)
def test_sdpa_recipe_accuracy(device, variant):
    check_accuracy(device, variant, SHAPES["subtile_tails"])


@pytest.mark.parametrize("variant, mask_kind", [("standard", "key_padding"), ("fast_bfp8", "random")])
def test_sdpa_recipe_attn_mask(device, variant, mask_kind):
    check_attn_mask(device, variant, mask_kind)


# Key ranges, one case per recipe family: the reference-max recipes' fused chunks between masked edge chunks, and the
# FP32-state recipes. Sliding windows leave rows with no visible key in a Q chunk's first K chunk.
@pytest.mark.parametrize("variant", ["standard", "accurate"])
def test_sdpa_recipe_causal(device, variant):
    check_key_range(device, variant, CAUSAL_SHAPES["subtile_tails"], causal=True)


@pytest.mark.parametrize("variant, causal", [("fast_bfp8", True), ("balanced", False)], ids=["causal", "centred"])
def test_sdpa_recipe_sliding_window(device, variant, causal):
    check_key_range(device, variant, CAUSAL_SHAPES["q256_k512"], causal=causal, window=300)


@pytest.mark.parametrize("device_params", [{"trace_region_size": 4194304}], indirect=True)
def test_chunked_sdpa_recipe_trace(device):
    check_chunked_trace(device, "standard", [512, 96])


# Paged K/V: five shuffled cache blocks per sequence, declared in another layer's geometry (paged_cache_geometry).
def test_chunked_sdpa_recipe_paged(device):
    check_chunked(device, "balanced", 320, blocks_per_seq=5, block=128, cache_shape=(1, 256, 128))


# MLA: V is K's first 128 columns of 192.
def test_flash_mla_prefill_recipe(device):
    check_mla(device, "standard", MLA_SHAPES["d192_v128"])


def test_sdpa_recipe_attention_sink(device):
    check_sink(device, "accurate", causal=True, shape=(1, 4, 2, 512, 512, 128, 256, 256))


def test_sdpa_recipe_output_concat_heads(device):
    check_concat_heads(device, "fast_bfp8")


def test_windowed_sdpa_recipe(device):
    check_windowed(device, "accurate", [0, 100, 356, 357, 800, 1024], causal=True, q_rows=512, q_offset=256)


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


# A cache hit with new inputs rebinds every buffer: the dense layout (mask, sink) and the key-range layout (page table,
# Q offset tensor, sink). Joint and windowed are in the nightly file.
@pytest.mark.parametrize("case", ["dense_mask_sink", "chunked_paged_sink"])
def test_sdpa_recipe_cache_hit_rebinds(device, case):
    check_cache_hit_rebinds(device, case)


# What legacy callers pass, on the recipes the legacy routes move to: STANDARD and ACCURATE with BFP8 Q/K/V (the
# output comes back as BFP8), a custom scale with an attn_mask, L1 inputs and output, an FP32-dest HiFi4
# compute_kernel_config with exp_approx_mode=False (ignored), and six batch/heads (GQA) on a 2x2 grid, so each
# core runs Q chunks of several heads. The sweeps are in the nightly file.
@pytest.mark.parametrize("variant", ["standard", "accurate"])
def test_sdpa_recipe_legacy_arguments(device, variant):
    check_legacy_arguments(
        device,
        variant,
        q_dtype=ttnn.bfloat8_b,
        kv_dtype=ttnn.bfloat8_b,
        scale=0.3,
        mask=True,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        compute_kernel_config=True,
        shape=(2, 3, 1, 288, 640, 64, 96, 160),
        grid=(2, 2),
    )


# Precision routing (no `precision`): FP32 dest runs ACCURATE at op-chosen blocking (dense causal with BFP8 Q/K/V and
# a custom scale; an attn_mask call passing Q2048; an attn_mask call whose 500 keys fit one K chunk; chunked prefill
# from a start tensor), non-ring joint runs STANDARD (ACCURATE with FP32 dest), also at op-chosen blocking.
@pytest.mark.parametrize(
    "case",
    [
        "dense_causal_bfp8",
        "dense_mask_q2048",
        "dense_mask_short_k",
        "chunked_tensor_start",
        "joint_bf16_dest",
        "joint_fp32_dest",
        "dense_d512",
        "dense_mixed_kv",
        "joint_empty",
    ],
)
def test_sdpa_precision_routing(device, case):
    check_routing(device, case)


# K and V in different formats: the fused reference-max path (K BFP8, V BF16) and the FP32-state path (K BF16, V BFP8).
@pytest.mark.parametrize(
    "variant, k_dtype, v_dtype",
    [("standard", ttnn.bfloat8_b, ttnn.bfloat16), ("accurate", ttnn.bfloat16, ttnn.bfloat8_b)],
    ids=["standard_kbfp8_vbf16", "accurate_kbf16_vbfp8"],
)
def test_sdpa_recipe_mixed_kv_dtypes(device, variant, k_dtype, v_dtype):
    check_mixed_kv(device, variant, k_dtype, v_dtype)


# Routed calls with L1 outputs back to back in one trace capture: the program hash sees the L1 the layout may use,
# not the raw free L1, so the second call (with the first output still live) reuses the warm-up's program.
@pytest.mark.parametrize("device_params", [{"trace_region_size": 4194304}], indirect=True)
def test_sdpa_recipe_trace_l1_outputs(device):
    device.enable_program_cache()
    q, k, v = randn(2, 4, 384, 64, seed=36), randn(2, 4, 384, 64, seed=37), randn(2, 4, 384, 64, seed=38)
    tensors = [to_device(device, x) for x in (q, k, v)]
    run = lambda: ttnn.transformer.scaled_dot_product_attention(
        *tensors,
        is_causal=False,
        program_config=program_config(device, 128, 256),
        compute_kernel_config=fp32_dest_config(device),
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    warm = run()
    expected = ttnn.to_torch(warm)
    warm.deallocate()
    entries = device.num_program_cache_entries()
    trace = ttnn.begin_trace_capture(device, cq_id=0)
    outputs = [run() for _ in range(3)]
    ttnn.end_trace_capture(device, trace, cq_id=0)
    try:
        ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
        for output in outputs:
            assert torch.equal(ttnn.to_torch(output), expected)
    finally:
        ttnn.release_trace(device, trace)
    assert device.num_program_cache_entries() == entries
    assert l2_pct(expected, reference(q, k, v)) < L2_PCT_BOUND["accurate"]


@pytest.mark.parametrize(
    "invalid",
    [
        "causal_sq_ne_sk",
        "causal_with_attn_mask",
        "attention_sink_shape",
        "sub_core_grids",
        "zero_scale",
        "sharded_output",
        "padded_head_dim",
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
    if invalid == "causal_sq_ne_sk":
        kwargs["is_causal"] = True
    elif invalid == "causal_with_attn_mask":
        kwargs["is_causal"] = True
        tensors[1], tensors[2] = tensors[0], tensors[0]
        kwargs["attn_mask"] = to_device(device, torch.zeros(1, 1, 256, 256))
    elif invalid == "attention_sink_shape":
        kwargs["attention_sink"] = tensors[0]
    elif invalid == "sub_core_grids":
        cfg["sub_core_grids"] = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    elif invalid == "zero_scale":
        kwargs["scale"] = 0.0
    elif invalid == "sharded_output":
        kwargs["memory_config"] = ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG
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
