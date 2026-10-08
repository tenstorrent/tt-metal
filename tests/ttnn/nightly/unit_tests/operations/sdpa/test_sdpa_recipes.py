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
    CAUSAL_SHAPES,
    L2_PCT_BOUND,
    OP_SELECTED_SHAPES,
    SHAPES,
    VARIANTS,
    blackhole_only,
    check_accuracy,
    check_attn_mask,
    check_chunked,
    check_chunked_trace,
    check_concat_heads,
    check_joint,
    check_key_range,
    check_legacy_arguments,
    check_mla,
    check_op_selected_blocking,
    check_sink,
    check_windowed,
    MLA_SHAPES,
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


@pytest.mark.parametrize("shape", CAUSAL_SHAPES.values(), ids=CAUSAL_SHAPES.keys())
@pytest.mark.parametrize("variant", VARIANTS)
def test_sdpa_recipe_causal(device, variant, shape):
    check_key_range(device, variant, shape, causal=True)


# Windows narrower than a K chunk (rows with no visible key in a Q chunk's first K chunk), off tile and chunk
# boundaries, and wider than the sequence's chunks; causal (left) and centred.
@pytest.mark.parametrize("window", [64, 300, 1100])
@pytest.mark.parametrize("causal", [True, False], ids=["causal", "centred"])
@pytest.mark.parametrize("shape", ["q256_k512", "odd_q96_k160_d64", "gqa_batch2"])
@pytest.mark.parametrize("variant", VARIANTS)
def test_sdpa_recipe_sliding_window(device, variant, shape, causal, window):
    check_key_range(device, variant, CAUSAL_SHAPES[shape], causal=causal, window=window)


# Chunk offsets on and off chunk boundaries, Q chunks over and under the K chunk, GQA and batch (the page table
# reverses the cache blocks), a sliding window, and the offset read on device.
CHUNKED_CASES = {
    "start0": dict(start=0),
    "start512": dict(start=512),
    "start96_off_chunk": dict(start=96),
    "start700_q256_k128": dict(start=700, q_chunk=256, k_chunk=128),
    "gqa_batch2": dict(start=384, b=2, nh=8, nkv=2),
    "tail_sq200": dict(start=320, sq=200, block=640),
    "window300": dict(start=640, window=300),
    "more_heads_than_cores": dict(start=128, b=8, nh=24, nkv=8, block=512),
    # Paged K/V: shuffled blocks per sequence, smaller than and not aligned to the K chunk; a cache declared in another
    # layer's geometry (paged_cache_geometry); attention sinks.
    "paged_5x128": dict(start=320, blocks_per_seq=5, block=128),
    "paged_gqa_batch2_4x96": dict(start=200, b=2, nh=8, nkv=2, sq=96, blocks_per_seq=4, block=96),
    "paged_geometry": dict(start=256, blocks_per_seq=2, block=256, cache_shape=(1, 512, 128)),
    "sink": dict(start=384, sink=True),
}


@pytest.mark.parametrize("as_tensor", [False, True], ids=["scalar_start", "tensor_start"])
@pytest.mark.parametrize("case", CHUNKED_CASES.values(), ids=CHUNKED_CASES.keys())
@pytest.mark.parametrize("variant", VARIANTS)
def test_chunked_sdpa_recipe(device, variant, case, as_tensor):
    check_chunked(device, variant, as_tensor=as_tensor, **case)


@pytest.mark.parametrize("variant", ["standard", "balanced", "accurate", "fast_bfp8"])
@pytest.mark.parametrize("device_params", [{"trace_region_size": 4194304}], indirect=True)
def test_chunked_sdpa_recipe_trace(device, variant):
    check_chunked_trace(device, variant, [512, 0, 96, 768])


# Windows of length 1 and off tile boundaries; Q slices at an offset (scalar or per-device tensor).
WINDOWED_CASES = {
    "full_q": dict(cu=[0, 100, 356, 357, 800, 1024]),
    "q_slice": dict(cu=[0, 100, 356, 357, 800, 1024], q_rows=512, q_offset=256),
    "q_slice_tensor": dict(cu=[0, 100, 356, 357, 800, 1024], q_rows=512, q_offset=512, offset_as_tensor=True),
    "uniform_q64_k512": dict(cu=list(range(0, 2049, 256)), chunks=(64, 512)),
}


@pytest.mark.parametrize("causal", [False, True], ids=["bidir", "causal"])
@pytest.mark.parametrize("case", WINDOWED_CASES.values(), ids=WINDOWED_CASES.keys())
@pytest.mark.parametrize("variant", VARIANTS)
def test_windowed_sdpa_recipe(device, variant, case, causal):
    check_windowed(device, variant, causal=causal, **case)


@pytest.mark.parametrize("variant", VARIANTS)
def test_chunked_flash_mla_prefill_recipe(device, variant):
    """chunked_flash_mla_prefill: paged K [blocks, 1, block, 192] shared by four heads, V its first 128 columns."""
    check_chunked(device, variant, 256, nh=4, nkv=1, sq=128, d=192, head_dim_v=128, blocks_per_seq=3, block=128)


@pytest.mark.parametrize("v_tensor", [False, True], ids=["v_from_k", "v_tensor"])
@pytest.mark.parametrize("causal", [True, False], ids=["causal", "noncausal"])
@pytest.mark.parametrize("shape", ["d192_v128", "d576_v512"])
@pytest.mark.parametrize("variant", VARIANTS)
def test_flash_mla_prefill_recipe(device, variant, shape, causal, v_tensor):
    check_mla(device, variant, MLA_SHAPES[shape], causal=causal, v_tensor=v_tensor)


@pytest.mark.parametrize("variant", VARIANTS)
def test_flash_mla_prefill_recipe_tails(device, variant):
    """Sequence lengths off the tile and chunk sizes, batch 2, three heads, noncausal (Sq != Sk)."""
    check_mla(device, variant, MLA_SHAPES["d128_v64_tails"], causal=False)


# Sinks: noncausal, causal and sliding-window rows; a dominant sink (scaled logit about 5 above the row maxima, most of
# each row's weight); an FP32 sink tensor.
SINK_CASES = {
    "noncausal": dict(),
    "causal": dict(causal=True, shape=(1, 4, 2, 1024, 1024, 128, 256, 512)),
    "window": dict(causal=True, window=200, shape=(1, 4, 2, 1024, 1024, 128, 256, 512)),
    "dominant": dict(sink_offset=90.0),
    "fp32_sink": dict(sink_dtype=ttnn.float32),
}


@pytest.mark.parametrize("case", SINK_CASES.values(), ids=SINK_CASES.keys())
@pytest.mark.parametrize("variant", VARIANTS)
def test_sdpa_recipe_attention_sink(device, variant, case):
    check_sink(device, variant, **case)


@pytest.mark.parametrize("causal", [False, True], ids=["noncausal", "causal"])
@pytest.mark.parametrize("variant", VARIANTS)
def test_sdpa_recipe_output_concat_heads(device, variant, causal):
    shape = (2, 4, 2, 640, 640, 64, 128, 256) if causal else (2, 4, 2, 300, 640, 64, 128, 256)
    check_concat_heads(device, variant, causal=causal, shape=shape)
