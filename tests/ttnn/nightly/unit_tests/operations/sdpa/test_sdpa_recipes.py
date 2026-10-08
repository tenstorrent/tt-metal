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

pytestmark = recipe_hardware


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


@pytest.mark.parametrize("scale", [0.125, 1.0, 0.3])
@pytest.mark.parametrize("mask_dtype", [ttnn.bfloat16, ttnn.bfloat8_b], ids=["bf16", "bfp8"])
@pytest.mark.parametrize("variant", ["balanced", "accurate"])
def test_sdpa_recipe_narrow_mask_prescale(device, variant, mask_dtype, scale):
    """FP32-state recipes keep a BF16/BFP8 attn_mask narrow when its 1/scale pre-scale is exact (a power of two);
    other scales widen it to FP32 first. Either way the result is bitwise that of the same mask values given as
    FP32. FP64 bound on the stored mask."""
    q, k, v = randn(2, 4, 600, 64, seed=60), randn(2, 4, 600, 64, seed=61), randn(2, 4, 600, 64, seed=62)
    generator = torch.Generator().manual_seed(63)
    mask = torch.randn(2, 1, 600, 600, generator=generator) * 3
    mask[torch.rand(mask.shape, generator=generator) < 0.2] = -1e9
    mask = stored(mask.bfloat16(), mask_dtype).float()
    narrow, wide = to_device(device, mask, mask_dtype), to_device(device, mask, ttnn.float32)
    tensors = [to_device(device, x) for x in (q, k, v)]
    run = lambda m: ttnn.to_torch(
        ttnn.transformer.scaled_dot_product_attention(
            *tensors,
            is_causal=False,
            attn_mask=m,
            scale=scale,
            program_config=program_config(device, 128, 256),
            precision=VARIANTS[variant][0],
        )
    )
    actual = run(narrow)
    assert torch.equal(actual, run(wide))
    assert l2_pct(actual, reference(q, k, v, mask, scale)) < L2_PCT_BOUND[variant]


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


# Key ranges on small grids: many snake rounds (alternating deal direction, a partial last round) with K/V passed
# between the cores of one head (causal, sliding window, chunked prefill, windowed), and more heads than cores (no
# sharing).
KEY_RANGE_GRID_CASES = {
    "causal_5_cores_7_rounds": lambda d, v: check_key_range(
        d, v, (1, 2, 2, 1024, 64, 64, 128), causal=True, grid=ttnn.CoreCoord(5, 1)
    ),
    "causal_gqa_batch2_9_cores": lambda d, v: check_key_range(
        d, v, (2, 4, 2, 768, 128, 128, 256), causal=True, grid=ttnn.CoreCoord(3, 3)
    ),
    "causal_heads_over_cores": lambda d, v: check_key_range(
        d, v, (1, 4, 4, 512, 64, 128, 128), causal=True, grid=ttnn.CoreCoord(2, 1)
    ),
    "window300_8_cores": lambda d, v: check_key_range(
        d, v, (1, 3, 3, 1000, 128, 128, 256), causal=True, window=300, grid=ttnn.CoreCoord(4, 2)
    ),
    "centred_window300_7_cores": lambda d, v: check_key_range(
        d, v, (1, 2, 2, 1000, 64, 64, 128), causal=False, window=300, grid=ttnn.CoreCoord(7, 1)
    ),
    "chunked_start700_6_cores": lambda d, v: check_chunked(
        d, v, 700, sq=1024, block=2048, nh=2, nkv=1, grid=ttnn.CoreCoord(3, 2)
    ),
    "windowed_5_cores": lambda d, v: check_windowed(
        d, v, [0, 100, 356, 357, 800, 1024], causal=True, chunks=(64, 128), grid=ttnn.CoreCoord(5, 1)
    ),
}


@pytest.mark.parametrize("case", KEY_RANGE_GRID_CASES.keys())
@pytest.mark.parametrize("variant", ["standard", "accurate", "fast_bfp8"])
def test_sdpa_recipe_key_range_small_grid(device, variant, case):
    KEY_RANGE_GRID_CASES[case](device, variant)


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


# Program-cache hits on new inputs (check_cache_hit_rebinds), the layouts the unit file does not cover.
@pytest.mark.parametrize("case", ["joint", "windowed_offset_tensor"])
def test_sdpa_recipe_cache_hit_rebinds(device, case):
    check_cache_hit_rebinds(device, case)
