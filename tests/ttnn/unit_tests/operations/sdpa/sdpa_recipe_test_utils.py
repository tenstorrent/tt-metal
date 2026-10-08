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


def reference(q, k, v, mask=None, scale=None):
    q, k, v = q.double(), k.double(), v.double()
    rep = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
    scores = (q @ k.transpose(-1, -2)) * (1 / math.sqrt(q.shape[-1]) if scale is None else scale)
    if mask is not None:
        scores = scores + mask.double()
    return torch.softmax(scores, -1) @ v


def l2_pct(actual, expected):
    actual, expected = actual.double(), expected.double()
    assert torch.isfinite(actual).all()
    return 100 * ((actual - expected).norm() / expected.norm()).item()


def to_device(device, x, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)


def stored(x, dtype):
    """The values x holds once stored as dtype (BFP8/BFP4 share an exponent per 16 values); the FP64 reference
    of a packed-input call uses these, so the bound measures the recipe, not the input format."""
    if dtype == ttnn.bfloat16:
        return x
    return ttnn.to_torch(ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT)).bfloat16()


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


def sdpa(device, variant, q, k, v, q_chunk, k_chunk, mask=None, scale=None):
    return ttnn.transformer.scaled_dot_product_attention(
        *inputs_for(device, variant, q, k, v),
        is_causal=False,
        scale=scale,
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


def check_legacy_arguments(
    device,
    variant,
    *,
    q_dtype=ttnn.bfloat16,
    kv_dtype=ttnn.bfloat16,
    scale=None,
    mask=False,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    compute_kernel_config=False,
    shape=(1, 2, 2, 288, 640, 64, 96, 160),
    grid=None,
    q_multiplier=1.0,
):
    """Arguments legacy SDPA callers pass, with a recipe: a custom scale (with an attn_mask, which is pre-scaled by
    1/scale), BFP8/BFP4 Q and K/V, L1 inputs and output, and a compute_kernel_config plus exp_approx_mode=False
    (ignored: the recipe owns the numerics). FP64 reference on the stored input values. Returns the output."""
    b, nh, nkv, sq, sk, d, q_chunk, k_chunk = shape
    precision = VARIANTS[variant][0]
    q, k, v = randn(b, nh, sq, d, seed=33), randn(b, nkv, sk, d, seed=34), randn(b, nkv, sk, d, seed=35)
    q = (q * q_multiplier).bfloat16()
    if precision == ttnn.SDPAPrecision.FAST:
        assert q_dtype == ttnn.bfloat16 and kv_dtype == ttnn.bfloat16, "FAST inputs come from prepare_sdpa_input"
        tq, tk, tv = inputs_for(device, variant, q, k, v)
        if memory_config != ttnn.DRAM_MEMORY_CONFIG:
            tq, tk, tv = (ttnn.to_memory_config(x, memory_config) for x in (tq, tk, tv))
    else:
        q, k, v = stored(q, q_dtype), stored(k, kv_dtype), stored(v, kv_dtype)
        tq = to_device(device, q, q_dtype, memory_config)
        tk, tv = (to_device(device, x, kv_dtype, memory_config) for x in (k, v))
    host_mask, kwargs = None, {}
    if mask:
        generator = torch.Generator().manual_seed(36)
        host_mask = torch.randn(1, 1, sq, sk, generator=generator)
        host_mask[torch.rand(host_mask.shape, generator=generator) < 0.2] = -math.inf
        host_mask = host_mask.bfloat16()
        kwargs["attn_mask"] = to_device(device, host_mask, memory_config=memory_config)
    if compute_kernel_config:
        kwargs["compute_kernel_config"] = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
    out = ttnn.transformer.scaled_dot_product_attention(
        tq,
        tk,
        tv,
        is_causal=False,
        scale=scale,
        memory_config=memory_config,
        program_config=ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid or device.compute_with_storage_grid_size(),
            q_chunk_size=q_chunk,
            k_chunk_size=k_chunk,
            exp_approx_mode=False if compute_kernel_config else None,
        ),
        precision=precision,
        **kwargs,
    )
    assert out.dtype == q_dtype and out.memory_config() == memory_config
    actual = ttnn.to_torch(out)
    if precision == ttnn.SDPAPrecision.FAST:
        q, k, v = (ttnn.to_torch(x) for x in (tq, tk, tv))
    expected = reference(q, k, v, host_mask, scale)
    # A BFP8/BFP4 output adds its own rounding (legacy SDPA returns Q's dtype too): allow a multiple of the error of
    # storing the exact result in that format on the host. The device's BFP4 packing loses about 1.8x that.
    factor = {ttnn.bfloat16: 0.0, ttnn.bfloat8_b: 1.5, ttnn.bfloat4_b: 2.2}[q_dtype]
    output_rounding = factor * l2_pct(stored(expected.bfloat16(), q_dtype), expected) if factor else 0.0
    assert l2_pct(actual, expected) < L2_PCT_BOUND[variant] + output_rounding
    return out


def key_mask(sq, sk, *, causal=False, window=None, q_offset=0, cu=None):
    """Additive {0, -inf} mask of the recipes' key ranges (ttnn's golden semantics): query row i sits at global
    position q_offset + i; cu are windowed mode's cumulative window bounds (cu_window_seqlens)."""
    q_pos = torch.arange(q_offset, q_offset + sq).unsqueeze(1)
    k_pos = torch.arange(sk).unsqueeze(0)
    allowed = torch.ones(sq, sk, dtype=torch.bool)
    if cu is not None:
        bounds = torch.tensor(cu)
        allowed &= torch.bucketize(q_pos, bounds, right=True) == torch.bucketize(k_pos, bounds, right=True)
    if causal:
        allowed &= k_pos <= q_pos
    if window:
        if causal:
            allowed &= k_pos > q_pos - window
        else:
            allowed &= (k_pos >= q_pos - window // 2) & (k_pos <= q_pos + window // 2)
    return torch.zeros(sq, sk).masked_fill(~allowed, -math.inf)


def int_tensor(device, values):
    return ttnn.from_torch(torch.tensor(values, dtype=torch.int32), dtype=ttnn.int32, device=device)


# Causal / sliding-window shapes: b, nh, nkv, s, d, q_chunk, k_chunk (Sq == Sk).
CAUSAL_SHAPES = {
    "q256_k512": (1, 2, 2, 2048, 128, 256, 512),
    "q512_k128": (1, 2, 2, 1024, 128, 512, 128),
    "subtile_tails": (1, 2, 2, 1000, 128, 256, 512),
    "odd_q96_k160_d64": (1, 2, 2, 864, 64, 96, 160),
    "gqa_batch2": (2, 8, 2, 768, 128, 128, 256),
    "more_heads_than_cores": (8, 24, 8, 512, 64, 128, 128),
}


def check_key_range(device, variant, shape, *, causal, window=None):
    """Dense causal and/or sliding-window SDPA (scaled_dot_product_attention with is_causal / sliding_window_size)."""
    b, nh, nkv, s, d, q_chunk, k_chunk = shape
    q, k, v = randn(b, nh, s, d, seed=33), randn(b, nkv, s, d, seed=34), randn(b, nkv, s, d, seed=35)
    out = ttnn.transformer.scaled_dot_product_attention(
        *inputs_for(device, variant, q, k, v),
        is_causal=causal,
        sliding_window_size=window,
        program_config=program_config(device, q_chunk, k_chunk),
        precision=VARIANTS[variant][0],
    )
    expected = reference(q, k, v, key_mask(s, s, causal=causal, window=window))
    assert l2_pct(ttnn.to_torch(out), expected) < L2_PCT_BOUND[variant]


def check_windowed(device, variant, cu, *, causal, q_rows=None, q_offset=0, offset_as_tensor=False, chunks=(128, 256)):
    """Windowed (block-diagonal) SDPA from cu_window_seqlens, optionally on a Q slice at q_offset."""
    s, d = cu[-1], 128
    q, k, v = randn(1, 4, s, d, seed=36), randn(1, 2, s, d, seed=37), randn(1, 2, s, d, seed=38)
    q_rows = q_rows or s
    q = q[:, :, q_offset : q_offset + q_rows].contiguous()
    kwargs = dict(cu_window_seqlens=int_tensor(device, cu))
    if offset_as_tensor:
        kwargs["windowed_q_token_offset_tensor"] = int_tensor(device, [q_offset])
    else:
        kwargs["windowed_q_token_offset"] = q_offset
    out = ttnn.transformer.scaled_dot_product_attention(
        *inputs_for(device, variant, q, k, v),
        is_causal=causal,
        program_config=program_config(device, *chunks),
        precision=VARIANTS[variant][0],
        **kwargs,
    )
    expected = reference(q, k, v, key_mask(q_rows, s, causal=causal, q_offset=q_offset, cu=cu))
    assert l2_pct(ttnn.to_torch(out), expected) < L2_PCT_BOUND[variant]


def sink_reference(q, k, v, sink, mask=None, scale=None):
    """FP64 attention with per-head sink logits [1, H, 1, 1] (legacy attention_sink: unscaled logits whose
    exp(scale * sink) joins each row's softmax denominator)."""
    q, k, v = q.double(), k.double(), v.double()
    rep = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
    scale = 1 / math.sqrt(q.shape[-1]) if scale is None else scale
    scores = (q @ k.transpose(-1, -2)) * scale
    if mask is not None:
        scores = scores + mask.double()
    sinks = (sink.double() * scale).expand(q.shape[0], -1, q.shape[2], 1)
    weights = torch.softmax(torch.cat([scores, sinks], -1), -1)[..., :-1]
    return weights @ v


class ChunkedCase:
    """Chunked prefill: Q rows [start, start + sq) of each sequence over a paged K/V cache of blocks_per_seq blocks of
    `block` rows per sequence. With one block per sequence the page table maps batch b to block blocks[b]; with more,
    to a shuffled set. `cache_shape` (heads, rows, head dim) declares the cache in another layer's geometry of the
    same elements per block; the call then passes its view as paged_cache_geometry. head_dim_v (MLA): run
    chunked_flash_mla_prefill, V being K's first head_dim_v columns. sink: per-head attention sink logits."""

    def __init__(
        self,
        device,
        variant,
        *,
        b=1,
        nh=2,
        nkv=2,
        sq=256,
        block=1024,
        d=128,
        blocks=None,
        blocks_per_seq=1,
        cache_shape=None,
        head_dim_v=None,
        sink=False,
        seed=40,
    ):
        self.device, self.variant, self.head_dim_v = device, variant, head_dim_v
        if blocks_per_seq == 1:
            table = [[x] for x in (blocks or list(reversed(range(b))))]
        else:
            order = torch.randperm(b * blocks_per_seq + 1, generator=torch.Generator().manual_seed(seed))
            table = order[: b * blocks_per_seq].reshape(b, blocks_per_seq).tolist()
        self.table = torch.tensor(table)
        count = int(self.table.max()) + 1
        self.q = randn(b, nh, sq, d, seed=seed)
        self.k, self.v = randn(count, nkv, block, d, seed=seed + 1), randn(count, nkv, block, d, seed=seed + 2)
        if head_dim_v:
            self.v = self.k[..., :head_dim_v]
        stored_k, stored_v = self.k, self.v
        self.geometry = None
        if cache_shape:
            stored_k, stored_v = (x.reshape(count, *cache_shape) for x in (self.k, self.v))
            self.geometry = ttnn.PagedCacheGeometryOverride(block_size=block, num_kv_heads=nkv)
        self.inputs = inputs_for(device, variant, self.q, stored_k, stored_v)
        if variant.startswith("fast"):
            # FAST stores K/V as prepare_sdpa_input packs them: the reference uses those values.
            self.k, self.v = (ttnn.to_torch(x).reshape(x.shape[0], nkv, block, -1) for x in self.inputs[1:])
            if head_dim_v:
                self.v = self.k[..., :head_dim_v]
        self.page_table = int_tensor(device, table)
        self.sink = randn(1, nh, 1, 1, seed=seed + 3) * 4 if sink else None

    def run(self, start, q_chunk, k_chunk, *, window=None, start_tensor=None):
        config = program_config(self.device, q_chunk, k_chunk)
        precision = VARIANTS[self.variant][0]
        if self.head_dim_v:
            return ttnn.transformer.chunked_flash_mla_prefill(
                self.inputs[0],
                self.inputs[1],
                self.head_dim_v,
                self.page_table,
                start,
                program_config=config,
                precision=precision,
            )
        kwargs = dict(chunk_start_idx_tensor=start_tensor) if start_tensor is not None else dict(chunk_start_idx=start)
        if self.geometry is not None:
            kwargs["paged_cache_geometry"] = self.geometry
        if self.sink is not None:
            kwargs["attention_sink"] = to_device(self.device, self.sink)
        return ttnn.transformer.chunked_scaled_dot_product_attention(
            *self.inputs,
            self.page_table,
            program_config=config,
            sliding_window_size=window,
            precision=precision,
            **kwargs,
        )

    def expected(self, start, window=None):
        sq = self.q.shape[2]
        keys = start + sq
        # Sequence b's keys: its blocks in page-table order.
        k, v = (
            torch.cat([x[self.table[:, j]] for j in range(self.table.shape[1])], 2)[:, :, :keys]
            for x in (self.k, self.v)
        )
        mask = key_mask(sq, keys, causal=True, window=window, q_offset=start)
        if self.sink is not None:
            return sink_reference(self.q, k, v, self.sink, mask)
        return reference(self.q, k, v, mask)

    def check(self, out, start, window=None):
        assert l2_pct(ttnn.to_torch(out), self.expected(start, window)) < L2_PCT_BOUND[self.variant]


def check_chunked(device, variant, start, *, q_chunk=128, k_chunk=256, window=None, as_tensor=False, **case):
    chunked = ChunkedCase(device, variant, **case)
    start_tensor = int_tensor(device, [start]) if as_tensor else None
    chunked.check(chunked.run(start, q_chunk, k_chunk, window=window, start_tensor=start_tensor), start, window)


# MLA prefill (flash_mla_prefill): b, nh, s_q, s_k, QK head dim, V head dim, q_chunk, k_chunk.
MLA_SHAPES = {
    "d192_v128": (1, 4, 512, 512, 192, 128, 128, 256),
    "d576_v512": (1, 2, 256, 256, 576, 512, 64, 128),
    "d128_v64_tails": (2, 3, 300, 700, 128, 64, 128, 256),
}


def check_mla(device, variant, shape, *, causal=True, v_tensor=False):
    """flash_mla_prefill: K [b, 1, s, d] shared by every head; V is K's first head_dim_v columns, or (v_tensor) its own
    tensor [b, 1, s, head_dim_v]."""
    b, nh, sq, sk, d, dv, q_chunk, k_chunk = shape
    q, k = randn(b, nh, sq, d, seed=50), randn(b, 1, sk, d, seed=51)
    v = randn(b, 1, sk, dv, seed=52) if v_tensor else k[..., :dv]
    tq, tk, tv = inputs_for(device, variant, q, k, v)
    if variant.startswith("fast"):
        k = ttnn.to_torch(tk)
        v = ttnn.to_torch(tv) if v_tensor else k[..., :dv]
    kwargs = dict(
        is_causal=causal, program_config=program_config(device, q_chunk, k_chunk), precision=VARIANTS[variant][0]
    )
    if v_tensor:
        out = ttnn.transformer.flash_mla_prefill(tq, tk, tv, **kwargs)
    else:
        out = ttnn.transformer.flash_mla_prefill(tq, tk, dv, **kwargs)
    assert tuple(out.shape) == (b, nh, sq, dv)
    expected = reference(q, k, v, key_mask(sq, sk, causal=True) if causal else None)
    assert l2_pct(ttnn.to_torch(out), expected) < L2_PCT_BOUND[variant]


def check_sink(
    device,
    variant,
    *,
    causal=False,
    window=None,
    shape=(1, 4, 2, 512, 1024, 128, 256, 512),
    sink_offset=0.0,
    sink_dtype=ttnn.bfloat16,
):
    """scaled_dot_product_attention with attention_sink [1, H, 1, 1] (unscaled per-head logits, 4 x normal plus
    sink_offset)."""
    b, nh, nkv, sq, sk, d, q_chunk, k_chunk = shape
    q, k, v = randn(b, nh, sq, d, seed=60), randn(b, nkv, sk, d, seed=61), randn(b, nkv, sk, d, seed=62)
    sink = (randn(1, nh, 1, 1, seed=63) * 4 + sink_offset).bfloat16()
    tq, tk, tv = inputs_for(device, variant, q, k, v)
    if variant.startswith("fast"):
        k, v = ttnn.to_torch(tk), ttnn.to_torch(tv)
    out = ttnn.transformer.scaled_dot_product_attention(
        tq,
        tk,
        tv,
        is_causal=causal,
        sliding_window_size=window,
        attention_sink=to_device(device, sink, sink_dtype),
        program_config=program_config(device, q_chunk, k_chunk),
        precision=VARIANTS[variant][0],
    )
    mask = key_mask(sq, sk, causal=causal, window=window) if causal or window else None
    sink = stored(sink, sink_dtype) if sink_dtype != ttnn.float32 else sink
    assert l2_pct(ttnn.to_torch(out), sink_reference(q, k, v, sink, mask)) < L2_PCT_BOUND[variant]


def check_concat_heads(device, variant, *, causal=False, shape=(2, 4, 2, 300, 640, 64, 128, 256)):
    """output_concat_heads: the output [b, 1, s, nh * d] holds the heads side by side."""
    b, nh, nkv, sq, sk, d, q_chunk, k_chunk = shape
    q, k, v = randn(b, nh, sq, d, seed=70), randn(b, nkv, sk, d, seed=71), randn(b, nkv, sk, d, seed=72)
    tq, tk, tv = inputs_for(device, variant, q, k, v)
    if variant.startswith("fast"):
        k, v = ttnn.to_torch(tk), ttnn.to_torch(tv)
    out = ttnn.transformer.scaled_dot_product_attention(
        tq,
        tk,
        tv,
        is_causal=causal,
        output_concat_heads=True,
        program_config=program_config(device, q_chunk, k_chunk),
        precision=VARIANTS[variant][0],
    )
    assert tuple(out.shape) == (b, 1, sq, nh * d)
    expected = reference(q, k, v, key_mask(sq, sk, causal=True) if causal else None)
    expected = expected.permute(0, 2, 1, 3).reshape(b, 1, sq, nh * d)
    assert l2_pct(ttnn.to_torch(out), expected) < L2_PCT_BOUND[variant]


def check_chunked_trace(device, variant, starts, *, q_chunk=128, k_chunk=256):
    """chunk_start_idx_tensor is read on device: one captured trace replays at every start offset."""
    device.enable_program_cache()
    chunked = ChunkedCase(device, variant)
    start_tensor = int_tensor(device, [starts[0]])
    chunked.check(chunked.run(starts[0], q_chunk, k_chunk, start_tensor=start_tensor), starts[0])
    trace = ttnn.begin_trace_capture(device, cq_id=0)
    traced = chunked.run(starts[0], q_chunk, k_chunk, start_tensor=start_tensor)
    ttnn.end_trace_capture(device, trace, cq_id=0)
    try:
        for start in starts:
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(torch.tensor([start], dtype=torch.int32), dtype=ttnn.int32), start_tensor
            )
            ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
            chunked.check(traced, start)
    finally:
        ttnn.release_trace(device, trace)


def fp32_dest_config(device):
    """The shared HiFi4 + FP32-dest config legacy callers pass (it selected the legacy kernels)."""
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def check_routing(device, case):
    """Precision routing: a call without `precision` that would reach a legacy loop runs a recipe (FP32 dest ->
    ACCURATE; non-ring joint -> STANDARD, or ACCURATE with FP32 dest). With chunk sizes the recipe supports, the
    routed call is bitwise the explicit recipe's; chunk sizes it does not support (Q2048) leave the blocking to the
    op. Either way the output meets the recipe's FP64 bound."""
    fp32 = fp32_dest_config(device)
    accurate, standard = ttnn.SDPAPrecision.ACCURATE, ttnn.SDPAPrecision.STANDARD
    if case in ("dense_causal_bfp8", "dense_mask_q2048"):
        causal = case == "dense_causal_bfp8"
        dtype = ttnn.bfloat8_b if causal else ttnn.bfloat16
        q, k, v = (stored(randn(1, h, 1000, 128, seed=80 + i), dtype) for i, h in enumerate((4, 2, 2)))
        mask = None if causal else key_mask(1000, 1000, window=512)[None, None]
        kwargs = dict(is_causal=causal, scale=0.1, attn_mask=None if causal else to_device(device, mask))
        tensors = [to_device(device, x, dtype) for x in (q, k, v)]
        cfg = program_config(device, 256, 512) if causal else program_config(device, 2048, 512)
        run = lambda **extra: ttnn.transformer.scaled_dot_product_attention(
            *tensors, program_config=cfg, **kwargs, **extra
        )
        routed, recipe = run(compute_kernel_config=fp32), accurate
        expected = reference(q, k, v, key_mask(1000, 1000, causal=True) if causal else mask, 0.1)
        explicit = run(precision=recipe) if causal else None
        tolerance = 1.5 * l2_pct(stored(expected.bfloat16(), dtype), expected)  # BFP8 output, as legacy
    elif case == "chunked_tensor_start":
        chunked = ChunkedCase(device, "accurate", sq=256, blocks_per_seq=4, block=128)
        start = int_tensor(device, [256])
        run = lambda **extra: ttnn.transformer.chunked_scaled_dot_product_attention(
            *chunked.inputs,
            chunked.page_table,
            chunk_start_idx_tensor=start,
            program_config=program_config(device, 128, 256),
            **extra,
        )
        routed, recipe = run(compute_kernel_config=fp32), accurate
        explicit, expected, tolerance = run(precision=recipe), chunked.expected(256), 0.0
    else:  # joint_bf16_dest / joint_fp32_dest
        recipe = accurate if case == "joint_fp32_dest" else standard
        q, k, v = randn(1, 2, 600, 128, seed=90), randn(1, 2, 600, 128, seed=91), randn(1, 2, 600, 128, seed=92)
        jq, jk, jv = randn(1, 2, 77, 128, seed=93), randn(1, 2, 77, 128, seed=94), randn(1, 2, 77, 128, seed=95)
        tensors = [to_device(device, x) for x in (q, k, v, jq, jk, jv)]
        run = lambda **extra: torch.cat(
            [
                ttnn.to_torch(x)
                for x in ttnn.transformer.joint_scaled_dot_product_attention(
                    *tensors, joint_strategy="rear", program_config=program_config(device, 128, 256), **extra
                )
            ],
            2,
        )
        routed = run(compute_kernel_config=fp32) if recipe == accurate else run()
        explicit, tolerance = run(precision=recipe), 0.0
        expected = reference(torch.cat([q, jq], 2), torch.cat([k, jk], 2), torch.cat([v, jv], 2))
    routed = routed if isinstance(routed, torch.Tensor) else ttnn.to_torch(routed)
    if explicit is not None:
        explicit = explicit if isinstance(explicit, torch.Tensor) else ttnn.to_torch(explicit)
        assert torch.equal(routed, explicit), "the routed call must run the explicit recipe"
    name = "accurate" if recipe == accurate else "standard"
    assert l2_pct(routed, expected) < L2_PCT_BOUND[name] + tolerance
