# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Fused head layout of the V4.1 attention (tt/v41/head_layout.py, bead 8y7.9.9) vs the composite ttnn path.

``q_heads`` replaces ``nlp_create_qkv_heads`` + the RoPE tail (slice, ``rotary_embedding_llama``, untilize,
``slice_write``); ``o_heads`` replaces the inverse RoPE tail (slice, tilize, rotate, untilize, ``slice_write``,
tilize) + ``nlp_concat_heads`` of the head groups. Both run the rotation's op sequence with the same bf16
intermediates, so they must match the composite path bit for bit. ``test_head_layout_traced_time`` logs the traced
per-call time of both paths at the production per-chip shape (the sequence shard of chunk 5120 on 2x4: 640 tokens,
64 heads of 512) as ``V41_HEADS_PERF`` JSON lines.
"""

import json
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.mla.rope import get_rot_transformation_mat
from models.demos.deepseek_v3_d_p.tt.v41.head_layout import o_heads, q_heads

ROPE = 64
REPLAYS = 20
# (tokens, heads, head_dim, groups): small = SmallV41Config on one chip of 2x4 (32 heads of 128, 8 groups)
SHAPES = {"small": (64, 32, 128, 8), "tiny": (32, 8, 128, 2), "production": (640, 64, 512, 8)}


def _upload(device, t, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(t.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=layout, device=device)


def _inputs(device, shape, seed=0):
    tokens, heads, head_dim, _ = shape
    gen = torch.Generator().manual_seed(seed)
    angle = torch.rand(tokens, ROPE // 2, generator=gen) * 6.3
    # interleaved pairs, as the V4.1 RoPE tables (rotary_embedding_llama's trans_mat convention)
    cos = torch.cos(angle).repeat_interleave(2, dim=-1).reshape(1, 1, tokens, ROPE)
    sin = torch.sin(angle).repeat_interleave(2, dim=-1).reshape(1, 1, tokens, ROPE)
    return dict(
        q=_upload(device, torch.randn(1, 1, tokens, heads * head_dim, generator=gen)),
        attn=_upload(device, torch.randn(1, heads, tokens, head_dim, generator=gen), ttnn.ROW_MAJOR_LAYOUT),
        cos=_upload(device, cos),
        sin=_upload(device, sin),
        trans=_upload(device, get_rot_transformation_mat()),
    )


def _composite_q(d, heads):
    """tt/v41/attention.py before 8y7.9.9: nlp_create_qkv_heads, then ``_rope_tail``."""
    q, _, _ = ttnn.experimental.nlp_create_qkv_heads(d["q"], num_heads=heads, num_kv_heads=0, transpose_k_heads=False)
    b, h, s, dim = q.shape
    tail = ttnn.slice(q, [0, 0, 0, dim - ROPE], [b, h, s, dim])
    tail = ttnn.experimental.rotary_embedding_llama(tail, d["cos"], d["sin"], d["trans"], is_decode_mode=False)
    out = ttnn.to_layout(q, ttnn.ROW_MAJOR_LAYOUT)
    ttnn.experimental.slice_write(
        ttnn.to_layout(tail, ttnn.ROW_MAJOR_LAYOUT), out, [0, 0, 0, dim - ROPE], [b, h, s, dim], [1, 1, 1, 1]
    )
    return out


def _composite_o(d, groups):
    """tt/v41/attention.py before 8y7.9.9: ``_inverse_rope_tail``, then the groups' nlp_concat_heads."""
    t = ttnn.clone(d["attn"])  # the composite path rotates in place
    b, h, s, dim = t.shape
    tail = ttnn.to_layout(ttnn.slice(t, [0, 0, 0, dim - ROPE], [b, h, s, dim]), ttnn.TILE_LAYOUT)
    tail = ttnn.experimental.rotary_embedding_llama(
        tail, d["cos"], ttnn.neg(d["sin"]), d["trans"], is_decode_mode=False
    )
    ttnn.experimental.slice_write(
        ttnn.to_layout(tail, ttnn.ROW_MAJOR_LAYOUT), t, [0, 0, 0, dim - ROPE], [b, h, s, dim], [1, 1, 1, 1]
    )
    t = ttnn.reshape(ttnn.to_layout(t, ttnn.TILE_LAYOUT), [groups, h // groups, s, dim])
    return ttnn.reshape(ttnn.experimental.nlp_concat_heads(t), [1, groups, s, h // groups * dim])


def _fused_q(d, heads):
    return q_heads(d["q"], d["cos"], d["sin"], d["trans"], heads, ROPE)


def _fused_o(d, groups):
    return o_heads(d["attn"], d["cos"], ttnn.neg(d["sin"]), d["trans"], groups, ROPE)


def _torch_q(d, heads):
    """fp32 torch of the same map (interleaved-pair RoPE on each head's tail) for a coarse semantic check."""
    q = ttnn.to_torch(d["q"]).float()
    s = q.shape[-2]
    q = q.reshape(s, heads, -1).transpose(0, 1)[None]
    cos, sin = (ttnn.to_torch(d[k]).float()[0, 0] for k in ("cos", "sin"))
    tail = q[..., -ROPE:]
    rot = torch.stack([-tail[..., 1::2], tail[..., 0::2]], dim=-1).flatten(-2)
    return torch.cat([q[..., :-ROPE], tail * cos + rot * sin], dim=-1)


@pytest.mark.parametrize("shape", list(SHAPES))
@pytest.mark.parametrize("side", ["q", "o"])
def test_head_layout_matches_composite(device, side, shape):
    tokens, heads, head_dim, groups = SHAPES[shape]
    d = _inputs(device, SHAPES[shape])
    t0 = time.perf_counter()
    fused_fn, composite_fn, arg = (_fused_q, _composite_q, heads) if side == "q" else (_fused_o, _composite_o, groups)
    fused = [fused_fn(d, arg) for _ in range(2)]
    logger.info(f"{side} {shape}: 2 fused calls {time.perf_counter() - t0:.2f}s (incl. compile)")
    want_layout = ttnn.ROW_MAJOR_LAYOUT if side == "q" else ttnn.TILE_LAYOUT
    want_shape = (1, heads, tokens, head_dim) if side == "q" else (1, groups, tokens, heads // groups * head_dim)
    assert fused[0].layout == want_layout and tuple(fused[0].shape) == want_shape, (fused[0].layout, fused[0].shape)
    got = [ttnn.to_torch(f) for f in fused]
    composite = ttnn.to_torch(composite_fn(d, arg))
    assert torch.equal(got[0], got[1]), "fused head layout is not deterministic"
    mismatch = (got[0] != composite).sum().item()
    logger.info(f"{side} {shape}: elements != composite {mismatch} of {composite.numel()}")
    if side == "q":
        ref = _torch_q(d, heads)
        err = (got[0].float() - ref).abs().max().item()
        logger.info(f"q {shape}: max |fused - fp32 torch| {err:.3e}")
        assert err < 0.05, err  # bf16 intermediates of |x| <~ 5
    assert mismatch == 0, f"{mismatch} elements differ from the composite path"


def _traced_us(device, fn):
    fn()
    ttnn.synchronize_device(device)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    out = fn()
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    for _ in range(REPLAYS):
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    us = (time.perf_counter() - t0) / REPLAYS * 1e6
    ttnn.release_trace(device, tid)
    out.deallocate()
    return us


@pytest.mark.parametrize("impl", ["fused", "composite"])
@pytest.mark.parametrize("side", ["q", "o"])
@pytest.mark.parametrize("device_params", [{"trace_region_size": 32 << 20}], indirect=True)
def test_head_layout_traced_time(device, side, impl):
    tokens, heads, head_dim, groups = SHAPES["production"]
    d = _inputs(device, SHAPES["production"])
    fns = {("q", "fused"): _fused_q, ("q", "composite"): _composite_q}
    fns |= {("o", "fused"): _fused_o, ("o", "composite"): _composite_o}
    arg = heads if side == "q" else groups
    us = _traced_us(device, lambda: fns[side, impl](d, arg))
    moved = 2 * 2 * tokens * heads * head_dim  # read + write the heads once, bf16
    bound_us = moved / (512.0 * 1e3)  # Blackhole p150 DRAM peak
    record = dict(side=side, impl=impl, tokens=tokens, heads=heads, head_dim=head_dim, traced_us=round(us, 1))
    record |= dict(dram_bound_us=round(bound_us, 1), dram_util=round(bound_us / us, 3))
    logger.info("V41_HEADS_PERF " + json.dumps(record))
