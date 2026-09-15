# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""`sol_attn` against the op it was forked from.

The fork starts as an exact clone of `scaled_dot_product_attention`, so every check here should hold
bit-for-bit. A failure is a forking mistake -- a renamed symbol bound to the wrong body, a kernel path
still pointing at the parent's file, a compile-time arg that shifted -- not an algorithm difference.

Single device on purpose: the hybrid slots in after the K/V all-gather, so its base needs no mesh,
semaphores or fabric, and iterating here cannot disturb a shared Galaxy.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc, skip_for_wormhole_b0


def _qkv(b, nh, seq, d, seed=0):
    torch.manual_seed(seed)
    shape = (b, nh, seq, d)
    return tuple(torch.randn(shape, dtype=torch.bfloat16) for _ in range(3))


def _run_both(device, b, nh, seq, d, program_config=None, attn_mask=None, is_causal=False):
    q, k, v = _qkv(b, nh, seq, d)
    dev = lambda x: ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_q, tt_k, tt_v = dev(q), dev(k), dev(v)
    tt_mask = dev(attn_mask) if attn_mask is not None else None

    # attn_mask is keyword-only on both ops, and is_causal defaults to True -- pass both explicitly
    # so the two calls cannot differ by a default.
    kwargs = dict(attn_mask=tt_mask, is_causal=is_causal, program_config=program_config)
    parent = ttnn.to_torch(ttnn.transformer.scaled_dot_product_attention(tt_q, tt_k, tt_v, **kwargs))
    fork = ttnn.to_torch(ttnn.transformer.sol_attn(tt_q, tt_k, tt_v, **kwargs))
    return q, k, v, parent, fork


@skip_for_wormhole_b0("sol_attn targets Blackhole")
@pytest.mark.parametrize(
    ("b", "nh", "seq", "d"),
    [(1, 2, 512, 128), (1, 8, 1024, 128), (1, 2, 2048, 64)],
    ids=["s512", "s1024h8", "s2048d64"],
)
def test_sol_attn_matches_parent(device, b, nh, seq, d):
    """Bit-identical to the parent op on the same inputs."""
    _q, _k, _v, parent, fork = _run_both(device, b, nh, seq, d)
    assert torch.equal(parent, fork), "sol_attn diverged from scaled_dot_product_attention"


@skip_for_wormhole_b0("sol_attn targets Blackhole")
@pytest.mark.parametrize(("b", "nh", "seq", "d"), [(1, 2, 1024, 128)], ids=["s1024"])
def test_sol_attn_matches_torch(device, b, nh, seq, d):
    """And against torch, so the comparison above is not two copies of the same bug."""
    q, k, v, _parent, fork = _run_both(device, b, nh, seq, d)
    gt = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=False)
    ok, pcc = comp_pcc(gt, fork.float(), 0.99)
    logger.info(f"sol_attn vs torch PCC {pcc}")
    assert ok, f"PCC {pcc}"


@skip_for_wormhole_b0("sol_attn targets Blackhole")
@pytest.mark.parametrize(("b", "nh", "seq", "d"), [(1, 2, 1024, 128)], ids=["s1024"])
def test_sol_attn_matches_parent_with_provided_mask(device, b, nh, seq, d):
    """The provided-mask path is what the block mask will ride, so pin it before changing it.

    A block-shaped mask (whole 64-token blocks kept or masked) is the shape the routing produces, so
    this is the case that matters rather than an arbitrary mask.
    """
    torch.manual_seed(5)
    n_blocks = seq // 64
    keep = torch.rand(n_blocks) > 0.5
    keep[0] = True  # a row with nothing selected has no finite softmax
    mask = torch.where(keep, 0.0, float("-inf")).repeat_interleave(64)
    attn_mask = mask.view(1, 1, 1, seq).expand(b, 1, seq, seq).to(torch.bfloat16).contiguous()

    _q, _k, _v, parent, fork = _run_both(device, b, nh, seq, d, attn_mask=attn_mask)
    assert torch.equal(parent, fork), "sol_attn diverged from its parent on the provided-mask path"


@skip_for_wormhole_b0("sol_attn targets Blackhole")
def test_sol_attn_is_deterministic(device):
    """Repeated launches must be bitwise equal, or no later PCC gate has a fixed target."""
    q, k, v = _qkv(1, 2, 1024, 128, seed=9)
    dev = lambda x: ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_q, tt_k, tt_v = dev(q), dev(k), dev(v)
    outs = [ttnn.to_torch(ttnn.transformer.sol_attn(tt_q, tt_k, tt_v, is_causal=False)) for _ in range(3)]
    for i, o in enumerate(outs[1:], start=1):
        assert torch.equal(outs[0], o), f"launch {i} differs from launch 0"


@skip_for_wormhole_b0("sol_attn targets Blackhole")
@pytest.mark.parametrize(("b", "nh", "seq", "d"), [(1, 2, 1024, 128)], ids=["s1024"])
def test_sol_attn_keep_every_one_is_dense(device, b, nh, seq, d):
    """`chunk_keep_every=1` instantiates the skip branches with a predicate that never fires.

    So a failure is the loop breaking, not skipping being wrong -- in particular the accumulator
    alias parity, which `KV_chunks_processed_in_iter` drives and which a misplaced skip would desync.
    """
    q, k, v = _qkv(b, nh, seq, d)
    dev = lambda x: ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_q, tt_k, tt_v = dev(q), dev(k), dev(v)
    parent = ttnn.to_torch(ttnn.transformer.scaled_dot_product_attention(tt_q, tt_k, tt_v, is_causal=False))
    fork = ttnn.to_torch(ttnn.transformer.sol_attn(tt_q, tt_k, tt_v, is_causal=False, chunk_keep_every=1))
    assert torch.equal(parent, fork), "chunk_keep_every=1 diverged from dense"


@skip_for_wormhole_b0("sol_attn targets Blackhole")
@pytest.mark.parametrize(("b", "nh", "seq", "d"), [(1, 2, 1024, 128)], ids=["s1024"])
def test_sol_attn_chunk_stride_is_in_the_program_hash(device, b, nh, seq, d):
    """Different strides must produce different programs, in one process.

    `chunk_keep_every` is a compile-time arg, so if it were missing from the program hash the cache
    would hand a later call the earlier program and every density would time identically -- a
    fabricated "skipping is free" result rather than a crash.
    """
    q, k, v = _qkv(b, nh, seq, d, seed=3)
    dev = lambda x: ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_q, tt_k, tt_v = dev(q), dev(k), dev(v)
    dense = ttnn.to_torch(ttnn.transformer.sol_attn(tt_q, tt_k, tt_v, is_causal=False, chunk_keep_every=1))
    sparse = ttnn.to_torch(ttnn.transformer.sol_attn(tt_q, tt_k, tt_v, is_causal=False, chunk_keep_every=4))
    assert not torch.equal(
        dense, sparse
    ), "keep_every=1 and 4 gave identical output; the stride is not reaching the program hash"
    assert torch.isfinite(sparse.float()).all(), "skipped-chunk output must still normalise"
