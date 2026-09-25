# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Trace-safe metadata for sparse_sdpa_msa, on one Blackhole.

chunk_start_idx / cache_batch_idx are host ints patched into the launch on every dispatch, which a metal trace
replay never re-runs. The tensor forms are 1-element uint32 DRAM tensors the kernels NoC-read each dispatch
(the reader derives the causal start; reader AND writer select the K/V slot, since they co-gather each block).
These tests pin, for one cached program:
  * tensor path == host-int path bit-exactly over 2 users x 3 depths (one depth not block- or tile-aligned),
  * a program-cache hit follows FRESHLY allocated metadata tensors,
  * an in-place rewrite of the same tensors retargets the next dispatch,
  * a captured trace retargets user / depth on replay,
  * illegal host/tensor mixes and bad metadata containers are refused.
The sp>1 rotated-start geometry is covered in the nightly multi-device file.
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from tests.ttnn.unit_tests.operations.sdpa.sparse_sdpa_msa_test_utils import (
    BLK_KV,
    SENTINEL,
    pcc,
    sparse_attention_ref_msa_sampled_tokens,
)

H, N_KV, S, D, T, TOPK = 32, 1, 2 * BLK_KV, 128, 8 * BLK_KV, 16
USERS, LAYERS, LAYER = 2, 2, 1  # user-major [USERS*LAYERS, n_kv, T, d] cache; this op instance is layer 1
DEPTHS = (0, 200, 512)  # 200: mid-block AND mid-tile start
SCALE = D**-0.5
TARGETS = [(u, d) for u in range(USERS) for d in DEPTHS]
SAMPLE_TOKENS = [0, 1, 55, 56, 127, 128, 200, S - 1]


def _causal_indices(depth, seed):
    """Per-token block ids for a chunk starting at `depth`: only visible blocks, own (diagonal) block always
    selected, -1 tail. The diagonal block holds future tokens, so the token-level mask is exercised."""
    gen = torch.Generator().manual_seed(seed)
    idx = torch.full((1, N_KV, S, TOPK), SENTINEL, dtype=torch.int32)
    for s in range(S):
        local = (depth + s) // BLK_KV
        visible = local + 1
        if visible <= TOPK:
            chosen = torch.arange(visible)
        else:
            pool = torch.randperm(visible, generator=gen)[:TOPK]
            if local not in pool.tolist():
                pool[-1] = local
            chosen = pool.sort().values
        idx[0, 0, s, : chosen.numel()] = chosen.to(torch.int32)
    return idx


def _inputs(seed=3):
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(1, H, S, D, generator=gen)
    k = torch.randn(USERS * LAYERS, N_KV, T, D, generator=gen)  # distinct slots: a wrong-slot read changes out
    v = torch.randn(USERS * LAYERS, N_KV, T, D, generator=gen)
    indices = {depth: _causal_indices(depth, seed=depth + 1) for depth in DEPTHS}
    return q, k, v, indices


def _rm(device, x, dtype):
    return ttnn.from_torch(
        x, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def _tile(device, x):
    return ttnn.from_torch(
        x.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _host_u32(value):
    return ttnn.from_torch(
        torch.tensor([[[[value]]]], dtype=torch.int64), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
    )


def _dev_u32(device, value, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.uint32, shape=(1, 1, 1, 1)):
    return ttnn.from_torch(
        torch.full(shape, value, dtype=torch.int64),
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=memory_config,
    )


class _Dev:
    """Device copies of the fixed inputs (q, the multi-slot K/V cache) and one indices tensor per depth."""

    def __init__(self, device, q, k, v, indices):
        self.q = _rm(device, q.to(torch.float32), ttnn.bfloat16)
        self.k, self.v = _tile(device, k), _tile(device, v)
        self.idx = {depth: _rm(device, i, ttnn.uint32) for depth, i in indices.items()}


def _run(dev, idx, **kw):
    return ttnn.transformer.sparse_sdpa_msa(dev.q, dev.k, dev.v, idx, scale=SCALE, block_size=BLK_KV, **kw)


def _meta_kwargs(start_t, user_t):
    return dict(
        chunk_start_idx_tensor=start_t,
        cache_batch_idx_tensor=user_t,
        index_cache_num_layers=LAYERS,
        index_cache_layer_idx=LAYER,
    )


def _host_references(dev, q, k, v, indices):
    """Host-int outputs per target, each anchored to the causal golden on its K/V slot."""
    refs = {}
    for user, depth in TARGETS:
        slot = user * LAYERS + LAYER
        out = ttnn.to_torch(_run(dev, dev.idx[depth], chunk_start_idx=depth, cache_batch_idx=slot))[:, :H]
        golden = sparse_attention_ref_msa_sampled_tokens(
            q,
            k[slot : slot + 1],
            v[slot : slot + 1],
            indices[depth],
            SCALE,
            SAMPLE_TOKENS,
            causal=True,
            chunk_start_idx=depth,
        )
        p = pcc(out[:, :, SAMPLE_TOKENS], golden)
        assert p >= 0.99, f"host path vs golden user={user} depth={depth}: pcc={p:.5f}"
        refs[(user, depth)] = out
    return refs


def _assert_same(out, refs, user, depth):
    assert torch.equal(out[:, :H], refs[(user, depth)]), f"metadata path != host path (user={user}, depth={depth})"


@run_for_blackhole()
def test_sparse_sdpa_msa_metadata_matches_host(device):
    """2 users x 3 depths on ONE cached metadata program, bit-exact vs the host-int path. Every dispatch passes
    FRESHLY allocated tensors (earlier ones kept alive -> new addresses), so a cache hit that kept the
    build-time addresses would mask / gather for the wrong user or depth. Then the same pair of tensors is
    rewritten in place between dispatches (the trace-replay access pattern)."""
    q, k, v, indices = _inputs()
    dev = _Dev(device, q, k, v, indices)
    refs = _host_references(dev, q, k, v, indices)

    live, entries = [], None
    for user, depth in TARGETS:
        start_t, user_t = _dev_u32(device, depth), _dev_u32(device, user)
        live += [start_t, user_t]
        _assert_same(ttnn.to_torch(_run(dev, dev.idx[depth], **_meta_kwargs(start_t, user_t))), refs, user, depth)
        if entries is None:
            entries = device.num_program_cache_entries()
    assert device.num_program_cache_entries() == entries, "switching user / depth tensors recompiled"
    assert len({t.buffer_address() for t in live}) == len(live), "metadata tensors were not distinct allocations"

    start_t, user_t = _dev_u32(device, 0), _dev_u32(device, 0)
    for user, depth in TARGETS[::-1]:
        ttnn.copy_host_to_device_tensor(_host_u32(depth), start_t)
        ttnn.copy_host_to_device_tensor(_host_u32(user), user_t)
        _assert_same(ttnn.to_torch(_run(dev, dev.idx[depth], **_meta_kwargs(start_t, user_t))), refs, user, depth)
    assert device.num_program_cache_entries() == entries, "in-place rewrite recompiled"


@run_for_blackhole()
def test_sparse_sdpa_msa_metadata_single_tensors(device):
    """Each tensor alone: chunk-start tensor with a host cache_batch_idx, slot tensor with a host chunk_start."""
    q, k, v, indices = _inputs(seed=7)
    dev = _Dev(device, q, k, v, indices)
    refs = _host_references(dev, q, k, v, indices)
    user, depth = 1, 200
    out = _run(
        dev, dev.idx[depth], chunk_start_idx_tensor=_dev_u32(device, depth), cache_batch_idx=user * LAYERS + LAYER
    )
    _assert_same(ttnn.to_torch(out), refs, user, depth)
    out = _run(
        dev,
        dev.idx[depth],
        chunk_start_idx=depth,
        cache_batch_idx_tensor=_dev_u32(device, user),
        index_cache_num_layers=LAYERS,
        index_cache_layer_idx=LAYER,
    )
    _assert_same(ttnn.to_torch(out), refs, user, depth)


@run_for_blackhole()
@pytest.mark.parametrize("device_params", [{"trace_region_size": 1 << 20}], indirect=True)
def test_sparse_sdpa_msa_metadata_trace_retarget(device):
    """One captured trace, replayed across users and depths by rewriting the SAME metadata (and indices) tensors
    in place. With host ints the replay would keep the capture-time slot and causal start."""
    q, k, v, indices = _inputs(seed=11)
    dev = _Dev(device, q, k, v, indices)
    refs = _host_references(dev, q, k, v, indices)

    idx_t = _rm(device, indices[DEPTHS[0]], ttnn.uint32)
    host_idx = {d: ttnn.from_torch(i, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT) for d, i in indices.items()}
    start_t, user_t = _dev_u32(device, DEPTHS[0]), _dev_u32(device, 0)
    _run(dev, idx_t, **_meta_kwargs(start_t, user_t))  # compile outside the capture
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    try:
        traced_out = _run(dev, idx_t, **_meta_kwargs(start_t, user_t))
    finally:
        ttnn.end_trace_capture(device, trace_id, cq_id=0)
    try:
        for user, depth in TARGETS + TARGETS[::-1]:
            ttnn.copy_host_to_device_tensor(host_idx[depth], idx_t)
            ttnn.copy_host_to_device_tensor(_host_u32(depth), start_t)
            ttnn.copy_host_to_device_tensor(_host_u32(user), user_t)
            ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
            _assert_same(ttnn.to_torch(traced_out), refs, user, depth)
    finally:
        ttnn.release_trace(device, trace_id)


@run_for_blackhole()
def test_sparse_sdpa_msa_metadata_rejects(device, expect_error):
    """Host ints and their tensors are mutually exclusive, and the metadata container is pinned."""
    q, k, v, indices = _inputs()
    dev = _Dev(device, q, k, v, indices)
    idx = dev.idx[DEPTHS[0]]
    start_t, user_t = _dev_u32(device, 0), _dev_u32(device, 1)
    meta = _meta_kwargs(start_t, user_t)

    with expect_error(RuntimeError, "chunk_start_idx and chunk_start_idx_tensor are mutually exclusive"):
        _run(dev, idx, chunk_start_idx=0, **meta)
    with expect_error(RuntimeError, "cache_batch_idx and cache_batch_idx_tensor are mutually exclusive"):
        _run(dev, idx, cache_batch_idx=1, **meta)
    with expect_error(RuntimeError, "must be DRAM interleaved"):
        _run(dev, idx, **_meta_kwargs(_dev_u32(device, 0, ttnn.L1_MEMORY_CONFIG), user_t))
    with expect_error(RuntimeError, "must hold exactly 1 element"):
        _run(dev, idx, **_meta_kwargs(_dev_u32(device, 0, shape=(1, 1, 1, 8)), user_t))
    with expect_error(RuntimeError, "must be UINT32"):
        _run(dev, idx, **_meta_kwargs(start_t, _dev_u32(device, 1, dtype=ttnn.int32)))
    with expect_error(RuntimeError, "index_cache_num_layers"):  # 3 does not divide the 4 user-major slots
        _run(dev, idx, chunk_start_idx_tensor=start_t, cache_batch_idx_tensor=user_t, index_cache_num_layers=3)
    with expect_error(RuntimeError, "batch must be 1"):  # a multi-slot cache still needs a slot selector
        _run(dev, idx, chunk_start_idx_tensor=start_t)
