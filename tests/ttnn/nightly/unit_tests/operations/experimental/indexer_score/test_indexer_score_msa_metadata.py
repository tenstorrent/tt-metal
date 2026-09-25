# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Trace-safe metadata for indexer_score_msa (classic factory, caller pre-gathered K), on one Blackhole.

The host ints chunk_start_idx / kv_len / cache_batch_idx are patched into the launch on every dispatch, which a
metal trace replay never re-runs. The tensor forms are 1-element uint32 DRAM tensors the reader NoC-reads each
dispatch. These tests pin, for one cached program:
  * tensor path == host-int path bit-exactly over 2 users x 3 depths (kv_len derived = depth + Sq),
  * a program-cache hit follows FRESHLY allocated metadata tensors (not the ones the program was built with),
  * an in-place rewrite of the same tensors retargets the next dispatch,
  * a captured trace retargets user / depth on replay after an in-place rewrite,
  * illegal host/tensor mixes and bad metadata containers are refused.
"""

import pytest
import torch

import ttnn
import tests.ttnn.nightly.unit_tests.operations.experimental.indexer_score.test_indexer_score as base

pytestmark = pytest.mark.skipif(not ttnn.device.is_blackhole(), reason="indexer_score is Blackhole-only")

HEADS, DIM, SQ, T = 4, 64, 64, 512
USERS, LAYERS, LAYER = 2, 2, 1  # user-major [USERS*LAYERS, 1, T, D] cache; this op instance is layer 1
DEPTHS = (0, 128, 256)  # history lengths: tile- and block-aligned, chunk ends inside T
SCALE = DIM**-0.5
TARGETS = [(u, d) for u in range(USERS) for d in DEPTHS]

# (num_groups, block_size, program_config): grouped unpooled planes, and the pooled M3 block selection.
CASES = {
    "grouped": (2, 0, dict(q_chunk_size=32, k_chunk_size=64, head_group_size=0)),
    "pooled": (1, 32, dict(q_chunk_size=32, k_chunk_size=256, head_group_size=0)),
}


def _inputs(seed=5):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(1, HEADS, SQ, DIM, generator=g, dtype=torch.bfloat16)
    # Distinct slots, so a wrong-slot read changes the scores.
    k_cache = torch.randn(USERS * LAYERS, 1, T, DIM, generator=g, dtype=torch.bfloat16)
    return q, k_cache


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


def _run(q_dev, k_dev, case, **kw):
    num_groups, block_size, cfg = CASES[case]
    return ttnn.experimental.indexer_score_msa(
        q_dev,
        k_dev,
        num_groups=num_groups,
        scale=SCALE,
        block_size=block_size,
        program_config=ttnn.IndexerScoreProgramConfig(**cfg),
        **kw,
    )


def _host_kwargs(user, depth):
    return dict(chunk_start_idx=depth, kv_len=depth + SQ, cache_batch_idx=user * LAYERS + LAYER)


def _meta_kwargs(chunk_start_tensor, user_tensor):
    return dict(
        chunk_start_idx_tensor=chunk_start_tensor,
        cache_batch_idx_tensor=user_tensor,
        index_cache_num_layers=LAYERS,
        index_cache_layer_idx=LAYER,
    )


def _valid_cols(case, depth):
    """Output columns written this dispatch: [0, kv_len) keys, or its whole blocks when pooling."""
    block_size = CASES[case][1]
    kv_len = depth + SQ
    return kv_len // block_size if block_size else kv_len


def _host_references(q, k_cache, q_dev, k_dev, case):
    """Host-int outputs per target, each anchored to the torch reference on its slot + valid prefix."""
    num_groups, block_size, _ = CASES[case]
    w_scale = torch.full((1, HEADS, SQ, 1), SCALE, dtype=torch.bfloat16)
    refs = {}
    for user, depth in TARGETS:
        out = ttnn.to_torch(_run(q_dev, k_dev, case, **_host_kwargs(user, depth)))
        cols = _valid_cols(case, depth)
        slot = user * LAYERS + LAYER
        kv_len = depth + SQ
        golden = base.indexer_score_msa_ref(
            q, k_cache[slot : slot + 1, :, :kv_len], w_scale, depth, num_groups, block_size=block_size
        )
        if block_size:  # block-max amplifies the bf16 raw-dot error: same floor as test_indexer_score_block_pool
            base.assert_pooled_match(out[..., :cols], golden, num_groups, SQ, cols, pcc_floor=0.995)
        else:
            base.assert_grouped_match(out[..., :cols], golden, num_groups, SQ, cols)
        refs[(user, depth)] = out
    return refs


def _assert_same(out, refs, case, user, depth):
    cols = _valid_cols(case, depth)
    assert torch.equal(
        out[..., :cols], refs[(user, depth)][..., :cols]
    ), f"{case}: metadata path != host-int path for user={user} depth={depth}"


@pytest.mark.parametrize("case", list(CASES))
def test_indexer_score_msa_metadata_matches_host(device, case):
    """2 users x 3 depths on ONE cached metadata program: bit-exact vs the host-int path. Each dispatch passes
    FRESHLY allocated tensors (the earlier ones stay alive so every allocation lands at a new address), so a
    cache hit that kept the build-time addresses would read the wrong user / depth. Then the same pair of
    tensors is rewritten in place between dispatches (the trace-replay access pattern)."""
    q, k_cache = _inputs()
    q_dev, k_dev = base.to_device(q, device), base.to_device(k_cache, device)
    refs = _host_references(q, k_cache, q_dev, k_dev, case)

    live = []  # keep every metadata tensor alive -> each new one is a distinct allocation
    entries = None
    for user, depth in TARGETS:
        start_t, user_t = _dev_u32(device, depth), _dev_u32(device, user)
        live += [start_t, user_t]
        out = ttnn.to_torch(_run(q_dev, k_dev, case, **_meta_kwargs(start_t, user_t)))
        _assert_same(out, refs, case, user, depth)
        if entries is None:
            entries = device.num_program_cache_entries()
    assert device.num_program_cache_entries() == entries, "switching user / depth tensors recompiled"
    assert len({t.buffer_address() for t in live}) == len(live), "metadata tensors were not distinct allocations"

    start_t, user_t = _dev_u32(device, DEPTHS[0]), _dev_u32(device, 0)
    for user, depth in TARGETS[::-1]:
        ttnn.copy_host_to_device_tensor(_host_u32(depth), start_t)
        ttnn.copy_host_to_device_tensor(_host_u32(user), user_t)
        out = ttnn.to_torch(_run(q_dev, k_dev, case, **_meta_kwargs(start_t, user_t)))
        _assert_same(out, refs, case, user, depth)
    assert device.num_program_cache_entries() == entries, "in-place rewrite recompiled"


def test_indexer_score_msa_metadata_contiguous_single_tensors(device):
    """Each tensor alone: the chunk-start tensor with a host cache_batch_idx, and the slot tensor with a host
    chunk_start_idx / kv_len (the classic reader selects the slot on its own)."""
    case = "grouped"
    q, k_cache = _inputs(seed=9)
    q_dev, k_dev = base.to_device(q, device), base.to_device(k_cache, device)
    refs = _host_references(q, k_cache, q_dev, k_dev, case)
    user, depth = 1, 128
    slot = user * LAYERS + LAYER
    out = ttnn.to_torch(_run(q_dev, k_dev, case, chunk_start_idx_tensor=_dev_u32(device, depth), cache_batch_idx=slot))
    _assert_same(out, refs, case, user, depth)
    out = ttnn.to_torch(
        _run(
            q_dev,
            k_dev,
            case,
            chunk_start_idx=depth,
            kv_len=depth + SQ,
            cache_batch_idx_tensor=_dev_u32(device, user),
            index_cache_num_layers=LAYERS,
            index_cache_layer_idx=LAYER,
        )
    )
    _assert_same(out, refs, case, user, depth)


def test_indexer_score_msa_metadata_valid_end_caps(device):
    """valid_end_tensor caps the derived kv_len at ceil32(valid_end): the capped dispatch equals the host path
    run with kv_len = ceil32(valid_end), on the columns both write."""
    case = "grouped"
    q, k_cache = _inputs(seed=13)
    q_dev, k_dev = base.to_device(q, device), base.to_device(k_cache, device)
    user, depth, valid_end = 0, 128, 128 + 20  # partial final chunk: 20 real tokens -> ceil32 = 160
    capped_len = 160
    host = ttnn.to_torch(
        _run(q_dev, k_dev, case, chunk_start_idx=depth, kv_len=capped_len, cache_batch_idx=user * LAYERS + LAYER)
    )
    meta = ttnn.to_torch(
        _run(
            q_dev,
            k_dev,
            case,
            valid_end_tensor=_dev_u32(device, valid_end),
            **_meta_kwargs(_dev_u32(device, depth), _dev_u32(device, user)),
        )
    )
    assert torch.equal(meta[..., :capped_len], host[..., :capped_len])


@pytest.mark.parametrize("device_params", [{"trace_region_size": 1 << 20}], indirect=True)
def test_indexer_score_msa_metadata_trace_retarget(device):
    """One captured trace, replayed across users and depths by rewriting the SAME metadata tensors in place.
    With host ints the replay would keep the capture-time slot and start; with tensors it follows each rewrite."""
    case = "pooled"
    q, k_cache = _inputs(seed=17)
    q_dev, k_dev = base.to_device(q, device), base.to_device(k_cache, device)
    refs = _host_references(q, k_cache, q_dev, k_dev, case)

    start_t, user_t = _dev_u32(device, DEPTHS[0]), _dev_u32(device, 0)
    _run(q_dev, k_dev, case, **_meta_kwargs(start_t, user_t))  # compile outside the capture
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    try:
        traced_out = _run(q_dev, k_dev, case, **_meta_kwargs(start_t, user_t))
    finally:
        ttnn.end_trace_capture(device, trace_id, cq_id=0)
    try:
        for user, depth in TARGETS + TARGETS[::-1]:
            ttnn.copy_host_to_device_tensor(_host_u32(depth), start_t)
            ttnn.copy_host_to_device_tensor(_host_u32(user), user_t)
            ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
            _assert_same(ttnn.to_torch(traced_out), refs, case, user, depth)
    finally:
        ttnn.release_trace(device, trace_id)


def test_indexer_score_msa_metadata_rejects(device, expect_error):
    """Host ints and their tensors are mutually exclusive, and the metadata container is pinned."""
    case = "grouped"
    q, k_cache = _inputs()
    q_dev, k_dev = base.to_device(q, device), base.to_device(k_cache, device)
    start_t, user_t = _dev_u32(device, 128), _dev_u32(device, 1)
    meta = _meta_kwargs(start_t, user_t)

    with expect_error(RuntimeError, "chunk_start_idx and chunk_start_idx_tensor are mutually exclusive"):
        _run(q_dev, k_dev, case, chunk_start_idx=128, **meta)
    with expect_error(RuntimeError, "kv_len must not be set alongside chunk_start_idx_tensor"):
        _run(q_dev, k_dev, case, kv_len=192, **meta)
    with expect_error(RuntimeError, "cache_batch_idx and cache_batch_idx_tensor are mutually exclusive"):
        _run(q_dev, k_dev, case, cache_batch_idx=1, **meta)
    with expect_error(RuntimeError, "valid_end_tensor requires chunk_start_idx_tensor"):
        _run(q_dev, k_dev, case, chunk_start_idx=128, cache_batch_idx=1, valid_end_tensor=_dev_u32(device, 150))
    with expect_error(RuntimeError, "must be in DRAM"):
        _run(q_dev, k_dev, case, **_meta_kwargs(_dev_u32(device, 128, ttnn.L1_MEMORY_CONFIG), user_t))
    with expect_error(RuntimeError, "must hold exactly 1 element"):
        _run(q_dev, k_dev, case, **_meta_kwargs(_dev_u32(device, 128, shape=(1, 1, 1, 8)), user_t))
    with expect_error(RuntimeError, "must be UINT32"):
        _run(q_dev, k_dev, case, **_meta_kwargs(_dev_u32(device, 128, dtype=ttnn.int32), user_t))
    with expect_error(RuntimeError, "index_cache_num_layers"):  # 3 does not divide the 4 user-major slots
        _run(
            q_dev, k_dev, case, chunk_start_idx_tensor=start_t, cache_batch_idx_tensor=user_t, index_cache_num_layers=3
        )
