# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""sp>1 coverage for indexer_score_msa trace-safe metadata on a block-cyclic K cache (multi-device).

Runs on (1,2) and (1,4) meshes (SP along cols; the mesh_device fixture skips when the devices are absent, so
single-card CI skips this file). K is replicated in block-cyclic (shard-major) order, as the SP AllGather'd
chunked-prefill cache is; q is SP-sharded. For each chunk start -- slab-aligned, and mid-slab starts that
rotate slab ownership and make the boundary rank straddle two slabs -- every rank must:
  * match a golden whose query positions come from the block-cyclic WRITER (token g lands on rank
    (g // chunk_local) % sp), not from the op's closed form, on the host-int path, and
  * match the host-int path bit-exactly on the metadata path (chunk_start_idx_tensor + cache_batch_idx_tensor),
    whose per-rank start, rotation, kv_len and K slot are derived in-kernel.

The tests below that one repeat the single-device metadata contract (one cached program across users and
starts, freshly allocated tensors on a cache hit, in-place rewrite, trace replay, valid_end_tensor, each
tensor alone) at sp>1, where device_index is no longer always 0: a per-coordinate patch that dropped a rank's
geometry, or a replay that fell back to rank 0's start, is invisible on one device.
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
import tests.ttnn.nightly.unit_tests.operations.experimental.indexer_score.test_indexer_score as base

HEADS, DIM, NUM_GROUPS = 4, 64, 2
CHUNK_LOCAL = 64  # == q seq-len per rank (pure-SP mesh)
N_CHUNKS = 4
SLOTS, SLOT = 2, 1  # 2-slot K cache; the op reads slot 1 (user 1, one layer)
SCALE = DIM**-0.5

# (num_groups, block_size, program_config)
CASES = {
    "grouped": (NUM_GROUPS, 0, dict(q_chunk_size=32, k_chunk_size=64, head_group_size=0)),
    "pooled": (1, 32, dict(q_chunk_size=32, k_chunk_size=256, head_group_size=0)),
}


def _natural_to_block_cyclic(t, sp, n_chunks, chunk_local):
    """Natural [B,1,T,d] (order (chunk, shard, local)) -> shard-major (shard, chunk, local)."""
    b, h, tt, d = t.shape
    t = t.reshape(b, h, n_chunks, sp, chunk_local, d).permute(0, 1, 3, 2, 4, 5)
    return t.reshape(b, h, tt, d).contiguous()


def _owned_positions(chunk_start, sp, chunk_local):
    owned = [[] for _ in range(sp)]
    for g in range(chunk_start, chunk_start + sp * chunk_local):
        owned[(g // chunk_local) % sp].append(g)
    return owned


def _golden(q, k, positions, kv_len, num_groups, block_size):
    """Per-group raw-dot scores of queries at explicit global positions over keys [0, kv_len); causal -inf;
    block-max-pooled with the forced-local +inf block when block_size > 0."""
    hog = HEADS // num_groups
    keys = k[0, 0, :kv_len].float()
    pos = torch.tensor(positions)
    future = torch.arange(kv_len).view(1, -1) > pos.view(-1, 1)
    planes = []
    for g in range(num_groups):
        s = sum(q[0, h].float() @ keys.T for h in range(g * hog, (g + 1) * hog)) * SCALE
        planes.append(s.masked_fill(future, float("-inf")))
    scores = torch.stack(planes).unsqueeze(0)  # [1, G, Sq, kv_len]
    if not block_size:
        return scores
    pooled = scores.reshape(1, num_groups, len(positions), kv_len // block_size, block_size).amax(dim=-1)
    pooled[:, :, torch.arange(len(positions)), pos // block_size] = float("inf")
    return pooled


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [(1, 2), (1, 4)], indirect=True)
@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize(
    "slab,chip,offset",
    [(1, 0, 0), (1, 1, 32), (2, 1, 0), (0, 0, 32)],
    ids=["aligned", "rotated_mid_slab", "rotated_slab_aligned", "chip0_mid_slab"],
)
def test_indexer_score_msa_metadata_block_cyclic_sp(mesh_device, case, slab, chip, offset):
    rows, cols = tuple(mesh_device.shape)
    sp_axis, sp = 1, cols
    if sp < 2:
        pytest.skip(f"needs sp>1 (mesh shape {(rows, cols)})")
    num_groups, block_size, cfg = CASES[case]
    T = sp * N_CHUNKS * CHUNK_LOCAL
    chunk_start = slab * sp * CHUNK_LOCAL + (chip % sp) * CHUNK_LOCAL + offset
    kv_len = chunk_start + sp * CHUNK_LOCAL  # history + this chunk: what the metadata path derives
    assert kv_len <= T
    if block_size and chunk_start % block_size:
        pytest.skip("pooling needs a block-aligned start")
    owned = _owned_positions(chunk_start, sp, CHUNK_LOCAL)

    gen = torch.Generator().manual_seed(chunk_start + 7 * sp)
    q_ranks = [torch.randn(1, HEADS, CHUNK_LOCAL, DIM, generator=gen, dtype=torch.bfloat16) for _ in range(sp)]
    k_cache = torch.randn(SLOTS, 1, T, DIM, generator=gen, dtype=torch.bfloat16)  # natural order, distinct slots

    shard = ttnn.ShardTensorToMesh(mesh_device, dim=2)  # (1, sp): device r along cols = SP rank r
    repl = ttnn.ReplicateTensorToMesh(mesh_device)
    q_dev = ttnn.from_torch(
        torch.cat(q_ranks, dim=2), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=shard
    )
    k_dev = ttnn.from_torch(
        _natural_to_block_cyclic(k_cache, sp, N_CHUNKS, CHUNK_LOCAL),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=repl,
    )

    def u32(value):
        return ttnn.from_torch(
            torch.tensor([[[[value]]]], dtype=torch.int64),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=repl,
        )

    def run(**kw):
        out = ttnn.experimental.indexer_score_msa(
            q_dev,
            k_dev,
            num_groups=num_groups,
            scale=SCALE,
            block_size=block_size,
            program_config=ttnn.IndexerScoreProgramConfig(**cfg),
            seq_shard_axes=[sp_axis],
            block_cyclic_sp_axis=sp_axis,
            block_cyclic_chunk_local=CHUNK_LOCAL,
            **kw,
        )
        return [ttnn.to_torch(t) for t in ttnn.get_device_tensors(out)]

    cols_valid = kv_len // block_size if block_size else kv_len
    host = run(chunk_start_idx=chunk_start, kv_len=kv_len, cache_batch_idx=SLOT)
    for r in range(sp):
        gold = _golden(q_ranks[r], k_cache[SLOT : SLOT + 1], owned[r], kv_len, num_groups, block_size)
        got = host[r][..., :cols_valid]
        if block_size:
            base.assert_pooled_match(got, gold, num_groups, CHUNK_LOCAL, cols_valid, pcc_floor=0.995)
        else:
            base.assert_grouped_match(got, gold, num_groups, CHUNK_LOCAL, cols_valid)

    meta = run(
        chunk_start_idx_tensor=u32(chunk_start),
        cache_batch_idx_tensor=u32(SLOT),  # user 1, one layer -> slot 1
        index_cache_num_layers=1,
        index_cache_layer_idx=0,
    )
    for r in range(sp):
        assert torch.equal(
            meta[r][..., :cols_valid], host[r][..., :cols_valid]
        ), f"sp={sp} start={chunk_start} {case}: rank {r} metadata != host path"


# --- retargeting the metadata at sp>1: cache hits, in-place rewrites, trace replay -------------------------
#
# A 4-slot user-major cache with a NON-TRIVIAL layer fold (slot = user * LAYERS + LAYER): with one layer at
# index 0 the fold is the identity, so a reader that ignored it would still land on the right slot.
USERS, LAYERS, LAYER = 2, 2, 1


def _retarget_starts(sp):
    """Slab-aligned, rotated mid-slab (boundary chip 1, so ownership rotates and that rank straddles), and a
    later slab-aligned start. All block-aligned, so the pooled case takes them too."""
    return (0, sp * CHUNK_LOCAL + CHUNK_LOCAL + 32, sp * CHUNK_LOCAL)


def _mesh_u32(mesh_device, value, *, on_device=True):
    kwargs = {
        "dtype": ttnn.uint32,
        "layout": ttnn.ROW_MAJOR_LAYOUT,
        "mesh_mapper": ttnn.ReplicateTensorToMesh(mesh_device),
    }
    if on_device:
        kwargs.update(device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.from_torch(torch.tensor([[[[value]]]], dtype=torch.int64), **kwargs)


def _mesh_inputs(mesh_device, sp, seed, *, block_cyclic=True):
    """SP-sharded q (one CHUNK_LOCAL chunk per rank) and a replicated USERS*LAYERS-slot K cache, laid out
    block-cyclic (shard-major) or left in natural token order for the contiguous path."""
    T = sp * N_CHUNKS * CHUNK_LOCAL
    gen = torch.Generator().manual_seed(seed)
    q_ranks = [torch.randn(1, HEADS, CHUNK_LOCAL, DIM, generator=gen, dtype=torch.bfloat16) for _ in range(sp)]
    k_cache = torch.randn(USERS * LAYERS, 1, T, DIM, generator=gen, dtype=torch.bfloat16)  # distinct slots
    q_dev = ttnn.from_torch(
        torch.cat(q_ranks, dim=2),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=2),
    )
    k_dev = ttnn.from_torch(
        _natural_to_block_cyclic(k_cache, sp, N_CHUNKS, CHUNK_LOCAL) if block_cyclic else k_cache,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    return q_ranks, k_cache, q_dev, k_dev


def _owned(start, sp, *, block_cyclic):
    """Global query positions per rank. Block-cyclic: the writer's striping (rotation + straddle). Contiguous:
    one unbroken run per rank at start + rank*Sq, which is what the no-block-cyclic geometry branch produces."""
    if block_cyclic:
        return _owned_positions(start, sp, CHUNK_LOCAL)
    return [list(range(start + r * CHUNK_LOCAL, start + (r + 1) * CHUNK_LOCAL)) for r in range(sp)]


def _dispatch(q_dev, k_dev, case, sp_axis, *, block_cyclic=True, **kw):
    num_groups, block_size, cfg = CASES[case]
    layout = dict(block_cyclic_sp_axis=sp_axis, block_cyclic_chunk_local=CHUNK_LOCAL) if block_cyclic else {}
    return ttnn.experimental.indexer_score_msa(
        q_dev,
        k_dev,
        num_groups=num_groups,
        scale=SCALE,
        block_size=block_size,
        program_config=ttnn.IndexerScoreProgramConfig(**cfg),
        seq_shard_axes=[sp_axis],
        **layout,
        **kw,
    )


def _shards(out):
    return [ttnn.to_torch(t) for t in ttnn.get_device_tensors(out)]


def _cols(case, kv_len):
    """Output columns this dispatch writes: [0, kv_len) keys, or its whole blocks when pooling."""
    block_size = CASES[case][1]
    return kv_len // block_size if block_size else kv_len


def _meta_kwargs(start_t, user_t):
    return dict(
        chunk_start_idx_tensor=start_t,
        cache_batch_idx_tensor=user_t,
        index_cache_num_layers=LAYERS,
        index_cache_layer_idx=LAYER,
    )


def _host_references(q_ranks, k_cache, q_dev, k_dev, case, sp, sp_axis, targets, *, block_cyclic=True):
    """Host-int output per (user, start), every rank anchored to the writer-derived golden on its own slot."""
    num_groups, block_size, _ = CASES[case]
    refs = {}
    for user, start in targets:
        slot = user * LAYERS + LAYER
        kv_len = start + sp * CHUNK_LOCAL
        cols = _cols(case, kv_len)
        shards = _shards(
            _dispatch(
                q_dev,
                k_dev,
                case,
                sp_axis,
                block_cyclic=block_cyclic,
                chunk_start_idx=start,
                kv_len=kv_len,
                cache_batch_idx=slot,
            )
        )
        owned = _owned(start, sp, block_cyclic=block_cyclic)
        for r in range(sp):
            gold = _golden(q_ranks[r], k_cache[slot : slot + 1], owned[r], kv_len, num_groups, block_size)
            if block_size:
                base.assert_pooled_match(shards[r][..., :cols], gold, num_groups, CHUNK_LOCAL, cols, pcc_floor=0.995)
            else:
                base.assert_grouped_match(shards[r][..., :cols], gold, num_groups, CHUNK_LOCAL, cols)
        refs[(user, start)] = shards
    _assert_targets_distinguishable(refs, case, sp)
    return refs


def _assert_targets_distinguishable(refs, case, sp):
    """Every (user, start) must produce a different score tensor, otherwise 'metadata == host path' below
    would hold even if the metadata tensors were never read: a stale slot or start would look identical."""
    flat = {key: torch.cat([shard.flatten() for shard in shards]) for key, shards in refs.items()}
    keys = list(refs)
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            assert not torch.equal(flat[a], flat[b]), (
                f"sp={sp} {case}: targets {a} and {b} scored identically, so the retarget assertions would be "
                "vacuous -- pick starts/users that actually change the output"
            )


def _assert_same(shards, refs, key, case, sp):
    user, start = key
    cols = _cols(case, start + sp * CHUNK_LOCAL)
    for r in range(sp):
        assert torch.equal(
            shards[r][..., :cols], refs[key][r][..., :cols]
        ), f"sp={sp} {case}: rank {r} metadata != host path for user={user} start={start}"


def _sp_or_skip(mesh_device):
    rows, cols = tuple(mesh_device.shape)
    if cols < 2:
        pytest.skip(f"needs sp>1 (mesh shape {(rows, cols)})")
    return 1, cols  # SP along cols: device r = SP rank r


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [(1, 2), (1, 4)], indirect=True)
@pytest.mark.parametrize("case", list(CASES))
def test_indexer_score_msa_metadata_sp_cache_hit(mesh_device, case):
    """2 users x 3 starts on ONE cached mesh program. Every dispatch passes FRESHLY allocated metadata tensors
    (earlier ones kept alive -> new addresses), so a hit that kept the build-time addresses would score the
    wrong user or start; and override_runtime_arguments has to repoint them on every coordinate while leaving
    that coordinate's own device_index and rotation intact. Then the same pair is rewritten in place."""
    sp_axis, sp = _sp_or_skip(mesh_device)
    q_ranks, k_cache, q_dev, k_dev = _mesh_inputs(mesh_device, sp, seed=23)
    targets = [(user, start) for user in range(USERS) for start in _retarget_starts(sp)]
    refs = _host_references(q_ranks, k_cache, q_dev, k_dev, case, sp, sp_axis, targets)

    live, entries = [], None
    for key in targets:
        user, start = key
        start_t, user_t = _mesh_u32(mesh_device, start), _mesh_u32(mesh_device, user)
        live += [start_t, user_t]
        out = _dispatch(q_dev, k_dev, case, sp_axis, **_meta_kwargs(start_t, user_t))
        _assert_same(_shards(out), refs, key, case, sp)
        if entries is None:
            entries = mesh_device.num_program_cache_entries()
    assert mesh_device.num_program_cache_entries() == entries, "switching user / start tensors recompiled"
    assert len({t.buffer_address() for t in live}) == len(live), "metadata tensors were not distinct allocations"

    start_t, user_t = _mesh_u32(mesh_device, 0), _mesh_u32(mesh_device, 0)
    for key in targets[::-1]:
        user, start = key
        ttnn.copy_host_to_device_tensor(_mesh_u32(mesh_device, start, on_device=False), start_t)
        ttnn.copy_host_to_device_tensor(_mesh_u32(mesh_device, user, on_device=False), user_t)
        out = _dispatch(q_dev, k_dev, case, sp_axis, **_meta_kwargs(start_t, user_t))
        _assert_same(_shards(out), refs, key, case, sp)
    assert mesh_device.num_program_cache_entries() == entries, "in-place rewrite recompiled"


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [(1, 2), (1, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 1 << 20}], indirect=True)
def test_indexer_score_msa_metadata_sp_trace_retarget(mesh_device):
    """One captured MESH trace, replayed across users and starts by rewriting the same replicated metadata
    tensors in place. Host ints would freeze the capture-time slot and start; a replay that lost the
    per-coordinate geometry and fell back to rank 0's start would diverge on every rotated rank."""
    case = "pooled"
    sp_axis, sp = _sp_or_skip(mesh_device)
    q_ranks, k_cache, q_dev, k_dev = _mesh_inputs(mesh_device, sp, seed=29)
    targets = [(user, start) for user in range(USERS) for start in _retarget_starts(sp)]
    refs = _host_references(q_ranks, k_cache, q_dev, k_dev, case, sp, sp_axis, targets)

    start_t, user_t = _mesh_u32(mesh_device, targets[0][1]), _mesh_u32(mesh_device, 0)
    _dispatch(q_dev, k_dev, case, sp_axis, **_meta_kwargs(start_t, user_t))  # compile outside the capture
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    try:
        traced = _dispatch(q_dev, k_dev, case, sp_axis, **_meta_kwargs(start_t, user_t))
    finally:
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    try:
        for key in targets + targets[::-1]:
            user, start = key
            ttnn.copy_host_to_device_tensor(_mesh_u32(mesh_device, start, on_device=False), start_t)
            ttnn.copy_host_to_device_tensor(_mesh_u32(mesh_device, user, on_device=False), user_t)
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
            _assert_same(_shards(traced), refs, key, case, sp)
    finally:
        ttnn.release_trace(mesh_device, trace_id)


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [(1, 2), (1, 4)], indirect=True)
@pytest.mark.parametrize("case", list(CASES))
def test_indexer_score_msa_metadata_sp_valid_end_caps(mesh_device, case):
    """valid_end_tensor caps the bound the reader DERIVES at sp>1, which is start + sp*chunk_local (the whole
    global chunk), not one rank's Sq. On a rotated mid-slab start, the capped dispatch must equal the host
    path run with kv_len = ceil32(valid_end) on every rank -- and must stop there: capping only SHRINKS the
    written extent, so comparing the shared columns alone would hold even if valid_end were never read. The
    uncapped dispatch (same start, no valid_end_tensor) is run too, and must differ past the cap."""
    sp_axis, sp = _sp_or_skip(mesh_device)
    q_ranks, _, q_dev, k_dev = _mesh_inputs(mesh_device, sp, seed=31)
    user, start = 1, _retarget_starts(sp)[1]  # rotated mid-slab: the cap rides the rotated geometry
    derived_len = start + sp * CHUNK_LOCAL  # what the reader derives with no cap
    capped_len = derived_len - 32  # one write grid short of it
    valid_end = capped_len - 20  # partial final chunk: 12 real tokens in it -> ceil32 == capped_len
    cols = _cols(case, capped_len)
    tail = slice(cols, _cols(case, derived_len))

    host = _shards(
        _dispatch(
            q_dev,
            k_dev,
            case,
            sp_axis,
            chunk_start_idx=start,
            kv_len=capped_len,
            cache_batch_idx=user * LAYERS + LAYER,
        )
    )
    meta = _shards(
        _dispatch(
            q_dev,
            k_dev,
            case,
            sp_axis,
            valid_end_tensor=_mesh_u32(mesh_device, valid_end),
            **_meta_kwargs(_mesh_u32(mesh_device, start), _mesh_u32(mesh_device, user)),
        )
    )
    uncapped = _shards(
        _dispatch(
            q_dev,
            k_dev,
            case,
            sp_axis,
            **_meta_kwargs(_mesh_u32(mesh_device, start), _mesh_u32(mesh_device, user)),
        )
    )
    for r in range(sp):
        assert torch.equal(
            meta[r][..., :cols], host[r][..., :cols]
        ), f"sp={sp} {case}: rank {r} capped metadata != host path at kv_len={capped_len}"
        # The uncapped run scores this column range; the capped run must leave it alone, so the two cannot
        # agree there. Equality means valid_end_tensor never reached the reader's bound.
        assert not torch.equal(
            meta[r][..., tail], uncapped[r][..., tail]
        ), f"sp={sp} {case}: rank {r} scored past valid_end -- the cap was ignored"


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [(1, 2), (1, 4)], indirect=True)
def test_indexer_score_msa_metadata_sp_single_tensors(mesh_device):
    """Each tensor alone on a rotated start: the start tensor with a host cache_batch_idx (geometry derived
    on device, slot from the host arg), and the slot tensor with host chunk_start_idx / kv_len (geometry from
    the host's per-coordinate patch, slot recomposed on device)."""
    case = "grouped"
    sp_axis, sp = _sp_or_skip(mesh_device)
    q_ranks, k_cache, q_dev, k_dev = _mesh_inputs(mesh_device, sp, seed=37)
    user, start = 1, _retarget_starts(sp)[1]
    key = (user, start)
    refs = _host_references(q_ranks, k_cache, q_dev, k_dev, case, sp, sp_axis, [key])

    out = _dispatch(
        q_dev,
        k_dev,
        case,
        sp_axis,
        chunk_start_idx_tensor=_mesh_u32(mesh_device, start),
        cache_batch_idx=user * LAYERS + LAYER,
    )
    _assert_same(_shards(out), refs, key, case, sp)

    out = _dispatch(
        q_dev,
        k_dev,
        case,
        sp_axis,
        chunk_start_idx=start,
        kv_len=start + sp * CHUNK_LOCAL,
        cache_batch_idx_tensor=_mesh_u32(mesh_device, user),
        index_cache_num_layers=LAYERS,
        index_cache_layer_idx=LAYER,
    )
    _assert_same(_shards(out), refs, key, case, sp)


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [(1, 2), (1, 4)], indirect=True)
@pytest.mark.parametrize("case", list(CASES))
def test_indexer_score_msa_metadata_sp_contiguous(mesh_device, case):
    """The CONTIGUOUS (no block_cyclic_*) SP path at sp>1. K stays in natural token order and each rank owns one
    unbroken run at start + rank*Sq, so the geometry takes the no-block-cyclic branch (linear, both straddle
    fields zero) and the reader derives kv_len from the seq_ring*Sq extent -- a ring size it recovers from q's
    device coordinates rather than from a block-cyclic descriptor. Same three claims as the block-cyclic file:
    host-int vs golden on every rank, metadata bit-exact with it, and one cached program across all targets."""
    sp_axis, sp = _sp_or_skip(mesh_device)
    q_ranks, k_cache, q_dev, k_dev = _mesh_inputs(mesh_device, sp, seed=41, block_cyclic=False)
    # No slab structure here, so any tile-aligned start works; the last one is not a multiple of Sq.
    starts = (0, sp * CHUNK_LOCAL, sp * CHUNK_LOCAL + 32)
    targets = [(user, start) for user in range(USERS) for start in starts]
    refs = _host_references(q_ranks, k_cache, q_dev, k_dev, case, sp, sp_axis, targets, block_cyclic=False)

    entries = None
    for key in targets:
        user, start = key
        out = _dispatch(
            q_dev,
            k_dev,
            case,
            sp_axis,
            block_cyclic=False,
            **_meta_kwargs(_mesh_u32(mesh_device, start), _mesh_u32(mesh_device, user)),
        )
        _assert_same(_shards(out), refs, key, case, sp)
        if entries is None:
            entries = mesh_device.num_program_cache_entries()
    assert mesh_device.num_program_cache_entries() == entries, "switching user / start tensors recompiled"
