# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""sp>1 block-cyclic remap coverage for sparse_sdpa_msa (multi-device).

The post-commit file only covers sp=1, where the invP block remap is the identity. This runs the REAL
permutation at sp=2 and sp=4 via the `mesh_device` fixture (auto-skips when the devices aren't present).
Inputs are replicated across the mesh; K/V are laid out block-cyclic (shard-major, as the AllGather'd
chunked-prefill cache is), block-ids stay NATURAL. The check is remap-transparency: the block-cyclic op
must match the PLAIN op (natural-order K/V, no remap) run with identical inputs and the same per-device
chunk_start. Under causal this is the key case — the diagonal-block mask must stay keyed on the logical
block id, not the remapped physical one; if the remap leaked into the mask, bc would diverge from plain.
Op-vs-golden correctness is covered single-device (post-commit file); here plain is additionally checked
against the layout-agnostic golden in the non-causal case.
"""

import pytest
import torch

import ttnn

from models.common.utility_functions import run_for_blackhole
from tests.ttnn.unit_tests.operations.sdpa.sparse_sdpa_msa_test_utils import (
    BLK_KV,
    make_msa_inputs,
    pcc,
    sparse_attention_ref_msa,
)

DEVICE_PCC = 0.99


def _natural_to_block_cyclic(t, sp, n_chunks, chunk_local):
    """Natural [1,H,T,d] (T = n_chunks*sp*chunk_local, order (chunk, shard, local)) -> block-cyclic
    shard-major (shard, chunk, local), the layout AllGather produces from the per-shard cache."""
    H, T, d = t.shape[1], t.shape[2], t.shape[3]
    t = t.reshape(1, H, n_chunks, sp, chunk_local, d)
    t = t.permute(0, 1, 3, 2, 4, 5)  # (chunk, shard) -> (shard, chunk)
    return t.reshape(1, H, T, d).contiguous()


@run_for_blackhole()  # sparse_sdpa_msa is Blackhole-only; nightly runs this dir on wh_n300 too
@pytest.mark.parametrize("mesh_device", [(1, 2), (1, 4)], indirect=True)  # SP along cols; fixture skips if absent
@pytest.mark.parametrize("n_chunks", [8])
@pytest.mark.parametrize("causal", [False, True])  # True: diagonal-block mask must stay on the logical id
def test_msa_native_block_cyclic_sp_gt1_matches_plain(mesh_device, n_chunks, causal):
    rows, cols = tuple(mesh_device.shape)
    sp_axis, sp = 1, cols
    if sp < 2:
        pytest.skip(f"needs sp>1 (mesh shape {(rows, cols)})")

    H, n_kv, S, d = 32, 1, 2 * BLK_KV, 128  # S = 2 blocks -> chunk_local spans >1 block (non-trivial invP divide)
    chunk_local = S  # tp=1 (pure-SP mesh) -> guard requires chunk_local == q_isl (= S)
    T = sp * n_chunks * chunk_local
    nblk = T // BLK_KV
    topk = 16  # multiple of 16 (indices row 64B-aligned) and <= nblk
    assert nblk >= topk and chunk_local // BLK_KV > 1 and n_chunks > 1, f"degenerate remap params: nblk={nblk}"
    q, k, v, indices = make_msa_inputs(H, n_kv, S, T, topk=topk, d=d, causal=causal, seed=T)
    k_bc = _natural_to_block_cyclic(k, sp, n_chunks, chunk_local)
    v_bc = _natural_to_block_cyclic(v, sp, n_chunks, chunk_local)

    repl = ttnn.ReplicateTensorToMesh(mesh_device)

    def dev_rm(x, dt):
        return ttnn.from_torch(
            x,
            dtype=dt,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=repl,
        )

    def dev_tile(x):
        return ttnn.from_torch(
            x.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=repl,
        )

    # Both runs get the same per-device chunk_start (compute_chunk_start_local, from the mesh coord); the only
    # difference is K/V layout + the remap, so equal outputs prove the remap is correctness-transparent.
    def run_op(k_in, v_in, bc):
        kw = dict(scale=d**-0.5, block_size=BLK_KV, chunk_start_idx=0 if causal else None)
        if bc:
            kw.update(block_cyclic_sp_axis=sp_axis, block_cyclic_chunk_local=chunk_local)
        out = ttnn.transformer.sparse_sdpa_msa(
            dev_rm(q.to(torch.float32), ttnn.bfloat16),
            dev_tile(k_in),
            dev_tile(v_in),
            dev_rm(indices.to(torch.int32), ttnn.uint32),
            **kw,
        )
        return [ttnn.to_torch(s)[:, :H] for s in ttnn.get_device_tensors(out)]

    plain = run_op(k, v, bc=False)
    blockc = run_op(k_bc, v_bc, bc=True)
    for i, (p_out, b_out) in enumerate(zip(plain, blockc)):
        p = pcc(b_out, p_out)
        assert p >= DEVICE_PCC, f"sp={sp} causal={causal}: block-cyclic != plain on dev {i} (pcc={p:.5f}, T={T})"

    if not causal:  # non-causal is device-uniform -> also anchor plain to the layout-agnostic golden (correctness)
        gold = sparse_attention_ref_msa(q, k, v, indices, d**-0.5)
        p = pcc(plain[0], gold)
        assert p >= DEVICE_PCC, f"sp={sp}: plain op vs golden pcc={p:.5f}"


def _block_cyclic_chunk_positions(chunk_start, sp, chunk_local):
    """Global positions of the chunk_local query rows each SP rank holds for the chunk
    [chunk_start, chunk_start + sp*chunk_local), in the rank's local-row order.

    Brute force from the KV writer's placement (update_padded_kv_cache: position g lives on rank
    (g // chunk_local) % sp at local row (g // chunk_global) * chunk_local + g % chunk_local), NOT from the
    closed form under test. A mid-slab chunk_start rotates which rank holds the chunk's first block, and the
    boundary rank's rows jump from the tail of one slab block to the head of its next one."""
    chunk_global = sp * chunk_local
    per_rank = [[] for _ in range(sp)]
    for g in range(chunk_start, chunk_start + chunk_global):
        local_row = (g // chunk_global) * chunk_local + g % chunk_local
        per_rank[(g // chunk_local) % sp].append((local_row, g))
    return [torch.tensor([g for _, g in sorted(rows)]) for rows in per_rank]


def _diag_plus_past_indices(positions, n_kv, topk, n_past, gen):
    """Per-query block ids: the query's own (diagonal) block plus up to n_past random past blocks, sentinel
    tail. Keeping the selection small makes the diagonal block a large share of the attention, so a query
    masked at the wrong position (future keys leaking into, or valid keys cut from, its own block) moves the
    output well past the PCC threshold instead of hiding under 16 blocks of averaging."""
    S = positions.numel()
    idx = torch.full((1, n_kv, S, topk), -1, dtype=torch.int32)
    for g in range(n_kv):
        for s, p in enumerate(positions.tolist()):
            diag = p // BLK_KV
            past = torch.randperm(diag, generator=gen)[: min(n_past, diag)].tolist() if diag else []
            chosen = sorted([diag] + past)
            idx[0, g, s, : len(chosen)] = torch.tensor(chosen, dtype=torch.int32)
    return idx


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [(1, 2), (1, 4)], indirect=True)  # SP along cols; fixture skips if absent
@pytest.mark.parametrize(
    "start_offset",
    [0, 32, 128, 256, 352],
    ids=["slab_aligned", "mid_block_straddle", "block_aligned_straddle", "rotated", "rotated_straddle"],
)
def test_msa_block_cyclic_mid_slab_causal(mesh_device, start_offset):
    """Causal sparse_sdpa_msa over a block-cyclic cache when the chunk starts mid-slab (a multi-turn resume at
    a 32-token boundary). Each SP rank's query rows sit at the KV writer's rotated positions, not the linear
    chunk_start + rank*S; the op must derive them (compute_causal_geometry) so the diagonal-block mask lands on
    each query's true position. Checked per rank against the golden at the brute-force positions."""
    rows, cols = tuple(mesh_device.shape)
    sp_axis, sp = 1, cols
    if rows != 1 or sp < 2:
        pytest.skip(f"needs a (1, sp>1) mesh (got {(rows, cols)})")

    H, n_kv, d = 32, 1, 128
    chunk_local = S = 2 * BLK_KV  # one rank's query rows == the block-cyclic per-shard chunk
    chunk_global = sp * chunk_local
    n_slabs = 4
    T = n_slabs * chunk_global
    chunk_start = chunk_global + start_offset  # one whole prior slab + the mid-slab offset
    assert chunk_start + chunk_global <= T
    topk, n_past = 16, 3

    gen = torch.Generator().manual_seed(1000 + start_offset)
    k = torch.randn(1, n_kv, T, d, generator=gen)
    v = torch.randn(1, n_kv, T, d, generator=gen)
    positions = _block_cyclic_chunk_positions(chunk_start, sp, chunk_local)
    qs = [torch.randn(1, H, S, d, generator=gen) for _ in range(sp)]
    idxs = [_diag_plus_past_indices(pos, n_kv, topk, n_past, gen) for pos in positions]
    scale = d**-0.5

    golds = [
        sparse_attention_ref_msa(q_r, k, v, i_r, scale, causal=True, q_positions=pos)
        for q_r, i_r, pos in zip(qs, idxs, positions)
    ]
    if start_offset % chunk_global:
        # Sanity: the case must discriminate -- masking at the old linear positions would fail the threshold.
        linear = [
            sparse_attention_ref_msa(q_r, k, v, i_r, scale, causal=True, chunk_start_idx=chunk_start + r * S)
            for r, (q_r, i_r) in enumerate(zip(qs, idxs))
        ]
        assert min(pcc(lin, gold) for lin, gold in zip(linear, golds)) < DEVICE_PCC

    seq_shard = ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 2), mesh_shape=(rows, cols))
    repl = ttnn.ReplicateTensorToMesh(mesh_device)

    def dev(x, dt, layout, mapper):
        return ttnn.from_torch(
            x,
            dtype=dt,
            layout=layout,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )

    out = ttnn.transformer.sparse_sdpa_msa(
        dev(torch.cat(qs, dim=2).to(torch.float32), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, seq_shard),
        dev(
            _natural_to_block_cyclic(k, sp, n_slabs, chunk_local).to(torch.bfloat16),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            repl,
        ),
        dev(
            _natural_to_block_cyclic(v, sp, n_slabs, chunk_local).to(torch.bfloat16),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            repl,
        ),
        dev(torch.cat(idxs, dim=2), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, seq_shard),
        scale=scale,
        block_size=BLK_KV,
        chunk_start_idx=chunk_start,
        cluster_axis=sp_axis,
        block_cyclic_sp_axis=sp_axis,
        block_cyclic_chunk_local=chunk_local,
    )
    for r, (dev_out, gold) in enumerate(zip(ttnn.get_device_tensors(out), golds)):
        p = pcc(ttnn.to_torch(dev_out)[:, :H], gold)
        assert p >= DEVICE_PCC, f"sp={sp} chunk_start={chunk_start}: rank {r} diverges from golden (pcc={p:.5f})"


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [(2, 2), (2, 4)], indirect=True)  # SP along cols, TP sub-shard along rows
@pytest.mark.parametrize("start_offset", [0, 32, 288], ids=["slab_aligned", "mid_block_straddle", "rotated_straddle"])
def test_msa_block_cyclic_mid_slab_causal_tp_subshard(mesh_device, start_offset):
    """Causal sparse_sdpa_msa with q seq-sharded over BOTH mesh axes (block_cyclic_chunk_local == tp*S): device
    (tp r, sp c) holds rows [r*S, (r+1)*S) of SP rank c's chunk_local rotated rows. The mask must use that
    [SP, TP] position (as indexer_score does with seq_shard_axes=[SP, TP]), not chunk_start + sp_rank*S."""
    rows, cols = tuple(mesh_device.shape)
    sp_axis, sp, tp = 1, cols, rows
    if tp < 2 or sp < 2:
        pytest.skip(f"needs a (tp>1, sp>1) mesh (got {(rows, cols)})")

    H, n_kv, d = 32, 1, 128
    S = BLK_KV
    chunk_local = tp * S
    chunk_global = sp * chunk_local
    n_slabs = 4
    T = n_slabs * chunk_global
    chunk_start = chunk_global + start_offset
    topk, n_past = 16, 3

    gen = torch.Generator().manual_seed(2000 + start_offset)
    k = torch.randn(1, n_kv, T, d, generator=gen)
    v = torch.randn(1, n_kv, T, d, generator=gen)
    sp_positions = _block_cyclic_chunk_positions(chunk_start, sp, chunk_local)
    positions = [[sp_positions[c][r * S : (r + 1) * S] for c in range(sp)] for r in range(tp)]
    qs = [[torch.randn(1, H, S, d, generator=gen) for _ in range(sp)] for _ in range(tp)]
    idxs = [[_diag_plus_past_indices(positions[r][c], n_kv, topk, n_past, gen) for c in range(sp)] for r in range(tp)]
    scale = d**-0.5

    # [tp, ., sp*S, .] with rows sharding dim 0 and cols dim 2 -> device (r, c) gets its own [1, ., S, .].
    q_all = torch.cat([torch.cat(qs[r], dim=2) for r in range(tp)], dim=0)
    idx_all = torch.cat([torch.cat(idxs[r], dim=2) for r in range(tp)], dim=0)
    shard = ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 2), mesh_shape=(rows, cols))
    repl = ttnn.ReplicateTensorToMesh(mesh_device)

    def dev(x, dt, layout, mapper):
        return ttnn.from_torch(
            x, dtype=dt, layout=layout, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mapper
        )

    out = ttnn.transformer.sparse_sdpa_msa(
        dev(q_all.to(torch.float32), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, shard),
        dev(
            _natural_to_block_cyclic(k, sp, n_slabs, chunk_local).to(torch.bfloat16),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            repl,
        ),
        dev(
            _natural_to_block_cyclic(v, sp, n_slabs, chunk_local).to(torch.bfloat16),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            repl,
        ),
        dev(idx_all, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, shard),
        scale=scale,
        block_size=BLK_KV,
        chunk_start_idx=chunk_start,
        cluster_axis=sp_axis,
        block_cyclic_sp_axis=sp_axis,
        block_cyclic_chunk_local=chunk_local,
    )
    dev_outs = ttnn.get_device_tensors(out)
    for r in range(tp):
        for c in range(sp):
            gold = sparse_attention_ref_msa(qs[r][c], k, v, idxs[r][c], scale, causal=True, q_positions=positions[r][c])
            p = pcc(ttnn.to_torch(dev_outs[r * cols + c])[:, :H], gold)
            assert p >= DEVICE_PCC, f"mesh {(rows, cols)} chunk_start={chunk_start}: device ({r},{c}) pcc={p:.5f}"


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
def test_msa_block_cyclic_causal_guards(mesh_device, expect_error):
    """A mid-slab causal start needs cluster_axis (the rotated positions come from the SP rank), and that
    cluster_axis must be the cache's block-cyclic SP axis."""
    rows, cols = tuple(mesh_device.shape)
    H, n_kv, d = 32, 1, 128
    chunk_local = S = 2 * BLK_KV
    sp, n_slabs = cols, 4  # T >= topk blocks
    T = n_slabs * sp * chunk_local
    q, k, v, indices = make_msa_inputs(H, n_kv, S, T, topk=16, d=d, causal=False, seed=1)
    repl = ttnn.ReplicateTensorToMesh(mesh_device)

    def dev(x, dt, layout):
        return ttnn.from_torch(
            x, dtype=dt, layout=layout, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=repl
        )

    def run(**kw):
        return ttnn.transformer.sparse_sdpa_msa(
            dev(q.to(torch.float32), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
            dev(k.to(torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT),
            dev(v.to(torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT),
            dev(indices.to(torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
            scale=d**-0.5,
            block_size=BLK_KV,
            block_cyclic_sp_axis=1,
            block_cyclic_chunk_local=chunk_local,
            **kw,
        )

    with expect_error(RuntimeError, "needs cluster_axis"):
        run(chunk_start_idx=32)
    with expect_error(RuntimeError, "must equal block_cyclic_sp_axis"):
        run(chunk_start_idx=0, cluster_axis=0)


def _owned_positions(chunk_start, sp, chunk_local):
    """Global positions of each SP rank's queries, from the block-cyclic WRITER's semantics (independent of the
    op's closed form): the global chunk [chunk_start, chunk_start + sp*chunk_local) is striped so token g lands on
    rank (g // chunk_local) % sp, and each rank's queries are its tokens in order. A mid-slab start rotates which
    rank owns which slab and splits the boundary rank's queries across two slabs."""
    owned = [[] for _ in range(sp)]
    for g in range(chunk_start, chunk_start + sp * chunk_local):
        owned[(g // chunk_local) % sp].append(g)
    return owned


def _causal_indices_at(positions, topk, gen):
    """Block ids for queries at the given global positions: visible blocks only, own block always selected."""
    idx = torch.full((1, 1, len(positions), topk), -1, dtype=torch.int32)
    for s, p in enumerate(positions):
        local = p // BLK_KV
        visible = local + 1
        if visible <= topk:
            chosen = torch.arange(visible)
        else:
            pool = torch.randperm(visible, generator=gen)[:topk]
            if local not in pool.tolist():
                pool[-1] = local
            chosen = pool.sort().values
        idx[0, 0, s, : chosen.numel()] = chosen.to(torch.int32)
    return idx


def _ref_at_positions(q, k, v, indices, scale, positions):
    """Causal MSA golden with an explicit global position per query row (n_kv == 1)."""
    out = torch.zeros(1, q.shape[1], q.shape[2], v.shape[-1])
    for s, p in enumerate(positions):
        blocks = [int(b) for b in indices[0, 0, s] if b >= 0]
        keys = torch.cat([torch.arange(b * BLK_KV, (b + 1) * BLK_KV) for b in blocks])
        scores = (q[0, :, s].float() * scale) @ k[0, 0, keys].float().T
        scores = scores.masked_fill(keys.view(1, -1) > p, float("-inf"))
        out[0, :, s] = scores.softmax(dim=-1) @ v[0, 0, keys].float()
    return out


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [(1, 2), (1, 4)], indirect=True)
@pytest.mark.parametrize(
    "slab,chip,offset",
    [(1, 0, 0), (1, 1, 96), (2, 1, 160), (0, 0, 64)],
    ids=["aligned", "rotated_mid_block", "rotated_straddle_tile", "chip0_mid_slab"],
)
def test_msa_block_cyclic_rotated_start(mesh_device, slab, chip, offset):
    """EAGER causal correctness for a chunk that starts MID-SLAB on a block-cyclic cache. Each rank's queries sit
    at the positions the cache writer gave them (rotated ownership + the boundary rank's slab straddle), not at
    the linear chunk_start + rank*S. The host-int path (the rotation-exact geometry the indexer also uses) must
    match the writer-derived golden on every rank, and the metadata path (chunk_start_idx_tensor +
    cache_batch_idx_tensor) must match the host-int path bit-exactly: its per-rank start and rotation are derived
    in-kernel, and reader AND writer select the K/V slot of a 2-slot cache from the user-id tensor."""
    rows, cols = tuple(mesh_device.shape)
    sp_axis, sp = 1, cols
    if sp < 2:
        pytest.skip(f"needs sp>1 (mesh shape {(rows, cols)})")
    H, S, d, topk, n_chunks = 32, 2 * BLK_KV, 128, 16, 6
    chunk_local = S
    T = sp * n_chunks * chunk_local
    chip = chip % sp
    chunk_start = slab * sp * chunk_local + chip * chunk_local + offset
    assert chunk_start + sp * chunk_local <= T
    owned = _owned_positions(chunk_start, sp, chunk_local)
    assert all(len(p) == S for p in owned)

    gen = torch.Generator().manual_seed(chunk_start + sp)
    q_ranks = [torch.randn(1, H, S, d, generator=gen) for _ in range(sp)]
    slots, slot = 2, 1  # distinct slots, so a wrong-slot gather on either kernel changes the output
    k = torch.randn(slots, 1, T, d, generator=gen)
    v = torch.randn(slots, 1, T, d, generator=gen)
    idx_ranks = [_causal_indices_at(owned[r], topk, gen) for r in range(sp)]

    shard = ttnn.ShardTensorToMesh(mesh_device, dim=2)  # (1, sp) mesh: device r along cols = SP rank r
    repl = ttnn.ReplicateTensorToMesh(mesh_device)

    def dev(x, dt, layout, mapper):
        return ttnn.from_torch(
            x, dtype=dt, layout=layout, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mapper
        )

    q_dev = dev(torch.cat(q_ranks, dim=2), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, shard)
    idx_dev = dev(torch.cat(idx_ranks, dim=2), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, shard)
    k_bc = torch.cat([_natural_to_block_cyclic(k[b : b + 1], sp, n_chunks, chunk_local) for b in range(slots)])
    v_bc = torch.cat([_natural_to_block_cyclic(v[b : b + 1], sp, n_chunks, chunk_local) for b in range(slots)])
    k_dev = dev(k_bc.to(torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT, repl)
    v_dev = dev(v_bc.to(torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT, repl)

    def run(**kw):
        out = ttnn.transformer.sparse_sdpa_msa(
            q_dev,
            k_dev,
            v_dev,
            idx_dev,
            scale=d**-0.5,
            block_size=BLK_KV,
            cluster_axis=sp_axis,
            block_cyclic_sp_axis=sp_axis,
            block_cyclic_chunk_local=chunk_local,
            **kw,
        )
        return [ttnn.to_torch(t)[:, :H] for t in ttnn.get_device_tensors(out)]

    host = run(chunk_start_idx=chunk_start, cache_batch_idx=slot)
    for r in range(sp):
        gold = _ref_at_positions(q_ranks[r], k[slot : slot + 1], v[slot : slot + 1], idx_ranks[r], d**-0.5, owned[r])
        p = pcc(host[r], gold)
        assert p >= DEVICE_PCC, f"sp={sp} start={chunk_start}: rank {r} vs writer-derived golden pcc={p:.5f}"

    def u32(value):
        return dev(torch.tensor([[[[value]]]], dtype=torch.int64), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, repl)

    meta = run(
        chunk_start_idx_tensor=u32(chunk_start),
        cache_batch_idx_tensor=u32(slot),  # user 1, one layer -> slot 1
        index_cache_num_layers=1,
        index_cache_layer_idx=0,
    )
    for r in range(sp):
        assert torch.equal(meta[r], host[r]), f"sp={sp} start={chunk_start}: rank {r} metadata != host path"


# --- retargeting the metadata at sp>1: cache hits, in-place rewrites, trace replay -------------------------
#
# The single-device file covers this contract where device_index is always 0. Here each rank has its own
# rotated start, and reader AND writer recompose the K/V slot from the same user-id tensor, so a hit or a
# replay that kept one rank's geometry (or one kernel's slot) shows up as a diverged rank.
#
# A 4-slot user-major cache with a NON-TRIVIAL layer fold (slot = user * LAYERS + LAYER): with one layer at
# index 0 the fold is the identity, so a kernel that ignored it would still land on the right slot.
USERS, LAYERS, LAYER = 2, 2, 1
META_H, META_D, META_TOPK, META_CHUNKS = 32, 128, 16, 6
META_S = 2 * BLK_KV  # q seq-len per rank == chunk_local (tp=1), spanning >1 block
META_SCALE = META_D**-0.5


def _meta_starts(sp, chunk_local):
    """Slab-aligned, and rotated mid-block (boundary chip 1, so ownership rotates and that rank straddles)."""
    return (sp * chunk_local, sp * chunk_local + chunk_local + 96)


def _sp_or_skip(mesh_device):
    rows, cols = tuple(mesh_device.shape)
    if cols < 2:
        pytest.skip(f"needs sp>1 (mesh shape {(rows, cols)})")
    return 1, cols  # SP along cols: device r = SP rank r


def _mesh_tensor(mesh_device, x, dtype, layout, mapper, *, on_device=True):
    kwargs = {"dtype": dtype, "layout": layout, "mesh_mapper": mapper}
    if on_device:
        kwargs.update(device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.from_torch(x, **kwargs)


def _meta_u32(mesh_device, value, *, on_device=True):
    return _mesh_tensor(
        mesh_device,
        torch.tensor([[[[value]]]], dtype=torch.int64),
        ttnn.uint32,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.ReplicateTensorToMesh(mesh_device),
        on_device=on_device,
    )


def _meta_inputs(mesh_device, sp, seed, *, block_cyclic=True):
    """SP-sharded q and a replicated USERS*LAYERS-slot K/V cache (distinct slots, so a wrong-slot gather on
    either the reader or the writer changes the output), block-cyclic or in natural token order."""
    T = sp * META_CHUNKS * META_S
    gen = torch.Generator().manual_seed(seed)
    q_ranks = [torch.randn(1, META_H, META_S, META_D, generator=gen) for _ in range(sp)]
    k = torch.randn(USERS * LAYERS, 1, T, META_D, generator=gen)
    v = torch.randn(USERS * LAYERS, 1, T, META_D, generator=gen)

    def to_bc(t):  # the cache is block-cyclic within each slot
        if not block_cyclic:
            return t
        return torch.cat([_natural_to_block_cyclic(t[b : b + 1], sp, META_CHUNKS, META_S) for b in range(t.shape[0])])

    repl = ttnn.ReplicateTensorToMesh(mesh_device)
    q_dev = _mesh_tensor(
        mesh_device,
        torch.cat(q_ranks, dim=2),
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.ShardTensorToMesh(mesh_device, dim=2),
    )
    k_dev = _mesh_tensor(mesh_device, to_bc(k).to(torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT, repl)
    v_dev = _mesh_tensor(mesh_device, to_bc(v).to(torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT, repl)
    return q_ranks, k, v, q_dev, k_dev, v_dev


def _meta_owned(start, sp, *, block_cyclic):
    """Global query positions per rank. Block-cyclic: the writer's striping (rotation + straddle). Contiguous:
    one unbroken run per rank at start + rank*S, which the no-block-cyclic geometry branch produces."""
    if block_cyclic:
        return _owned_positions(start, sp, META_S)
    return [list(range(start + r * META_S, start + (r + 1) * META_S)) for r in range(sp)]


def _meta_indices(mesh_device, sp, start, gen, *, on_device=True, block_cyclic=True):
    """Per-rank causal block ids for the queries each rank actually owns, sharded over the mesh."""
    owned = _meta_owned(start, sp, block_cyclic=block_cyclic)
    idx_ranks = [_causal_indices_at(owned[r], META_TOPK, gen) for r in range(sp)]
    idx_dev = _mesh_tensor(
        mesh_device,
        torch.cat(idx_ranks, dim=2),
        ttnn.uint32,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.ShardTensorToMesh(mesh_device, dim=2),
        on_device=on_device,
    )
    return owned, idx_ranks, idx_dev


def _meta_dispatch(q_dev, k_dev, v_dev, idx_dev, sp_axis, *, block_cyclic=True, flat=False, **kw):
    """`flat=True` drops cluster_axis, which is the predicate selecting the rotation-exact SP geometry -- so it
    routes the causal start through the flat both-axes branch (linear + within-block straddle) instead."""
    layout = dict(block_cyclic_sp_axis=sp_axis, block_cyclic_chunk_local=META_S) if block_cyclic else {}
    return ttnn.transformer.sparse_sdpa_msa(
        q_dev,
        k_dev,
        v_dev,
        idx_dev,
        scale=META_SCALE,
        block_size=BLK_KV,
        cluster_axis=None if flat else sp_axis,
        **layout,
        **kw,
    )


def _meta_shards(out):
    return [ttnn.to_torch(t)[:, :META_H] for t in ttnn.get_device_tensors(out)]


def _meta_kwargs(start_t, user_t):
    return dict(
        chunk_start_idx_tensor=start_t,
        cache_batch_idx_tensor=user_t,
        index_cache_num_layers=LAYERS,
        index_cache_layer_idx=LAYER,
    )


def _meta_references(mesh_device, sp, sp_axis, q_ranks, k, v, q_dev, k_dev, v_dev, targets, seed, *, block_cyclic=True):
    """Host-int output per (user, start), plus the per-start indices. Each rank is anchored to a golden built
    from the writer-derived query positions, so the references themselves cannot inherit the op's formula."""
    refs, indices = {}, {}
    for user, start in targets:
        if start not in indices:
            gen = torch.Generator().manual_seed(seed + start)
            indices[start] = _meta_indices(mesh_device, sp, start, gen, block_cyclic=block_cyclic)
        owned, idx_ranks, idx_dev = indices[start]
        slot = user * LAYERS + LAYER
        shards = _meta_shards(
            _meta_dispatch(
                q_dev,
                k_dev,
                v_dev,
                idx_dev,
                sp_axis,
                block_cyclic=block_cyclic,
                chunk_start_idx=start,
                cache_batch_idx=slot,
            )
        )
        for r in range(sp):
            gold = _ref_at_positions(
                q_ranks[r], k[slot : slot + 1], v[slot : slot + 1], idx_ranks[r], META_SCALE, owned[r]
            )
            p = pcc(shards[r], gold)
            assert p >= DEVICE_PCC, f"sp={sp} start={start} user={user}: rank {r} host vs golden pcc={p:.5f}"
        refs[(user, start)] = shards
    # Every (user, start) must produce a different output, otherwise "metadata == host path" below would hold
    # even if the metadata tensors were never read: a stale slot or start would look identical.
    flat = {key: torch.cat([shard.flatten() for shard in shards]) for key, shards in refs.items()}
    keys = list(refs)
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            assert not torch.equal(flat[a], flat[b]), (
                f"sp={sp}: targets {a} and {b} produced identical output, so the retarget assertions would be "
                "vacuous -- pick starts/users that actually change the output"
            )
    return refs, indices


def _assert_meta_same(shards, refs, key, sp):
    user, start = key
    for r in range(sp):
        assert torch.equal(
            shards[r], refs[key][r]
        ), f"sp={sp}: rank {r} metadata != host path for user={user} start={start}"


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [(1, 2), (1, 4)], indirect=True)
def test_msa_block_cyclic_sp_metadata_cache_hit(mesh_device):
    """2 users x 2 starts on ONE cached mesh program. Every dispatch passes FRESHLY allocated metadata tensors
    (earlier ones kept alive -> new addresses), so a hit that kept the build-time addresses would mask or
    gather for the wrong user or start; override_runtime_arguments has to repoint BOTH the reader and the
    writer on every coordinate while leaving that coordinate's device_index and rotation intact. Then the
    same pair is rewritten in place."""
    sp_axis, sp = _sp_or_skip(mesh_device)
    q_ranks, k, v, q_dev, k_dev, v_dev = _meta_inputs(mesh_device, sp, seed=61)
    targets = [(user, start) for user in range(USERS) for start in _meta_starts(sp, META_S)]
    refs, indices = _meta_references(mesh_device, sp, sp_axis, q_ranks, k, v, q_dev, k_dev, v_dev, targets, seed=61)

    live, entries = [], None
    for key in targets:
        user, start = key
        start_t, user_t = _meta_u32(mesh_device, start), _meta_u32(mesh_device, user)
        live += [start_t, user_t]
        out = _meta_dispatch(q_dev, k_dev, v_dev, indices[start][2], sp_axis, **_meta_kwargs(start_t, user_t))
        _assert_meta_same(_meta_shards(out), refs, key, sp)
        if entries is None:
            entries = mesh_device.num_program_cache_entries()
    assert mesh_device.num_program_cache_entries() == entries, "switching user / start tensors recompiled"
    assert len({t.buffer_address() for t in live}) == len(live), "metadata tensors were not distinct allocations"

    start_t, user_t = _meta_u32(mesh_device, 0), _meta_u32(mesh_device, 0)
    for key in targets[::-1]:
        user, start = key
        ttnn.copy_host_to_device_tensor(_meta_u32(mesh_device, start, on_device=False), start_t)
        ttnn.copy_host_to_device_tensor(_meta_u32(mesh_device, user, on_device=False), user_t)
        out = _meta_dispatch(q_dev, k_dev, v_dev, indices[start][2], sp_axis, **_meta_kwargs(start_t, user_t))
        _assert_meta_same(_meta_shards(out), refs, key, sp)
    assert mesh_device.num_program_cache_entries() == entries, "in-place rewrite recompiled"


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [(1, 2), (1, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 1 << 20}], indirect=True)
def test_msa_block_cyclic_sp_metadata_trace_retarget(mesh_device):
    """One captured MESH trace, replayed across users and starts by rewriting the same metadata (and indices)
    tensors in place. Host ints would freeze the capture-time slot and start; a replay that lost the
    per-coordinate geometry and fell back to rank 0's start would diverge on every rotated rank."""
    sp_axis, sp = _sp_or_skip(mesh_device)
    q_ranks, k, v, q_dev, k_dev, v_dev = _meta_inputs(mesh_device, sp, seed=67)
    targets = [(user, start) for user in range(USERS) for start in _meta_starts(sp, META_S)]
    refs, indices = _meta_references(mesh_device, sp, sp_axis, q_ranks, k, v, q_dev, k_dev, v_dev, targets, seed=67)

    # The buffers the capture bakes in, plus a host-side copy per start to rewrite the indices with.
    first_start = targets[0][1]
    _, _, idx_t = _meta_indices(mesh_device, sp, first_start, torch.Generator().manual_seed(67 + first_start))
    host_idx = {
        start: _mesh_tensor(
            mesh_device,
            torch.cat(idx_ranks, dim=2),
            ttnn.uint32,
            ttnn.ROW_MAJOR_LAYOUT,
            ttnn.ShardTensorToMesh(mesh_device, dim=2),
            on_device=False,
        )
        for start, (_, idx_ranks, _) in indices.items()
    }
    start_t, user_t = _meta_u32(mesh_device, first_start), _meta_u32(mesh_device, 0)

    _meta_dispatch(q_dev, k_dev, v_dev, idx_t, sp_axis, **_meta_kwargs(start_t, user_t))  # compile first
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    try:
        traced = _meta_dispatch(q_dev, k_dev, v_dev, idx_t, sp_axis, **_meta_kwargs(start_t, user_t))
    finally:
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    try:
        for key in targets + targets[::-1]:
            user, start = key
            ttnn.copy_host_to_device_tensor(host_idx[start], idx_t)
            ttnn.copy_host_to_device_tensor(_meta_u32(mesh_device, start, on_device=False), start_t)
            ttnn.copy_host_to_device_tensor(_meta_u32(mesh_device, user, on_device=False), user_t)
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
            _assert_meta_same(_meta_shards(traced), refs, key, sp)
    finally:
        ttnn.release_trace(mesh_device, trace_id)


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [(1, 2), (1, 4)], indirect=True)
def test_msa_block_cyclic_sp_metadata_single_tensors(mesh_device):
    """Each tensor alone on a rotated start: the start tensor with a host cache_batch_idx (geometry derived
    on device, slot from the host arg), and the slot tensor with a host chunk_start_idx (geometry from the
    host's per-coordinate patch, slot recomposed on device by both the reader and the writer)."""
    sp_axis, sp = _sp_or_skip(mesh_device)
    q_ranks, k, v, q_dev, k_dev, v_dev = _meta_inputs(mesh_device, sp, seed=71)
    user, start = 1, _meta_starts(sp, META_S)[1]
    key = (user, start)
    refs, indices = _meta_references(mesh_device, sp, sp_axis, q_ranks, k, v, q_dev, k_dev, v_dev, [key], seed=71)
    idx_dev = indices[start][2]

    out = _meta_dispatch(
        q_dev,
        k_dev,
        v_dev,
        idx_dev,
        sp_axis,
        chunk_start_idx_tensor=_meta_u32(mesh_device, start),
        cache_batch_idx=user * LAYERS + LAYER,
    )
    _assert_meta_same(_meta_shards(out), refs, key, sp)

    out = _meta_dispatch(
        q_dev,
        k_dev,
        v_dev,
        idx_dev,
        sp_axis,
        chunk_start_idx=start,
        cache_batch_idx_tensor=_meta_u32(mesh_device, user),
        index_cache_num_layers=LAYERS,
        index_cache_layer_idx=LAYER,
    )
    _assert_meta_same(_meta_shards(out), refs, key, sp)


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [(1, 2), (1, 4)], indirect=True)
def test_msa_block_cyclic_sp_metadata_contiguous(mesh_device):
    """The CONTIGUOUS (no block_cyclic_*) SP path at sp>1: K/V stay in natural token order and each rank owns
    one unbroken run at start + rank*S, so the causal start takes the no-block-cyclic branch (linear, both
    straddle fields zero). Host-int vs golden on every rank, then metadata bit-exact with it."""
    sp_axis, sp = _sp_or_skip(mesh_device)
    q_ranks, k, v, q_dev, k_dev, v_dev = _meta_inputs(mesh_device, sp, seed=79, block_cyclic=False)
    starts = (sp * META_S, sp * META_S + META_S)  # no slab structure: any block-aligned start is valid
    targets = [(user, start) for user in range(USERS) for start in starts]
    refs, indices = _meta_references(
        mesh_device, sp, sp_axis, q_ranks, k, v, q_dev, k_dev, v_dev, targets, seed=79, block_cyclic=False
    )
    for key in targets:
        user, start = key
        out = _meta_dispatch(
            q_dev,
            k_dev,
            v_dev,
            indices[start][2],
            sp_axis,
            block_cyclic=False,
            **_meta_kwargs(_meta_u32(mesh_device, start), _meta_u32(mesh_device, user)),
        )
        _assert_meta_same(_meta_shards(out), refs, key, sp)


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [(1, 2), (1, 4)], indirect=True)
def test_msa_block_cyclic_sp_flat_linearization(mesh_device):
    """`cluster_axis=None` on a block-cyclic cache: the causal start takes the FLAT both-axes branch (linear +
    within-block straddle) instead of the rotation-exact one. Two claims:

    1. On a SLAB-ALIGNED start the two branches must agree EXACTLY -- that is the PR's compatibility claim
       ("identical to before for slab-aligned starts, which covers every current caller"), and it is checked
       by running the same inputs with and without cluster_axis and comparing bit-for-bit.
    2. On a mid-slab start, where the flat branch's straddle fields go nonzero on every rank, the metadata
       path must still reproduce the host-int path bit-exactly. This is the only coverage of the kernel's
       meta_rotation_exact=0 compile-time branch at sp>1; the rest of this file always sets cluster_axis.
    """
    sp_axis, sp = _sp_or_skip(mesh_device)
    q_ranks, k, v, q_dev, k_dev, v_dev = _meta_inputs(mesh_device, sp, seed=83)
    user, slot = 1, 1 * LAYERS + LAYER
    gen = torch.Generator().manual_seed(83)

    # (1) slab-aligned: rotation-exact and flat must produce identical output.
    aligned = 2 * sp * META_S
    _, _, idx_dev = _meta_indices(mesh_device, sp, aligned, gen)
    exact = _meta_shards(
        _meta_dispatch(q_dev, k_dev, v_dev, idx_dev, sp_axis, chunk_start_idx=aligned, cache_batch_idx=slot)
    )
    flat = _meta_shards(
        _meta_dispatch(q_dev, k_dev, v_dev, idx_dev, sp_axis, flat=True, chunk_start_idx=aligned, cache_batch_idx=slot)
    )
    for r in range(sp):
        assert torch.equal(
            flat[r], exact[r]
        ), f"sp={sp} start={aligned}: cluster_axis=None diverged from the rotation-exact geometry on a slab-aligned start"

    # (2) mid-slab: the flat branch straddles on every rank (offset = start % S != 0), and the metadata path
    # must derive the same geometry on device as the host patched in.
    mid = 2 * sp * META_S + 96
    assert mid % META_S != 0, "start must be mid-block for the flat branch to straddle"
    _, _, idx_mid = _meta_indices(mesh_device, sp, mid, gen)
    host = _meta_shards(
        _meta_dispatch(q_dev, k_dev, v_dev, idx_mid, sp_axis, flat=True, chunk_start_idx=mid, cache_batch_idx=slot)
    )
    meta = _meta_shards(
        _meta_dispatch(
            q_dev,
            k_dev,
            v_dev,
            idx_mid,
            sp_axis,
            flat=True,
            **_meta_kwargs(_meta_u32(mesh_device, mid), _meta_u32(mesh_device, user)),
        )
    )
    for r in range(sp):
        assert torch.equal(meta[r], host[r]), f"sp={sp} start={mid}: flat-branch metadata != host path on rank {r}"
    # The mid-slab flat start must not coincide with the aligned run, or the checks above are vacuous.
    assert not torch.equal(host[0], exact[0]), "mid-slab flat output matched the aligned run; pick a different start"


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [(1, 2), (1, 4)], indirect=True)
def test_msa_block_cyclic_sp_rejects_wrong_cluster_axis(mesh_device, expect_error):
    """The rotation-exact geometry reads this device's SP rank off cluster_axis, so that axis must BE the axis
    the cache was striped over. Naming a different one (here axis 0, extent 1 on a 1xN mesh) would hand every
    device rank 0 and silently score every rank against rank 0's causal window."""
    sp_axis, sp = _sp_or_skip(mesh_device)
    q_ranks, _, _, q_dev, k_dev, v_dev = _meta_inputs(mesh_device, sp, seed=89)
    _, _, idx_dev = _meta_indices(mesh_device, sp, sp * META_S, torch.Generator().manual_seed(89))
    with expect_error(RuntimeError, "must be the SP axis the cache was striped over"):
        ttnn.transformer.sparse_sdpa_msa(
            q_dev,
            k_dev,
            v_dev,
            idx_dev,
            scale=META_SCALE,
            block_size=BLK_KV,
            cluster_axis=0,  # extent 1 on a (1, sp) mesh, but the cache was striped over axis 1
            block_cyclic_sp_axis=sp_axis,
            block_cyclic_chunk_local=META_S,
            chunk_start_idx=sp * META_S,
            cache_batch_idx=1 * LAYERS + LAYER,
        )


def _system_mesh_is_2d():
    """Whether a (2,2) submesh can be opened at all. The mesh_device fixture only compares the device COUNT,
    so a 4-chip box exposed as a 1x4 mesh passes that check and then FATALs inside open_mesh_device; decide
    from the system mesh shape instead and skip cleanly."""
    try:
        shape = ttnn._ttnn.multi_device.SystemMeshDescriptor().shape()
    except Exception:
        return False
    return shape.dims() >= 2 and shape[0] >= 2 and shape[1] >= 2


@run_for_blackhole()
@pytest.mark.skipif(not _system_mesh_is_2d(), reason="a (2,2) submesh requires a 2D system mesh")
@pytest.mark.parametrize("mesh_device", [(2, 2)], ids=["sp2xtp2"], indirect=True)
@pytest.mark.parametrize("warm", [False, True], ids=["miss", "hit"])
def test_msa_block_cyclic_tp_subshard_reject(mesh_device, warm, expect_error):
    """A TP sub-shard of the query chunk (block_cyclic_chunk_local == tp*Sq) needs a per-device row offset
    WITHIN the slab that this op does not take, so causal block-cyclic with cluster_axis rejects it -- the same
    combination indexer_score_msa rejects. Needs a 2D mesh: on a 1xN mesh tp is 1 and the two are equal.

    `hit`: a legal cluster_axis=None call builds the program first. cluster_axis is not hashed on the host-int
    path, so the rejected call reuses that program -- the check must run on the hit too, not only on a miss."""
    rows, cols = tuple(mesh_device.shape)
    sp_axis, sp = 0, rows
    tp = (rows * cols) // sp
    if sp < 2 or tp < 2:
        pytest.skip(f"needs sp>1 and tp>1 (mesh shape {(rows, cols)})")
    chunk_local = tp * META_S  # the sub-shard form the op must refuse
    n_chunks = 3
    T = sp * n_chunks * chunk_local
    gen = torch.Generator().manual_seed(97)
    repl = ttnn.ReplicateTensorToMesh(mesh_device)
    q_dev = _mesh_tensor(
        mesh_device,
        torch.randn(1, META_H, rows * cols * META_S, META_D, generator=gen),
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.ShardTensorToMesh(mesh_device, dim=2),
    )
    kv = torch.randn(1, 1, T, META_D, generator=gen).to(torch.bfloat16)
    k_dev = _mesh_tensor(mesh_device, kv, ttnn.bfloat16, ttnn.TILE_LAYOUT, repl)
    v_dev = _mesh_tensor(mesh_device, kv.clone(), ttnn.bfloat16, ttnn.TILE_LAYOUT, repl)
    idx = _causal_indices_at(list(range(chunk_local, chunk_local + META_S)), META_TOPK, gen)
    idx_dev = _mesh_tensor(mesh_device, idx, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, repl)

    def call(cluster_axis):
        return ttnn.transformer.sparse_sdpa_msa(
            q_dev,
            k_dev,
            v_dev,
            idx_dev,
            scale=META_SCALE,
            block_size=BLK_KV,
            cluster_axis=cluster_axis,
            block_cyclic_sp_axis=sp_axis,
            block_cyclic_chunk_local=chunk_local,
            chunk_start_idx=sp * chunk_local,
        )

    if warm:
        entries = mesh_device.num_program_cache_entries()
        ttnn.deallocate(call(None))  # the flat linearization accepts the sub-shard
        assert mesh_device.num_program_cache_entries() == entries + 1
    with expect_error(RuntimeError, "needs block_cyclic_chunk_local"):
        call(sp_axis)
