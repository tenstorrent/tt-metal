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
