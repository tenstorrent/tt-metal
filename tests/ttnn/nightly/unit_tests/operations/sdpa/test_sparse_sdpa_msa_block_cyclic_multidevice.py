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
