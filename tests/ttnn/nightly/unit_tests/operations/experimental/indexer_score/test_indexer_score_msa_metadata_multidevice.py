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
