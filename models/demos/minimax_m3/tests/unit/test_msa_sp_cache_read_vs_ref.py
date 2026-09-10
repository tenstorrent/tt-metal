# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MSA chunked CACHE-READ at TP=4 × SP=8 — current chunk attends the BLOCK-CYCLIC cached prefix.

The deployed multi-chunk read path is ``msa_sp_attention_cache_read``: the accumulated K/V/index_k live in
the packed ND-sharded SP cache in DeepSeek block-cyclic order (chip r holds [chunk0_r, chunk1_r, ...]) and
``high_bw_all_gather`` gathers the (user, layer) slot straight out of the cache into a persistent worst-case
buffer; the indexer + sparse_sdpa_msa decode the block-cyclic layout IN-KERNEL (block_cyclic_* args,
#49490/#48772) and are bounded to the written prefix (kv_len).

The reference here is the path this one replaced (#55668): gather the slot's block-cyclic shards with
``all_gather_async`` on an exact-size copy, typecast to bf16 and feed the same in-kernel remap. Identical
inputs, so the check is exact-PCC.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.minimax_m3.config import MeshConfig
from models.demos.minimax_m3.tt.attention.msa import msa_indexer_sparse
from models.demos.minimax_m3.tt.ccl import CCLManager
from models.demos.minimax_m3.utils.general_utils import get_default_num_links

from ..test_factory import parametrize_mesh_with_fabric

NQ, NKV, NIDX, HEAD_DIM = 64, 4, 4, 128


def reference_msa_block_cyclic(
    q,
    k_acc,
    v_acc,
    index_q,
    index_k_acc,
    *,
    mesh_config,
    ccl_manager,
    cached_len,
    chunk_local,
    scale,
    block_size,
    topk_blocks,
    num_groups=1,
):
    """Reference cross-chunk MSA over exact-size BLOCK-CYCLIC SP shards of the accumulated context
    ([1, n, n_chunks*chunk_local, hd] per device): all_gather_async across SP, typecast to bf16, then the
    same in-kernel block-cyclic remap the deployed path uses. This is the pre-#55668 production read path,
    kept as the golden for the high_bw_all_gather cache read."""
    sp_axis = mesh_config.sp_axis

    def gather(t):
        t = ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG)
        full_bc = mesh_config.allgather(t, ccl_manager, axis=sp_axis, dim=2)
        if full_bc.dtype != ttnn.bfloat16:
            full_bc = ttnn.typecast(full_bc, ttnn.bfloat16)
        return full_bc

    return msa_indexer_sparse(
        index_q,
        gather(index_k_acc),
        q,
        gather(k_acc),
        gather(v_acc),
        chunk_start_idx=cached_len,
        scale=scale,
        num_groups=num_groups,
        block_size=block_size,
        topk_blocks=topk_blocks,
        device=ccl_manager.mesh_device,
        cluster_axis=sp_axis,
        block_cyclic_sp_axis=sp_axis,
        block_cyclic_chunk_local=chunk_local,
    )


@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)], linear_fabric=True)
@pytest.mark.parametrize(
    "chunk_local,n_prior,capacity_chunks,slot",
    [(640, 1, 2, 0), (640, 1, 4, 3), (640, 3, 6, 1)],
    ids=["prior1_cap2_slot0", "prior1_cap4_slot3", "prior3_cap6_slot1"],
)
@pytest.mark.parametrize("cache_dtype", [ttnn.bfloat16, ttnn.bfloat8_b], ids=["bf16cache", "bf8cache"])
def test_msa_sp_cache_read_high_bw_pcc(
    mesh_device, device_params, chunk_local, n_prior, capacity_chunks, slot, cache_dtype, reset_seeds
):
    """``msa_sp_attention_cache_read`` (high_bw_all_gather straight from the cache slot) vs the reference
    block-cyclic read, exact-PCC, over the SAME real multi-slot ND-sharded cache.

    Reference input: bf16 cache -> the contiguous SP shard of the block-cyclic-permuted context (independent
    of the cache write); bf8 cache -> the slot read back out of the cache (whole-tensor NdShard->interleaved,
    then slice: slicing the NdShard tensor directly scrambles the round-robin banks), so both paths see the
    identical device-quantized bf8 values.

    The layouts pin what is new in the deployed path: capacity > written prefix (fixed-slot stride is
    seq_local, not n_rows), a non-zero slot in a multi-slot cache (in-op slot select), and a 4-chunk prefix.
    ``bf8cache`` feeds the deployed bf8 cache to the consumers natively (the reference typecasts to bf16).
    """
    from models.common.utility_functions import comp_pcc
    from models.demos.minimax_m3.tt.attention.kv_cache import allocate_kv_caches, write_index_k_chunk, write_kv_chunk
    from models.demos.minimax_m3.tt.attention.msa import msa_sp_attention_cache_read

    rows, cols = mesh_device.shape
    assert (rows, cols) == (8, 4)
    sp, tp, sp_axis = rows, cols, 0
    chunk = sp * chunk_local
    cached_len = n_prior * chunk
    n_chunks = n_prior + 1
    T = n_chunks * chunk  # written prefix
    capacity = capacity_chunks * chunk  # cache capacity (>= T)
    assert capacity >= T
    num_layers = slot + 1  # enough slots to place the tested one last
    G = NQ // NKV
    scale = HEAD_DIM**-0.5

    torch.manual_seed(0)
    q = torch.randn(1, NQ, chunk, HEAD_DIM, dtype=torch.bfloat16) * 0.1
    iq = torch.randn(1, NIDX, chunk, HEAD_DIM, dtype=torch.bfloat16) * 0.1
    k = torch.randn(1, NKV, T, HEAD_DIM, dtype=torch.bfloat16) * 0.1
    v = torch.randn(1, NKV, T, HEAD_DIM, dtype=torch.bfloat16) * 0.1
    ik = torch.randn(1, 1, T, HEAD_DIM, dtype=torch.bfloat16) * 0.1

    mesh_config = MeshConfig((rows, cols), tp=tp)
    ccl = CCLManager(mesh_device, num_links=get_default_num_links(mesh_device), topology=ttnn.Topology.Linear)

    def shard(t, split_heads, dtype=ttnn.bfloat16):
        dims = [None, None]
        dims[sp_axis] = 2
        dims[1] = 1 if split_heads else None
        return ttnn.from_torch(
            t,
            device=mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=dims),
        )

    # natural token idx -> block-cyclic slot (chip r holds [chunk0_r, chunk1_r, ...]).
    bc_idx = torch.tensor(
        [
            chunk_c * chunk + chip * chunk_local + c
            for chip in range(sp)
            for chunk_c in range(n_chunks)
            for c in range(chunk_local)
        ],
        dtype=torch.long,
    )

    def shard_bc(t, split_heads):
        return shard(t[:, :, bc_idx, :], split_heads)

    def collect(out_t):
        dts = ttnn.get_device_tensors(out_t)
        groups = [
            torch.cat([ttnn.to_torch(dts[r * cols + c]).float()[:, :G] for r in range(rows)], dim=2)
            for c in range(cols)
        ]
        return torch.cat(groups, dim=1)  # [1, NQ, chunk, HD]

    common = dict(mesh_config=mesh_config, ccl_manager=ccl, cached_len=cached_len, scale=scale, num_groups=1)
    common.update(block_size=128, topk_blocks=16)

    # The real multi-slot ND-sharded cache at capacity > prefix, tested slot last.
    kv = allocate_kv_caches(
        mesh_device,
        num_layers=num_layers,
        max_seq_len=capacity,
        sp_axis=sp_axis,
        head_dim=HEAD_DIM,
        cache_dtype=cache_dtype,
    )
    # Poison the OTHER slots so a wrong slot select / stride shows up as a PCC failure, not a lucky zero.
    for other in range(num_layers - 1):
        for c in range(n_chunks):
            sl = slice(c * chunk, (c + 1) * chunk)
            noise_k = torch.randn_like(k[:, :, sl, :])
            noise_v = torch.randn_like(v[:, :, sl, :])
            noise_ik = torch.randn_like(ik[:, :, sl, :])
            write_kv_chunk(
                kv,
                shard(noise_k, True),
                shard(noise_v, True),
                slot_idx=0,
                layer_idx=other,
                kv_actual=c * chunk,
                sp_axis=sp_axis,
            )
            write_index_k_chunk(
                kv, shard(noise_ik, False), slot_idx=0, layer_idx=other, kv_actual=c * chunk, sp_axis=sp_axis
            )
    for c in range(n_chunks):
        sl = slice(c * chunk, (c + 1) * chunk)
        write_kv_chunk(
            kv,
            shard(k[:, :, sl, :], True),
            shard(v[:, :, sl, :], True),
            slot_idx=0,
            layer_idx=slot,
            kv_actual=c * chunk,
            sp_axis=sp_axis,
        )
        write_index_k_chunk(
            kv, shard(ik[:, :, sl, :], False), slot_idx=0, layer_idx=slot, kv_actual=c * chunk, sp_axis=sp_axis
        )

    if cache_dtype == ttnn.bfloat16:
        # REFERENCE (bf16): contiguous block-cyclic shard, independent of the cache.
        ref_in = (shard_bc(k, True), shard_bc(v, True), shard_bc(ik, False))
    else:
        # REFERENCE (bf8): the slot read back out of the SAME cache — whole-tensor de-shard first, then slice.
        n_rows = n_chunks * chunk_local
        ints = [ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG) for t in (kv.k, kv.v, kv.index_k)]
        ref_in = tuple(ttnn.slice(t, (slot, 0, 0, 0), (slot + 1, 1, n_rows, HEAD_DIM)) for t in ints)
    out_ref = collect(
        reference_msa_block_cyclic(
            shard(q, True), ref_in[0], ref_in[1], shard(iq, True), ref_in[2], chunk_local=chunk_local, **common
        )
    )

    # DEPLOYED: gathered in-op straight from the cache slot.
    out_hbw = collect(
        msa_sp_attention_cache_read(
            shard(q, True),
            shard(iq, True),
            kv,
            slot=slot,
            chunk_local=chunk_local,
            **common,
        )
    )

    passing, pcc_msg = comp_pcc(out_ref, out_hbw, 0.99)
    logger.info(
        f"[cache-read-high-bw-pcc] {cache_dtype} capacity={capacity} prefix={T} slot={slot}/{num_layers}: {pcc_msg}"
    )
    assert passing, f"high_bw_all_gather cache-read diverges from the reference block-cyclic read ({pcc_msg})"
