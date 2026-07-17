# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
GPT-OSS GQA chunked-KV cache write + read-back PCC vs torch golden (round-trip + slot/layout).

Allocates a small ``GptOssKVCache`` (a couple of users x layers), writes random natural-order K/V
into each (user, layer) slot via the M3-style ``write_kv_chunk`` wrapper (which drives DeepSeek's
``update_padded_kv_cache``), reads the caches back, inverts the block-cyclic SP layout to natural
token order, and PCC-checks each slot against the torch tensor that was written.

This validates: the two persistent DRAM NdShard caches, the user-major slot packing
(slot = user_id * num_layers + layer_idx), the head-sharded (1 KV head / TP col) + SP-sharded
sequence layout, and the bf8 round-trip. Threshold PCC >= 0.99.

Single card (1x1 mesh): sp=1, tp=1 -> 1 KV head/chip, block-cyclic == identity (kv_actual=0, cache
sized to the chunk), so readback is in natural order. The same code also exercises SP block-cyclic +
TP head-sharding on a multi-row/col mesh (sp=rows, tp=cols, NKV=cols), if the parent runs it there.

Run:
    pytest models/demos/gpt_oss_d_p/tests/unit/test_kv_cache_vs_ref.py
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.tt.mla.utils import blockcyclic_positions
from models.demos.gpt_oss_d_p.tt.attention import allocate_kv_cache, write_kv_chunk

HEAD_DIM = 64


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("num_users, num_layers", [(2, 2)], ids=["u2xl2"])
@pytest.mark.parametrize("seq_len", [128], ids=["s128"])
def test_kv_cache_write_read_vs_ref(mesh_device, num_users, num_layers, seq_len, reset_seeds):
    """Write GQA K/V into each (user, layer) slot, read back, PCC vs natural-order torch."""
    rows, cols = tuple(mesh_device.shape)
    sp, tp = rows, cols
    sp_axis, tp_axis = 0, 1  # matches gpt_oss_d_p MeshConfig (tp_axis=1 -> sp_axis=0)
    nkv = tp  # 1 KV head per TP col (the cache's per-chip head slot is 1)

    # Single chunk sized to the whole cache => block-cyclic layout is the identity (kv_actual=0).
    assert seq_len % (32 * sp) == 0, f"seq_len {seq_len} must be a multiple of 32*sp ({32 * sp})"
    max_seq_len = seq_len

    torch.manual_seed(0)
    # Natural-order per-(user, layer) K/V, one head per TP col: [num_users, num_layers, nkv, seq, HEAD_DIM].
    sent_k = torch.randn(num_users, num_layers, nkv, seq_len, HEAD_DIM)
    sent_v = torch.randn(num_users, num_layers, nkv, seq_len, HEAD_DIM)

    kv_cache = allocate_kv_cache(
        mesh_device,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        sp_axis=sp_axis,
        num_users=num_users,
        head_dim=HEAD_DIM,
    )

    # Chunk input sharding: sequence on SP rows (dim 2), heads on TP cols (dim 1).
    in_dims = [None, None]
    in_dims[sp_axis] = 2
    in_dims[tp_axis] = 1

    def to_chunk(nat):  # nat: [nkv, seq, HEAD_DIM] -> device [1, nkv, seq, HEAD_DIM], SP+TP sharded
        chunk = nat.reshape(1, nkv, seq_len, HEAD_DIM)
        return ttnn.from_torch(
            chunk,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=in_dims),
        )

    for u in range(num_users):
        for l in range(num_layers):
            tt_k = to_chunk(sent_k[u, l])
            tt_v = to_chunk(sent_v[u, l])
            write_kv_chunk(kv_cache, tt_k, tt_v, slot_idx=u, layer_idx=l, kv_actual=0, sp_axis=sp_axis)
            tt_k.deallocate(True)
            tt_v.deallocate(True)
    ttnn.synchronize_device(mesh_device)

    # Read back: per chip the cache is [num_users*num_layers, 1, seq_local, HEAD_DIM]; KV head c on col
    # c, seq (block-cyclic) split across SP rows. Concat rows -> invert block-cyclic -> natural order.
    p = blockcyclic_positions(sp, seq_len, max_seq_len)  # global pos held by each block-cyclic shard row

    def gather(cache_tensor, slot, col):
        dts = ttnn.get_device_tensors(cache_tensor)
        dev = torch.cat([ttnn.to_torch(dts[r * cols + col])[slot, 0].float() for r in range(rows)], dim=0)
        nat = torch.empty_like(dev)
        nat[p] = dev
        return nat[:seq_len]  # [seq, HEAD_DIM]

    for u in range(num_users):
        for l in range(num_layers):
            slot = u * num_layers + l
            host_k = torch.stack([gather(kv_cache.k, slot, c) for c in range(nkv)], dim=0)  # [nkv, seq, HD]
            host_v = torch.stack([gather(kv_cache.v, slot, c) for c in range(nkv)], dim=0)
            ok_k, pcc_k = comp_pcc(sent_k[u, l], host_k, 0.99)
            ok_v, pcc_v = comp_pcc(sent_v[u, l], host_v, 0.99)
            logger.info(f"(user={u}, layer={l}) slot={slot}: K pcc={pcc_k} V pcc={pcc_v}")
            assert ok_k, f"K cache mismatch (user={u}, layer={l}): {pcc_k}"
            assert ok_v, f"V cache mismatch (user={u}, layer={l}): {pcc_v}"

    logger.info(f"GQA chunked-KV cache round-trip ({num_users} users x {num_layers} layers): all slots PCC>=0.99")
