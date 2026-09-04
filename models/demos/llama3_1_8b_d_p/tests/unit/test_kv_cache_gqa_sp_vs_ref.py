# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Write AND read-back for this model's own cache shape on the chunked-KV substrate, at SP=8 x TP=4.

``test_kv_cache_write_vs_ref`` writes a single chunk sized to the whole cache, which makes the
block-cyclic layout the identity. This test writes SEVERAL chunks into a cache larger than one
chunk, which is where the block-cyclic ordering is real: SP row ``r``'s contiguous local cache rows
hold block ``r`` of chunk 0, then block ``r`` of chunk 1, and so on. Reading the cache back and
un-permuting must recover the original sequence in natural order.

Getting this wrong does not raise — the ring SDPA still runs and reads the wrong tokens. It is the
single highest-value host-side check on the layout, which is why it is separated from the
seam-level write test.

The cache is this package's own: ``[num_users*num_layers, 2, seq_local, 128]``, i.e. 2 KV heads per
chip, which is the shape neither donor has.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.tt.mla.utils import rotated_chip_positions
from models.demos.llama3_1_8b_d_p.tt.attention.kv_cache import allocate_kv_cache, write_kv_chunk

from ..test_factory import llama_config, make_mesh_config, parametrize_mesh_with_fabric
from .test_kv_cache_write_vs_ref import gather_kv_cache

PCC = 0.99


@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)])
@pytest.mark.parametrize("n_chunks, chunk_local", [(4, 64), (2, 256)], ids=["4x512", "2x2048"])
@pytest.mark.parametrize("num_layers", [1, 3], ids=["L1", "L3"])
def test_kv_cache_gqa_sp_vs_ref(mesh_device, device_params, n_chunks, chunk_local, num_layers, reset_seeds):
    """Round-trip several chunks per layer through the block-cyclic SP cache.

    ``num_layers=3`` also pins the slot packing ``slot = user*num_layers + layer``: each layer's
    writes must land in its own rows, so a layer-indexing bug shows up as another layer's data.
    """
    cfg = llama_config()
    n_kv, hd = cfg.num_key_value_heads, cfg.head_dim
    mesh_config = make_mesh_config(mesh_device)
    rows, cols = tuple(mesh_device.shape)
    sp, sp_axis, tp_axis = mesh_config.sp, mesh_config.sp_axis, mesh_config.tp_axis
    n_kv_local = n_kv // mesh_config.tp

    chunk_global = sp * chunk_local
    cache_global = n_chunks * chunk_global

    # A distinct sequence per layer so a cross-layer write is visible, not masked by equal data.
    per_layer_k = [torch.randn(1, n_kv, cache_global, hd) * 0.1 for _ in range(num_layers)]
    per_layer_v = [torch.randn(1, n_kv, cache_global, hd) * 0.1 for _ in range(num_layers)]

    kv_cache = allocate_kv_cache(
        mesh_device,
        num_layers=num_layers,
        max_seq_len=cache_global,
        sp_axis=sp_axis,
        num_users=1,
        head_dim=hd,
        num_kv_heads_local=n_kv_local,
    )

    dims = [None, None]
    dims[sp_axis], dims[tp_axis] = 2, 1

    def bc_index(kv_actual):
        pos = rotated_chip_positions(kv_actual, sp, chunk_local)
        return torch.tensor([pos[c][r] for c in range(sp) for r in range(chunk_local)], dtype=torch.long)

    def to_device_chunk(src, kv_actual):
        chunk = src[:, :, bc_index(kv_actual), :]
        return ttnn.from_torch(
            chunk,
            device=mesh_device,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=dims),
        )

    for layer_idx in range(num_layers):
        for c in range(n_chunks):
            kv_actual = c * chunk_global
            write_kv_chunk(
                kv_cache,
                to_device_chunk(per_layer_k[layer_idx], kv_actual),
                to_device_chunk(per_layer_v[layer_idx], kv_actual),
                slot_idx=0,
                layer_idx=layer_idx,
                kv_actual=kv_actual,
                sp_axis=sp_axis,
            )
    ttnn.synchronize_device(mesh_device)

    for layer_idx in range(num_layers):
        # slot = user*num_layers + layer, user 0 -> row layer_idx of the packed batch dim.
        host_k = gather_kv_cache(mesh_device, kv_cache.k, n_kv_local, slot_row=layer_idx, chunk_local=chunk_local)
        host_v = gather_kv_cache(mesh_device, kv_cache.v, n_kv_local, slot_row=layer_idx, chunk_local=chunk_local)
        ok_k, pcc_k = comp_pcc(per_layer_k[layer_idx], host_k, PCC)
        ok_v, pcc_v = comp_pcc(per_layer_v[layer_idx], host_v, PCC)
        logger.info(f"KV round-trip L{layer_idx} {n_chunks}x{chunk_global}: K={pcc_k} V={pcc_v}")
        assert ok_k, f"L{layer_idx} K round-trip mismatch: {pcc_k}"
        assert ok_v, f"L{layer_idx} V round-trip mismatch: {pcc_v}"
