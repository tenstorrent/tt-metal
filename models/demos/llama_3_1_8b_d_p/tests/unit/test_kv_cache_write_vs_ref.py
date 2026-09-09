# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Cache contents after a write through the PRODUCTION prefill seam, PCC'd against the reference K/V.

Structure follows `minimax_m3/tests/unit/test_kv_cache_write_vs_ref.py`.

The sibling `test_kv_cache_gqa_sp_vs_ref.py` writes synthetic K/V to exercise the substrate at this
model's shape. This one instead takes the K/V the torch reference actually produces — post-RoPE K
and raw V from `RefAttention` — and pushes them through `write_kv_chunk`, the same seam the decoder
layer calls. What it proves is that the write LANDS AT THE RIGHT SLOT AND OFFSET for real per-layer
data, across more than one layer and more than one slot.

Slot packing is user-major: `slot = user_id * num_layers + layer_idx`, layers contiguous. Getting
that arithmetic wrong is the failure that makes every layer read layer 0's cache — correct at layer
0 by coincidence, stale everywhere after it.
"""

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.mla.utils import rotated_chip_positions
from models.demos.llama_3_1_8b_d_p.reference.model import REF_DTYPE, RefAttention, RefRotaryEmbedding, causal_mask
from models.demos.llama_3_1_8b_d_p.tt.attention.kv_cache import allocate_kv_caches, write_kv_chunk

from ..test_factory import KV_DTYPE, assert_pcc, parametrize_target_mesh

CHUNK_LOCAL = 64
NUM_LAYERS = 4


@parametrize_target_mesh()
@pytest.mark.parametrize("layer_idx", [0, 1, NUM_LAYERS - 1], ids=["layer0", "layer1", "layer_last"])
def test_kv_cache_write_vs_ref(mesh_device, device_params, config, mesh_config, topology_name, layer_idx):
    """Reference K/V written through the seam, read back, PCC'd — at several layer slots."""
    sp, tp = mesh_config.sp, mesh_config.tp
    sp_axis, tp_axis = mesh_config.sp_axis, mesh_config.tp_axis
    n_kv, hd = config.num_key_value_heads, config.head_dim
    chunk_global = sp * CHUNK_LOCAL

    # Real reference K/V: post-RoPE K and raw V, exactly what the cache stores.
    torch.manual_seed(layer_idx)
    x = torch.randn(1, chunk_global, config.hidden_size, dtype=REF_DTYPE)
    attn = RefAttention(config, layer_idx)
    rope = RefRotaryEmbedding(config)
    cos, sin = rope(torch.arange(chunk_global)[None, :], dtype=REF_DTYPE)
    with torch.no_grad():
        _, k_rope, v = attn(x, (cos, sin), causal_mask(chunk_global), return_kv=True)

    kv = allocate_kv_caches(
        mesh_device,
        num_layers=NUM_LAYERS,
        max_seq_len=chunk_global,
        chunk_size=chunk_global,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        num_kv_heads=n_kv,
        head_dim=hd,
        cache_dtype=KV_DTYPE,
    )

    dims = [None, None]
    dims[sp_axis] = 2
    dims[tp_axis] = 1
    positions = rotated_chip_positions(0, sp, CHUNK_LOCAL)
    idx = torch.tensor([positions[c][r] for c in range(sp) for r in range(CHUNK_LOCAL)], dtype=torch.long)

    def to_dev(t):
        return ttnn.from_torch(
            t[:, :, idx, :].reshape(1, n_kv, chunk_global, hd),
            device=mesh_device,
            dtype=KV_DTYPE,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=dims),
        )

    write_kv_chunk(kv, to_dev(k_rope), to_dev(v), slot_idx=0, layer_idx=layer_idx, kv_actual=0, sp_axis=sp_axis)
    ttnn.synchronize_device(mesh_device)

    concat = [None, None]
    concat[sp_axis] = 2
    concat[tp_axis] = 1

    def readback(cache):
        full = ttnn.to_torch(
            cache, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=tuple(concat), mesh_shape=tuple(mesh_device.shape))
        ).to(REF_DTYPE)
        # Slot dim is user-major: slot 0, layer `layer_idx` -> batch row `0 * NUM_LAYERS + layer_idx`.
        return full[0 * NUM_LAYERS + layer_idx]

    tokens_per_dev = chunk_global // sp
    p = torch.arange(chunk_global)
    chip = (p % chunk_global) // CHUNK_LOCAL
    local_row = (p // chunk_global) * CHUNK_LOCAL + (p % CHUNK_LOCAL)
    dim2_idx = chip * tokens_per_dev + local_row

    assert_pcc(f"kv_write.k[layer{layer_idx}]", k_rope[0], readback(kv.k)[:, dim2_idx, :], topology_name)
    assert_pcc(f"kv_write.v[layer{layer_idx}]", v[0], readback(kv.v)[:, dim2_idx, :], topology_name)


@parametrize_target_mesh()
def test_layer_slots_do_not_alias(mesh_device, device_params, config, mesh_config):
    """Writing layer N must leave the other layers' slots untouched.

    The user-major packing puts a user's layers contiguously in the batch dim; an off-by-one in
    `slot * num_layers + layer` overwrites a neighbouring layer instead of raising.
    """
    sp = mesh_config.sp
    n_kv, hd = config.num_key_value_heads, config.head_dim
    chunk_global = sp * CHUNK_LOCAL
    kv = allocate_kv_caches(
        mesh_device,
        num_layers=NUM_LAYERS,
        max_seq_len=chunk_global,
        chunk_size=chunk_global,
        sp_axis=mesh_config.sp_axis,
        tp_axis=mesh_config.tp_axis,
        num_kv_heads=n_kv,
        head_dim=hd,
        cache_dtype=KV_DTYPE,
    )
    dims = [None, None]
    dims[mesh_config.sp_axis] = 2
    dims[mesh_config.tp_axis] = 1
    ones = ttnn.from_torch(
        torch.ones(1, n_kv, chunk_global, hd),
        device=mesh_device,
        dtype=KV_DTYPE,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=dims),
    )
    write_kv_chunk(kv, ones, ones, slot_idx=0, layer_idx=2, kv_actual=0, sp_axis=mesh_config.sp_axis)
    ttnn.synchronize_device(mesh_device)

    per_chip = ttnn.to_torch(ttnn.get_device_tensors(kv.k)[0]).float()
    for layer in range(NUM_LAYERS):
        written = per_chip[layer].abs().sum().item()
        if layer == 2:
            assert written > 0, "layer 2 was written but its slot is empty"
        else:
            assert written == 0, f"writing layer 2 also touched layer {layer}'s slot"
