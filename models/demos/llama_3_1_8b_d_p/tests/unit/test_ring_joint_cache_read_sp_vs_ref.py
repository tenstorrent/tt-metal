# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Ring SDPA reading K/V OUT OF the block-cyclic KV cache: short Q against a longer prefix.

Structure follows `minimax_m3/tests/unit/test_ring_joint_cache_read_sp_vs_ref.py`.

**This is the mechanism chunked prefill depends on.** In the donor's bring-up it was also the single
worst test in the whole stage — a borrowed program config that assumed the wrong q_chunk stalled it
at PCC 0.91 — so the chunk sizes here are re-derived rather than inherited: this spec's head_dim 128
and chunk_size 5120 at sp8 give chunk_local 640, which happens to match the envelope the donor's
`q_chunk=128 / k_chunk=512` was measured at. That coincidence is recorded in
`tt/attention/config.py`, not relied on silently.

Two heads per chip again: the cache is `[.., 2, seq_local, 128]` and the ring gather buffer must
agree with it.
"""

import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.mla.utils import rotated_chip_positions
from models.demos.llama_3_1_8b_d_p.reference.model import REF_DTYPE, eager_attention_forward
from models.demos.llama_3_1_8b_d_p.tt.attention.config import LlamaAttentionProgramConfig
from models.demos.llama_3_1_8b_d_p.tt.attention.dense_sp import dense_sp_attention
from models.demos.llama_3_1_8b_d_p.tt.attention.kv_cache import allocate_kv_caches

from ..test_factory import ACT_DTYPE, KV_DTYPE, assert_pcc, parametrize_target_mesh

CHUNK_LOCAL = 128  # per SP row; chunk_global = 8 * 128 = 1024
N_CHUNKS = 2


@parametrize_target_mesh()
def test_ring_joint_cache_read_sp_vs_ref(
    mesh_device, device_params, config, mesh_config, ccl_manager, topology_name
):
    """Fill the cache with chunk 0, then attend chunk 1's queries over the whole cached prefix."""
    sp, tp = mesh_config.sp, mesh_config.tp
    sp_axis, tp_axis = mesh_config.sp_axis, mesh_config.tp_axis
    n_q, n_kv, hd = config.num_attention_heads, config.num_key_value_heads, config.head_dim
    chunk_global = sp * CHUNK_LOCAL
    total = N_CHUNKS * chunk_global

    torch.manual_seed(0)
    q_all = torch.randn(1, n_q, total, hd, dtype=REF_DTYPE)
    k_all = torch.randn(1, n_kv, total, hd, dtype=REF_DTYPE)
    v_all = torch.randn(1, n_kv, total, hd, dtype=REF_DTYPE)

    # Golden: full causal attention over the whole sequence; we compare only the LAST chunk's rows,
    # which is what the cache-read path computes.
    mask = torch.triu(torch.full((total, total), torch.finfo(REF_DTYPE).min, dtype=REF_DTYPE), diagonal=1)[None, None]
    golden_all, _ = eager_attention_forward(q_all, k_all, v_all, mask, config.attn_scale, config.num_key_value_groups)
    golden_all = golden_all.transpose(1, 2)  # [1, n_q, total, hd]

    kv = allocate_kv_caches(
        mesh_device,
        num_layers=1,
        max_seq_len=total,
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

    def block_cyclic(t, kv_actual, heads):
        """Reorder a chunk's tokens into the block-cyclic chip-concat order the writer expects."""
        positions = rotated_chip_positions(kv_actual, sp, CHUNK_LOCAL)
        idx = torch.tensor([positions[c][r] for c in range(sp) for r in range(CHUNK_LOCAL)], dtype=torch.long)
        return t[:, :, idx, :].reshape(1, heads, chunk_global, hd)

    def to_dev(t, dtype):
        return ttnn.from_torch(
            t,
            device=mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=dims),
        )

    program_config = LlamaAttentionProgramConfig()

    out = None
    for c in range(N_CHUNKS):
        kv_actual = c * chunk_global
        logical_n = (c + 1) * chunk_global
        k_chunk = to_dev(block_cyclic(k_all, kv_actual, n_kv), KV_DTYPE)
        v_chunk = to_dev(block_cyclic(v_all, kv_actual, n_kv), KV_DTYPE)
        q_chunk = to_dev(block_cyclic(q_all, kv_actual, n_q), ACT_DTYPE)
        out = dense_sp_attention(
            q_chunk,
            kv.k,
            kv.v,
            k_chunk,
            v_chunk,
            kv_actual=kv_actual,
            logical_n=logical_n,
            n_kv=n_kv,
            cache_global=kv.capacity,
            head_dim=hd,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            program_config=program_config.get_ring_sdpa_config(),
            compute_kernel_config=program_config.get_ring_compute_kernel_config(),
            scale=config.attn_scale,
            cluster_axis=sp_axis,
            slot_idx=0,
            layer_idx=0,
            num_layers=1,
            write_chunk=True,
            cache_dtype=KV_DTYPE,
        )
    ttnn.synchronize_device(mesh_device)

    got = ttnn.to_torch(
        out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 1), mesh_shape=tuple(mesh_device.shape))
    ).to(REF_DTYPE)

    # Undo the block-cyclic ordering of the LAST chunk's rows to compare against natural order.
    last_actual = (N_CHUNKS - 1) * chunk_global
    positions = rotated_chip_positions(last_actual, sp, CHUNK_LOCAL)
    order = [positions[c][r] for c in range(sp) for r in range(CHUNK_LOCAL)]
    inverse = torch.empty(chunk_global, dtype=torch.long)
    for row, pos in enumerate(order):
        inverse[pos - last_actual] = row
    got_natural = got[:, :, inverse, :]
    golden_last = golden_all[:, :, last_actual : last_actual + chunk_global, :]

    assert_pcc("ring_joint_cache_read_sp", golden_last, got_natural, topology_name)
