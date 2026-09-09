# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""SP-sharded ring SDPA with LIVE Q/K/V (no cache) vs an unsharded torch reference.

Structure follows `minimax_m3/tests/unit/test_ring_joint_sp_vs_ref.py`.

The claim under test: gathering K/V across the SP axis by ONLINE SOFTMAX over the ring gives the
same answer as ordinary unsharded causal attention. Each device holds `seq/sp` query rows and
`seq/sp` K/V rows; the op reconstructs the full sequence internally, so there is no explicit
AllGather and no `repeat_kv` head inflation — the kernel is GQA-causal and consumes grouped V at
`n_kv` heads.

At this model's TP=4 each chip carries **2** KV heads and 8 Q heads, which is a shape the donor's
version of this test never ran: it had 1 KV head per chip.
"""

import torch

import ttnn
from models.demos.llama_3_1_8b_d_p.reference.model import REF_DTYPE, eager_attention_forward
from models.demos.llama_3_1_8b_d_p.tt.attention.config import LlamaAttentionProgramConfig
from models.demos.llama_3_1_8b_d_p.tt.attention.dense_sp import dense_sp_attention_nocache

from ..test_factory import ACT_DTYPE, assert_pcc, parametrize_target_mesh

SEQ = 2048  # 256 tokens per SP row


def _golden(config, seq_len, seed=0):
    """Unsharded causal GQA attention over the whole sequence, fp16."""
    n_q, n_kv, hd = config.num_attention_heads, config.num_key_value_heads, config.head_dim
    torch.manual_seed(seed)
    q = torch.randn(1, n_q, seq_len, hd, dtype=REF_DTYPE)
    k = torch.randn(1, n_kv, seq_len, hd, dtype=REF_DTYPE)
    v = torch.randn(1, n_kv, seq_len, hd, dtype=REF_DTYPE)
    mask = torch.triu(
        torch.full((seq_len, seq_len), torch.finfo(REF_DTYPE).min, dtype=REF_DTYPE), diagonal=1
    )[None, None]
    out, _ = eager_attention_forward(q, k, v, mask, config.attn_scale, config.num_key_value_groups)
    return q, k, v, out.transpose(1, 2)  # -> [1, n_q, seq, hd]


@parametrize_target_mesh()
def test_ring_joint_sp_vs_ref(mesh_device, device_params, config, mesh_config, ccl_manager, topology_name):
    """Ring SDPA over live SP-sharded Q/K/V vs unsharded torch attention."""
    sp, tp = mesh_config.sp, mesh_config.tp
    assert SEQ % sp == 0
    q, k, v, golden = _golden(config, SEQ)

    # Q heads shard on TP cols, sequence on SP rows. Same for K/V (2 kv heads per chip).
    dims = [None, None]
    dims[mesh_config.sp_axis] = 2
    dims[mesh_config.tp_axis] = 1

    def to_dev(t):
        return ttnn.from_torch(
            t,
            device=mesh_device,
            dtype=ACT_DTYPE,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=dims),
        )

    program_config = LlamaAttentionProgramConfig()
    out = dense_sp_attention_nocache(
        to_dev(q),
        to_dev(k),
        to_dev(v),
        mesh_config=mesh_config,
        ccl_manager=ccl_manager,
        logical_n=SEQ,
        n_kv=config.num_key_value_heads,  # GLOBAL count; the buffer shards it across TP
        head_dim=config.head_dim,
        scale=config.attn_scale,
        program_config=program_config.get_ring_sdpa_config(),
        compute_kernel_config=program_config.get_ring_compute_kernel_config(),
    )
    ttnn.synchronize_device(mesh_device)

    got = ttnn.to_torch(
        out,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 1), mesh_shape=tuple(mesh_device.shape)),
    ).to(REF_DTYPE)
    assert_pcc("ring_joint_sp", golden, got, topology_name)
