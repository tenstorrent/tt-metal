# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Decoder suite rows 5 and 6: SP ring-joint SDPA, with live K/V and reading the KV cache.

Recipe rows ``test_ring_joint_sp_vs_ref.py`` ("SP-sharded ring SDPA vs torch ref with **live
Q/K/V** (no cache)") and ``test_ring_joint_cache_read_sp_vs_ref.py`` ("same op reading K/V **out
of the block-cyclic KV cache** — short Q against a longer accumulated prefix"). Same op, same
shapes, one prefix written or not — so they live in one file and the second reuses the first's
golden.

These are deliberately **op-level**: Q/K/V are already-projected randoms, not the output of a
projection, so a failure is the ring op's parameterisation (cluster axis, column-major CCL,
balance flag, semaphores, carved core grid) and nothing else. Row 5 calls the ttnn op directly;
row 6 goes through ``dense_sp_attention``, the package's own seam, because the cache read is
inseparable from the slot/offset arithmetic that wrapper owns.

Mistral shapes: 96 Q heads / 8 KV heads / head_dim 128, i.e. 24 Q and 2 KV heads per chip at
TP=4, sequence over the 8 SP rows. ``is_balanced=False`` because causal work per chip is
triangular. The cache is bfloat8_b, so row 6's PCC is measured against a bf16 torch golden with
the cache's own quantisation in the loop — that is the number the README reports, not a defect.
"""

import pytest
import torch

import ttnn
from models.demos.mistral_medium_3_5_128b.tests.device_utils import (
    MESH_SHAPE,
    assert_pcc,
    gather_heads_seq,
    shard_heads_seq,
    torch_gqa_causal,
)
from models.demos.mistral_medium_3_5_128b.tt.attention.dense_sp import dense_sp_attention
from models.demos.mistral_medium_3_5_128b.tt.attention.kv_cache import allocate_kv_cache, write_kv_chunk

NQ, NKV, HEAD_DIM = 96, 8, 128
SP_AXIS = 0
SCALE = HEAD_DIM**-0.5


def _qkv(seq, seed=0):
    torch.manual_seed(seed)
    q = (torch.randn(1, NQ, seq, HEAD_DIM) * 0.1).to(torch.bfloat16)
    k = (torch.randn(1, NKV, seq, HEAD_DIM) * 0.1).to(torch.bfloat16)
    v = (torch.randn(1, NKV, seq, HEAD_DIM) * 0.1).to(torch.bfloat16)
    return q, k, v


def _sdpa_configs(mesh):
    grid = mesh.compute_with_storage_grid_size()
    prog = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid.x - 1, grid.y),  # leave the CCL column free
        q_chunk_size=128,
        k_chunk_size=128,
        exp_approx_mode=False,
    )
    kcfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,  # the ring op does not support fp32 dest accumulation
        packer_l1_acc=False,
    )
    return prog, kcfg


@pytest.mark.parametrize("seq", [2048, 5120], ids=["s2048", "s5120"])
def test_ring_joint_sp_live_qkv_vs_ref(galaxy, ccl, seq):
    """The ring op over live SP-sharded Q/K/V, no cache, vs a torch GQA-causal golden."""
    q, k, v = _qkv(seq)
    ref = torch_gqa_causal(q, k, v, SCALE)

    tt_q, tt_k, tt_v = (shard_heads_seq(galaxy, t) for t in (q, k, v))

    # The ring gather lands the full sequence in these per-chip buffers: heads over TP, seq whole.
    def pbuf(n_heads):
        return ttnn.from_torch(
            torch.zeros(1, n_heads, seq, HEAD_DIM),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=galaxy,
            mesh_mapper=ttnn.ShardTensor2dMesh(galaxy, mesh_shape=MESH_SHAPE, dims=[None, 1]),
        )

    prog, kcfg = _sdpa_configs(galaxy)
    out, _, _ = ttnn.transformer.ring_joint_scaled_dot_product_attention(
        tt_q,
        tt_k,
        tt_v,
        None,
        None,
        None,
        persistent_output_buffer_k=pbuf(NKV),
        persistent_output_buffer_v=pbuf(NKV),
        joint_strategy="rear",
        logical_n=seq,
        program_config=prog,
        compute_kernel_config=kcfg,
        dim=2,
        multi_device_global_semaphore=ccl.ring_attention_ccl_semaphore_handles,
        num_links=ccl.num_links,
        cluster_axis=SP_AXIS,
        mesh_device=galaxy,
        topology=ccl.topology,
        ccl_core_grid_offset=ccl.ring_attention_ccl_core_grid_offset,
        use_column_major_ccl=True,
        is_causal=True,
        scale=SCALE,
        is_balanced=False,
    )

    assert_pcc(f"ring_joint_live[s{seq}]", ref, gather_heads_seq(galaxy, out))


@pytest.mark.parametrize("chunk_size", [2048], ids=["c2048"])
def test_ring_joint_cache_read_sp_vs_ref(galaxy, ccl, chunk_size):
    """Short Q against a longer accumulated prefix, K/V read out of the block-cyclic cache.

    Two chunks: the first is written and left alone, the second is written by
    ``dense_sp_attention`` itself and its queries run over the whole two-chunk prefix. The golden
    is the last chunk's slice of a full-sequence causal SDPA, which is exactly the property
    chunked prefill has to have.
    """
    n_chunks = 2
    total = n_chunks * chunk_size
    q, k, v = _qkv(total, seed=1)
    ref = torch_gqa_causal(q, k, v, SCALE)[:, :, chunk_size:, :]

    kv = allocate_kv_cache(
        galaxy,
        num_layers=1,
        max_seq_len=total,
        chunk_size=chunk_size,
        sp_axis=SP_AXIS,
        num_kv_heads=NKV,
        head_dim=HEAD_DIM,
    )
    # Chunk 0 goes in through the production write seam; chunk 1 is written by the attention call.
    write_kv_chunk(
        kv,
        shard_heads_seq(galaxy, k[:, :, :chunk_size], dtype=ttnn.bfloat8_b),
        shard_heads_seq(galaxy, v[:, :, :chunk_size], dtype=ttnn.bfloat8_b),
        slot_idx=0,
        layer_idx=0,
        kv_actual=0,
        sp_axis=SP_AXIS,
    )
    ttnn.synchronize_device(galaxy)

    prog, kcfg = _sdpa_configs(galaxy)
    out = dense_sp_attention(
        shard_heads_seq(galaxy, q[:, :, chunk_size:]),
        kv.k,
        kv.v,
        shard_heads_seq(galaxy, k[:, :, chunk_size:], dtype=ttnn.bfloat8_b),
        shard_heads_seq(galaxy, v[:, :, chunk_size:], dtype=ttnn.bfloat8_b),
        kv_actual=chunk_size,
        logical_n=total,
        n_kv=NKV,
        cache_global=kv.max_seq_len,
        head_dim=HEAD_DIM,
        mesh_device=galaxy,
        ccl_manager=ccl,
        program_config=prog,
        compute_kernel_config=kcfg,
        scale=SCALE,
        cluster_axis=SP_AXIS,
        slot_idx=0,
        layer_idx=0,
        num_layers=1,
    )

    assert_pcc(f"ring_joint_cache_read[c{chunk_size}]", ref, gather_heads_seq(galaxy, out))
