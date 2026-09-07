# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The SP-sharded ring SDPA against a torch reference with LIVE Q/K/V (no cache).
Pattern: ``minimax_m3/tests/unit/test_ring_joint_sp_vs_ref.py``.

The claim under test: gathering K/V across the SP axis by online softmax gives the same answer as
unsharded attention. The op is called directly on Mistral's GQA shapes — Q[1, 96, S, 128],
K/V[1, 8, S, 128] — sharded across the target mesh with heads on the TP cols and the sequence on the
SP rows, and PCC'd against a torch GQA-causal SDPA golden.

This de-risks the mechanism IN ISOLATION, before any cache is involved: it exercises the pieces of
``CCLManager`` the ring path needs (``ring_attention_ccl_semaphore_handles`` and
``ring_attention_ccl_core_grid_offset`` — CCL workers in the last compute column, SDPA on the carved
grid) with ``cluster_axis = sp_axis``. Q/K/V are already-projected randoms, so no rope, projection
or cache layout can be blamed for a failure here.
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.mistral_3_5_d_p.reference.mistral_config import MistralMedium35Config as C
from models.demos.mistral_3_5_d_p.spec import SPEC
from models.demos.mistral_3_5_d_p.tt.attention.config import ProgramConfig

from ..test_factory import build_mesh_and_ccl, parametrize_mesh, sp_tp_shard_mapper

NQ, NKV, HEAD_DIM = C.NUM_ATTENTION_HEADS, C.NUM_KEY_VALUE_HEADS, C.HEAD_DIM


def torch_gqa_causal(q, k, v):
    """fp32 GQA causal SDPA golden. q[1,NQ,S,HD], k/v[1,NKV,S,HD]."""
    rep = NQ // NKV
    k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
    s = q.shape[2]
    scores = (q @ k.transpose(-1, -2)) * (HEAD_DIM**-0.5)
    causal = torch.triu(torch.full((s, s), float("-inf")), diagonal=1)
    return torch.softmax(scores + causal, dim=-1) @ v


def gather_heads_and_seq(tt_out, mesh_device):
    """out per chip is [1, NQ/tp, S/sp, HD]: concat the TP cols on the head dim, the SP rows on seq."""
    rows, cols = tuple(mesh_device.shape)
    shards = ttnn.get_device_tensors(tt_out)
    per_row = [
        torch.cat([ttnn.to_torch(shards[r * cols + c]).float() for c in range(cols)], dim=1) for r in range(rows)
    ]
    return torch.cat(per_row, dim=2)


@parametrize_mesh()
@pytest.mark.parametrize("seq_len", [512, 5120], ids=["s512", "s5120"])
def test_ring_joint_sp_vs_ref(mesh_device, device_params, seq_len, reset_seeds):
    """Ring-joint GQA causal attention at SP=4 x TP=8 vs the torch golden.

    ``s5120`` is the spec's real ``chunk_size`` (1280 tokens per SP row), so the production geometry
    is covered and not just a small stand-in.
    """
    rows, cols = tuple(mesh_device.shape)
    sp = rows
    assert cols == SPEC.tp and rows == SPEC.sp, f"expected the spec's mesh {SPEC.mesh_shape}, got {(rows, cols)}"
    assert seq_len % (ttnn.TILE_SIZE * sp) == 0, f"seq_len {seq_len} must be a multiple of {ttnn.TILE_SIZE * sp}"

    q = torch.randn(1, NQ, seq_len, HEAD_DIM) * 0.1
    k = torch.randn(1, NKV, seq_len, HEAD_DIM) * 0.1
    v = torch.randn(1, NKV, seq_len, HEAD_DIM) * 0.1
    ref = torch_gqa_causal(q.float(), k.float(), v.float())

    mesh_config, ccl = build_mesh_and_ccl(mesh_device)
    program_config = ProgramConfig()
    mapper = sp_tp_shard_mapper(mesh_device, seq_dim=2, head_dim=1)

    def shard(t):
        return ttnn.from_torch(
            t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )

    tt_q, tt_k, tt_v = shard(q), shard(k), shard(v)

    # The persistent gather buffers hold the reconstructed FULL sequence per chip: heads on the TP
    # cols, sequence replicated across the SP rows. Live-QKV mode is bf16 (no bf8 cache involved),
    # so these are built here rather than taken from the CCL manager's bf8 cache buffers.
    def persistent_buffer(n_heads):
        return ttnn.from_torch(
            torch.zeros(1, n_heads, seq_len, HEAD_DIM),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=sp_tp_shard_mapper(mesh_device, head_dim=1),
        )

    out, _, _ = ttnn.transformer.ring_joint_scaled_dot_product_attention(
        tt_q,
        tt_k,
        tt_v,
        None,
        None,
        None,
        persistent_output_buffer_k=persistent_buffer(NKV),
        persistent_output_buffer_v=persistent_buffer(NKV),
        joint_strategy="rear",
        logical_n=seq_len,  # the full sequence, reconstructed across the ring
        program_config=program_config.get_ring_sdpa_config(mesh_device),
        compute_kernel_config=program_config.get_ring_compute_kernel_config(mesh_device),
        dim=2,
        multi_device_global_semaphore=ccl.ring_attention_ccl_semaphore_handles,
        num_links=ccl.num_links,
        cluster_axis=mesh_config.sp_axis,
        mesh_device=mesh_device,
        topology=ccl.topology,
        ccl_core_grid_offset=ccl.ring_attention_ccl_core_grid_offset,
        use_column_major_ccl=True,
        is_causal=True,
        scale=HEAD_DIM**-0.5,
        is_balanced=False,
    )

    full = gather_heads_and_seq(out, mesh_device)
    passing, pcc = comp_pcc(ref, full, SPEC.pcc)
    logger.info(f"ring_joint SP={rows} x TP={cols} seq={seq_len}: pcc={pcc} shape={tuple(full.shape)}")
    assert passing, f"ring_joint SP PCC fail (seq={seq_len}): {pcc}"
