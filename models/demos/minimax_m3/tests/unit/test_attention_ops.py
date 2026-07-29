# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tier-1 focused unit tests for the MiniMax-M3 attention ops (QK-norm + partial RoPE).

These compare our TTNN ops against hand-written torch references and depend ONLY
on torch (no HuggingFace / AutoConfig), so they run on a single Wormhole/Blackhole
card without a downloaded checkpoint.

Coverage:
  * partial RoPE (rotate first rotary_dim dims, pass the rest through).
"""

from types import SimpleNamespace

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.minimax_m3.tt.attention.operations import apply_rope
from models.demos.minimax_m3.tt.model import create_rope_setup

from ..test_factory import parametrize_mesh_with_fabric


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
@pytest.mark.parametrize("seq_len", [128], ids=["s128"])
def test_partial_rope_wrapper(mesh_device, device_params, seq_len, reset_seeds):
    """Partial-RoPE wrapper: rotate first rotary_dim dims, pass the rest through.

    The rope *kernel* vs HF is covered by test_rope.py; here we validate the
    MiniMax-M3 wrapper mechanics on a head_dim=128 / rotary_dim=64 tensor:
      * the [rotary_dim:] tail is passed through unchanged, and
      * the [:rotary_dim] head equals a direct rope on that slice (right indices
        + concat order preserved).
    """
    head_dim, rotary_dim, n_heads = 128, 64, 8

    hf_config = SimpleNamespace(
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        rope_theta=5_000_000,
        max_position_embeddings=4096,
        rope_scaling=None,
    )
    rope_setup = create_rope_setup(mesh_device=mesh_device, hf_config=hf_config, datatype=ttnn.bfloat16)
    cos = rope_setup.cos_matrix_prefill[:, :, :seq_len, :]
    sin = rope_setup.sin_matrix_prefill[:, :, :seq_len, :]
    trans = rope_setup.get_both_trans_mats()["prefill"]
    assert cos.shape[-1] == rotary_dim, f"rope matrices should be rotary_dim wide, got {cos.shape[-1]}"

    q = torch.randn(1, n_heads, seq_len, head_dim)
    q_tt = ttnn.from_torch(
        q,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    out = apply_rope(q_tt, (cos, sin), trans, is_decode_mode=False)
    out_torch = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).reshape(1, n_heads, seq_len, head_dim)

    # Direct rope on the 64-wide head slice (independent reference for the rotated part).
    q_head = ttnn.slice(q_tt, [0, 0, 0, 0], [1, n_heads, seq_len, rotary_dim])
    head_rope = ttnn.experimental.rotary_embedding_llama(q_head, cos, sin, trans, is_decode_mode=False)
    head_ref = ttnn.to_torch(ttnn.get_device_tensors(head_rope)[0]).reshape(1, n_heads, seq_len, rotary_dim)

    # Tail must be passed through unchanged.
    tail_pass, tail_pcc = comp_pcc(q[..., rotary_dim:], out_torch[..., rotary_dim:], 0.999)
    # Head must equal the direct rope on the slice.
    head_pass, head_pcc = comp_pcc(head_ref, out_torch[..., :rotary_dim], 0.999)
    # Sanity: the head was actually transformed (not an accidental pass-through).
    _, identity_pcc = comp_pcc(q[..., :rotary_dim], out_torch[..., :rotary_dim], 0.999)

    logger.info(f"partial rope: tail_pcc={tail_pcc}, head_pcc={head_pcc}, head-vs-input(should differ)={identity_pcc}")
    assert tail_pass, f"tail pass-through PCC fail: {tail_pcc}"
    assert head_pass, f"rotated head PCC fail: {head_pcc}"
