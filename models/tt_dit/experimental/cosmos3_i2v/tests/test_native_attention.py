# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC test for the native Cosmos3 joint-attention block, TP variants.

Builds the torch reference (`Cosmos3PackedMoTAttention` + `Cosmos3AttnProcessor`)
with random weights and runs the same weights through `Cosmos3JointAttention`
on submeshes of a BH Galaxy 4x8 parent mesh — (1,1), (1,2), (1,4), (1,8).
Compares both und (causal) and gen (full) pathway outputs via PCC.

Test config has 32 Q heads + 8 K/V heads (matches the 64:8 GQA ratio of the
real 64B trunk at smaller scale) so all four mesh shapes exercise GQA. SP is
not exercised here — that lands in a follow-up alongside ring_joint_sdpa.
"""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.check import assert_quality
from models.tt_dit.utils.tensor import bf16_tensor
from models.tt_dit.utils.test import line_params

_PARENT_MESH = (4, 8)


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [pytest.param(_PARENT_MESH, line_params, id="bh_galaxy_parent")],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    "submesh_shape",
    [
        pytest.param((1, 1), id="1x1"),
        pytest.param((1, 2), id="1x2"),
        pytest.param((1, 4), id="1x4"),
        pytest.param((1, 8), id="1x8"),
    ],
)
@pytest.mark.timeout(300)
def test_native_joint_attention(mesh_device: ttnn.MeshDevice, submesh_shape: tuple[int, int]) -> None:
    mesh_device = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    from models.tt_dit.experimental.cosmos3_i2v.model.attention import Cosmos3JointAttention
    from models.tt_dit.experimental.cosmos3_i2v.reference.transformer_cosmos3 import (
        Cosmos3PackedMoTAttention,
        Cosmos3VLTextRotaryEmbedding,
    )

    torch.manual_seed(42)

    mesh_shape = tuple(mesh_device.shape)
    tp_factor = mesh_shape[1]
    sp_factor = mesh_shape[0]

    hidden_size = 2048
    head_dim = 64
    num_attention_heads = 32  # divisible by 1/2/4/8
    num_key_value_heads = 8  # divisible by 1/2/4/8, 4:1 GQA against Q
    rms_norm_eps = 1e-6
    rope_theta = 5_000_000.0
    rope_axes_dim = (12, 10, 10)  # sums to head_dim/2 = 32
    N_und = 128
    N_gen = 128
    N_total = N_und + N_gen

    torch_attn = Cosmos3PackedMoTAttention(
        hidden_size=hidden_size,
        head_dim=head_dim,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        attention_bias=False,
        rms_norm_eps=rms_norm_eps,
    )
    torch_attn.eval()
    torch_attn.to(dtype=torch.bfloat16)

    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(1, 0),
        sequence_parallel=ParallelFactor(sp_factor, 0),
        tensor_parallel=ParallelFactor(tp_factor, 1),
    )
    ccl_manager = (
        CCLManager(mesh_device=mesh_device, num_links=1, topology=ttnn.Topology.Linear) if tp_factor > 1 else None
    )

    tt_attn = Cosmos3JointAttention(
        hidden_size=hidden_size,
        head_dim=head_dim,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        attention_bias=False,
        rms_norm_eps=rms_norm_eps,
        mesh_device=mesh_device,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
    )
    tt_attn.load_torch_state_dict(torch_attn.state_dict())

    # cos/sin for the JOINT sequence, then slice into und + gen halves —
    # matches `Cosmos3OmniTransformer.forward`'s rotary slicing.
    rope = Cosmos3VLTextRotaryEmbedding(
        head_dim=head_dim,
        rope_theta=rope_theta,
        rope_axes_dim=list(rope_axes_dim),
    )
    position_ids = torch.arange(N_total).unsqueeze(0)
    cos_all, sin_all = rope(position_ids, device=torch.device("cpu"), dtype=torch.bfloat16)
    cos_all = cos_all.squeeze(0)
    sin_all = sin_all.squeeze(0)
    cos_und_NE = cos_all[:N_und]
    sin_und_NE = sin_all[:N_und]
    cos_gen_NE = cos_all[N_und:]
    sin_gen_NE = sin_all[N_und:]

    und_seq = torch.randn(N_und, hidden_size, dtype=torch.bfloat16)
    gen_seq = torch.randn(N_gen, hidden_size, dtype=torch.bfloat16)

    with torch.no_grad():
        torch_und_out, torch_gen_out = torch_attn(
            und_seq,
            gen_seq,
            (cos_und_NE, sin_und_NE, cos_gen_NE, sin_gen_NE),
        )

    und_seq_tt = bf16_tensor(und_seq.reshape(1, 1, N_und, hidden_size), device=mesh_device)
    gen_seq_tt = bf16_tensor(gen_seq.reshape(1, 1, N_gen, hidden_size), device=mesh_device)
    cos_und_tt = bf16_tensor(cos_und_NE.reshape(1, 1, N_und, head_dim), device=mesh_device)
    sin_und_tt = bf16_tensor(sin_und_NE.reshape(1, 1, N_und, head_dim), device=mesh_device)
    cos_gen_tt = bf16_tensor(cos_gen_NE.reshape(1, 1, N_gen, head_dim), device=mesh_device)
    sin_gen_tt = bf16_tensor(sin_gen_NE.reshape(1, 1, N_gen, head_dim), device=mesh_device)

    tt_und_out, tt_gen_out = tt_attn(
        und_seq_tt,
        gen_seq_tt,
        cos_und_tt,
        sin_und_tt,
        cos_gen_tt,
        sin_gen_tt,
    )

    # Output is replicated across the mesh; first device tensor is enough.
    und_torch_view = ttnn.to_torch(ttnn.get_device_tensors(tt_und_out)[0]).reshape(N_und, hidden_size)
    gen_torch_view = ttnn.to_torch(ttnn.get_device_tensors(tt_gen_out)[0]).reshape(N_gen, hidden_size)

    assert_quality(torch_und_out, und_torch_view, pcc=0.98)
    assert_quality(torch_gen_out, gen_torch_view, pcc=0.98)
