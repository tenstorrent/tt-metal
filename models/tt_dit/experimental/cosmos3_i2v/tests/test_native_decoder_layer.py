# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC test for the full native Cosmos3 decoder layer (TP variants).

Builds the torch reference (`Cosmos3VLTextMoTDecoderLayer`) with random
weights and runs the same weights through the native
`Cosmos3VLTextMoTDecoderLayer` on subsets of a WH LoudBox mesh —
(1,1), (1,2), (1,4), (1,8).

This validates the full per-layer pipeline end-to-end: 4 RMSNorms, the
native joint-attention, two native MLPs, and the two residual adds, all
composed on device. PCC bound is slightly looser than per-module tests
(0.97) because errors compound across the stages.
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


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param((1, 1), line_params, id="1x1"),
        pytest.param((1, 2), line_params, id="1x2"),
        pytest.param((1, 4), line_params, id="1x4"),
        pytest.param((1, 8), line_params, id="1x8"),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.timeout(300)
def test_native_decoder_layer(mesh_device: ttnn.MeshDevice) -> None:
    from models.tt_dit.experimental.cosmos3_i2v.model.decoder_layer import (
        Cosmos3VLTextMoTDecoderLayer as TTDecoderLayer,
    )
    from models.tt_dit.experimental.cosmos3_i2v.reference.transformer_cosmos3 import (
        Cosmos3VLTextMoTDecoderLayer as RefDecoderLayer,
    )
    from models.tt_dit.experimental.cosmos3_i2v.reference.transformer_cosmos3 import Cosmos3VLTextRotaryEmbedding

    torch.manual_seed(42)

    mesh_shape = tuple(mesh_device.shape)
    tp_factor = mesh_shape[1]
    sp_factor = mesh_shape[0]

    hidden_size = 256
    head_dim = 64
    num_attention_heads = 32
    num_key_value_heads = 8
    intermediate_size = 512
    rms_norm_eps = 1e-6
    rope_theta = 5_000_000.0
    rope_axes_dim = (12, 10, 10)
    N_und = 128
    N_gen = 128
    N_total = N_und + N_gen

    torch_layer = RefDecoderLayer(
        hidden_size=hidden_size,
        head_dim=head_dim,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        intermediate_size=intermediate_size,
        attention_bias=False,
        rms_norm_eps=rms_norm_eps,
    )
    torch_layer.eval()
    torch_layer.to(dtype=torch.bfloat16)

    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(1, 0),
        sequence_parallel=ParallelFactor(sp_factor, 0),
        tensor_parallel=ParallelFactor(tp_factor, 1),
    )
    ccl_manager = (
        CCLManager(mesh_device=mesh_device, num_links=1, topology=ttnn.Topology.Linear) if tp_factor > 1 else None
    )

    tt_layer = TTDecoderLayer(
        hidden_size=hidden_size,
        head_dim=head_dim,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        intermediate_size=intermediate_size,
        attention_bias=False,
        rms_norm_eps=rms_norm_eps,
        mesh_device=mesh_device,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
    )
    tt_layer.load_torch_state_dict(torch_layer.state_dict())

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
        torch_und_out, torch_gen_out = torch_layer(
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

    tt_und_out, tt_gen_out = tt_layer(
        und_seq_tt,
        gen_seq_tt,
        cos_und_tt,
        sin_und_tt,
        cos_gen_tt,
        sin_gen_tt,
    )

    und_torch_view = ttnn.to_torch(ttnn.get_device_tensors(tt_und_out)[0]).reshape(N_und, hidden_size)
    gen_torch_view = ttnn.to_torch(ttnn.get_device_tensors(tt_gen_out)[0]).reshape(N_gen, hidden_size)

    assert_quality(torch_und_out, und_torch_view, pcc=0.97)
    assert_quality(torch_gen_out, gen_torch_view, pcc=0.97)
