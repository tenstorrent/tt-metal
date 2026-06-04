# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC for DiT ``ttnn.linear`` with production LoFi + ``bfloat8_b`` weights (real tiny weights)."""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.experimental.ace_step_v1_5.tests._dit_decoder_pcc_common import assert_pcc_print, tiny_dit_decoder_fixture
from models.experimental.ace_step_v1_5.tt_device import ace_step_dit_weight_mesh_mapper
from models.experimental.ace_step_v1_5.ttnn_impl.math_perf_env import (
    ace_step_dit_attn_linear_program_config,
    ace_step_dit_weight_dtype,
    ace_step_dit_weight_layout,
    ace_step_dit_weight_memory_config,
    ace_step_init_dit_linear_compute_kernel_config,
    ace_step_linear_kwargs_memory_config,
    ace_step_linear_l1_memory_config,
    ace_step_matmul_activation,
)


@pytest.mark.parametrize("proj", ["q_proj", "o_proj", "gate_proj"])
def test_pcc_dit_linear_lofi_bfloat8(mesh_device, proj: str):
    """Single DiT linear: BF16 activations × BFP8 weights, L1 in0/out, DRAM weights — PCC ≥ 0.99."""
    _, sd, _, seq_len, _ = tiny_dit_decoder_fixture(seq_len=64, intermediate=256)
    mesh = mesh_device
    act_dtype = ttnn.bfloat16
    w_dtype = ace_step_dit_weight_dtype(ttnn, act_dtype)
    dram = ace_step_dit_weight_memory_config(ttnn)
    l1 = ace_step_linear_l1_memory_config(ttnn)
    mapper = ace_step_dit_weight_mesh_mapper(mesh)

    w_host = None
    for prefix in ("layers.0.self_attn", "layers.0.mlp"):
        key = f"{prefix}.{proj}.weight"
        if key in sd:
            w_host = sd[key]
            break
    assert w_host is not None, proj

    in_dim = int(w_host.shape[1])
    out_dim = int(w_host.shape[0])
    b = 2
    s = int(seq_len)

    w_tt = ttnn.as_tensor(
        w_host,
        device=mesh,
        dtype=w_dtype,
        layout=ace_step_dit_weight_layout(ttnn, w_dtype, default_layout=ttnn.TILE_LAYOUT),
        memory_config=dram,
        mesh_mapper=mapper,
    )

    x_host = torch.randn(b, 1, s, in_dim, dtype=torch.float32)
    w_t = torch.as_tensor(w_host, dtype=torch.bfloat16)
    x_ref = (
        torch.nn.functional.linear(
            x_host.reshape(b * s, in_dim).to(torch.bfloat16),
            w_t,
        )
        .reshape(b, 1, s, out_dim)
        .float()
    )

    x_tt = ttnn.from_torch(
        x_host.to(torch.bfloat16),
        device=mesh,
        dtype=act_dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=l1,
    )

    pc = ace_step_dit_attn_linear_program_config(mesh, seq_len=s, in_dim=in_dim, out_dim=out_dim, batch_size=b)
    ck = ace_step_init_dit_linear_compute_kernel_config(mesh)
    lin_kw: dict = {"transpose_b": True}
    if ck is not None:
        lin_kw["compute_kernel_config"] = ck
    if pc is not None:
        lin_kw["program_config"] = pc
    lin_kw["memory_config"] = ace_step_linear_kwargs_memory_config(pc, linear_out_l1=l1, dram=dram)

    x_in = ace_step_matmul_activation(ttnn, x_tt, lin_kw, l1_fn=lambda t: ttnn.to_memory_config(t, l1), dram_mc=dram)
    y_tt = ttnn.linear(x_in, w_tt, **lin_kw)

    assert_pcc_print(f"dit_linear_{proj}_lofi_bfp8", x_ref, y_tt, pcc=0.99)
