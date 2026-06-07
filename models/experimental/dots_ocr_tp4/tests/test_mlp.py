# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TP4 MLP correctness vs a torch SwiGLU reference (dots.ocr text-decoder dims)."""

import pytest
import torch
import torch.nn as nn

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

from models.experimental.dots_ocr_tp4.tt.common import DotsOCRConfig, from_replicated_to_torch, to_replicated
from models.experimental.dots_ocr_tp4.tt.mlp import DotsOCRMLPTP4
from models.experimental.dots_ocr_tp4.tests.common import device_params, resolve_mesh_shape


class TorchSwiGLUMLP(nn.Module):
    """Reference Qwen2/dots.ocr SwiGLU MLP (no bias)."""

    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


@pytest.mark.parametrize("device_params", [device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [resolve_mesh_shape()], indirect=True)
@pytest.mark.parametrize("seq_len", [2816])
def test_dots_ocr_mlp_tp4(mesh_device, seq_len):
    torch.manual_seed(0)
    torch.set_grad_enabled(False)

    config = DotsOCRConfig()
    H, I = config.hidden_size, config.intermediate_size

    # Reference stays float32; ttnn casts the same weights to bf16 on load.
    torch_mlp = TorchSwiGLUMLP(H, I).eval()

    x = torch.randn(1, seq_len, H, dtype=torch.bfloat16)
    torch_out = torch_mlp(x.to(torch.float32))

    tt_mlp = DotsOCRMLPTP4.from_torch(mesh_device, config, torch_mlp)
    x_tt = to_replicated(x, mesh_device, dtype=ttnn.bfloat16)

    out_tt = tt_mlp.forward(x_tt)
    ttnn.synchronize_device(mesh_device)

    out_torch = from_replicated_to_torch(out_tt, mesh_device).to(torch.float32).reshape(torch_out.shape)

    # MLP uses the production low-precision recipe (BFP4 gate/up + BFP4 down,
    # BFP8 activations); a single SwiGLU at this width lands ~0.99 PCC.
    assert_with_pcc(torch_out.to(torch.float32), out_torch, pcc=0.98)
