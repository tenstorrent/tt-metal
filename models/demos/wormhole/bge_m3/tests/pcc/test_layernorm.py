# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

import ttnn
from models.common.auto_compose import to_torch_auto_compose
from models.common.modules.lazy_weight import LazyWeight
from models.common.utility_functions import comp_allclose, comp_pcc
from models.demos.wormhole.bge_m3.tt.norm import LayerNorm1D


def _lazy_weight_from_vector(weight_1d: torch.Tensor, device) -> LazyWeight:
    return LazyWeight(
        source=weight_1d,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _to_ttnn_activation(x: torch.Tensor, device) -> ttnn.Tensor:
    return ttnn.from_torch(
        x,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _to_torch_output(tt_output: ttnn.Tensor, expected_shape: tuple[int, int, int, int]) -> torch.Tensor:
    out = to_torch_auto_compose(tt_output).to(torch.float32)
    assert tuple(out.shape) == expected_shape, f"Expected output shape {expected_shape}, got {tuple(out.shape)}"
    return out


def _skip_if_not_single_device(device) -> None:
    if hasattr(device, "get_num_devices") and device.get_num_devices() != 1:
        pytest.skip("BGE-M3 LayerNorm tests currently target single-device execution")


def _reference_layer_norm(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    hidden_size = x.shape[-1]
    return F.layer_norm(
        x.to(torch.float32),
        normalized_shape=(hidden_size,),
        weight=gamma.to(torch.float32),
        bias=beta.to(torch.float32),
        eps=eps,
    )


def test_layernorm_vs_pytorch(device):
    """
    Validate LayerNorm parity against torch.nn.functional.layer_norm on [B,1,S,D].
    Also validates non-trivial affine parameters (gamma + beta).
    """
    _skip_if_not_single_device(device)
    torch.manual_seed(11)

    batch_size, seq_len, hidden_size = 2, 8192, 1024
    eps = 1e-5

    x = torch.randn((batch_size, 1, seq_len, hidden_size), dtype=torch.bfloat16)
    gamma = torch.randn((hidden_size,), dtype=torch.bfloat16)
    beta = torch.randn((hidden_size,), dtype=torch.bfloat16)

    tt_x = _to_ttnn_activation(x, device=device)
    model = LayerNorm1D(
        weight=_lazy_weight_from_vector(gamma, device),
        bias=_lazy_weight_from_vector(beta, device),
        eps=eps,
    )
    tt_out = model.forward(tt_x)
    tt_out_torch = _to_torch_output(tt_out, expected_shape=(batch_size, 1, seq_len, hidden_size))

    ref_out = _reference_layer_norm(x=x, gamma=gamma, beta=beta, eps=eps)
    passing, pcc_message = comp_pcc(ref_out, tt_out_torch, 0.999)
    allclose, allclose_message = comp_allclose(ref_out, tt_out_torch)
    assert passing, f"PCC check failed: {pcc_message}; {allclose_message}; allclose={allclose}"
