# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from ttnn.operations.layer_norm_tile import layer_norm

from tests.ttnn.utils_for_testing import assert_with_pcc

DRAM = ttnn.DRAM_MEMORY_CONFIG
L1 = ttnn.L1_MEMORY_CONFIG


def _torch_layer_norm(x, residual, gamma, beta, eps):
    z = x.float() if residual is None else x.float() + residual.float()
    return torch.nn.functional.layer_norm(z, (z.shape[-1],), gamma.float(), beta.float(), eps)


def _to_device(tensor, device, memory_config, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(tensor, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=memory_config)


@pytest.mark.parametrize(
    "shape",
    [(1, 1, 32, 32), (1, 1, 100, 768), (8, 1, 512, 768), (1, 1, 4224, 768), (2, 1, 48, 1024), (1, 1, 64, 2048)],
    ids=["one_tile", "h_non_aligned", "nomic_8x512", "nomic_long", "odd_bands", "wide"],
)
@pytest.mark.parametrize("has_residual", [True, False], ids=["residual", "no_residual"])
@pytest.mark.parametrize(
    "input_config, residual_config, output_config",
    [(DRAM, DRAM, DRAM), (L1, DRAM, DRAM), (DRAM, DRAM, L1)],
    ids=["all_dram", "l1_input", "l1_output"],
)
@pytest.mark.parametrize("affine_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT], ids=["affine_rm", "affine_tile"])
def test_layer_norm_tile(device, shape, has_residual, input_config, residual_config, output_config, affine_layout):
    if shape[0] * shape[2] * shape[3] * 2 > 8 * 1024 * 1024 and L1 in (input_config, output_config):
        pytest.skip("too large to hold in L1")
    torch.manual_seed(0)
    width = shape[-1]
    x = torch.randn(shape, dtype=torch.bfloat16)
    residual = torch.randn(shape, dtype=torch.bfloat16) if has_residual else None
    gamma = torch.randn(width, dtype=torch.bfloat16)
    beta = torch.randn(width, dtype=torch.bfloat16)
    affine_shape = (1, 1, 1, width) if affine_layout == ttnn.TILE_LAYOUT else (width // 32, 32)

    out = layer_norm(
        _to_device(x, device, input_config),
        _to_device(gamma.reshape(affine_shape), device, DRAM, affine_layout),
        _to_device(beta.reshape(affine_shape), device, DRAM, affine_layout),
        _to_device(residual, device, residual_config) if has_residual else None,
        epsilon=1e-5,
        memory_config=output_config,
    )
    assert out.memory_config().buffer_type == output_config.buffer_type
    assert_with_pcc(_torch_layer_norm(x, residual, gamma, beta, 1e-5), ttnn.to_torch(out).float(), 0.999)


@pytest.mark.parametrize("has_residual", [True, False], ids=["residual", "no_residual"])
@pytest.mark.parametrize("fp32_dest_acc_en", [True, False], ids=["fp32_dest", "bf16_dest"])
def test_layer_norm_tile_large_mean(device, has_residual, fp32_dest_acc_en):
    """Rows with |mean| >> std, where E[z^2] - E[z]^2 cancels. Under bf16 DEST, z = x + res rounds to 0.5 at
    |z| = 100 (ttnn.layer_norm measures PCC 0.992 on this input)."""
    torch.manual_seed(0)
    shape = (8, 1, 512, 768)
    x = 100 + torch.randn(shape, dtype=torch.bfloat16)
    residual = torch.randn(shape, dtype=torch.bfloat16) if has_residual else None
    gamma = torch.ones(768, dtype=torch.bfloat16)
    beta = torch.zeros(768, dtype=torch.bfloat16)
    config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=fp32_dest_acc_en, math_approx_mode=False
    )

    out = layer_norm(
        _to_device(x, device, DRAM),
        _to_device(gamma.reshape(24, 32), device, DRAM, ttnn.ROW_MAJOR_LAYOUT),
        _to_device(beta.reshape(24, 32), device, DRAM, ttnn.ROW_MAJOR_LAYOUT),
        _to_device(residual, device, DRAM) if has_residual else None,
        epsilon=1e-5,
        compute_kernel_config=config,
    )
    pcc = 0.999 if fp32_dest_acc_en or not has_residual else 0.99
    assert_with_pcc(_torch_layer_norm(x, residual, gamma, beta, 1e-5), ttnn.to_torch(out).float(), pcc)
