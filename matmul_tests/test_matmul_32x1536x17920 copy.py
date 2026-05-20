# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_numeric_metrics

# MatmulDeviceOperation dimensions: M x K x N
M = 32
K = 1536
N = 17920


@pytest.mark.parametrize(
    "m_size, k_size, n_size",
    [(M, K, N)],
    ids=[f"MatmulDeviceOperation_{M}x{K}x{N}"],
)
def test_matmul_device_operation_32x1536x17920(device, m_size, k_size, n_size):
    """PCC test for MatmulDeviceOperation with M=32, K=1536, N=17920."""
    torch.manual_seed(0)

    torch_input_a = torch.randn((1, 1, m_size, k_size), dtype=torch.bfloat16)
    torch_input_b = torch.randn((1, 1, k_size, n_size), dtype=torch.bfloat16)
    torch_output = torch.matmul(torch_input_a, torch_input_b)

    input_a = ttnn.from_torch(
        torch_input_a,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        device=device,
    )
    input_b = ttnn.from_torch(
        torch_input_b,
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        device=device,
    )

    output = ttnn.matmul(input_a, input_b)
    output = ttnn.to_torch(output)

    assert output.shape == torch_output.shape
    assert_numeric_metrics(
        torch_output,
        output,
        atol=0.004 * k_size,
        rtol=0.004 * k_size,
        frobenius_threshold=0.003 * k_size,
        pcc_threshold=0.999,
        check_ulp=False,
        check_allclose=False,
    )
