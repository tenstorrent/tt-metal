# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest

import ttnn
import tt_lib as ttl

from models.demos.mamba.tt.transforms import MambaSsmBlockTransformer
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)

N = 32
HIDDEN_SIZE = 2560


@pytest.mark.parametrize(
    "batch, pcc",
    (
        (
            32,
            0.99,
        ),
    ),
)
def test_mamba_ssm_block_repeat_interleave(
    device: ttnn.Device,
    use_program_cache,
    batch: int,
    pcc: float,
):
    input = torch.rand(1, 1, batch, HIDDEN_SIZE * 2)

    expected = torch.repeat_interleave(input, N, dim=3)

    transformer = MambaSsmBlockTransformer(device, batch, HIDDEN_SIZE * 2, N)
    input = ttnn.to_device(
        ttnn.from_torch(input, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16),
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    actual = transformer.repeat_interleave(
        input,
        memory_config=ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
    )

    assert list(actual.get_legacy_shape()) == [1, 1, batch, 2 * HIDDEN_SIZE * N]

    actual = ttnn.to_torch(actual)
    passing_pcc, output_pcc = comp_pcc(actual, expected, 0.9999)
    assert passing_pcc


@pytest.mark.parametrize(
    "batch, pcc",
    (
        (
            32,
            0.99,
        ),
    ),
)
def test_mamba_ssm_block_repeat(
    device: ttnn.Device,
    batch: int,
    pcc: float,
    use_program_cache,
):
    input = torch.rand(1, 1, batch, N)

    # (1, 1, B, n) -> (1, 1, B, hidden * 2 * n)
    expected = input.repeat((1, 1, 1, HIDDEN_SIZE * 2))

    transformer = MambaSsmBlockTransformer(device, batch, HIDDEN_SIZE * 2, N)
    input = ttnn.to_device(
        ttnn.from_torch(input, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16),
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    actual = transformer.repeat(
        input,
        memory_config=ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
    )

    assert list(actual.get_legacy_shape()) == [1, 1, batch, 2 * HIDDEN_SIZE * N]

    actual = ttnn.to_torch(actual)
    passing_pcc, output_pcc = comp_pcc(actual, expected, 0.9999)
    assert passing_pcc
