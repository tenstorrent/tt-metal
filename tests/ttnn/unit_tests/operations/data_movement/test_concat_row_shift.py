# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# A TILE concat along the rows whose inputs' rows are not whole tiles: bit-exact against torch, and one
# program for every row count.

import pytest
import torch

import ttnn


def _run(device, row_counts, width, batch, dtype):
    torch_dtype = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
    torch.manual_seed(sum(row_counts) + width)
    torch_inputs = [torch.randn(*batch, rows, width, dtype=torch.float32).to(torch_dtype) for rows in row_counts]
    inputs = [
        ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for t in torch_inputs
    ]
    output = ttnn.concat(inputs, dim=-2)
    expected = torch.cat(torch_inputs, dim=-2)
    actual = ttnn.to_torch(output)
    assert actual.shape == expected.shape
    assert torch.equal(actual, expected), f"rows {row_counts}, width {width}, batch {batch}"


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize(
    "row_counts",
    [
        [1, 1],
        [32, 64],
        [3, 32],
        [17, 15],
        [31, 33],
        [33, 1],
        [64, 7],
        [100, 45],
        [3317, 1003],
        [5, 6, 7],
        [40, 1, 70, 16],
    ],
)
@pytest.mark.parametrize("width", [32, 128, 96])
def test_concat_row_shift_exact(device, row_counts, width, dtype):
    _run(device, row_counts, width, [1], dtype)


@pytest.mark.parametrize("batch", [[2], [2, 3]])
def test_concat_row_shift_batched(device, batch):
    _run(device, [45, 19], 64, batch, ttnn.bfloat16)


def test_concat_row_shift_one_program(device):
    device.enable_program_cache()
    # Warmed at whole tiles, as the pipeline is at its default size.
    _run(device, [1024, 1024], 128, [1], ttnn.bfloat16)
    entries = device.num_program_cache_entries()
    for row_counts in [[3317, 1003], [3685, 987], [17, 15], [1, 1], [4500, 2000], [1003, 3317], [33, 64], [64, 32]]:
        _run(device, row_counts, 128, [1], ttnn.bfloat16)
    assert device.num_program_cache_entries() == entries
