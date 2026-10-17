# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# Probe only (not for merge): measures the sharded row-major WH transpose for H values that reach the
# narrow-row pack untilize, printing the mismatch count per H on shipped code.
import pytest
import torch

import ttnn


def _height_sharded_rm_input(device, h, w, c=4):
    torch.manual_seed(2005)
    torch_input = torch.rand((1, c, h, w), dtype=torch.bfloat16)
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    shard_spec = ttnn.ShardSpec(shard_grid, (c * h, w), ttnn.ShardOrientation.ROW_MAJOR)
    sharded_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec
    )
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=sharded_mem_config,
    )
    return torch_input, tt_input


@pytest.mark.parametrize("h", [4, 8, 16, 20, 24, 28, 36, 40, 48, 100])
@pytest.mark.parametrize("w", [64])
def test_probe_transpose_hw_sharded_rm_narrow_row(device, h, w):
    torch_input, tt_input = _height_sharded_rm_input(device, h, w)
    expected = torch_input.transpose(2, 3)
    try:
        tt_output = ttnn.transpose(tt_input, 2, 3)
    except RuntimeError as e:
        print(f"\nPROBE arch={device.arch()} h={h} w={w} REJECTED: {str(e).splitlines()[0][:200]}")
        raise
    tt_output = ttnn.to_torch(ttnn.from_device(ttnn.to_memory_config(tt_output, ttnn.L1_MEMORY_CONFIG)))
    mismatches = int((expected != tt_output).sum())
    print(f"\nPROBE arch={device.arch()} h={h} w={w} row_bytes={2 * h} mismatches={mismatches}/{expected.numel()}")
    assert mismatches == 0
