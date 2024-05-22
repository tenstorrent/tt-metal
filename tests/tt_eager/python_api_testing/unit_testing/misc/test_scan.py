# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import tt_lib as ttl
from tt_lib import tensor as tt

from models.utility_functions import skip_for_wormhole_b0, torch2tt_tensor, tt2torch_tensor
from tests.ttnn.utils_for_testing import assert_with_pcc


def pretty_print(tensor, file=None, window=((0, None), (0, None))):
    (start_row, end_row), (start_col, end_col) = window
    tensor_np = tensor[start_row:end_row, start_col:end_col].to(torch.float32).numpy()
    max_width = max(len(f"{val:g}") for val in tensor_np.flatten())

    for row in tensor_np:
        print(" ".join(f"{val:{max_width}g}" for val in row), file=file)


def seq_matrix(height, width, dtype, step=1):
    if height % 32 != 0 or width % 32 != 0:
        raise ValueError("Dimensions must be divisible by 32")

    patches_height = height // 32
    patches_width = width // 32

    result = torch.zeros(height, width, dtype=dtype)
    current_value = 1

    for i in range(patches_height):
        for j in range(patches_width):
            result[i * 32 : (i + 1) * 32, j * 32 : (j + 1) * 32] = current_value
            current_value += step

    return result


def test_scan(device):
    torch.manual_seed(0)

    shape = (64, 2048)
    torch_input = torch.rand(*shape, dtype=torch.bfloat16)

    expected = torch.cumprod(torch_input, dim=0)

    shard_grid = tt.CoreRangeSet(
        {
            tt.CoreRange(
                tt.CoreCoord(0, 0),
                tt.CoreCoord(0, 0),
            )
        }
    )

    shard_spec = tt.ShardSpec(shard_grid, [64, 2048], tt.ShardOrientation.ROW_MAJOR, False)

    tt_input = torch2tt_tensor(
        torch_input,
        device,
        tt.Layout.TILE,
        tt_memory_config=tt.MemoryConfig(tt.TensorMemoryLayout.HEIGHT_SHARDED, tt.BufferType.L1, shard_spec),
    )

    tt_input = tt.scan(tt_input)
    after_scan = tt2torch_tensor(tt_input)

    assert_with_pcc(expected, after_scan.squeeze(), 0.999997)


def test_retile_to_row_major(device):
    torch.manual_seed(0)

    shape = [192, 2048]
    torch_input = seq_matrix(*shape, dtype=torch.bfloat16)  # torch.rand(*shape, dtype=torch.bfloat16) #

    expected = torch.cumprod(torch_input, dim=0)

    shard_grid = tt.CoreRangeSet(
        {
            tt.CoreRange(
                tt.CoreCoord(0, 0),
                tt.CoreCoord(0, 2),
            )
        }
    )

    shard_spec = tt.ShardSpec(shard_grid, [64, 2048], tt.ShardOrientation.ROW_MAJOR, False)

    tt_input = torch2tt_tensor(
        torch_input,
        device,
        tt.Layout.TILE,
        tt_memory_config=tt.MemoryConfig(tt.TensorMemoryLayout.HEIGHT_SHARDED, tt.BufferType.L1, shard_spec),
    )

    # tt_input = tt.scan(tt_input)
    # tt_input = tt.retile_to_row_major(tt_input)
    # tt_input = tt.scan_only(tt_input)
    # tt_input = tt.undo_retile_to_row_major(tt_input)
    tt_input = tt.scan_communicate(tt_input)
    after_scan = tt2torch_tensor(tt_input)

    with open("expected.txt", "w") as f:
        pretty_print(expected, f)

    with open("after_scan.txt", "w") as f:
        pretty_print(after_scan.squeeze(), f)

    # assert_with_pcc(expected, after_scan.squeeze(), 0.999997)
    # torch.testing.assert_allclose(expected, after_scan.squeeze(), rtol=1e-5, atol=1e-5, equal_nan=True)
