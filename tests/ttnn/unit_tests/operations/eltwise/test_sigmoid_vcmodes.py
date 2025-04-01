# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc, assert_equal
from tests.ttnn.unit_tests.operations.eltwise.backward.utility_funcs import data_gen_with_range, compare_pcc
from models.utility_functions import torch_random, skip_for_grayskull, is_wormhole_b0, is_blackhole


def run_unary_test_sharded(device, hw, out_channels, ttnn_function, pcc=0.9999):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((hw, out_channels), dtype=torch.bfloat16)
    golden_function = ttnn.get_golden_function(ttnn_function)
    torch_output_tensor = golden_function(torch_input_tensor, device=device)

    num_cores = 64
    core_range_set = ttnn.num_cores_to_corerangeset(num_cores, device.compute_with_storage_grid_size())
    shard_shape = [hw // num_cores, out_channels if out_channels >= 32 else 32]

    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            core_range_set,
            shard_shape,
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )

    input_tensor = ttnn.from_torch(
        torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
    )
    output_tensor = ttnn_function(input_tensor, mode=2, fast_and_approximate_mode=False)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    pcc_v, pcc_message = assert_with_pcc(torch_output_tensor, output_tensor, pcc)
    print(pcc_v)
    print(pcc_message)


@pytest.mark.parametrize("h", [2048])
@pytest.mark.parametrize("w", [128])
@pytest.mark.parametrize("out_channels", [2])
def test_sigmoid_two_out_channels(device, h, w, out_channels):
    torch.manual_seed(0)
    run_unary_test_sharded(device, h * w, out_channels, ttnn.sigmoid, pcc=0.991)
