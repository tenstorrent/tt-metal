# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
import itertools

from tests.ttnn.utils_for_testing import assert_with_pcc, assert_equal
from models.common.utility_functions import is_blackhole


# constants
RM = ttnn.ROW_MAJOR_LAYOUT
TILE = ttnn.TILE_LAYOUT
DRAM = ttnn.DRAM_MEMORY_CONFIG
L1 = ttnn.L1_MEMORY_CONFIG


def skip_if_not_20_cores(device):
    compute_grid = device.compute_with_storage_grid_size()
    if compute_grid.x != 5 or compute_grid.y != 4:
        pytest.skip(
            f"This test is intended to run only with 20 cores. Core grid [{compute_grid.x},{compute_grid.y}] must be [5, 4]."
        )


def random_torch_tensor(dtype, shape):
    if dtype == ttnn.int32:
        return torch.randint(-(2**31), 2**31, shape, dtype=torch.int32)
    if dtype == ttnn.uint32:
        return torch.randint(0, 2**31, shape, dtype=torch.int32)
    if dtype == ttnn.float32:
        return torch.rand(shape, dtype=torch.float32)
    if dtype == ttnn.bfloat16:
        return torch.rand(shape, dtype=torch.bfloat16)


@pytest.mark.parametrize(
    "shape, perm, dtype, layout, in_mem_config, out_mem_config",
    [
        ([1, 48, 160, 256], [2, 1, 0, 3], ttnn.uint32, TILE, DRAM, DRAM),
        ([160, 48, 1, 256], [2, 1, 0, 3], ttnn.uint32, TILE, DRAM, DRAM),
        ([1, 24, 80, 256], [2, 1, 0, 3], ttnn.uint32, TILE, DRAM, DRAM),
        ([80, 24, 1, 256], [2, 1, 0, 3], ttnn.uint32, TILE, DRAM, DRAM),
        ([1, 12, 40, 256], [2, 1, 0, 3], ttnn.uint32, TILE, DRAM, DRAM),
        ([40, 12, 1, 256], [2, 1, 0, 3], ttnn.uint32, TILE, DRAM, DRAM),
        ([1, 1, 25281, 9], [0, 3, 1, 2], ttnn.bfloat16, RM, DRAM, L1),
        ([1, 3, 159, 160], [0, 2, 3, 1], ttnn.bfloat16, TILE, L1, L1),
        ([1, 1, 159, 160], [0, 2, 3, 1], ttnn.bfloat16, TILE, L1, L1),
    ],
)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 100000}], indirect=True)
def test_permute(device, shape, perm, dtype, layout, in_mem_config, out_mem_config):
    skip_if_not_20_cores(device)

    torch.manual_seed(2005)
    torch_input_tensor = random_torch_tensor(dtype, shape)
    torch_output_tensor = torch.permute(torch_input_tensor, perm)

    input_tensor = ttnn.from_torch(
        torch_input_tensor, layout=layout, dtype=dtype, device=device, memory_config=in_mem_config
    )
    output_tensor = ttnn.permute(input_tensor, perm, memory_config=out_mem_config)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_equal(torch_output_tensor, output_tensor)

    ## Compile run
    # output_tensor = ttnn.permute(input_tensor, perm)
    # ttnn.synchronize_device(device)

    ## Capture trace
    # num_iters = 20
    # trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    # for _ in range(num_iters):
    #    output_tensor = ttnn.permute(input_tensor, perm)
    # ttnn.end_trace_capture(device, trace_id)
    # ttnn.synchronize_device(device)

    ## Run the trace
    # ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
    # ttnn.release_trace(device, trace_id)
    # ttnn.synchronize_device(device)
