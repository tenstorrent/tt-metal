# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
import ttnn
from models.utility_functions import is_wormhole_b0, is_grayskull, skip_for_wormhole_b0
import torch
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_equal,
)

from tests.ttnn.unit_tests.operations.speculative_execution.sfd_common import (
    get_buffer_address,
)


@pytest.mark.skipif(is_grayskull(), reason="GS does not support fp32")
@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 1, 1],
        [1, 1, 256, 256],
        [1, 1, 1, 1024],
        [1, 1, 512, 16],
        [1, 1024, 1, 16],
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.uint32,
        ttnn.bfloat16,
    ],
)
def test_priority_tensor(
    device,
    shape,
    dtype,
    function_level_defaults,
    use_program_cache,
):
    num_cores = 64  # TODO: Get this from the device
    shape[0] = num_cores

    # Create the priority tensor
    torch_tensor = torch.randn(shape)
    tensor_mem_config = ttnn.create_sharded_memory_config(
        shape=(shape[1] * shape[2], shape[-1]),
        core_grid=ttnn.num_cores_to_corerangeset(num_cores, device.compute_with_storage_grid_size(), row_wise=True),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    # Create the src and dst tensors
    src_tensor = ttnn.from_torch(
        torch_tensor,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=dtype,
        memory_config=tensor_mem_config,
    )
    dst_tensor = ttnn.ones_like(src_tensor, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT)

    logger.info(f"src_tensor.shape: {src_tensor.shape}")
    ttnn.copy_tensor(src_tensor, dst_tensor)

    # Check the output
    tt_out_src = ttnn.to_torch(src_tensor)
    tt_out_dst = ttnn.to_torch(dst_tensor)

    passing, output = comp_equal(tt_out_src, tt_out_dst)
    logger.info(output)

    assert passing
