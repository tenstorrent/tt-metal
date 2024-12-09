# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger

from tests.ttnn.utils_for_testing import assert_with_pcc
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_equal,
    comp_pcc,
)


@pytest.mark.parametrize(
    "num_tensors, input_shape",
    [
        (2, (512, 512)),
    ],
)
@pytest.mark.parametrize(
    "pcc_threshold",
    [
        1.0,
    ],
)
def test_run_prefetcher(
    device,
    num_tensors,
    input_shape,
    pcc_threshold,
    use_program_cache,
    function_level_defaults,
):
    K, N = input_shape

    ##### Set up the Global CB #####
    dram_cores = [ttnn.CoreCoord(1, 0), ttnn.CoreCoord(2, 0)]  # DRAM banks 1 and 2
    sender_cores = [ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 4)]
    receiver_cores = [
        ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(1, 0),
                    ttnn.CoreCoord(2, 0),
                ),
            }
        ),
        ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(1, 4),
                    ttnn.CoreCoord(2, 4),
                ),
            }
        ),
    ]
    sender_receiver_mapping = dict(zip(sender_cores, receiver_cores))
    global_circular_buffer = ttnn.create_global_circular_buffer(device, sender_receiver_mapping, 2048 * 400)

    ##### Set up the input tensors #####
    dram_core_range_set = ttnn.CoreRangeSet([ttnn.CoreRange(core_coord, core_coord) for core_coord in dram_cores])
    core_range_set = ttnn.CoreRangeSet([ttnn.CoreRange(core_coord, core_coord) for core_coord in sender_cores])
    input_sharded_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(
            dram_core_range_set,
            [K, N // len(sender_cores)],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )

    pt_tensors = [torch.randn(input_shape) for _ in range(num_tensors)]
    tt_tensors = []
    for i in range(num_tensors):
        tt_tensor = ttnn.as_tensor(
            pt_tensors[i],
            device=device,
            dtype=ttnn.bfloat16,
            memory_config=input_sharded_mem_config,
        )
        tt_tensors.append(tt_tensor)

    tensor_addrs = torch.tensor([x.buffer_address() for x in tt_tensors])
    tensor_addrs = tensor_addrs.repeat(len(dram_cores), 1)
    tensor_addrs_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            core_range_set,
            [tensor_addrs.shape[0] // len(dram_cores), tensor_addrs.shape[1]],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )
    tt_tensor_addrs = ttnn.as_tensor(
        tensor_addrs, device=device, dtype=ttnn.uint32, memory_config=tensor_addrs_mem_config
    )

    tt_out = ttnn.dram_prefetcher(tt_tensors, tt_tensor_addrs, global_circular_buffer)
    tt_out = ttnn.to_torch(tt_out)

    # Check the output of DRAM Prefetcher
    all_passing = True
    for i in range(num_tensors):  # TODO: Update this when output tensor is returning more than just one tensor
        pt_out = pt_tensors[i]
        passing, output = comp_pcc(pt_out, tt_out, pcc_threshold)
        logger.info(output)

        all_passing = all_passing and passing

    assert all_passing
