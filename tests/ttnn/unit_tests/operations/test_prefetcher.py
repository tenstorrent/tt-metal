# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

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
def test_run_prefetcher(
    device,
    num_tensors,
    input_shape,
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
    global_circular_buffer = ttnn.create_global_circular_buffer(device, sender_receiver_mapping, 3200)

    ##### Set up the input tensors #####
    core_range_set = ttnn.CoreRangeSet([ttnn.CoreRange(core_coord, core_coord) for core_coord in dram_cores])
    input_sharded_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(
            core_range_set,
            [K, N // len(sender_cores)],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )

    tt_tensors = []
    for i in range(num_tensors):
        tt_tensor = ttnn.as_tensor(
            torch.randn(input_shape),
            device=device,
            dtype=ttnn.bfloat16,
            memory_config=input_sharded_mem_config,
        )
        tt_tensors.append(tt_tensor)

    tensor_addrs = torch.tensor([x.buffer_address() for x in tt_tensors])
    tensor_addrs_mem_config = None  # TODO: implement
    tt_tensor_addrs = ttnn.as_tensor(
        tensor_addrs, device=device, dtype=ttnn.uint32, memory_config=tensor_addrs_mem_config
    )

    tt_out = ttnn.dram_prefetcher(tt_tensors, global_circular_buffer)
    tt_out = ttnn.to_torch(tt_out)

    # Check the output of DRAM Prefetcher
    all_passing = True

    pt_tensors = [ttnn.to_torch(x) for x in tt_tensors]
    pt_tensors = torch.cat(pt_tensors, dim=0)

    for i in range(num_tensors):
        pt_out = ttnn.to_torch(tt_tensors[i])
        passing, output = comp_pcc(pt_out, tt_out, pcc_threshold)
        logger.info(output)

        all_passing = all_passing and passing

    assert all_passing
