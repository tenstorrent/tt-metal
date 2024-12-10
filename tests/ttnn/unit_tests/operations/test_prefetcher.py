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

"""
Things to test:
- BFP8
- Different dataformats/shapes
    - Need to add support for multiple output tenosrs
    - Base it off of the input tensor shapes
- Multiple layers
    - Need to change how output tensor is tested?


Testing for writer side:
- Create and output_memory_config (maybe a new arg) across the receiver cores
- Alternative: Replace current output_tensor with output tensor
 sharded on the receiver cores (instead of the sender cores)
  - Requires a new CB (on just the receiver cores), and a new kernel that copies
  data on the global cb (local to the receiver cores) to the output cb on those cores
  -

"""


@pytest.mark.parametrize(
    "num_tensors, input_shapes, num_layers",
    [  # TODO: test different shapes etc
        (2, [(512, 512), (512, 512)], 1),
        # (2, [(128, 128), (128, 128)], 1), # Hangs
        # (1, [(512, 512)], 2), # Hangs
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
    input_shapes,
    num_layers,
    pcc_threshold,
    use_program_cache,
    function_level_defaults,
):
    logger.info(f"Running test_run_prefetcher with num_tensors={num_tensors}, input_shape={input_shapes[0]}")
    K, N = input_shapes[0]

    ##### Set up the Global CB #####
    dram_cores = [ttnn.CoreCoord(1, 0), ttnn.CoreCoord(2, 0)]  # DRAM banks 1 and 2
    sender_cores = [ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 4)]
    receiver_cores_list = [(1, 0), (2, 0), (1, 4), (2, 4)]
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

    receiver_core_range_set = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(
                ttnn.CoreCoord(x, y),
                ttnn.CoreCoord(x, y),
            )
            for x, y in receiver_cores_list
        ]
    )

    sender_receiver_mapping = dict(zip(sender_cores, receiver_cores))
    global_circular_buffer = ttnn.create_global_circular_buffer(device, sender_receiver_mapping, 2048 * 400)

    ##### Set up the input tensors #####
    dram_core_range_set = ttnn.CoreRangeSet([ttnn.CoreRange(core_coord, core_coord) for core_coord in dram_cores])
    sender_core_range_set = ttnn.CoreRangeSet([ttnn.CoreRange(core_coord, core_coord) for core_coord in sender_cores])

    pt_tensors = [torch.randn(input_shapes[tid]) for tid in range(num_tensors) for _ in range(num_layers)]
    tt_tensors = []

    for tid in range(num_tensors):
        K, N = input_shapes[tid]
        input_sharded_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(
                dram_core_range_set,
                [K, N // len(dram_cores)],
                ttnn.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )

        tt_tensor = ttnn.as_tensor(
            pt_tensors[tid * num_layers],  # Add a loop for num_layers
            device=device,
            dtype=ttnn.bfloat16,
            memory_config=input_sharded_mem_config,
            layout=ttnn.TILE_LAYOUT,
        )
        tt_tensors.append(tt_tensor)

    # Set up the tensor addrs
    tensor_addrs = torch.tensor([x.buffer_address() for x in tt_tensors])
    tensor_addrs = tensor_addrs.repeat(len(dram_cores), 1)
    tensor_addrs_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            sender_core_range_set,
            [tensor_addrs.shape[0] // len(dram_cores), tensor_addrs.shape[1]],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )
    tt_tensor_addrs = ttnn.as_tensor(
        tensor_addrs, device=device, dtype=ttnn.uint32, memory_config=tensor_addrs_mem_config
    )

    ##### Output mem config #####
    reader_output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            sender_core_range_set,
            [
                K * num_tensors * num_layers,
                N // len(sender_cores),
            ],  # Assuming all tensors have the same shape TODO: extend to different shapes
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )

    writer_output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            receiver_core_range_set,
            [K * num_tensors, N // receiver_core_range_set.num_cores()],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )

    breakpoint()

    tt_outs = ttnn.dram_prefetcher(
        tt_tensors,
        tt_tensor_addrs,
        num_layers,
        global_circular_buffer,
        reader_output_mem_config,
        writer_output_mem_config,
    )
    tt_outs_pt = ttnn.to_torch(tt_outs)
    tt_outs_pt = torch.chunk(tt_outs_pt, num_tensors, dim=0)

    # Check the output of DRAM Prefetcher
    all_passing = True
    for i in range(num_tensors):
        pt_out = ttnn.to_torch(tt_tensors[i])
        tt_out = tt_outs_pt[i]
        passing, output = comp_pcc(pt_out, tt_out, pcc_threshold)
        logger.info(output)

        all_passing = all_passing and passing

    assert all_passing
