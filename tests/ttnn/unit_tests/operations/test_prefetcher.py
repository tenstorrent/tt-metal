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

from tests.tt_eager.python_api_testing.unit_testing.misc.test_matmul_1d_gather_in0 import run_multi_core_matmul_1d

"""
Things to test:
- BFP8
- Different dataformats/shapes
    - Need to add support for multiple output tenosrs
    - Base it off of the input tensor shapes
- Multiple layers
    - Need to change how output tensor is tested?
- Non-square shapes


Testing for writer side:
- Create and output_memory_config (maybe a new arg) across the receiver cores
- Alternative: Replace current output_tensor with output tensor
 sharded on the receiver cores (instead of the sender cores)
  - Requires a new CB (on just the receiver cores), and a new kernel that copies
  data on the global cb (local to the receiver cores) to the output cb on those cores
  -

"""


def get_core_ranges(num_reader_cores):
    # all_dram_cores = [
    #     ttnn.CoreCoord(1, 0),
    #     ttnn.CoreCoord(2, 0),
    #     ttnn.CoreCoord(3, 0),
    #     ttnn.CoreCoord(0, 0),
    #     ttnn.CoreCoord(4, 0),
    #     ttnn.CoreCoord(6, 0),
    #     ttnn.CoreCoord(9, 0),
    #     ttnn.CoreCoord(10, 0),
    #     ttnn.CoreCoord(11, 0),
    #     ttnn.CoreCoord(8, 0),
    #     ttnn.CoreCoord(7, 0),
    #     ttnn.CoreCoord(5, 0),
    # ]
    # all_sender_cores = [
    #     ttnn.CoreCoord(0, 0),
    #     ttnn.CoreCoord(0, 4),
    #     ttnn.CoreCoord(0, 5),
    #     ttnn.CoreCoord(0, 9),
    #     ttnn.CoreCoord(4, 0),
    #     ttnn.CoreCoord(4, 1),
    #     ttnn.CoreCoord(4, 2),
    #     ttnn.CoreCoord(4, 4),
    #     ttnn.CoreCoord(4, 5),
    #     ttnn.CoreCoord(4, 6),
    #     ttnn.CoreCoord(4, 7),
    #     ttnn.CoreCoord(4, 9),
    # ]
    # all_receiver_cores_list = [
    #     (1, 0),
    #     # (2, 0),
    #     (1, 4),
    #     # (2, 4),
    #     (1, 5),
    #     # (2, 5),
    #     (1, 9),
    #     # (2, 9),
    #     (5, 0),
    #     # (6, 0),
    #     (5, 1),
    #     # (6, 1),
    #     (5, 2),
    #     (6, 2),
    #     (5, 4),
    #     (6, 4),
    #     (5, 5),
    #     (6, 5),
    #     (5, 6),
    #     (6, 6),
    #     (5, 7),
    #     (6, 7),
    #     (5, 9),
    #     (6, 9),
    # ]
    # all_receiver_cores = [
    #     ttnn.CoreRangeSet(
    #         [
    #             ttnn.CoreRange(
    #                 ttnn.CoreCoord(1, 0),
    #                 ttnn.CoreCoord(1, 0),
    #                 # ttnn.CoreCoord(2, 0),
    #             ),
    #         ]
    #     ),
    #     ttnn.CoreRangeSet(
    #         [
    #             ttnn.CoreRange(
    #                 ttnn.CoreCoord(1, 4),
    #                 ttnn.CoreCoord(1, 4),
    #                 # ttnn.CoreCoord(2, 4),
    #             ),
    #         ]
    #     ),
    #     ttnn.CoreRangeSet(
    #         [
    #             ttnn.CoreRange(
    #                 ttnn.CoreCoord(1, 5),
    #                 ttnn.CoreCoord(1, 5),
    #                 # ttnn.CoreCoord(2, 5),
    #             ),
    #         ]
    #     ),
    #     ttnn.CoreRangeSet(
    #         [
    #             ttnn.CoreRange(
    #                 ttnn.CoreCoord(1, 9),
    #                 ttnn.CoreCoord(1, 9),
    #                 # ttnn.CoreCoord(2, 9),
    #             ),
    #         ]
    #     ),
    #     ttnn.CoreRangeSet(
    #         [
    #             ttnn.CoreRange(
    #                 ttnn.CoreCoord(5, 0),
    #                 ttnn.CoreCoord(5, 0),
    #                 # ttnn.CoreCoord(6, 0),
    #             ),
    #         ]
    #     ),
    #     ttnn.CoreRangeSet(
    #         [
    #             ttnn.CoreRange(
    #                 ttnn.CoreCoord(5, 1),
    #                 ttnn.CoreCoord(5, 1),
    #                 # ttnn.CoreCoord(6, 1),
    #             ),
    #         ]
    #     ),
    #     ttnn.CoreRangeSet(
    #         [
    #             ttnn.CoreRange(
    #                 ttnn.CoreCoord(5, 2),
    #                 ttnn.CoreCoord(6, 2),
    #             ),
    #         ]
    #     ),
    #     ttnn.CoreRangeSet(
    #         [
    #             ttnn.CoreRange(
    #                 ttnn.CoreCoord(5, 4),
    #                 ttnn.CoreCoord(6, 4),
    #             ),
    #         ]
    #     ),
    #     ttnn.CoreRangeSet(
    #         [
    #             ttnn.CoreRange(
    #                 ttnn.CoreCoord(5, 5),
    #                 ttnn.CoreCoord(6, 5),
    #             ),
    #         ]
    #     ),
    #     ttnn.CoreRangeSet(
    #         [
    #             ttnn.CoreRange(
    #                 ttnn.CoreCoord(5, 6),
    #                 ttnn.CoreCoord(6, 6),
    #             ),
    #         ]
    #     ),
    #     ttnn.CoreRangeSet(
    #         [
    #             ttnn.CoreRange(
    #                 ttnn.CoreCoord(5, 7),
    #                 ttnn.CoreCoord(6, 7),
    #             ),
    #         ]
    #     ),
    #     ttnn.CoreRangeSet(
    #         [
    #             ttnn.CoreRange(
    #                 ttnn.CoreCoord(5, 9),
    #                 ttnn.CoreCoord(6, 9),
    #             ),
    #         ]
    #     ),
    # ]
    all_dram_cores = [
        ttnn.CoreCoord(0, 0),
        ttnn.CoreCoord(1, 0),
        ttnn.CoreCoord(2, 0),
        ttnn.CoreCoord(3, 0),
        ttnn.CoreCoord(4, 0),
        ttnn.CoreCoord(5, 0),
        ttnn.CoreCoord(6, 0),
        ttnn.CoreCoord(7, 0),
        ttnn.CoreCoord(8, 0),
        ttnn.CoreCoord(9, 0),
        ttnn.CoreCoord(10, 0),
        ttnn.CoreCoord(11, 0),
    ]
    all_sender_cores = [
        ttnn.CoreCoord(0, 9),
        ttnn.CoreCoord(0, 0),
        ttnn.CoreCoord(0, 4),
        ttnn.CoreCoord(0, 5),
        ttnn.CoreCoord(4, 0),
        ttnn.CoreCoord(4, 9),
        ttnn.CoreCoord(4, 1),
        ttnn.CoreCoord(4, 8),
        ttnn.CoreCoord(4, 3),
        ttnn.CoreCoord(4, 2),
        ttnn.CoreCoord(4, 4),
        ttnn.CoreCoord(4, 5),
    ]
    all_receiver_cores_list = [
        (1, 9),
        (2, 9),
        (1, 0),
        (2, 0),
        (1, 4),
        (2, 4),
        (1, 5),
        (2, 5),
        (5, 0),
        (6, 0),
        (5, 9),
        (6, 9),
        (5, 1),
        (6, 1),
        (5, 8),
        (6, 8),
        (5, 3),
        (6, 3),
        (5, 2),
        (6, 2),
        (5, 4),
        (6, 4),
        (5, 5),
        (6, 5),
    ]
    all_receiver_cores = [
        ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(
                    ttnn.CoreCoord(1, 9),
                    # ttnn.CoreCoord(1, 9),
                    ttnn.CoreCoord(2, 9),
                ),
            ]
        ),
        ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(
                    ttnn.CoreCoord(1, 0),
                    # ttnn.CoreCoord(1, 0),
                    ttnn.CoreCoord(2, 0),
                ),
            ]
        ),
        ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(
                    ttnn.CoreCoord(1, 4),
                    # ttnn.CoreCoord(1, 4),
                    ttnn.CoreCoord(2, 4),
                ),
            ]
        ),
        ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(
                    ttnn.CoreCoord(1, 5),
                    # ttnn.CoreCoord(1, 5),
                    ttnn.CoreCoord(2, 5),
                ),
            ]
        ),
        ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(
                    ttnn.CoreCoord(5, 0),
                    # ttnn.CoreCoord(5, 0),
                    ttnn.CoreCoord(6, 0),
                ),
            ]
        ),
        ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(
                    ttnn.CoreCoord(5, 9),
                    # ttnn.CoreCoord(5, 9),
                    ttnn.CoreCoord(6, 9),
                ),
            ]
        ),
        ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(
                    ttnn.CoreCoord(5, 1),
                    # ttnn.CoreCoord(5, 1),
                    ttnn.CoreCoord(6, 1),
                ),
            ]
        ),
        ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(
                    ttnn.CoreCoord(5, 8),
                    ttnn.CoreCoord(6, 8),
                ),
            ]
        ),
        ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(
                    ttnn.CoreCoord(5, 3),
                    ttnn.CoreCoord(6, 3),
                ),
            ]
        ),
        ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(
                    ttnn.CoreCoord(5, 2),
                    ttnn.CoreCoord(6, 2),
                ),
            ]
        ),
        ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(
                    ttnn.CoreCoord(5, 4),
                    ttnn.CoreCoord(6, 4),
                ),
            ]
        ),
        ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(
                    ttnn.CoreCoord(5, 5),
                    ttnn.CoreCoord(6, 5),
                ),
            ]
        ),
    ]

    worker_cores_range_set = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
            ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 9)),
        ]
    )

    dram_cores = all_dram_cores[:num_reader_cores]
    sender_cores = all_sender_cores[:num_reader_cores]
    receiver_cores_list = all_receiver_cores_list[: num_reader_cores * 2]
    # receiver_cores_list = all_receiver_cores_list[:num_reader_cores]
    receiver_cores = all_receiver_cores[:num_reader_cores]

    return dram_cores, sender_cores, receiver_cores_list, receiver_cores, worker_cores_range_set


@pytest.mark.parametrize(
    "num_reader_cores, num_tensors, input_shapes, num_layers",
    [  # TODO: test different shapes etc
        # (2, 2, [(256, 512), (256, 512)], 1),
        # (2, 2, [(1024, 256), (1024, 256)], 1),
        # (2, 2, [(128, 128), (128, 128)], 1),
        # (2, 2, [(256, 1024), (256, 1024)], 1),
        # (2, 2, [(512, 512), (512, 512)], 1),  # good PCC
        (12, 1, [(768, 768), (768, 384)], 1),  # bad PCC
        # (12, 1, [(2304, 768), (2304, 768)], 1),  # bad PCC
        #
        # (2, 2, [(192, 320), (192, 320)], 1),  # passes (1/2 banks)
        # (2, 2, [(384, 640), (384, 640)], 1),  # passes (2 banks)
        # (12, 2, [(1536, 768), (1536, 768)], 1),
        # (12, 1, [(2304, 768), (2304, 768)], 1),
        # (12, 1, [(2304, 3840), (2304, 3840)], 1),  # FF1/3 = 72 tiles x 120 tiles = 8640 tiles / 24 cores = 360 tiles per receiver core
        # (12, 2, [(7680, 2304), (7680, 2304)], 1),  # FF2
        # (12, 2, [(2304, 1536), (2304, 1536)], 1),  # QKV
        # (12, 2, [(2304, 2304), (2304, 2304)], 1),  # DO
        # (1, 2, [(512, 512)], 2), # Hangs
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
        # ttnn.bfloat8_b,
        # ttnn.bfloat4_b,
    ],
)
@pytest.mark.parametrize(
    "pcc_threshold",
    [
        0.999,
    ],
)
@pytest.mark.parametrize("device_params", [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}], indirect=True)
def test_run_prefetcher(
    device,
    num_tensors,
    input_shapes,
    num_layers,
    num_reader_cores,
    dtype,
    pcc_threshold,
    use_program_cache,
    function_level_defaults,
):
    logger.info(f"Running test_run_prefetcher with num_tensors={num_tensors}, input_shape={input_shapes[0]}")
    K, N = input_shapes[0]

    dram_cores, sender_cores, receiver_cores_list, receiver_cores, worker_cores_range_set = get_core_ranges(
        num_reader_cores
    )

    receiver_core_range_set = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(
                ttnn.CoreCoord(x, y),
                ttnn.CoreCoord(x, y),
            )
            for x, y in receiver_cores_list
        ]
    )

    print(f"sender_cores: {sender_cores}")
    print(f"receiver_cores: {receiver_cores}")
    print(f"receiver_cores_list: {receiver_cores_list}")

    sender_receiver_mapping = list(zip(sender_cores, receiver_cores))
    global_circular_buffer = ttnn.create_global_circular_buffer(device, sender_receiver_mapping, 2048 * (576))
    # global_circular_buffer = ttnn.create_global_circular_buffer(device, sender_receiver_mapping, 1088 * (360))
    # global_circular_buffer = ttnn.create_global_circular_buffer(device, sender_receiver_mapping, 576 * (360))

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
            dtype=dtype,
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

    ##### Setup up sub devices #####
    prefetcher_sub_device = ttnn.SubDevice([sender_core_range_set])
    worker_sub_device = ttnn.SubDevice([worker_cores_range_set])
    sub_device_manager = device.create_sub_device_manager([prefetcher_sub_device, worker_sub_device], 0)
    device.load_sub_device_manager(sub_device_manager)

    # Run the prefetcher
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
        # breakpoint()
        passing, output = comp_pcc(pt_out, tt_out, pcc_threshold)
        logger.info(output)

        all_passing = all_passing and passing

    # Run matmul
    for i in range(num_tensors):
        run_multi_core_matmul_1d(
            device,
            ttnn.bfloat16,
            dtype,
            ttnn.MathFidelity.HiFi4,
            False,
            True,
            True,
            B=1,
            M=32,
            K=K,
            N=N,
            activation=None,
            grid=receiver_cores_list,
            use_arbitrary_cores=True,
            num_iters=1,
            max_dst_tiles=8,
            pcc_threshold=0.98,
            mm_chain=False,
            use_physical_to_logical_mapping=False,
            global_cb=global_circular_buffer,
            prefetched_weights_pt=pt_tensors[0],
        )

    device.clear_loaded_sub_device_manager()
    device.remove_sub_device_manager(sub_device_manager)

    assert all_passing
