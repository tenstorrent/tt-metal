# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
import random
import math
from models.utility_functions import is_wormhole_b0, is_grayskull, skip_for_wormhole_b0, is_blackhole
from tracy import signpost
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from tests.ttnn.utils_for_testing import assert_with_pcc

from models.demos.llama3_subdevices.tt.model_config import (
    PREFETCHER_NOC1_GRID,
)

from models.demos.llama3_subdevices.tt.model_config import TtModelArgs

from models.demos.llama3_subdevices.tt.prefetcher_common import TtLlamaPrefetcherSetup

from models.utility_functions import skip_for_grayskull
from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler

# physical coords
PREFETCHER_GRID = [
    (8, 11),
    (8, 9),
    (8, 8),
    (8, 7),
    (8, 5),
    (8, 3),
    (8, 2),
    (8, 1),
    (7, 1),
    (7, 2),
    (7, 3),
    (7, 5),
    (7, 7),
    (7, 8),
    (7, 9),
    (7, 11),
    (3, 11),
    (3, 7),
    (3, 5),
    (3, 1),
    (2, 1),
    (2, 5),
    (2, 7),
    (2, 11),
]

# dram sharded MM output logical coords
PREFETCHER_NOC1_OUTPUT_GRID = [
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
    (5, 7),
    (6, 7),
    (5, 6),
    (6, 6),
    (5, 2),
    (6, 2),
    (5, 4),
    (6, 4),
    (5, 5),
    (6, 5),
]


def num_cores_to_rectangle_grid(num_cores, mesh_device):
    """
    Find a rectangular core grid size, given an number of cores.

    Return None if rectangle grid is not possible.
    """
    x = mesh_device.compute_with_storage_grid_size().x
    while x > 0 and num_cores % x != 0:
        x -= 1

    if x == 0:
        return None

    y = num_cores // x
    return (x, y)


def get_physical_to_logical_core_mapping(mesh_device):
    """
    Get a mapping from physical core coords to logical core coords

    Returns a dictionary.
    """
    mapping = {}
    grid = mesh_device.compute_with_storage_grid_size()
    for x in range(grid.x):
        for y in range(grid.y):
            physical_core = mesh_device.get_devices()[0].worker_core_from_logical_core(ttnn.CoreCoord(x, y))
            mapping[(physical_core.x, physical_core.y)] = (x, y)
    return mapping


def round_up(a, b):
    """
    Round up a to the nearest multiple of b
    """
    return b * math.ceil(a / b)


from tests.ttnn.unit_tests.operations.ccl.test_new_all_reduce import (
    SUB_DEVICE_CRS,
    QKV_CRS,
    RING_CRS,
    FF1_CRS,
    FF1_CRS_RS_OUT,
    NORM_CRS,
)

PACKET_WORKER_CRS = ttnn.CoreRangeSet(
    [
        ttnn.CoreRange(ttnn.CoreCoord(1, 1), ttnn.CoreCoord(3, 2)),
        ttnn.CoreRange(ttnn.CoreCoord(1, 3), ttnn.CoreCoord(2, 3)),
    ]
)
from tracy import signpost

PACKET_WORKER_CRS = ttnn.CoreRangeSet(
    [
        ttnn.CoreRange(ttnn.CoreCoord(1, 1), ttnn.CoreCoord(3, 2)),
        ttnn.CoreRange(ttnn.CoreCoord(1, 3), ttnn.CoreCoord(2, 3)),
    ]
)
MATMUL_CRS = ttnn.CoreRangeSet(
    [
        ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(2, 0)),
        ttnn.CoreRange(ttnn.CoreCoord(1, 4), ttnn.CoreCoord(2, 5)),
        ttnn.CoreRange(ttnn.CoreCoord(1, 9), ttnn.CoreCoord(2, 9)),
        ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 2)),
        ttnn.CoreRange(ttnn.CoreCoord(5, 4), ttnn.CoreCoord(6, 7)),
        ttnn.CoreRange(ttnn.CoreCoord(5, 9), ttnn.CoreCoord(6, 9)),
    ]
)


def gen_tensor(dim, shard_height, shard_width, num_devices_scatter, num_devices_fracture, num_cores, scheme="random"):
    factor = 0
    torch_fracture_tensors = []
    for _ in range(num_devices_fracture):
        torch_scatter_tensors = []
        for _ in range(num_devices_scatter):
            torch_input_tensors = []
            for _ in range(num_cores):
                for _ in range(shard_width // 32):
                    if scheme == "random":
                        torch_input_tensors.append(torch.rand(1, 1, shard_height, 32))
                    elif scheme == "sequential":
                        torch_input_tensors.append(torch.ones(1, 1, shard_height, 32) * factor)
                        factor += 1
                    else:
                        raise ValueError(f"Invalid scheme: {scheme}")
            torch_scatter_tensors.append(torch.cat(torch_input_tensors, dim=dim))

        torch_fracture_tensors.append(torch.cat(torch_scatter_tensors, dim=1))

    return torch.cat(torch_fracture_tensors, dim=0)


def run_reduce_scatter_test(
    mesh_device,
    dim,
    shard_height,
    shard_width,
    num_devices_scatter,
    num_devices_fracture,
    num_cores,
    num_iters,
    warmup_iters,
    trace_mode,
    B,
    M,
    K,
    N,
    in0_dtype,
    in1_dtype,
    output_dtype,
    fidelity,
    packer_l1_acc,
    fp32_acc_mode,
    grid,
    in1_is_dram_interleaved,
    in1_is_in_dram,
    num_links=3,
    scheme="random",
    use_regular_grid=False,
    input_grid=None,
    output_grid=None,
    dtype=ttnn.bfloat8_b,
    profiler=BenchmarkProfiler(),
    hop_grid=None,
    max_dst_tiles=8,
    activation=None,
    pcc_threshold=0.98,
    use_physical_to_logical_mapping=True,
    untilize_out=False,
):
    use_arbitrary_cores = True
    mesh_device.enable_program_cache()
    num_pages_per_packet = 4
    cyclic_buffer_size = 8
    if not isinstance(grid, tuple) and not use_arbitrary_cores:
        pytest.skip("Grid is not a tuple and not using arbitrary cores")
    if output_dtype is None:
        output_dtype = in0_dtype
    in0_shape = [1, B, M, K]
    in1_shape = [1, 1, K, N]
    num_cores_mm = grid[0] * grid[1] if isinstance(grid, tuple) else len(grid)
    storage_grid = num_cores_to_rectangle_grid(num_cores, mesh_device)
    if storage_grid is None:
        pytest.skip(f"Could not find a rectangle grid for num_cores: {num_cores}")
    M *= B  # Fuse batch always enabled
    K_per_shard = round_up(math.ceil(K / num_cores), ttnn.TILE_SIZE)
    K_padded = K_per_shard * num_cores
    N_per_shard = round_up(math.ceil(N / num_cores), ttnn.TILE_SIZE)
    N_per_shard_in_dram = N_per_shard * 2
    N_padded = N_per_shard * num_cores
    in0_block_h = M // ttnn.TILE_SIZE
    in0_block_w = K // num_cores // ttnn.TILE_SIZE
    while (K / ttnn.TILE_SIZE) % in0_block_w != 0:
        in0_block_w -= 1
    out_block_h = M // ttnn.TILE_SIZE
    out_block_w = N_padded // num_cores // ttnn.TILE_SIZE
    num_blocks_y = (M // ttnn.TILE_SIZE - 1) // out_block_h + 1
    num_blocks_x = (N_padded // ttnn.TILE_SIZE - 1) // out_block_w + 1
    num_blocks_total = num_blocks_y * num_blocks_x

    out_subblock_h = 1
    out_subblock_w = max_dst_tiles
    while out_block_w % out_subblock_w != 0:
        out_subblock_w -= 1
    hop_core_range_set = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(3, 6), ttnn.CoreCoord(3, 6))])

    in0_sharded_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            MATMUL_CRS,
            [M, K_per_shard],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    if in1_is_dram_interleaved:
        in1_sharded_mem_config = ttnn.DRAM_MEMORY_CONFIG
    else:
        in1_shard_grid = ttnn.CoreCoord(mesh_device.dram_grid_size().x - 1, mesh_device.dram_grid_size().y - 1)
        in1_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), in1_shard_grid)})
        in1_sharded_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(
                in1_shard_grid,
                [K, N_per_shard_in_dram],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
    dram_sharded_output_core_range_set = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(
                ttnn.CoreCoord(x, y),
                ttnn.CoreCoord(x, y),
            )
            for x, y in PREFETCHER_NOC1_OUTPUT_GRID
        ]
    )
    output_sharded_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            dram_sharded_output_core_range_set,
            [M, N_per_shard],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    in0 = torch.randn(in0_shape)
    in1 = torch.randn(in1_shape)

    in0_t = ttnn.from_torch(
        in0,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=in0_dtype,
        memory_config=in0_sharded_mem_config,
    )
    print("In 0 mem config is")
    print(in0_sharded_mem_config)
    in1_t = ttnn.from_torch(
        in1,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=in1_dtype,
        memory_config=in1_sharded_mem_config,
    )
    print("In 1 mem config is")
    print(in1_sharded_mem_config)
    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=storage_grid,
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=out_block_h,
        per_core_N=out_block_w,
        fuse_batch=True,
        fused_activation=activation,
        mcast_in0=False,
        gather_in0=True,
        hop_cores=hop_core_range_set,
        untilize_out=untilize_out,
    )
    print("program_config is")
    print(program_config)
    print("With storage grid")
    print(storage_grid)
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=fidelity,
        math_approx_mode=True,
        fp32_dest_acc_en=fp32_acc_mode,
        packer_l1_acc=packer_l1_acc,
        dst_full_sync_en=True,
    )

    # input, output, interm core range set
    compute_grid = (mesh_device.compute_with_storage_grid_size().x, mesh_device.compute_with_storage_grid_size().y)
    subdevice_shard_cores_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(compute_grid[0] - 1, compute_grid[1] - 1),
            ),
        }
    )
    if input_grid is not None:
        input_shard_cores_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(input_grid[0] - 1, input_grid[1] - 1),
                ),
            }
        )
    if output_grid is not None:
        output_shard_cores_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(output_grid[0] - 1, output_grid[1] - 1),
                ),
            }
        )
        tensor_width_in_tiles = num_cores * shard_width
        output_num_cores = output_grid[0] * output_grid[1]

    # input, output, interm memory config
    sharded_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            input_shard_cores_grid if use_regular_grid else RING_CRS,
            [shard_height, shard_width],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    packet_workers_persistent_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            PACKET_WORKER_CRS,
            [shard_height, num_devices_scatter * num_pages_per_packet * 32],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            output_shard_cores_grid if use_regular_grid else FF1_CRS_RS_OUT,
            [
                shard_height,
                tensor_width_in_tiles // output_num_cores // num_devices_scatter if use_regular_grid else 32,
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )

    output_tensor_goldens_list = []
    tt_input_tensors_list = []
    tt_intermediate_tensors_list = []
    for iter in range(num_iters):
        input = gen_tensor(
            dim, shard_height, shard_width, num_devices_scatter, num_devices_fracture, num_cores, scheme=scheme
        )

        intermediate_tensor = torch.zeros(
            [
                num_devices_fracture,
                num_devices_scatter,
                shard_height,
                num_devices_scatter
                * num_pages_per_packet
                * 32
                * packet_workers_persistent_mem_config.shard_spec.num_cores(),
            ]
        )

        intermediate_outputs = torch.chunk(input, chunks=num_devices_scatter, dim=1)
        output = torch.zeros(intermediate_outputs[0].shape)

        for i in range(0, len(intermediate_outputs)):
            output += intermediate_outputs[i]

        scattered_output = torch.chunk(output, chunks=num_devices_scatter, dim=dim)
        scattered_output = torch.cat(scattered_output, dim=1)

        output_tensor_goldens_list.append(scattered_output)

        tt_input = ttnn.from_torch(
            input,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            memory_config=sharded_mem_config,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device, dims=(0, 1), mesh_shape=[num_devices_fracture, num_devices_scatter]
            ),
        )
        if iter < cyclic_buffer_size:
            tt_intermediate = ttnn.from_torch(
                intermediate_tensor,
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=dtype,
                memory_config=packet_workers_persistent_mem_config,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    mesh_device, dims=(0, 1), mesh_shape=[num_devices_fracture, num_devices_scatter]
                ),
            )
            tt_intermediate_tensors_list.append(tt_intermediate)
        tt_input_tensors_list.append(tt_input)

    ccl_sub_device_crs = subdevice_shard_cores_grid if use_regular_grid is not None else SUB_DEVICE_CRS
    worker_sub_device = ttnn.SubDevice(
        [
            ccl_sub_device_crs,
        ]
    )
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_stall_group = [worker_sub_device_id]
    sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
    mesh_device.load_sub_device_manager(sub_device_manager)
    mesh_device.set_sub_device_stall_group(sub_device_stall_group)

    # create global semaphore handles
    ccl_semaphore_handles = [ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0) for _ in range(num_iters)]

    tt_out_tensor_list = []
    model_args = TtModelArgs(mesh_device, max_batch_size=32, max_seq_len=256, dummy_weights=True)
    prefetcher_setup = TtLlamaPrefetcherSetup(
        mesh_device,
        n_tensors=5,
        n_layers=80,
    )
    prefetcher_setup.create_global_cb()

    def run_op(n_iters, store_all_results=True):
        tt_output_list = []
        for i in range(n_iters):
            buffer_index = 0 if trace_mode else i
            _, tt_out_tensor = ttnn.experimental.rs_matmul(
                in0_t,
                in1_t,
                tt_input_tensors_list[buffer_index],
                tt_intermediate_tensors_list[buffer_index % cyclic_buffer_size],
                dim,
                ccl_semaphore_handles[buffer_index],
                1,
                mesh_device,
                num_links,
                global_cb=prefetcher_setup.global_circular_buffer,
                memory_config_rs=output_mem_config,
                memory_config_mm=output_sharded_mem_config,
                program_config=program_config,
                compute_kernel_config=compute_kernel_config,
                dtype=output_dtype,
                topology=ttnn.Topology.Linear,
            )
            if not trace_mode:
                ttnn.synchronize_device(mesh_device)
            if store_all_results:
                tt_output_list.append(tt_out_tensor)
        if store_all_results:
            return tt_output_list
        else:
            return [tt_out_tensor]

    signpost("start")
    tt_out_tensor_list = run_op(num_iters, store_all_results=True)
    signpost("stop")

    mesh_device.reset_sub_device_stall_group()

    passed = True
    first_failed_tensor_index = None
    failed_indices = []
    expected_pcc = 0.999 if dtype == ttnn.bfloat8_b else 0.9999
    for tensor_index in range(len(tt_out_tensor_list)):
        tt_torch_tensor = ttnn.to_torch(
            tt_out_tensor_list[tensor_index],
            mesh_composer=ttnn.ConcatMesh2dToTensor(
                mesh_device, mesh_shape=[num_devices_fracture, num_devices_scatter], dims=(0, 1)
            ),
        )
        eq, output_results = comp_pcc(tt_torch_tensor, output_tensor_goldens_list[tensor_index], expected_pcc)
        logger.info(f"Output tensor {tensor_index} has result {output_results}")
        if not eq:
            passed = False
            first_failed_tensor_index = tensor_index
            failed_indices = torch.where(tt_torch_tensor != output_tensor_goldens_list[tensor_index])
            break

    logger.info(f"Device has {mesh_device.num_program_cache_entries()} program cache entries")
    assert (
        mesh_device.num_program_cache_entries() == 1 or mesh_device.num_program_cache_entries() == num_iters
    ), f"Device {i} has {mesh_device.num_program_cache_entries()} program cache entries"

    if not passed:
        logger.info(f"Failed indices: {failed_indices}")
        assert eq, f"{first_failed_tensor_index} FAILED: {output_results}"


@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
@pytest.mark.parametrize("trace_mode", [False])
@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "B, M, K, N, in0_dtype, in1_dtype, output_dtype, fidelity, packer_l1_acc, fp32_acc_mode, grid, in1_is_dram_interleaved, in1_is_in_dram",
    [
        (
            1,
            32,
            2048,
            16 * 1024,
            ttnn.bfloat8_b,
            ttnn.bfloat8_b,
            ttnn.bfloat8_b,
            ttnn.MathFidelity.HiFi2,
            True,
            True,
            PREFETCHER_NOC1_GRID,
            False,
            True,
        ),
    ],
    ids=[
        "lm_head",
    ],
)
def test_fabric_reduce_scatter_tg_no_trace(
    mesh_device,
    trace_mode,
    B,
    M,
    K,
    N,
    in0_dtype,
    in1_dtype,
    output_dtype,
    fidelity,
    packer_l1_acc,
    fp32_acc_mode,
    grid,
    in1_is_dram_interleaved,
    in1_is_in_dram,
):
    # Only run these tests on unharvested TG
    device_grid = (mesh_device.compute_with_storage_grid_size().x, mesh_device.compute_with_storage_grid_size().y)
    if device_grid != (7, 10):
        pytest.skip("Not TG!")

    dim = 3
    shard_height = 32
    shard_width = 160
    num_devices_scatter = 4
    num_devices_fracture = 8
    num_cores = 24
    num_iters = 5
    warmup_iters = 0
    trace_mode = trace_mode
    if in1_is_dram_interleaved:
        hop_grid = None
    else:
        hop_grid = [
            (3, 6),
        ]

    run_reduce_scatter_test(
        mesh_device,
        dim,
        shard_height,
        shard_width,
        num_devices_scatter,
        num_devices_fracture,
        num_cores,
        num_iters,
        warmup_iters,
        trace_mode,
        B,
        M,
        K,
        N,
        in0_dtype,
        in1_dtype,
        output_dtype,
        fidelity,
        packer_l1_acc,
        fp32_acc_mode,
        grid,
        in1_is_dram_interleaved,
        in1_is_in_dram,
        num_links=3,
        scheme="random",
        use_physical_to_logical_mapping=False,
        hop_grid=hop_grid,
    )
