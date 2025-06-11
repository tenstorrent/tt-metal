# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
import ttnn
from models.utility_functions import is_wormhole_b0, is_grayskull, skip_for_wormhole_b0, is_blackhole
from models.utility_functions import torch2tt_tensor, tt2torch_tensor, pad_by_zero, roundup32
import torch
import itertools
from models.demos.llama3_subdevices.tt.model_config import LlamaOptimizations

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_equal,
    comp_pcc,
)
from models.demos.llama3_subdevices.tt.model_config import TtModelArgs
import random
import math
from models.utility_functions import is_wormhole_b0, is_grayskull, is_wormhole_b0, is_blackhole
from tracy import signpost

from models.demos.llama3_subdevices.tt.model_config import (
    PREFETCHER_NOC1_GRID,
)


from tests.ttnn.unit_tests.operations.ccl.test_new_all_reduce import (
    SUB_DEVICE_CRS,
    QKV_CRS,
    RING_CRS,
    FF1_CRS,
    FF1_CRS_RS_OUT,
    NORM_CRS,
)
from tracy import signpost

PACKET_WORKER_CRS = ttnn.CoreRangeSet(
    [
        ttnn.CoreRange(ttnn.CoreCoord(1, 1), ttnn.CoreCoord(3, 2)),
        ttnn.CoreRange(ttnn.CoreCoord(1, 3), ttnn.CoreCoord(2, 3)),
    ]
)


def gen_tensor(dim, shard_height, shard_width, num_devices_scatter, num_devices_fracture, num_cores, scheme="random"):
    torch.manual_seed(2005)
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


random.seed(10)


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
            physical_core = mesh_device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
            mapping[(physical_core.x, physical_core.y)] = (x, y)
    return mapping


def round_up(a, b):
    """
    Round up a to the nearest multiple of b
    """
    return b * math.ceil(a / b)


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

LM_HEAD_32_GRID = list(
    itertools.chain(
        itertools.product([1, 2, 3], range(10)),  # Generates (1,0)-(1,9), (2,0)-(2,9), (3,0)-(3,9)
        itertools.product([5], range(2)),  # Generates (5,0), (5,1)
    )
)


def run_multi_core_matmul_1d(
    mesh_device,
    in0_dtype,
    in1_dtype,
    fidelity,
    has_bias,
    fp32_acc_mode,
    packer_l1_acc,
    B,
    M,
    K,
    N,
    activation,
    grid,
    use_arbitrary_cores,
    num_iters,
    shard_height,
    shard_width,
    input_grid,
    output_grid,
    dtype,
    output_dtype=None,
    max_dst_tiles=8,
    pcc_threshold=0.98,
    use_physical_to_logical_mapping=True,
    hop_grid=None,
    in1_is_dram_interleaved=False,
    in1_is_in_dram=False,
    untilize_out=False,
    use_regular_grid=True,
    scheme="random",
    num_links=3,
):
    print("Making model args")
    model_args = TtModelArgs(
        mesh_device,
        instruct=True,
        max_batch_size=32,
        optimizations=LlamaOptimizations.performance,
        max_seq_len=128 * 1024,
        dummy_weights="random",
    )
    print("Model args made")
    model_args.n_layers = 1
    model_configuration = model_args.get_model_config()

    dim = 3
    shard_height = 32
    shard_width = 160
    num_devices_scatter = 4
    num_devices_fracture = 8
    num_cores = 24
    num_iters = 75
    warmup_iters = 10
    mesh_device.enable_program_cache()
    num_pages_per_packet = 4
    cyclic_buffer_size = 8
    compute_grid = (mesh_device.compute_with_storage_grid_size().x, mesh_device.compute_with_storage_grid_size().y)
    subdevice_shard_cores_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(1, 1), ttnn.CoreCoord(3, 2)),
            ttnn.CoreRange(ttnn.CoreCoord(1, 3), ttnn.CoreCoord(2, 3)),
        ]
    )
    packet_workers_persistent_mem_config = ttnn.create_sharded_memory_config(
        shape=(32, 512),
        core_grid=PACKET_WORKER_CRS,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            FF1_CRS_RS_OUT,
            [32, 32],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    assert not has_bias, "Bias not supported for gather_in0 mode."
    if not isinstance(grid, tuple) and not use_arbitrary_cores:
        pytest.skip("Grid is not a tuple and not using arbitrary cores")

    if output_dtype is None:
        output_dtype = in0_dtype

    in0_shape = [1, B, M, K]
    in1_shape = [1, 1, K, N]
    num_cores = grid[0] * grid[1] if isinstance(grid, tuple) else len(grid)

    storage_grid = num_cores_to_rectangle_grid(num_cores, mesh_device)
    if storage_grid is None:
        pytest.skip(f"Could not find a rectangle grid for num_cores: {num_cores}")

    M *= B  # Fuse batch always enabled

    K_per_shard = round_up(math.ceil(K / num_cores), ttnn.TILE_SIZE)
    K_padded = K_per_shard * num_cores
    N_per_shard = round_up(math.ceil(N / num_cores), ttnn.TILE_SIZE)
    N_per_shard_in_dram = N_per_shard * 2
    N_padded = N_per_shard * num_cores

    logger.info(f"K_per_shard {K_per_shard}")
    logger.info(f"K_padded {K_padded}")
    logger.info(f"N_per_shard {N_per_shard}")
    logger.info(f"N_padded {N_padded}")

    in0_block_h = M // ttnn.TILE_SIZE
    in0_block_w = K // num_cores // ttnn.TILE_SIZE
    while (K / ttnn.TILE_SIZE) % in0_block_w != 0:
        in0_block_w -= 1

    out_block_h = M // ttnn.TILE_SIZE
    out_block_w = N_padded // num_cores // ttnn.TILE_SIZE

    num_blocks_y = (M // ttnn.TILE_SIZE - 1) // out_block_h + 1
    num_blocks_x = (N_padded // ttnn.TILE_SIZE - 1) // out_block_w + 1
    num_blocks_total = num_blocks_y * num_blocks_x

    if num_blocks_total != num_cores:
        pytest.skip(f"num_blocks_total {num_blocks_total} != num_cores {num_cores}")

    out_subblock_h = 1
    out_subblock_w = max_dst_tiles
    while out_block_w % out_subblock_w != 0:
        out_subblock_w -= 1

    logger.debug("in0 block h w " + str(in0_block_h) + " " + str(in0_block_w))
    logger.debug("in1 block h w " + str(in0_block_w) + " " + str(out_block_w))
    logger.debug("out block h w " + str(out_block_h) + " " + str(out_block_w))
    logger.debug("out subblock h w " + str(out_subblock_h) + " " + str(out_subblock_w))

    if use_arbitrary_cores:
        # x, y
        if isinstance(grid, tuple):  # Generate random grid
            CORE_RANGE = [(x, y) for y in range(storage_grid[1]) for x in range(storage_grid[0])]
            random.shuffle(CORE_RANGE)
        else:  # Use custom grid
            if use_physical_to_logical_mapping:
                mapping = get_physical_to_logical_core_mapping(mesh_device)
                CORE_RANGE = [mapping[physical_coord] for physical_coord in grid]
            else:
                CORE_RANGE = grid

        core_range_set = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(
                    ttnn.CoreCoord(x, y),
                    ttnn.CoreCoord(x, y),
                )
                for x, y in CORE_RANGE
            ]
        )
    else:
        core_range_set = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(storage_grid[0] - 1, storage_grid[1] - 1),
                ),
            }
        )

    hop_core_range_set = ttnn.CoreRangeSet([])
    if hop_grid is not None:
        hop_core_range_set = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(x, y),
                    ttnn.CoreCoord(x, y),
                )
                for x, y in hop_grid
            }
        )

    in0_sharded_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            core_range_set,
            [M, K_per_shard],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )

    if in1_is_in_dram:
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
    else:
        in1_sharded_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                core_range_set,
                [K, N_per_shard],
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
            dram_sharded_output_core_range_set if (in1_is_in_dram and not in1_is_dram_interleaved) else core_range_set,
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
    in1_t = ttnn.from_torch(
        in1,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=in1_dtype,
        memory_config=in1_sharded_mem_config,
    )

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

    if is_grayskull():
        compute_kernel_config = ttnn.GrayskullComputeKernelConfig(
            math_fidelity=fidelity,
            math_approx_mode=True,
        )
    else:
        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=fidelity,
            math_approx_mode=True,
            fp32_dest_acc_en=fp32_acc_mode,
            packer_l1_acc=packer_l1_acc,
            dst_full_sync_en=True,
        )
    rs_input_mem_config = model_configuration["SHARDED_FF12_OUT_RING_MEMCFG"]
    input = gen_tensor(
        dim,
        shard_height,
        shard_width,
        num_devices_scatter,
        num_devices_fracture,
        rs_input_mem_config.shard_spec.num_cores(),
        scheme=scheme,
    )
    intermediate_outputs = torch.chunk(input, chunks=num_devices_scatter, dim=dim)
    output = torch.zeros(intermediate_outputs[0].shape)
    for i in range(0, len(intermediate_outputs)):
        output += intermediate_outputs[i]

    scattered_output = torch.chunk(output, chunks=num_devices_scatter, dim=dim)
    scattered_output = torch.cat(scattered_output, dim=dim)
    cluster_shape = (8, 4)
    tt_input = ttnn.from_torch(
        input,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        memory_config=rs_input_mem_config,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=cluster_shape),
    )
    buffer_mem_cfg = model_configuration["REDUCE_SCATTER_INTERIM_MEMCFG"]
    tt_intermediate = ttnn.from_torch(
        torch.zeros((*cluster_shape, 32, 512 * buffer_mem_cfg.shard_spec.num_cores())),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        memory_config=buffer_mem_cfg,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=cluster_shape),
    )
    ccl_sub_device_crs = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(6, 9)),
        ]
    )
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
    ccl_semaphore_handle = ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0)
    worker_cores_range_set = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
            ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 9)),
        ]
    )

    worker_sub_device = ttnn.SubDevice([worker_cores_range_set])
    worker_sub_device_id = ttnn.SubDeviceId(0)
    signpost("start")
    for _ in range(num_iters):
        rs_out, matmul_out = ttnn.experimental.llama_rs_matmul(
            in0_t,
            in1_t,
            tt_input,
            tt_intermediate,
            dim,
            ccl_semaphore_handle,
            1,
            mesh_device,
            num_links,
            worker_sub_device_id,
            program_config=program_config,
            memory_config_mm=output_sharded_mem_config,
            compute_kernel_config=compute_kernel_config,
            dtype=output_dtype,
            memory_config_rs=model_configuration["REDUCE_SCATTER_OUT_MEMCFG"],
            topology=ttnn.Topology.Linear,
        )


@pytest.mark.skipif(is_grayskull(), reason="Test suite for WH only")
@pytest.mark.skipif(is_blackhole(), reason="Test suite for WH only")
@pytest.mark.parametrize("has_bias", [False], ids=["no_bias"])
@pytest.mark.parametrize(
    "B, M, K, N, in0_dtype, in1_dtype, output_dtype, fidelity, packer_l1_acc, fp32_acc_mode, grid, in1_is_dram_interleaved, untilize_out",
    [
        (
            1,
            32,
            2048,
            3584,
            ttnn.bfloat16,
            ttnn.bfloat4_b,
            ttnn.bfloat8_b,
            ttnn.MathFidelity.LoFi,
            True,
            False,
            PREFETCHER_NOC1_GRID,
            False,
            False,
        ),
    ],
    ids=[
        "ff13",
    ],
)
@pytest.mark.parametrize(
    "num_iters",
    [50],
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "trace_region_size": 300000,
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("shard_height", [32])
@pytest.mark.parametrize("shard_width", [64])
@pytest.mark.parametrize("input_grid", [(5, 5)])
@pytest.mark.parametrize("output_grid", [(5, 1)])
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b])
@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),  # TODO: Once fabric can be initialized on a SubMesh, revert to (1, 4)
    ],
    indirect=True,
)
def test_matmul_1d_ring_llama_with_rs_perf(
    mesh_device,
    in0_dtype,
    in1_dtype,
    output_dtype,
    fidelity,
    has_bias,
    fp32_acc_mode,
    packer_l1_acc,
    B,
    M,
    K,
    N,
    grid,
    in1_is_dram_interleaved,
    untilize_out,
    num_iters,
    use_program_cache,
    function_level_defaults,
    shard_height,
    shard_width,
    input_grid,
    output_grid,
    dtype,
):
    # Only run these tests on unharvested TG
    device_grid = (mesh_device.compute_with_storage_grid_size().x, mesh_device.compute_with_storage_grid_size().y)
    if device_grid != (7, 10):
        pytest.skip("Skipping test_run_prefetcher because it only works with a 7x10 grid")

    if in1_is_dram_interleaved:
        hop_grid = None
    else:
        hop_grid = [
            (3, 6),
        ]

    run_multi_core_matmul_1d(
        mesh_device,
        in0_dtype,
        in1_dtype,
        fidelity,
        has_bias,
        fp32_acc_mode,
        packer_l1_acc,
        B,
        M,
        K,
        N,
        None,  # activation,
        grid,
        True,
        num_iters,
        shard_height,
        shard_width,
        input_grid,
        output_grid,
        dtype,
        output_dtype=output_dtype,
        use_physical_to_logical_mapping=False,
        hop_grid=hop_grid,
        in1_is_dram_interleaved=in1_is_dram_interleaved,
        untilize_out=untilize_out,
        use_regular_grid=True,
    )
