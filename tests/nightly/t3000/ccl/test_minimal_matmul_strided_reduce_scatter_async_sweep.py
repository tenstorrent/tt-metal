# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from dataclasses import dataclass
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from models.common.utility_functions import skip_for_blackhole

import tracy
import os


@dataclass
class MinimalMatmulStridedReduceScatterTestConfig:
    """Test configuration for fused matmul + strided reduce scatter.

    Using a dataclass ensures parameters can't be provided in wrong order
    and makes test cases self-documenting.
    """

    M: int
    K: int
    N: int
    dim: int
    mm_block_m: int
    mm_block_k: int
    mm_block_n: int
    mm_core_grid: object  # ttnn.CoreCoord
    chunk_width_in_mm_blocks: int
    subblock_h: int = 1
    subblock_w: int = 1
    layout: object = None  # ttnn.Layout, set in __post_init__
    input_dtype: object = None  # ttnn.DataType, set in __post_init__
    num_workers_per_link: object = None  # Optional[int]

    def __post_init__(self):
        if self.layout is None:
            self.layout = ttnn.TILE_LAYOUT
        if self.input_dtype is None:
            self.input_dtype = ttnn.bfloat16


def create_global_semaphores(mesh_device, cores, initial_value):
    """Create 3 global semaphores needed by strided reduce-scatter."""
    ccl_semaphore_handles = [ttnn.create_global_semaphore(mesh_device, cores, initial_value) for _ in range(3)]
    return ccl_semaphore_handles


def run_minimal_matmul_strided_reduce_scatter_impl(
    mesh_device,
    M,
    K,
    N,
    dim,
    num_links,
    input_dtype,
    layout,
    mem_config_input,
    mem_config_mm,
    mem_config_rs,
    topology,
    mm_block_m,
    mm_block_k,
    mm_block_n,
    subblock_h,
    subblock_w,
    mm_core_grid,
    num_iters=1,
    enable_trace=False,
    cluster_axis=1,
    num_workers_per_link=None,
    num_buffers_per_channel=None,
    chunk_width_in_mm_blocks=None,
    rs_mode="fused",
    use_barrier=True,
    math_fidelity=ttnn.MathFidelity.HiFi2,
    fp32_acc=True,
    rs_core_grid_offset=None,
    allowed_pcc=0.99,
    sweep_key=None,
):
    torch.manual_seed(0)

    TILE_SIZE = 32

    num_devices = mesh_device.shape[cluster_axis]

    # Default RS core grid offset: place RS cores below MM cores
    if rs_core_grid_offset is None:
        rs_core_grid_offset = ttnn.CoreCoord(0, mm_core_grid.y)

    ##### Fabric / sub-device setup (matching standalone RS test) #####
    compute_grid_size = mesh_device.compute_with_storage_grid_size()
    all_cores = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
    )
    worker_sub_device_id = ttnn.SubDeviceId(0)
    mesh_device.set_sub_device_stall_group([worker_sub_device_id])

    # RS needs 3 semaphores per iteration
    ccl_semaphore_handles = [create_global_semaphores(mesh_device, all_cores, 0) for _ in range(num_iters)]
    barrier_semaphore_handles = [ttnn.create_global_semaphore(mesh_device, all_cores, 0) for _ in range(num_iters)]

    ##### Input setup #####
    # Input (activations): replicated on all devices (same activations on every device)
    # Weight: unique per device. Shape [num_devices, 1, K, N] where dim 0 is the device-shard
    #   dimension (not batch) — ShardTensor2dMesh splits it so each device gets [1, 1, K, N].
    #   This makes each device compute a different MM output, giving the reduce-scatter real work.
    input_shape = [1, 1, M, K]
    weight_shape_global = [num_devices, 1, K, N]  # dim 0 is device-shard dimension

    # logger.info(f"Input shape per device: {input_shape}")
    # logger.info(f"Weight shape per device: [1, 1, {K}, {N}]")
    # logger.info(f"MM output shape per device: [1, 1, {M}, {N}]")
    # logger.info(f"RS scatter dim: {dim}, ring_size: {num_devices}")
    # logger.info(f"RS output shape per device: [1, 1, {M}, {N // num_devices}]")

    input_tensor_mesh_list = []
    weight_tensor_mesh_list = []
    torch_mm_output_per_device_list = []  # list of lists, [iter][device]
    torch_rs_output_list = []
    shard_dims = [None, None]
    shard_dims[cluster_axis] = 0

    for i in range(num_iters):
        torch_input = torch.randn(input_shape, dtype=torch.float32)
        torch_weight_global = torch.randn(weight_shape_global, dtype=torch.float32)

        # Golden: per-device MM outputs (each device has different weights)
        torch_weight_chunks = torch.chunk(torch_weight_global, num_devices, dim=0)  # each [1, 1, K, N]
        mm_outputs = []
        for d in range(num_devices):
            mm_out_d = torch.matmul(torch_input, torch_weight_chunks[d])
            mm_outputs.append(mm_out_d)
        torch_mm_output_per_device_list.append(mm_outputs)

        # Golden: RS reduce (sum across devices) then scatter
        torch_rs_reduced = torch.sum(torch.stack(mm_outputs), dim=0)  # [1, 1, M, N]
        torch_rs_scattered = torch.chunk(torch_rs_reduced, num_devices, dim=dim)
        torch_rs_output_list.append(torch_rs_scattered)

        # Create device tensors
        # Input: replicated (same on all devices)
        input_tensor_mesh = ttnn.from_torch(
            torch_input,
            device=mesh_device,
            layout=layout,
            dtype=input_dtype,
            memory_config=mem_config_input,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        # Weight: dim 0 sharded across devices so each device gets unique [1, 1, K, N] weights
        weight_tensor_mesh = ttnn.from_torch(
            torch_weight_global,
            device=mesh_device,
            layout=layout,
            dtype=input_dtype,
            memory_config=mem_config_input,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=shard_dims, mesh_shape=tuple(mesh_device.shape)),
        )

        input_tensor_mesh_list.append(input_tensor_mesh)
        weight_tensor_mesh_list.append(weight_tensor_mesh)

    ##### Compute config #####
    compute_config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=math_fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_acc,
        packer_l1_acc=True,
    )
    matmul_config = ttnn.MinimalMatmulConfig(
        M_block_size=mm_block_m // TILE_SIZE,
        K_block_size=mm_block_k // TILE_SIZE,
        N_block_size=mm_block_n // TILE_SIZE,
        subblock_h=subblock_h,
        subblock_w=subblock_w,
        compute_with_storage_grid_size=mm_core_grid,
    )

    ##### Run the op #####
    # Auto-compute num_workers_per_link from available RS cores if not explicitly set.
    # This fills the remaining rows below the MM grid (RS zone) with bidirectional workers.
    if num_workers_per_link is None:
        rs_zone_capacity = (compute_grid_size.y - mm_core_grid.y) * compute_grid_size.x
        num_workers_per_link = rs_zone_capacity // (2 * num_links) - 1

    # NOTE: This sweep function does not perform correctness verification (no comp_pcc).
    # It is intended for performance measurement only. Use test_minimal_matmul_strided_reduce_scatter_async
    # for functional correctness checks.
    def run_op(i):
        (
            tt_mm_out,
            tt_rs_out,
        ) = ttnn.experimental.minimal_matmul_strided_reduce_scatter_async(
            input_tensor_mesh_list[i],
            weight_tensor_mesh_list[i],
            dim,
            ccl_semaphore_handles[i],
            rs_core_grid_offset,
            num_links=num_links,
            memory_config_mm=mem_config_mm,
            rs_output_mem_config=mem_config_rs,
            topology=topology,
            cluster_axis=cluster_axis,
            config=matmul_config,
            compute_kernel_config=compute_config,
            barrier_semaphore=barrier_semaphore_handles[i] if use_barrier else None,
            num_workers_per_link=num_workers_per_link,
            num_buffers_per_channel=num_buffers_per_channel,
            chunk_width_in_mm_blocks=chunk_width_in_mm_blocks,
        )
        return tt_mm_out, tt_rs_out

    for i in range(num_iters):
        ttnn.synchronize_device(mesh_device)
        tracy.signpost(f"{sweep_key}-start")
        tt_mm_out, tt_rs_out = run_op(i)
        ttnn.synchronize_device(mesh_device)
        tracy.signpost(f"{sweep_key}-end")


def write_error_to_file(error):
    with open("test_minimal_matmul_strided_reduce_scatter_async_sweep_errors.log", "a") as ef:
        ef.write(error + "\n")
        ef.flush()


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True, ids=["4x8"])
@pytest.mark.parametrize("num_links", [2], ids=["2link"])
@pytest.mark.parametrize("cluster_axis", [0], ids=["axis_0"])  # Wan is currently TP 0
@pytest.mark.parametrize(
    "test_config",
    [
        pytest.param(
            MinimalMatmulStridedReduceScatterTestConfig(
                M=9472 // 4,  # Sweep for quad
                K=3456,
                N=5120,
                dim=3,
                mm_block_m=256,
                mm_block_k=128,
                mm_block_n=256,
                mm_core_grid=ttnn.CoreCoord(12, 8),
                chunk_width_in_mm_blocks=2,
                num_workers_per_link=3,
            ),
            id="xlarge_9472_3456_5120_x7_y7_cwimb1_rs3_fullgrid",
        )
    ],
)
@pytest.mark.parametrize(
    "mem_config_input, mem_config_mm, mem_config_rs",
    [
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        )
    ],
)
@pytest.mark.parametrize(
    "enable_trace, num_iters",
    [
        (False, 1),
    ],
    ids=["check"],
)
@pytest.mark.parametrize(
    "rs_mode",
    [
        "fused",
    ],
)
@pytest.mark.parametrize(
    "device_params, topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}, ttnn.Topology.Ring),
    ],
    indirect=["device_params"],
    ids=["fabric_ring"],
)
def test_minimal_matmul_strided_reduce_scatter_async(
    mesh_device,
    test_config,
    num_links,
    mem_config_input,
    mem_config_mm,
    mem_config_rs,
    enable_trace,
    topology,
    num_iters,
    rs_mode,
    cluster_axis,
):
    cfg = test_config
    TILE_SIZE = 32
    Nt = cfg.N // TILE_SIZE
    Nt_per_core = Nt // cfg.mm_core_grid.x

    M_Block_range = range(TILE_SIZE + 32, 512 + TILE_SIZE, TILE_SIZE)
    K_Block_range = range(TILE_SIZE + 32, 512 + TILE_SIZE, TILE_SIZE)
    N_Block_range = range(TILE_SIZE + 32, 512 + TILE_SIZE, TILE_SIZE)
    sub_h_range = range(1, 8 + 1)
    sub_w_range = range(1, 8 + 1)

    cache_file = f"test_minimal_matmul_strided_reduce_scatter_async_sweep_cache.log"
    processed_cache = set()
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            processed_cache = {line.strip() for line in f}

    with open(cache_file, "a") as f:
        for mm_block_m in M_Block_range:
            for mm_block_k in K_Block_range:
                for mm_block_n in N_Block_range:
                    for sub_h in sub_h_range:
                        for sub_w in sub_w_range:
                            cache_key = f"M_block={mm_block_m}-K_block={mm_block_k}-N_block={mm_block_n}-sub_h={sub_h}-sub_w={sub_w}"
                            if cache_key in processed_cache:
                                continue
                            if Nt_per_core < (mm_block_n // TILE_SIZE):
                                write_error_to_file(
                                    f"{cache_key} - block_n size is {mm_block_n // TILE_SIZE} tiles, but only {Nt_per_core} tiles of work per core"
                                )
                                continue

                            f.write(
                                cache_key + "\n"
                            )  # No need to add to processed_cache since it's already in the file, and combinations are unique
                            f.flush()
                            try:
                                run_minimal_matmul_strided_reduce_scatter_impl(
                                    mesh_device,
                                    cfg.M,
                                    cfg.K,
                                    cfg.N,
                                    cfg.dim,
                                    num_links,
                                    cfg.input_dtype,
                                    cfg.layout,
                                    mem_config_input,
                                    mem_config_mm,
                                    mem_config_rs,
                                    topology=topology,
                                    enable_trace=enable_trace,
                                    num_iters=num_iters,
                                    num_workers_per_link=cfg.num_workers_per_link,
                                    mm_block_m=mm_block_m,
                                    mm_block_k=mm_block_k,
                                    mm_block_n=mm_block_n,
                                    subblock_h=sub_h,  # cfg.subblock_h,
                                    subblock_w=sub_w,  # cfg.subblock_w,
                                    mm_core_grid=cfg.mm_core_grid,
                                    chunk_width_in_mm_blocks=cfg.chunk_width_in_mm_blocks,
                                    rs_mode=rs_mode,
                                    cluster_axis=cluster_axis,
                                    sweep_key=cache_key,
                                )
                            except Exception as e:
                                write_error_to_file(f"{cache_key} - Error: {e}")

            ttnn.ReadDeviceProfiler(mesh_device)
