# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from dataclasses import dataclass
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from models.common.utility_functions import skip_for_blackhole

from tracy import signpost


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
    num_devices,
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
):
    torch.manual_seed(0)

    TILE_SIZE = 32

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

    logger.info(f"Input shape per device: {input_shape}")
    logger.info(f"Weight shape per device: [1, 1, {K}, {N}]")
    logger.info(f"MM output shape per device: [1, 1, {M}, {N}]")
    logger.info(f"RS scatter dim: {dim}, ring_size: {num_devices}")
    logger.info(f"RS output shape per device: [1, 1, {M}, {N // num_devices}]")

    input_tensor_mesh_list = []
    weight_tensor_mesh_list = []
    torch_mm_output_per_device_list = []  # list of lists, [iter][device]
    torch_rs_output_list = []

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
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=[None, 0], mesh_shape=tuple(mesh_device.shape)),
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
    tt_mm_out_tensor_list = []
    tt_rs_out_tensor_list = []

    # In separate mode, the matmul can use the full compute grid since RS runs independently.
    # The RS parameters (mm_cores_y etc.) still use the original mm_core_grid for internal consistency.
    matmul_config_separate = ttnn.MinimalMatmulConfig(
        M_block_size=mm_block_m // TILE_SIZE,
        K_block_size=mm_block_k // TILE_SIZE,
        N_block_size=mm_block_n // TILE_SIZE,
        subblock_h=subblock_h,
        subblock_w=subblock_w,
        compute_with_storage_grid_size=ttnn.CoreCoord(compute_grid_size.x, compute_grid_size.y),
    )

    def run_op(i):
        if rs_mode == "fused":
            # Fused path: matmul and strided reduce-scatter run concurrently
            (
                tt_mm_out,
                tt_rs_intermediate,
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
        else:
            # Non-fused: run matmul to completion, then reduce-scatter sequentially.
            tt_mm_out = ttnn.experimental.minimal_matmul(
                input_tensor_mesh_list[i],
                weight_tensor_mesh_list[i],
                compute_kernel_config=compute_config,
                config=matmul_config_separate,
            )

            if rs_mode == "separate_strided":
                # Strided reduce-scatter on the materialized matmul output.
                # Tests the strided access pattern independently from fusion.
                tt_rs_out_tensor = ttnn.experimental.strided_reduce_scatter_async(
                    tt_mm_out,
                    None,  # persistent_output_buffers
                    dim,
                    ccl_semaphore_handles[i],
                    barrier_semaphore=barrier_semaphore_handles[i] if use_barrier else None,
                    num_links=num_links,
                    memory_config=mem_config_rs,
                    topology=topology,
                    cluster_axis=cluster_axis,
                    num_workers_per_link=num_workers_per_link,
                    num_buffers_per_channel=num_buffers_per_channel,
                    mm_cores_y=mm_core_grid.y,
                    mm_block_ht=mm_block_m // TILE_SIZE,
                    mm_block_wt=mm_block_n // TILE_SIZE,
                    mm_N_block_wt=N // TILE_SIZE // mm_core_grid.x,
                    chunk_width_in_mm_blocks=chunk_width_in_mm_blocks,
                )
            elif rs_mode == "separate":
                # Standard (non-strided) reduce-scatter on the materialized matmul output.
                # Baseline reference that doesn't depend on strided access at all.
                tt_rs_out_tensor = ttnn.experimental.reduce_scatter_minimal_async(
                    tt_mm_out,
                    None,  # persistent_output_buffers
                    dim,
                    ccl_semaphore_handles[i],
                    barrier_semaphore=barrier_semaphore_handles[i] if use_barrier else None,
                    num_links=num_links,
                    memory_config=mem_config_rs,
                    topology=topology,
                    cluster_axis=cluster_axis,
                    num_workers_per_link=num_workers_per_link,
                    num_buffers_per_channel=num_buffers_per_channel,
                )
            else:
                raise ValueError(f"Unknown rs_mode: {rs_mode!r}. Expected 'fused', 'separate_strided', or 'separate'.")

            return tt_mm_out, tt_rs_out_tensor

    if enable_trace:
        # Compile
        run_op(0)
        ttnn.synchronize_device(mesh_device)
        logger.info("Done compiling op")

        # Capture trace
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        tt_mm_out, tt_rs_out = run_op(0)
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh_device)
        logger.info("Done capturing trace")

        # Execute trace
        signpost("start")
        for i in range(num_iters):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            tt_mm_out_tensor_list.append(tt_mm_out)
            tt_rs_out_tensor_list.append(tt_rs_out)
        signpost("stop")
        logger.info("Done executing trace")
    else:
        for i in range(num_iters):
            ttnn.synchronize_device(mesh_device)
            tt_mm_out, tt_rs_out = run_op(i)
            tt_mm_out_tensor_list.append(tt_mm_out)
            tt_rs_out_tensor_list.append(tt_rs_out)

            logger.info(f"Waiting for op")
            ttnn.synchronize_device(mesh_device)
            logger.info(f"Done iteration {i}")

    ##### Verify results #####
    for i in range(num_iters):
        golden_idx = i if not enable_trace else 0

        # Check MM output (each device has different output since weights differ)
        tt_mm_out = ttnn.from_device(tt_mm_out_tensor_list[i])
        tt_mm_out_torch = ttnn.to_torch(
            tt_mm_out,
            mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
        )
        mm_goldens = torch_mm_output_per_device_list[golden_idx]

        for device_id in range(num_devices):
            tt_mm_slice = tt_mm_out_torch[device_id : device_id + 1, :, :, :]
            eq, output = comp_pcc(tt_mm_slice, mm_goldens[device_id], allowed_pcc)
            logger.info(f"MM output device {device_id}, iter {i}: {output}")
            assert eq, f"iter {i} device {device_id} MM FAILED: {output}"

        # Check RS output
        tt_rs_out = ttnn.from_device(tt_rs_out_tensor_list[i])
        tt_rs_out_torch = ttnn.to_torch(
            tt_rs_out,
            mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=dim),
        )
        torch_rs_golden = torch_rs_output_list[golden_idx]

        tt_rs_chunks = torch.chunk(tt_rs_out_torch, num_devices, dim=dim)
        for device_id in range(num_devices):
            eq, output = comp_pcc(tt_rs_chunks[device_id], torch_rs_golden[device_id], allowed_pcc)
            logger.info(f"RS output device {device_id}, iter {i}: {output}")
            assert eq, f"iter {i} device {device_id} RS FAILED: {output}"

    logger.info("All checks passed!")


@skip_for_blackhole("Requires wormhole_b0 to run")
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("num_links", [1], ids=["1link"])
@pytest.mark.parametrize(
    "test_config",
    [
        pytest.param(
            MinimalMatmulStridedReduceScatterTestConfig(
                M=128,
                K=256,
                N=512,
                dim=3,
                mm_block_m=64,
                mm_block_k=64,
                mm_block_n=64,
                mm_core_grid=ttnn.CoreCoord(8, 2),
                chunk_width_in_mm_blocks=1,
            ),
            id="small_Nwt2_cwimb1",
        ),
        pytest.param(
            MinimalMatmulStridedReduceScatterTestConfig(
                M=128,
                K=256,
                N=1024,
                dim=3,
                mm_block_m=64,
                mm_block_k=64,
                mm_block_n=64,
                mm_core_grid=ttnn.CoreCoord(8, 2),
                chunk_width_in_mm_blocks=1,
            ),
            id="medium_Nwt4_cwimb1",
        ),
        pytest.param(
            MinimalMatmulStridedReduceScatterTestConfig(
                M=512,
                K=512,
                N=2048,
                dim=3,
                mm_block_m=128,
                mm_block_k=128,
                mm_block_n=128,
                mm_core_grid=ttnn.CoreCoord(8, 2),
                chunk_width_in_mm_blocks=2,
            ),
            id="large_Nwt8_cwimb2",
        ),
        pytest.param(
            MinimalMatmulStridedReduceScatterTestConfig(
                M=512,
                K=256,
                N=2560,
                dim=3,
                mm_block_m=64,
                mm_block_k=64,
                mm_block_n=64,
                mm_core_grid=ttnn.CoreCoord(8, 2),
                chunk_width_in_mm_blocks=4,
            ),
            id="large_Nwt10_cwimb4",
        ),
        pytest.param(
            MinimalMatmulStridedReduceScatterTestConfig(
                M=4096,
                K=512,
                N=2048,
                dim=3,
                mm_block_m=256,
                mm_block_k=256,
                mm_block_n=256,
                mm_core_grid=ttnn.CoreCoord(8, 4),
                chunk_width_in_mm_blocks=1,
            ),
            id="xlarge_4k_Nwt8_cwimb1",
        ),
        pytest.param(
            MinimalMatmulStridedReduceScatterTestConfig(
                M=4096,
                K=512,
                N=4096,
                dim=3,
                mm_block_m=256,
                mm_block_k=256,
                mm_block_n=256,
                mm_core_grid=ttnn.CoreCoord(8, 4),
                chunk_width_in_mm_blocks=2,
            ),
            id="xlarge_4k_Nwt16_cwimb2",
        ),
        pytest.param(
            MinimalMatmulStridedReduceScatterTestConfig(
                M=3072,
                K=512,
                N=4096,
                dim=3,
                mm_block_m=256,
                mm_block_k=256,
                mm_block_n=256,
                mm_core_grid=ttnn.CoreCoord(8, 6),
                chunk_width_in_mm_blocks=2,
            ),
            id="xlarge_4k_y6_Nwt16_cwimb2",
        ),
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
        "separate",
        "separate_strided",
        "fused",
    ],
)
@pytest.mark.parametrize(
    "device_params, topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 1531456}, ttnn.Topology.Ring),
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
):
    cfg = test_config
    TILE_SIZE = 32
    Nt = cfg.N // TILE_SIZE
    Nt_per_core = Nt // cfg.mm_core_grid.x
    assert Nt_per_core >= (
        cfg.mm_block_n // TILE_SIZE
    ), f"block_n size is {cfg.mm_block_n // TILE_SIZE} tiles, but only {Nt_per_core} tiles of work per core"

    run_minimal_matmul_strided_reduce_scatter_impl(
        mesh_device,
        mesh_device.get_num_devices(),
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
        mm_block_m=cfg.mm_block_m,
        mm_block_k=cfg.mm_block_k,
        mm_block_n=cfg.mm_block_n,
        subblock_h=cfg.subblock_h,
        subblock_w=cfg.subblock_w,
        mm_core_grid=cfg.mm_core_grid,
        chunk_width_in_mm_blocks=cfg.chunk_width_in_mm_blocks,
        rs_mode=rs_mode,
    )
