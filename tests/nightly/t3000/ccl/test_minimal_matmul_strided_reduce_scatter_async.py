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

    logger.info(f"Input shape per device: {input_shape}")
    logger.info(f"Weight shape per device: [1, 1, {K}, {N}]")
    logger.info(f"MM output shape per device: [1, 1, {M}, {N}]")
    logger.info(f"RS scatter dim: {dim}, ring_size: {num_devices}")
    logger.info(f"RS output shape per device: [1, 1, {M}, {N // num_devices}]")

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
                    mm_N_full_block_wt=N // TILE_SIZE // mm_core_grid.x,
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
    # Setup concat mesh to use 1D mesh concatenation.
    concat_mesh_shape = list(mesh_device.shape)
    concat_mesh_shape[1 - cluster_axis] = 1  # Set replicated mesh axis to 1 to prevent concatenation
    for i in range(num_iters):
        golden_idx = i if not enable_trace else 0

        # Check MM output (each device has different output since weights differ)
        # Setup concatenation dimension per axis
        concat_dims = [0, 0]
        concat_dims[
            1 - cluster_axis
        ] = 1  # Dimensions have to be unique. Set to anything but the concatenation dimension.
        tt_mm_out_torch = ttnn.to_torch(
            tt_mm_out_tensor_list[i],
            mesh_composer=ttnn.create_mesh_composer(
                mesh_device, ttnn.MeshComposerConfig(concat_dims, ttnn.MeshShape(concat_mesh_shape))
            ),
        )
        mm_goldens = torch_mm_output_per_device_list[golden_idx]

        for device_id in range(num_devices):
            tt_mm_slice = tt_mm_out_torch[device_id : device_id + 1, :, :, :]
            eq, output = comp_pcc(tt_mm_slice, mm_goldens[device_id], allowed_pcc)
            logger.info(f"MM output device {device_id}, iter {i}: {output}")
            assert eq, f"iter {i} device {device_id} MM FAILED: {output}"

        # Check RS output
        # Setup concatenation dimension per axis
        concat_dims = [dim, dim]
        concat_dims[1 - cluster_axis] = (
            0 if dim != 0 else 1
        )  # Get any other index not dim (needs to be unique). Setting the number of devices to 1 on that dimension will prevent concatenations.
        tt_rs_out_torch = ttnn.to_torch(
            tt_rs_out_tensor_list[i],
            mesh_composer=ttnn.create_mesh_composer(
                mesh_device, ttnn.MeshComposerConfig(concat_dims, ttnn.MeshShape(concat_mesh_shape))
            ),
        )
        torch_rs_golden = torch_rs_output_list[golden_idx]

        tt_rs_chunks = torch.chunk(tt_rs_out_torch, num_devices, dim=dim)
        for device_id in range(num_devices):
            eq, output = comp_pcc(tt_rs_chunks[device_id], torch_rs_golden[device_id], allowed_pcc)
            logger.info(f"RS output device {device_id}, iter {i}: {output}")
            assert eq, f"iter {i} device {device_id} RS FAILED: {output}"

    logger.info("All checks passed!")


@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True, ids=["1x8"])
@pytest.mark.parametrize("num_links", [1], ids=["1link"])
@pytest.mark.parametrize("cluster_axis", [1], ids=["axis_1"])
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
        # Same shape as above but with explicit RS worker counts to explore
        # the MM/RS core budget tradeoff in the fused case.
        # RS cores = 2 + 2*num_workers_per_link (mux + workers, 1 link).
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
                num_workers_per_link=6,  # 14 RS cores, 62 total
            ),
            id="xlarge_4k_y6_Nwt16_cwimb2_rs6",
        ),
        pytest.param(
            MinimalMatmulStridedReduceScatterTestConfig(
                # Non-divisible slice_Wt: mm_core_grid.x=6 (not a power of 2) breaks the
                # usual alignment between slice_Wt and mm_N_full_block_wt.
                # N_tiles=48, mm_N_full_block_wt=48/6=8, slice_Wt=48/8=6. 6 % 8 ≠ 0.
                # chip k: skip_cols_left = (k*6) % 8 = {0,6,4,2,0,6,4,2} — 4 distinct values.
                # chunk_width_in_tiles=8 > effective_advance=2 → cross-column 2-tile packets
                # in the fused reader/writer. Tests all three rs_mode paths.
                M=512,
                K=256,
                N=1536,
                dim=3,
                mm_block_m=128,
                mm_block_k=64,
                mm_block_n=256,
                mm_core_grid=ttnn.CoreCoord(6, 2),
                chunk_width_in_mm_blocks=1,
            ),
            id="non_div_Wt_6x2_cwimb1",
        ),
        pytest.param(
            MinimalMatmulStridedReduceScatterTestConfig(
                # Non-divisible slice_Wt with multi-worker cross-column packets.
                # mm_core_grid.x=5, N_tiles=160, mm_N_full_block_wt=32, slice_Wt=20. 20%32≠0.
                # chip k: skip = (k*20)%32 = {0,20,8,28,16,4,24,12} — 8 distinct skip values.
                # chunk_width_in_mm_blocks=2 → chunk_width_in_tiles=16; num_workers=4 →
                # effective_advance=8. 8 % 16 ≠ 0 → consecutive packet slots visit different
                # columns; with skip=20 a packet (col=0,col=8) spans (invalid,invalid) but
                # (col=4,col=12) spans (invalid,valid) — exercises the CB-packing fix.
                M=3072,
                K=512,
                N=5120,
                dim=3,
                mm_block_m=256,
                mm_block_k=128,
                mm_block_n=256,
                mm_core_grid=ttnn.CoreCoord(5, 6),
                chunk_width_in_mm_blocks=2,
                num_workers_per_link=4,
            ),
            id="non_div_Wt_large_5x6_cwimb2_rs4",
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
        # "separate",
        # "separate_strided",
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
@skip_for_blackhole("t3000 tests are wormhole_b0 only")
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
    assert Nt_per_core >= (
        cfg.mm_block_n // TILE_SIZE
    ), f"block_n size is {cfg.mm_block_n // TILE_SIZE} tiles, but only {Nt_per_core} tiles of work per core"

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
        mm_block_m=cfg.mm_block_m,
        mm_block_k=cfg.mm_block_k,
        mm_block_n=cfg.mm_block_n,
        subblock_h=cfg.subblock_h,
        subblock_w=cfg.subblock_w,
        mm_core_grid=cfg.mm_core_grid,
        chunk_width_in_mm_blocks=cfg.chunk_width_in_mm_blocks,
        rs_mode=rs_mode,
        cluster_axis=cluster_axis,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Optional-feature tests: addcmul, fused_activation, bias, persistent_buffers
# ──────────────────────────────────────────────────────────────────────────────

_FEATURE_BASE_CFG = MinimalMatmulStridedReduceScatterTestConfig(
    M=128,
    K=256,
    N=512,
    dim=3,
    mm_block_m=64,
    mm_block_k=64,
    mm_block_n=64,
    mm_core_grid=ttnn.CoreCoord(8, 2),
    chunk_width_in_mm_blocks=1,
)


def _run_optional_feature_test(mesh_device, topology, cluster_axis, feature):
    """Exercise one optional feature of minimal_matmul_strided_reduce_scatter_async.

    Shares setup with the main test helper but wires up the optional inputs and
    adjusts the golden computation accordingly.
    """
    torch.manual_seed(42)
    TILE_SIZE = 32

    cfg = _FEATURE_BASE_CFG
    num_links = 1
    num_devices = mesh_device.shape[cluster_axis]
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

    compute_grid_size = mesh_device.compute_with_storage_grid_size()
    all_cores = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
    )
    mesh_device.set_sub_device_stall_group([ttnn.SubDeviceId(0)])
    ccl_semaphores = create_global_semaphores(mesh_device, all_cores, 0)
    barrier_semaphore = ttnn.create_global_semaphore(mesh_device, all_cores, 0)
    rs_core_grid_offset = ttnn.CoreCoord(0, cfg.mm_core_grid.y)

    compute_config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    matmul_config = ttnn.MinimalMatmulConfig(
        M_block_size=cfg.mm_block_m // TILE_SIZE,
        K_block_size=cfg.mm_block_k // TILE_SIZE,
        N_block_size=cfg.mm_block_n // TILE_SIZE,
        subblock_h=1,
        subblock_w=1,
        compute_with_storage_grid_size=cfg.mm_core_grid,
    )

    # Input: replicated; weight: sharded so each device has different weights
    shard_dims = [None, None]
    shard_dims[cluster_axis] = 0
    torch_input = torch.randn([1, 1, cfg.M, cfg.K])
    torch_weight_global = torch.randn([num_devices, 1, cfg.K, cfg.N])
    torch_weight_chunks = torch.chunk(torch_weight_global, num_devices, dim=0)

    input_mesh = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    weight_mesh = ttnn.from_torch(
        torch_weight_global,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=mem_config,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=shard_dims, mesh_shape=tuple(mesh_device.shape)),
    )

    # Per-device MM golden outputs
    mm_golds = [torch.matmul(torch_input, torch_weight_chunks[d]).float() for d in range(num_devices)]

    # Feature-specific setup: build extra_kwargs and adjust rs_input_per_device
    extra_kwargs = {}
    rs_input_per_device = mm_golds  # default: RS on raw MM output

    if feature == "fused_activation_relu":
        # Applied to MM output before RS
        rs_input_per_device = [torch.relu(mm_out) for mm_out in mm_golds]
        extra_kwargs = {"fused_activation": ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)}

    elif feature == "bias":
        # Bias added to each device's MM output before RS
        torch_bias = torch.randn([1, 1, 1, cfg.N])
        bias_mesh = ttnn.from_torch(
            torch_bias,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=mem_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        rs_input_per_device = [mm_out + torch_bias for mm_out in mm_golds]
        extra_kwargs = {"bias": bias_mesh}

    elif feature == "addcmul":
        # The reader reads only the first N/num_devices columns from each tensor (b=0 in the kernel
        # batch loop). ternary_a must be full [M, N] and ternary_b must be [1, N] to pass validation
        # in minimal_matmul_device_operation.cpp (N is derived from the weight shape). Replicate the
        # full tensors; every device reads the same first-N/num_devices slice from its local copy.
        scalar = 0.5
        torch_addcmul_a = torch.randn([1, 1, cfg.M, cfg.N])
        torch_addcmul_b = torch.randn([1, 1, 1, cfg.N])
        addcmul_a_mesh = ttnn.from_torch(
            torch_addcmul_a,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=mem_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        addcmul_b_mesh = ttnn.from_torch(
            torch_addcmul_b,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=mem_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        extra_kwargs = {
            "fused_ternary_scalar": scalar,
            "addcmul_input_tensor1": addcmul_a_mesh,
            "addcmul_input_tensor2": addcmul_b_mesh,
        }

    # Golden: reduce-scatter over rs_input_per_device
    torch_rs_reduced = torch.sum(torch.stack(rs_input_per_device), dim=0)
    torch_rs_scattered = list(torch.chunk(torch_rs_reduced, num_devices, dim=cfg.dim))
    if feature == "addcmul":
        # Each device reads the first N/num_devices columns of the replicated tensors.
        slice_n = cfg.N // num_devices
        addcmul_a_slice = torch_addcmul_a[..., :slice_n]  # [1, 1, M, N/num_devices]
        addcmul_b_slice = torch_addcmul_b[..., :slice_n]  # [1, 1, 1, N/num_devices] broadcast
        torch_rs_scattered = [addcmul_a_slice + scalar * chunk * addcmul_b_slice for chunk in torch_rs_scattered]

    # Run the op
    mm_out, rs_out = ttnn.experimental.minimal_matmul_strided_reduce_scatter_async(
        input_mesh,
        weight_mesh,
        cfg.dim,
        ccl_semaphores,
        rs_core_grid_offset,
        num_links=num_links,
        memory_config_mm=mem_config,
        rs_output_mem_config=mem_config,
        topology=topology,
        cluster_axis=cluster_axis,
        config=matmul_config,
        compute_kernel_config=compute_config,
        barrier_semaphore=barrier_semaphore,
        chunk_width_in_mm_blocks=cfg.chunk_width_in_mm_blocks,
        **extra_kwargs,
    )
    ttnn.synchronize_device(mesh_device)

    # Verify RS output on each device
    concat_dims = [cfg.dim, cfg.dim]
    concat_dims[1 - cluster_axis] = 0 if cfg.dim != 0 else 1
    concat_mesh_shape = list(mesh_device.shape)
    concat_mesh_shape[1 - cluster_axis] = 1
    tt_rs_torch = ttnn.to_torch(
        rs_out,
        mesh_composer=ttnn.create_mesh_composer(
            mesh_device, ttnn.MeshComposerConfig(concat_dims, ttnn.MeshShape(concat_mesh_shape))
        ),
    )
    tt_rs_chunks = torch.chunk(tt_rs_torch, num_devices, dim=cfg.dim)
    for d in range(num_devices):
        eq, msg = comp_pcc(tt_rs_chunks[d], torch_rs_scattered[d], 0.99)
        logger.info(f"feature={feature} device {d}: {msg}")
        assert eq, f"feature={feature} device {d} FAILED: {msg}"

    logger.info(f"All checks passed for feature={feature}!")


@skip_for_blackhole("t3000 tests are wormhole_b0 only")
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True, ids=["1x8"])
@pytest.mark.parametrize("cluster_axis", [1], ids=["axis_1"])
@pytest.mark.parametrize(
    "device_params, topology",
    [({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 1531456}, ttnn.Topology.Ring)],
    indirect=["device_params"],
    ids=["fabric_ring"],
)
@pytest.mark.parametrize(
    "feature",
    ["fused_activation_relu", "bias", "addcmul"],
)
def test_minimal_matmul_strided_reduce_scatter_async_optional_features(mesh_device, topology, cluster_axis, feature):
    _run_optional_feature_test(mesh_device, topology, cluster_axis, feature)
