# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Test for MM+RS fused op with the original "xlarge" default big size
# Testing different core grid configurations on Galaxy (8x4 mesh)
#
# Biggest size from bklockiewicz's branch: M=4096, K=512, N=4096

import torch
import pytest
from dataclasses import dataclass
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from models.common.utility_functions import skip_for_blackhole

from tracy import signpost


@dataclass
class TestConfig:
    M: int
    K: int
    N: int
    dim: int
    mm_block_m: int
    mm_block_k: int
    mm_block_n: int
    mm_core_grid: object  # ttnn.CoreCoord
    chunk_width_in_mm_blocks: int
    num_workers_per_link: int = 3
    subblock_h: int = 1
    subblock_w: int = 1
    input_dtype: object = None
    math_fidelity: object = None
    fp32_acc: bool = True

    def __post_init__(self):
        if self.input_dtype is None:
            self.input_dtype = ttnn.bfloat16
        if self.math_fidelity is None:
            self.math_fidelity = ttnn.MathFidelity.HiFi2


def create_global_semaphores(mesh_device, cores, initial_value):
    return [ttnn.create_global_semaphore(mesh_device, cores, initial_value) for _ in range(3)]


def run_test(
    mesh_device,
    cfg,
    num_links=1,
    num_iters=1,
    enable_trace=False,
    cluster_axis=1,
    rs_mode="fused",
    allowed_pcc=0.99,
):
    torch.manual_seed(0)
    TILE_SIZE = 32

    # Extract config
    M, K, N = cfg.M, cfg.K, cfg.N
    dim = cfg.dim
    mm_core_grid = cfg.mm_core_grid

    # Memory configs
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    topology = ttnn.Topology.Ring

    # Mesh info
    mesh_shape = tuple(mesh_device.shape)
    ring_size = mesh_shape[cluster_axis]
    total_devices = mesh_shape[0] * mesh_shape[1]

    logger.info(f"=== Test Config ===")
    logger.info(f"Tensor: M={M}, K={K}, N={N}")
    logger.info(f"Core grid: {mm_core_grid.x}x{mm_core_grid.y}")
    logger.info(f"Blocks: {cfg.mm_block_m}/{cfg.mm_block_k}/{cfg.mm_block_n}")
    logger.info(f"Mesh: {mesh_shape}, cluster_axis={cluster_axis}, ring_size={ring_size}")
    logger.info(f"RS output N per device: {N // ring_size}")

    # RS core grid offset
    rs_core_grid_offset = ttnn.CoreCoord(0, mm_core_grid.y)

    # Setup
    compute_grid_size = mesh_device.compute_with_storage_grid_size()
    all_cores = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
    )
    worker_sub_device_id = ttnn.SubDeviceId(0)
    mesh_device.set_sub_device_stall_group([worker_sub_device_id])

    ccl_semaphore_handles = [create_global_semaphores(mesh_device, all_cores, 0) for _ in range(num_iters)]
    barrier_semaphore_handles = [ttnn.create_global_semaphore(mesh_device, all_cores, 0) for _ in range(num_iters)]

    # Create tensors
    input_shape = [1, 1, M, K]
    weight_shape = [1, 1, K, N]

    torch_input = torch.randn(input_shape, dtype=torch.float32)
    torch_weight = torch.randn(weight_shape, dtype=torch.float32)
    torch_mm_out = torch.matmul(torch_input, torch_weight)
    torch_rs_reduced = torch_mm_out * ring_size
    torch_rs_scattered = torch.chunk(torch_rs_reduced, ring_size, dim=dim)

    input_tensor = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=cfg.input_dtype,
        memory_config=mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    weight_tensor = ttnn.from_torch(
        torch_weight,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=cfg.input_dtype,
        memory_config=mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # Compute config
    compute_config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=cfg.math_fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=cfg.fp32_acc,
        packer_l1_acc=True,
    )
    matmul_config = ttnn.MinimalMatmulConfig(
        M_block_size=cfg.mm_block_m // TILE_SIZE,
        K_block_size=cfg.mm_block_k // TILE_SIZE,
        N_block_size=cfg.mm_block_n // TILE_SIZE,
        subblock_h=cfg.subblock_h,
        subblock_w=cfg.subblock_w,
        compute_with_storage_grid_size=mm_core_grid,
    )

    # Run the fused op
    import time

    start_time = time.perf_counter()

    tt_mm_out, tt_rs_intermediate, tt_rs_out = ttnn.experimental.minimal_matmul_strided_reduce_scatter_async(
        input_tensor,
        weight_tensor,
        dim,
        ccl_semaphore_handles[0],
        rs_core_grid_offset,
        num_links=num_links,
        memory_config_mm=mem_config,
        rs_output_mem_config=mem_config,
        topology=topology,
        cluster_axis=cluster_axis,
        config=matmul_config,
        compute_kernel_config=compute_config,
        barrier_semaphore=barrier_semaphore_handles[0],
        num_workers_per_link=cfg.num_workers_per_link,
        chunk_width_in_mm_blocks=cfg.chunk_width_in_mm_blocks,
    )

    ttnn.synchronize_device(mesh_device)
    end_time = time.perf_counter()
    elapsed_ms = (end_time - start_time) * 1000
    logger.info(f"Host-side time: {elapsed_ms:.2f} ms")

    # Verify MM output
    tt_mm_out_cpu = ttnn.from_device(tt_mm_out)
    tt_mm_out_torch = ttnn.to_torch(tt_mm_out_cpu, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))

    for device_id in range(total_devices):
        tt_mm_slice = tt_mm_out_torch[device_id : device_id + 1, :, :, :]
        eq, output = comp_pcc(tt_mm_slice, torch_mm_out, allowed_pcc)
        if not eq:
            logger.error(f"MM output device {device_id} FAILED: {output}")
        assert eq, f"device {device_id} MM FAILED: {output}"
    logger.info(f"MM output: PASS (PCC >= {allowed_pcc})")

    # Verify RS output
    tt_rs_out_cpu = ttnn.from_device(tt_rs_out)
    tt_rs_out_torch = ttnn.to_torch(tt_rs_out_cpu, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=dim))
    tt_rs_chunks = torch.chunk(tt_rs_out_torch, total_devices, dim=dim)

    for device_id in range(total_devices):
        ring_pos = device_id % ring_size if cluster_axis == 1 else device_id // mesh_shape[1]
        eq, output = comp_pcc(tt_rs_chunks[device_id], torch_rs_scattered[ring_pos], allowed_pcc)
        if not eq:
            logger.error(f"RS output device {device_id} (ring_pos={ring_pos}) FAILED: {output}")
        assert eq, f"device {device_id} (ring_pos={ring_pos}) RS FAILED: {output}"
    logger.info(f"RS output: PASS (PCC >= {allowed_pcc})")

    logger.info("=== TEST PASSED ===")


# ============================================================================
# Test configurations: Original xlarge size with different core grids
# Original: M=3584, K=512, N=4096, blocks=256/256/256
# ============================================================================


@skip_for_blackhole("Requires wormhole_b0 to run")
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)  # Galaxy
@pytest.mark.parametrize("num_links", [1], ids=["1link"])
@pytest.mark.parametrize(
    "test_config",
    [
        pytest.param(
            TestConfig(
                M=4096,
                K=512,
                N=4096,
                dim=3,
                mm_block_m=256,
                mm_block_k=256,
                mm_block_n=256,
                mm_core_grid=ttnn.CoreCoord(4, 8),
                chunk_width_in_mm_blocks=2,
                num_workers_per_link=3,
            ),
            id="xlarge_4x8",
        ),
        pytest.param(
            TestConfig(
                M=4096,
                K=512,
                N=4096,
                dim=3,
                mm_block_m=256,
                mm_block_k=256,
                mm_block_n=256,
                mm_core_grid=ttnn.CoreCoord(8, 4),
                chunk_width_in_mm_blocks=2,
                num_workers_per_link=3,
            ),
            id="xlarge_8x4",
        ),
        pytest.param(
            TestConfig(
                M=4096,
                K=512,
                N=4096,
                dim=3,
                mm_block_m=256,
                mm_block_k=256,
                mm_block_n=256,
                mm_core_grid=ttnn.CoreCoord(7, 8),
                chunk_width_in_mm_blocks=2,
                num_workers_per_link=3,
            ),
            id="xlarge_7x8",
        ),
        pytest.param(
            TestConfig(
                M=4096,
                K=512,
                N=4096,
                dim=3,
                mm_block_m=256,
                mm_block_k=256,
                mm_block_n=256,
                mm_core_grid=ttnn.CoreCoord(8, 6),
                chunk_width_in_mm_blocks=2,
                num_workers_per_link=3,
            ),
            id="xlarge_8x6",
        ),
        pytest.param(
            TestConfig(
                M=4096,
                K=512,
                N=4096,
                dim=3,
                mm_block_m=256,
                mm_block_k=256,
                mm_block_n=256,
                mm_core_grid=ttnn.CoreCoord(8, 7),
                chunk_width_in_mm_blocks=2,
                num_workers_per_link=3,
            ),
            id="xlarge_8x7",
        ),
        pytest.param(
            TestConfig(
                M=4096,
                K=512,
                N=4096,
                dim=3,
                mm_block_m=256,
                mm_block_k=256,
                mm_block_n=256,
                mm_core_grid=ttnn.CoreCoord(8, 8),
                chunk_width_in_mm_blocks=2,
                num_workers_per_link=3,
            ),
            id="xlarge_8x8",
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
    ids=["dram"],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 266240}],
    indirect=True,
)
def test_default_big_size(
    mesh_device,
    num_links,
    test_config,
    mem_config_input,
    mem_config_mm,
    mem_config_rs,
):
    run_test(
        mesh_device=mesh_device,
        cfg=test_config,
        num_links=num_links,
        num_iters=1,
        enable_trace=False,
        cluster_axis=1,
        rs_mode="fused",
        allowed_pcc=0.99,
    )
