# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Explicit 64k ISL AGMM: M=65536, blocks (8,8,8), configurable subblock, 6x8 grid, wh_galaxy (3 links).

- (2,2): forced subblock; sweep Pass 1 uses pick_subblock → often (1,4).
- (2,4): **not valid** for fused `all_gather_minimal_matmul_async` — Metal enforces
  `subblock_h * subblock_w <= max_dest_volume` (4). Model `PREFILL_FF2_MINIMAL_MATMUL_CONFIG` uses (2,4)
  for **standalone** `minimal_matmul` only. See skipped test below for the exact assert.

Usage (from tt-metal repo root, Galaxy):
  pytest tests/ttnn/unit_tests/operations/ccl/test_agmm_64k_explicit_8_8_8.py::test_agmm_64k_blocks_8_8_8_subblock_2_2 -x -s
"""

import pytest
import torch
from loguru import logger

import ttnn

DEVICE_CONFIG = {
    "mesh_shape": (8, 4),
    "fabric_config": "FABRIC_1D_RING",
    "fabric_router_config_payload": 7168,
    "topology": "Ring",
    "num_links": 3,
    "num_workers_per_link": 2,
    "cluster_axis": 1,
    "ring_size": 4,
}

M = 65536
K = 3584
N = 2048
CORE_GRID = (6, 8)

M_BLOCK = 8
K_BLOCK = 8
N_BLOCK = 8

WARMUP_ITERS = 1
MEASURED_ITERS = 2


def create_fabric_router_config(max_payload_size):
    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = max_payload_size
    return config


def create_global_semaphores(mesh_device, num_devices, core_range_set, initial_value=0):
    return [ttnn.create_global_semaphore(mesh_device, core_range_set, initial_value) for _ in range(num_devices)]


def open_mesh(cfg):
    fabric_kwargs = [
        getattr(ttnn.FabricConfig, cfg["fabric_config"]),
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
    ]
    if cfg["fabric_router_config_payload"] is not None:
        fabric_kwargs.append(create_fabric_router_config(cfg["fabric_router_config_payload"]))
    ttnn.set_fabric_config(*fabric_kwargs)
    rows, cols = cfg["mesh_shape"]
    return ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(rows, cols))


def close_mesh(mesh_device):
    ttnn.close_mesh_device(mesh_device)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _run_agmm_64k_explicit_subblock(subblock_h: int, subblock_w: int):
    cfg = DEVICE_CONFIG
    cluster_axis = cfg["cluster_axis"]
    topology = getattr(ttnn.Topology, cfg["topology"])

    logger.info(
        f"Explicit AGMM 64k: M={M} K={K} N={N} grid={CORE_GRID} "
        f"links={cfg['num_links']} blocks=({M_BLOCK},{K_BLOCK},{N_BLOCK}) subblock=({subblock_h},{subblock_w})"
    )

    mesh_device = open_mesh(cfg)
    try:
        dtype = ttnn.bfloat8_b
        core_grid = ttnn.CoreCoord(CORE_GRID[0], CORE_GRID[1])

        tt_input = ttnn.from_torch(
            torch.randn((M, K), dtype=torch.float32),
            dtype=dtype,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device, mesh_shape=tuple(mesh_device.shape), dims=[None, cluster_axis]
            ),
        )
        tt_weight = ttnn.from_torch(
            torch.randn((K, N), dtype=torch.float32),
            dtype=dtype,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
        )
        tt_bias = ttnn.from_torch(
            torch.randn((1, N), dtype=torch.float32),
            dtype=dtype,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
        )
        persistent_output_buffer = ttnn.from_torch(
            torch.zeros((M, K), dtype=torch.float32),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=[None, None]),
        )
        ccl_cores = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(core_grid.x - 1, core_grid.y - 1))}
        )
        ccl_semaphore_handles = create_global_semaphores(mesh_device, mesh_device.get_num_devices(), ccl_cores, 0)
        compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        matmul_config = ttnn.MinimalMatmulConfig(
            M_block_size=M_BLOCK,
            K_block_size=K_BLOCK,
            N_block_size=N_BLOCK,
            subblock_h=subblock_h,
            subblock_w=subblock_w,
            compute_with_storage_grid_size=core_grid,
        )

        for _ in range(WARMUP_ITERS):
            ttnn.experimental.all_gather_minimal_matmul_async(
                tt_input,
                tt_weight,
                bias_tensor=tt_bias,
                fused_activation=None,
                compute_kernel_config=compute_config,
                config=matmul_config,
                persistent_output_buffer=persistent_output_buffer,
                multi_device_global_semaphore=ccl_semaphore_handles,
                num_links=cfg["num_links"],
                topology=topology,
                cluster_axis=cluster_axis,
                barrier_semaphore=None,
                force_transpose=True,
                num_workers_per_link=cfg["num_workers_per_link"],
                num_buffers_per_channel=8,
                scalar=None,
                addcmul_input_tensor1=None,
                addcmul_input_tensor2=None,
                chunks=1,
            )
            ttnn.synchronize_device(mesh_device)
        logger.info("Warmup done")

        from tracy import signpost

        signpost("start")
        for _ in range(MEASURED_ITERS):
            ttnn.experimental.all_gather_minimal_matmul_async(
                tt_input,
                tt_weight,
                bias_tensor=tt_bias,
                fused_activation=None,
                compute_kernel_config=compute_config,
                config=matmul_config,
                persistent_output_buffer=persistent_output_buffer,
                multi_device_global_semaphore=ccl_semaphore_handles,
                num_links=cfg["num_links"],
                topology=topology,
                cluster_axis=cluster_axis,
                barrier_semaphore=None,
                force_transpose=True,
                num_workers_per_link=cfg["num_workers_per_link"],
                num_buffers_per_channel=8,
                scalar=None,
                addcmul_input_tensor1=None,
                addcmul_input_tensor2=None,
                chunks=1,
            )
        ttnn.synchronize_device(mesh_device)
        signpost("stop")

        logger.info(
            f"Done: blocks=({M_BLOCK},{K_BLOCK},{N_BLOCK}) subblock=({subblock_h},{subblock_w}) "
            f"measured_iters={MEASURED_ITERS}"
        )
    finally:
        close_mesh(mesh_device)


@pytest.mark.timeout(3600)
def test_agmm_64k_blocks_8_8_8_subblock_2_2():
    _run_agmm_64k_explicit_subblock(2, 2)


@pytest.mark.timeout(3600)
def test_agmm_64k_blocks_8_8_8_subblock_2_4():
    """(2,4) is valid for standalone minimal_matmul in the model; fused AGMM rejects 2*4 > 4."""
    pytest.skip(
        "Fused all_gather_minimal_matmul_async requires subblock_h*subblock_w<=4; (2,4)=8 invalid "
        "(all_gather_minimal_matmul_async_device_operation.cpp). Model (2,4) is for standalone minimal_matmul. "
        "For fused, use subblock_2_2 or sweep default (1,4)."
    )
