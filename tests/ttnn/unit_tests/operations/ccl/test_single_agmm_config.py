# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Single config test for all_gather_minimal_matmul_async (fused AG+MM)
Matches Llama 70B FF2 model config exactly: K_block=8, N_block=8, subblock=(2,2)

Usage:
  pytest tests/ttnn/unit_tests/operations/ccl/test_single_agmm_config.py -x -s

  # With profiler:
  python tools/tracy/profile_this.py -c "pytest tests/ttnn/unit_tests/operations/ccl/test_single_agmm_config.py -x -s" -o agmm_profile
"""

import pytest
import torch
from loguru import logger

import ttnn


# Device config for WH Galaxy
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

# Llama 70B FF2 shape
M = 8192
K = 3584
N = 2048
CORE_GRID = (6, 8)

# Block config matching model exactly
M_BLOCK = 8
K_BLOCK = 8
N_BLOCK = 8
SUBBLOCK_H = 2
SUBBLOCK_W = 2

# Profiling: one compile warmup, then few measured ops between signposts (avoid long runs / buffer fill)
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


@pytest.mark.timeout(600)
def test_single_agmm_config():
    """Run single all_gather_minimal_matmul_async with exact model config."""
    cfg = DEVICE_CONFIG
    cluster_axis = cfg["cluster_axis"]
    topology = getattr(ttnn.Topology, cfg["topology"])

    logger.info(f"Testing AGMM: M={M}, K={K}, N={N}, grid={CORE_GRID}")
    logger.info(f"Block config: M_block={M_BLOCK}, K_block={K_BLOCK}, N_block={N_BLOCK}")
    logger.info(f"Subblock: ({SUBBLOCK_H}, {SUBBLOCK_W})")

    mesh_device = open_mesh(cfg)
    try:
        dtype = ttnn.bfloat8_b
        core_grid = ttnn.CoreCoord(CORE_GRID[0], CORE_GRID[1])

        # Input tensor: (M, K) = (8192, 3584), sharded to (8192, 896) per device
        tt_input = ttnn.from_torch(
            torch.randn((M, K), dtype=torch.float32),
            dtype=dtype,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device, mesh_shape=tuple(mesh_device.shape), dims=[None, cluster_axis]
            ),
        )

        # Weight: (K, N) = (3584, 2048), replicated
        tt_weight = ttnn.from_torch(
            torch.randn((K, N), dtype=torch.float32),
            dtype=dtype,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
        )

        # Bias: (1, N) = (1, 2048), replicated
        tt_bias = ttnn.from_torch(
            torch.randn((1, N), dtype=torch.float32),
            dtype=dtype,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
        )

        # Persistent output buffer: (M, K) = (8192, 3584), replicated
        persistent_output_buffer = ttnn.from_torch(
            torch.zeros((M, K), dtype=torch.float32),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=[None, None]),
        )

        # Semaphores
        ccl_cores = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(core_grid.x - 1, core_grid.y - 1))}
        )
        ccl_semaphore_handles = create_global_semaphores(mesh_device, mesh_device.get_num_devices(), ccl_cores, 0)

        # Compute config
        compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        # Matmul config with exact model parameters
        matmul_config = ttnn.MinimalMatmulConfig(
            M_block_size=M_BLOCK,
            K_block_size=K_BLOCK,
            N_block_size=N_BLOCK,
            subblock_h=SUBBLOCK_H,
            subblock_w=SUBBLOCK_W,
            compute_with_storage_grid_size=core_grid,
        )

        logger.info(f"Warmup ({WARMUP_ITERS} iteration(s), outside signposts)...")
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

        # Measured runs with signpost markers
        from tracy import signpost

        logger.info(f"Measured region: {MEASURED_ITERS} iteration(s) between signposts")

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

        logger.info(f"Done! {MEASURED_ITERS} measured iteration(s) completed")
        logger.info(
            f"Config: M_block={M_BLOCK}, K_block={K_BLOCK}, N_block={N_BLOCK}, subblock=({SUBBLOCK_H},{SUBBLOCK_W})"
        )

    finally:
        close_mesh(mesh_device)


if __name__ == "__main__":
    test_single_agmm_config()
