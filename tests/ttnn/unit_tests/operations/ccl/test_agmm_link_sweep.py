# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Sweep num_links and num_workers_per_link for all_gather_minimal_matmul_async
on Llama 70B FF2 shape using the best block config from previous sweep.

Constraints for this op with force_transpose=True and compute grid (CORE_GRID.x, CORE_GRID.y)
(see all_gather_minimal_matmul_async_program_factory.cpp):

1) num_links must divide CORE_GRID.x (the in0 row count in transpose layout).
   For FF2 grid 6x8: valid num_links are 1, 2, 3, 6 — not 4.

2) num_workers_per_link should not exceed cores per link along that axis:
   cores_per_link = CORE_GRID.x // num_links. Example: num_links=3 -> 2 cores/link.
   Using num_workers_per_link=4 with num_links=3 only has 2 physical in0 cores per link;
   mux/handshake logic expects one worker index per core in the chain -> hang or bad behavior.
   For num_links=3 use num_workers_per_link in {1, 2} (Llama Galaxy uses 3 links, 2 workers).

Usage:
  pytest tests/ttnn/unit_tests/operations/ccl/test_agmm_link_sweep.py -x -s

  # With profiler:
  python tools/tracy/profile_this.py -c "pytest tests/ttnn/unit_tests/operations/ccl/test_agmm_link_sweep.py -x -s" -o agmm_link_sweep
"""

import pytest
import torch
from loguru import logger

import ttnn


# Llama 70B FF2 shape
M = 8192
K = 3584
N = 2048
CORE_GRID = (6, 8)

# Best block config from sweep
M_BLOCK = 16
K_BLOCK = 8
N_BLOCK = 8
SUBBLOCK_H = 1
SUBBLOCK_W = 4


def create_fabric_router_config(max_payload_size):
    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = max_payload_size
    return config


def create_global_semaphores(mesh_device, num_devices, core_range_set, initial_value=0):
    return [ttnn.create_global_semaphore(mesh_device, core_range_set, initial_value) for _ in range(num_devices)]


def open_mesh(num_links):
    """Open mesh with appropriate fabric config for num_links."""
    fabric_kwargs = [
        ttnn.FabricConfig.FABRIC_1D_RING,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
    ]
    # Adjust payload size based on num_links
    payload_sizes = {1: 4096, 2: 4096, 3: 7168, 4: 4096}
    payload = payload_sizes.get(num_links, 7168)
    fabric_kwargs.append(create_fabric_router_config(payload))
    ttnn.set_fabric_config(*fabric_kwargs)

    return ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(8, 4))


def close_mesh(mesh_device):
    ttnn.close_mesh_device(mesh_device)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def run_agmm_test(num_links, num_workers_per_link, num_iters=2):
    """Run AGMM with specified link config and return average duration."""
    cluster_axis = 1
    topology = ttnn.Topology.Ring

    logger.info(f"Testing: num_links={num_links}, num_workers_per_link={num_workers_per_link}")

    mesh_device = open_mesh(num_links)
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

        # Matmul config
        matmul_config = ttnn.MinimalMatmulConfig(
            M_block_size=M_BLOCK,
            K_block_size=K_BLOCK,
            N_block_size=N_BLOCK,
            subblock_h=SUBBLOCK_H,
            subblock_w=SUBBLOCK_W,
            compute_with_storage_grid_size=core_grid,
        )

        def run_op():
            ttnn.experimental.all_gather_minimal_matmul_async(
                tt_input,
                tt_weight,
                bias_tensor=tt_bias,
                fused_activation=None,
                compute_kernel_config=compute_config,
                config=matmul_config,
                persistent_output_buffer=persistent_output_buffer,
                multi_device_global_semaphore=ccl_semaphore_handles,
                num_links=num_links,
                topology=topology,
                cluster_axis=cluster_axis,
                barrier_semaphore=None,
                force_transpose=True,
                num_workers_per_link=num_workers_per_link,
                num_buffers_per_channel=8,
                scalar=None,
                addcmul_input_tensor1=None,
                addcmul_input_tensor2=None,
                chunks=1,
            )
            ttnn.synchronize_device(mesh_device)

        # Warmup
        logger.info("Warmup...")
        run_op()

        # Measured runs with signpost markers
        from tracy import signpost

        logger.info(f"Running {num_iters} measured iterations...")
        signpost("start")
        for _ in range(num_iters):
            run_op()
        signpost("stop")

        logger.info(f"Done: num_links={num_links}, num_workers_per_link={num_workers_per_link}")

    finally:
        close_mesh(mesh_device)


# Link configurations valid for CORE_GRID.x == 6 and force_transpose=True
# (see module docstring — do not use e.g. (3, 4) or num_links=4 here)
LINK_CONFIGS = [
    (1, 2),
    (1, 4),
    (1, 6),
    (2, 2),
    (2, 3),
    (3, 1),
    (3, 2),  # Llama 70B Galaxy FF2 default
    (6, 1),
]


@pytest.mark.timeout(600)
@pytest.mark.parametrize(
    "num_links,num_workers_per_link", LINK_CONFIGS, ids=[f"links{l}_workers{w}" for l, w in LINK_CONFIGS]
)
def test_agmm_link_config(num_links, num_workers_per_link):
    """Test AGMM with different link configurations."""
    run_agmm_test(num_links, num_workers_per_link)


if __name__ == "__main__":
    for num_links, num_workers_per_link in LINK_CONFIGS:
        try:
            run_agmm_test(num_links, num_workers_per_link)
        except Exception as e:
            logger.error(f"Failed: num_links={num_links}, num_workers_per_link={num_workers_per_link}: {e}")
