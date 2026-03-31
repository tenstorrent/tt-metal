# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Single-shot AGMM compare: 64k ISL-shaped M vs 6×8 (3 links) vs 7×8 (1 link).

Fixed block sizes from your sweep row: M_block=64, K_block=1, N_block=1.
Subblock: for N_block=1 only subblock_w=1 divides N; max subblock_h with h*w<=4 is (4,1).

Usage (from tt-metal repo root, Galaxy 8×4):
  pytest tests/ttnn/unit_tests/operations/ccl/test_agmm_64k_grid_compare.py::test_agmm_64k_6x8_vs_7x8_same_blocks -x -s

For Tracy/device kernel timing, profile the same node:
  python tools/tracy/profile_this.py -c "pytest tests/ttnn/unit_tests/operations/ccl/test_agmm_64k_grid_compare.py::test_agmm_64k_6x8_vs_7x8_same_blocks -x -s" -o agmm_64k_grid
"""

import time

import pytest
import torch
from loguru import logger

import ttnn

# 64k tokens → M rows (tile height 32)
M = 65536
K = 3584
N = 2048

# Sweep row you called out
M_BLOCK = 64
K_BLOCK = 1
N_BLOCK = 1
# Best valid dest subblock for (M_block=64, N_block=1): w|1 => w=1; h|64, h*w<=4 => h in {1,2,4} -> (4,1)
SUBBLOCK_H = 4
SUBBLOCK_W = 1

DEVICE_6x8_3LINKS = {
    "mesh_shape": (8, 4),
    "fabric_config": "FABRIC_1D_RING",
    "fabric_router_config_payload": 7168,
    "topology": "Ring",
    "num_links": 3,
    "num_workers_per_link": 2,
    "cluster_axis": 1,
    "ring_size": 4,
}

DEVICE_7x8_1LINK = {
    "mesh_shape": (8, 4),
    "fabric_config": "FABRIC_1D_RING",
    "fabric_router_config_payload": 7168,
    "topology": "Ring",
    "num_links": 1,
    "num_workers_per_link": 7,
    "cluster_axis": 1,
    "ring_size": 4,
}


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


def _run_one_variant(label, cfg, core_x, core_y, warmup_iters=1, measured_iters=2):
    """Open mesh, run AGMM warmup + measured loop; return list of host-side seconds per measured iter."""
    cluster_axis = cfg["cluster_axis"]
    topology = getattr(ttnn.Topology, cfg["topology"])
    core_grid = ttnn.CoreCoord(core_x, core_y)
    dtype = ttnn.bfloat8_b

    logger.info(
        f"=== {label}: grid={core_x}x{core_y}, num_links={cfg['num_links']}, "
        f"blocks=({M_BLOCK},{K_BLOCK},{N_BLOCK}), subblock=({SUBBLOCK_H},{SUBBLOCK_W}) ==="
    )

    mesh_device = open_mesh(cfg)
    try:
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

        for _ in range(warmup_iters):
            run_op()
        logger.info(f"{label}: warmup done ({warmup_iters} iters)")

        times_s = []
        for i in range(measured_iters):
            t0 = time.perf_counter()
            run_op()
            t1 = time.perf_counter()
            dt = t1 - t0
            times_s.append(dt)
            logger.info(f"{label}: measured iter {i + 1}/{measured_iters} host_wall_s={dt:.4f}s")

        avg_ms = sum(times_s) / len(times_s) * 1000
        logger.info(f"{label}: avg host_wall={avg_ms:.2f} ms over {measured_iters} iters (not device kernel ns)")
        return times_s
    finally:
        close_mesh(mesh_device)


@pytest.mark.timeout(3600)
def test_agmm_64k_6x8_vs_7x8_same_blocks():
    """
    One pytest: first 6×8 + 3 links, then 7×8 + 1 link.
    Same M,K,N and MinimalMatmulConfig blocks (64,1,1) + subblock (4,1).

    Note: (16,8,8) + subblock (1,4) was Pass-2 winner in CSV for a different M_block row;
    that combo is not the same as (64,1,1). Add another test if you want that pair.
    """
    t_6x8 = _run_one_variant("6x8_3links", DEVICE_6x8_3LINKS, 6, 8)
    t_7x8 = _run_one_variant("7x8_1link", DEVICE_7x8_1LINK, 7, 8)

    a6 = sum(t_6x8) / len(t_6x8) * 1000
    a7 = sum(t_7x8) / len(t_7x8) * 1000
    logger.info(f"SUMMARY avg host wall: 6x8_3links={a6:.2f} ms, 7x8_1link={a7:.2f} ms")
    if a6 > 0 and a7 > 0:
        faster = "6x8_3links" if a6 < a7 else "7x8_1link"
        logger.info(f"SUMMARY faster (host wall): {faster} (ratio max/min={max(a6,a7)/min(a6,a7):.3f}x)")
