# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Test old vs new fused AG+MM configs for Llama 70B FF2.

Old config: K_block=8 — invalid on current kernel (K_tiles_per_device=28, 28%8!=0).
New config: K_block=7 — valid (28%7==0), found via block-size sweep.

12 tests total: 6 ISLs × {old_config, new_config}
  - old_config tests are marked xfail (expected TT_FATAL due to K_block constraint)
  - new_config tests should pass

Usage:
  pytest tests/ttnn/unit_tests/operations/ccl/test_agmm_old_vs_new_config.py -v -s
"""

import pytest
import torch

import ttnn


# ============================================================================
# DEVICE CONFIG (Wormhole Galaxy 8×4 mesh)
# ============================================================================

DEVICE_CFG = {
    "mesh_shape": (8, 4),
    "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
    "fabric_router_config_payload": 7168,
    "topology": ttnn.Topology.Ring,
    "num_links": 3,
    "num_workers_per_link": 2,
    "cluster_axis": 1,
    "ring_size": 4,
}


# ============================================================================
# ISL SHAPE TABLE: (ISL_label, M, K, N, grid_x, grid_y)
# ============================================================================

ISLS = [
    ("4k", 4096, 3584, 2048, 6, 8),
    ("8k", 8192, 3584, 2048, 6, 8),
    ("16k", 16384, 3584, 2048, 6, 8),
    ("32k", 32768, 3584, 2048, 6, 8),
    ("64k", 65536, 3584, 2048, 6, 8),
    ("128k", 131072, 3584, 2048, 6, 8),
]


# ============================================================================
# OLD CONFIGS (K_block=8, pre-sweep — will fail K_tiles_per_device constraint)
# ============================================================================

OLD_CONFIGS = {
    "4k": dict(M_block_size=8, K_block_size=8, N_block_size=8, subblock_h=1, subblock_w=4),
    "8k": dict(M_block_size=8, K_block_size=8, N_block_size=8, subblock_h=1, subblock_w=4),
    "16k": dict(M_block_size=8, K_block_size=8, N_block_size=8, subblock_h=1, subblock_w=4),
    "32k": dict(M_block_size=16, K_block_size=8, N_block_size=8, subblock_h=1, subblock_w=4),
    "64k": dict(M_block_size=16, K_block_size=8, N_block_size=8, subblock_h=1, subblock_w=4),
    "128k": dict(M_block_size=16, K_block_size=8, N_block_size=8, subblock_h=1, subblock_w=4),
}


# ============================================================================
# NEW CONFIGS (K_block=7, sweep-optimized — valid and faster)
# ============================================================================

NEW_CONFIGS = {
    "4k": dict(M_block_size=8, K_block_size=7, N_block_size=8, subblock_h=1, subblock_w=4),
    "8k": dict(M_block_size=8, K_block_size=7, N_block_size=8, subblock_h=1, subblock_w=4),
    "16k": dict(M_block_size=8, K_block_size=7, N_block_size=8, subblock_h=1, subblock_w=4),
    "32k": dict(M_block_size=16, K_block_size=7, N_block_size=8, subblock_h=1, subblock_w=4),
    "64k": dict(M_block_size=16, K_block_size=7, N_block_size=8, subblock_h=1, subblock_w=4),
    "128k": dict(M_block_size=16, K_block_size=7, N_block_size=8, subblock_h=1, subblock_w=4),
}


# ============================================================================
# HELPERS
# ============================================================================


def _create_fabric_router_config(max_payload_size):
    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = max_payload_size
    return config


def _open_mesh():
    cfg = DEVICE_CFG
    fabric_kwargs = [
        cfg["fabric_config"],
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
    ]
    if cfg["fabric_router_config_payload"] is not None:
        fabric_kwargs.append(_create_fabric_router_config(cfg["fabric_router_config_payload"]))
    ttnn.set_fabric_config(*fabric_kwargs)
    rows, cols = cfg["mesh_shape"]
    return ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(rows, cols))


def _close_mesh(mesh_device):
    ttnn.close_mesh_device(mesh_device)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _run_agmm(mesh_device, M, K, N, cgx, cgy, block_cfg):
    """Execute one all_gather_minimal_matmul_async call with given block config."""
    cfg = DEVICE_CFG
    dtype = ttnn.bfloat8_b
    core_grid = ttnn.CoreCoord(cgx, cgy)

    matmul_config = ttnn.MinimalMatmulConfig(
        M_block_size=block_cfg["M_block_size"],
        K_block_size=block_cfg["K_block_size"],
        N_block_size=block_cfg["N_block_size"],
        subblock_h=block_cfg["subblock_h"],
        subblock_w=block_cfg["subblock_w"],
        compute_with_storage_grid_size=core_grid,
    )

    compute_config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    tt_input = ttnn.from_torch(
        torch.randn((M, K), dtype=torch.float32),
        dtype=dtype,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device, mesh_shape=tuple(mesh_device.shape), dims=[None, cfg["cluster_axis"]]
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
    ccl_semaphore_handles = [
        ttnn.create_global_semaphore(mesh_device, ccl_cores, 0) for _ in range(mesh_device.get_num_devices())
    ]

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
        topology=cfg["topology"],
        cluster_axis=cfg["cluster_axis"],
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


# ============================================================================
# OLD CONFIG TESTS — K_block=8, expected to fail with TT_FATAL
# ============================================================================


@pytest.mark.timeout(600)
@pytest.mark.parametrize("isl", ISLS, ids=[f"{s[0]}_old_config" for s in ISLS])
def test_agmm_old_config(isl):
    """Old config (K_block=8): will FAIL — K_block_size(8) does not divide K_tiles_per_device(28)."""
    isl_name, M, K, N, cgx, cgy = isl
    block_cfg = OLD_CONFIGS[isl_name]

    mesh_device = _open_mesh()
    try:
        _run_agmm(mesh_device, M, K, N, cgx, cgy, block_cfg)
    finally:
        _close_mesh(mesh_device)


# ============================================================================
# NEW CONFIG TESTS — K_block=7, sweep-optimized, should pass
# ============================================================================


@pytest.mark.timeout(600)
@pytest.mark.parametrize("isl", ISLS, ids=[f"{s[0]}_new_config" for s in ISLS])
def test_agmm_new_config(isl):
    """New config (K_block=7): should PASS — sweep-optimized block sizes."""
    isl_name, M, K, N, cgx, cgy = isl
    block_cfg = NEW_CONFIGS[isl_name]

    mesh_device = _open_mesh()
    try:
        _run_agmm(mesh_device, M, K, N, cgx, cgy, block_cfg)
    finally:
        _close_mesh(mesh_device)
