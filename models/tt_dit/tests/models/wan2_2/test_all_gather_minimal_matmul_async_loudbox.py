# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

import ttnn
from models.common.utility_functions import is_wormhole_b0
from models.tt_dit.tests.models.wan2_2.test_all_gather_minimal_matmul_async import (
    create_fabric_router_config,
    run_test_linear,
)

LOUDBOX_MESH_CONFIG = {
    "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
    "fabric_router_config": create_fabric_router_config(4096),
    "trace_region_size": 90112,
}


@pytest.mark.requires_grid_size((11, 10))
@pytest.mark.parametrize(
    "mesh_device, device_params, topology, num_links, num_workers_per_link, sp_axis, tp_axis, core_grid_x, core_grid_y, cluster_axis",
    [
        [
            (1, 8),
            LOUDBOX_MESH_CONFIG,
            ttnn.Topology.Ring,
            1,
            6,
            0,
            1,
            11,
            10,
            1,
        ],
        [
            (1, 8),
            LOUDBOX_MESH_CONFIG,
            ttnn.Topology.Ring,
            2,
            4,
            0,
            1,
            8,
            8,
            1,
        ],
    ],
    ids=["bh1x8fullgrid", "bh1x8partialgrid"],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    "M, K, N, force_transpose, use_bias, activation, chunks, fuse_addcmul, M_block_size, K_block_size, N_block_size, subblock_h, subblock_w",
    [
        # M/K/N tiles: 32/192/24 — K_block=8 → 24 K-blocks, 3 per device on 1x8
        (1024, 6144, 768, True, True, None, 1, False, 8, 8, 6, 2, 2),
        # M/K/N tiles: 16/192/24
        (512, 6144, 768, True, True, None, 1, False, 4, 8, 6, 2, 2),
        # M/K/N tiles: 32/192/144 — N_block=16 → 144/16=9, 16 tiles/core on 10-wide grid
        (1024, 6144, 4608, True, True, None, 1, False, 1, 4, 24, 1, 4),
    ],
    ids=["m1024_k6144_n768", "m512_k6144_n768", "ltx"],
)
@pytest.mark.parametrize("use_non_fused", [False, True], ids=["fused", "separate"])
@pytest.mark.parametrize("enable_trace,num_iters", [(False, 1)], ids=["check"])
def test_linear_loudbox(
    mesh_device,
    M,
    K,
    N,
    M_block_size,
    K_block_size,
    N_block_size,
    subblock_h,
    subblock_w,
    topology,
    core_grid_x,
    core_grid_y,
    num_workers_per_link,
    num_links,
    use_non_fused,
    force_transpose,
    sp_axis,
    tp_axis,
    use_bias,
    activation,
    enable_trace,
    num_iters,
    cluster_axis,
    fuse_addcmul,
    chunks,
):
    if is_wormhole_b0():
        pytest.skip("Blackhole Loudbox config: core grid (11, 10) exceeds wormhole_b0 compute grid (8x8)")

    assert mesh_device.shape == ttnn.MeshShape(1, 8)

    check_result = run_test_linear(
        mesh_device,
        M,
        K,
        N,
        M_block_size,
        K_block_size,
        N_block_size,
        subblock_h,
        subblock_w,
        topology,
        core_grid=ttnn.CoreCoord(core_grid_x, core_grid_y),
        num_workers_per_link=num_workers_per_link,
        num_links=num_links,
        use_non_fused=use_non_fused,
        force_transpose=force_transpose,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        use_bias=use_bias,
        activation=activation,
        enable_trace=enable_trace,
        num_iters=num_iters,
        cluster_axis=cluster_axis,
        fuse_addcmul=fuse_addcmul,
        chunks=chunks,
    )

    for n in range(num_iters):
        for c in range(chunks):
            for i in range(mesh_device.get_num_devices()):
                assert check_result[n][c][i]["pcc"] > 0.999_500
                assert check_result[n][c][i]["relative_rmse"] < 0.02


#
# TODO: is this test needed for loudbox?
# @pytest.mark.requires_grid_size((11, 10))
# @pytest.mark.parametrize(
#     "mesh_device, device_params, topology, num_links, num_workers_per_link, sp_axis, tp_axis, core_grid_x, core_grid_y, cluster_axis",
#     [
#         [
#             (1, 8),
#             LOUDBOX_MESH_CONFIG,
#             ttnn.Topology.Ring,
#             2,
#             6,
#             0,
#             1,
#             11,
#             10,
#             1,
#         ],
#     ],
#     ids=["bh1x8links2"],
#     indirect=["mesh_device", "device_params"],
# )
# @pytest.mark.parametrize("broadcast_gate", [True, False], ids=["broadcast_gate", "full_gate"])
# def test_linear_addcmul_gate_loudbox(
#     mesh_device,
#     topology,
#     num_links,
#     num_workers_per_link,
#     sp_axis,
#     tp_axis,
#     core_grid_x,
#     core_grid_y,
#     cluster_axis,
#     broadcast_gate,
# ):
#     if is_wormhole_b0():
#         pytest.skip("Blackhole Loudbox config: core grid (11, 10) exceeds wormhole_b0 compute grid (8x8)")

#     assert mesh_device.shape == ttnn.MeshShape(1, 8)

#     check_result = run_test_linear(
#         mesh_device,
#         M=768,
#         K=5120,
#         N=1280,
#         M_block_size=2,
#         K_block_size=8,
#         N_block_size=8,
#         subblock_h=2,
#         subblock_w=1,
#         topology=topology,
#         core_grid=ttnn.CoreCoord(core_grid_x, core_grid_y),
#         num_workers_per_link=num_workers_per_link,
#         num_links=num_links,
#         use_bias=True,
#         fuse_addcmul=True,
#         addcmul_scalar=1.0,
#         broadcast_gate=broadcast_gate,
#         use_non_fused=False,
#         sp_axis=sp_axis,
#         tp_axis=tp_axis,
#         cluster_axis=cluster_axis,
#     )
#     for c in range(1):
#         for i in range(mesh_device.get_num_devices()):
#             assert check_result[0][c][i]["pcc"] > 0.999_500
#             assert check_result[0][c][i]["relative_rmse"] < 0.02
