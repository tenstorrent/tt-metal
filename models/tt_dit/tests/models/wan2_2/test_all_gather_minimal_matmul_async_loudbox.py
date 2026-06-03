# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import os

import pytest

import ttnn
from models.common.utility_functions import is_slow_dispatch, is_wormhole_b0
from models.tt_dit.tests.models.wan2_2.test_all_gather_minimal_matmul_async import (
    create_fabric_router_config,
    run_test_linear,
)

LOUDBOX_MESH_CONFIG = {
    "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
    "fabric_router_config": create_fabric_router_config(4096),
    "trace_region_size": 90112,
}


@pytest.mark.parametrize(
    "mesh_device, device_params, topology, num_links, num_workers_per_link, sp_axis, tp_axis, core_grid_x, core_grid_y, cluster_axis",
    [
        [
            (1, 8),
            LOUDBOX_MESH_CONFIG,
            ttnn.Topology.Ring,
            2,
            6,  # full grid requires force_transpose=True to divide core_grid_x=12
            0,
            1,
            12,
            9,
            1,
        ],
    ],
    ids=["bh1x8fullgrid"],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    "M, K, N, force_transpose, use_bias, activation, chunks, fuse_addcmul, M_block_size, K_block_size, N_block_size, subblock_h, subblock_w",
    [
        # M/K/N tiles: 32/192/24 — K_block=8 → 24 K-blocks, 3 per device on 1x8
        (1024, 6144, 768, True, True, None, 1, False, 8, 8, 6, 2, 2),
        # M/K/N tiles: 16/192/24
        (512, 6144, 768, True, True, None, 1, False, 4, 8, 6, 2, 2),
        # M/K/N tiles: 32/192/144 — N_block=24 on full grid (10-wide)
        (1024, 6144, 4608, True, True, None, 1, False, 1, 4, 24, 1, 4),
        # "good" test cases
        (18944, 5120, 1280, True, True, None, 1, False, 10, 5, 6, 2, 1),
        (18944, 5120, 3456, True, True, None, 1, False, 7, 5, 12, 1, 2),
        (12480, 5120, 3840, True, True, None, 3, False, 7, 5, 16, 1, 2),
        (28800, 5120, 3840, True, True, None, 3, False, 7, 5, 16, 1, 2),
        (4096, 4096, 4096, True, True, None, 1, False, 8, 8, 8, 2, 2),
        (1024, 6144, 4608, False, True, None, 1, False, 1, 4, 24, 1, 4),  # force_transpose=False
    ],
    ids=[
        "m1024_k6144_n768",
        "m512_k6144_n768",
        "ltx",
        "1xdenseattn1",
        "1xff1",
        "1xssg480pqkv",
        "1xssg720pqkv",
        "4ksquare",
        "ltx_ftfalse",
    ],
)
@pytest.mark.parametrize(
    "mode",
    ["full_fused", "matmul_isolation_fused", "separate"],
    ids=["full_fused", "matmul_isolation_fused", "separate"],
)
@pytest.mark.parametrize(
    "enable_trace,num_iters",
    [(False, 3), (True, 2)],
    ids=["check", "perf"],
)
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
    mode,
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
        pytest.skip("Blackhole Loudbox config not supported on wormhole_b0")

    assert mesh_device.shape == ttnn.MeshShape(1, 8)

    if core_grid_x > 11 and not is_slow_dispatch():
        pytest.skip("Fast dispatch mode not supported for core_grid_x > 11")

    use_non_fused = mode == "separate"
    matmul_isolation = mode == "matmul_isolation_fused" and os.environ.get("AGMM_MATMUL_ISOLATION") == "1"

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
        skip_result_check=matmul_isolation or enable_trace,
    )

    if matmul_isolation or enable_trace:
        return

    for n in range(num_iters):
        for c in range(chunks):
            for i in range(mesh_device.get_num_devices()):
                assert check_result[n][c][i]["pcc"] > 0.999_500
                assert check_result[n][c][i]["relative_rmse"] < 0.02


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

# if core_grid_x > 11 and not is_slow_dispatch():
#    pytest.skip("Fast dispatch mode not supported for core_grid_x > 11")

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
