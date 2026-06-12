# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import os

import pytest

import ttnn
from models.common.utility_functions import is_slow_dispatch, is_wormhole_b0
from models.tt_dit.tests.models.wan2_2.test_all_gather_minimal_matmul_async_dev import (
    create_fabric_router_config,
    run_test_linear,
)

LOUDBOX_MESH_CONFIG = {
    "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
    "fabric_router_config": create_fabric_router_config(4096),
    "trace_region_size": 90112,
}


def get_1x8_mesh(mesh_device):
    """Normalize both platforms to a 1x8 mesh.

    Loudbox allocates a 1x8 mesh directly; Galaxy allocates 4x8 and carves a 1x8 submesh.
    Either way the test runs on a single 1x8 mesh, so the per-device work (and therefore the
    M/K/N params) is identical across both configs.
    """
    if mesh_device.shape == ttnn.MeshShape(4, 8):
        return mesh_device.create_submeshes(ttnn.MeshShape(1, 8))[0]
    assert mesh_device.shape == ttnn.MeshShape(1, 8)
    return mesh_device


# This single test exercises both matmul output orientations via the force_transpose axis:
#   transpose    -> 12x9 compute grid, fabric mux on the bottom row    (6 workers/link)
#   no-transpose -> 11x10 compute grid, fabric mux on the right column (5 workers/link)
# Either orientation needs a 12-wide full grid. On Loudbox that 12th column only exists in slow
# dispatch (fast dispatch reserves a column for dispatch); on Galaxy the wider physical grid exposes
# it in fast dispatch too. The grid/worker config is derived from force_transpose in the test body,
# so the parametrize below carries only the per-platform mesh config.
@pytest.mark.parametrize(
    "mesh_device, device_params, topology, num_links, sp_axis, tp_axis, cluster_axis, is_loudbox",
    [
        # Loudbox (1x8): the 12-wide full grid only exists in slow dispatch.
        [(1, 8), LOUDBOX_MESH_CONFIG, ttnn.Topology.Ring, 2, 0, 1, 1, True],
        # Galaxy (1x8): wider physical grid exposes the full grid in fast dispatch too.
        [(1, 8), LOUDBOX_MESH_CONFIG, ttnn.Topology.Ring, 2, 0, 1, 1, False],
        # Galaxy (4x8): allocated as 4x8, carved down to a 1x8 submesh by get_1x8_mesh().
        [(4, 8), LOUDBOX_MESH_CONFIG, ttnn.Topology.Ring, 2, 0, 1, 1, False],
    ],
    ids=["lb1x8", "glx1x8", "glx4x8"],
    indirect=["mesh_device", "device_params"],
)

# 1024, 6144, 2304      # QKV spatial
# 512,  6144, 2304      # QKV prompt
# 1024, 6144, 768.      # to_out spatial
# 512,  6144, 768.      # to_out prompt
# 1024, 6144, 4608.     # ff1/swiglu spatial
# 512,  6144, 4608.     # ff1/swiglu prompt
@pytest.mark.parametrize(
    "M, K, N, use_bias, activation, chunks, fuse_addcmul, M_block_size, K_block_size, N_block_size, subblock_h, subblock_w",
    [
        # TODO: run perf tests
        (1024, 6144, 2304, True, None, 1, False, 12, 4, 7, 4, 1),
        (512, 6144, 2304, True, None, 1, False, 2, 8, 7, 2, 1),
        (1024, 6144, 768, True, None, 1, False, 8, 12, 3, 4, 1),  # DOES TRANSPOSE, NOT useful
        (512, 6144, 768, True, None, 1, False, 10, 8, 3, 1, 3),
        (1024, 6144, 4608, False, None, 1, False, 8, 3, 14, 2, 2),  # OG test case
        (768, 6144, 1536, True, "gelu", 2, False, 3, 4, 5, 3, 1),
    ],
    ids=["flux22-1", "flux22-2", "flux22-3", "flux22-4", "flux22-5", "flux22-6"],
)
@pytest.mark.parametrize("force_transpose", [False, True], ids=["no_transpose", "transpose"])
@pytest.mark.parametrize(
    "mode",
    ["full_fused", "matmul_isolation_fused", "separate"],
    ids=["full_fused", "matmul_isolation_fused", "separate"],
)
@pytest.mark.parametrize(
    # No trace/perf variant: the non-transposed 11x10 config requires slow dispatch (for the 12-wide
    # full grid that hosts the right-column mux), and trace capture is not supported in slow dispatch.
    "enable_trace,num_iters",
    [(False, 2)],
    ids=["check"],
)
def test_linear_AGMM_perf(
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
    is_loudbox,
):
    if is_wormhole_b0():
        pytest.skip("Blackhole config not supported on wormhole_b0")

    # Full 4x8 Galaxy: run on the whole mesh (M sharded across the 4 rows, K gathered across the 8
    # columns). All other configs carve down to a 1x8. The M params are per-device, so on the full
    # 4x8 scale M by the sp-axis size (x4) to keep per-device M at the swept/tuned size.
    if mesh_device.shape == ttnn.MeshShape(4, 8):
        M = M * mesh_device.shape[sp_axis]
    else:
        mesh_device = get_1x8_mesh(mesh_device)

    # Mirror the program factory's grid heuristic (all_gather_minimal_matmul_async_program_factory.cpp):
    # the matmul transposes its core grid when force_transpose is set OR the per-device output is taller
    # than wide (per_device_M > N). The in0 parallelization axis (x when transposed, y otherwise) hosts
    # the fabric mux and must be a multiple of num_links, so we size num_workers_per_link from it rather
    # than skipping mismatched configs.
    #   transpose    -> 12x9 grid,  mux on the bottom row,    in0 axis = x (12)
    #   no-transpose -> 11x10 grid, mux on the right column, in0 axis = y (10)
    # Either orientation needs a 12-wide full grid. On Loudbox that 12th column only exists in slow
    # dispatch (fast dispatch reserves a column for dispatch); on Galaxy the wider physical grid exposes
    # it in fast dispatch too, so only Loudbox forces slow dispatch.
    per_device_M = M // mesh_device.shape[sp_axis]
    transpose_core_grid = force_transpose or (per_device_M > N)
    core_grid_x, core_grid_y = (12, 9) if transpose_core_grid else (11, 10)
    in0_axis_cores = core_grid_x if transpose_core_grid else core_grid_y
    num_workers_per_link = in0_axis_cores // num_links

    if is_loudbox and not is_slow_dispatch():
        pytest.skip("Loudbox needs a 12-wide full grid for the fabric mux (slow dispatch only)")

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
