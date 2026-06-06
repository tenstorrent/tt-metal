# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Run the fused AGMM no-transpose path on a 1x8 SUBMESH of the Galaxy 4x8.

Purpose: discriminate whether the ~2x fused matmul slowdown on Galaxy (vs LoudBox 1x8 parity) is
from the 4 concurrent sp-row rings / 32-chip-active environment, by running the SAME per-device
matmul on just one 1x8 row of the Galaxy (8 of 32 chips active).

Opening a (1,8) mesh directly on the Galaxy fails (partial-mesh ethernet handshake / fabric router
sync timeout), so we open the full (4,8) mesh and carve a (1,8) submesh. M is the per-device size
(= Galaxy M // 4) so the per-chip matmul matches the 4x8 dev.py no_transpose run exactly, with the
same block sizes (grid 11x10, links=2, workers=5).

  full_fused           : pytest <this> -k full_fused
  matmul_isolation     : AGMM_MATMUL_ISOLATION=1 pytest <this> -k matmul_isolation_fused
  separate             : pytest <this> -k separate
"""

import os

import pytest

import ttnn
from models.tt_dit.tests.models.wan2_2.test_all_gather_minimal_matmul_async_dev import (
    create_fabric_router_config,
    run_test_linear,
)

GALAXY_MESH_CONFIG = {
    "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
    "fabric_router_config": create_fabric_router_config(4096),
    "trace_region_size": 90112,
}


@pytest.mark.parametrize(
    "mesh_device, device_params, topology, num_links, num_workers_per_link, sp_axis, tp_axis, core_grid_x, core_grid_y, cluster_axis",
    [
        [
            (4, 8),  # open the FULL Galaxy; submesh is carved below
            GALAXY_MESH_CONFIG,
            ttnn.Topology.Ring,
            2,
            5,
            0,
            1,
            11,
            10,
            1,
        ],
    ],
    ids=["bh_submesh_1x8"],
    indirect=["mesh_device", "device_params"],
)
# Per-device shapes (= Galaxy full-M // 4) with the same block sizes as dev.py test_linear_no_transpose.
@pytest.mark.parametrize(
    "M, K, N, use_bias, activation, chunks, M_block_size, K_block_size, N_block_size, subblock_h, subblock_w",
    [
        (1024, 6144, 2304, True, None, 1, 4, 4, 7, 4, 1),
        (512, 6144, 2304, True, None, 1, 2, 6, 7, 2, 1),
        (1024, 6144, 768, True, None, 1, 3, 8, 3, 3, 1),  # m>N -> transposes -> skipped
        (512, 6144, 768, True, None, 1, 2, 8, 3, 1, 3),
        (1024, 6144, 4608, False, None, 1, 4, 6, 7, 4, 1),
        (768, 6144, 1536, True, "gelu", 2, 3, 8, 5, 3, 1),
    ],
    ids=["flux22-1", "flux22-2", "flux22-3", "flux22-4", "flux22-5", "flux22-6"],
)
@pytest.mark.parametrize(
    "mode",
    ["full_fused", "matmul_isolation_fused", "separate"],
    ids=["full_fused", "matmul_isolation_fused", "separate"],
)
def test_linear_submesh_1x8(
    mesh_device,
    M,
    K,
    N,
    use_bias,
    activation,
    chunks,
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
    sp_axis,
    tp_axis,
    cluster_axis,
):
    assert mesh_device.shape == ttnn.MeshShape(4, 8)
    submesh = mesh_device.create_submesh(ttnn.MeshShape(1, 8))

    use_non_fused = mode == "separate"
    matmul_isolation = mode == "matmul_isolation_fused" and os.environ.get("AGMM_MATMUL_ISOLATION") == "1"

    # Same transpose/skip rule as dev.py no_transpose (force_transpose=False -> per_device_M > N).
    per_device_M = M // submesh.shape[sp_axis]
    transpose_core_grid = per_device_M > N
    relevant_core_grid_dim = core_grid_x if transpose_core_grid else core_grid_y
    if relevant_core_grid_dim % num_links != 0:
        pytest.skip(
            f"num_links ({num_links}) does not divide relevant core grid dim "
            f"({'x' if transpose_core_grid else 'y'}={relevant_core_grid_dim})"
        )

    check_result = run_test_linear(
        submesh,
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
        force_transpose=False,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        use_bias=use_bias,
        activation=activation,
        enable_trace=False,
        num_iters=3,
        cluster_axis=cluster_axis,
        chunks=chunks,
        skip_result_check=matmul_isolation,
    )

    if matmul_isolation:
        return
    for n in range(num_iters_default := 3):
        for c in range(chunks):
            for i in range(submesh.get_num_devices()):
                assert check_result[n][c][i]["pcc"] > 0.999_500
