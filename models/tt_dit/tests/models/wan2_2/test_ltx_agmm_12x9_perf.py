# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""AG / MM / AGMM perf sweep for the LTX `grid_12_9_configs` matmul shapes.

Reuses the helpers from ``test_all_gather_minimal_matmul_async`` and pins the
``bh4x8links2`` (mesh 4x8, cluster_axis=0 -> 4-device K-gather ring, core grid
12x9) configuration.

* ``separate`` (use_non_fused=True): runs all_gather_async (AG) then
  minimal_matmul (MM) as two ops -> gives AG and MM device times.
* ``fused`` (use_non_fused=False): runs all_gather_minimal_matmul_async (AGMM)
  as one op -> gives AGMM device time. Works for all shapes including N=32.

Run under tracy to capture per-op device durations, e.g.::

    python -m tracy -r -p -v -o benchmarks/tracy/ltx_agmm_12x9 -n ltx_agmm_12x9 \
        -m pytest models/tt_dit/tests/models/wan2_2/test_ltx_agmm_12x9_perf.py -v -s
"""

import pytest

import ttnn
from models.tt_dit.tests.models.wan2_2.test_all_gather_minimal_matmul_async import (
    create_fabric_router_config,
    run_test_linear,
)

# (M, K, N, M_block_size, K_block_size, N_block_size, subblock_h, subblock_w)
# Taken verbatim from grid_12_9_configs in models/tt_dit/utils/matmul.py (lines 159-175).
LTX_12X9_SHAPES = [
    (1216, 4096, 32, 8, 8, 1, 4, 1),
    (1216, 4096, 3072, 4, 8, 12, 1, 4),
    (1216, 4096, 1024, 4, 8, 4, 1, 4),
    (1216, 4096, 512, 4, 8, 2, 2, 2),
    (1216, 2048, 1024, 4, 8, 4, 1, 4),
    (1216, 4096, 4096, 4, 8, 16, 1, 4),
    (4864, 4096, 32, 20, 8, 1, 4, 1),
    (4864, 4096, 3072, 10, 4, 12, 1, 4),
    (4864, 4096, 1024, 8, 8, 8, 2, 2),
    (4864, 4096, 512, 8, 8, 2, 4, 1),
    (4864, 2048, 1024, 5, 8, 4, 1, 4),
    (4864, 4096, 4096, 5, 8, 16, 1, 4),
    (256, 2048, 1024, 2, 8, 4, 1, 4),
    (32, 2048, 32, 1, 8, 1, 1, 1),
    (32, 2048, 1536, 1, 4, 16, 1, 4),
    (32, 2048, 512, 1, 8, 2, 1, 2),
    (32, 2048, 2048, 1, 4, 12, 1, 4),
]

_SHAPE_IDS = [f"{m}x{k}x{n}" for (m, k, n, *_rest) in LTX_12X9_SHAPES]


@pytest.mark.parametrize(
    "mesh_device, device_params, topology, num_links, num_workers_per_link, sp_axis, tp_axis, core_grid_x, core_grid_y, cluster_axis",
    [
        [
            (4, 8),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "fabric_router_config": create_fabric_router_config(4096),
                "trace_region_size": 90112,
            },
            ttnn.Topology.Ring,
            2,
            6,
            1,
            0,
            12,
            9,
            0,
        ],
    ],
    ids=["bh4x8links2"],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    "M, K, N, M_block_size, K_block_size, N_block_size, subblock_h, subblock_w",
    LTX_12X9_SHAPES,
    ids=_SHAPE_IDS,
)
@pytest.mark.parametrize("use_non_fused", [True, False], ids=["separate", "fused"])
def test_ltx_agmm_12x9_perf(
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
    sp_axis,
    tp_axis,
    cluster_axis,
    use_non_fused,
):
    from models.tt_dit.tests.models.wan2_2.test_all_gather_minimal_matmul_async import _create_cluster_submesh

    submesh = _create_cluster_submesh(mesh_device, cluster_axis)
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
        use_bias=True,
        use_non_fused=use_non_fused,
        force_transpose=True,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        cluster_axis=cluster_axis,
        enable_trace=True,
        num_iters=2,
    )

    for n in range(len(check_result)):
        for c in range(len(check_result[n])):
            for i in range(len(check_result[n][c])):
                assert check_result[n][c][i]["pcc"] > 0.999_500


# N-fold variant: fold M into two 12x4 bands on a 12x8 grid (m_fold=2), so each core owns 2x the N
# (more matmul to hide the in0 fabric gather). num_buffers_per_channel halved 48->24 keeps mux L1 flat
# while channels/mux double 6->12. Fused AGMM only.
@pytest.mark.parametrize(
    "mesh_device, device_params, topology, num_links, num_workers_per_link, sp_axis, tp_axis, core_grid_x, core_grid_y, cluster_axis",
    [
        [
            (4, 8),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "fabric_router_config": create_fabric_router_config(4096),
                "trace_region_size": 90112,
            },
            ttnn.Topology.Ring,
            2,
            6,
            1,
            0,
            12,
            8,
            0,
        ],
    ],
    ids=["bh4x8links2"],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    "M, K, N, M_block_size, K_block_size, N_block_size, subblock_h, subblock_w",
    LTX_12X9_SHAPES,
    ids=_SHAPE_IDS,
)
def test_ltx_agmm_fold_12x8_perf(
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
    sp_axis,
    tp_axis,
    cluster_axis,
):
    from models.tt_dit.tests.models.wan2_2.test_all_gather_minimal_matmul_async import _create_cluster_submesh

    submesh = _create_cluster_submesh(mesh_device, cluster_axis)
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
        use_bias=True,
        use_non_fused=False,
        force_transpose=True,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        cluster_axis=cluster_axis,
        enable_trace=True,
        num_iters=2,
        m_fold=2,
        num_buffers_per_channel=24,
    )

    for n in range(len(check_result)):
        for c in range(len(check_result[n])):
            for i in range(len(check_result[n][c])):
                assert check_result[n][c][i]["pcc"] > 0.999_500


# Buffer-starvation isolation: plain 12x9 (no fold), vary ONLY num_buffers_per_channel. If fab-send and
# WAIT-CONSUMER balloon at 24 vs 48, mux buffer depth is the mechanism behind the fold's fabric slowdown.
@pytest.mark.parametrize(
    "mesh_device, device_params, topology, num_links, num_workers_per_link, sp_axis, tp_axis, core_grid_x, core_grid_y, cluster_axis",
    [
        [
            (4, 8),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "fabric_router_config": create_fabric_router_config(4096),
                "trace_region_size": 90112,
            },
            ttnn.Topology.Ring,
            2,
            6,
            1,
            0,
            12,
            4,
            0,
        ],
    ],
    ids=["bh4x8links2"],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    "M, K, N, M_block_size, K_block_size, N_block_size, subblock_h, subblock_w",
    LTX_12X9_SHAPES,
    ids=_SHAPE_IDS,
)
@pytest.mark.parametrize("num_buffers", [48, 24], ids=["buf48", "buf24"])
def test_ltx_agmm_buffer_proof(
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
    sp_axis,
    tp_axis,
    cluster_axis,
    num_buffers,
):
    from models.tt_dit.tests.models.wan2_2.test_all_gather_minimal_matmul_async import _create_cluster_submesh

    submesh = _create_cluster_submesh(mesh_device, cluster_axis)
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
        use_bias=True,
        use_non_fused=False,
        force_transpose=True,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        cluster_axis=cluster_axis,
        enable_trace=True,
        num_iters=2,
        m_fold=1,
        num_buffers_per_channel=num_buffers,
    )

    for n in range(len(check_result)):
        for c in range(len(check_result[n])):
            for i in range(len(check_result[n][c])):
                assert check_result[n][c][i]["pcc"] > 0.999_500
