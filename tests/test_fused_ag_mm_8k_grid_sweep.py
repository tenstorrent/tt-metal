# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Separate test: fused AG+MM at 8k sizes (M=65536, K=3584, N=2048), mesh (8,4),
sweeping grid 8x8 and 7x8 with different num_links and num_workers_per_link.

Run:
  pytest tests/test_fused_ag_mm_8k_grid_sweep.py -sv
  pytest tests/test_fused_ag_mm_8k_grid_sweep.py -k "8x8" -sv
  pytest tests/test_fused_ag_mm_8k_grid_sweep.py -k "7x8" -sv

Verify K_blocks is the cause of hang: short run with K=4096 (K_blocks=16, divisible by ring_size=4):
  pytest tests/test_fused_ag_mm_8k_grid_sweep.py::test_fused_ag_mm_short_k4096 -sv
"""

import pytest
import time
import ttnn

from models.tt_dit.tests.models.wan2_2.test_all_gather_minimal_matmul_async import (
    create_fabric_router_config,
    run_test_linear,
)


def _device_params_8k():
    return {
        "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
        "fabric_router_config": create_fabric_router_config(2048),
        "trace_region_size": 90112,
    }


def _device_params_original_8x8():
    """Match original test_linear 8x8 rows exactly (fabric_router_config 4096)."""
    return {
        "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
        "fabric_router_config": create_fabric_router_config(4096),
        "trace_region_size": 90112,
    }


def _sweep_params():
    """(core_grid_x, core_grid_y, num_links, num_workers_per_link)."""
    out = []
    for num_links in (1, 2, 4):
        for num_workers in (1, 4, 8):
            out.append((8, 8, num_links, num_workers))
    for num_workers in (1, 4, 7):
        out.append((7, 8, 1, num_workers))
    return out


@pytest.mark.parametrize(
    "mesh_device, device_params, topology, num_links, num_workers_per_link, sp_axis, tp_axis, core_grid_x, core_grid_y",
    [
        [
            (8, 4),
            _device_params_original_8x8(),
            ttnn.Topology.Ring,
            1,
            8,
            0,
            1,
            8,
            8,
        ],
    ],
    ids=["8x8_1link_8w"],
    indirect=["mesh_device", "device_params"],
)
def test_fused_ag_mm_short_k4096(
    mesh_device,
    topology,
    core_grid_x,
    core_grid_y,
    num_workers_per_link,
    num_links,
    sp_axis,
    tp_axis,
):
    """Short run M=32768, K=4096, N=4096 with 8x8, 1 link, 8 workers (valid combo). K_blocks=16. Verifies K_blocks divisible by ring_size when using a supported num_workers_per_link."""
    M, K, N = 32768, 4096, 4096
    M_block_size, K_block_size, N_block_size = 8, 8, 8
    subblock_h, subblock_w = 4, 2
    num_iters = 1

    print(f"\n[Fused AG+MM short K=4096] M={M}, K={K}, N={N}, grid=(8x8), num_links=1, num_workers_per_link=8")
    t0 = time.perf_counter()
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
        use_non_fused=False,
        force_transpose=True,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        use_bias=False,
        activation=None,
        num_iters=num_iters,
        fp32_acc=False,
        sync_timeout_sec=120,
    )
    elapsed_s = time.perf_counter() - t0
    print(f"  [PERF] ~{1000*elapsed_s/num_iters:.2f} ms/iter (K=4096 -> K_blocks=16, passes)")

    for n in range(num_iters):
        for i in range(mesh_device.get_num_devices()):
            pcc, rmse = check_result[n][i]["pcc"], check_result[n][i]["relative_rmse"]
            assert pcc > 0.999_500, f"PCC too low: {pcc}"
            assert rmse < 0.02, f"RMSE too high: {rmse}"


@pytest.mark.parametrize(
    "mesh_device, device_params, topology, num_links, num_workers_per_link, sp_axis, tp_axis, core_grid_x, core_grid_y",
    [
        [
            (8, 4),
            _device_params_8k(),
            ttnn.Topology.Ring,
            num_links,
            num_workers,
            0,
            1,
            gx,
            gy,
        ]
        for (gx, gy, num_links, num_workers) in _sweep_params()
    ],
    ids=[f"{gx}x{gy}_{nl}link_{nw}w" for (gx, gy, nl, nw) in _sweep_params()],
    indirect=["mesh_device", "device_params"],
)
def test_fused_ag_mm_8k_padded_grid_sweep(
    mesh_device,
    topology,
    core_grid_x,
    core_grid_y,
    num_workers_per_link,
    num_links,
    sp_axis,
    tp_axis,
):
    """Our sizes: M=65536, K=3584, N=2048, mesh (8,4). Sweep 8x8 and 7x8 grids with different num_links / num_workers_per_link."""
    M, K, N = 65536, 3584, 2048
    M_block_size, K_block_size, N_block_size = 8, 8, 8
    subblock_h, subblock_w = 4, 2
    num_iters = 1

    print(
        f"\n[Fused AG+MM 8k sweep] M={M}, K={K}, N={N}, grid=({core_grid_x}x{core_grid_y}), "
        f"num_links={num_links}, num_workers_per_link={num_workers_per_link}"
    )
    t0 = time.perf_counter()
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
        use_non_fused=False,
        force_transpose=True,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        use_bias=False,
        activation=None,
        num_iters=num_iters,
        fp32_acc=False,
        sync_timeout_sec=120,
    )
    elapsed_s = time.perf_counter() - t0
    print(
        f"  [PERF] grid=({core_grid_x}x{core_grid_y}) {num_links}link_{num_workers_per_link}w: ~{1000*elapsed_s/num_iters:.2f} ms/iter"
    )

    for n in range(num_iters):
        for i in range(mesh_device.get_num_devices()):
            pcc, rmse = check_result[n][i]["pcc"], check_result[n][i]["relative_rmse"]
            assert pcc > 0.999_500, f"PCC too low: {pcc}"
            assert rmse < 0.02, f"RMSE too high: {rmse}"
