# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Standalone pytest for fused AllGather+MinimalMatmul with LLaMA 70B W2 dimensions.
Tests both fused and non-fused (separate AG + MM) paths for direct comparison.

Quantifying 4x8 vs 8x8 gain for fused AG+MM (two separate tests; devices close between runs):
  pytest tests/test_fused_ag_mm_w2.py::test_w2_fused_ag_mm_4x8_perf -sv
  pytest tests/test_fused_ag_mm_w2.py::test_w2_fused_ag_mm_8x8_perf -sv
  Compare the two [PERF] ms/iter values; gain = (ms_4x8 - ms_8x8) / ms_4x8 * 100%.

W2 per-device dimensions:
  - M = seqlen (varies)
  - K = 3584 (gathered across 4 devices on tp_axis=1)
  - N = 2048 (output dim per device)

Run examples:
  # Quick correctness check with 8x8 grid, 8k seqlen
  pytest tests/test_fused_ag_mm_w2.py -k "w2_8k and fused and check and wh8x4links1" -sv

  # Fused vs separate comparison at 8k
  pytest tests/test_fused_ag_mm_w2.py -k "w2_8k and wh8x4links1 and check" -sv

  # 4x8 vs 8x8 fused perf (two separate tests)
  pytest tests/test_fused_ag_mm_w2.py::test_w2_fused_ag_mm_4x8_perf -sv
  pytest tests/test_fused_ag_mm_w2.py::test_w2_fused_ag_mm_8x8_perf -sv

  # Try 4x8 grid (harvested-compatible)
  pytest tests/test_fused_ag_mm_w2.py -k "w2_8k and fused and check and 4x8" -sv

  # Run all W2 sizes
  pytest tests/test_fused_ag_mm_w2.py -sv
"""

import pytest
import time
import ttnn

from models.tt_dit.tests.models.wan2_2.test_all_gather_minimal_matmul_async import (
    create_fabric_router_config,
    run_test_linear,
)


@pytest.mark.parametrize(
    "mesh_device, device_params, topology, num_links, num_workers_per_link, sp_axis, tp_axis, core_grid_x, core_grid_y",
    [
        # 8x8 grid (unharvested only, 64 cores)
        [
            (8, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "fabric_router_config": create_fabric_router_config(4096),
                "trace_region_size": 90112,
            },
            ttnn.Topology.Ring,
            1,
            4,
            0,
            1,
            8,
            8,
        ],
        # 4x8 grid (harvested-compatible, 32 cores)
        [
            (8, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "fabric_router_config": create_fabric_router_config(4096),
                "trace_region_size": 90112,
            },
            ttnn.Topology.Ring,
            1,
            4,
            0,
            1,
            4,
            8,
        ],
    ],
    ids=[
        "wh8x4links1",
        "4x8",
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    "M, K, N, force_transpose, use_bias, activation",
    [
        (8192 * 8, 3584, 2048, True, False, None),  # 8k seqlen * 8 devices on sp_axis
        (16384 * 8, 3584, 2048, True, False, None),  # 16k
        (32768 * 8, 3584, 2048, True, False, None),  # 32k
    ],
    ids=[
        "w2_8k",
        "w2_16k",
        "w2_32k",
    ],
)
@pytest.mark.parametrize(
    "use_non_fused",
    [
        True,
        False,
    ],
    ids=["separate", "fused"],
)
@pytest.mark.parametrize(
    "enable_trace,num_iters",
    [
        (False, 1),
        (True, 2),
    ],
    ids=["check", "perf"],
)
def test_w2_fused_ag_mm(
    mesh_device,
    M,
    K,
    N,
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
):
    M_block_size, K_block_size, N_block_size = 8, 8, 8
    subblock_h, subblock_w = 4, 2

    print(
        f"\nW2 Test: M={M}, K={K}, N={N}, grid=({core_grid_x},{core_grid_y}), "
        f"fused={not use_non_fused}, trace={enable_trace}"
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
        use_non_fused=use_non_fused,
        force_transpose=force_transpose,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        use_bias=use_bias,
        activation=activation,
        enable_trace=enable_trace,
        num_iters=num_iters,
        fp32_acc=False,
    )
    elapsed_s = time.perf_counter() - t0
    if enable_trace and num_iters >= 1:
        # Per-iteration time for proof table (elapsed / num_iters)
        per_iter_ms = 1000.0 * elapsed_s / num_iters
        print(
            f"  [PERF] grid=({core_grid_x}x{core_grid_y}) fused={not use_non_fused}: "
            f"elapsed={elapsed_s:.3f}s, {num_iters} iters → ~{per_iter_ms:.2f} ms/iter"
        )

    for n in range(num_iters):
        for i in range(mesh_device.get_num_devices()):
            pcc = check_result[n][i]["pcc"]
            rmse = check_result[n][i]["relative_rmse"]
            print(f"  iter={n}, device={i}: PCC={pcc:.6f}, RMSE={rmse:.4f}")
            assert pcc > 0.999_000, f"PCC too low: {pcc}"
            assert rmse < 0.05, f"RMSE too high: {rmse}"


# ---- Separate perf tests for 4x8 vs 8x8 (devices open/close per test) ----


@pytest.mark.parametrize(
    "mesh_device, device_params, topology, num_links, num_workers_per_link, sp_axis, tp_axis",
    [
        [
            (8, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "fabric_router_config": create_fabric_router_config(4096),
                "trace_region_size": 90112,
            },
            ttnn.Topology.Ring,
            1,
            4,
            0,
            1,
        ],
    ],
    ids=["4x8"],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    "M, K, N",
    [
        (8192 * 8, 3584, 2048),
        (16384 * 8, 3584, 2048),
        (32768 * 8, 3584, 2048),
    ],
    ids=["w2_8k", "w2_16k", "w2_32k"],
)
def test_w2_fused_ag_mm_4x8_perf(
    mesh_device,
    M,
    K,
    N,
    topology,
    num_workers_per_link,
    num_links,
    sp_axis,
    tp_axis,
):
    """Fused AG+MM with 4x8 grid only (32 cores). Run separately from 8x8; compare [PERF] ms/iter."""
    core_grid_x, core_grid_y = 4, 8
    M_block_size, K_block_size, N_block_size = 8, 8, 8
    subblock_h, subblock_w = 4, 2
    enable_trace, num_iters = True, 2

    print(f"\nW2 fused AG+MM 4x8 perf: M={M}, K={K}, N={N}")
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
        enable_trace=enable_trace,
        num_iters=num_iters,
        fp32_acc=False,
    )
    elapsed_s = time.perf_counter() - t0
    per_iter_ms = 1000.0 * elapsed_s / num_iters
    per_iter_us = per_iter_ms * 1000
    print(f"  [PERF] grid=4x8 (32 cores) fused: ~{per_iter_ms:.2f} ms/iter (~{per_iter_us:.0f} us/iter)")

    for n in range(num_iters):
        for i in range(mesh_device.get_num_devices()):
            pcc, rmse = check_result[n][i]["pcc"], check_result[n][i]["relative_rmse"]
            assert pcc > 0.999_000, f"PCC too low: {pcc}"
            assert rmse < 0.05, f"RMSE too high: {rmse}"


@pytest.mark.parametrize(
    "mesh_device, device_params, topology, num_links, num_workers_per_link, sp_axis, tp_axis",
    [
        [
            (8, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "fabric_router_config": create_fabric_router_config(4096),
                "trace_region_size": 90112,
            },
            ttnn.Topology.Ring,
            1,
            4,
            0,
            1,
        ],
    ],
    ids=["8x8"],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    "M, K, N",
    [
        (8192 * 8, 3584, 2048),
        (16384 * 8, 3584, 2048),
        (32768 * 8, 3584, 2048),
    ],
    ids=["w2_8k", "w2_16k", "w2_32k"],
)
def test_w2_fused_ag_mm_8x8_perf(
    mesh_device,
    M,
    K,
    N,
    topology,
    num_workers_per_link,
    num_links,
    sp_axis,
    tp_axis,
):
    """Fused AG+MM with 8x8 grid only (64 cores). Run separately from 4x8; compare [PERF] ms/iter."""
    core_grid_x, core_grid_y = 8, 8
    M_block_size, K_block_size, N_block_size = 8, 8, 8
    subblock_h, subblock_w = 4, 2
    enable_trace, num_iters = True, 2

    print(f"\nW2 fused AG+MM 8x8 perf: M={M}, K={K}, N={N}")
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
        enable_trace=enable_trace,
        num_iters=num_iters,
        fp32_acc=False,
    )
    elapsed_s = time.perf_counter() - t0
    per_iter_ms = 1000.0 * elapsed_s / num_iters
    per_iter_us = per_iter_ms * 1000
    print(f"  [PERF] grid=8x8 (64 cores) fused: ~{per_iter_ms:.2f} ms/iter (~{per_iter_us:.0f} us/iter)")

    for n in range(num_iters):
        for i in range(mesh_device.get_num_devices()):
            pcc, rmse = check_result[n][i]["pcc"], check_result[n][i]["relative_rmse"]
            assert pcc > 0.999_000, f"PCC too low: {pcc}"
            assert rmse < 0.05, f"RMSE too high: {rmse}"
