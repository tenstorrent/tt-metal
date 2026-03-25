# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import math
from itertools import product

import torch

import ttnn
from loguru import logger
import pytest
from models.tt_dit.tests.unit.test_ring_joint_attention import (
    run_ring_joint_sdpa,
    run_ring_joint_sdpa_model_config,
    run_test_ring_joint_sdpa,
    create_ring_joint_sdpa_submesh,
    wh_t3k_unit_test_params,
    mesh_device_map,
    benchmark_model_input_shapes,
    parallel_config_map,
)


@wh_t3k_unit_test_params
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [
        (
            {"worker_l1_size": 1344544, "trace_region_size": 1000000, "fabric_config": ttnn.FabricConfig.FABRIC_1D},
            ttnn.Topology.Linear,
        ),
    ],
    indirect=["device_params"],
    ids=[
        "line",
    ],
)
@pytest.mark.parametrize("mesh_device, num_links", [mesh_device_map["wh_t3k"]], ids=["2x4"], indirect=["mesh_device"])
def test_ring_joint_sdpa_dit_wh_t3k(
    mesh_device,
    input_shape,
    parallel_config,
    chunk_sizes,
    expected_correctness,
    num_links,
    all_gather_topology,
    reset_seeds,
):
    dtype = ttnn.bfloat16
    n_iters = 1
    trace_enabled = False
    skip_check = False
    pcc_threshold, max_mse = expected_correctness
    q_chunk_size, k_chunk_size = chunk_sizes

    run_test_ring_joint_sdpa(
        mesh_device,
        input_shape,
        parallel_config,
        q_chunk_size,
        k_chunk_size,
        n_iters,
        trace_enabled,
        num_links,
        all_gather_topology,
        skip_check,
        dtype,
        pcc_threshold=pcc_threshold,
        max_mse=max_mse,
    )


@pytest.mark.parametrize(
    "dtype, pcc_threshold",
    [(ttnn.bfloat16, 0.994), (ttnn.bfloat8_b, 0.994), (ttnn.bfloat4_b, 0.8)],
    ids=["bf16", "bf8_b", "bf4_b"],
)
@pytest.mark.parametrize(
    "b, nh, seq_len, joint_seq_len, d, q_chunk_size, k_chunk_size, n_iters, trace_enabled",
    [
        (1, 40, 4096, 333, 64, 128, 512, 1, False),  # SD3.5, no_trace
        (1, 10, 4096, 333, 64, 128, 512, 10, True),  # SD3.5 TG, yes_trace
        (1, 40, 8192, 128, 128, 256, 256, 1, False),
    ],
    ids=["sd35_full-no_trace", "sd35_tg-yes_trace", "small_wan_no_trace"],
)
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [
        (
            {"worker_l1_size": 1344544, "trace_region_size": 200000, "fabric_config": ttnn.FabricConfig.FABRIC_1D},
            ttnn.Topology.Linear,
        ),
    ],
    indirect=["device_params"],
    ids=[
        "line",
    ],
)
@pytest.mark.parametrize(
    "mesh_device",
    [(2, 4)],
    indirect=True,
)
@pytest.mark.parametrize(
    "rp_axis, rp_factor, up_axis, up_factor",
    [
        [0, 2, 1, 4],  # 2x4 RP x UP
        [0, 2, 1, 2],  # 2x2 RP x UP
        [0, 2, 1, 1],  # 2x1 RP x UP
        [1, 2, 0, 2],  # 2x2 UP x RP
        [1, 2, 0, 1],  # 1x2 UP x RP
        [1, 4, 0, 1],  # 1x4 UP x RP
        [1, 8, 0, 1],  # 1x8 UP x RP
    ],
    ids=[
        "2rpx4up",
        "2rpx2up",
        "2rpx1up",
        "2upx2rp",
        "1upx2rp",
        "1upx4rp",
        "1upx8rp",
    ],
)
def test_ring_joint_sdpa(
    mesh_device,
    b,
    nh,
    seq_len,
    joint_seq_len,
    d,
    q_chunk_size,
    k_chunk_size,
    dtype,
    pcc_threshold,
    n_iters,
    trace_enabled,
    num_links,
    rp_axis,
    rp_factor,
    up_axis,
    up_factor,
    all_gather_topology,
    reset_seeds,
):
    if nh % up_factor != 0:
        pytest.skip("nh must be divisible by up_factor")
    if rp_factor == 8 and rp_axis == 1:
        mesh_device.reshape(ttnn.MeshShape(1, 8))

    mesh_device_shape = list(mesh_device.shape)
    assert mesh_device_shape[rp_axis] >= rp_factor and mesh_device_shape[up_axis] >= up_factor

    submesh = create_ring_joint_sdpa_submesh(mesh_device, rp_axis, rp_factor, up_axis, up_factor)

    logger.debug(f"RP axis: {rp_axis} factor: {rp_factor}, UP axis: {up_axis} factor: {up_factor}")
    logger.debug(f"submesh: {submesh.shape}")

    skip_check = False

    run_ring_joint_sdpa(
        submesh,
        b,
        nh,
        seq_len,
        seq_len,
        joint_seq_len,
        d,
        q_chunk_size,
        k_chunk_size,
        dtype,
        n_iters,
        trace_enabled,
        num_links,
        rp_axis,
        up_axis,
        all_gather_topology,
        skip_check,
        pcc_threshold,
    )


@pytest.mark.parametrize(
    "dtype, pcc_threshold",
    [
        (ttnn.bfloat16, 0.994),
        (ttnn.bfloat8_b, 0.944),
        (ttnn.bfloat4_b, 0.8),
    ],
    ids=["bf16", "bf8_b", "bf4_b"],
)
@pytest.mark.parametrize(
    "b, nh, seq_len, joint_seq_len, d, q_chunk_size, k_chunk_size",
    [
        (1, 40, 4096, 333, 64, 64, 128),  # SD3.5
    ],
    ids=["sd35"],
)
@pytest.mark.parametrize("n_iters, trace_enabled", [(1, False)], ids=["no_trace"])
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [
        (
            {"worker_l1_size": 1344544, "trace_region_size": 200000, "fabric_config": ttnn.FabricConfig.FABRIC_1D},
            ttnn.Topology.Linear,
        ),
    ],
    indirect=["device_params"],
    ids=[
        "line",
    ],
)
@pytest.mark.parametrize(
    "mesh_device",
    [(2, 4)],
    indirect=True,
)
@pytest.mark.parametrize(
    "rp_axis, rp_factor, up_axis, up_factor",
    [
        [0, 2, 1, 4],  # 2x4 RP x UP
    ],
    ids=[
        "2rpx4up",
    ],
)
def test_ring_joint_sdpa_program_cache(
    mesh_device,
    b,
    nh,
    seq_len,
    joint_seq_len,
    d,
    q_chunk_size,
    k_chunk_size,
    dtype,
    pcc_threshold,
    n_iters,
    trace_enabled,
    num_links,
    rp_axis,
    rp_factor,
    up_axis,
    up_factor,
    all_gather_topology,
):
    if rp_factor == 8 and rp_axis == 1:
        mesh_device.reshape(ttnn.MeshShape(1, 8))

    mesh_device_shape = list(mesh_device.shape)
    assert mesh_device_shape[rp_axis] >= rp_factor and mesh_device_shape[up_axis] >= up_factor
    submesh = create_ring_joint_sdpa_submesh(mesh_device, rp_axis, rp_factor, up_axis, up_factor)

    logger.debug(f"RP axis: {rp_axis} factor: {rp_factor}, UP axis: {up_axis} factor: {up_factor}")
    logger.debug(f"submesh: {submesh.shape}")

    skip_check = False

    dummy_tensors = []
    for i in range(3):
        dummy_tensors.append(
            ttnn.from_torch(
                torch.rand((b, nh, seq_len, d)),
                device=submesh,
                layout=ttnn.TILE_LAYOUT,
                dtype=dtype,
                mesh_mapper=ttnn.ShardTensor2dMesh(submesh, mesh_shape=tuple(submesh.shape), dims=[None, None]),
            )
        )

        run_ring_joint_sdpa(
            submesh,
            b,
            nh,
            seq_len,
            seq_len,
            joint_seq_len,
            d,
            q_chunk_size,
            k_chunk_size,
            dtype,
            n_iters,
            trace_enabled,
            num_links,
            rp_axis,
            up_axis,
            all_gather_topology,
            skip_check,
            pcc_threshold,
        )

    assert submesh.num_program_cache_entries() == 1


# ===========================================================================
# Model-config regression tests — reproduce code-size failures with exact
# model shapes, chunk sizes, grid layouts, and default device_params.
# ===========================================================================


@pytest.mark.parametrize("mesh_device", [(2, 4)], ids=["2x4"], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
def test_ring_joint_sdpa_sd35_model_config(mesh_device, reset_seeds):
    """SD3.5: heads=38, d=64, seq=4096, joint=333, sp4×tp2, chunks=(256,512)."""
    run_ring_joint_sdpa_model_config(
        mesh_device,
        b=1,
        nh=38,
        base_seq_len=4096,
        joint_seq_len=333,
        d=64,
        q_chunk_size=256,
        k_chunk_size=512,
        rp_axis=1,
        rp_factor=4,
        up_axis=0,
        up_factor=2,
        num_links=1,
        ccl_reserve_last_column=False,
        use_column_major_ccl=False,
        use_wormhole_compute_kernel_config=True,
    )


@pytest.mark.parametrize("mesh_device", [(2, 4)], ids=["2x4"], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
def test_ring_joint_sdpa_wan_14b_720p_model_config(mesh_device, reset_seeds):
    """Wan2.2 14B-720p: heads=40, d=128, seq=75600, joint=0, sp2×tp4, chunks=(256,256)."""
    run_ring_joint_sdpa_model_config(
        mesh_device,
        b=1,
        nh=40,
        base_seq_len=75600,
        joint_seq_len=0,
        d=128,
        q_chunk_size=256,
        k_chunk_size=256,
        rp_axis=0,
        rp_factor=2,
        up_axis=1,
        up_factor=4,
        num_links=1,
        ccl_reserve_last_column=True,
        use_column_major_ccl=True,
        use_wormhole_compute_kernel_config=False,
        pcc_threshold=0.9994,
    )


@pytest.mark.parametrize("mesh_device", [(2, 4)], ids=["2x4"], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
def test_ring_joint_sdpa_mochi_model_config(mesh_device, reset_seeds):
    """Mochi: heads=24, d=128, seq=4000, joint=118, sp2×tp2, chunks=(256,256)."""
    run_ring_joint_sdpa_model_config(
        mesh_device,
        b=1,
        nh=24,
        base_seq_len=4000,
        joint_seq_len=118,
        d=128,
        q_chunk_size=256,
        k_chunk_size=256,
        rp_axis=0,
        rp_factor=2,
        up_axis=1,
        up_factor=2,
        num_links=1,
        ccl_reserve_last_column=False,
        use_column_major_ccl=False,
        use_wormhole_compute_kernel_config=False,
    )


# ============================================================================
# PERFORMANCE TABLE TEST — Math Utilization for WH T3K
# ============================================================================

# WH T3K grid constants (logical, before harvesting)
WH_T3K_GRID_COLS = 8
WH_T3K_GRID_ROWS = 8
WH_T3K_CCL_COLUMN = 1  # Last column reserved for CCL

PERF_Q_CHUNK_SIZES = [128, 256]
PERF_K_CHUNK_SIZES = [256, 512]


from tests.nightly.sdpa_perf_utils import post_process_ops_log, compute_cores_used, compute_math_utilization


# --- Sweep perf impl: runs one config with skip_check for profiling ---
@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Performance test - skip on CI")
@pytest.mark.parametrize("q_chunk_size", PERF_Q_CHUNK_SIZES, ids=[f"q{s}" for s in PERF_Q_CHUNK_SIZES])
@pytest.mark.parametrize("k_chunk_size", PERF_K_CHUNK_SIZES, ids=[f"k{s}" for s in PERF_K_CHUNK_SIZES])
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [
        (
            {"worker_l1_size": 1344544, "trace_region_size": 1000000, "fabric_config": ttnn.FabricConfig.FABRIC_1D},
            ttnn.Topology.Linear,
        ),
    ],
    indirect=["device_params"],
    ids=["line"],
)
@pytest.mark.parametrize("mesh_device, num_links", [mesh_device_map["wh_t3k"]], ids=["2x4"], indirect=["mesh_device"])
@pytest.mark.parametrize(
    "input_shape, parallel_config",
    [(benchmark_model_input_shapes[k], parallel_config_map["wh_t3k"][k]) for k in benchmark_model_input_shapes],
    ids=list(benchmark_model_input_shapes.keys()),
)
def test_ring_joint_sdpa_sweep_perf(
    mesh_device,
    input_shape,
    parallel_config,
    q_chunk_size,
    k_chunk_size,
    num_links,
    all_gather_topology,
    reset_seeds,
):
    """Run ring joint SDPA with skip_check for performance profiling."""
    run_test_ring_joint_sdpa(
        mesh_device,
        input_shape,
        parallel_config,
        q_chunk_size,
        k_chunk_size,
        n_iters=1,
        trace_enabled=False,
        num_links=num_links,
        all_gather_topology=all_gather_topology,
        skip_check=True,
        dtype=ttnn.bfloat16,
    )


# --- Perf table: spawns profiled subprocesses and computes math utilization ---
@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Performance test - skip on CI")
@pytest.mark.timeout(1000)
@pytest.mark.parametrize(
    "model_id",
    list(benchmark_model_input_shapes.keys()),
)
def test_ring_joint_sdpa_create_perf_table(model_id):
    """
    Sweep chunk sizes for ring joint attention on WH T3K and print a performance table
    with math utilization. Requires TT_METAL_DEVICE_PROFILER=1.
    """
    from tracy.process_model_log import run_device_profiler

    shape = benchmark_model_input_shapes[model_id]
    parallel = parallel_config_map["wh_t3k"][model_id]
    b, nh, base_seq_len, joint_seq_len, d = shape
    rp_axis, rp_factor, up_axis, up_factor = parallel
    ring_size = rp_factor
    heads_per_device = nh / up_factor
    local_seq_len = base_seq_len // ring_size

    subdir = "t3k_ring_joint_sdpa_perf"
    perf_results = []

    for q_chunk_size, k_chunk_size in product(PERF_Q_CHUNK_SIZES, PERF_K_CHUNK_SIZES):
        # Parametrize ID: input_shape-mesh_device-device_params-k_chunk-q_chunk
        test_id = f"wormhole_b0-{model_id}-2x4-line-k{k_chunk_size}-q{q_chunk_size}"
        command = (
            f"pytest tests/nightly/t3000/ccl/"
            f"test_ring_joint_attention.py::"
            f"test_ring_joint_sdpa_sweep_perf"
            f"[{test_id}]"
        )

        try:
            run_device_profiler(command, subdir, device_analysis_types=["device_kernel_duration"])

            float_cols = ["CORE COUNT", "DEVICE KERNEL DURATION [ns]"]
            r = post_process_ops_log(subdir, float_columns=float_cols, sum_vals=False)

            measured_core_count = int(r["CORE COUNT"][0]) if len(r["CORE COUNT"]) > 0 else 0
            duration_ns = (
                int(r["DEVICE KERNEL DURATION [ns]"].max()) if len(r["DEVICE KERNEL DURATION [ns]"]) > 0 else 0
            )

            # Use measured core count from Tracy (accounts for harvesting).
            # Subtract CCL column cores to get SDPA-only count.
            effective_cores = measured_core_count - measured_core_count % WH_T3K_GRID_COLS
            sdpa_cores = effective_cores - WH_T3K_CCL_COLUMN * (effective_cores // WH_T3K_GRID_COLS)
            cores_used = compute_cores_used(base_seq_len, q_chunk_size, sdpa_cores, heads_per_device, ring_size)
            cores_idle = sdpa_cores - cores_used
            utilization = compute_math_utilization(
                local_seq_len,
                base_seq_len,
                d,
                heads_per_device,
                duration_ns,
                effective_cores,
                arch="wormhole_b0",
            )

            q_num_chunks = math.ceil(local_seq_len / q_chunk_size)
            max_q_parallel = sdpa_cores // int(heads_per_device) if heads_per_device > 0 else 1
            q_per_core = math.ceil(q_num_chunks / max_q_parallel) if max_q_parallel > 0 else q_num_chunks
            k_num_chunks = math.ceil(base_seq_len / k_chunk_size)
            iters_per_core = q_per_core * k_num_chunks

            perf_results.append(
                {
                    "q_chunk_size": q_chunk_size,
                    "k_chunk_size": k_chunk_size,
                    "measured_core_count": measured_core_count,
                    "cores_used": cores_used,
                    "cores_idle": cores_idle,
                    "iters_per_core": iters_per_core,
                    "duration_ns": duration_ns,
                    "duration_ms": duration_ns / 1e6,
                    "utilization": utilization,
                }
            )
            logger.info(
                f"q={q_chunk_size}, k={k_chunk_size}: {duration_ns/1e6:.3f} ms, "
                f"util={utilization:.1f}%, cores={cores_used}/{sdpa_cores}"
            )

        except Exception as e:
            if isinstance(e, KeyboardInterrupt):
                raise
            logger.error(f"Error with q={q_chunk_size}, k={k_chunk_size}: {e}")
            perf_results.append({"q_chunk_size": q_chunk_size, "k_chunk_size": k_chunk_size, "duration_ns": None})

    # Sort by duration (best first)
    valid_results = [r for r in perf_results if r["duration_ns"] is not None]
    valid_results.sort(key=lambda x: x["duration_ns"])

    mm_flops = 4 * base_seq_len * base_seq_len * d * nh

    print(f"\n{'='*140}")
    print(f"WH T3K Ring Joint Attention Perf: {model_id} — b={b}, nh={nh}, s={base_seq_len}, d={d}")
    print(f"Ring size: {ring_size}, TP: {up_factor}, Heads/device: {heads_per_device:.0f}, Local seq: {local_seq_len}")
    measured_sdpa = valid_results[0]["cores_used"] + valid_results[0]["cores_idle"] if valid_results else 0
    print(
        f"Total MM FLOPs: {mm_flops/1e9:.2f} GFLOPs, SDPA cores: {measured_sdpa} (from Tracy, accounts for harvesting)"
    )
    print(f"{'='*140}")
    header = "| Rank | q_chunk | k_chunk | Duration (ms) | Cores Used | Cores Idle | Iters/Core | Math Util |"
    sep = "|------|---------|---------|---------------|------------|------------|------------|-----------|"
    print(header)
    print(sep)

    for rank, result in enumerate(valid_results, 1):
        print(
            f"| {rank:4d} | {int(result['q_chunk_size']):7d} | {int(result['k_chunk_size']):7d} | "
            f"{result['duration_ms']:13.3f} | {int(result['cores_used']):10d} | {int(result['cores_idle']):10d} | "
            f"{int(result['iters_per_core']):10d} | {result['utilization']:8.1f}% |"
        )

    if valid_results:
        best = valid_results[0]
        print(
            f"\nBest: q={best['q_chunk_size']}, k={best['k_chunk_size']} — "
            f"{best['duration_ms']:.3f} ms, {best['utilization']:.1f}% math util, "
            f"{best['cores_used']} cores, {best['iters_per_core']} iters/core"
        )
    print(f"{'='*140}\n")
