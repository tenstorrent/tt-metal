# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from loguru import logger
import pytest
from models.tt_dit.tests.unit.test_ring_joint_attention import (
    run_ring_joint_sdpa,
    run_test_ring_joint_sdpa,
    create_ring_joint_sdpa_submesh,
    bh_qb_ge_unit_test_params,
    mesh_device_map,
)


@pytest.fixture(autouse=True)
def skip_on_large_clusters():
    num_devices = ttnn.get_num_devices()
    if num_devices > 4:
        pytest.skip(
            f"2x2 submesh fabric init not supported on clusters with {num_devices} devices (e.g. 8xP150 LoudBox)"
        )


@bh_qb_ge_unit_test_params
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
@pytest.mark.parametrize("mesh_device, num_links", [mesh_device_map["bh_qb_ge"]], ids=["2x2"], indirect=["mesh_device"])
def test_ring_joint_sdpa_dit_bh_qb_ge(
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
    [
        (ttnn.bfloat16, 0.994),
        (ttnn.bfloat8_b, 0.994),
        (ttnn.bfloat4_b, 0.8),
    ],
    ids=["bf16", "bf8_b", "bf4_b"],
)
@pytest.mark.parametrize(
    "b, nh, seq_len, joint_seq_len, d, q_chunk_size, k_chunk_size",
    [
        (1, 38, 4096, 333, 64, 256, 512),  # SD3.5
    ],
    ids=["sd35"],
)
@pytest.mark.parametrize("n_iters, trace_enabled", [(3, False)], ids=["no_trace"])
@pytest.mark.parametrize("num_links", [2])
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
    [(2, 2)],
    indirect=True,
)
@pytest.mark.parametrize(
    "rp_axis, rp_factor, up_axis, up_factor",
    [
        [0, 2, 1, 2],  # 2x2 RP x UP
    ],
    ids=[
        "2rpx2up",
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
    mesh_device_shape = list(mesh_device.shape)
    assert mesh_device_shape[rp_axis] >= rp_factor and mesh_device_shape[up_axis] >= up_factor
    submesh = create_ring_joint_sdpa_submesh(mesh_device, rp_axis, rp_factor, up_axis, up_factor)

    logger.debug(f"RP axis: {rp_axis} factor: {rp_factor}, UP axis: {up_axis} factor: {up_factor}")
    logger.debug(f"submesh: {submesh.shape}")

    skip_check = False

    # Run the op n_iters (>1) times within a SINGLE run_ring_joint_sdpa invocation. The op must be
    # exercised under one sub-device-manager lifetime: run_ring_joint_sdpa loads a sub-device manager
    # on entry, which clears the program cache (mesh_device.cpp clear_program_cache on manager switch),
    # so looping the whole helper would clear the cache between calls and defeat the reuse check.
    # run_ring_joint_sdpa's internal loop runs the op n_iters times with distinct per-iter persistent
    # buffers and global semaphores, so cache reuse is still validated against address variation.
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

    assert submesh.cache_entries_counter.total == 1
