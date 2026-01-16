# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from loguru import logger
import pytest
from models.experimental.tt_dit.tests.unit.test_ring_joint_attention import (
    run_ring_joint_sdpa,
    run_test_ring_joint_sdpa,
    create_ring_joint_sdpa_submesh,
    wh_t3k_unit_test_params,
    mesh_device_map,
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
