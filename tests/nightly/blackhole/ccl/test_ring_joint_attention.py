# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch

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
@pytest.mark.parametrize("n_iters, trace_enabled", [(1, False)], ids=["no_trace"])
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
