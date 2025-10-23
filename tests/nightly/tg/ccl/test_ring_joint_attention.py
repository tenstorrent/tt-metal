# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from loguru import logger
import pytest
from tests.nightly.t3000.ccl.test_ring_joint_attention import run_ring_joint_sdpa, create_ring_joint_sdpa_submesh


@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["bf16"])
@pytest.mark.parametrize(
    "b, nh, seq_len, joint_seq_len, d, q_chunk_size, k_chunk_size",
    [
        (1, 10, 4096, 333, 64, 128, 512),  # SD3.5
    ],
)
@pytest.mark.parametrize("n_iters, trace_enabled", [(1, False)], ids=["no_trace"])
@pytest.mark.parametrize("num_links", [4], ids=["4link"])
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
@pytest.mark.parametrize(
    "mesh_device",
    [(8, 4)],
    indirect=True,
)
@pytest.mark.parametrize(
    "rp_axis, rp_factor, up_axis, up_factor",
    [
        [0, 8, 1, 1],  # 8x1 RP x UP
    ],
    ids=[
        "8rpx1up",
    ],
)
@pytest.mark.parametrize(
    "mesh_column_index", [0, 1, 2, 3], ids=["mesh_column_0", "mesh_column_1", "mesh_column_2", "mesh_column_3"]
)
def test_mesh_column_ring_joint_sdpa(
    mesh_device,
    b,
    nh,
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
    rp_factor,
    up_axis,
    up_factor,
    all_gather_topology,
    mesh_column_index,
    reset_seeds,
):
    mesh_device_shape = list(mesh_device.shape)
    assert mesh_device_shape[rp_axis] >= rp_factor and mesh_device_shape[up_axis] >= up_factor

    # submesh = create_ring_joint_sdpa_submesh(mesh_device, rp_axis, rp_factor, up_axis, up_factor)
    submesh_shape = [0, 0]
    submesh_shape[rp_axis] = rp_factor
    submesh_shape[up_axis] = up_factor
    submesh = mesh_device.create_submeshes(ttnn.MeshShape(submesh_shape[0], submesh_shape[1]))[mesh_column_index]

    logger.debug(f"RP axis: {rp_axis} factor: {rp_factor}, UP axis: {up_axis} factor: {up_factor}")
    logger.debug(f"submesh: {submesh.shape}")

    run_ring_joint_sdpa(
        submesh,
        b,
        nh,
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
    )


@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["bf16"])
@pytest.mark.parametrize(
    "b, nh, seq_len, joint_seq_len, d, q_chunk_size, k_chunk_size",
    [
        (1, 40, 4096, 333, 64, 128, 512),  # SD3.5
        (1, 40, 32768, 118, 128, 256, 256),  # SD3.5 with hi
        (1, 40, 65536, 118, 128, 256, 256),  # SD3.5 with hi
    ],
    ids=["sd35", "32k", "64k"],
)
@pytest.mark.parametrize("n_iters, trace_enabled", [(1, False), (10, True)], ids=["no_trace", "yes_trace"])
@pytest.mark.parametrize("num_links", [1, 2, 3, 4], ids=["1link", "2link", "3link", "4link"])
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
    [(8, 4)],
    indirect=True,
)
@pytest.mark.parametrize(
    "rp_axis, rp_factor, up_axis, up_factor",
    [
        [0, 8, 1, 4],  # 8x4 RP x UP
        # [0, 8, 1, 2],  # 8x2 RP x UP
        # [0, 8, 1, 1],  # 8x1 RP x UP
        # [1, 4, 0, 4],  # 4x4 RP x UP
        # [1, 4, 0, 2],  # 4x2 RP x UP
        # [1, 4, 0, 1],  # 4x1 RP x UP
        # [1, 2, 0, 2],  # 2x2 RP x UP
    ],
    ids=[
        "8rpx4up",
        #     "8rpx2up",
        #     "8rpx1up",
        #     "4rpx4up",
        #     "4rpx2up",
        #     "4rpx1up",
        #     "2rpx2up",
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

    run_ring_joint_sdpa(
        submesh,
        b,
        nh,
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
    )


@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["bf16"])
@pytest.mark.parametrize(
    "b, nh, seq_len, joint_seq_len, d, q_chunk_size, k_chunk_size",
    [
        (1, 40, 4096, 333, 64, 64, 128),  # SD3.5
    ],
    ids=["sd35"],
)
@pytest.mark.parametrize("n_iters, trace_enabled", [(1, False)], ids=["no_trace"])
@pytest.mark.parametrize("num_links", [1, 2, 3], ids=["1link", "2link", "3link"])
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
    [(8, 4)],
    indirect=True,
)
@pytest.mark.parametrize(
    "rp_axis, rp_factor, up_axis, up_factor",
    [
        [0, 4, 1, 4],  # 4x4 RP x UP
    ],
    ids=[
        "4rpx4up",
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
        )

    assert submesh.num_program_cache_entries() == 1
