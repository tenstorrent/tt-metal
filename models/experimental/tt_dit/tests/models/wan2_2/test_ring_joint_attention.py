# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from loguru import logger
import pytest
from tests.nightly.t3000.ccl.test_ring_joint_attention import run_ring_joint_sdpa, create_ring_joint_sdpa_submesh
from models.experimental.tt_dit.utils.padding import get_padded_vision_seq_len


@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["bf16"])
@pytest.mark.parametrize(
    "b, nh, base_seq_len, joint_seq_len, d",
    [
        (1, 40, 75776, 0, 128),
        (1, 10, 75600, 0, 128),
    ],
    ids=["wan_baseline", "wan_one_column"],
)
@pytest.mark.parametrize("q_chunk_size", [64, 128, 256], ids=["q64", "q128", "q256"])
@pytest.mark.parametrize("k_chunk_size", [128, 256, 512], ids=["k128", "k256", "k512"])
@pytest.mark.parametrize(
    "n_iters, trace_enabled, skip_check", [(1, False, True), (10, True, True)], ids=["no_trace", "yes_trace"]
)
@pytest.mark.parametrize("num_links", [2, 4], ids=["2link", "4link"])
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
        [0, 8, 1, 4],  # 8x4 RP x UP
        [0, 8, 1, 1],
    ],
    ids=[
        "8rpx4up",
        "8rpx1up",
    ],
)
def test_ring_joint_sdpa(
    mesh_device,
    b,
    nh,
    base_seq_len,
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
    skip_check,
    reset_seeds,
):
    mesh_device_shape = list(mesh_device.shape)
    assert mesh_device_shape[rp_axis] >= rp_factor and mesh_device_shape[up_axis] >= up_factor

    submesh = create_ring_joint_sdpa_submesh(mesh_device, rp_axis, rp_factor, up_axis, up_factor)

    chunk_divisor = max(q_chunk_size, k_chunk_size)
    padded_seq_len = get_padded_vision_seq_len(base_seq_len, chunk_divisor, mesh_device_shape[rp_axis])

    logger.debug(f"RP axis: {rp_axis} factor: {rp_factor}, UP axis: {up_axis} factor: {up_factor}")
    logger.debug(f"submesh: {submesh.shape}")

    run_ring_joint_sdpa(
        submesh,
        b,
        nh,
        padded_seq_len,
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
    )
