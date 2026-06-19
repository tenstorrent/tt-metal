# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import math

import pytest

import ttnn
from models.tt_dit.utils.padding import get_padded_vision_seq_len

# Op-level test for ttnn.transformer.exp_ring_joint_scaled_dot_product_attention.
# The shared runners (run_exp_ring_joint_sdpa / run_test_exp_ring_joint_sdpa) live in the
# sibling ring_joint_sdpa_test_common module, alongside the regular ring_joint runners they
# share input-setup and verification helpers with.
from tests.ttnn.unit_tests.operations.sdpa.ring_joint_sdpa_test_common import (
    create_ring_joint_sdpa_submesh,
    run_exp_ring_joint_sdpa,
)


def create_fabric_router_config(max_payload_size=8192):
    config = ttnn.FabricRouterConfig()
    config.max_packet_payload_size_bytes = max_payload_size
    return config


@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [
        (
            {
                "worker_l1_size": 1344544,
                "trace_region_size": 1000000,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "fabric_router_config": create_fabric_router_config(8192),
            },
            ttnn.Topology.Ring,
        ),
    ],
    indirect=["device_params"],
    ids=["ring"],
)
@pytest.mark.parametrize(
    "mesh_device, num_links, nh, base_seq_len, rp_axis, rp_factor, up_axis, up_factor",
    [
        ((4, 32), 2, 40, 75600, 1, 32, 0, 4),
        ((4, 8), 2, 40, 18944, 1, 8, 0, 4),
        ((1, 4), 2, 10, 8960, 1, 4, 0, 1),
    ],
    ids=["4x32", "4x8", "1x4"],
    indirect=["mesh_device"],
)
@pytest.mark.skipif(
    ttnn.cluster.get_cluster_type() != ttnn.cluster.ClusterType.BLACKHOLE_GALAXY,
    reason="test_ring_joint_sdpa_dit_bh_glx requires a Blackhole Galaxy cluster",
)
def test_exp_ring_joint_sdpa_dit_bh_glx_custom(
    mesh_device,
    num_links,
    nh,
    base_seq_len,
    rp_axis,
    rp_factor,
    up_axis,
    up_factor,
    all_gather_topology,
    reset_seeds,
):
    dtype = ttnn.bfloat16
    b, joint_seq_len, d = 1, 0, 128
    q_chunk_size = 224
    k_chunk_size = 512
    n_iters = 5
    trace_enabled = False
    skip_check = False
    pcc_threshold = 0.9993
    max_mse = 8e-5

    if nh % up_factor != 0:
        nh = math.ceil(nh / up_factor) * up_factor
    submesh = create_ring_joint_sdpa_submesh(mesh_device, rp_axis, rp_factor, up_axis, up_factor)
    padded_seq_len = get_padded_vision_seq_len(base_seq_len, list(mesh_device.shape)[rp_axis])

    run_exp_ring_joint_sdpa(
        submesh,
        b,
        nh,
        base_seq_len,
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
        pcc_threshold,
        max_mse=max_mse,
    )
