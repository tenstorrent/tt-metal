# SPDX-License-Identifier: Apache-2.0
"""whcheck: exp ring joint SDPA on a Wormhole T3K ring (1x8), legacy config. Same file on BASE and HEAD."""
import math
import pytest
import ttnn
from .test_exp_ring_joint_attention import create_fabric_router_config, run_exp_ring_joint_sdpa
from .test_ring_joint_attention import create_ring_joint_sdpa_submesh
from ...utils.padding import get_padded_vision_seq_len


@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [({"trace_region_size": 1000000, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
       "fabric_router_config": create_fabric_router_config(8192)}, ttnn.Topology.Ring)],
    indirect=["device_params"], ids=["ring"],
)
@pytest.mark.parametrize(
    "mesh_device, num_links, nh, base_seq_len, joint_seq_len, rp_factor, q_chunk_size, k_chunk_size, fp32",
    [
        ((1, 2), 2, 8, 2048, 0, 2, 256, 256, False),
        ((1, 2), 2, 8, 4000, 128, 2, 512, 256, False),
        ((1, 2), 2, 4, 1024, 0, 2, 128, 128, False),
    ],
    ids=["1x2_s2048", "1x2_s4000_j128", "1x2_s1024"],
    indirect=["mesh_device"],
)
def test_exp_ring_wh(mesh_device, num_links, nh, base_seq_len, joint_seq_len, rp_factor, q_chunk_size, k_chunk_size,
                     fp32, all_gather_topology, reset_seeds):
    submesh = create_ring_joint_sdpa_submesh(mesh_device, 1, rp_factor, 0, 1)
    padded = get_padded_vision_seq_len(base_seq_len, rp_factor)
    run_exp_ring_joint_sdpa(submesh, 1, nh, base_seq_len, padded, joint_seq_len, 128, q_chunk_size, k_chunk_size,
                            ttnn.bfloat16, 2, False, num_links, 1, 0, all_gather_topology, False, 0.994, max_mse=None)
