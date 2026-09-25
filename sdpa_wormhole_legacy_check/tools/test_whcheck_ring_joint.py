# SPDX-License-Identifier: Apache-2.0
"""whcheck: ring joint SDPA on one Wormhole n300 (1x2 mesh, linear), legacy config. Same file on BASE and HEAD."""
import pytest
import ttnn
from .test_ring_joint_attention import run_test_ring_joint_sdpa

LINE = {"worker_l1_size": 1344544, "trace_region_size": 1000000, "fabric_config": ttnn.FabricConfig.FABRIC_1D}


@pytest.mark.parametrize("device_params, all_gather_topology", [(LINE, ttnn.Topology.Linear)], indirect=["device_params"], ids=["line"])
@pytest.mark.parametrize("mesh_device", [(1, 2)], ids=["1x2"], indirect=True)
@pytest.mark.parametrize("fp32", [False, True], ids=["bf16acc", "fp32acc"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b], ids=["bf16", "bf8b"])
@pytest.mark.parametrize(
    "shape, q_chunk, k_chunk",
    [
        ((1, 10, 4096, 333, 64), 128, 512),   # sd35-like, joint
        ((1, 8, 8192, 0, 128), 256, 256),     # wan-like, no joint
        ((1, 6, 4000, 118, 128), 128, 256),   # mochi-like, padded + joint
    ],
    ids=["sd35", "wan", "mochi"],
)
def test_ring_joint_wh_1x2(mesh_device, shape, q_chunk, k_chunk, dtype, fp32, all_gather_topology, reset_seeds):
    run_test_ring_joint_sdpa(mesh_device, shape, (1, 2, 0, 1), q_chunk, k_chunk, 1, False, 1, all_gather_topology,
                             False, dtype, pcc_threshold=0.99, fp32_dest_acc_en=fp32)
