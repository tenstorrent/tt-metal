# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn

from models.tt_dit.tests.models.wan2_2.test_all_gather_minimal_matmul_async import (
    create_fabric_router_config,
    run_test_linear,
)


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True, ids=["4x8"])
@pytest.mark.parametrize("broadcast_gate", [True, False], ids=["broadcast_gate", "full_gate"])
@pytest.mark.parametrize(
    "device_params, topology",
    [
        (
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "fabric_router_config": create_fabric_router_config(4096),
                "trace_region_size": 90112,
            },
            ttnn.Topology.Ring,
        ),
    ],
    indirect=["device_params"],
    ids=["fabric_ring"],
)
def test_all_gather_minimal_matmul_addcmul(
    mesh_device,
    topology,
    broadcast_gate,
):
    check_result = run_test_linear(
        mesh_device,
        M=3072,
        K=5120,
        N=1280,
        M_block_size=8,
        K_block_size=8,
        N_block_size=8,
        subblock_h=2,
        subblock_w=1,
        topology=topology,
        core_grid=ttnn.CoreCoord(12, 9),
        num_workers_per_link=6,
        num_links=2,
        use_bias=True,
        fuse_addcmul=True,
        addcmul_scalar=1.0,
        broadcast_gate=broadcast_gate,
        use_non_fused=False,
        sp_axis=1,
        tp_axis=0,
        cluster_axis=0,
    )
    for c in range(1):
        for i in range(mesh_device.get_num_devices()):
            assert check_result[0][c][i]["pcc"] > 0.999_500
            assert check_result[0][c][i]["relative_rmse"] < 0.02
