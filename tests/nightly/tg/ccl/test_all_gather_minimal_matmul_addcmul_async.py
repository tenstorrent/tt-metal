# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
from models.common.utility_functions import is_wormhole_b0

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
    if is_wormhole_b0():
        pytest.skip("core grid (12, 9) exceeds wormhole_b0 compute grid (8x8), blackhole-only config")

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


def _assert_addcmul_quality(mesh_device, check_result):
    for c in range(1):
        for i in range(mesh_device.get_num_devices()):
            assert check_result[0][c][i]["pcc"] > 0.999_500
            assert check_result[0][c][i]["relative_rmse"] < 0.02


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True, ids=["4x8"])
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
def test_all_gather_minimal_matmul_addcmul_cache_hit(mesh_device, topology):
    if is_wormhole_b0():
        pytest.skip("core grid (12, 9) exceeds wormhole_b0 compute grid (8x8), blackhole-only config")

    mesh_device.enable_program_cache()
    common = dict(
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
        broadcast_gate=False,
        share_addcmul_inputs=True,
        use_non_fused=False,
        sp_axis=1,
        tp_axis=0,
        cluster_axis=0,
    )
    # Miss uses aliased full-size ternary inputs so resolve_bindings returns no address bindings.
    first = run_test_linear(mesh_device, addcmul_scalar=1.0, **common)
    entries = mesh_device.num_program_cache_entries()
    # Hit: new allocations, new semaphores, and a different excluded scalar must still match.
    second = run_test_linear(mesh_device, addcmul_scalar=0.5, **common)
    assert mesh_device.num_program_cache_entries() == entries
    _assert_addcmul_quality(mesh_device, first)
    _assert_addcmul_quality(mesh_device, second)
