# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
#  SPDX-License-Identifier: Apache-2.0

"""All-to-all combine (MoE) tests for multi-galaxy exabox mesh configurations (DUAL_BH / QUAD_BH).

Tests cover:
- all_to_all_combine on submeshes from 16x4 and 32x4 meshes
- Both cluster axes (0 and 1) via submesh creation
- MoE-specific parameters (experts, batches, select_k, local_reduce)
- FABRIC_1D + Linear topology
"""

import pytest

import ttnn
from tests.nightly.t3000.ccl.test_all_to_all_combine import run_all_to_all_combine_test


# ---------------------------------------------------------------------------
# Test: all_to_all_combine on 16x4 mesh (DUAL_BH)
# ---------------------------------------------------------------------------


@pytest.mark.requires_device(["DUAL_BH"])
@pytest.mark.parametrize(
    "device_params, num_links, topology",
    [
        pytest.param(
            {"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "fabric_config": ttnn.FabricConfig.FABRIC_1D},
            1,
            ttnn.Topology.Linear,
            id="fabric_1d-linear",
        ),
    ],
    indirect=["device_params"],
)
@pytest.mark.parametrize("mesh_device", [pytest.param((16, 4), id="16x4_grid")], indirect=True)
@pytest.mark.parametrize(
    "num_devices, mesh_shape, cluster_axis",
    [
        pytest.param(16, (16, 1), 0, id="axis0_16dev"),
        pytest.param(4, (1, 4), 1, id="axis1_4dev"),
    ],
)
@pytest.mark.parametrize("batches_per_device", [8])
@pytest.mark.parametrize("experts_per_device", [8])
@pytest.mark.parametrize("select_experts_k", [8])
@pytest.mark.parametrize("hidden_size", [7000])
@pytest.mark.parametrize("seq", [2])
@pytest.mark.parametrize("local_reduce", [False, True])
@pytest.mark.parametrize("scheme", ["random"])
@pytest.mark.parametrize("num_iters", [2])
@pytest.mark.parametrize("input_memory_config", [ttnn.L1_MEMORY_CONFIG], ids=["l1"])
@pytest.mark.parametrize("output_memory_config", [ttnn.L1_MEMORY_CONFIG], ids=["l1"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
def test_all_to_all_combine_16x4(
    mesh_device,
    mesh_shape,
    num_devices,
    cluster_axis,
    batches_per_device,
    seq,
    local_reduce,
    experts_per_device,
    select_experts_k,
    hidden_size,
    num_iters,
    scheme,
    input_memory_config,
    output_memory_config,
    num_links,
    topology,
    dtype,
):
    devices = mesh_shape[0] * mesh_shape[1]
    batch = batches_per_device * devices
    experts = experts_per_device * devices

    mesh_device.disable_and_clear_program_cache()
    submesh = mesh_device.create_submesh(ttnn.MeshShape(mesh_shape))

    run_all_to_all_combine_test(
        submesh,
        mesh_shape,
        cluster_axis,
        batch,
        seq,
        local_reduce,
        experts,
        select_experts_k,
        hidden_size,
        num_iters,
        num_links=num_links,
        scheme=scheme,
        topology=topology,
        input_memory_config=input_memory_config,
        output_memory_config=output_memory_config,
        test_skew=False,
    )


# ---------------------------------------------------------------------------
# Test: all_to_all_combine on 32x4 mesh (QUAD_BH)
# ---------------------------------------------------------------------------


@pytest.mark.requires_device(["QUAD_BH"])
@pytest.mark.parametrize(
    "device_params, num_links, topology",
    [
        pytest.param(
            {"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "fabric_config": ttnn.FabricConfig.FABRIC_1D},
            1,
            ttnn.Topology.Linear,
            id="fabric_1d-linear",
        ),
    ],
    indirect=["device_params"],
)
@pytest.mark.parametrize("mesh_device", [pytest.param((32, 4), id="32x4_grid")], indirect=True)
@pytest.mark.parametrize(
    "num_devices, mesh_shape, cluster_axis",
    [
        pytest.param(32, (32, 1), 0, id="axis0_32dev"),
        pytest.param(4, (1, 4), 1, id="axis1_4dev"),
    ],
)
@pytest.mark.parametrize("batches_per_device", [8])
@pytest.mark.parametrize("experts_per_device", [8])
@pytest.mark.parametrize("select_experts_k", [8])
@pytest.mark.parametrize("hidden_size", [7000])
@pytest.mark.parametrize("seq", [2])
@pytest.mark.parametrize("local_reduce", [False, True])
@pytest.mark.parametrize("scheme", ["random"])
@pytest.mark.parametrize("num_iters", [2])
@pytest.mark.parametrize("input_memory_config", [ttnn.L1_MEMORY_CONFIG], ids=["l1"])
@pytest.mark.parametrize("output_memory_config", [ttnn.L1_MEMORY_CONFIG], ids=["l1"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
def test_all_to_all_combine_32x4(
    mesh_device,
    mesh_shape,
    num_devices,
    cluster_axis,
    batches_per_device,
    seq,
    local_reduce,
    experts_per_device,
    select_experts_k,
    hidden_size,
    num_iters,
    scheme,
    input_memory_config,
    output_memory_config,
    num_links,
    topology,
    dtype,
):
    devices = mesh_shape[0] * mesh_shape[1]
    batch = batches_per_device * devices
    experts = experts_per_device * devices

    mesh_device.disable_and_clear_program_cache()
    submesh = mesh_device.create_submesh(ttnn.MeshShape(mesh_shape))

    run_all_to_all_combine_test(
        submesh,
        mesh_shape,
        cluster_axis,
        batch,
        seq,
        local_reduce,
        experts,
        select_experts_k,
        hidden_size,
        num_iters,
        num_links=num_links,
        scheme=scheme,
        topology=topology,
        input_memory_config=input_memory_config,
        output_memory_config=output_memory_config,
        test_skew=False,
    )
