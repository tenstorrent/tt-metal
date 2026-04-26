# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
#  SPDX-License-Identifier: Apache-2.0

"""All-to-all dispatch (MoE) tests for multi-galaxy exabox mesh configurations (DUAL_BH / QUAD_BH).

Tests cover:
- all_to_all_dispatch on submeshes from 16x4 and 32x4 meshes
- Both cluster axes (0 and 1) via submesh creation
- MoE-specific parameters (experts, batches, select_k)
- FABRIC_1D + Linear topology
"""

import pytest

import ttnn
from tests.nightly.t3000.ccl.test_all_to_all_dispatch import run_all_to_all_dispatch_test


# ---------------------------------------------------------------------------
# Test: all_to_all_dispatch on 16x4 mesh (DUAL_BH)
# ---------------------------------------------------------------------------


@pytest.mark.requires_device(["DUAL_BH"])
@pytest.mark.parametrize(
    "device_params",
    [
        {"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "fabric_config": ttnn.FabricConfig.FABRIC_1D},
    ],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [pytest.param((16, 4), id="16x4_grid")], indirect=True)
@pytest.mark.parametrize("trace_mode", [False])
@pytest.mark.parametrize(
    "num_devices, mesh_shape, cluster_axis",
    [
        pytest.param(16, (16, 1), 0, id="axis0_16dev"),
        pytest.param(4, (1, 4), 1, id="axis1_4dev"),
    ],
)
@pytest.mark.parametrize("experts_per_device", [8])
@pytest.mark.parametrize("select_experts_k", [4])
@pytest.mark.parametrize("hidden_size", [7168])
@pytest.mark.parametrize(
    "batches_per_device, seq_len, num_iters, warmup_iters",
    [
        (16, 2, 2, 1),
    ],
    ids=["b16s2"],
)
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("input_memory_config", [ttnn.L1_MEMORY_CONFIG], ids=["l1"])
@pytest.mark.parametrize("output_memory_config", [ttnn.L1_MEMORY_CONFIG], ids=["l1"])
def test_all_to_all_dispatch_16x4(
    mesh_device,
    trace_mode,
    mesh_shape,
    num_devices,
    cluster_axis,
    batches_per_device,
    experts_per_device,
    select_experts_k,
    hidden_size,
    seq_len,
    num_iters,
    warmup_iters,
    num_links,
    dtype,
    input_memory_config,
    output_memory_config,
    device_params,
):
    topology = ttnn.Topology.Linear
    if cluster_axis is None:
        dispatch_devices = mesh_shape[0] * mesh_shape[1]
    else:
        dispatch_devices = mesh_shape[cluster_axis]
    batch = batches_per_device * dispatch_devices
    experts = experts_per_device * dispatch_devices

    submesh = mesh_device.create_submesh(ttnn.MeshShape(mesh_shape))
    run_all_to_all_dispatch_test(
        submesh,
        mesh_shape,
        batch,
        experts,
        select_experts_k,
        hidden_size,
        seq_len,
        num_iters,
        warmup_iters,
        trace_mode,
        num_links=num_links,
        scheme="random",
        topology=topology,
        input_memory_config=input_memory_config,
        output_memory_config=output_memory_config,
        dtype=dtype,
        cluster_axis=cluster_axis,
    )


# ---------------------------------------------------------------------------
# Test: all_to_all_dispatch on 32x4 mesh (QUAD_BH)
# ---------------------------------------------------------------------------


@pytest.mark.requires_device(["QUAD_BH"])
@pytest.mark.parametrize(
    "device_params",
    [
        {"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "fabric_config": ttnn.FabricConfig.FABRIC_1D},
    ],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [pytest.param((32, 4), id="32x4_grid")], indirect=True)
@pytest.mark.parametrize("trace_mode", [False])
@pytest.mark.parametrize(
    "num_devices, mesh_shape, cluster_axis",
    [
        pytest.param(32, (32, 1), 0, id="axis0_32dev"),
        pytest.param(4, (1, 4), 1, id="axis1_4dev"),
    ],
)
@pytest.mark.parametrize("experts_per_device", [8])
@pytest.mark.parametrize("select_experts_k", [4])
@pytest.mark.parametrize("hidden_size", [7168])
@pytest.mark.parametrize(
    "batches_per_device, seq_len, num_iters, warmup_iters",
    [
        (16, 2, 2, 1),
    ],
    ids=["b16s2"],
)
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("input_memory_config", [ttnn.L1_MEMORY_CONFIG], ids=["l1"])
@pytest.mark.parametrize("output_memory_config", [ttnn.L1_MEMORY_CONFIG], ids=["l1"])
def test_all_to_all_dispatch_32x4(
    mesh_device,
    trace_mode,
    mesh_shape,
    num_devices,
    cluster_axis,
    batches_per_device,
    experts_per_device,
    select_experts_k,
    hidden_size,
    seq_len,
    num_iters,
    warmup_iters,
    num_links,
    dtype,
    input_memory_config,
    output_memory_config,
    device_params,
):
    topology = ttnn.Topology.Linear
    if cluster_axis is None:
        dispatch_devices = mesh_shape[0] * mesh_shape[1]
    else:
        dispatch_devices = mesh_shape[cluster_axis]
    batch = batches_per_device * dispatch_devices
    experts = experts_per_device * dispatch_devices

    submesh = mesh_device.create_submesh(ttnn.MeshShape(mesh_shape))
    run_all_to_all_dispatch_test(
        submesh,
        mesh_shape,
        batch,
        experts,
        select_experts_k,
        hidden_size,
        seq_len,
        num_iters,
        warmup_iters,
        trace_mode,
        num_links=num_links,
        scheme="random",
        topology=topology,
        input_memory_config=input_memory_config,
        output_memory_config=output_memory_config,
        dtype=dtype,
        cluster_axis=cluster_axis,
    )
