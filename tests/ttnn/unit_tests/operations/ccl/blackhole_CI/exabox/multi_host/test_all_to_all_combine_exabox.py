# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
#  SPDX-License-Identifier: Apache-2.0

"""All-to-all combine (MoE) tests for multi-galaxy exabox mesh configurations (DUAL_BH / QUAD_BH).

Tests cover:
- Sync API (ttnn.all_to_all_combine) on full 16x4 and 32x4 meshes
- cluster_axis=1 only (matches all other multi-host all_to_all_combine tests in
  the repo: test_all_to_all_combine_6U.py::test_all_to_all_combine_8x8_dual_galaxy,
  ::test_all_to_all_combine_quad_host_mesh, and test_selective_combine_6U.py
  all parametrize cluster_axis=[1] only)
- L1 memory configs
- bfloat16 dtype
- MoE-specific parameters (experts, batches, select_k, local_reduce)

The underlying op + golden helper (run_all_to_all_combine_test) is vendored in
the sibling _a2a_moe_helpers.py so this folder is self-contained and does not
depend on tests/nightly/t3000/... It seeds torch and random internally so
multi-host MPI ranks generate consistent goldens.
"""

import pytest

import ttnn
from tests.ttnn.unit_tests.operations.ccl.blackhole_CI.exabox.multi_host._a2a_moe_helpers import (
    run_all_to_all_combine_test,
)

# ---------------------------------------------------------------------------
# Fabric / topology parametrize combos (reused by multiple tests)
# ---------------------------------------------------------------------------

FABRIC_TOPOLOGY_COMBOS = [
    pytest.param(
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "trace_region_size": 90112,
        },
        ttnn.Topology.Linear,
        id="fabric_1d-linear",
    ),
]

# ---------------------------------------------------------------------------
# Test: all_to_all_combine on 16x4 mesh (DUAL_BH)
# ---------------------------------------------------------------------------


@pytest.mark.requires_device(["DUAL_BH"])
@pytest.mark.parametrize(
    "device_params, topology",
    FABRIC_TOPOLOGY_COMBOS,
    indirect=["device_params"],
)
@pytest.mark.parametrize("mesh_device", [pytest.param((16, 4), id="16x4_grid")], indirect=True)
@pytest.mark.parametrize("cluster_axis", [pytest.param(1, id="axis1")])
@pytest.mark.parametrize("num_links", [1], ids=["1link"])
# Per-device params reduced from t3000 defaults (batches_per_device=8, experts_per_device=8,
# hidden_size=7000) because the full 64-device mesh would scale the aggregate batch and
# experts to 512 each, overflowing per-device L1.
@pytest.mark.parametrize("batches_per_device", [2])
@pytest.mark.parametrize("experts_per_device", [2])
@pytest.mark.parametrize("select_experts_k", [2])
@pytest.mark.parametrize("hidden_size", [1024])
@pytest.mark.parametrize("seq", [2])
@pytest.mark.parametrize("local_reduce", [False, True])
@pytest.mark.parametrize("scheme", ["random"])
@pytest.mark.parametrize("num_iters", [2])
@pytest.mark.parametrize("input_memory_config", [ttnn.L1_MEMORY_CONFIG], ids=["l1"])
@pytest.mark.parametrize("output_memory_config", [ttnn.L1_MEMORY_CONFIG], ids=["l1"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
def test_all_to_all_combine_16x4(
    mesh_device,
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
    mesh_shape = tuple(mesh_device.shape)
    devices = mesh_shape[0] * mesh_shape[1]
    batch = batches_per_device * devices
    experts = experts_per_device * devices

    mesh_device.disable_and_clear_program_cache()

    run_all_to_all_combine_test(
        mesh_device,
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
    "device_params, topology",
    FABRIC_TOPOLOGY_COMBOS,
    indirect=["device_params"],
)
@pytest.mark.parametrize("mesh_device", [pytest.param((32, 4), id="32x4_grid")], indirect=True)
@pytest.mark.parametrize("cluster_axis", [pytest.param(1, id="axis1")])
@pytest.mark.parametrize("num_links", [1], ids=["1link"])
# Per-device params reduced for the same reason as the 16x4 test (128 devices on QUAD_BH).
@pytest.mark.parametrize("batches_per_device", [2])
@pytest.mark.parametrize("experts_per_device", [2])
@pytest.mark.parametrize("select_experts_k", [2])
@pytest.mark.parametrize("hidden_size", [1024])
@pytest.mark.parametrize("seq", [2])
@pytest.mark.parametrize("local_reduce", [False, True])
@pytest.mark.parametrize("scheme", ["random"])
@pytest.mark.parametrize("num_iters", [2])
@pytest.mark.parametrize("input_memory_config", [ttnn.L1_MEMORY_CONFIG], ids=["l1"])
@pytest.mark.parametrize("output_memory_config", [ttnn.L1_MEMORY_CONFIG], ids=["l1"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
def test_all_to_all_combine_32x4(
    mesh_device,
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
    mesh_shape = tuple(mesh_device.shape)
    devices = mesh_shape[0] * mesh_shape[1]
    batch = batches_per_device * devices
    experts = experts_per_device * devices

    mesh_device.disable_and_clear_program_cache()

    run_all_to_all_combine_test(
        mesh_device,
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
