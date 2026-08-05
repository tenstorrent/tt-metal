# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
#  SPDX-License-Identifier: Apache-2.0

"""All-to-all dispatch (MoE) tests for multi-galaxy exabox mesh configurations (DUAL_BH / QUAD_BH).

Tests cover:
- Sync API (ttnn.all_to_all_dispatch) on full 16x4 and 32x4 meshes
- cluster_axis=1 only (matches the multi-host all_to_all_dispatch tests in the
  repo: test_all_to_all_dispatch_6U.py::test_all_to_all_dispatch_8x8_dual_galaxy
  and the dual_galaxy CI runner both parametrize cluster_axis=[1] only;
  cluster_axis=[0, 1] is exercised only on single-galaxy 8x4 in
  test_all_to_all_dispatch_6U.py::test_all_to_all_dispatch_trace)
- L1 memory configs
- bfloat16 dtype
- MoE-specific parameters (experts, batches, select_k)

The underlying op + golden helper (run_all_to_all_dispatch_test) is vendored in
the sibling _a2a_moe_helpers.py so this folder is self-contained and does not
depend on tests/nightly/t3000/...
"""

import pytest

import ttnn
from tests.ttnn.unit_tests.operations.ccl.blackhole_CI.exabox.multi_host._a2a_moe_helpers import (
    run_all_to_all_dispatch_test,
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
# Test: all_to_all_dispatch on 16x4 mesh (DUAL_BH)
# ---------------------------------------------------------------------------


@pytest.mark.requires_device(["DUAL_BH"])
@pytest.mark.parametrize(
    "device_params, topology",
    FABRIC_TOPOLOGY_COMBOS,
    indirect=["device_params"],
)
@pytest.mark.parametrize("mesh_device", [pytest.param((16, 4), id="16x4_grid")], indirect=True)
@pytest.mark.parametrize("cluster_axis", [pytest.param(1, id="axis1")])
@pytest.mark.parametrize("trace_mode", [False])
@pytest.mark.parametrize("num_links", [1], ids=["1link"])
# Per-device params reduced from t3000 defaults (batches_per_device=16, experts_per_device=8,
# hidden_size=7168) because the full 64-device mesh would scale aggregate batch/experts to
# 1024/512 and overflow per-device L1.
@pytest.mark.parametrize("batches_per_device", [2])
@pytest.mark.parametrize("experts_per_device", [2])
@pytest.mark.parametrize("select_experts_k", [2])
@pytest.mark.parametrize("hidden_size", [1024])
@pytest.mark.parametrize(
    "seq_len, num_iters, warmup_iters",
    [(2, 2, 1)],
    ids=["s2"],
)
@pytest.mark.parametrize("input_memory_config", [ttnn.L1_MEMORY_CONFIG], ids=["l1"])
@pytest.mark.parametrize("output_memory_config", [ttnn.L1_MEMORY_CONFIG], ids=["l1"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
def test_all_to_all_dispatch_16x4(
    mesh_device,
    cluster_axis,
    trace_mode,
    batches_per_device,
    experts_per_device,
    select_experts_k,
    hidden_size,
    seq_len,
    num_iters,
    warmup_iters,
    num_links,
    topology,
    dtype,
    input_memory_config,
    output_memory_config,
    device_params,
):
    mesh_shape = tuple(mesh_device.shape)
    # Helper computes per-device counts as batch/devices and experts/devices, where
    # devices = mesh_shape[0] * mesh_shape[1]. Scale by total mesh size accordingly,
    # not just by mesh_shape[cluster_axis] (which was the submesh-era convention).
    devices = mesh_shape[0] * mesh_shape[1]
    batch = batches_per_device * devices
    experts = experts_per_device * devices

    run_all_to_all_dispatch_test(
        mesh_device,
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
    "device_params, topology",
    FABRIC_TOPOLOGY_COMBOS,
    indirect=["device_params"],
)
@pytest.mark.parametrize("mesh_device", [pytest.param((32, 4), id="32x4_grid")], indirect=True)
@pytest.mark.parametrize("cluster_axis", [pytest.param(1, id="axis1")])
@pytest.mark.parametrize("trace_mode", [False])
@pytest.mark.parametrize("num_links", [1], ids=["1link"])
# Per-device params reduced for the same reason as 16x4 (128 devices on QUAD_BH).
@pytest.mark.parametrize("batches_per_device", [2])
@pytest.mark.parametrize("experts_per_device", [2])
@pytest.mark.parametrize("select_experts_k", [2])
@pytest.mark.parametrize("hidden_size", [1024])
@pytest.mark.parametrize(
    "seq_len, num_iters, warmup_iters",
    [(2, 2, 1)],
    ids=["s2"],
)
@pytest.mark.parametrize("input_memory_config", [ttnn.L1_MEMORY_CONFIG], ids=["l1"])
@pytest.mark.parametrize("output_memory_config", [ttnn.L1_MEMORY_CONFIG], ids=["l1"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
def test_all_to_all_dispatch_32x4(
    mesh_device,
    cluster_axis,
    trace_mode,
    batches_per_device,
    experts_per_device,
    select_experts_k,
    hidden_size,
    seq_len,
    num_iters,
    warmup_iters,
    num_links,
    topology,
    dtype,
    input_memory_config,
    output_memory_config,
    device_params,
):
    mesh_shape = tuple(mesh_device.shape)
    # Helper computes per-device counts as batch/devices and experts/devices, where
    # devices = mesh_shape[0] * mesh_shape[1]. Scale by total mesh size accordingly.
    devices = mesh_shape[0] * mesh_shape[1]
    batch = batches_per_device * devices
    experts = experts_per_device * devices

    run_all_to_all_dispatch_test(
        mesh_device,
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
