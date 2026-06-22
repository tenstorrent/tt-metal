# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Parity tests for the C++ ttnn-tensor port of ``moe_compute_utils``.

Each test runs the torch reference (``ttnn._experimental.moe_compute_utils``)
and the C++ ttnn-tensor port (``ttnn.experimental.<fn>``) on the same inputs
and checks for byte-exact output parity. Inputs are sharded across the TG
mesh on the experts dim (dim 1) so the tests exercise the actual multi-device
sharded path the ttnn helpers are designed for.
"""

import os

import pytest
import torch
import ttnn

# Force torch CPU ops to use all cores. bf16 matmul on CPU is single-thread on
# some torch builds even with OMP_NUM_THREADS set; this is the runtime knob that
# keeps host-side golden compute fast. See test_moe_compute_6U.py for context.
torch.set_num_threads(os.cpu_count() or 1)

from ttnn.experimental.moe_compute_utils import (
    add_shared_expert_weights as torch_add_shared_expert_weights,
    get_weight_core_shard_maps,
    get_weight_mem_configs as torch_get_weight_mem_configs,
    prepare_w0_w1_tensor_for_moe_compute as torch_prepare_w0_w1,
    prepare_w0_w1_tensor_with_bias as torch_prepare_w0_w1_with_bias,
    prepare_w2_tensor_for_moe_compute as torch_prepare_w2,
    prepare_w2_tensor_with_bias as torch_prepare_w2_with_bias,
)

MESH_GRAPH_DESC_1x16 = (
    "tests/tt_metal/tt_fabric/custom_mesh_descriptors/single_galaxy_1x16_torus_graph_descriptor.textproto"
)
MESH_GRAPH_DESC_1x8 = (
    "tests/tt_metal/tt_fabric/custom_mesh_descriptors/single_galaxy_1x8_torus_graph_descriptor.textproto"
)


def is_mesh_graph_descriptor_set(expected_path):
    return os.environ.get("TT_MESH_GRAPH_DESC_PATH") == expected_path


_MESH_PARAM_8x4 = [
    pytest.param(
        (8, 4),
        (8, 4),
        marks=pytest.mark.skipif(
            is_mesh_graph_descriptor_set(MESH_GRAPH_DESC_1x16),
            reason="8x4 mesh breaks with 16 device MGD",
        ),
        id="8x4",
    ),
]

_MESH_PARAM_1x16 = [
    pytest.param(
        (1, 16),
        (1, 16),
        marks=pytest.mark.skipif(
            not is_mesh_graph_descriptor_set(MESH_GRAPH_DESC_1x16),
            reason=f"1x16 mesh requires TT_MESH_GRAPH_DESC_PATH={MESH_GRAPH_DESC_1x16}",
        ),
        id="1x16",
    ),
]

MESH_PARAMS = _MESH_PARAM_8x4 + _MESH_PARAM_1x16

# (hidden_size, intermediate_size) — DeepSeek matches the production layout; the
# small config catches edge cases in the shard formulas (Nt = num_cores, etc.).
DIM_PARAMS = [
    pytest.param(7168, 2048, id="deepseek"),
    pytest.param(512, 384, id="small"),
]

DEVICE_PARAMS = [
    {
        "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
        "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
        "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
        "trace_region_size": 500000,
    }
]


def _shard_to_mesh(t: torch.Tensor, mesh_device, dim: int) -> ttnn.Tensor:
    return ttnn.from_torch(
        t,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=dim),
    )


def _concat_from_mesh(tt: ttnn.Tensor, mesh_device, dim: int) -> torch.Tensor:
    return ttnn.to_torch(tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=dim))


def _torch_prepare_w0_w1_per_device_stacked(
    torch_w0, torch_w1, num_devices, num_layers, experts_per_device, hidden_size, N, shard_map
):
    """Apply torch prepare_w0_w1 to each device's E-chunk, stack along the (post-permute) E dim 2."""
    chunks = []
    for d in range(num_devices):
        s = slice(d * experts_per_device, (d + 1) * experts_per_device)
        chunks.append(
            torch_prepare_w0_w1(
                torch_w0[:, s], torch_w1[:, s], num_layers, experts_per_device, hidden_size, N, shard_map
            )
        )
    return torch.cat(chunks, dim=2)


def _torch_prepare_w2_per_device_stacked(
    torch_w2, num_devices, num_layers, experts_per_device, N, hidden_size, w2_shard_map, w0_w1_shard_map
):
    chunks = []
    for d in range(num_devices):
        s = slice(d * experts_per_device, (d + 1) * experts_per_device)
        chunks.append(
            torch_prepare_w2(
                torch_w2[:, s], num_layers, experts_per_device, N, hidden_size, w2_shard_map, w0_w1_shard_map
            )
        )
    return torch.cat(chunks, dim=2)


def _torch_prepare_w0_w1_with_bias_per_device_stacked(
    torch_w0, torch_w1, torch_b0, torch_b1, num_devices, num_layers, experts_per_device, hidden_size, N, shard_map
):
    chunks = []
    for d in range(num_devices):
        s = slice(d * experts_per_device, (d + 1) * experts_per_device)
        chunks.append(
            torch_prepare_w0_w1_with_bias(
                torch_w0[:, s],
                torch_w1[:, s],
                torch_b0[:, s],
                torch_b1[:, s],
                num_layers,
                experts_per_device,
                hidden_size,
                N,
                shard_map,
            )
        )
    return torch.cat(chunks, dim=2)


def _torch_prepare_w2_with_bias_per_device_stacked(
    torch_w2, torch_b2, num_devices, num_layers, experts_per_device, N, hidden_size, w2_shard_map, w0_w1_shard_map
):
    chunks = []
    for d in range(num_devices):
        s = slice(d * experts_per_device, (d + 1) * experts_per_device)
        chunks.append(
            torch_prepare_w2_with_bias(
                torch_w2[:, s],
                torch_b2[:, s],
                num_layers,
                experts_per_device,
                N,
                hidden_size,
                w2_shard_map,
                w0_w1_shard_map,
            )
        )
    return torch.cat(chunks, dim=2)


# ---------------------------------------------------------------------------
# get_weight_core_shard_maps
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_shape, mesh_device", MESH_PARAMS, indirect=["mesh_device"])
@pytest.mark.parametrize("hidden_size, N", DIM_PARAMS)
def test_get_weight_core_shard_maps_parity(mesh_device, mesh_shape, hidden_size, N):
    py_w0_w1, py_w2, py_crs = get_weight_core_shard_maps(mesh_device, hidden_size, N)
    tt_result = ttnn.experimental.get_weight_core_shard_maps(mesh_device, hidden_size, N)
    assert list(tt_result.w0_w1_shard_map) == list(py_w0_w1)
    assert [tuple(p) for p in tt_result.w2_shard_map] == [tuple(p) for p in py_w2]
    assert tt_result.dram_core_range_set == py_crs


# ---------------------------------------------------------------------------
# get_weight_mem_configs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_shape, mesh_device", MESH_PARAMS, indirect=["mesh_device"])
@pytest.mark.parametrize("hidden_size, N", DIM_PARAMS)
@pytest.mark.parametrize("num_layers, experts_per_device", [(1, 2)])
@pytest.mark.parametrize("has_bias", [False, True])
def test_get_weight_mem_configs_parity(
    mesh_device, mesh_shape, hidden_size, N, num_layers, experts_per_device, has_bias
):
    py_w0_w1_map, py_w2_map, py_crs = get_weight_core_shard_maps(mesh_device, hidden_size, N)
    py_w0_w1_mc, py_w2_mc, _, _ = torch_get_weight_mem_configs(
        num_layers, experts_per_device, hidden_size, N, py_w0_w1_map, py_w2_map, py_crs, has_bias=has_bias
    )
    tt_result = ttnn.experimental.get_weight_mem_configs(
        mesh_device,
        num_layers=num_layers,
        experts_per_device=experts_per_device,
        hidden_size=hidden_size,
        intermediate_size=N,
        has_bias=has_bias,
    )
    assert tt_result.w0_w1 == py_w0_w1_mc
    assert tt_result.w2 == py_w2_mc


# ---------------------------------------------------------------------------
# prepare_w0_w1_tensor_for_moe_compute
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_shape, mesh_device", MESH_PARAMS, indirect=["mesh_device"])
@pytest.mark.parametrize("hidden_size, N", DIM_PARAMS)
@pytest.mark.parametrize("num_layers, experts_per_device", [(1, 2)])
@torch.no_grad()
def test_prepare_w0_w1_tensor_parity(mesh_device, mesh_shape, hidden_size, N, num_layers, experts_per_device):
    torch.manual_seed(2003)
    num_devices = mesh_shape[0] * mesh_shape[1]
    E_total = experts_per_device * num_devices

    torch_w0 = torch.randn(num_layers, E_total, hidden_size, N, dtype=torch.bfloat16)
    torch_w1 = torch.randn(num_layers, E_total, hidden_size, N, dtype=torch.bfloat16)

    w0_w1_shard_map, _, _ = get_weight_core_shard_maps(mesh_device, hidden_size, N)

    torch_ref = _torch_prepare_w0_w1_per_device_stacked(
        torch_w0, torch_w1, num_devices, num_layers, experts_per_device, hidden_size, N, w0_w1_shard_map
    )

    tt_w0 = _shard_to_mesh(torch_w0, mesh_device, dim=1)
    tt_w1 = _shard_to_mesh(torch_w1, mesh_device, dim=1)
    tt_out = ttnn.experimental.prepare_w0_w1_tensor_for_moe_compute(
        tt_w0, tt_w1, L=num_layers, E=experts_per_device, K=hidden_size, N=N
    )
    ttnn.synchronize_device(mesh_device)
    tt_pulled = _concat_from_mesh(tt_out, mesh_device, dim=2)

    assert tt_pulled.shape == torch_ref.shape, f"shape mismatch: {tt_pulled.shape} vs {torch_ref.shape}"
    assert torch.equal(tt_pulled, torch_ref)


# ---------------------------------------------------------------------------
# prepare_w2_tensor_for_moe_compute
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_shape, mesh_device", MESH_PARAMS, indirect=["mesh_device"])
@pytest.mark.parametrize("hidden_size, N", DIM_PARAMS)
@pytest.mark.parametrize("num_layers, experts_per_device", [(1, 2)])
@torch.no_grad()
def test_prepare_w2_tensor_parity(mesh_device, mesh_shape, hidden_size, N, num_layers, experts_per_device):
    torch.manual_seed(2003)
    num_devices = mesh_shape[0] * mesh_shape[1]
    E_total = experts_per_device * num_devices

    torch_w2 = torch.randn(num_layers, E_total, N, hidden_size, dtype=torch.bfloat16)

    w0_w1_shard_map, w2_shard_map, _ = get_weight_core_shard_maps(mesh_device, hidden_size, N)

    torch_ref = _torch_prepare_w2_per_device_stacked(
        torch_w2, num_devices, num_layers, experts_per_device, N, hidden_size, w2_shard_map, w0_w1_shard_map
    )

    tt_w2 = _shard_to_mesh(torch_w2, mesh_device, dim=1)
    tt_out = ttnn.experimental.prepare_w2_tensor_for_moe_compute(
        tt_w2,
        L=num_layers,
        E=experts_per_device,
        N=N,
        K=hidden_size,
    )
    ttnn.synchronize_device(mesh_device)

    tt_pulled = _concat_from_mesh(tt_out, mesh_device, dim=2)

    assert tt_pulled.shape == torch_ref.shape, f"shape mismatch: {tt_pulled.shape} vs {torch_ref.shape}"
    assert torch.equal(tt_pulled, torch_ref)


# ---------------------------------------------------------------------------
# prepare_w0_w1_tensor_with_bias
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_shape, mesh_device", MESH_PARAMS, indirect=["mesh_device"])
@pytest.mark.parametrize("hidden_size, N", DIM_PARAMS)
@pytest.mark.parametrize("num_layers, experts_per_device", [(1, 2)])
@torch.no_grad()
def test_prepare_w0_w1_tensor_with_bias_parity(mesh_device, mesh_shape, hidden_size, N, num_layers, experts_per_device):
    torch.manual_seed(2003)
    num_devices = mesh_shape[0] * mesh_shape[1]
    E_total = experts_per_device * num_devices

    torch_w0 = torch.randn(num_layers, E_total, hidden_size, N, dtype=torch.bfloat16)
    torch_w1 = torch.randn(num_layers, E_total, hidden_size, N, dtype=torch.bfloat16)
    torch_b0 = torch.randn(num_layers, E_total, N, dtype=torch.bfloat16)
    torch_b1 = torch.randn(num_layers, E_total, N, dtype=torch.bfloat16)

    w0_w1_shard_map, _, _ = get_weight_core_shard_maps(mesh_device, hidden_size, N)

    torch_ref = _torch_prepare_w0_w1_with_bias_per_device_stacked(
        torch_w0,
        torch_w1,
        torch_b0,
        torch_b1,
        num_devices,
        num_layers,
        experts_per_device,
        hidden_size,
        N,
        w0_w1_shard_map,
    )

    tt_w0 = _shard_to_mesh(torch_w0, mesh_device, dim=1)
    tt_w1 = _shard_to_mesh(torch_w1, mesh_device, dim=1)
    tt_b0 = _shard_to_mesh(torch_b0, mesh_device, dim=1)
    tt_b1 = _shard_to_mesh(torch_b1, mesh_device, dim=1)
    tt_out = ttnn.experimental.prepare_w0_w1_tensor_with_bias(
        tt_w0,
        tt_w1,
        tt_b0,
        tt_b1,
        L=num_layers,
        E=experts_per_device,
        K=hidden_size,
        N=N,
    )
    ttnn.synchronize_device(mesh_device)

    tt_pulled = _concat_from_mesh(tt_out, mesh_device, dim=2)

    assert tt_pulled.shape == torch_ref.shape, f"shape mismatch: {tt_pulled.shape} vs {torch_ref.shape}"
    assert torch.equal(tt_pulled, torch_ref)


# ---------------------------------------------------------------------------
# prepare_w2_tensor_with_bias
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_shape, mesh_device", MESH_PARAMS, indirect=["mesh_device"])
@pytest.mark.parametrize("hidden_size, N", DIM_PARAMS)
@pytest.mark.parametrize("num_layers, experts_per_device", [(1, 2)])
@torch.no_grad()
def test_prepare_w2_tensor_with_bias_parity(mesh_device, mesh_shape, hidden_size, N, num_layers, experts_per_device):
    torch.manual_seed(2003)
    num_devices = mesh_shape[0] * mesh_shape[1]
    E_total = experts_per_device * num_devices

    torch_w2 = torch.randn(num_layers, E_total, N, hidden_size, dtype=torch.bfloat16)
    torch_b2 = torch.randn(num_layers, E_total, hidden_size, dtype=torch.bfloat16)

    w0_w1_shard_map, w2_shard_map, _ = get_weight_core_shard_maps(mesh_device, hidden_size, N)

    torch_ref = _torch_prepare_w2_with_bias_per_device_stacked(
        torch_w2,
        torch_b2,
        num_devices,
        num_layers,
        experts_per_device,
        N,
        hidden_size,
        w2_shard_map,
        w0_w1_shard_map,
    )

    tt_w2 = _shard_to_mesh(torch_w2, mesh_device, dim=1)
    tt_b2 = _shard_to_mesh(torch_b2, mesh_device, dim=1)
    tt_out = ttnn.experimental.prepare_w2_tensor_with_bias(
        tt_w2,
        tt_b2,
        L=num_layers,
        E=experts_per_device,
        N=N,
        K=hidden_size,
    )
    ttnn.synchronize_device(mesh_device)
    tt_pulled = _concat_from_mesh(tt_out, mesh_device, dim=2)

    assert tt_pulled.shape == torch_ref.shape, f"shape mismatch: {tt_pulled.shape} vs {torch_ref.shape}"
    assert torch.equal(tt_pulled, torch_ref)


# ---------------------------------------------------------------------------
# add_shared_expert_weights
# ---------------------------------------------------------------------------


def _build_shared_test_setup(num_layers, num_devices, hidden_size, N, E_total):
    """Build a shared-expert test setup with 2 distinct shared experts split across even/odd devices.

    Returns:
        shared_w0_dict, shared_w1_dict, shared_w2_dict, shared_expert_ids_to_device,
        shared_w0_arranged, shared_w1_arranged, shared_w2_arranged
        where the `*_arranged` tensors are per-device-arranged (L, num_devices * num_shared_per_device, ...)
        ready to be sharded on dim 1.
    """
    # 2 distinct shared experts, ids E_total and E_total+1.
    # Shared E_total goes on even-indexed devices; E_total+1 on odd.
    # Each device gets exactly 1 shared expert → num_shared_per_device = 1.
    shared_ids = [E_total, E_total + 1]
    shared_expert_ids_to_device = {
        E_total: [d for d in range(num_devices) if d % 2 == 0],
        E_total + 1: [d for d in range(num_devices) if d % 2 == 1],
    }

    shared_w0_dict = {sid: torch.randn(num_layers, 1, hidden_size, N, dtype=torch.bfloat16) for sid in shared_ids}
    shared_w1_dict = {sid: torch.randn(num_layers, 1, hidden_size, N, dtype=torch.bfloat16) for sid in shared_ids}
    shared_w2_dict = {sid: torch.randn(num_layers, 1, N, hidden_size, dtype=torch.bfloat16) for sid in shared_ids}

    device_to_shared_experts = [[] for _ in range(num_devices)]
    for sid in sorted(shared_expert_ids_to_device.keys()):
        for d in shared_expert_ids_to_device[sid]:
            device_to_shared_experts[d].append(sid)

    # Sanity: even distribution.
    assert all(len(x) == len(device_to_shared_experts[0]) for x in device_to_shared_experts)

    def arrange(shared_dict):
        # For each device in order, concat its assigned shared experts (in sorted-id order) on dim 1.
        per_device_chunks = []
        for d in range(num_devices):
            per_device_chunks.append(torch.cat([shared_dict[sid] for sid in device_to_shared_experts[d]], dim=1))
        return torch.cat(per_device_chunks, dim=1)

    return (
        shared_w0_dict,
        shared_w1_dict,
        shared_w2_dict,
        shared_expert_ids_to_device,
        arrange(shared_w0_dict),
        arrange(shared_w1_dict),
        arrange(shared_w2_dict),
    )


# Note, only test this on linear meshes as the C++ version now includes TP sharding across the replicate axis and has
# thus diverged from the OG python.
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_shape, mesh_device", _MESH_PARAM_1x16, indirect=["mesh_device"])
@pytest.mark.parametrize("hidden_size, N", DIM_PARAMS)
@pytest.mark.parametrize("num_layers, experts_per_device", [(1, 2)])
@torch.no_grad()
def test_add_shared_expert_weights_parity(mesh_device, mesh_shape, hidden_size, N, num_layers, experts_per_device):
    torch.manual_seed(2003)
    num_devices = mesh_shape[0] * mesh_shape[1]
    E_total = experts_per_device * num_devices
    cluster_axis = 1

    routed_w0 = torch.randn(num_layers, E_total, hidden_size, N, dtype=torch.bfloat16)
    routed_w1 = torch.randn(num_layers, E_total, hidden_size, N, dtype=torch.bfloat16)
    routed_w2 = torch.randn(num_layers, E_total, N, hidden_size, dtype=torch.bfloat16)

    (
        shared_w0_dict,
        shared_w1_dict,
        shared_w2_dict,
        shared_expert_ids_to_device,
        shared_w0_arranged,
        shared_w1_arranged,
        shared_w2_arranged,
    ) = _build_shared_test_setup(num_layers, num_devices, hidden_size, N, E_total)

    torch_ref_w0, torch_ref_w1, torch_ref_w2 = torch_add_shared_expert_weights(
        routed_w0,
        routed_w1,
        routed_w2,
        shared_w0_dict,
        shared_w1_dict,
        shared_w2_dict,
        shared_expert_ids_to_device,
        num_devices,
    )

    tt_routed_w0 = _shard_to_mesh(routed_w0, mesh_device, dim=1)
    tt_routed_w1 = _shard_to_mesh(routed_w1, mesh_device, dim=1)
    tt_routed_w2 = _shard_to_mesh(routed_w2, mesh_device, dim=1)
    tt_shared_w0 = _shard_to_mesh(shared_w0_arranged, mesh_device, dim=1)
    tt_shared_w1 = _shard_to_mesh(shared_w1_arranged, mesh_device, dim=1)
    tt_shared_w2 = _shard_to_mesh(shared_w2_arranged, mesh_device, dim=1)

    tt_out_w0, tt_out_w1, tt_out_w2 = ttnn.experimental.add_shared_expert_weights(
        tt_routed_w0,
        tt_routed_w1,
        tt_routed_w2,
        tt_shared_w0,
        tt_shared_w1,
        tt_shared_w2,
        cluster_axis=cluster_axis,
    )
    ttnn.synchronize_device(mesh_device)

    tt_pulled_w0 = _concat_from_mesh(tt_out_w0, mesh_device, dim=1)
    tt_pulled_w1 = _concat_from_mesh(tt_out_w1, mesh_device, dim=1)
    tt_pulled_w2 = _concat_from_mesh(tt_out_w2, mesh_device, dim=1)

    assert tt_pulled_w0.shape == torch_ref_w0.shape
    assert tt_pulled_w1.shape == torch_ref_w1.shape
    assert tt_pulled_w2.shape == torch_ref_w2.shape
    assert torch.equal(tt_pulled_w0, torch_ref_w0)
    assert torch.equal(tt_pulled_w1, torch_ref_w1)
    assert torch.equal(tt_pulled_w2, torch_ref_w2)
