# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Acceptance test (immutable spec) for the point_to_point CCL data-movement op.

point_to_point sends one mesh device's shard of an interleaved tensor to another
device over the Tenstorrent fabric. It performs NO arithmetic: after the call the
receiver device's shard is bit-identical to the sender device's input shard, and
every non-participating device's shard is untouched.

This op is inherently multi-device, so it uses the `mesh_device` fixture (with a
fabric config) rather than the single-device `device` fixture. On machines with
fewer devices than requested, the fixture skips automatically.

DO NOT MODIFY THIS FILE — it is the acceptance specification the implementer codes against.
"""

import pytest
import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc

# The op under test — newly authored, self-contained Python op (NOT the bound C++ op).
from ttnn.operations.point_to_point import point_to_point


# PCC tolerances keyed by dtype (same thresholds as the golden suite). The transfer is
# a pure copy, so PCC is ~1.0; bf8b's threshold accounts for its block quantization.
DTYPES = [
    pytest.param(ttnn.bfloat16, torch.bfloat16, 0.995, id="bfloat16"),
    pytest.param(ttnn.float32, torch.float32, 0.999, id="float32"),
    pytest.param(ttnn.bfloat8_b, torch.bfloat16, 0.99, id="bfloat8_b"),
]

# Per-device (logical) shard shapes: single-tile, multi-tile, non-square, multi-batch.
SHAPES = [
    (1, 1, 32, 32),
    (1, 1, 64, 128),
    (1, 1, 96, 32),
    (2, 3, 32, 64),
]

# Topology paired with the fabric config it needs (device_params is indirect to mesh_device).
TOPOLOGY_FABRIC = [
    pytest.param({"fabric_config": ttnn.FabricConfig.FABRIC_1D}, ttnn.Topology.Linear, id="linear"),
    pytest.param({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}, ttnn.Topology.Ring, id="ring"),
]


def _linear(coord, mesh_shape):
    """Row-major linear index of a 2-D mesh coordinate."""
    return coord[0] * mesh_shape[1] + coord[1]


def _run_p2p(mesh_device, topology, shape, layout, ttnn_dtype, torch_dtype, *, output_tensor=None):
    """Shard a tensor across the mesh, send sender->receiver, return (out_torch, sender_lin, receiver_lin, S)."""
    torch.manual_seed(42)

    mesh_shape = tuple(mesh_device.shape)
    num_devices = mesh_shape[0] * mesh_shape[1]
    assert num_devices >= 2, "point_to_point requires at least 2 mesh devices"

    sender = (0, 0)
    receiver = (0, 1)  # same row as sender -> 1-D routable
    sender_lin = _linear(sender, mesh_shape)
    receiver_lin = _linear(receiver, mesh_shape)

    S = shape[0]
    global_shape = (S * num_devices,) + tuple(shape[1:])

    # Distinct, recognizable data per device shard via randn.
    torch_input = torch.randn(global_shape, dtype=torch.float32).to(torch_dtype)

    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn_dtype,
        layout=layout,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )

    out = point_to_point(
        input_tensor,
        ttnn.MeshCoordinate(*sender),
        ttnn.MeshCoordinate(*receiver),
        topology=topology,
        output_tensor=output_tensor,
    )

    out_torch = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    return torch_input, out_torch, sender_lin, receiver_lin, S


@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
@pytest.mark.parametrize("device_params, topology", TOPOLOGY_FABRIC, indirect=["device_params"])
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "row_major"])
@pytest.mark.parametrize("ttnn_dtype, torch_dtype, pcc", DTYPES)
def test_point_to_point(mesh_device, topology, shape, layout, ttnn_dtype, torch_dtype, pcc):
    """Receiver shard must equal the sender's input shard, for every dtype/layout/topology."""
    if ttnn_dtype == ttnn.bfloat8_b and layout == ttnn.ROW_MAJOR_LAYOUT:
        pytest.skip("bfloat8_b is a block-quantized format and requires TILE layout")

    torch_input, out_torch, sender_lin, receiver_lin, S = _run_p2p(
        mesh_device, topology, shape, layout, ttnn_dtype, torch_dtype
    )

    expected = torch_input[sender_lin * S : (sender_lin + 1) * S]
    received = out_torch[receiver_lin * S : (receiver_lin + 1) * S]
    assert_with_pcc(expected, received, pcc)


@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
@pytest.mark.parametrize("device_params, topology", TOPOLOGY_FABRIC, indirect=["device_params"])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "row_major"])
def test_point_to_point_output_tensor(mesh_device, topology, layout):
    """The output_tensor path writes into the supplied tensor and returns it."""
    shape = (1, 1, 64, 128)
    ttnn_dtype, torch_dtype, pcc = ttnn.bfloat16, torch.bfloat16, 0.995

    mesh_shape = tuple(mesh_device.shape)
    num_devices = mesh_shape[0] * mesh_shape[1]

    preallocated = ttnn.from_torch(
        torch.zeros((shape[0] * num_devices,) + tuple(shape[1:]), dtype=torch_dtype),
        dtype=ttnn_dtype,
        layout=layout,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )

    torch_input, out_torch, sender_lin, receiver_lin, S = _run_p2p(
        mesh_device, topology, shape, layout, ttnn_dtype, torch_dtype, output_tensor=preallocated
    )

    expected = torch_input[sender_lin * S : (sender_lin + 1) * S]
    received = out_torch[receiver_lin * S : (receiver_lin + 1) * S]
    assert_with_pcc(expected, received, pcc)


@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
@pytest.mark.parametrize("device_params, topology", TOPOLOGY_FABRIC, indirect=["device_params"])
def test_point_to_point_program_cache(mesh_device, topology):
    """Second call with identical shape/dtype/coords/topology must be a cache hit and stay correct.

    This exercises the cache-reuse semaphore-reset footgun: the first run can pass while a
    missing reset hangs/corrupts the second. Both runs must produce the correct receiver shard.
    """
    shape = (1, 1, 64, 128)
    ttnn_dtype, torch_dtype, pcc = ttnn.bfloat16, torch.bfloat16, 0.995

    for _ in range(2):
        torch_input, out_torch, sender_lin, receiver_lin, S = _run_p2p(
            mesh_device, topology, shape, ttnn.TILE_LAYOUT, ttnn_dtype, torch_dtype
        )
        expected = torch_input[sender_lin * S : (sender_lin + 1) * S]
        received = out_torch[receiver_lin * S : (receiver_lin + 1) * S]
        assert_with_pcc(expected, received, pcc)


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize(
    "device_params, topology",
    [pytest.param({"fabric_config": ttnn.FabricConfig.FABRIC_1D}, ttnn.Topology.Linear, id="linear")],
    indirect=["device_params"],
)
def test_point_to_point_other_shards_untouched(mesh_device, topology):
    """Non-participating devices' shards are unchanged (needs >= 4 devices; skips otherwise)."""
    shape = (1, 1, 64, 128)
    ttnn_dtype, torch_dtype, pcc = ttnn.bfloat16, torch.bfloat16, 0.995

    mesh_shape = tuple(mesh_device.shape)
    num_devices = mesh_shape[0] * mesh_shape[1]
    if num_devices < 4:
        pytest.skip("requires a mesh with at least 4 devices to have a non-participating device")

    torch.manual_seed(42)
    S = shape[0]
    global_shape = (S * num_devices,) + tuple(shape[1:])

    # Preallocate the output with a recognizable sentinel so we can detect any stray write.
    sentinel = torch.full(global_shape, 7.0, dtype=torch_dtype)
    output_tensor = ttnn.from_torch(
        sentinel,
        dtype=ttnn_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )

    torch_input = torch.randn(global_shape, dtype=torch.float32).to(torch_dtype)
    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )

    sender, receiver = (0, 0), (0, 1)
    out = point_to_point(
        input_tensor,
        ttnn.MeshCoordinate(*sender),
        ttnn.MeshCoordinate(*receiver),
        topology=topology,
        output_tensor=output_tensor,
    )
    out_torch = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))

    sender_lin = _linear(sender, mesh_shape)
    receiver_lin = _linear(receiver, mesh_shape)

    # Receiver shard == sender input shard.
    assert_with_pcc(
        torch_input[sender_lin * S : (sender_lin + 1) * S],
        out_torch[receiver_lin * S : (receiver_lin + 1) * S],
        pcc,
    )

    # Every non-participating device's output shard still holds the sentinel.
    for d in range(num_devices):
        if d in (sender_lin, receiver_lin):
            continue
        shard = out_torch[d * S : (d + 1) * S]
        assert_with_pcc(sentinel[d * S : (d + 1) * S], shard, pcc)
