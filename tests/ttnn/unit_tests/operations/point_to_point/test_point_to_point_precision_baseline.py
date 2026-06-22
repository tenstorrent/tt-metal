# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Precision baseline for point_to_point.

point_to_point is a pure cross-chip byte copy — it performs NO arithmetic. The receiver
shard is expected to be BIT-IDENTICAL to the sender's input shard (the same ttnn tensor,
moved over the fabric). So for every dtype the analytical expectation is:

    PCC = 1.0, max_abs_err = 0, mean_abs_err = 0, relative_rms_err = 0

The "precision" here is the copy's fidelity to the *sent* (already-quantized) payload, not
to an idealized float reference — a copy that round-trips bf8b bytes unchanged is perfect
even though bf8b itself is lossy versus float32. Any non-zero error in this test therefore
flags a real transfer bug (dropped/duplicated/mis-framed pages), not quantization.

This op is multi-device; the `mesh_device` fixture auto-skips on machines with < 2 devices.
Metrics are printed (use `-s`) and recorded in the verification report.
"""

import pytest
import torch

import ttnn

from models.common.utility_functions import comp_allclose
from tests.ttnn.utils_for_testing import comp_pcc

from ttnn.operations.point_to_point import point_to_point


def _linear(coord, mesh_shape):
    return coord[0] * mesh_shape[1] + coord[1]


# small, medium, non-square, one larger
SHAPES = [
    (1, 1, 32, 32),
    (1, 1, 64, 128),
    (2, 3, 32, 64),
    (1, 1, 128, 1024),
]

DTYPES = [
    pytest.param(ttnn.bfloat16, torch.bfloat16, 0.999, id="bfloat16"),
    pytest.param(ttnn.float32, torch.float32, 0.9999, id="float32"),
    pytest.param(ttnn.bfloat8_b, torch.bfloat16, 0.99, id="bfloat8_b"),
]


def _relative_rms(expected: torch.Tensor, received: torch.Tensor) -> float:
    e = expected.to(torch.float32)
    r = received.to(torch.float32)
    denom = torch.sqrt(torch.mean(e * e)).item()
    rms = torch.sqrt(torch.mean((e - r) ** 2)).item()
    return rms / denom if denom > 0 else rms


@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
@pytest.mark.parametrize("ttnn_dtype, torch_dtype, pcc_threshold", DTYPES)
def test_precision_baseline(mesh_device, shape, ttnn_dtype, torch_dtype, pcc_threshold):
    torch.manual_seed(42)
    mesh_shape = tuple(mesh_device.shape)
    num_devices = mesh_shape[0] * mesh_shape[1]
    S = shape[0]
    global_shape = (S * num_devices,) + tuple(shape[1:])

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
        topology=ttnn.Topology.Linear,
    )
    out_torch = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))

    sl, rl = _linear(sender, mesh_shape), _linear(receiver, mesh_shape)
    # Compare against the *sent* shard read back through ttnn so both sides share the
    # exact same quantization — a correct copy is then bit-identical.
    sent_back = ttnn.to_torch(input_tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    expected = sent_back[sl * S : (sl + 1) * S].to(torch.float32)
    received = out_torch[rl * S : (rl + 1) * S].to(torch.float32)

    pcc_passed, pcc_msg = comp_pcc(expected, received, pcc_threshold)
    _, allclose_msg = comp_allclose(expected, received)
    max_abs = (expected - received).abs().max().item()
    mean_abs = (expected - received).abs().mean().item()
    rel_rms = _relative_rms(expected, received)

    print(
        f"\n[precision] shape={shape} dtype={ttnn_dtype} | {pcc_msg} | {allclose_msg} | "
        f"max_abs_err={max_abs:.3e} mean_abs_err={mean_abs:.3e} rel_rms_err={rel_rms:.3e}"
    )

    assert pcc_passed, pcc_msg
