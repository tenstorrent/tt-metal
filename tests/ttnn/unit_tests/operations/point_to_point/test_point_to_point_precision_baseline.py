# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for the point_to_point CCL op.

point_to_point is PURE byte movement (no arithmetic): the receiver device's
output shard must equal the sender device's input shard. So the precision
oracle is identity — for float dtypes we expect PCC ~ 1.0 and (because the
transfer copies the stored bytes verbatim) bit-exact equality against the
*device-resident* input shard, including any dtype quantization that already
happened at ``from_torch`` time (e.g. bfloat8_b).

This is a multi-device op (needs a >=2-device ``ttnn.MeshDevice`` with the
fabric enabled), so — exactly like the acceptance suite — it must be run under
the deterministic multi-device sim runner, NOT ``run_safe_pytest.sh``. It opens
exactly the graded topology's mesh shape (2, 4) + FABRIC_1D, so drive it by the
matching topology (``--op`` fans across all p2p topologies whose mesh shapes
differ and would hang fabric init):

    scripts/run_multidevice_sim_pytest.py --topology bh_8xP150_p2p -- \
        tests/ttnn/unit_tests/operations/point_to_point/test_point_to_point_precision_baseline.py -v

It records PCC, max abs error, mean abs error, and relative RMS error per shape.
"""

from math import prod

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc

from ttnn.operations.point_to_point import point_to_point


# Identity transfer: float dtypes should be effectively exact end to end.
PCC = {
    ttnn.float32: 0.9999,
    ttnn.bfloat16: 0.999,
    ttnn.bfloat8_b: 0.99,
}

LINEAR = ({"fabric_config": ttnn.FabricConfig.FABRIC_1D}, ttnn.Topology.Linear)

# small / multi-tile / non-square / larger — a compact 4-shape sweep.
SHAPES = [
    (1, 1, 32, 32),
    (1, 1, 64, 128),
    (1, 1, 96, 64),
    (1, 1, 512, 512),
]


def _linear_index(coord, mesh_shape):
    mesh_shape = tuple(mesh_shape)
    return coord[0] * mesh_shape[1] + coord[1]


def _make_input(mesh_device, shard_shape, dtype, layout):
    num_devices = prod(tuple(mesh_device.shape))
    full_shape = (shard_shape[0] * num_devices, *shard_shape[1:])

    torch.manual_seed(42)
    torch_full = torch.randn(full_shape, dtype=torch.float32)
    if dtype == ttnn.bfloat16:
        torch_full = torch_full.to(torch.bfloat16)

    input_tensor = ttnn.from_torch(
        torch_full,
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    ttnn.synchronize_device(mesh_device)
    input_shards = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(input_tensor)]
    return input_tensor, input_shards


def _metrics(golden: torch.Tensor, calc: torch.Tensor):
    """PCC + max/mean abs error + relative RMS error against the golden shard."""
    g = golden.to(torch.float32)
    c = calc.to(torch.float32)
    _, pcc = comp_pcc(g, c)
    diff = (g - c).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    denom = g.pow(2).mean().sqrt().item()
    rms = diff.pow(2).mean().sqrt().item()
    rel_rms = rms / denom if denom > 0 else rms
    return pcc, max_abs, mean_abs, rel_rms


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("dtype, layout", [(ttnn.bfloat16, ttnn.TILE_LAYOUT), (ttnn.float32, ttnn.TILE_LAYOUT)])
@pytest.mark.parametrize("shard_shape", SHAPES)
def test_point_to_point_precision_baseline(mesh_device, topology, dtype, layout, shard_shape):
    if prod(tuple(mesh_device.shape)) < 2:
        pytest.skip("point_to_point requires at least 2 mesh devices")

    sender_coord = ttnn.MeshCoordinate(0, 0)
    receiver_coord = ttnn.MeshCoordinate(0, 1)
    send_idx = _linear_index(sender_coord, mesh_device.shape)
    recv_idx = _linear_index(receiver_coord, mesh_device.shape)

    input_tensor, input_shards = _make_input(mesh_device, shard_shape, dtype, layout)

    output_tensor = point_to_point(input_tensor, sender_coord, receiver_coord, topology=topology)
    ttnn.synchronize_device(mesh_device)
    output_shards = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(output_tensor)]

    golden = input_shards[send_idx]  # device-resident sender shard (post-quantization)
    calc = output_shards[recv_idx]  # receiver shard after the fabric transfer

    pcc, max_abs, mean_abs, rel_rms = _metrics(golden, calc)
    _, allclose_msg = comp_allclose(golden, calc)

    logger.info(
        f"p2p precision [{dtype} {layout} {shard_shape}]: "
        f"PCC={pcc:.6f} max_abs={max_abs:.3e} mean_abs={mean_abs:.3e} "
        f"rel_rms={rel_rms:.3e} | {allclose_msg}"
    )

    # Pure copy: receiver shard equals the (device-resident) sender shard.
    assert pcc >= PCC[dtype], f"PCC {pcc} below {PCC[dtype]} for {dtype} {shard_shape}"
