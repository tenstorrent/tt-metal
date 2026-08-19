# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for the point_to_point CCL op.

point_to_point is PURE byte movement (no arithmetic): the receiver device's output
shard must equal the sender device's input shard. So the precision oracle is
IDENTITY, and the interesting number is not "how close" but "is it exact" — the
transfer copies the stored bytes verbatim, including any dtype quantization that
already happened at ``from_torch`` time (e.g. ``bfloat8_b``). The reference is
therefore the *device-resident* sender shard, not the original torch tensor.

Each case records PCC, max abs error, mean abs error and relative RMS error, and
additionally asserts bit-exactness (``max_abs == 0``) — the real contract. A
tolerance-only assertion would let a single corrupted page through on a large
shard, which is exactly the failure mode the framing/alignment logic can produce.

Verification topology (MUST match ``scripts/multidevice_sim_topologies.yaml``)
-----------------------------------------------------------------------------
Topology ``bh_quietbox_1x4_hw``: a 4-chip Blackhole mesh of shape ``(1, 4)`` with
``fabric_config = ttnn.FabricConfig.FABRIC_1D``. Opening any *other* shape on that
box hangs fabric init with ``Fabric Router Sync: Timeout ... Ethernet handshake
likely failed`` (measured: a ``(1, 2)`` mesh errors every case), so ``MESH_SHAPE``
is pinned to the topology's shape exactly like the acceptance suite.

Run it under the multi-device runner, NOT ``run_safe_pytest.sh``::

    scripts/run_multidevice_sim_pytest.py --op point_to_point --runtime hardware -- \
        tests/ttnn/unit_tests/operations/point_to_point/test_point_to_point_precision_baseline.py -v
"""

from math import prod

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from tests.ttnn.utils_for_testing import assert_with_pcc

from ttnn.operations.point_to_point import point_to_point

# --- verification topology contract -------------------------------------------------
MESH_SHAPE = (1, 4)
FABRIC = {"fabric_config": ttnn.FabricConfig.FABRIC_1D}

# Identity transfer: every float dtype should be effectively exact end to end. These
# are the tolerance floors the assertions use; the measured values are far above them.
PCC = {
    ttnn.float32: 0.9999,
    ttnn.bfloat16: 0.999,
    ttnn.bfloat8_b: 0.99,
}

# small / multi-tile / non-square / larger — a compact 4-shape sweep.
SHAPES = [
    (1, 1, 32, 32),
    (1, 1, 64, 128),
    (1, 1, 96, 64),
    (1, 1, 512, 512),
]

# bfloat16 (primary), float32 (widest float page), bfloat8_b (block-float: proves the
# copy is exact even for a format with no torch equivalent). All TILE — the layout
# axis is covered by the acceptance + golden suites; this file measures numerics.
DTYPES = [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b]


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


@pytest.mark.parametrize("device_params", [FABRIC], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shard_shape", SHAPES)
def test_point_to_point_precision_baseline(mesh_device, dtype, shard_shape):
    if prod(tuple(mesh_device.shape)) < 2:
        pytest.skip("point_to_point requires at least 2 mesh devices")

    layout = ttnn.TILE_LAYOUT
    sender_coord = ttnn.MeshCoordinate(0, 0)
    receiver_coord = ttnn.MeshCoordinate(0, 1)
    send_idx = _linear_index(sender_coord, mesh_device.shape)
    recv_idx = _linear_index(receiver_coord, mesh_device.shape)

    input_tensor, input_shards = _make_input(mesh_device, shard_shape, dtype, layout)

    output_tensor = point_to_point(input_tensor, sender_coord, receiver_coord, topology=ttnn.Topology.Linear)
    ttnn.synchronize_device(mesh_device)
    output_shards = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(output_tensor)]

    golden = input_shards[send_idx]  # device-resident sender shard (post-quantization)
    calc = output_shards[recv_idx]  # receiver shard after the fabric transfer

    pcc, max_abs, mean_abs, rel_rms = _metrics(golden, calc)
    _, allclose_msg = comp_allclose(golden, calc)

    logger.info(
        f"PRECISION_BASELINE point_to_point | shape={tuple(shard_shape)} dtype={dtype} "
        f"PCC={pcc:.7f} max_abs={max_abs:.3e} mean_abs={mean_abs:.3e} "
        f"rel_rms={rel_rms:.3e} | {allclose_msg}"
    )

    # 1. Tolerance floor (shared with the acceptance + golden suites).
    assert_with_pcc(golden, calc, PCC[dtype])
    # 2. The real contract: a pure byte copy is BIT-EXACT, so every error metric is
    #    identically zero. Asserted after the PCC check so a partial-page corruption
    #    (which PCC would smear out on a large shard) still fails loudly.
    assert max_abs == 0.0, f"point_to_point is a byte copy but max|diff| = {max_abs} for {dtype} {shard_shape}"
    assert rel_rms == 0.0, f"non-zero relative RMS ({rel_rms}) for {dtype} {shard_shape}"
