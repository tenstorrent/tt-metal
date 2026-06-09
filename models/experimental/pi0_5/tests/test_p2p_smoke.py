# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Smoke test for `ttnn.point_to_point` on the Option C BH Galaxy parent mesh.

Mirrors the canonical pattern in tests/nightly/t3000/ccl/test_point_to_point.py:
shard a torch tensor along dim 0 so each chip gets a unique slice, P2P from
one chip's slice to another, verify the destination shard received the source
shard's bytes.

Run with:
    PI0_P2P_SMOKE=1 pytest models/experimental/pi0_5/tests/test_p2p_smoke.py -s
"""

from __future__ import annotations

import os

import pytest
import torch
import ttnn

from models.experimental.pi0_5.tt.option_c.transport import send_shard_via_p2p


pytestmark = pytest.mark.skipif(
    os.environ.get("PI0_P2P_SMOKE") != "1",
    reason="set PI0_P2P_SMOKE=1 to run the point_to_point smoke",
)


GALAXY_SHAPE = ttnn.MeshShape(8, 4)


def _linear_coord(row: int, col: int, mesh_cols: int) -> int:
    return row * mesh_cols + col


def _p2p_one(parent, src_row: int, src_col: int, dst_row: int, dst_col: int):
    """Shard a tensor over the parent mesh, P2P src→dst, verify."""
    devices_total = GALAXY_SHAPE[0] * GALAXY_SHAPE[1]
    shape = (1, 1, 32, 128)  # tile-aligned for TILE_LAYOUT
    multi_device_shape = (devices_total, 1, 32, 128)  # shard dim 0 across devices

    # Build a tensor where each chip's slice is filled with its linear index +1.
    # That way after P2P, the receiver shard's elements should match the
    # sender's linear index, not the receiver's.
    src_lin = _linear_coord(src_row, src_col, GALAXY_SHAPE[1])
    dst_lin = _linear_coord(dst_row, dst_col, GALAXY_SHAPE[1])
    input_torch = torch.zeros(multi_device_shape, dtype=torch.bfloat16)
    for i in range(devices_total):
        input_torch[i].fill_(float(i + 1))

    tensor = ttnn.from_torch(
        input_torch,
        layout=ttnn.TILE_LAYOUT,
        device=parent,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(parent, dim=0),
    )

    print(
        f"\n[transfer] (r={src_row},c={src_col})[lin={src_lin}, val={src_lin+1}] → "
        f"(r={dst_row},c={dst_col})[lin={dst_lin}, val_before={dst_lin+1}]"
    )

    # Use the transport.send_shard_via_p2p wrapper (same call site signature
    # as the future inter-layer transport).
    sent = send_shard_via_p2p(tensor, (src_row, src_col), (dst_row, dst_col))
    ttnn.synchronize_device(parent)

    # Read back all shards and check src + dst values
    full = ttnn.to_torch(sent, mesh_composer=ttnn.ConcatMeshToTensor(parent, dim=0))
    src_after = float(full[src_lin].flatten()[0])
    dst_after = float(full[dst_lin].flatten()[0])
    print(
        f"  after p2p: src shard val={src_after}  dst shard val={dst_after}  "
        f"(expected src≈{src_lin+1}, dst≈{src_lin+1})"
    )

    # P2P semantics (per canonical t3000 test): the returned tensor has the
    # sender's data at the dest coord's shard. Other shards (including the
    # sender's own coord) are NOT guaranteed to preserve the input — the op
    # is a "send X from A to B, returning a fresh tensor with X at B".
    # Only assert the destination received the sender's bytes.
    assert abs(dst_after - (src_lin + 1)) < 0.1, f"receiver did not get sender's data: {dst_after} vs {src_lin+1}"
    print("  ✓ dst shard received sender's bytes")
    ttnn.deallocate(sent if sent is not tensor else tensor)
    if sent is not tensor:
        ttnn.deallocate(tensor)


def test_p2p_basic_transfers():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    parent = ttnn.open_mesh_device(GALAXY_SHAPE)
    try:
        # Same row (canonical prefill layer-paired transfer pattern):
        _p2p_one(parent, 2, 0, 2, 1)
        # Same column (e.g. prefill row-to-row):
        _p2p_one(parent, 2, 0, 3, 0)
        # Longer hop same row:
        _p2p_one(parent, 2, 0, 2, 2)
        print("\n[PASS] all 3 P2P transfers (same-row, same-col, multi-hop) verified")
    finally:
        ttnn.close_mesh_device(parent)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
