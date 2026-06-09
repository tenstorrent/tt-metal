# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Smoke: can `ttnn.point_to_point` operate on a tensor allocated on the
vision SUBMESH (not the parent mesh)?

Decision blocker for SigLIP D2D. If P2P works on the 2x4 vision submesh,
SigLIP weights can stay chunk-local at the existing ~75 MB/chip budget.
If P2P requires the parent mesh, we'd have to pad to a [32, ...] sharded
tensor, which inflates non-vision chips' L1 by 32-130 MB and pushes the
prefill chips (~120 MB current) over the 180 MB L1 cap.

Run with:
    PI0_VISION_P2P_SMOKE=1 pytest -s \
      models/experimental/pi0_5/tests/test_vision_submesh_p2p_smoke.py
"""

from __future__ import annotations

import os

import pytest
import torch
import ttnn


pytestmark = pytest.mark.skipif(
    os.environ.get("PI0_VISION_P2P_SMOKE") != "1",
    reason="set PI0_VISION_P2P_SMOKE=1 to run the vision-submesh P2P smoke",
)


GALAXY_SHAPE = ttnn.MeshShape(8, 4)
VISION_OFFSET = (0, 0)
VISION_SHAPE = (2, 4)


def test_p2p_within_vision_submesh():
    """Try P2P (0,0)→(0,1) on a tensor allocated on the 2x4 vision submesh.
    If this passes we can do chunk-local SigLIP weight uploads."""
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    parent = ttnn.open_mesh_device(GALAXY_SHAPE)
    try:
        vision_submesh = parent.create_submesh(
            ttnn.MeshShape(*VISION_SHAPE),
            ttnn.MeshCoordinate(*VISION_OFFSET),
        )
        # Allocate a tensor on the vision submesh, sharded across its 8 chips.
        num_chips = VISION_SHAPE[0] * VISION_SHAPE[1]
        M, N = 32, 32  # tile-sized to keep the kernel happy
        torch_t = torch.zeros(num_chips, M, N, dtype=torch.bfloat16)
        # Put a recognizable signal at submesh-coord (0, 0) = lin idx 0.
        torch_t[0, :, :] = 42.0
        tensor_on_vision = ttnn.from_torch(
            torch_t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=vision_submesh,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(vision_submesh, dim=0),
        )
        print(f"\n[setup] allocated [num_chips={num_chips}, {M}, {N}] on vision submesh ({VISION_SHAPE})")

        # Try ttnn.point_to_point from submesh-coord (0,0) → (0,1).
        # The submesh has its own coordinate space starting at (0,0).
        src = ttnn.MeshCoordinate(0, 0)
        dst = ttnn.MeshCoordinate(0, 1)
        print(f"[p2p] attempting point_to_point src={src} → dst={dst} on vision submesh...")
        result = ttnn.point_to_point(tensor_on_vision, src, dst, topology=ttnn.Topology.Linear)
        ttnn.synchronize_device(vision_submesh)

        # Read back: dst shard (lin idx 1) should now hold the sender's bytes (42.0).
        shards = ttnn.get_device_tensors(result)
        dst_shard = ttnn.to_torch(shards[1])
        max_val = dst_shard.max().item()
        min_val = dst_shard.min().item()
        print(f"[result] dst shard: min={min_val:.2f} max={max_val:.2f}")
        assert abs(max_val - 42.0) < 0.5, f"P2P on vision submesh did NOT deliver sender's bytes; got max={max_val}"
        print("\n[PASS] P2P works on vision submesh — SigLIP D2D can use chunk-local weight uploads")
        ttnn.deallocate(result)
        ttnn.deallocate(tensor_on_vision)
    finally:
        ttnn.close_mesh_device(parent)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
