# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Probe: can we run a matmul on the (6, 3) prefill submesh with weights
sharded per-chip and validate that each chip computes its own layer's
result?

This is the core building block for the parent-mesh D2D refactor (Option A
in docs/D2D_INTEGRATION_STATUS.md). If this works, we know how to load
weights for 18 layers across 18 chips and run ttnn.linear with per-chip
slices. If not, we know the ttnn limitation precisely.

Run with:
    PI0_PMM_SMOKE=1 pytest models/experimental/pi0_5/tests/test_parent_mesh_matmul_smoke.py -s
"""

from __future__ import annotations

import os

import pytest
import torch
import ttnn


pytestmark = pytest.mark.skipif(
    os.environ.get("PI0_PMM_SMOKE") != "1",
    reason="set PI0_PMM_SMOKE=1 to run the parent-mesh matmul smoke",
)


GALAXY_SHAPE = ttnn.MeshShape(8, 4)
PREFILL_SHAPE = ttnn.MeshShape(6, 3)
PREFILL_OFFSET = ttnn.MeshCoordinate(2, 0)


def test_parent_mesh_matmul():
    """Each of 18 prefill chips has DIFFERENT weights; each computes its
    own matmul on a SHARED replicated activation; verify per-chip outputs
    are distinct and correct.

    This proves the architectural premise of Option A. If outputs are all
    identical (replicated weights), the per-chip-different-weights model
    doesn't work as expected. If outputs are distinct, we know the model
    is viable and can build on it.
    """
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    parent = ttnn.open_mesh_device(GALAXY_SHAPE)
    try:
        prefill = parent.create_submesh(PREFILL_SHAPE, PREFILL_OFFSET)
        num_chips = PREFILL_SHAPE[0] * PREFILL_SHAPE[1]  # 18
        print(f"\n[mesh] prefill {PREFILL_SHAPE} = {num_chips} chips")

        # Activation: replicated across all 18 prefill chips, all ones.
        M, K, N = 32, 64, 32
        act_torch = torch.ones(1, 1, M, K, dtype=torch.bfloat16)
        act = ttnn.from_torch(
            act_torch,
            layout=ttnn.TILE_LAYOUT,
            device=prefill,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(prefill),
        )
        print(f"[act] shape={list(act.shape)} replicated on {num_chips} chips")

        # Weights: SHARDED along dim 0. Each chip gets one [K, N] slice with
        # value = chip_idx + 1 (so chip 0's weight is all 1.0, chip 17's is
        # all 18.0).
        # Stack 18 different [K, N] weights into a [18, K, N] tensor along
        # dim 0; ShardTensorToMesh(dim=0) sends slice i to chip i.
        weights_stacked = torch.zeros(num_chips, K, N, dtype=torch.bfloat16)
        for i in range(num_chips):
            weights_stacked[i].fill_(float(i + 1))
        weight = ttnn.from_torch(
            weights_stacked.view(num_chips, 1, K, N).contiguous(),
            layout=ttnn.TILE_LAYOUT,
            device=prefill,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(prefill, dim=0),
        )
        print(f"[weight] sharded shape={list(weight.shape)} (one [K,N] slice per chip, val=i+1)")

        # Run matmul. Expected: per-chip output[i] = act @ weight[i] =
        # ones_MxK @ ((i+1)*ones_KxN) = (i+1) * K * ones_MxN. With K=64,
        # chip i should produce a uniform matrix of value 64*(i+1).
        out = ttnn.matmul(act, weight)
        ttnn.synchronize_device(prefill)
        print(f"[out] shape={list(out.shape)}")

        # Read each chip's shard and check.
        all_ok = True
        for chip_idx in range(num_chips):
            shards = ttnn.get_device_tensors(out)
            t = ttnn.to_torch(shards[chip_idx])
            actual = float(t.flatten()[0])
            expected = float(K * (chip_idx + 1))
            ok = abs(actual - expected) <= max(0.5, 0.01 * expected)
            print(f"  chip {chip_idx:2d}: out[0]={actual:8.1f} (expected {expected:8.1f}) " f"{'✓' if ok else '✗'}")
            all_ok = all_ok and ok

        assert all_ok, "per-chip matmul outputs don't match expected — per-chip-different-weights model fails"
        print("\n[PASS] per-chip-different-weights matmul model is VIABLE on prefill submesh")
    finally:
        ttnn.close_mesh_device(parent)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
