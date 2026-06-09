# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Smoke test for `KVMigration.migrate_layer_paired_d2d`.

Validates the routing math + fabric transfer for cross-submesh KV migration
between the prefill submesh (rows 2-7, cols 0-2) and the denoise submesh
(rows 2-7, col 3). Synthetic K/V tensors are allocated on the galaxy parent
with a per-layer signature, P2P'd to the denoise coord, and verified that
the right denoise chip received the right layer's data.

Run with:
    PI0_KV_MIG_D2D_SMOKE=1 pytest models/experimental/pi0_5/tests/test_kv_migration_d2d_smoke.py -s
"""

from __future__ import annotations

import os

import pytest
import torch
import ttnn

from models.experimental.pi0_5.tt.option_c.kv_migration import KVMigration


pytestmark = pytest.mark.skipif(
    os.environ.get("PI0_KV_MIG_D2D_SMOKE") != "1",
    reason="set PI0_KV_MIG_D2D_SMOKE=1 to run the kv_migration D2D smoke",
)


GALAXY_SHAPE = ttnn.MeshShape(8, 4)
NUM_LAYERS = 18  # Gemma-2B VLM depth


def _linear_coord(row: int, col: int, mesh_cols: int) -> int:
    return row * mesh_cols + col


def test_kv_migration_d2d_routing():
    """All 18 layers' K and V routed via D2D; verify each denoise chip
    received the right layer's bytes."""
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    parent = ttnn.open_mesh_device(GALAXY_SHAPE)
    try:
        devices_total = GALAXY_SHAPE[0] * GALAXY_SHAPE[1]
        # Per-layer K/V with a unique signature value at each layer's source
        # parent coord. Zeros everywhere else. After P2P, the dest coord's
        # shard should hold the source's signature.
        shape = (1, 1, 32, 64)  # tile-aligned small KV
        multi_device_shape = (devices_total, 1, 32, 64)

        # Build NUM_LAYERS K-tensors and NUM_LAYERS V-tensors on the parent.
        # For layer i, the sender is at parent coord (2+i//3, i%3) — only that
        # shard gets the signature (i+1)*10 for K, (i+1)*100 for V.
        kvs_on_parent = []
        for layer_idx in range(NUM_LAYERS):
            src_row = 2 + layer_idx // 3
            src_col = layer_idx % 3
            src_lin = _linear_coord(src_row, src_col, GALAXY_SHAPE[1])

            k_torch = torch.zeros(multi_device_shape, dtype=torch.bfloat16)
            v_torch = torch.zeros(multi_device_shape, dtype=torch.bfloat16)
            k_torch[src_lin].fill_(float((layer_idx + 1) * 10))
            v_torch[src_lin].fill_(float((layer_idx + 1) * 100))

            k = ttnn.from_torch(
                k_torch,
                layout=ttnn.TILE_LAYOUT,
                device=parent,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(parent, dim=0),
            )
            v = ttnn.from_torch(
                v_torch,
                layout=ttnn.TILE_LAYOUT,
                device=parent,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(parent, dim=0),
            )
            kvs_on_parent.append((k, v))

        # Denoise submesh is needed only for the constructor signature; the
        # D2D path doesn't actually copy to its own mesh (the parent mesh's
        # denoise-coord shards hold the result).
        denoise_submesh = parent.create_submesh(ttnn.MeshShape(6, 1), ttnn.MeshCoordinate(2, 3))
        migrator = KVMigration(denoise_submesh=denoise_submesh)
        migrator.migrate_layer_paired_d2d(kvs_on_parent)

        ttnn.synchronize_device(parent)

        # Verify: for each layer, the dest coord's shard (in the returned
        # migrated tensor) should match the source signature.
        all_ok = True
        for layer_idx in range(NUM_LAYERS):
            k_mig, v_mig = migrator.get(layer_idx)
            denoise_chip = layer_idx // 3
            dst_row = 2 + denoise_chip
            dst_col = 3
            dst_lin = _linear_coord(dst_row, dst_col, GALAXY_SHAPE[1])

            k_full = ttnn.to_torch(k_mig, mesh_composer=ttnn.ConcatMeshToTensor(parent, dim=0))
            v_full = ttnn.to_torch(v_mig, mesh_composer=ttnn.ConcatMeshToTensor(parent, dim=0))
            k_dst = float(k_full[dst_lin].flatten()[0])
            v_dst = float(v_full[dst_lin].flatten()[0])
            k_expected = float((layer_idx + 1) * 10)
            v_expected = float((layer_idx + 1) * 100)
            # bf16 has ~3 decimal digits of mantissa precision; for V values
            # ≥1024 the quantization step is 4. Use relative tolerance.
            ok = abs(k_dst - k_expected) <= max(0.5, 0.01 * k_expected) and abs(v_dst - v_expected) <= max(
                0.5, 0.01 * v_expected
            )
            print(
                f"  layer {layer_idx:2d}: src=(r={2+layer_idx//3},c={layer_idx%3}) "
                f"→ denoise chip {denoise_chip} (parent r={dst_row},c={dst_col})  "
                f"K_dst={k_dst} (exp {k_expected})  V_dst={v_dst} (exp {v_expected})  "
                f"{'✓' if ok else '✗'}"
            )
            all_ok = all_ok and ok

        assert all_ok, "one or more layers' D2D KV migration produced wrong bytes at destination"
        print("\n[PASS] all 18 layers routed correctly via D2D")
    finally:
        ttnn.close_mesh_device(parent)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
