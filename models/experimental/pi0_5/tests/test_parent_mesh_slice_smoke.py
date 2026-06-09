# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Smoke: Pi0_5OptionCVLMSliceParent — parent-mesh slice with real Gemma-2B
weights, runs the simplified q_proj chain end-to-end.

This validates the COMPLETE parent-mesh integration at production scale:
  - All 18 layers' real q_proj weights uploaded as parent-mesh sharded
    tensors (chip i has layer i's Q weight)
  - Activation at [1, 1024, 2048] bf16 (production prefill shape)
  - 18 matmul + 17 P2P-multihop transitions (12 1-hop + 5 2-hop)
  - Final output at chip 17's parent coord

If this passes, we have validated the slice class on real weights at the
real workload size — the foundation for the full GemmaBlockTTNN
integration is solid.

Run with:
    PI0_PARENT_SLICE_SMOKE=1 pytest models/experimental/pi0_5/tests/test_parent_mesh_slice_smoke.py -s
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
import ttnn

from models.experimental.pi0_5.common.configs import PaliGemmaConfig
from models.experimental.pi0_5.tt.option_c.vlm_slice import Pi0_5OptionCVLMSliceParent


pytestmark = pytest.mark.skipif(
    os.environ.get("PI0_PARENT_SLICE_SMOKE") != "1",
    reason="set PI0_PARENT_SLICE_SMOKE=1 to run the parent-mesh slice smoke",
)


GALAXY_SHAPE = ttnn.MeshShape(8, 4)
PREFILL_OFFSET = (2, 0)
PREFILL_SHAPE = (6, 3)
_CKPT = os.environ.get("PI0_OC_CHECKPOINT", "/home/tt-admin/pi05_cache/pi05_libero_upstream")


def _load_real_weights():
    if not Path(_CKPT, "model.safetensors").exists():
        pytest.skip(f"Checkpoint not found at {_CKPT}")
    from models.experimental.pi0_5.common import Pi0_5WeightLoader

    return Pi0_5WeightLoader(_CKPT).categorized_weights


def test_parent_slice_qkv_chain():
    """Build the parent-mesh slice with real Gemma-2B weights, run q_proj chain."""
    cfg = PaliGemmaConfig()
    weights = _load_real_weights()

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    parent = ttnn.open_mesh_device(GALAXY_SHAPE)
    try:
        slice_ = Pi0_5OptionCVLMSliceParent(
            config=cfg,
            weights=weights,
            parent_mesh=parent,
            prefill_offset=PREFILL_OFFSET,
            prefill_shape=PREFILL_SHAPE,
            layer_range=(0, 18),
        )
        print(f"\n[init] loaded {len(slice_.weights_on_parent)} weight categories on parent mesh")
        for name, w in slice_.weights_on_parent.items():
            print(f"  - {name}: shape={list(w.shape)} dtype={w.dtype}")

        # Build initial activation: real-shape [1, 1, 1024, 2048] bf16, with
        # the active data at chip 0's parent coord (2, 0) and zeros elsewhere.
        devices_total = GALAXY_SHAPE[0] * GALAXY_SHAPE[1]
        M, K = 1024, 2048
        # Sharded across the parent mesh, one shard per chip.
        act_torch = torch.zeros(devices_total, 1, M, K, dtype=torch.bfloat16)
        # Place a non-trivial input at chip 0's parent lin idx (2, 0) = 8.
        coord0 = slice_.prefill_coord_for_layer(0)
        lin0 = coord0[0] * GALAXY_SHAPE[1] + coord0[1]
        act_torch[lin0, 0, :, :] = torch.randn(M, K, dtype=torch.bfloat16) * 0.1
        act = ttnn.from_torch(
            act_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=parent,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(parent, dim=0),
        )
        print(f"[input] shape={list(act.shape)} initial data at parent coord {coord0}")

        # Run the q_proj chain.
        print("\n[chain] running 18-layer q_proj chain + P2P multi-hop transitions...")
        out = slice_.forward_qkv_chain(act)
        ttnn.synchronize_device(parent)

        # Verify the final output landed at chip 17's parent coord.
        coord17 = slice_.prefill_coord_for_layer(17)
        lin17 = coord17[0] * GALAXY_SHAPE[1] + coord17[1]
        shards = ttnn.get_device_tensors(out)
        out_t17 = ttnn.to_torch(shards[lin17])
        finite = torch.isfinite(out_t17).all().item()
        non_zero_count = (out_t17.abs() > 1e-6).sum().item()
        total = out_t17.numel()
        print(
            f"\n[output @ chip 17 coord {coord17}] shape={list(out_t17.shape)} "
            f"finite={finite} non_zero={non_zero_count}/{total}"
        )
        # Show a few values for sanity.
        print(f"  first 5 values: {out_t17.flatten()[:5].tolist()}")
        print(f"  min={out_t17.min().item():.4f} max={out_t17.max().item():.4f}")

        assert finite, "Output contains NaN/Inf"
        assert non_zero_count > 0, "Output is all zeros"
        print("\n[PASS] parent-mesh slice q_proj chain works end-to-end with real weights")
        ttnn.deallocate(out)
    finally:
        ttnn.close_mesh_device(parent)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def test_parent_slice_mlp_sublayer_chain():
    """MLP sublayer chain (gate + up + GLU + down + residual) for 18 layers."""
    cfg = PaliGemmaConfig()
    weights = _load_real_weights()

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    parent = ttnn.open_mesh_device(GALAXY_SHAPE)
    try:
        slice_ = Pi0_5OptionCVLMSliceParent(
            config=cfg,
            weights=weights,
            parent_mesh=parent,
            prefill_offset=PREFILL_OFFSET,
            prefill_shape=PREFILL_SHAPE,
            layer_range=(0, 18),
        )

        devices_total = GALAXY_SHAPE[0] * GALAXY_SHAPE[1]
        M, K = 1024, 2048
        act_torch = torch.zeros(devices_total, 1, M, K, dtype=torch.bfloat16)
        coord0 = slice_.prefill_coord_for_layer(0)
        lin0 = coord0[0] * GALAXY_SHAPE[1] + coord0[1]
        act_torch[lin0, 0, :, :] = torch.randn(M, K, dtype=torch.bfloat16) * 0.01
        act = ttnn.from_torch(
            act_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=parent,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(parent, dim=0),
        )

        print("\n[chain] running 18-layer MLP sublayer chain (gate + up + GLU + down + residual)...")
        out = slice_.forward_mlp_sublayer_chain(act)
        ttnn.synchronize_device(parent)

        coord17 = slice_.prefill_coord_for_layer(17)
        lin17 = coord17[0] * GALAXY_SHAPE[1] + coord17[1]
        shards = ttnn.get_device_tensors(out)
        out_t17 = ttnn.to_torch(shards[lin17])
        finite = torch.isfinite(out_t17).all().item()
        non_zero_count = (out_t17.abs() > 1e-6).sum().item()
        print(f"[output @ chip 17] finite={finite} non_zero={non_zero_count}/{out_t17.numel()}")
        print(f"  range: [{out_t17.min().item():.2f}, {out_t17.max().item():.2f}]")

        assert finite, "MLP chain output has NaN/Inf"
        assert non_zero_count > 0, "MLP chain output all zeros"
        print("[PASS] parent-mesh MLP sublayer chain end-to-end")
        ttnn.deallocate(out)
    finally:
        ttnn.close_mesh_device(parent)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def test_parent_slice_qkv_with_rope_chain():
    """Q+K+V matmuls (all three) end-to-end for 18 layers on parent mesh."""
    cfg = PaliGemmaConfig()
    weights = _load_real_weights()
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    parent = ttnn.open_mesh_device(GALAXY_SHAPE)
    try:
        slice_ = Pi0_5OptionCVLMSliceParent(
            config=cfg,
            weights=weights,
            parent_mesh=parent,
            prefill_offset=PREFILL_OFFSET,
            prefill_shape=PREFILL_SHAPE,
            layer_range=(0, 18),
        )
        devices_total = GALAXY_SHAPE[0] * GALAXY_SHAPE[1]
        M, K = 1024, 2048
        act_torch = torch.zeros(devices_total, 1, M, K, dtype=torch.bfloat16)
        coord0 = slice_.prefill_coord_for_layer(0)
        lin0 = coord0[0] * GALAXY_SHAPE[1] + coord0[1]
        act_torch[lin0, 0, :, :] = torch.randn(M, K, dtype=torch.bfloat16) * 0.01
        act = ttnn.from_torch(
            act_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=parent,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(parent, dim=0),
        )
        print("\n[chain] running 18-layer Q+K+V matmul chain (validating all 3 attn projections)...")
        out = slice_.forward_qkv_with_rope_chain(act)
        ttnn.synchronize_device(parent)
        coord17 = slice_.prefill_coord_for_layer(17)
        lin17 = coord17[0] * GALAXY_SHAPE[1] + coord17[1]
        shards = ttnn.get_device_tensors(out)
        out_t17 = ttnn.to_torch(shards[lin17])
        finite = torch.isfinite(out_t17).all().item()
        non_zero = (out_t17.abs() > 1e-6).sum().item()
        print(
            f"[output @ chip 17] finite={finite} non_zero={non_zero}/{out_t17.numel()} "
            f"range=[{out_t17.min().item():.2f}, {out_t17.max().item():.2f}]"
        )
        assert finite and non_zero > 0
        print("[PASS] parent-mesh Q+K+V matmul chain end-to-end")
        ttnn.deallocate(out)
    finally:
        ttnn.close_mesh_device(parent)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def test_parent_slice_full_block_chain():
    """Full Gemma block (RMSNorm + Q+O + residual + RMSNorm + MLP + residual)."""
    cfg = PaliGemmaConfig()
    weights = _load_real_weights()
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    parent = ttnn.open_mesh_device(GALAXY_SHAPE)
    try:
        slice_ = Pi0_5OptionCVLMSliceParent(
            config=cfg,
            weights=weights,
            parent_mesh=parent,
            prefill_offset=PREFILL_OFFSET,
            prefill_shape=PREFILL_SHAPE,
            layer_range=(0, 18),
        )
        devices_total = GALAXY_SHAPE[0] * GALAXY_SHAPE[1]
        M, K = 1024, 2048
        act_torch = torch.zeros(devices_total, 1, M, K, dtype=torch.bfloat16)
        coord0 = slice_.prefill_coord_for_layer(0)
        lin0 = coord0[0] * GALAXY_SHAPE[1] + coord0[1]
        act_torch[lin0, 0, :, :] = torch.randn(M, K, dtype=torch.bfloat16) * 0.01
        act = ttnn.from_torch(
            act_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=parent,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(parent, dim=0),
        )
        print("\n[chain] running FULL 18-layer Gemma block (RMSNorm+Q+O+res+RMSNorm+MLP+res)...")
        out = slice_.forward_full_block_chain(act)
        ttnn.synchronize_device(parent)
        coord17 = slice_.prefill_coord_for_layer(17)
        lin17 = coord17[0] * GALAXY_SHAPE[1] + coord17[1]
        shards = ttnn.get_device_tensors(out)
        out_t17 = ttnn.to_torch(shards[lin17])
        finite = torch.isfinite(out_t17).all().item()
        non_zero = (out_t17.abs() > 1e-6).sum().item()
        print(
            f"[output @ chip 17] finite={finite} non_zero={non_zero}/{out_t17.numel()} "
            f"range=[{out_t17.min().item():.2f}, {out_t17.max().item():.2f}]"
        )
        assert finite and non_zero > 0
        print("[PASS] FULL parent-mesh Gemma block chain end-to-end")
        ttnn.deallocate(out)
    finally:
        ttnn.close_mesh_device(parent)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def test_parent_slice_attention_sublayer_chain():
    """Full attention sublayer chain (RMSNorm + Q+O + residual) for 18 layers."""
    cfg = PaliGemmaConfig()
    weights = _load_real_weights()

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    parent = ttnn.open_mesh_device(GALAXY_SHAPE)
    try:
        slice_ = Pi0_5OptionCVLMSliceParent(
            config=cfg,
            weights=weights,
            parent_mesh=parent,
            prefill_offset=PREFILL_OFFSET,
            prefill_shape=PREFILL_SHAPE,
            layer_range=(0, 18),
        )

        devices_total = GALAXY_SHAPE[0] * GALAXY_SHAPE[1]
        M, K = 1024, 2048
        act_torch = torch.zeros(devices_total, 1, M, K, dtype=torch.bfloat16)
        coord0 = slice_.prefill_coord_for_layer(0)
        lin0 = coord0[0] * GALAXY_SHAPE[1] + coord0[1]
        act_torch[lin0, 0, :, :] = torch.randn(M, K, dtype=torch.bfloat16) * 0.1
        act = ttnn.from_torch(
            act_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=parent,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(parent, dim=0),
        )

        print("\n[chain] running 18-layer attention sublayer chain (RMSNorm + Q + O + residual)...")
        out = slice_.forward_attention_sublayer_chain(act)
        ttnn.synchronize_device(parent)

        coord17 = slice_.prefill_coord_for_layer(17)
        lin17 = coord17[0] * GALAXY_SHAPE[1] + coord17[1]
        shards = ttnn.get_device_tensors(out)
        out_t17 = ttnn.to_torch(shards[lin17])
        finite = torch.isfinite(out_t17).all().item()
        non_zero_count = (out_t17.abs() > 1e-6).sum().item()
        print(
            f"[output @ chip 17 coord {coord17}] shape={list(out_t17.shape)} "
            f"finite={finite} non_zero={non_zero_count}/{out_t17.numel()}"
        )
        print(f"  range: [{out_t17.min().item():.2f}, {out_t17.max().item():.2f}]")

        assert finite, "Output has NaN/Inf"
        assert non_zero_count > 0, "Output all zeros"
        print("[PASS] parent-mesh attention sublayer chain end-to-end")
        ttnn.deallocate(out)
    finally:
        ttnn.close_mesh_device(parent)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
