# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Smoke: Pi0_5OptionCVisionSliceSplitParent — parent-mesh SigLIP-27 slice.

This commit lands the SCAFFOLD only:
  - Chunk-stacked sharded weight upload onto the 2x4 vision submesh
  - Per-chunk coord helpers (chunk_coord, chunk_for_layer, position_in_chunk)
  - Validates that real layer weights live at the owning chunk's chip slot
    (and not at non-owner chip slots).

The forward + P2P transitions land in follow-up commits, mirroring the VLM
parent slice rollout (cf55e9ea2c7 → 9f9dbb00cf2 → 0e154616eff → ...).

Run with:
    PI0_PARENT_VISION_SLICE_SMOKE=1 pytest -s \\
      models/experimental/pi0_5/tests/test_parent_mesh_vision_slice_smoke.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import ttnn

from models.experimental.pi0_5.common.configs import PaliGemmaConfig
from models.experimental.pi0_5.tt.option_c.stages import (
    VISION_SUBMESH_OFFSET,
    VISION_SUBMESH_SHAPE,
)
from models.experimental.pi0_5.tt.option_c.vision_slice import (
    Pi0_5OptionCVisionSliceSplitParent,
)


pytestmark = pytest.mark.skipif(
    os.environ.get("PI0_PARENT_VISION_SLICE_SMOKE") != "1",
    reason="set PI0_PARENT_VISION_SLICE_SMOKE=1 to run the parent vision slice smoke",
)


GALAXY_SHAPE = ttnn.MeshShape(8, 4)
_CKPT = os.environ.get("PI0_OC_CHECKPOINT", "/home/tt-admin/pi05_cache/pi05_libero_upstream")


def _load_real_weights():
    if not Path(_CKPT, "model.safetensors").exists():
        pytest.skip(f"Checkpoint not found at {_CKPT}")
    from models.experimental.pi0_5.common import Pi0_5WeightLoader

    return Pi0_5WeightLoader(_CKPT).categorized_weights


def test_parent_vision_slice_weight_placement():
    """Build Pi0_5OptionCVisionSliceSplitParent with real SigLIP weights.
    Validate that each weight category was uploaded as a chunk-stacked tensor
    and that for the q_proj at position 0, chunk i's slot has non-zero data
    iff chunk i owns at least one layer (always true for position 0 here)."""
    cfg = PaliGemmaConfig()
    weights = _load_real_weights()

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    parent = ttnn.open_mesh_device(GALAXY_SHAPE)
    try:
        vision_submesh = parent.create_submesh(
            ttnn.MeshShape(*VISION_SUBMESH_SHAPE),
            ttnn.MeshCoordinate(*VISION_SUBMESH_OFFSET),
        )
        slice_ = Pi0_5OptionCVisionSliceSplitParent(
            config=cfg,
            weights=weights,
            vision_submesh=vision_submesh,
        )
        print(
            f"\n[init] uploaded {len(slice_.weights_on_vision)} weight categories on vision submesh "
            f"({slice_.submesh_shape}); max_layers_per_chunk={slice_.max_layers_per_chunk}"
        )
        for cat, ws in slice_.weights_on_vision.items():
            shapes = [list(w.shape) if w is not None else None for w in ws]
            print(f"  - {cat}: positions={len(ws)} shapes={shapes}")

        # Sanity: chunk → coord mapping for the canonical 4+4+4+4+3+3+3+2 split.
        expected_coords = {
            0: (0, 0),
            1: (0, 1),
            2: (0, 2),
            3: (0, 3),
            4: (1, 0),
            5: (1, 1),
            6: (1, 2),
            7: (1, 3),
        }
        for chunk_idx, expected in expected_coords.items():
            got = slice_.chunk_coord(chunk_idx)
            assert got == expected, f"chunk_coord({chunk_idx}) = {got}, expected {expected}"
        print(f"\n[coords] chunk_coord mapping validated for 8 chunks")

        # Validate q_proj position 0 lives at the right chips (non-zero per chunk).
        q_w_p0 = slice_.weights_on_vision["q_proj"][0]
        assert q_w_p0 is not None, "q_proj[0] is None"
        shards = ttnn.get_device_tensors(q_w_p0)
        print(f"\n[layout] q_proj[position=0] has {len(shards)} shards (one per vision chip)")
        for chunk_idx in range(slice_.num_chunks):
            shard_t = ttnn.to_torch(shards[chunk_idx])
            non_zero = (shard_t.abs() > 1e-6).sum().item()
            total = shard_t.numel()
            # Position 0 exists for every chunk (smallest chunk has 2 layers).
            assert non_zero > 0, (
                f"chunk {chunk_idx} (coord {slice_.chunk_coord(chunk_idx)}): "
                f"q_proj[0] is all zeros at this chip — weight upload broken"
            )
            if chunk_idx < 2 or chunk_idx == 7:
                print(
                    f"  chunk {chunk_idx} @ coord {slice_.chunk_coord(chunk_idx)}: "
                    f"q_proj[0] non_zero={non_zero}/{total}"
                )

        # Validate q_proj position 3 — only chunks 0..3 have a layer at position 3.
        q_w_p3 = slice_.weights_on_vision["q_proj"][3]
        if q_w_p3 is not None:
            shards = ttnn.get_device_tensors(q_w_p3)
            print(f"\n[layout] q_proj[position=3] sparse validation (chunks 0..3 have layer, 4..7 don't):")
            for chunk_idx in range(slice_.num_chunks):
                shard_t = ttnn.to_torch(shards[chunk_idx])
                non_zero = (shard_t.abs() > 1e-6).sum().item()
                expected_nonzero = chunk_idx < 4  # only chunks with 4 layers
                got_nonzero = non_zero > 0
                print(
                    f"  chunk {chunk_idx}: q_proj[3] non_zero={non_zero}/{shard_t.numel()} "
                    f"expected_nonzero={expected_nonzero} got_nonzero={got_nonzero}"
                )
                assert got_nonzero == expected_nonzero, (
                    f"chunk {chunk_idx}: q_proj[3] non_zero placement wrong "
                    f"(expected_nonzero={expected_nonzero}, got_nonzero={got_nonzero})"
                )
        print("\n[PASS] parent-mesh vision slice scaffold: weights upload + chunk-local placement validated")
    finally:
        ttnn.close_mesh_device(parent)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def test_parent_vision_slice_forward_chunk0():
    """Run chunk 0's 4 SigLIP encoder layers via forward_chunk(0) on the vision
    submesh. Validates the inline LayerNorm + fused wqkv + nlp_create_qkv_heads
    + SDPA + nlp_concat_heads + out_proj + MLP block chain on real SigLIP-base
    weights, with the live shard at chunk 0's chip coord (0, 0)."""
    import torch as _torch

    cfg = PaliGemmaConfig()
    weights = _load_real_weights()

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    parent = ttnn.open_mesh_device(GALAXY_SHAPE)
    try:
        vision_submesh = parent.create_submesh(
            ttnn.MeshShape(*VISION_SUBMESH_SHAPE),
            ttnn.MeshCoordinate(*VISION_SUBMESH_OFFSET),
        )
        slice_ = Pi0_5OptionCVisionSliceSplitParent(
            config=cfg,
            weights=weights,
            vision_submesh=vision_submesh,
        )

        # Build a hidden_states input on the vision submesh, live at chunk 0's coord (0,0).
        # Shape: [B, 1, num_patches, hidden] = [1, 1, 256, 1152] (production SigLIP-base
        # patch count + hidden). 4D so nlp_create_qkv_heads' shape contract holds.
        num_chips = VISION_SUBMESH_SHAPE[0] * VISION_SUBMESH_SHAPE[1]
        B, M, H = 1, 256, slice_.hidden_size
        h_torch = _torch.zeros(num_chips, B, M, H, dtype=_torch.bfloat16)
        c0 = slice_.chunk_coord(0)
        lin0 = c0[0] * VISION_SUBMESH_SHAPE[1] + c0[1]
        h_torch[lin0, 0, :, :] = _torch.randn(M, H, dtype=_torch.bfloat16) * 0.02
        # ttnn.from_torch with ShardTensorToMesh(dim=0) on [num_chips, B, M, H] gives
        # per-chip ttnn shape [1, B, M, H] = 4D — exactly what nlp_create_qkv_heads wants.
        hidden = ttnn.from_torch(
            h_torch.contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=vision_submesh,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(vision_submesh, dim=0),
        )
        print(f"\n[input] hidden shape={list(hidden.shape)} live at chunk 0 coord {c0}")

        # Run chunk 0's 4 SigLIP encoder layers.
        print(f"\n[chain] running forward_chunk(0) — 4 SigLIP encoder layers on chip {c0}...")
        out = slice_.forward_chunk(hidden, chunk_idx=0)
        ttnn.synchronize_device(vision_submesh)

        # Read back chunk 0's chip slot to validate output.
        shards = ttnn.get_device_tensors(out)
        out_chip0 = ttnn.to_torch(shards[lin0])
        non_zero = (out_chip0.abs() > 1e-6).sum().item()
        nan_count = _torch.isnan(out_chip0).sum().item()
        finite_count = _torch.isfinite(out_chip0).sum().item()
        total = out_chip0.numel()
        print(
            f"\n[output @ chunk 0 coord {c0}] shape={list(out_chip0.shape)} "
            f"non_zero={non_zero}/{total} nan={nan_count} finite={finite_count}/{total}"
        )
        print(f"  first 5 values: {out_chip0.flatten()[:5].tolist()}")
        if finite_count > 0:
            finite_t = out_chip0[_torch.isfinite(out_chip0)]
            print(f"  finite range: [{finite_t.min().item():.4e}, {finite_t.max().item():.4e}]")

        assert nan_count == 0, "Output contains NaN (chunk forward broke)"
        assert non_zero > 0, "Output is all zeros — chunk forward did not propagate"
        print("\n[PASS] parent-mesh vision slice chunk-0 forward: 4 SigLIP encoder layers ran end-to-end")
        ttnn.deallocate(out)
    finally:
        ttnn.close_mesh_device(parent)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def test_parent_vision_slice_forward_all_chunks():
    """Run all 8 SigLIP chunks (27 encoder layers total) + 7 P2P transitions
    on the vision submesh. The 7 transitions: 6 same-row hops + 1 diagonal
    multihop (chunk 3 → 4, (0,3) → (1,0), routed via (0,0)). Validates the
    full SigLIP encoder D2D chain end-to-end on real weights, with the
    final activation landing at chunk 7's chip coord (1, 3)."""
    import torch as _torch

    cfg = PaliGemmaConfig()
    weights = _load_real_weights()

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    parent = ttnn.open_mesh_device(GALAXY_SHAPE)
    try:
        vision_submesh = parent.create_submesh(
            ttnn.MeshShape(*VISION_SUBMESH_SHAPE),
            ttnn.MeshCoordinate(*VISION_SUBMESH_OFFSET),
        )
        slice_ = Pi0_5OptionCVisionSliceSplitParent(
            config=cfg,
            weights=weights,
            vision_submesh=vision_submesh,
        )

        num_chips = VISION_SUBMESH_SHAPE[0] * VISION_SUBMESH_SHAPE[1]
        B, M, H = 1, 256, slice_.hidden_size
        h_torch = _torch.zeros(num_chips, B, M, H, dtype=_torch.bfloat16)
        c0 = slice_.chunk_coord(0)
        lin0 = c0[0] * VISION_SUBMESH_SHAPE[1] + c0[1]
        h_torch[lin0, 0, :, :] = _torch.randn(M, H, dtype=_torch.bfloat16) * 0.02
        hidden = ttnn.from_torch(
            h_torch.contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=vision_submesh,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(vision_submesh, dim=0),
        )
        print(f"\n[input] hidden shape={list(hidden.shape)} live at chunk 0 coord {c0}")

        print("\n[chain] running forward_all_chunks — 27 SigLIP layers + 7 P2P transitions...")
        out = slice_.forward_all_chunks(hidden)
        ttnn.synchronize_device(vision_submesh)

        # Final live data should be at chunk 7's chip coord (1, 3).
        c_last = slice_.chunk_coord(slice_.num_chunks - 1)
        lin_last = c_last[0] * VISION_SUBMESH_SHAPE[1] + c_last[1]
        shards = ttnn.get_device_tensors(out)
        out_last = ttnn.to_torch(shards[lin_last])
        non_zero = (out_last.abs() > 1e-6).sum().item()
        nan_count = _torch.isnan(out_last).sum().item()
        finite_count = _torch.isfinite(out_last).sum().item()
        total = out_last.numel()
        print(
            f"\n[output @ chunk 7 coord {c_last}] shape={list(out_last.shape)} "
            f"non_zero={non_zero}/{total} nan={nan_count} finite={finite_count}/{total}"
        )
        print(f"  first 5 values: {out_last.flatten()[:5].tolist()}")
        if finite_count > 0:
            finite_t = out_last[_torch.isfinite(out_last)]
            print(f"  finite range: [{finite_t.min().item():.4e}, {finite_t.max().item():.4e}]")

        assert nan_count == 0, "Output contains NaN — chunk chain broke somewhere"
        assert non_zero > 0, "Output all zeros — P2P chain did not deliver live shard to chunk 7"
        print(
            "\n[PASS] parent-mesh vision slice: 27 SigLIP layers + 7 P2P transitions ran end-to-end "
            "on the vision submesh, final activation landed at chunk 7's coord"
        )
        ttnn.deallocate(out)
    finally:
        ttnn.close_mesh_device(parent)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
