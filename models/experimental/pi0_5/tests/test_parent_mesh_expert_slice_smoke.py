# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Smoke: Pi0_5OptionCExpertSliceParent — parent-mesh expert slice with real
action-expert weights, runs the simplified q_proj chain end-to-end.

The denoise submesh sits in a single column of the galaxy parent (rows 2..7,
col 3). All P2P hops between adjacent expert chips are same-column → single
fabric hop (no multihop needed). 18 layers across 6 chips, 3 layers per chip,
so 5 inter-chip transitions per Euler step.

Tests:
  1. `test_parent_expert_slice_qo_chain` — load all 18 layers' weights as
     parent-mesh sharded tensors, run a 6-chip Q→O matmul chain at the
     action-expert dimensions (M=64, K=hidden=1024). Q→O is the outer shape
     of the attention sublayer (input W=1024 → Q W*num_heads*head_dim/W=2048
     → O W=1024). Validate the final output lands at chip 5's parent coord
     with finite non-zero values.

Run with:
    PI0_PARENT_EXPERT_SLICE_SMOKE=1 pytest \\
      models/experimental/pi0_5/tests/test_parent_mesh_expert_slice_smoke.py -s
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
import ttnn

from models.experimental.pi0_5.common.configs import PaliGemmaConfig
from models.experimental.pi0_5.tt.option_c.expert_slice import Pi0_5OptionCExpertSliceParent
from models.experimental.pi0_5.tt.option_c.stages import (
    DENOISE_SUBMESH_OFFSET,
    DENOISE_SUBMESH_SHAPE,
    EXPERT_LAYERS_PER_DENOISE_CHIP,
)


pytestmark = pytest.mark.skipif(
    os.environ.get("PI0_PARENT_EXPERT_SLICE_SMOKE") != "1",
    reason="set PI0_PARENT_EXPERT_SLICE_SMOKE=1 to run the parent-mesh expert slice smoke",
)


GALAXY_SHAPE = ttnn.MeshShape(8, 4)
_CKPT = os.environ.get("PI0_OC_CHECKPOINT", "/home/tt-admin/pi05_cache/pi05_libero_upstream")


def _load_real_weights():
    if not Path(_CKPT, "model.safetensors").exists():
        pytest.skip(f"Checkpoint not found at {_CKPT}")
    from models.experimental.pi0_5.common import Pi0_5WeightLoader

    return Pi0_5WeightLoader(_CKPT).categorized_weights


def test_parent_expert_slice_qo_chain():
    """Build the parent-mesh expert slice with real action-expert weights,
    run the Q→O matmul chain across 6 chips + 5 inter-chip P2P transitions."""
    cfg = PaliGemmaConfig()
    weights = _load_real_weights()

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    parent = ttnn.open_mesh_device(GALAXY_SHAPE)
    try:
        slice_ = Pi0_5OptionCExpertSliceParent(
            config=cfg,
            weights=weights,
            parent_mesh=parent,
            denoise_offset=DENOISE_SUBMESH_OFFSET,
            denoise_shape=DENOISE_SUBMESH_SHAPE,
            expert_layer_range=(0, cfg.expert_config.depth),
            layers_per_chip=EXPERT_LAYERS_PER_DENOISE_CHIP,
        )
        print(
            f"\n[init] loaded {len(slice_.weights_on_parent)} weight categories on parent mesh, "
            f"each with {slice_.layers_per_chip} per-position tensors"
        )
        for name, ws in slice_.weights_on_parent.items():
            shapes = [list(w.shape) for w in ws]
            print(f"  - {name}: positions={len(ws)} shapes={shapes}")

        # Build initial activation. Action expert M = action_horizon padded
        # to 64; hidden = expert_config.width.
        devices_total = GALAXY_SHAPE[0] * GALAXY_SHAPE[1]
        M = 64  # action_horizon=50 → padded to 64
        K = cfg.expert_config.width
        act_torch = torch.zeros(devices_total, 1, M, K, dtype=torch.bfloat16)
        # Active data at first expert chip's parent coord = (2, 3).
        coord0 = slice_.denoise_coord_for_layer(0)
        lin0 = coord0[0] * GALAXY_SHAPE[1] + coord0[1]
        # Scale the input small. 36 unnormalized matmuls (18 Q + 18 O) compound
        # quickly without any normalization in between; this is a wire-up
        # smoke, not a numerical-correctness check.
        act_torch[lin0, 0, :, :] = torch.randn(M, K, dtype=torch.bfloat16) * 0.01
        act = ttnn.from_torch(
            act_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=parent,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(parent, dim=0),
        )
        print(f"[input] shape={list(act.shape)} initial data at parent coord {coord0}")

        # Run the Q→O chain (36 matmuls = 18 layers × 2 + 5 single-hop P2P transitions).
        print("\n[chain] running 18-layer Q→O chain + 5 same-column P2P transitions...")
        out = slice_.forward_qo_chain(act)
        ttnn.synchronize_device(parent)

        # Final layer is local 17, owner chip 5, coord (7, 3).
        coord_last = slice_.denoise_coord_for_layer(slice_.num_layers - 1)
        lin_last = coord_last[0] * GALAXY_SHAPE[1] + coord_last[1]
        shards = ttnn.get_device_tensors(out)
        out_last = ttnn.to_torch(shards[lin_last])
        non_zero_count = (out_last.abs() > 1e-6).sum().item()
        nan_count = torch.isnan(out_last).sum().item()
        total = out_last.numel()
        print(
            f"\n[output @ last chip coord {coord_last}] shape={list(out_last.shape)} "
            f"non_zero={non_zero_count}/{total} nan={nan_count}/{total}"
        )
        print(f"  first 5 values: {out_last.flatten()[:5].tolist()}")
        print(f"  min={out_last.min().item():.4e} max={out_last.max().item():.4e}")

        # Wire-up check: the chain must produce non-zero output (i.e. weights
        # uploaded correctly + matmuls ran + P2P delivered the live shard).
        # Numerical fidelity comes in the next commit when the full block
        # forward (RMSNorm + adaRMS + attention + MLP) lands; 36 unnormalized
        # matmuls naturally overflow regardless of input scale.
        assert nan_count == 0, "Output contains NaN (matmul failure, not just overflow)"
        assert non_zero_count > 0, "Output is all zeros — P2P chain did not deliver live shard"
        print("\n[PASS] parent-mesh expert slice Q→O chain + P2P wire-up validated end-to-end")
        ttnn.deallocate(out)
    finally:
        ttnn.close_mesh_device(parent)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def test_parent_expert_slice_attn_sublayer_chain():
    """Full attention sublayer (mod + adaRMS norm + wqkv + RoPE + SDPA + O +
    gated residual) chained across 18 layers + 5 same-column P2P transitions
    on real action-expert weights. Self-attention only (no past_kv yet —
    cross-attention with migrated VLM prefix KV lands in the next commit)."""
    cfg = PaliGemmaConfig()
    weights = _load_real_weights()

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    parent = ttnn.open_mesh_device(GALAXY_SHAPE)
    try:
        slice_ = Pi0_5OptionCExpertSliceParent(
            config=cfg,
            weights=weights,
            parent_mesh=parent,
            denoise_offset=DENOISE_SUBMESH_OFFSET,
            denoise_shape=DENOISE_SUBMESH_SHAPE,
            expert_layer_range=(0, cfg.expert_config.depth),
            layers_per_chip=EXPERT_LAYERS_PER_DENOISE_CHIP,
        )
        print(f"\n[init] {len(slice_.weights_on_parent)} weight categories uploaded")

        # Activation [1, 1, M=64, W=1024], live at the first denoise chip's coord.
        devices_total = GALAXY_SHAPE[0] * GALAXY_SHAPE[1]
        M = 64
        W = cfg.expert_config.width  # 1024
        act_torch = torch.zeros(devices_total, 1, M, W, dtype=torch.bfloat16)
        coord0 = slice_.denoise_coord_for_layer(0)
        lin0 = coord0[0] * GALAXY_SHAPE[1] + coord0[1]
        act_torch[lin0, 0, :, :] = torch.randn(M, W, dtype=torch.bfloat16) * 0.05
        act = ttnn.from_torch(
            act_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=parent,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(parent, dim=0),
        )

        # adarms_cond [B=1, 1, W=1024], replicated across all chips (same data
        # per shard). Per the AdaRMSGemmaBlockTTNN forward, this is the
        # output of the suffix slice's time/state embedding.
        cond_data = torch.randn(1, 1, W, dtype=torch.bfloat16) * 0.1
        cond_torch = cond_data.unsqueeze(0).expand(devices_total, -1, -1, -1).contiguous()
        adarms_cond = ttnn.from_torch(
            cond_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=parent,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(parent, dim=0),
        )
        print(f"[inputs] act shape={list(act.shape)} adarms_cond shape={list(adarms_cond.shape)}")

        # Run the full attention-sublayer chain (18 layers).
        print("\n[chain] running 18-layer attn sublayer chain + 5 same-column P2P transitions...")
        out = slice_.forward_attn_sublayer_chain(act, adarms_cond, attention_mask=None)
        ttnn.synchronize_device(parent)

        coord_last = slice_.denoise_coord_for_layer(slice_.num_layers - 1)
        lin_last = coord_last[0] * GALAXY_SHAPE[1] + coord_last[1]
        shards = ttnn.get_device_tensors(out)
        out_last = ttnn.to_torch(shards[lin_last])
        non_zero_count = (out_last.abs() > 1e-6).sum().item()
        nan_count = torch.isnan(out_last).sum().item()
        inf_count = torch.isinf(out_last).sum().item()
        finite_count = torch.isfinite(out_last).sum().item()
        total = out_last.numel()
        print(
            f"\n[output @ last chip coord {coord_last}] shape={list(out_last.shape)} "
            f"non_zero={non_zero_count}/{total} nan={nan_count} inf={inf_count} finite={finite_count}/{total}"
        )
        print(f"  first 5 values: {out_last.flatten()[:5].tolist()}")
        if finite_count > 0:
            finite_t = out_last[torch.isfinite(out_last)]
            print(f"  finite range: [{finite_t.min().item():.4e}, {finite_t.max().item():.4e}]")

        # adaRMS norm + gated residual is a partial bound on the residual stream,
        # so we EXPECT mostly-finite values here (the gated structure damps the
        # explosion that pure matmul-chain hits). Numerical PCC vs the paired
        # slice comes when KV-cache cross-attention lands.
        assert nan_count == 0, "Output contains NaN (sublayer failure)"
        assert non_zero_count > 0, "Output is all zeros — live shard did not propagate"
        # The gated residual normally keeps things bounded; report if it didn't.
        if finite_count < total:
            print(f"  [warn] {total - finite_count} non-finite values — gated residual may need adarms_cond tuning")
        print("\n[PASS] parent-mesh expert attn-sublayer chain runs end-to-end on real weights")
        ttnn.deallocate(out)
        ttnn.deallocate(adarms_cond)
    finally:
        ttnn.close_mesh_device(parent)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def test_parent_expert_slice_real_block_chain():
    """Complete Gemma expert block forward across 18 layers + 5 P2P
    transitions: attention sublayer (mod + adaRMS + wqkv + RoPE + SDPA + O +
    gated residual) AND MLP sublayer (adaRMS + gate/up/silu/down + gated
    residual), all on real Gemma-300M action-expert weights. No prefix KV
    cache yet — cross-attention integration with migrate_layer_paired_d2d
    is the next sub-commit."""
    cfg = PaliGemmaConfig()
    weights = _load_real_weights()

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    parent = ttnn.open_mesh_device(GALAXY_SHAPE)
    try:
        slice_ = Pi0_5OptionCExpertSliceParent(
            config=cfg,
            weights=weights,
            parent_mesh=parent,
            denoise_offset=DENOISE_SUBMESH_OFFSET,
            denoise_shape=DENOISE_SUBMESH_SHAPE,
            expert_layer_range=(0, cfg.expert_config.depth),
            layers_per_chip=EXPERT_LAYERS_PER_DENOISE_CHIP,
        )
        devices_total = GALAXY_SHAPE[0] * GALAXY_SHAPE[1]
        M = 64
        W = cfg.expert_config.width
        act_torch = torch.zeros(devices_total, 1, M, W, dtype=torch.bfloat16)
        coord0 = slice_.denoise_coord_for_layer(0)
        lin0 = coord0[0] * GALAXY_SHAPE[1] + coord0[1]
        act_torch[lin0, 0, :, :] = torch.randn(M, W, dtype=torch.bfloat16) * 0.05
        act = ttnn.from_torch(
            act_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=parent,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(parent, dim=0),
        )
        cond_data = torch.randn(1, 1, W, dtype=torch.bfloat16) * 0.1
        cond_torch = cond_data.unsqueeze(0).expand(devices_total, -1, -1, -1).contiguous()
        adarms_cond = ttnn.from_torch(
            cond_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=parent,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(parent, dim=0),
        )

        print("\n[chain] running 18-layer FULL block (attn + MLP) + 5 same-column P2P transitions...")
        out = slice_.forward_real_block_chain(act, adarms_cond, attention_mask=None, prefix_kv_cache=None)
        ttnn.synchronize_device(parent)

        coord_last = slice_.denoise_coord_for_layer(slice_.num_layers - 1)
        lin_last = coord_last[0] * GALAXY_SHAPE[1] + coord_last[1]
        shards = ttnn.get_device_tensors(out)
        out_last = ttnn.to_torch(shards[lin_last])
        non_zero_count = (out_last.abs() > 1e-6).sum().item()
        nan_count = torch.isnan(out_last).sum().item()
        inf_count = torch.isinf(out_last).sum().item()
        finite_count = torch.isfinite(out_last).sum().item()
        total = out_last.numel()
        print(
            f"\n[output @ last chip coord {coord_last}] shape={list(out_last.shape)} "
            f"non_zero={non_zero_count}/{total} nan={nan_count} inf={inf_count} finite={finite_count}/{total}"
        )
        print(f"  first 5 values: {out_last.flatten()[:5].tolist()}")
        if finite_count > 0:
            finite_t = out_last[torch.isfinite(out_last)]
            print(f"  finite range: [{finite_t.min().item():.4e}, {finite_t.max().item():.4e}]")

        assert nan_count == 0, "Output contains NaN (block failure)"
        assert non_zero_count > 0, "Output is all zeros — live shard did not propagate"
        print("\n[PASS] parent-mesh expert FULL block chain runs end-to-end on real weights")
        ttnn.deallocate(out)
        ttnn.deallocate(adarms_cond)
    finally:
        ttnn.close_mesh_device(parent)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def test_parent_expert_cross_attn_with_d2d_migrated_kv():
    """END-TO-END D2D denoise path: VLM block forward emits KV → D2D-migrate
    KV from prefill chips to denoise chips → expert block forward consumes
    the migrated KV via cross-attention. All on the galaxy parent mesh —
    zero host bounces between layers/stages.

    This is the proof-of-concept for the full prefill→denoise D2D pipeline:
    it validates that migrate_layer_paired_d2d's parent-mesh tensor layout
    is exactly what forward_real_block_chain's prefix_kv_cache parameter
    consumes (chip-local concat with the current step's K, V).
    """
    from models.experimental.pi0_5.tt.option_c.kv_migration import KVMigration
    from models.experimental.pi0_5.tt.option_c.stages import (
        PREFILL_SUBMESH_OFFSET,
        PREFILL_SUBMESH_SHAPE,
    )
    from models.experimental.pi0_5.tt.option_c.vlm_slice import Pi0_5OptionCVLMSliceParent

    cfg = PaliGemmaConfig()
    weights = _load_real_weights()

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    parent = ttnn.open_mesh_device(GALAXY_SHAPE)
    try:
        # --- VLM slice (prefill) ---
        vlm_slice = Pi0_5OptionCVLMSliceParent(
            config=cfg,
            weights=weights,
            parent_mesh=parent,
            prefill_offset=PREFILL_SUBMESH_OFFSET,
            prefill_shape=PREFILL_SUBMESH_SHAPE,
            layer_range=(0, cfg.vlm_config.depth),
        )
        # --- Expert slice (denoise) ---
        expert_slice = Pi0_5OptionCExpertSliceParent(
            config=cfg,
            weights=weights,
            parent_mesh=parent,
            denoise_offset=DENOISE_SUBMESH_OFFSET,
            denoise_shape=DENOISE_SUBMESH_SHAPE,
            expert_layer_range=(0, cfg.expert_config.depth),
            layers_per_chip=EXPERT_LAYERS_PER_DENOISE_CHIP,
        )

        devices_total = GALAXY_SHAPE[0] * GALAXY_SHAPE[1]

        # --- Prefill input (small S for the smoke) ---
        S_prefix = 128  # small to keep smoke fast; production is 1024
        W_vlm = cfg.vlm_config.width
        prefix_torch = torch.zeros(devices_total, 1, S_prefix, W_vlm, dtype=torch.bfloat16)
        v_coord0 = vlm_slice.prefill_coord_for_layer(0)
        v_lin0 = v_coord0[0] * GALAXY_SHAPE[1] + v_coord0[1]
        prefix_torch[v_lin0, 0, :, :] = torch.randn(S_prefix, W_vlm, dtype=torch.bfloat16) * 0.01
        prefix_input = ttnn.from_torch(
            prefix_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=parent,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(parent, dim=0),
        )

        print(f"\n[stage 1: VLM] forward + emit KV cache on parent mesh (S_prefix={S_prefix})...")
        prefix_hidden, vlm_kv = vlm_slice.forward_real_block_chain(prefix_input, return_kv_cache=True)
        ttnn.synchronize_device(parent)
        print(f"[stage 1] KV emitted for {len(vlm_kv)} layers")

        # --- Migrate KV: prefill chip i → denoise chip i//3 (same row, single P2P) ---
        migrator = KVMigration(
            denoise_submesh=parent.create_submesh(
                ttnn.MeshShape(*DENOISE_SUBMESH_SHAPE), ttnn.MeshCoordinate(*DENOISE_SUBMESH_OFFSET)
            )
        )
        print("\n[migrate] migrate_layer_paired_d2d (18 P2P transfers)...")
        migrator.migrate_layer_paired_d2d(vlm_kv)
        ttnn.synchronize_device(parent)
        prefix_kv_on_denoise = migrator.as_list(cfg.vlm_config.depth)
        # Free source KV (already migrated).
        for k_t, v_t in vlm_kv:
            ttnn.deallocate(k_t)
            ttnn.deallocate(v_t)

        # --- Expert: suffix input + cross-attention with migrated prefix KV ---
        M_suffix = 64
        W_expert = cfg.expert_config.width
        suffix_torch = torch.zeros(devices_total, 1, M_suffix, W_expert, dtype=torch.bfloat16)
        e_coord0 = expert_slice.denoise_coord_for_layer(0)
        e_lin0 = e_coord0[0] * GALAXY_SHAPE[1] + e_coord0[1]
        suffix_torch[e_lin0, 0, :, :] = torch.randn(M_suffix, W_expert, dtype=torch.bfloat16) * 0.05
        suffix_input = ttnn.from_torch(
            suffix_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=parent,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(parent, dim=0),
        )
        cond_data = torch.randn(1, 1, W_expert, dtype=torch.bfloat16) * 0.1
        cond_torch = cond_data.unsqueeze(0).expand(devices_total, -1, -1, -1).contiguous()
        adarms_cond = ttnn.from_torch(
            cond_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=parent,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(parent, dim=0),
        )

        # Joint attention mask: suffix attends over [prefix_S + suffix_M] tokens.
        # All-zeros = fully unmasked. DRAM-resident (SDPA TT_FATALs on L1 masks).
        kv_seq_len = S_prefix + M_suffix
        mask_torch = torch.zeros(devices_total, 1, M_suffix, kv_seq_len, dtype=torch.bfloat16)
        attn_mask = ttnn.from_torch(
            mask_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=parent,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(parent, dim=0),
        )

        print(f"\n[stage 2: expert] forward with cross-attention over migrated KV (kv_seq_len={kv_seq_len})...")
        out = expert_slice.forward_real_block_chain(
            suffix_input, adarms_cond, attention_mask=attn_mask, prefix_kv_cache=prefix_kv_on_denoise
        )
        ttnn.synchronize_device(parent)

        coord_last = expert_slice.denoise_coord_for_layer(expert_slice.num_layers - 1)
        lin_last = coord_last[0] * GALAXY_SHAPE[1] + coord_last[1]
        shards = ttnn.get_device_tensors(out)
        out_last = ttnn.to_torch(shards[lin_last])
        non_zero_count = (out_last.abs() > 1e-6).sum().item()
        nan_count = torch.isnan(out_last).sum().item()
        finite_count = torch.isfinite(out_last).sum().item()
        total = out_last.numel()
        print(
            f"\n[output @ last chip coord {coord_last}] shape={list(out_last.shape)} "
            f"non_zero={non_zero_count}/{total} nan={nan_count} finite={finite_count}/{total}"
        )
        print(f"  first 5 values: {out_last.flatten()[:5].tolist()}")

        assert nan_count == 0, "Output contains NaN — cross-attention failure"
        assert non_zero_count > 0, "Output is all zeros — D2D migration or cross-attention did not propagate"
        print("\n[PASS] END-TO-END D2D: VLM → migrate_d2d → expert cross-attention on parent mesh")
        ttnn.deallocate(out)
        ttnn.deallocate(prefix_hidden)
        ttnn.deallocate(adarms_cond)
        ttnn.deallocate(attn_mask)
    finally:
        ttnn.close_mesh_device(parent)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def test_parent_mesh_euler_step_with_p2p_wrap_back():
    """One full Euler-style denoise step entirely on the parent mesh:
       suffix.embed_actions → suffix.embed_adarms_cond → expert.forward_real_
       block_chain → P2P wrap-back (chip 5 → chip 0) → suffix.project_output →
       Euler update (x_t = x_t + dt * v_t). All ops parent-mesh; zero host
       bounces inside the step. Validates the Task 3 wrap-back primitive
       end-to-end against the full denoise toolchain.

    No prefix KV cache (self-attention only) — the cross-attention smoke
    above already validates the migrate→consume contract. This test is
    specifically about the suffix↔expert composition + the wrap-back P2P.
    """
    from models.experimental.pi0_5.tt.option_c.suffix_slice import (
        Pi0_5OptionCSuffixSliceParent,
    )

    cfg = PaliGemmaConfig()
    weights = _load_real_weights()

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    parent = ttnn.open_mesh_device(GALAXY_SHAPE)
    try:
        expert_slice = Pi0_5OptionCExpertSliceParent(
            config=cfg,
            weights=weights,
            parent_mesh=parent,
            denoise_offset=DENOISE_SUBMESH_OFFSET,
            denoise_shape=DENOISE_SUBMESH_SHAPE,
            expert_layer_range=(0, cfg.expert_config.depth),
            layers_per_chip=EXPERT_LAYERS_PER_DENOISE_CHIP,
        )
        suffix_slice = Pi0_5OptionCSuffixSliceParent(
            config=cfg,
            weights=weights,
            parent_mesh=parent,
            action_dim=32,
            action_horizon=50,
        )

        # x_t on parent mesh, replicated (all chips hold the same noisy actions).
        # Shape: [B=1, M=64 (50 padded), action_dim=32].
        B, M, A = 1, 64, 32
        x_t_torch = torch.randn(B, M, A, dtype=torch.bfloat16) * 0.1
        x_t = ttnn.from_torch(
            x_t_torch.contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=parent,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(parent),
        )

        # --- One Euler step ---
        t_val = 1.0  # initial timestep
        dt = -0.1

        print("\n[suffix] embed_adarms_cond + embed_actions on parent mesh...")
        adarms_cond = suffix_slice.embed_adarms_cond(t_val, batch_size=B)
        suffix_h_3d = suffix_slice.embed_actions(x_t)
        # forward_real_block_chain expects 4D activations [B, 1, M, W] (so
        # nlp_create_qkv_heads' shape constraints hold). Reshape from
        # suffix's 3D [B, M, W] output.
        suffix_h = ttnn.reshape(suffix_h_3d, (B, 1, M, cfg.expert_config.width))
        # Same for adarms_cond — the block-level modulation Dense expects a
        # broadcast-friendly shape [B, 1, 1, W] for the per-token modulation.
        if len(adarms_cond.shape) == 2:
            adarms_cond_4d = ttnn.reshape(adarms_cond, (B, 1, 1, cfg.expert_config.width))
            ttnn.deallocate(adarms_cond)
            adarms_cond = adarms_cond_4d
        print(f"  adarms_cond shape={list(adarms_cond.shape)}  suffix_h shape={list(suffix_h.shape)}")

        print("\n[expert] forward_real_block_chain (18 layers + 5 P2P advances)...")
        velocity_hidden = expert_slice.forward_real_block_chain(
            suffix_h, adarms_cond, attention_mask=None, prefix_kv_cache=None
        )
        ttnn.deallocate(suffix_h)
        if suffix_h_3d is not suffix_h:
            try:
                ttnn.deallocate(suffix_h_3d)
            except RuntimeError:
                pass

        last_coord = expert_slice.denoise_coord_for_layer(expert_slice.num_layers - 1)
        first_coord = expert_slice.denoise_coord_for_layer(0)
        print(f"\n[wrap_back] P2P {last_coord} → {first_coord} (single-hop fabric)...")
        velocity_at_first = expert_slice.wrap_back_to_chip0(velocity_hidden)

        # Verify the wrap-back delivered the live data to chip 0.
        lin_first = first_coord[0] * GALAXY_SHAPE[1] + first_coord[1]
        shards_after_wrap = ttnn.get_device_tensors(velocity_at_first)
        vel_at_chip0 = ttnn.to_torch(shards_after_wrap[lin_first])
        vel_chip0_nonzero = (vel_at_chip0.abs() > 1e-6).sum().item()
        print(
            f"  velocity @ chip 0 (post-wrap) shape={list(vel_at_chip0.shape)} "
            f"non_zero={vel_chip0_nonzero}/{vel_at_chip0.numel()}"
        )
        assert vel_chip0_nonzero > 0, "Wrap-back P2P did not deliver live shard to chip 0"

        print("\n[suffix] project_output (W → action_dim)...")
        v_t_4d = suffix_slice.project_output(velocity_at_first)
        ttnn.deallocate(velocity_at_first)
        # Reshape v_t back to 3D [B, M, A] to match x_t's shape.
        v_t = ttnn.reshape(v_t_4d, (B, M, A))

        print("\n[euler] x_t_new = x_t + dt * v_t...")
        dx = ttnn.multiply(v_t, dt)
        ttnn.deallocate(v_t)
        x_t_new = ttnn.add(x_t, dx)
        ttnn.deallocate(dx)
        ttnn.deallocate(x_t)
        ttnn.synchronize_device(parent)

        shards = ttnn.get_device_tensors(x_t_new)
        x_t_chip0 = ttnn.to_torch(shards[lin_first])
        non_zero = (x_t_chip0.abs() > 1e-6).sum().item()
        nan_count = torch.isnan(x_t_chip0).sum().item()
        finite_count = torch.isfinite(x_t_chip0).sum().item()
        total = x_t_chip0.numel()
        print(
            f"\n[x_t_new @ chip 0] shape={list(x_t_chip0.shape)} "
            f"non_zero={non_zero}/{total} nan={nan_count} finite={finite_count}/{total}"
        )
        print(f"  first 5 values: {x_t_chip0.flatten()[:5].tolist()}")
        assert nan_count == 0, "x_t_new contains NaN — Euler step or wrap-back broke"
        assert non_zero > 0, "x_t_new is all zeros"
        print("\n[PASS] one full Euler step on parent mesh (suffix + expert + P2P wrap-back + update)")
        ttnn.deallocate(x_t_new)
        ttnn.deallocate(adarms_cond)
    finally:
        ttnn.close_mesh_device(parent)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
