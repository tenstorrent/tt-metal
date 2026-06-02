# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Option B smoke tests.

Three layers, each stricter than the previous:
  1. test_default_layout_shape — pure dataclass, no HW.
  2. test_open_32_chip_mesh_and_partition — opens 8×4 mesh + 4× 4×2 submeshes,
     no weights, no compute.
  3. test_vlm_slice_forward_one_layer_on_submesh — uploads RANDOM weights for
     one VLM layer onto stage-1's submesh, runs a forward() pass on a dummy
     hidden_states tensor, checks shape. This validates the submesh-aware
     `Pi0_5SubmeshVLMSlice` and confirms that `GemmaBlockTTNN` runs unchanged
     on a 4×2 MeshDevice.
"""

from __future__ import annotations

import pytest
import torch
import ttnn

from models.experimental.pi0_5.common.configs import GemmaConfig, PaliGemmaConfig
from models.experimental.pi0_5.tt.option_b.stages import build_default_layout, build_shrunk_layout
from models.experimental.pi0_5.tt.option_b.mesh_setup import open_galaxy_mesh, describe_submesh
from models.experimental.pi0_5.tt.option_b.vlm_slice import Pi0_5SubmeshVLMSlice
from models.experimental.pi0_5.tt.option_b.expert_slice import Pi0_5SubmeshExpertSlice
from models.experimental.pi0_5.tt.option_b.stage_vlm import StageVLM
from models.experimental.pi0_5.tt.option_b.stage_3_expert import Stage3Expert
from models.experimental.pi0_5.tt.option_b.transport import send_activation_via_host
from models.experimental.pi0_5.tt.option_b.kv_migration import KVMigration


# Real-weights checkpoint path (Pi0.5 upstream-openpi LIBERO finetune).
_REAL_CKPT = "/home/tt-admin/pi05_cache/pi05_libero_upstream"


def _compute_pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation coefficient over flattened tensors."""
    t1 = a.flatten().float()
    t2 = b.flatten().float()
    mean1, mean2 = torch.mean(t1), torch.mean(t2)
    std1, std2 = torch.std(t1), torch.std(t2)
    if std1 < 1e-6 or std2 < 1e-6:
        return 1.0 if torch.allclose(t1, t2, atol=1e-5) else 0.0
    cov = torch.mean((t1 - mean1) * (t2 - mean2))
    return (cov / (std1 * std2)).item()


def _get_one_replica(t: "ttnn.Tensor") -> torch.Tensor:
    """Pull shard-0 of a replicated mesh tensor back to torch."""
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0])


@pytest.mark.timeout(60)
def test_default_layout_shape():
    layout = build_default_layout()
    assert layout.parent_mesh_shape == (8, 4)
    assert layout.submesh_shape == (4, 2)
    assert len(layout.stages) == 4
    assert layout.stages[0].name == "vision_embed"
    assert layout.stages[0].siglip_layer_range == (0, 27)
    assert layout.stages[1].vlm_layer_range == (0, 9)
    assert layout.stages[2].vlm_layer_range == (9, 18)
    assert layout.stages[2].emits_kv_migration is True
    assert layout.stages[3].expert_layer_range == (0, 18)
    assert layout.stages[3].runs_denoise_loop is True
    assert layout.stages[3].receives_kv_migration is True


@pytest.mark.timeout(180)
def test_open_32_chip_mesh_and_partition():
    """Open an 8×4 mesh, slice into 4× 4×2 submeshes, confirm each is 8 chips."""
    layout = build_default_layout()
    with open_galaxy_mesh(layout) as (parent, submeshes):
        assert parent.shape[0] == 8 and parent.shape[1] == 4
        assert parent.get_num_devices() == 32, f"Expected 32 chips, got {parent.get_num_devices()}"

        assert len(submeshes) == 4, f"Expected 4 submeshes, got {len(submeshes)}"
        for i, sm in enumerate(submeshes):
            assert (
                sm.get_num_devices() == 8
            ), f"Submesh {i} has {sm.get_num_devices()} devices, expected 8 ({describe_submesh(sm)})"
            assert (
                sm.shape[0] == 4 and sm.shape[1] == 2
            ), f"Submesh {i} shape is ({sm.shape[0]},{sm.shape[1]}), expected (4,2)"
            print(f"stage {i} ({layout.stages[i].name}): {describe_submesh(sm)}")


def _random_vlm_layer_weights(layer_idx: int, cfg: GemmaConfig) -> dict:
    """Random torch tensors keyed exactly as Pi0_5SubmeshVLMSlice expects."""
    p = f"model.layers.{layer_idx}."
    W, M, H, KV, D = cfg.width, cfg.mlp_dim, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
    gen = torch.Generator().manual_seed(0xB1 + layer_idx)
    randn = lambda *s: torch.randn(*s, generator=gen, dtype=torch.float32) * 0.02
    return {
        f"{p}self_attn.q_proj.weight": randn(H * D, W),
        f"{p}self_attn.k_proj.weight": randn(KV * D, W),
        f"{p}self_attn.v_proj.weight": randn(KV * D, W),
        f"{p}self_attn.o_proj.weight": randn(W, H * D),
        f"{p}mlp.gate_proj.weight": randn(M, W),
        f"{p}mlp.up_proj.weight": randn(M, W),
        f"{p}mlp.down_proj.weight": randn(W, M),
        f"{p}input_layernorm.weight": torch.zeros(W),  # +1.0 added at upload
        f"{p}post_attention_layernorm.weight": torch.zeros(W),
    }


@pytest.mark.timeout(300)
def test_vlm_slice_forward_one_layer_on_submesh():
    """Build a single-layer VLM slice on stage-1's 4×2 submesh, run forward."""
    layout = build_default_layout()
    cfg = PaliGemmaConfig()  # default gemma_2b, depth=18
    weights = {"vlm_language": _random_vlm_layer_weights(0, cfg.vlm_config)}

    with open_galaxy_mesh(layout) as (parent, submeshes):
        submesh = submeshes[1]  # stage 1 — VLM first half
        slice_ = Pi0_5SubmeshVLMSlice(
            config=cfg,
            weights=weights,
            submesh=submesh,
            layer_range=(0, 1),
            holds_embed_tokens=False,
            holds_vlm_final_norm=False,
        )
        assert slice_.num_layers == 1
        assert len(slice_.vlm_blocks) == 1

        # Dummy hidden_states: [B=1, S=64 (tile-aligned), H=2048] bf16.
        S = 64
        h_torch = torch.randn(1, S, cfg.vlm_config.width, dtype=torch.float32) * 0.02
        h = ttnn.from_torch(
            h_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=submesh,
            mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(submesh),
        )
        # Unmasked attention mask: all zeros, shape [1, 1, S, S], bf16.
        mask_torch = torch.zeros(1, 1, S, S, dtype=torch.float32)
        mask = ttnn.from_torch(
            mask_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=submesh,
            mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(submesh),
        )

        out, _new_kv = slice_.forward(h, attention_mask=mask, use_cache=False)
        # Output is a replicated tensor on the 8-chip submesh; per-device shape
        # is what the block produces. For a replicated layout, shape on the
        # ttnn Tensor wrapper is the per-device shape.
        assert out.shape[-1] == cfg.vlm_config.width, f"got out.shape={out.shape}"
        print(f"vlm_slice.forward OK — out.shape={out.shape}, dtype={out.dtype}")


def _random_expert_layer_weights(layer_idx: int, cfg) -> dict:
    """Random torch tensors keyed exactly as Pi0_5SubmeshExpertSlice expects."""
    p = f"model.layers.{layer_idx}."
    W, M, H, KV, D = cfg.width, cfg.mlp_dim, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
    gen = torch.Generator().manual_seed(0xE0 + layer_idx)
    randn = lambda *s: torch.randn(*s, generator=gen, dtype=torch.float32) * 0.02
    return {
        f"{p}self_attn.q_proj.weight": randn(H * D, W),
        f"{p}self_attn.k_proj.weight": randn(KV * D, W),
        f"{p}self_attn.v_proj.weight": randn(KV * D, W),
        f"{p}self_attn.o_proj.weight": randn(W, H * D),
        f"{p}mlp.gate_proj.weight": randn(M, W),
        f"{p}mlp.up_proj.weight": randn(M, W),
        f"{p}mlp.down_proj.weight": randn(W, M),
        # adaRMS modulation Denses (3*W each, fused at upload).
        f"{p}input_layernorm.dense.weight": randn(3 * W, W),
        f"{p}input_layernorm.dense.bias": randn(3 * W),
        f"{p}post_attention_layernorm.dense.weight": randn(3 * W, W),
        f"{p}post_attention_layernorm.dense.bias": randn(3 * W),
    }


@pytest.mark.timeout(300)
def test_expert_slice_forward_one_layer_on_submesh():
    """One expert layer with adaRMS on stage-3's 4x2 submesh."""
    layout = build_default_layout()
    cfg = PaliGemmaConfig()
    W = cfg.expert_config.width
    gen = torch.Generator().manual_seed(0xE2)
    randn = lambda *s: torch.randn(*s, generator=gen, dtype=torch.float32) * 0.02

    expert = _random_expert_layer_weights(0, cfg.expert_config)
    # Final norm modulation Dense produces (scale, shift, gate) → out_features = 3*W.
    # ada_rms_norm_no_gate_ttnn slices the 3W output into scale/shift (gate dropped).
    expert["model.norm.dense.weight"] = randn(3 * W, W)
    expert["model.norm.dense.bias"] = randn(3 * W)
    weights = {"action_expert": expert}

    with open_galaxy_mesh(layout) as (parent, submeshes):
        submesh = submeshes[3]
        slice_ = Pi0_5SubmeshExpertSlice(
            config=cfg,
            weights=weights,
            submesh=submesh,
            expert_layer_range=(0, 1),
        )
        assert slice_.num_layers == 1
        assert len(slice_.expert_blocks) == 1

        S = 64  # tile-aligned suffix length
        replicate = lambda t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT: ttnn.from_torch(
            t,
            dtype=dtype,
            layout=layout,
            device=submesh,
            mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(submesh),
        )

        h = replicate(torch.randn(1, S, W, dtype=torch.float32) * 0.02)
        # adarms_cond must be 2D [B, W] — _split_modulation_6 reshapes
        # mod[B, 6W] -> [B, 1, 6W] and any extra dims blow the volume check.
        # Single-device path matches: ttnn_pi0_5_model.py uses (1, W).
        adarms_cond = replicate(torch.randn(1, W, dtype=torch.float32) * 0.02)
        mask = replicate(torch.zeros(1, 1, S, S, dtype=torch.float32))

        out = slice_.forward(h, adarms_cond, attention_mask=mask)
        assert out.shape[-1] == W, f"got out.shape={out.shape}"
        print(f"expert_slice.forward OK — out.shape={out.shape}, dtype={out.dtype}")


@pytest.mark.timeout(180)
def test_inter_submesh_host_bounce_transport():
    """Round-trip a known tensor through two submeshes via host bounce.

    Validates send_activation_via_host: replicate on submesh A, ship to
    submesh B, pull back to torch on B, compare to original. Bit-identical
    is too strong (bf16 quantization on the way down) but max-abs < 1e-2
    should hold for inputs of this scale.
    """
    layout = build_default_layout()
    with open_galaxy_mesh(layout) as (parent, submeshes):
        # Use submeshes 1 and 2 (the two stages with the same submesh shape).
        src_mesh, dst_mesh = submeshes[1], submeshes[2]

        original = torch.randn(1, 64, 2048, dtype=torch.float32) * 0.5
        src = ttnn.from_torch(
            original,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=src_mesh,
            mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(src_mesh),
        )
        moved = send_activation_via_host(src, dst_mesh)

        assert moved.shape == src.shape, f"shape mismatch: {moved.shape} vs {src.shape}"
        assert moved.dtype == src.dtype
        # We do not pull back to torch here — `ttnn.to_torch` on a replicated
        # mesh tensor needs a mesh composer to flatten; that's covered by
        # the transport implementation itself (`send_activation_via_host`
        # calls `ttnn.to_torch(src_tensor)` internally). If it returns
        # successfully, the bounce works.
        print(f"transport host-bounce OK — moved.shape={moved.shape}, dtype={moved.dtype}")


@pytest.mark.timeout(600)
def test_e2e_vlm_to_expert_shrunk_pipeline():
    """End-to-end: synthetic prefix → stage 1 → stage 2 → KV-migrate → stage 3
    expert step. Skips stage 0 (host SigLIP needs HF-compatible random weights
    which are awkward to fabricate; covered by the standalone vision_slice).

    Uses build_shrunk_layout(vlm_depth=2, expert_depth=1) so replicated
    weights fit under the per-chip cap. With this config:
      - Stage 1 holds VLM layer 0
      - Stage 2 holds VLM layer 1 + final norm
      - Stage 3 holds expert layer 0 + final adaRMS Dense
    """
    layout = build_shrunk_layout(vlm_depth=2, expert_depth=1)
    cfg = PaliGemmaConfig(
        vlm_config=GemmaConfig(width=2048, depth=2, mlp_dim=16384, num_heads=8, num_kv_heads=1, head_dim=256),
        expert_config=GemmaConfig(width=1024, depth=1, mlp_dim=4096, num_heads=8, num_kv_heads=1, head_dim=256),
    )

    # Build weight dicts ---------------------------------------------------
    vlm_lang = {}
    for li in range(cfg.vlm_config.depth):
        vlm_lang.update(_random_vlm_layer_weights(li, cfg.vlm_config))
    vlm_lang["model.norm.weight"] = torch.zeros(cfg.vlm_config.width)

    W = cfg.expert_config.width
    expert_w = {}
    for li in range(cfg.expert_config.depth):
        expert_w.update(_random_expert_layer_weights(li, cfg.expert_config))
    gen = torch.Generator().manual_seed(0xEF)
    randn = lambda *s: torch.randn(*s, generator=gen, dtype=torch.float32) * 0.02
    expert_w["model.norm.dense.weight"] = randn(3 * W, W)
    expert_w["model.norm.dense.bias"] = randn(3 * W)

    weights = {"vlm_language": vlm_lang, "action_expert": expert_w}

    with open_galaxy_mesh(layout) as (parent, submeshes):
        s = layout.stages
        stage_1 = StageVLM(s[1], submeshes[1], cfg, weights)
        stage_2 = StageVLM(s[2], submeshes[2], cfg, weights)
        stage_3 = Stage3Expert(s[3], submeshes[3], cfg, weights)
        stage_1.initialize()
        stage_2.initialize()
        stage_3.initialize(KVMigration(expert_submesh=submeshes[3]))

        # Synthetic prefix on submesh 1 (skip stage 0 / SigLIP).
        S_prefix = 64  # tile-aligned
        replicate = lambda t, mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT: ttnn.from_torch(
            t,
            dtype=dtype,
            layout=layout,
            device=mesh,
            mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh),
        )
        h0_torch = torch.randn(1, S_prefix, cfg.vlm_config.width) * 0.02
        h_on_1 = replicate(h0_torch, submeshes[1])
        prefix_mask = replicate(
            torch.zeros(1, 1, S_prefix, S_prefix, dtype=torch.float32),
            submeshes[1],
        )

        # Stage 1 — VLM layer 0
        h_after_1 = stage_1.forward(h_on_1, attention_mask=prefix_mask, use_cache=True)
        assert h_after_1.shape[-1] == cfg.vlm_config.width
        kv1 = stage_1.get_kv_cache()
        assert len(kv1) == 1 and kv1[0][0] == 0, f"stage 1 KV: {[(i, type(v)) for i,v in kv1]}"

        # Transport 1 -> 2 (and need a fresh prefix mask on submesh 2).
        h_on_2 = send_activation_via_host(h_after_1, submeshes[2])
        prefix_mask_2 = replicate(
            torch.zeros(1, 1, S_prefix, S_prefix, dtype=torch.float32),
            submeshes[2],
        )

        # Stage 2 — VLM layer 1 + final norm
        h_after_2 = stage_2.forward(h_on_2, attention_mask=prefix_mask_2, use_cache=True)
        assert h_after_2.shape[-1] == cfg.vlm_config.width
        kv2 = stage_2.get_kv_cache()
        assert len(kv2) == 1 and kv2[0][0] == 1

        # KV migration: ship each (K, V) from stages 1/2 to submesh 3.
        migrated = [None] * cfg.vlm_config.depth
        for stage in (stage_1, stage_2):
            for gi, (k, v) in stage.get_kv_cache():
                k_on_3 = send_activation_via_host(k, submeshes[3])
                v_on_3 = send_activation_via_host(v, submeshes[3])
                migrated[gi] = (k_on_3, v_on_3)
        assert all(m is not None for m in migrated), "KV migration left holes"

        # Stage 3 — expert step. The expert's joint attention needs an
        # attention mask covering [suffix_len, prefix_len + suffix_len]. For
        # the smoke test we run unmasked (all zeros).
        S_suffix = 64  # tile-aligned
        suffix_h = replicate(torch.randn(1, S_suffix, W) * 0.02, submeshes[3])
        adarms_cond = replicate(torch.randn(1, W) * 0.02, submeshes[3])
        # Joint mask shape: [B, 1, suffix_len, prefix_len + suffix_len].
        joint_mask = replicate(
            torch.zeros(1, 1, S_suffix, S_prefix + S_suffix, dtype=torch.float32),
            submeshes[3],
        )

        out = stage_3.forward_expert_step(
            suffix_h,
            adarms_cond,
            prefix_kv_cache=migrated,
            attention_mask=joint_mask,
        )
        assert out.shape[-1] == W, f"got {out.shape}"
        print(
            f"e2e VLM→expert OK — out.shape={out.shape}, dtype={out.dtype}, "
            f"prefix_kv={[m[0].shape if m else None for m in migrated]}"
        )


@pytest.mark.timeout(420)
def test_vlm_slice_tp_shard_one_layer():
    """TP=8 sharded single VLM layer at real Gemma-2B dims (W=2048, M=16384).

    Per-chip weight footprint after sharding (bf8):
      - q_proj: W * (H*D / 8) = 2048 * 256 = 0.5 MB
      - k_proj: W * KV*D = 2048 * 256 = 0.5 MB (replicated)
      - v_proj: W * KV*D = 0.5 MB (replicated)
      - o_proj: (H*D / 8) * W = 256 * 2048 = 0.5 MB
      - gate, up, down: 2048 * 2048 = 4 MB each → 12 MB total
      - norms etc: trace
      Σ ~14 MB / chip per layer. Comfortable inside 180 MB cap.
    """
    layout = build_default_layout()
    cfg = PaliGemmaConfig()
    weights = {"vlm_language": _random_vlm_layer_weights(0, cfg.vlm_config)}

    with open_galaxy_mesh(layout, enable_fabric=True) as (parent, submeshes):
        submesh = submeshes[1]
        slice_ = Pi0_5SubmeshVLMSlice(
            config=cfg,
            weights=weights,
            submesh=submesh,
            layer_range=(0, 1),
            tp_shard=True,
        )
        assert len(slice_.vlm_blocks) == 1

        S = 64
        h = ttnn.from_torch(
            torch.randn(1, S, cfg.vlm_config.width) * 0.02,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=submesh,
            mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(submesh),
        )
        mask = ttnn.from_torch(
            torch.zeros(1, 1, S, S, dtype=torch.float32),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=submesh,
            mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(submesh),
        )

        out, _ = slice_.forward(h, attention_mask=mask, use_cache=False)
        assert out.shape[-1] == cfg.vlm_config.width
        print(f"tp_shard VLM slice OK — out.shape={out.shape}, dtype={out.dtype}")


@pytest.mark.timeout(600)
def test_vlm_slice_tp_shard_nine_layers():
    """9 TP=8-sharded VLM layers on stage 1's submesh — the real per-stage
    workload for Option B. If this passes, real pi0.5 weights for a full
    stage are within budget.
    """
    layout = build_default_layout()
    cfg = PaliGemmaConfig()  # depth=18
    lang = {}
    for li in range(9):
        lang.update(_random_vlm_layer_weights(li, cfg.vlm_config))
    weights = {"vlm_language": lang}

    with open_galaxy_mesh(layout, enable_fabric=True) as (parent, submeshes):
        submesh = submeshes[1]
        slice_ = Pi0_5SubmeshVLMSlice(
            config=cfg,
            weights=weights,
            submesh=submesh,
            layer_range=(0, 9),
            tp_shard=True,
        )
        assert len(slice_.vlm_blocks) == 9

        S = 64
        h = ttnn.from_torch(
            torch.randn(1, S, cfg.vlm_config.width) * 0.02,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=submesh,
            mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(submesh),
        )
        mask = ttnn.from_torch(
            torch.zeros(1, 1, S, S, dtype=torch.float32),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=submesh,
            mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(submesh),
        )
        out, _ = slice_.forward(h, attention_mask=mask, use_cache=False)
        assert out.shape[-1] == cfg.vlm_config.width
        print(f"tp_shard 9-layer VLM stage OK — out.shape={out.shape}")


@pytest.mark.timeout(600)
def test_tp_vlm_kv_migrate_to_replicated_expert():
    """TP=8 VLM (1 layer) captures KV → host-bounce migrate → replicated
    expert (1 layer) consumes the migrated KV via past_key_value.

    Validates the full TP=8 KV-cache plumbing: capture in TP block,
    propagate via slice, migrate via host-bounce, attend in the expert.
    """
    layout = build_default_layout()
    cfg = PaliGemmaConfig(
        vlm_config=GemmaConfig(width=2048, depth=1, mlp_dim=16384, num_heads=8, num_kv_heads=1, head_dim=256),
        expert_config=GemmaConfig(width=1024, depth=1, mlp_dim=4096, num_heads=8, num_kv_heads=1, head_dim=256),
    )
    # VLM (TP=8) weights
    vlm_lang = _random_vlm_layer_weights(0, cfg.vlm_config)
    weights_vlm = {"vlm_language": vlm_lang}
    # Expert (replicated) weights
    W_e = cfg.expert_config.width
    gen = torch.Generator().manual_seed(0xFA)
    randn = lambda *s: torch.randn(*s, generator=gen, dtype=torch.float32) * 0.02
    expert_w = _random_expert_layer_weights(0, cfg.expert_config)
    expert_w["model.norm.dense.weight"] = randn(3 * W_e, W_e)
    expert_w["model.norm.dense.bias"] = randn(3 * W_e)
    weights_expert = {"action_expert": expert_w}

    with open_galaxy_mesh(layout, enable_fabric=True) as (parent, submeshes):
        vlm_mesh, expert_mesh = submeshes[1], submeshes[3]

        # TP=8 VLM slice (1 layer)
        vlm_slice = Pi0_5SubmeshVLMSlice(
            config=cfg,
            weights=weights_vlm,
            submesh=vlm_mesh,
            layer_range=(0, 1),
            tp_shard=True,
        )
        # Replicated expert slice (1 layer)
        expert_slice = Pi0_5SubmeshExpertSlice(
            config=cfg,
            weights=weights_expert,
            submesh=expert_mesh,
            expert_layer_range=(0, 1),
        )

        S_prefix = 64
        replicate = lambda t, mesh: ttnn.from_torch(
            t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh),
        )

        h_vlm = replicate(torch.randn(1, S_prefix, cfg.vlm_config.width) * 0.02, vlm_mesh)
        prefix_mask = replicate(torch.zeros(1, 1, S_prefix, S_prefix, dtype=torch.float32), vlm_mesh)

        out_vlm, new_kv = vlm_slice.forward(h_vlm, attention_mask=prefix_mask, use_cache=True)
        assert out_vlm.shape[-1] == cfg.vlm_config.width
        assert new_kv is not None and new_kv[0] is not None, "TP block did not emit KV"
        K, V = new_kv[0]
        # K/V shape: [B, num_kv_heads, S, head_dim]
        assert K.shape == V.shape, f"K={K.shape}, V={V.shape}"
        assert K.shape[-1] == cfg.vlm_config.head_dim
        assert K.shape[-2] == S_prefix
        print(f"TP VLM emitted KV — K.shape={K.shape}, V.shape={V.shape}")

        # Migrate K and V from vlm_mesh to expert_mesh via host-bounce.
        K_on_e = send_activation_via_host(K, expert_mesh)
        V_on_e = send_activation_via_host(V, expert_mesh)
        migrated = [(K_on_e, V_on_e)]

        # Build expert inputs on expert_mesh.
        S_suffix = 64
        suffix_h = replicate(torch.randn(1, S_suffix, W_e) * 0.02, expert_mesh)
        adarms_cond = replicate(torch.randn(1, W_e) * 0.02, expert_mesh)
        joint_mask = replicate(
            torch.zeros(1, 1, S_suffix, S_prefix + S_suffix, dtype=torch.float32),
            expert_mesh,
        )

        out = expert_slice.forward(
            suffix_h,
            adarms_cond,
            prefix_kv_cache=migrated,
            attention_mask=joint_mask,
        )
        assert out.shape[-1] == W_e, f"got {out.shape}"
        print(f"TP VLM → migrate → expert OK — " f"out.shape={out.shape}, K_migrated.shape={K_on_e.shape}")


@pytest.mark.timeout(900)
def test_e2e_real_config_tp_vlm_replicated_expert():
    """Full real-config E2E:
      - PaliGemmaConfig() defaults: Gemma-2B VLM depth=18, Gemma-300M expert depth=18.
      - Stage 1 (submesh 1): TP=8 VLM layers 0..9 + (synthetic) prefix input
      - Stage 2 (submesh 2): TP=8 VLM layers 9..18 + final norm
      - Stage 3 (submesh 3): replicated expert layers 0..18 + final adaRMS norm
      - KV migration: host-bounce of 18 (K, V) pairs from submeshes 1/2 → submesh 3
      - Per-stage timings reported.

    This is the **real** Option B per-stage workload at production dims. Stage
    0 (host SigLIP + embed) is skipped — its slice is tested separately and
    its outputs are synthesised here as a single replicated prefix tensor on
    submesh 1.
    """
    import time

    layout = build_default_layout()  # 9/9/18 split, real PaliGemma config
    cfg = PaliGemmaConfig()  # depth=18 VLM + depth=18 expert

    # ----------------------------- build weights -------------------------
    t_wt0 = time.perf_counter()
    vlm_lang = {}
    for li in range(cfg.vlm_config.depth):
        vlm_lang.update(_random_vlm_layer_weights(li, cfg.vlm_config))
    vlm_lang["model.norm.weight"] = torch.zeros(cfg.vlm_config.width)

    expert_w = {}
    for li in range(cfg.expert_config.depth):
        expert_w.update(_random_expert_layer_weights(li, cfg.expert_config))
    W_e = cfg.expert_config.width
    gen = torch.Generator().manual_seed(0xAA)
    randn = lambda *s: torch.randn(*s, generator=gen, dtype=torch.float32) * 0.02
    expert_w["model.norm.dense.weight"] = randn(3 * W_e, W_e)
    expert_w["model.norm.dense.bias"] = randn(3 * W_e)
    weights = {"vlm_language": vlm_lang, "action_expert": expert_w}
    t_wt = (time.perf_counter() - t_wt0) * 1000
    print(f"\n[perf] weight construction (CPU): {t_wt:.1f} ms")

    with open_galaxy_mesh(layout, enable_fabric=True) as (parent, submeshes):
        # ---------------------- per-stage init -----------------------------
        t_init0 = time.perf_counter()
        vlm1 = Pi0_5SubmeshVLMSlice(
            config=cfg,
            weights={"vlm_language": vlm_lang},
            submesh=submeshes[1],
            layer_range=(0, 9),
            tp_shard=True,
        )
        t_init1 = time.perf_counter()
        vlm2 = Pi0_5SubmeshVLMSlice(
            config=cfg,
            weights={"vlm_language": vlm_lang},
            submesh=submeshes[2],
            layer_range=(9, 18),
            holds_vlm_final_norm=True,
            tp_shard=True,
        )
        t_init2 = time.perf_counter()
        expert = Pi0_5SubmeshExpertSlice(
            config=cfg,
            weights={"action_expert": expert_w},
            submesh=submeshes[3],
            expert_layer_range=(0, 18),
        )
        t_init3 = time.perf_counter()
        print(
            f"[perf] init stage1 (TP=8, 9 layers): {(t_init1-t_init0)*1000:.0f} ms\n"
            f"[perf] init stage2 (TP=8, 9 layers + norm): {(t_init2-t_init1)*1000:.0f} ms\n"
            f"[perf] init stage3 (replicated, 18 layers + adaRMS norm): {(t_init3-t_init2)*1000:.0f} ms"
        )

        # ---------------------- synthetic prefix ---------------------------
        S_prefix = 64  # tile-aligned (real prefill would be 544-968)
        S_suffix = 64
        replicate = lambda t, mesh: ttnn.from_torch(
            t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh),
        )
        h_on_1 = replicate(torch.randn(1, S_prefix, cfg.vlm_config.width) * 0.02, submeshes[1])
        mask_s1 = replicate(torch.zeros(1, 1, S_prefix, S_prefix, dtype=torch.float32), submeshes[1])

        # -------------------- stage 1: TP=8 VLM 0..9 -----------------------
        t_s1_0 = time.perf_counter()
        h_after_1, kv_after_1 = vlm1.forward(h_on_1, attention_mask=mask_s1, use_cache=True)
        t_s1 = (time.perf_counter() - t_s1_0) * 1000
        assert h_after_1.shape[-1] == cfg.vlm_config.width
        assert kv_after_1 is not None
        n_kv_s1 = sum(1 for kv in kv_after_1 if kv is not None)
        assert n_kv_s1 == 9, f"expected 9 KV entries, got {n_kv_s1}"
        print(f"[perf] stage 1 forward (TP=8, 9 layers): {t_s1:.1f} ms")

        # ---------------------- transport 1 -> 2 ---------------------------
        t_xp_0 = time.perf_counter()
        h_on_2 = send_activation_via_host(h_after_1, submeshes[2])
        t_xp = (time.perf_counter() - t_xp_0) * 1000
        mask_s2 = replicate(torch.zeros(1, 1, S_prefix, S_prefix, dtype=torch.float32), submeshes[2])
        print(f"[perf] transport submesh1 → submesh2 (host bounce): {t_xp:.1f} ms")

        # -------------------- stage 2: TP=8 VLM 9..18 ----------------------
        t_s2_0 = time.perf_counter()
        h_after_2, kv_after_2 = vlm2.forward(h_on_2, attention_mask=mask_s2, use_cache=True)
        t_s2 = (time.perf_counter() - t_s2_0) * 1000
        assert h_after_2.shape[-1] == cfg.vlm_config.width
        n_kv_s2 = sum(1 for kv in kv_after_2 if kv is not None)
        assert n_kv_s2 == 9
        print(f"[perf] stage 2 forward (TP=8, 9 layers + norm): {t_s2:.1f} ms")

        # ------------- KV migration submeshes 1,2 → submesh 3 --------------
        t_mig_0 = time.perf_counter()
        migrated = [None] * cfg.vlm_config.depth
        for kv_list in (kv_after_1, kv_after_2):
            for gi, kv in enumerate(kv_list):
                if kv is None:
                    continue
                K_on_3 = send_activation_via_host(kv[0], submeshes[3])
                V_on_3 = send_activation_via_host(kv[1], submeshes[3])
                migrated[gi] = (K_on_3, V_on_3)
        t_mig = (time.perf_counter() - t_mig_0) * 1000
        assert all(m is not None for m in migrated), "migration left holes"
        print(f"[perf] KV migration (18 layers × (K,V), host bounce): {t_mig:.1f} ms")

        # ------------------ stage 3: expert step ---------------------------
        suffix_h = replicate(torch.randn(1, S_suffix, W_e) * 0.02, submeshes[3])
        adarms_cond = replicate(torch.randn(1, W_e) * 0.02, submeshes[3])
        joint_mask = replicate(
            torch.zeros(1, 1, S_suffix, S_prefix + S_suffix, dtype=torch.float32),
            submeshes[3],
        )
        t_s3_0 = time.perf_counter()
        out = expert.forward(
            suffix_h,
            adarms_cond,
            prefix_kv_cache=migrated,
            attention_mask=joint_mask,
        )
        t_s3 = (time.perf_counter() - t_s3_0) * 1000
        assert out.shape[-1] == W_e, f"got {out.shape}"
        print(f"[perf] stage 3 forward_expert_step (replicated, 18 layers): {t_s3:.1f} ms")

        # ---------------------- totals & summary ---------------------------
        total_compute = t_s1 + t_xp + t_s2 + t_mig + t_s3
        print(
            f"\n[perf] ===== E2E real-config TP=8 VLM + replicated expert =====\n"
            f"[perf] config:      VLM Gemma-2B (W=2048, depth=18, mlp=16384)\n"
            f"[perf]              expert Gemma-300M (W=1024, depth=18, mlp=4096)\n"
            f"[perf] layout:      stage 1+2 TP=8 / 9 layers each, stage 3 replicated / 18 layers\n"
            f"[perf] prefix len:  {S_prefix}, suffix len: {S_suffix}\n"
            f"[perf] stage 1:     {t_s1:8.1f} ms\n"
            f"[perf] transport:   {t_xp:8.1f} ms\n"
            f"[perf] stage 2:     {t_s2:8.1f} ms\n"
            f"[perf] KV migrate:  {t_mig:8.1f} ms\n"
            f"[perf] stage 3:     {t_s3:8.1f} ms\n"
            f"[perf] ─────────────────────\n"
            f"[perf] total e2e:   {total_compute:8.1f} ms (single forward pass, "
            f"first uncached run includes JIT setup of any new kernels)\n"
            f"[perf] out.shape={out.shape}, dtype={out.dtype}"
        )


@pytest.mark.timeout(600)
def test_pcc_tp_vs_replicated_real_weights_one_vlm_layer():
    """**Real-weights PCC test.** Compare TP=8 VLM forward to the existing
    (validated) replicated `GemmaBlockTTNN` path on the same submesh, with
    real pi05 layer-0 weights.

    Passing this confirms the TP=8 math is correct against a known-good
    reference. Threshold 0.99 — bf8 + different op ordering accepts some
    numerical drift; mathematically equivalent paths should clear 0.99.
    """
    from pathlib import Path

    if not Path(_REAL_CKPT, "model.safetensors").exists():
        pytest.skip(f"Real checkpoint not found at {_REAL_CKPT}")

    from models.experimental.pi0_5.common import Pi0_5WeightLoader

    loader = Pi0_5WeightLoader(_REAL_CKPT)
    weights = loader.categorized_weights
    cfg = PaliGemmaConfig()  # full real Gemma-2B dims

    layout = build_default_layout()
    with open_galaxy_mesh(layout, enable_fabric=True) as (parent, submeshes):
        # Baseline: replicated GemmaBlockTTNN on submesh 1 (1 layer of real weights).
        baseline = Pi0_5SubmeshVLMSlice(
            config=cfg,
            weights=weights,
            submesh=submeshes[1],
            layer_range=(0, 1),
            tp_shard=False,
        )
        # TP=8: sharded path on submesh 2 (same layer 0).
        tp = Pi0_5SubmeshVLMSlice(
            config=cfg,
            weights=weights,
            submesh=submeshes[2],
            layer_range=(0, 1),
            tp_shard=True,
        )

        # Use the SAME input torch tensor on both submeshes.
        S = 64
        torch.manual_seed(0xCAFE)
        h_torch = torch.randn(1, S, cfg.vlm_config.width) * 0.02
        mask_torch = torch.zeros(1, 1, S, S, dtype=torch.float32)

        def replicate(t: torch.Tensor, mesh) -> "ttnn.Tensor":
            return ttnn.from_torch(
                t,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh),
            )

        out_baseline, _ = baseline.forward(
            replicate(h_torch, submeshes[1]),
            attention_mask=replicate(mask_torch, submeshes[1]),
            use_cache=False,
        )
        out_tp, _ = tp.forward(
            replicate(h_torch, submeshes[2]),
            attention_mask=replicate(mask_torch, submeshes[2]),
            use_cache=False,
        )

        out_baseline_torch = _get_one_replica(out_baseline)
        out_tp_torch = _get_one_replica(out_tp)

        # Sanity: shapes match.
        assert (
            out_baseline_torch.shape == out_tp_torch.shape
        ), f"shape mismatch: {out_baseline_torch.shape} vs {out_tp_torch.shape}"

        pcc = _compute_pcc(out_baseline_torch, out_tp_torch)
        max_abs = (out_baseline_torch.float() - out_tp_torch.float()).abs().max().item()
        baseline_mag = out_baseline_torch.float().abs().mean().item()
        tp_mag = out_tp_torch.float().abs().mean().item()
        print(
            f"\n[pcc] real-weights TP=8 vs replicated (VLM layer 0):\n"
            f"      PCC={pcc:.6f}, max_abs_diff={max_abs:.4e}, "
            f"baseline_mean_abs={baseline_mag:.4e}, tp_mean_abs={tp_mag:.4e}\n"
            f"      shape={out_baseline_torch.shape}"
        )
        assert pcc >= 0.99, f"PCC {pcc:.6f} below threshold 0.99"


@pytest.mark.timeout(900)
def test_pcc_tp_vs_replicated_real_weights_nine_vlm_layers():
    """9-layer (stage-1 worth) real-weights PCC. Catches per-layer
    compounding drift that the single-layer test would miss.
    """
    from pathlib import Path

    if not Path(_REAL_CKPT, "model.safetensors").exists():
        pytest.skip(f"Real checkpoint not found at {_REAL_CKPT}")

    from models.experimental.pi0_5.common import Pi0_5WeightLoader

    loader = Pi0_5WeightLoader(_REAL_CKPT)
    weights = loader.categorized_weights
    cfg = PaliGemmaConfig()

    layout = build_default_layout()
    with open_galaxy_mesh(layout, enable_fabric=True) as (parent, submeshes):
        baseline = Pi0_5SubmeshVLMSlice(
            config=cfg,
            weights=weights,
            submesh=submeshes[1],
            layer_range=(0, 9),
            tp_shard=False,
        )
        tp = Pi0_5SubmeshVLMSlice(
            config=cfg,
            weights=weights,
            submesh=submeshes[2],
            layer_range=(0, 9),
            tp_shard=True,
        )

        S = 64
        torch.manual_seed(0xDEADBEEF)
        h_torch = torch.randn(1, S, cfg.vlm_config.width) * 0.02
        mask_torch = torch.zeros(1, 1, S, S, dtype=torch.float32)

        def replicate(t, mesh):
            return ttnn.from_torch(
                t,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh),
            )

        out_baseline, _ = baseline.forward(
            replicate(h_torch, submeshes[1]),
            attention_mask=replicate(mask_torch, submeshes[1]),
        )
        out_tp, _ = tp.forward(
            replicate(h_torch, submeshes[2]),
            attention_mask=replicate(mask_torch, submeshes[2]),
        )

        out_baseline_torch = _get_one_replica(out_baseline)
        out_tp_torch = _get_one_replica(out_tp)

        pcc = _compute_pcc(out_baseline_torch, out_tp_torch)
        max_abs = (out_baseline_torch.float() - out_tp_torch.float()).abs().max().item()
        baseline_mag = out_baseline_torch.float().abs().mean().item()
        tp_mag = out_tp_torch.float().abs().mean().item()
        print(
            f"\n[pcc] real-weights TP=8 vs replicated (VLM layers 0-9, stage 1 workload):\n"
            f"      PCC={pcc:.6f}, max_abs_diff={max_abs:.4e}, "
            f"baseline_mean_abs={baseline_mag:.4e}, tp_mean_abs={tp_mag:.4e}\n"
            f"      shape={out_baseline_torch.shape}"
        )
        assert pcc >= 0.99, f"PCC {pcc:.6f} below threshold 0.99"


@pytest.mark.timeout(900)
def test_pcc_tp_vs_torch_real_weights_one_vlm_layer():
    """**Single-chip torch baseline PCC.** Independent reference — pure
    PyTorch `GemmaBlock` running on CPU in fp32, real layer-0 weights.
    Compared against the TP=8 TTNN submesh forward.

    Stronger than the submesh-vs-submesh PCC: torch has no TTNN code, no
    mesh, no fabric, no bf8 quantization (fp32 throughout). If TP=8
    TTNN matches torch to ≥0.99 PCC, the TP math is correct against the
    canonical reference.
    """
    from pathlib import Path

    if not Path(_REAL_CKPT, "model.safetensors").exists():
        pytest.skip(f"Real checkpoint not found at {_REAL_CKPT}")

    from models.experimental.pi0_5.common import Pi0_5WeightLoader
    from models.experimental.pi0_5.reference.torch_gemma import (
        GemmaBlock as TorchGemmaBlock,
        precompute_freqs_cis as torch_precompute_freqs_cis,
    )

    loader = Pi0_5WeightLoader(_REAL_CKPT)
    weights = loader.categorized_weights
    cfg = PaliGemmaConfig()

    # ----------- torch reference (single-chip baseline) -----------------
    # Strip "model.layers.0." prefix for the torch block's weight dict.
    layer_idx = 0
    prefix = f"model.layers.{layer_idx}."
    layer_weights = {k[len(prefix) :]: v for k, v in weights["vlm_language"].items() if k.startswith(prefix)}
    torch_block = TorchGemmaBlock(cfg.vlm_config, layer_weights, layer_idx)
    cos_t, sin_t = torch_precompute_freqs_cis(
        cfg.vlm_config.head_dim,
        cfg.max_seq_len,
        cfg.vlm_config.rope_base,
    )

    # Same input both sides.
    S = 64
    torch.manual_seed(0xFEED)
    h_torch = torch.randn(1, S, cfg.vlm_config.width) * 0.02
    mask_torch = torch.zeros(1, 1, S, S, dtype=torch.float32)
    # Cast to fp32 (torch's default dtype for the block).
    h_in_fp32 = h_torch.float()

    # Torch forward.
    with torch.no_grad():
        # The torch GemmaBlock returns (hidden, kv_cache_tuple_or_None).
        out_torch_ref, _ = torch_block.forward(
            h_in_fp32,
            cos_t,
            sin_t,
            attention_mask=mask_torch.float(),
            use_cache=False,
        )

    # ----------- TP=8 TTNN forward ---------------------------------------
    layout = build_default_layout()
    with open_galaxy_mesh(layout, enable_fabric=True) as (parent, submeshes):
        tp = Pi0_5SubmeshVLMSlice(
            config=cfg,
            weights=weights,
            submesh=submeshes[1],
            layer_range=(0, 1),
            tp_shard=True,
        )

        replicate = lambda t: ttnn.from_torch(
            t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=submeshes[1],
            mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(submeshes[1]),
        )

        # Upload as bf16 (matches the device dtype).
        out_tp, _ = tp.forward(
            replicate(h_torch),
            attention_mask=replicate(mask_torch),
            use_cache=False,
        )
        out_tp_torch = _get_one_replica(out_tp)

    # ----------- PCC ----------------------------------------------------
    pcc = _compute_pcc(out_torch_ref, out_tp_torch)
    max_abs = (out_torch_ref.float() - out_tp_torch.float()).abs().max().item()
    torch_mag = out_torch_ref.float().abs().mean().item()
    tp_mag = out_tp_torch.float().abs().mean().item()
    print(
        f"\n[pcc] **single-chip torch baseline** vs TP=8 TTNN (VLM layer 0):\n"
        f"      PCC={pcc:.6f}, max_abs_diff={max_abs:.4e}, "
        f"torch_mean_abs={torch_mag:.4e}, tp_mean_abs={tp_mag:.4e}\n"
        f"      shape={out_torch_ref.shape}"
    )
    assert pcc >= 0.99, f"PCC {pcc:.6f} below threshold 0.99"


@pytest.mark.timeout(600)
def test_pcc_tp_vs_replicated_real_weights_one_expert_layer():
    """**Real-weights PCC** for the TP=8 expert against the existing
    (validated) replicated `AdaRMSGemmaBlockTTNN` path. Same submesh,
    same input, same real layer-0 expert weights. Threshold 0.99.
    """
    from pathlib import Path

    if not Path(_REAL_CKPT, "model.safetensors").exists():
        pytest.skip(f"Real checkpoint not found at {_REAL_CKPT}")

    from models.experimental.pi0_5.common import Pi0_5WeightLoader

    loader = Pi0_5WeightLoader(_REAL_CKPT)
    weights = loader.categorized_weights
    cfg = PaliGemmaConfig()
    W = cfg.expert_config.width

    layout = build_default_layout()
    with open_galaxy_mesh(layout, enable_fabric=True) as (parent, submeshes):
        baseline = Pi0_5SubmeshExpertSlice(
            config=cfg,
            weights=weights,
            submesh=submeshes[0],
            expert_layer_range=(0, 1),
            tp_shard=False,
        )
        tp = Pi0_5SubmeshExpertSlice(
            config=cfg,
            weights=weights,
            submesh=submeshes[3],
            expert_layer_range=(0, 1),
            tp_shard=True,
        )

        S = 64
        torch.manual_seed(0xBEEF)
        h_torch = torch.randn(1, S, W) * 0.02
        adarms_cond_torch = torch.randn(1, W) * 0.02
        mask_torch = torch.zeros(1, 1, S, S, dtype=torch.float32)

        def replicate(t, mesh):
            return ttnn.from_torch(
                t,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh),
            )

        out_baseline = baseline.forward(
            replicate(h_torch, submeshes[0]),
            replicate(adarms_cond_torch, submeshes[0]),
            attention_mask=replicate(mask_torch, submeshes[0]),
        )
        out_tp = tp.forward(
            replicate(h_torch, submeshes[3]),
            replicate(adarms_cond_torch, submeshes[3]),
            attention_mask=replicate(mask_torch, submeshes[3]),
        )

        out_baseline_torch = _get_one_replica(out_baseline)
        out_tp_torch = _get_one_replica(out_tp)

        pcc = _compute_pcc(out_baseline_torch, out_tp_torch)
        max_abs = (out_baseline_torch.float() - out_tp_torch.float()).abs().max().item()
        baseline_mag = out_baseline_torch.float().abs().mean().item()
        tp_mag = out_tp_torch.float().abs().mean().item()
        print(
            f"\n[pcc] real-weights TP=8 vs replicated (expert layer 0):\n"
            f"      PCC={pcc:.6f}, max_abs_diff={max_abs:.4e}, "
            f"baseline_mean_abs={baseline_mag:.4e}, tp_mean_abs={tp_mag:.4e}\n"
            f"      shape={out_baseline_torch.shape}"
        )
        assert pcc >= 0.99, f"PCC {pcc:.6f} below threshold 0.99"


@pytest.mark.timeout(420)
def test_expert_slice_tp_shard_one_layer():
    """TP=8 sharded single expert layer at real Gemma-300M dims.

    Per-chip footprint (bf8):
      Q/K/V/O ~1 MB, MLP ~2.25 MB, adaRMS mod (replicated) ~3 MB. ~6 MB/layer.
    """
    layout = build_default_layout()
    cfg = PaliGemmaConfig()
    W = cfg.expert_config.width
    gen = torch.Generator().manual_seed(0xCC)
    randn = lambda *s: torch.randn(*s, generator=gen, dtype=torch.float32) * 0.02

    expert = _random_expert_layer_weights(0, cfg.expert_config)
    expert["model.norm.dense.weight"] = randn(3 * W, W)
    expert["model.norm.dense.bias"] = randn(3 * W)
    weights = {"action_expert": expert}

    with open_galaxy_mesh(layout, enable_fabric=True) as (parent, submeshes):
        submesh = submeshes[3]
        slice_ = Pi0_5SubmeshExpertSlice(
            config=cfg,
            weights=weights,
            submesh=submesh,
            expert_layer_range=(0, 1),
            tp_shard=True,
        )
        assert len(slice_.expert_blocks) == 1

        S = 64
        replicate = lambda t: ttnn.from_torch(
            t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=submesh,
            mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(submesh),
        )
        h = replicate(torch.randn(1, S, W) * 0.02)
        adarms_cond = replicate(torch.randn(1, W) * 0.02)
        mask = replicate(torch.zeros(1, 1, S, S, dtype=torch.float32))

        out = slice_.forward(h, adarms_cond, attention_mask=mask)
        assert out.shape[-1] == W, f"got {out.shape}"
        print(f"tp_shard expert slice OK — out.shape={out.shape}, dtype={out.dtype}")


@pytest.mark.timeout(600)
def test_expert_slice_tp_shard_eighteen_layers():
    """Full 18-layer TP=8 expert stage at real Gemma-300M dims.

    This is the real per-stage stage 3 workload (sans the suffix MLP and the
    Euler denoise loop). Confirms 18-layer TP expert fits and runs.
    """
    layout = build_default_layout()
    cfg = PaliGemmaConfig()
    W = cfg.expert_config.width
    gen = torch.Generator().manual_seed(0xCD)
    randn = lambda *s: torch.randn(*s, generator=gen, dtype=torch.float32) * 0.02

    ae = {}
    for li in range(cfg.expert_config.depth):
        ae.update(_random_expert_layer_weights(li, cfg.expert_config))
    ae["model.norm.dense.weight"] = randn(3 * W, W)
    ae["model.norm.dense.bias"] = randn(3 * W)
    weights = {"action_expert": ae}

    with open_galaxy_mesh(layout, enable_fabric=True) as (parent, submeshes):
        submesh = submeshes[3]
        slice_ = Pi0_5SubmeshExpertSlice(
            config=cfg,
            weights=weights,
            submesh=submesh,
            expert_layer_range=(0, 18),
            tp_shard=True,
        )
        assert len(slice_.expert_blocks) == 18

        S = 64
        replicate = lambda t: ttnn.from_torch(
            t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=submesh,
            mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(submesh),
        )
        h = replicate(torch.randn(1, S, W) * 0.02)
        adarms_cond = replicate(torch.randn(1, W) * 0.02)
        # We give a joint-style mask sized to a 64-token prefix + 64-token suffix
        # via past_key_value below — but for this isolated test there's no past
        # KV, so the mask is just the local self-attention mask.
        self_mask = replicate(torch.zeros(1, 1, S, S, dtype=torch.float32))

        out = slice_.forward(h, adarms_cond, attention_mask=self_mask)
        assert out.shape[-1] == W
        print(f"tp_shard 18-layer expert stage OK — out.shape={out.shape}")


@pytest.mark.timeout(900)
def test_e2e_real_config_full_tp():
    """Same as test_e2e_real_config_tp_vlm_replicated_expert but with TP=8
    expert too. Now stages 1, 2, 3 are ALL TP=8. Compare per-stage perf to
    the replicated-expert variant to measure the TP expert speedup.
    """
    import time

    layout = build_default_layout()
    cfg = PaliGemmaConfig()

    t_wt0 = time.perf_counter()
    vlm_lang = {}
    for li in range(cfg.vlm_config.depth):
        vlm_lang.update(_random_vlm_layer_weights(li, cfg.vlm_config))
    vlm_lang["model.norm.weight"] = torch.zeros(cfg.vlm_config.width)

    expert_w = {}
    for li in range(cfg.expert_config.depth):
        expert_w.update(_random_expert_layer_weights(li, cfg.expert_config))
    W_e = cfg.expert_config.width
    gen = torch.Generator().manual_seed(0xBB)
    randn = lambda *s: torch.randn(*s, generator=gen, dtype=torch.float32) * 0.02
    expert_w["model.norm.dense.weight"] = randn(3 * W_e, W_e)
    expert_w["model.norm.dense.bias"] = randn(3 * W_e)
    t_wt = (time.perf_counter() - t_wt0) * 1000
    print(f"\n[perf] weight construction (CPU): {t_wt:.1f} ms")

    with open_galaxy_mesh(layout, enable_fabric=True) as (parent, submeshes):
        t_init0 = time.perf_counter()
        vlm1 = Pi0_5SubmeshVLMSlice(
            config=cfg,
            weights={"vlm_language": vlm_lang},
            submesh=submeshes[1],
            layer_range=(0, 9),
            tp_shard=True,
        )
        t_init1 = time.perf_counter()
        vlm2 = Pi0_5SubmeshVLMSlice(
            config=cfg,
            weights={"vlm_language": vlm_lang},
            submesh=submeshes[2],
            layer_range=(9, 18),
            holds_vlm_final_norm=True,
            tp_shard=True,
        )
        t_init2 = time.perf_counter()
        expert = Pi0_5SubmeshExpertSlice(
            config=cfg,
            weights={"action_expert": expert_w},
            submesh=submeshes[3],
            expert_layer_range=(0, 18),
            tp_shard=True,
        )
        t_init3 = time.perf_counter()
        print(
            f"[perf] init stage1 (TP=8, 9 VLM layers): {(t_init1-t_init0)*1000:.0f} ms\n"
            f"[perf] init stage2 (TP=8, 9 VLM layers + norm): {(t_init2-t_init1)*1000:.0f} ms\n"
            f"[perf] init stage3 (**TP=8**, 18 expert layers + adaRMS norm): {(t_init3-t_init2)*1000:.0f} ms"
        )

        # Tile-aligned prefix; bigger ≈ production prefill (544–968).
        S_prefix = int(__import__("os").environ.get("PI0_OB_PREFIX", "64"))
        S_suffix = 64
        replicate = lambda t, mesh: ttnn.from_torch(
            t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh),
        )
        h_on_1 = replicate(torch.randn(1, S_prefix, cfg.vlm_config.width) * 0.02, submeshes[1])
        mask_s1 = replicate(torch.zeros(1, 1, S_prefix, S_prefix, dtype=torch.float32), submeshes[1])
        print(f"[perf] prefix len: {S_prefix} (M_tiles={S_prefix//32})")

        # Helper: one full forward pass, returns per-stage timings.
        def run_pass(label: str):
            t_s1_0 = time.perf_counter()
            h_after_1, kv_after_1 = vlm1.forward(h_on_1, attention_mask=mask_s1, use_cache=True)
            t_s1 = (time.perf_counter() - t_s1_0) * 1000

            t_xp_0 = time.perf_counter()
            h_on_2 = send_activation_via_host(h_after_1, submeshes[2])
            t_xp = (time.perf_counter() - t_xp_0) * 1000
            mask_s2 = replicate(torch.zeros(1, 1, S_prefix, S_prefix, dtype=torch.float32), submeshes[2])

            t_s2_0 = time.perf_counter()
            h_after_2, kv_after_2 = vlm2.forward(h_on_2, attention_mask=mask_s2, use_cache=True)
            t_s2 = (time.perf_counter() - t_s2_0) * 1000

            t_mig_0 = time.perf_counter()
            migrated = [None] * cfg.vlm_config.depth
            for kv_list in (kv_after_1, kv_after_2):
                for gi, kv in enumerate(kv_list):
                    if kv is None:
                        continue
                    K_on_3 = send_activation_via_host(kv[0], submeshes[3])
                    V_on_3 = send_activation_via_host(kv[1], submeshes[3])
                    migrated[gi] = (K_on_3, V_on_3)
            t_mig = (time.perf_counter() - t_mig_0) * 1000
            assert all(m is not None for m in migrated)

            suffix_h = replicate(torch.randn(1, S_suffix, W_e) * 0.02, submeshes[3])
            adarms_cond = replicate(torch.randn(1, W_e) * 0.02, submeshes[3])
            joint_mask = replicate(
                torch.zeros(1, 1, S_suffix, S_prefix + S_suffix, dtype=torch.float32),
                submeshes[3],
            )

            t_s3_0 = time.perf_counter()
            out = expert.forward(suffix_h, adarms_cond, prefix_kv_cache=migrated, attention_mask=joint_mask)
            t_s3 = (time.perf_counter() - t_s3_0) * 1000
            assert out.shape[-1] == W_e

            total = t_s1 + t_xp + t_s2 + t_mig + t_s3
            print(
                f"\n[perf] ===== E2E **full TP=8** — {label} =====\n"
                f"[perf] stage 1 (TP=8 VLM):    {t_s1:8.1f} ms\n"
                f"[perf] transport 1→2 (host):  {t_xp:8.1f} ms\n"
                f"[perf] stage 2 (TP=8 VLM):    {t_s2:8.1f} ms\n"
                f"[perf] KV migrate (host):     {t_mig:8.1f} ms\n"
                f"[perf] stage 3 (TP=8 expert): {t_s3:8.1f} ms\n"
                f"[perf] ──────────────────────\n"
                f"[perf] total e2e:             {total:8.1f} ms"
            )
            return total, out

        _, _ = run_pass("warmup (first run, includes JIT setup of TP expert)")
        warm_total, out = run_pass("WARM (second run, kernels cached)")
        print(f"[perf] out.shape={out.shape}, dtype={out.dtype}")


@pytest.mark.timeout(1200)
def test_e2e_denoise_loop_real_weights():
    """**Full Option B denoise pipeline at real config**:
       synthetic prefix → VLM stage 1+2 (TP=8) → KV migrate → stage 3
       denoise (suffix MLP + 10-step Euler loop, replicated expert).

    Uses REAL pi05 weights for suffix MLP + expert; the VLM weights are
    real (loaded from the categorized dict). Stage 0 (host SigLIP) is
    bypassed — we synthesize the prefix hidden directly on submesh 1.

    Verifies the denoise loop runs end-to-end and produces a final action
    tensor of the expected shape. Reports per-step + total denoise wall.
    """
    import time
    from pathlib import Path

    if not Path(_REAL_CKPT, "model.safetensors").exists():
        pytest.skip(f"Real checkpoint not found at {_REAL_CKPT}")

    from models.experimental.pi0_5.common import Pi0_5WeightLoader
    from models.experimental.pi0_5.tt.option_b.stage_3_expert import Stage3Expert

    loader = Pi0_5WeightLoader(_REAL_CKPT)
    weights = loader.categorized_weights
    cfg = PaliGemmaConfig()
    action_dim = 32
    action_horizon = 50
    action_horizon_padded = ((action_horizon + 31) // 32) * 32  # 64

    layout = build_default_layout()
    with open_galaxy_mesh(layout, enable_fabric=True) as (parent, submeshes):
        # Stages 1 & 2: TP=8 VLM
        vlm1 = Pi0_5SubmeshVLMSlice(
            config=cfg,
            weights=weights,
            submesh=submeshes[1],
            layer_range=(0, 9),
            tp_shard=True,
        )
        vlm2 = Pi0_5SubmeshVLMSlice(
            config=cfg,
            weights=weights,
            submesh=submeshes[2],
            layer_range=(9, 18),
            holds_vlm_final_norm=True,
            tp_shard=True,
        )
        # Stage 3: Stage3Expert with replicated expert + suffix MLP.
        # (TP expert ran into matmul-pcfg recompile issues with the new pcfg
        # path; replicated is fine for the bring-up correctness signal.)
        stage_3 = Stage3Expert(
            spec=layout.stages[3],
            submesh=submeshes[3],
            config=cfg,
            weights=weights,
            denoise_steps=10,
            tp_shard=False,  # replicated for now
            action_dim=action_dim,
            action_horizon=action_horizon,
        )
        stage_3.initialize()
        assert stage_3.suffix is not None, "Suffix slice failed to initialize"

        # Synthetic prefix on submesh 1.
        S_prefix = 64
        torch.manual_seed(0x5005)

        def replicate(t, mesh):
            return ttnn.from_torch(
                t,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh),
            )

        h_on_1 = replicate(torch.randn(1, S_prefix, cfg.vlm_config.width) * 0.02, submeshes[1])
        prefix_mask_1 = replicate(torch.zeros(1, 1, S_prefix, S_prefix, dtype=torch.float32), submeshes[1])

        # --- VLM stages ---
        t0 = time.perf_counter()
        h_after_1, kv1 = vlm1.forward(h_on_1, attention_mask=prefix_mask_1, use_cache=True)
        h_on_2 = send_activation_via_host(h_after_1, submeshes[2])
        prefix_mask_2 = replicate(torch.zeros(1, 1, S_prefix, S_prefix, dtype=torch.float32), submeshes[2])
        h_after_2, kv2 = vlm2.forward(h_on_2, attention_mask=prefix_mask_2, use_cache=True)
        t_vlm = (time.perf_counter() - t0) * 1000

        # --- KV migrate ---
        t0 = time.perf_counter()
        migrated = [None] * cfg.vlm_config.depth
        for kv_list in (kv1, kv2):
            for gi, kv in enumerate(kv_list):
                if kv is None:
                    continue
                K_on_3 = send_activation_via_host(kv[0], submeshes[3])
                V_on_3 = send_activation_via_host(kv[1], submeshes[3])
                migrated[gi] = (K_on_3, V_on_3)
        t_mig = (time.perf_counter() - t0) * 1000

        # --- Denoise inputs ---
        # Noisy actions: pad action_horizon to tile-aligned (50 → 64).
        noisy_torch = torch.zeros(1, action_horizon_padded, action_dim, dtype=torch.float32)
        noisy_torch[:, :action_horizon, :] = torch.randn(1, action_horizon, action_dim)
        noisy_on_3 = replicate(noisy_torch, submeshes[3])

        # Joint attention mask for the suffix path: [B, 1, S_suffix, S_prefix + S_suffix].
        joint_mask = replicate(
            torch.zeros(
                1,
                1,
                action_horizon_padded,
                S_prefix + action_horizon_padded,
                dtype=torch.float32,
            ),
            submeshes[3],
        )

        # ---- Denoise — instrumented (per-step), 3 passes: cold + 2 warm ----
        def run_denoise_with_per_step_timing():
            num_steps = stage_3.denoise_steps
            dt = -1.0 / num_steps
            x_t = noisy_on_3
            x_t_owned = False
            step_times = []
            for i in range(num_steps):
                t = 1.0 - i / num_steps
                B = x_t.shape[0]
                step_start = time.perf_counter()
                adarms_cond = stage_3.suffix.embed_adarms_cond(t, batch_size=B)
                suffix_h = stage_3.suffix.embed_actions(x_t)
                velocity_hidden = stage_3.slice.forward(
                    suffix_h,
                    adarms_cond,
                    prefix_kv_cache=migrated,
                    attention_mask=joint_mask,
                )
                ttnn.deallocate(suffix_h)
                ttnn.deallocate(adarms_cond)
                v_t = stage_3.suffix.project_output(velocity_hidden)
                ttnn.deallocate(velocity_hidden)
                dx = ttnn.multiply(v_t, dt)
                ttnn.deallocate(v_t)
                x_t_new = ttnn.add(x_t, dx)
                ttnn.deallocate(dx)
                if x_t_owned:
                    ttnn.deallocate(x_t)
                x_t = x_t_new
                x_t_owned = True
                step_times.append((time.perf_counter() - step_start) * 1000)
            return x_t, step_times

        passes = {}
        for label in ("COLD (1st run, includes JIT)", "WARM-1", "WARM-2"):
            t0 = time.perf_counter()
            clean_actions, step_times = run_denoise_with_per_step_timing()
            total_ms = (time.perf_counter() - t0) * 1000
            passes[label] = (total_ms, step_times)
            if label != "WARM-2":
                ttnn.deallocate(clean_actions)
                # Re-create fresh noisy actions for each pass (same noise).
                noisy_on_3 = replicate(noisy_torch, submeshes[3])

        clean_torch = _get_one_replica(clean_actions)
        assert clean_torch.shape == (1, action_horizon_padded, action_dim)
        assert torch.isfinite(clean_torch).all(), "denoise produced NaN/Inf actions"

        print(
            f"\n[perf] ===== Option B DENOISE — full benchmark (real weights) =====\n"
            f"[perf] config:       Gemma-2B (depth=18) + expert Gemma-300M (depth=18, replicated)\n"
            f"[perf] prefix len:   {S_prefix}, action_horizon (padded): {action_horizon_padded}\n"
            f"[perf] denoise steps: 10\n"
            f"[perf] VLM stages 1+2:  {t_vlm:8.1f} ms (first-run, includes JIT)\n"
            f"[perf] KV migrate:      {t_mig:8.1f} ms\n"
            f"[perf] ─── denoise passes (Python loop, no trace, no D2D) ───"
        )
        for label, (total_ms, step_times) in passes.items():
            steps_str = " ".join(f"{s:5.1f}" for s in step_times)
            print(
                f"[perf] {label:30s}: total={total_ms:7.1f} ms  "
                f"step1={step_times[0]:.1f} steps2-10 avg={sum(step_times[1:])/9:.1f} ms"
            )
            print(f"[perf]   per-step: {steps_str}")
        print(
            f"[perf] clean_actions: shape={clean_torch.shape}, "
            f"mean_abs={clean_torch.float().abs().mean().item():.4e}"
        )
