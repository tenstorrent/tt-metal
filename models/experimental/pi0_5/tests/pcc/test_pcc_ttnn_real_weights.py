# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
PI0.5 TTNN smoke tests on the real lerobot/pi05_base checkpoint.

These tests focus on the *new* pi0.5 code paths (suffix → adaRMS action
expert → action projection) without depending on the shared VLM / SigLIP
prefix path. A synthetic prefix KV cache stands in for the VLM output, so
we can validate the new modules end-to-end on a Blackhole device.

Skipped if the checkpoint isn't present locally.
"""

from pathlib import Path

import pytest
import torch
import ttnn

CHECKPOINT_DIR = Path(__file__).resolve().parents[2] / "weights" / "pi05_base"

pytestmark = pytest.mark.skipif(
    not (CHECKPOINT_DIR / "model.safetensors").exists(),
    reason=f"pi0.5 checkpoint not found at {CHECKPOINT_DIR}",
)


def _build_ttnn_model(device):
    from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig
    from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader
    from models.experimental.pi0_5.tt.ttnn_pi0_5_model import Pi0_5ModelTTNN

    loader = Pi0_5WeightLoader(str(CHECKPOINT_DIR))
    cfg = Pi0_5ModelConfig(num_denoising_steps=2)
    return cfg, Pi0_5ModelTTNN(cfg, loader, device)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_pi0_5_ttnn_model_constructs(device):
    """Loading 14GB of real pi0.5 weights and building all TTNN modules succeeds."""
    cfg, model = _build_ttnn_model(device)
    assert model.config.pi05 is True
    assert len(model.backbone.expert_blocks) == cfg.expert_config.depth
    # adaRMS modulation tensors (fused 6*W Dense per block) were loaded.
    for blk in model.backbone.expert_blocks:
        assert blk.mod_weight is not None
    # Final expert norm is adaRMS.
    assert model.backbone.expert_final_norm_mod_weight is not None


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_pi0_5_ttnn_suffix_runs_on_device(device):
    """Suffix embedding (sincos -> MLP -> adarms_cond) runs and produces the
    right shapes on the real weights."""
    cfg, model = _build_ttnn_model(device)

    batch_size = 1
    noisy_actions = torch.randn(batch_size, cfg.action_horizon, cfg.action_dim)
    timestep = torch.tensor([0.5])

    noisy_ttnn = ttnn.from_torch(
        noisy_actions,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    t_ttnn = ttnn.from_torch(
        timestep,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    suffix_embs, _, _, adarms_cond = model.embed_suffix(None, noisy_ttnn, t_ttnn)

    s_torch = ttnn.to_torch(suffix_embs)
    a_torch = ttnn.to_torch(adarms_cond)

    # Shape checks (TTNN may pad to tile boundaries, so check ranges).
    assert s_torch.shape[0] == batch_size
    assert s_torch.shape[1] >= cfg.action_horizon
    assert s_torch.shape[2] >= cfg.expert_config.width
    assert torch.isfinite(s_torch).all(), "suffix_embs contain NaN/Inf"

    assert a_torch.shape[0] == batch_size
    assert a_torch.shape[-1] >= cfg.expert_config.width
    assert torch.isfinite(a_torch).all(), "adarms_cond contains NaN/Inf"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_pi0_5_ttnn_expert_runs_with_synthetic_prefix(device):
    """
    Full pi0.5 action-expert path (adaRMS norms + gated residuals + final
    adaRMS) on the real weights, with a synthetic prefix KV cache.

    This is the most important pi0.5-specific validation: it proves the
    AdaRMSGemmaBlockTTNN, the modulation tensor loading, and the final
    adaRMS norm all execute correctly on device.
    """
    cfg, model = _build_ttnn_model(device)
    expert_cfg = cfg.expert_config

    batch_size = 1
    prefix_len = 32  # tile-aligned
    noisy_actions = torch.randn(batch_size, cfg.action_horizon, cfg.action_dim)
    timestep = torch.tensor([0.5])

    noisy_ttnn = ttnn.from_torch(
        noisy_actions,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    t_ttnn = ttnn.from_torch(
        timestep,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    suffix_embs, _, _, adarms_cond = model.embed_suffix(None, noisy_ttnn, t_ttnn)

    # Synthetic prefix KV cache: one (K, V) pair per expert layer.
    # Shape: (batch, num_kv_heads, prefix_len, head_dim)
    prefix_kv_cache = []
    for _ in range(expert_cfg.depth):
        k = torch.randn(batch_size, expert_cfg.num_kv_heads, prefix_len, expert_cfg.head_dim) * 0.1
        v = torch.randn(batch_size, expert_cfg.num_kv_heads, prefix_len, expert_cfg.head_dim) * 0.1
        k_t = ttnn.from_torch(k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        v_t = ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        prefix_kv_cache.append((k_t, v_t))

    expert_output, _ = model.backbone.forward_expert(
        suffix_embs,
        adarms_cond=adarms_cond,
        past_key_values=prefix_kv_cache,
    )

    velocity = model.suffix_embedding.project_output(expert_output)
    v_torch = ttnn.to_torch(velocity)
    v_torch = v_torch[:, : cfg.action_horizon, : cfg.action_dim]

    assert v_torch.shape == (batch_size, cfg.action_horizon, cfg.action_dim)
    assert torch.isfinite(v_torch).all(), "velocity contains NaN/Inf"
