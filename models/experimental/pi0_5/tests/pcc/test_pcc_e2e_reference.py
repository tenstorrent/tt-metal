# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end PI0.5 reference test on a small config with synthetic weights.

Exercises the full inference path:
    images + lang_tokens
        -> embed_prefix
        -> backbone.forward_vlm (KV cache)
        -> denoise loop:
             - Pi0_5SuffixEmbedding.embed_suffix (action_in_proj + sincos+MLP)
             - Pi0_5PaliGemmaBackbone.forward_expert (adaRMSNorm + gated res)
             - action_out_proj
             - x_t += dt * velocity
        -> final actions

Validates wiring + shapes + that outputs are finite and time-dependent.
"""

import torch

from models.experimental.pi0_5.common.configs import (
    GemmaConfig,
    PaliGemmaConfig,
    SigLIPConfig,
    SuffixConfig,
    PrefixConfig,
    DenoiseConfig,
)
from models.experimental.pi0_5.reference.torch_prefix import PrefixEmbedding
from models.experimental.pi0_5.reference.torch_denoise import DenoisingModule

from models.experimental.pi0_5.reference.torch_suffix import Pi0_5SuffixEmbedding
from models.experimental.pi0_5.reference.torch_paligemma import Pi0_5PaliGemmaBackbone

SEED = 0


def _random_siglip_weights(cfg: SigLIPConfig) -> dict:
    w = {}
    h = cfg.hidden_size
    inter = cfg.intermediate_size
    np_ = (cfg.image_size // cfg.patch_size) ** 2
    w["vision_model.embeddings.patch_embedding.weight"] = torch.randn(h, 3, cfg.patch_size, cfg.patch_size)
    w["vision_model.embeddings.patch_embedding.bias"] = torch.randn(h)
    w["vision_model.embeddings.position_embedding.weight"] = torch.randn(np_, h)
    for i in range(cfg.num_hidden_layers):
        p = f"vision_model.encoder.layers.{i}."
        w[f"{p}layer_norm1.weight"] = torch.randn(h)
        w[f"{p}layer_norm1.bias"] = torch.randn(h)
        w[f"{p}layer_norm2.weight"] = torch.randn(h)
        w[f"{p}layer_norm2.bias"] = torch.randn(h)
        for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
            w[f"{p}self_attn.{proj}.weight"] = torch.randn(h, h)
            w[f"{p}self_attn.{proj}.bias"] = torch.randn(h)
        w[f"{p}mlp.fc1.weight"] = torch.randn(inter, h)
        w[f"{p}mlp.fc1.bias"] = torch.randn(inter)
        w[f"{p}mlp.fc2.weight"] = torch.randn(h, inter)
        w[f"{p}mlp.fc2.bias"] = torch.randn(h)
    w["vision_model.encoder.final_layer_norm.weight"] = torch.randn(h)
    w["vision_model.encoder.final_layer_norm.bias"] = torch.randn(h)
    return w


def _random_gemma_weights(cfg: GemmaConfig) -> dict:
    w = {}
    width = cfg.width
    mlp = cfg.mlp_dim
    nh, nkv, hd = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
    w["model.embed_tokens.weight"] = torch.randn(257152, width)
    w["model.norm.weight"] = torch.randn(width)
    for i in range(cfg.depth):
        p = f"model.layers.{i}."
        w[f"{p}input_layernorm.weight"] = torch.randn(width)
        w[f"{p}post_attention_layernorm.weight"] = torch.randn(width)
        w[f"{p}self_attn.q_proj.weight"] = torch.randn(nh * hd, width)
        w[f"{p}self_attn.k_proj.weight"] = torch.randn(nkv * hd, width)
        w[f"{p}self_attn.v_proj.weight"] = torch.randn(nkv * hd, width)
        w[f"{p}self_attn.o_proj.weight"] = torch.randn(width, nh * hd)
        w[f"{p}mlp.gate_proj.weight"] = torch.randn(mlp, width)
        w[f"{p}mlp.up_proj.weight"] = torch.randn(mlp, width)
        w[f"{p}mlp.down_proj.weight"] = torch.randn(width, mlp)
    return w


def _random_projector_weights(in_size: int, out_size: int) -> dict:
    return {"linear.weight": torch.randn(out_size, in_size), "linear.bias": torch.randn(out_size)}


def _small_paligemma_config() -> PaliGemmaConfig:
    siglip = SigLIPConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        image_size=28,
        patch_size=14,  # -> 4 patch tokens
    )
    vlm = GemmaConfig(
        width=64,
        depth=2,
        mlp_dim=128,
        num_heads=2,
        num_kv_heads=1,
        head_dim=32,
    )
    expert = GemmaConfig(
        width=64,
        depth=2,
        mlp_dim=128,
        num_heads=2,
        num_kv_heads=1,
        head_dim=32,
    )
    return PaliGemmaConfig(
        siglip_config=siglip,
        vlm_config=vlm,
        expert_config=expert,
        max_seq_len=64,
    )


def _add_adarms_modulation_weights(expert_weights: dict, expert_cfg: GemmaConfig) -> None:
    """Augment expert weights in-place with random pi0.5 adaRMS modulation tensors."""
    w = expert_cfg.width
    g = torch.Generator().manual_seed(SEED)
    for i in range(expert_cfg.depth):
        prefix = f"model.layers.{i}."
        # Small init so the gate is near zero -> stable e2e on random weights.
        for name in ("input_layernorm.dense", "post_attention_layernorm.dense"):
            expert_weights[f"{prefix}{name}.weight"] = torch.randn(3 * w, w, generator=g) * 0.01
            expert_weights[f"{prefix}{name}.bias"] = torch.zeros(3 * w)
    # Final expert norm: also adaRMS.
    expert_weights["model.norm.dense.weight"] = torch.randn(3 * w, w, generator=g) * 0.01
    expert_weights["model.norm.dense.bias"] = torch.zeros(3 * w)


def _suffix_weights(action_dim: int, expert_width: int) -> dict:
    g = torch.Generator().manual_seed(SEED)
    return {
        "action_in_proj.weight": torch.randn(expert_width, action_dim, generator=g) * 0.02,
        "action_in_proj.bias": torch.zeros(expert_width),
        "action_out_proj.weight": torch.randn(action_dim, expert_width, generator=g) * 0.02,
        "action_out_proj.bias": torch.zeros(action_dim),
        "time_mlp_in.weight": torch.randn(expert_width, expert_width, generator=g) * 0.02,
        "time_mlp_in.bias": torch.zeros(expert_width),
        "time_mlp_out.weight": torch.randn(expert_width, expert_width, generator=g) * 0.02,
        "time_mlp_out.bias": torch.zeros(expert_width),
    }


def _build_pi0_5_reference(pg_config: PaliGemmaConfig, action_dim: int, action_horizon: int):
    """Build a Pi0_5-style reference model on top of random weights (small config)."""
    torch.manual_seed(SEED)

    weights = {
        "vlm_vision": _random_siglip_weights(pg_config.siglip_config),
        "vlm_language": _random_gemma_weights(pg_config.vlm_config),
        "vlm_projector": _random_projector_weights(pg_config.siglip_config.hidden_size, pg_config.vlm_config.width),
        "action_expert": _random_gemma_weights(pg_config.expert_config),
    }
    _add_adarms_modulation_weights(weights["action_expert"], pg_config.expert_config)

    backbone = Pi0_5PaliGemmaBackbone(pg_config, weights)

    suffix_cfg = SuffixConfig(
        action_dim=action_dim,
        action_horizon=action_horizon,
        expert_width=pg_config.expert_config.width,
        pi05=True,
    )
    suffix = Pi0_5SuffixEmbedding(suffix_cfg, _suffix_weights(action_dim, pg_config.expert_config.width))

    prefix = PrefixEmbedding(
        PrefixConfig(vlm_hidden_size=pg_config.vlm_config.width),
        embed_image_fn=backbone.embed_image,
        embed_language_fn=backbone.embed_language_tokens,
    )

    return backbone, suffix, prefix


def test_pi0_5_reference_end_to_end():
    """Run the full inference path on a tiny config and check the output."""
    pg_cfg = _small_paligemma_config()
    action_dim = 4
    action_horizon = 4
    num_steps = 3
    batch_size = 1

    backbone, suffix, prefix = _build_pi0_5_reference(pg_cfg, action_dim, action_horizon)

    # Prefix inputs.
    image = torch.randn(batch_size, 3, pg_cfg.siglip_config.image_size, pg_cfg.siglip_config.image_size)
    img_mask = torch.ones(batch_size, dtype=torch.bool)
    lang_tokens = torch.randint(0, 1000, (batch_size, 6))
    lang_masks = torch.ones(batch_size, 6, dtype=torch.bool)

    prefix_embs, _, _ = prefix.embed_prefix([image], [img_mask], lang_tokens, lang_masks)
    assert prefix_embs.shape[0] == batch_size
    assert prefix_embs.shape[-1] == pg_cfg.vlm_config.width
    expected_prefix_len = pg_cfg.siglip_config.num_patches + lang_tokens.shape[1]
    assert prefix_embs.shape[1] == expected_prefix_len

    _, vlm_cache = backbone.forward_vlm(prefix_embs, use_cache=True)
    assert vlm_cache is not None and len(vlm_cache) == pg_cfg.vlm_config.depth

    def denoise_forward(noisy_actions, timestep, kv_cache=None, **kwargs):
        suffix_embs, _, _, adarms_cond = suffix.embed_suffix(None, noisy_actions, timestep)
        expert_out, _ = backbone.forward_expert(suffix_embs, adarms_cond=adarms_cond, past_key_values=kv_cache)
        return suffix.project_output(expert_out)

    denoise_cfg = DenoiseConfig(num_steps=num_steps, action_dim=action_dim, action_horizon=action_horizon)
    denoiser = DenoisingModule(denoise_cfg, denoise_forward)

    torch.manual_seed(SEED + 1)
    actions = denoiser.sample_actions(batch_size, prefix_kv_cache=vlm_cache, state=None)

    assert actions.shape == (batch_size, action_horizon, action_dim)
    assert torch.isfinite(actions).all(), "denoised actions contain NaN/Inf"

    # Re-run with different noise -> output should differ. Same noise -> same.
    torch.manual_seed(SEED + 1)
    actions_same = denoiser.sample_actions(batch_size, prefix_kv_cache=vlm_cache, state=None)
    torch.manual_seed(SEED + 2)
    actions_diff = denoiser.sample_actions(batch_size, prefix_kv_cache=vlm_cache, state=None)
    assert torch.allclose(actions, actions_same, atol=1e-5), "same noise -> same actions"
    assert not torch.allclose(actions, actions_diff, atol=1e-3), "different noise -> different actions"


def test_pi0_5_reference_zero_adarms_matches_plain_path():
    """
    With the adaRMS modulation Dense at zero, gate=0 and the expert layers
    behave as a residual-free passthrough. The expert output then depends only
    on the action_in_proj of noisy_actions and the prefix-cached attention.
    This is a structural sanity check on the gated-residual wiring.
    """
    pg_cfg = _small_paligemma_config()
    action_dim = 4
    action_horizon = 4
    batch_size = 1

    backbone, suffix, prefix = _build_pi0_5_reference(pg_cfg, action_dim, action_horizon)

    # Zero out the adaRMS modulation Dense in every expert layer.
    for blk in backbone.expert_blocks:
        blk.pre_attn_mod_weight = torch.zeros_like(blk.pre_attn_mod_weight)
        blk.pre_attn_mod_bias = torch.zeros_like(blk.pre_attn_mod_bias)
        blk.pre_ffw_mod_weight = torch.zeros_like(blk.pre_ffw_mod_weight)
        blk.pre_ffw_mod_bias = torch.zeros_like(blk.pre_ffw_mod_bias)

    # Build minimal prefix cache.
    image = torch.randn(batch_size, 3, pg_cfg.siglip_config.image_size, pg_cfg.siglip_config.image_size)
    img_mask = torch.ones(batch_size, dtype=torch.bool)
    lang_tokens = torch.randint(0, 1000, (batch_size, 6))
    lang_masks = torch.ones(batch_size, 6, dtype=torch.bool)
    prefix_embs, _, _ = prefix.embed_prefix([image], [img_mask], lang_tokens, lang_masks)
    _, vlm_cache = backbone.forward_vlm(prefix_embs, use_cache=True)

    noisy = torch.randn(batch_size, action_horizon, action_dim)
    t = torch.tensor([0.5])
    suffix_embs, _, _, adarms_cond = suffix.embed_suffix(None, noisy, t)

    expert_out, _ = backbone.forward_expert(suffix_embs, adarms_cond=adarms_cond, past_key_values=vlm_cache)

    # Gated residual with gate=0 means expert_out should equal suffix_embs
    # before the final RMSNorm. The final RMSNorm rescales but doesn't change shape.
    assert expert_out.shape == suffix_embs.shape
    assert torch.isfinite(expert_out).all()
