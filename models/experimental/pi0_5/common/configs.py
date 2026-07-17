# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Common configurations for PI0 model components.

This module contains all dataclass configs used across reference and TTNN implementations.
"""

from dataclasses import dataclass, field


@dataclass
class GemmaConfig:
    """Configuration for Gemma transformer."""

    width: int = 2048
    depth: int = 18
    mlp_dim: int = 16384
    num_heads: int = 8
    num_kv_heads: int = 1
    head_dim: int = 256
    rms_norm_eps: float = 1e-6
    rope_base: float = 10000.0

    @classmethod
    def gemma_2b(cls) -> "GemmaConfig":
        """Gemma 2B configuration (VLM backbone)."""
        return cls(
            width=2048,
            depth=18,
            mlp_dim=16384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )

    @classmethod
    def gemma_300m(cls) -> "GemmaConfig":
        """Pi0.5 action expert: Gemma-300M-sized (18L, 1024H, 4096 MLP) but with
        head_dim=256 to MATCH the VLM, not the stock Gemma-300M head_dim=64.
        Required because pi0.5 cross-attention reuses the VLM's KV cache;
        expert Q heads must share head_dim with VLM K/V."""
        return cls(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )


@dataclass
class SigLIPConfig:
    """Configuration for SigLIP vision encoder."""

    hidden_size: int = 1152
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    image_size: int = 224
    patch_size: int = 14
    num_channels: int = 3
    intermediate_size: int = 4304
    layer_norm_eps: float = 1e-6

    def __post_init__(self):
        assert self.hidden_size % self.num_attention_heads == 0, (
            f"SigLIP hidden_size={self.hidden_size} not divisible by " f"num_attention_heads={self.num_attention_heads}"
        )

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


@dataclass
class SuffixConfig:
    """Configuration for suffix embedding."""

    action_dim: int = 32
    action_horizon: int = 50
    expert_width: int = 1024
    state_dim: int = 32  # padded for tile alignment; real state is 8-dim (see norm_stats.json)
    time_emb_dim: int = 1024  # Time embedding dimension
    pi05: bool = False  # PI05 uses different time handling


@dataclass
class PrefixConfig:
    """Configuration for prefix embedding."""

    vlm_hidden_size: int = 2048
    num_image_tokens: int = 256  # Tokens per image from SigLIP


@dataclass
class PaliGemmaConfig:
    """Configuration for PaliGemma backbone."""

    vlm_config: GemmaConfig = None
    expert_config: GemmaConfig = None
    siglip_config: SigLIPConfig = None
    max_seq_len: int = 2048

    def __post_init__(self):
        if self.vlm_config is None:
            self.vlm_config = GemmaConfig.gemma_2b()
        if self.expert_config is None:
            self.expert_config = GemmaConfig.gemma_300m()
        if self.siglip_config is None:
            self.siglip_config = SigLIPConfig()
        assert self.expert_config.head_dim == self.vlm_config.head_dim, (
            f"expert head_dim={self.expert_config.head_dim} must match "
            f"vlm head_dim={self.vlm_config.head_dim} (cross-attention KV-cache sharing)"
        )


@dataclass
class DenoiseConfig:
    """Configuration for denoising."""

    num_steps: int = 10
    noise_scale: float = 1.0
    action_dim: int = 32
    action_horizon: int = 50


@dataclass
class PI0ModelConfig:
    """Complete configuration for PI0 model."""

    # Core dimensions
    action_dim: int = 32
    action_horizon: int = 50
    state_dim: int = 32

    # Model variants
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"

    # Processing
    precision: str = "bfloat16"
    num_denoising_steps: int = 10
    max_seq_len: int = 2048

    # PI05 mode (uses adaRMS instead of fused action-time)
    pi05: bool = False

    # Component configs (auto-populated)
    vlm_config: GemmaConfig = field(default_factory=GemmaConfig.gemma_2b)
    expert_config: GemmaConfig = field(default_factory=GemmaConfig.gemma_300m)
    siglip_config: SigLIPConfig = field(default_factory=SigLIPConfig)

    def __post_init__(self):
        self.vlm_config = GemmaConfig.gemma_2b()
        self.expert_config = GemmaConfig.gemma_300m()
        self.siglip_config = SigLIPConfig()
        assert self.expert_config.head_dim == self.vlm_config.head_dim, (
            f"expert head_dim={self.expert_config.head_dim} must match "
            f"vlm head_dim={self.vlm_config.head_dim} (cross-attention KV-cache sharing)"
        )


# ============================================================================
# pi0.5-specific additions (subclasses, overrides)
# ============================================================================

from dataclasses import dataclass, field


@dataclass
class Pi0_5ModelConfig(PI0ModelConfig):
    """
    PI0.5 model configuration.

    Differences vs PI0:
      - pi05 = True (drives adaRMS path in suffix + expert)
      - max_token_len = 200 (pi0.5 default in openpi)
      - discrete_state_input = True (state encoded as language tokens)
    """

    pi05: bool = True
    max_token_len: int = 200
    discrete_state_input: bool = True

    vlm_config: GemmaConfig = field(default_factory=GemmaConfig.gemma_2b)
    expert_config: GemmaConfig = field(default_factory=GemmaConfig.gemma_300m)
    siglip_config: SigLIPConfig = field(default_factory=SigLIPConfig)

    def __post_init__(self):
        super().__post_init__()

    @classmethod
    def from_checkpoint(cls, ckpt_dir, **overrides) -> "Pi0_5ModelConfig":
        """Construct config with action_horizon auto-read from <ckpt_dir>/config.json.

        Use this in any test or production path that loads weights from a
        specific checkpoint. The pi05_libero upstream checkpoint trains at
        action_horizon=10, the lerobot finetune at 50 — instantiating with
        the default silently degrades PCC (see
        [[feedback_action_horizon_from_config]]).

        Any keyword in `overrides` wins over the auto-detected value.
        """
        from pathlib import Path

        from models.experimental.pi0_5.common.checkpoint_meta import action_horizon_from_checkpoint

        default_ah = cls.__dataclass_fields__["action_horizon"].default
        ah = action_horizon_from_checkpoint(Path(ckpt_dir), default=default_ah)
        overrides.setdefault("action_horizon", ah)
        return cls(**overrides)
