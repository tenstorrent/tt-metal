# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Configuration dataclasses for GR00T N1.6-3B model.

Architecture: Eagle-Block2A-2B-v2 backbone (SigLIP2 + Qwen3-1.7B) + AlternateVLDiT action head.

Reference: https://github.com/NVIDIA/Isaac-GR00T
Model card: https://huggingface.co/nvidia/GR00T-N1.6-3B
"""

from dataclasses import dataclass, field


@dataclass
class SigLIP2Config:
    """SigLIP2 vision encoder configuration."""
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    image_size: int = 224
    patch_size: int = 14
    layer_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    num_channels: int = 3

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2  # 756 for 384, or 256 for 224

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads  # 72


@dataclass
class Qwen3Config:
    """Qwen3-1.7B language model configuration (within Eagle backbone)."""
    hidden_size: int = 2048
    intermediate_size: int = 6144
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    vocab_size: int = 151936
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    max_position_embeddings: int = 40960
    head_dim: int = 128
    hidden_act: str = "silu"
    tie_word_embeddings: bool = True

    # GR00T uses intermediate features from this layer
    select_layer: int = 16


@dataclass
class EagleBackboneConfig:
    """Eagle-Block2A-2B-v2 VLM backbone configuration."""
    vision: SigLIP2Config = field(default_factory=SigLIP2Config)
    language: Qwen3Config = field(default_factory=Qwen3Config)

    # Pixel shuffle downsampling ratio (2x2 -> 1, so 256 -> 64 tokens per frame)
    pixel_shuffle_ratio: float = 0.5

    # MLP connector: vision_dim -> language_dim
    connector_hidden_size: int = 2048  # matches language hidden_size

    # Number of top LLM layers to tune (rest are frozen in training, all used in inference)
    tune_top_llm_layers: int = 4

    @property
    def num_image_tokens_per_frame(self) -> int:
        """After pixel shuffle: num_patches * ratio^2"""
        ratio_sq = self.pixel_shuffle_ratio ** 2  # 0.25
        return int(self.vision.num_patches * ratio_sq)  # 256 * 0.25 = 64


@dataclass
class DiTConfig:
    """AlternateVLDiT (Diffusion Transformer) action head configuration."""
    num_layers: int = 32
    num_attention_heads: int = 32
    attention_head_dim: int = 48
    norm_type: str = "ada_norm"
    dropout: float = 0.2
    final_dropout: bool = True
    output_dim: int = 1024
    cross_attention_dim: int = 2048  # backbone embedding dim

    # AlternateVLDiT specific
    attend_text_every_n_blocks: int = 2
    use_vlln: bool = True  # LayerNorm on backbone features

    @property
    def inner_dim(self) -> int:
        return self.num_attention_heads * self.attention_head_dim  # 32 * 48 = 1536


@dataclass
class EmbodimentConfig:
    """Per-embodiment MLP configuration."""
    max_num_embodiments: int = 32
    max_state_dim: int = 128
    max_action_dim: int = 128
    state_hidden_dim: int = 1024
    state_output_dim: int = 1536  # matches DiT inner_dim
    action_hidden_dim: int = 1536
    action_output_dim: int = 1536


@dataclass
class Gr00tN16Config:
    """Top-level GR00T N1.6-3B model configuration."""
    backbone: EagleBackboneConfig = field(default_factory=EagleBackboneConfig)
    dit: DiTConfig = field(default_factory=DiTConfig)
    embodiment: EmbodimentConfig = field(default_factory=EmbodimentConfig)

    # Flow matching inference
    num_inference_timesteps: int = 4
    num_timestep_buckets: int = 1000
    action_horizon: int = 50

    # Embedding dimensions
    hidden_size: int = 1024  # DiT output dim
    input_embedding_dim: int = 1536  # DiT inner_dim
    backbone_embedding_dim: int = 2048  # from Qwen3

    # Max sequence length for backbone
    max_seq_len: int = 1024

    # State dropout (0.0 for inference)
    state_dropout_prob: float = 0.0

    # Add positional embeddings to action tokens
    add_pos_embed: bool = True

    # HuggingFace model ID
    hf_model_id: str = "nvidia/GR00T-N1.6-3B"

    @classmethod
    def default(cls) -> "Gr00tN16Config":
        return cls()
