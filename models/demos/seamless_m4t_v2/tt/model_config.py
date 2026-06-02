# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Configuration for the SeamlessM4Tv2 S2TT TT port.

`SeamlessS2TTConfig` reads the HuggingFace `SeamlessM4Tv2Config` (config.json only,
no weights) and surfaces the subset of fields the TT speech encoder + text decoder
need. TT-specific knobs (dtypes, conv fallback, core grids) live here too so the
device modules stay free of magic numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_MODEL_ID = "facebook/seamless-m4t-v2-large"


@dataclass
class SeamlessS2TTConfig:
    # --- speech (conformer) encoder ---
    hidden_size: int = 1024
    speech_encoder_layers: int = 24
    speech_encoder_attention_heads: int = 16
    speech_encoder_intermediate_size: int = 4096
    speech_encoder_hidden_act: str = "swish"  # SiLU
    conv_depthwise_kernel_size: int = 31
    left_max_position_embeddings: int = 64
    right_max_position_embeddings: int = 8
    position_embeddings_type: str = "relative_key"
    feature_projection_input_dim: int = 160
    speech_encoder_chunk_size: int = 20000

    # --- adapter (stride-8 downsample) ---
    add_adapter: bool = True
    num_adapter_layers: int = 1
    adaptor_kernel_size: int = 8
    adaptor_stride: int = 8

    # --- text decoder ---
    decoder_layers: int = 24
    decoder_attention_heads: int = 16
    decoder_ffn_dim: int = 8192
    decoder_activation: str = "relu"
    vocab_size: int = 256102
    max_position_embeddings: int = 4096
    scale_embedding: bool = True

    # --- tokens ---
    pad_token_id: int = 0
    bos_token_id: int = 2
    eos_token_id: int = 3
    decoder_start_token_id: int = 3

    layer_norm_eps: float = 1e-5

    # --- TT-specific knobs ---
    model_id: str = DEFAULT_MODEL_ID
    # Phase 1/3: run depthwise (k=31) and adapter stride-8 convs on host until
    # Phase 7 hardens the on-device path. Toggle off to exercise ttnn.conv1d.
    conv_cpu_fallback: bool = True

    # derived
    head_dim: int = field(init=False)
    num_distance_positions: int = field(init=False)

    def __post_init__(self):
        self.head_dim = self.hidden_size // self.speech_encoder_attention_heads
        self.num_distance_positions = (
            self.left_max_position_embeddings + self.right_max_position_embeddings + 1
        )

    @classmethod
    def from_hf(cls, model_id: str = DEFAULT_MODEL_ID, **overrides) -> "SeamlessS2TTConfig":
        """Build from the HF config.json (downloads config only, not weights)."""
        from transformers import AutoConfig

        hf = AutoConfig.from_pretrained(model_id)
        fields = {
            "hidden_size",
            "speech_encoder_layers",
            "speech_encoder_attention_heads",
            "speech_encoder_intermediate_size",
            "speech_encoder_hidden_act",
            "conv_depthwise_kernel_size",
            "left_max_position_embeddings",
            "right_max_position_embeddings",
            "position_embeddings_type",
            "feature_projection_input_dim",
            "speech_encoder_chunk_size",
            "add_adapter",
            "num_adapter_layers",
            "adaptor_kernel_size",
            "adaptor_stride",
            "decoder_layers",
            "decoder_attention_heads",
            "decoder_ffn_dim",
            "vocab_size",
            "max_position_embeddings",
            "scale_embedding",
            "pad_token_id",
            "bos_token_id",
            "eos_token_id",
            "decoder_start_token_id",
            "layer_norm_eps",
        }
        kwargs = {k: getattr(hf, k) for k in fields if hasattr(hf, k)}
        kwargs["decoder_activation"] = getattr(hf, "activation_function", "relu")
        kwargs["model_id"] = model_id
        kwargs.update(overrides)
        return cls(**kwargs)
