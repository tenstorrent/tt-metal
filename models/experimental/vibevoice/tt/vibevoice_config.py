# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""VibeVoice-1.5B architecture configuration parsed from config.json."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class DecoderConfig:
    hidden_size: int = 1536
    num_hidden_layers: int = 28
    num_attention_heads: int = 12
    num_key_value_heads: int = 2
    intermediate_size: int = 8960
    vocab_size: int = 151936
    rope_theta: float = 1_000_000.0
    rms_norm_eps: float = 1e-06
    max_position_embeddings: int = 32768
    head_dim: int = 128  # hidden_size // num_attention_heads = 1536 // 12 = 128


@dataclass
class DiffusionHeadConfig:
    hidden_size: int = 1536
    head_layers: int = 4
    head_ffn_ratio: float = 3.0
    rms_norm_eps: float = 1e-05
    latent_size: int = 64


@dataclass
class TokenizerConfig:
    vae_dim: int = 64
    causal: bool = True
    encoder_n_filters: int = 32
    encoder_ratios: List[int] = field(default_factory=lambda: [8, 5, 5, 4, 2, 2])
    encoder_depths: str = "3-3-3-3-3-3-8"
    decoder_ratios: Optional[List[int]] = None
    decoder_n_filters: int = 32
    layernorm: str = "RMSNorm"
    layernorm_eps: float = 1e-05
    conv_bias: bool = True
    mixer_layer: str = "depthwise_conv"


@dataclass
class SemanticTokenizerConfig:
    vae_dim: int = 128
    causal: bool = True
    encoder_n_filters: int = 32
    encoder_ratios: List[int] = field(default_factory=lambda: [8, 5, 5, 4, 2, 2])
    encoder_depths: str = "3-3-3-3-3-3-8"
    layernorm: str = "RMSNorm"
    layernorm_eps: float = 1e-05
    conv_bias: bool = True
    mixer_layer: str = "depthwise_conv"


@dataclass
class VibeVoiceModelConfig:
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    diffusion_head: DiffusionHeadConfig = field(default_factory=DiffusionHeadConfig)
    acoustic_tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    semantic_tokenizer: SemanticTokenizerConfig = field(default_factory=SemanticTokenizerConfig)
    # Connector dims derived from tokenizer vae_dims → LM hidden_size
    acoustic_connector_input_dim: int = 64
    semantic_connector_input_dim: int = 128
    connector_output_dim: int = 1536  # = decoder.hidden_size


def load_vibevoice_model_config(model_path: str) -> VibeVoiceModelConfig:
    """Parse VibeVoice config.json into structured dataclasses."""
    cfg_path = Path(model_path) / "config.json"
    if not cfg_path.exists():
        # Return defaults for the 1.5B variant when no config.json is present
        return VibeVoiceModelConfig()

    with open(cfg_path) as f:
        raw = json.load(f)

    dec_raw = raw.get("decoder_config", {})
    dec = DecoderConfig(
        hidden_size=dec_raw.get("hidden_size", 1536),
        num_hidden_layers=dec_raw.get("num_hidden_layers", 28),
        num_attention_heads=dec_raw.get("num_attention_heads", 12),
        num_key_value_heads=dec_raw.get("num_key_value_heads", 2),
        intermediate_size=dec_raw.get("intermediate_size", 8960),
        vocab_size=dec_raw.get("vocab_size", 151936),
        rope_theta=float(dec_raw.get("rope_theta", 1_000_000.0)),
        rms_norm_eps=dec_raw.get("rms_norm_eps", 1e-6),
        max_position_embeddings=dec_raw.get("max_position_embeddings", 32768),
    )
    dec.head_dim = dec.hidden_size // dec.num_attention_heads

    diff_raw = raw.get("diffusion_head_config", {})
    diff = DiffusionHeadConfig(
        hidden_size=diff_raw.get("hidden_size", dec.hidden_size),
        head_layers=diff_raw.get("head_layers", 4),
        head_ffn_ratio=diff_raw.get("head_ffn_ratio", 3.0),
        rms_norm_eps=diff_raw.get("rms_norm_eps", 1e-5),
        latent_size=diff_raw.get("latent_size", 64),
    )

    def _tok(raw_key: str, default_vae: int) -> TokenizerConfig:
        t = raw.get(raw_key, {})
        return TokenizerConfig(
            vae_dim=t.get("vae_dim", default_vae),
            causal=t.get("causal", True),
            encoder_n_filters=t.get("encoder_n_filters", 32),
            encoder_ratios=t.get("encoder_ratios", [8, 5, 5, 4, 2, 2]),
            encoder_depths=t.get("encoder_depths", "3-3-3-3-3-3-8"),
            decoder_ratios=t.get("decoder_ratios", None),
            decoder_n_filters=t.get("decoder_n_filters", 32),
            layernorm=t.get("layernorm", "RMSNorm"),
            layernorm_eps=t.get("layernorm_eps", 1e-5),
            conv_bias=t.get("conv_bias", True),
            mixer_layer=t.get("mixer_layer", "depthwise_conv"),
        )

    def _sem_tok(raw_key: str) -> SemanticTokenizerConfig:
        t = raw.get(raw_key, {})
        return SemanticTokenizerConfig(
            vae_dim=t.get("vae_dim", 128),
            causal=t.get("causal", True),
            encoder_n_filters=t.get("encoder_n_filters", 32),
            encoder_ratios=t.get("encoder_ratios", [8, 5, 5, 4, 2, 2]),
            encoder_depths=t.get("encoder_depths", "3-3-3-3-3-3-8"),
            layernorm=t.get("layernorm", "RMSNorm"),
            layernorm_eps=t.get("layernorm_eps", 1e-5),
            conv_bias=t.get("conv_bias", True),
            mixer_layer=t.get("mixer_layer", "depthwise_conv"),
        )

    acoustic_cfg = _tok("acoustic_tokenizer_config", default_vae=64)
    semantic_cfg = _sem_tok("semantic_tokenizer_config")

    return VibeVoiceModelConfig(
        decoder=dec,
        diffusion_head=diff,
        acoustic_tokenizer=acoustic_cfg,
        semantic_tokenizer=semantic_cfg,
        acoustic_connector_input_dim=acoustic_cfg.vae_dim,
        semantic_connector_input_dim=semantic_cfg.vae_dim,
        connector_output_dim=dec.hidden_size,
    )
