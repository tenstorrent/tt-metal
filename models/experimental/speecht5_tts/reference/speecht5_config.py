# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass


@dataclass
class SpeechT5Config:
    """Configuration class for SpeechT5 model"""

    # Shared model configuration
    vocab_size: int = 81
    hidden_size: int = 768

    # Encoder configuration
    encoder_layers: int = 12
    encoder_attention_heads: int = 12
    encoder_ffn_dim: int = 3072

    # Decoder configuration
    decoder_layers: int = 6
    decoder_attention_heads: int = 12
    decoder_ffn_dim: int = 3072

    # Speech-specific configuration
    num_mel_bins: int = 80
    reduction_factor: int = 2
    speech_decoder_prenet_units: int = 256
    speech_decoder_prenet_layers: int = 2
    speech_decoder_postnet_units: int = 256
    speech_decoder_postnet_layers: int = 5
    speech_decoder_postnet_kernel: int = 5

    # Speaker embedding
    speaker_embedding_dim: int = 512

    # Other hyperparameters
    max_text_positions: int = 450
    max_speech_positions: int = 4000
    encoder_max_relative_position: int = 160
    decoder_max_relative_position: int = 160
    dropout: float = 0.1
    attention_dropout: float = 0.0
    activation_dropout: float = 0.0
    layer_norm_eps: float = 1e-5
    positional_dropout: float = 0.1
    scale_embedding: bool = False

    @classmethod
    def from_hf_config(cls, hf_config):
        """Create SpeechT5Config from HuggingFace config"""
        return cls(
            vocab_size=hf_config.vocab_size,
            hidden_size=hf_config.hidden_size,
            encoder_layers=hf_config.encoder_layers,
            encoder_attention_heads=hf_config.encoder_attention_heads,
            encoder_ffn_dim=hf_config.encoder_ffn_dim,
            decoder_layers=hf_config.decoder_layers,
            decoder_attention_heads=hf_config.decoder_attention_heads,
            decoder_ffn_dim=hf_config.decoder_ffn_dim,
            num_mel_bins=hf_config.num_mel_bins,
            reduction_factor=hf_config.reduction_factor,
            speech_decoder_prenet_units=hf_config.speech_decoder_prenet_units,
            speech_decoder_prenet_layers=hf_config.speech_decoder_prenet_layers,
            speech_decoder_postnet_units=hf_config.speech_decoder_postnet_units,
            speech_decoder_postnet_layers=hf_config.speech_decoder_postnet_layers,
            speaker_embedding_dim=getattr(hf_config, "speaker_embedding_dim", 512),
            dropout=getattr(hf_config, "dropout", 0.1),
            layer_norm_eps=getattr(hf_config, "layer_norm_epsilon", 1e-5),
        )
