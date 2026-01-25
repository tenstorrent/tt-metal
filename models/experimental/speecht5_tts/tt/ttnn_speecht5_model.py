# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Complete End-to-End TTNN SpeechT5 Model.

Combines:
1. Encoder (text/phoneme → encoder hidden states)
2. Decoder (mel + encoder states → decoder hidden states)
3. Post-Net (decoder states → mel spectrograms + stop predictions)

This is a pure TTNN implementation with NO torch operations in the forward pass.

Target PCC: > 0.94 vs PyTorch reference
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import ttnn

from models.experimental.speecht5_tts.tt.ttnn_speecht5_encoder import (
    TTNNSpeechT5Encoder,
    TTNNEncoderConfig,
    preprocess_encoder_parameters,
)
from models.experimental.speecht5_tts.tt.ttnn_speecht5_decoder import (
    TTNNSpeechT5Decoder,
    TTNNDecoderConfig,
    preprocess_decoder_parameters,
)
from models.experimental.speecht5_tts.tt.ttnn_speecht5_postnet import (
    TTNNSpeechT5SpeechDecoderPostnet,
    TTNNPostNetConfig,
    preprocess_postnet_parameters,
)


@dataclass
class TTNNSpeechT5Config:
    """Complete configuration for TTNN SpeechT5 model."""

    # Shared config
    hidden_size: int = 768
    vocab_size: int = 81

    # Encoder config
    encoder_num_layers: int = 12
    encoder_num_heads: int = 12
    encoder_ffn_dim: int = 3072
    max_relative_distance: int = 160

    # Decoder config
    decoder_num_layers: int = 6
    decoder_num_heads: int = 12
    decoder_ffn_dim: int = 3072
    num_mel_bins: int = 80
    reduction_factor: int = 2
    speech_decoder_prenet_units: int = 256
    speech_decoder_prenet_layers: int = 2
    speaker_embedding_dim: int = 512

    # Post-net config
    postnet_num_layers: int = 5
    postnet_conv_dim: int = 512
    postnet_kernel_size: int = 5

    # Common
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5


class TTNNSpeechT5Model:
    """
    Complete end-to-end TTNN SpeechT5 model.

    Pure TTNN implementation - no torch operations in forward pass.

    Usage:
        # Load PyTorch model
        from reference import load_speecht5_model
        torch_model = load_speecht5_model()

        # Convert to TTNN
        device = ttnn.open_device(device_id=0, l1_small_size=24576)
        config = TTNNSpeechT5Config()
        parameters = preprocess_speecht5_parameters(torch_model, config, device)
        ttnn_model = TTNNSpeechT5Model(device, parameters, config)

        # Run inference
        input_ids = torch.tensor([[1, 2, 3, ...]])  # Text tokens
        decoder_input = torch.randn(1, 10, 80)  # Previous mel frames

        # Convert to TTNN
        ttnn_input_ids = ttnn.from_torch(input_ids, ...)
        ttnn_decoder_input = ttnn.from_torch(decoder_input, ...)

        # Forward pass (all TTNN!)
        feat_out, outputs_after, stop = ttnn_model(
            input_ids=ttnn_input_ids,
            decoder_input_values=ttnn_decoder_input,
        )

        # Convert back to torch for analysis
        feat_out_torch = ttnn.to_torch(feat_out)
    """

    def __init__(self, device, parameters, config: TTNNSpeechT5Config):
        self.device = device
        self.config = config
        self.parameters = parameters

        # Initialize encoder
        encoder_config = TTNNEncoderConfig(
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            num_layers=config.encoder_num_layers,
            num_heads=config.encoder_num_heads,
            ffn_dim=config.encoder_ffn_dim,
            max_relative_distance=config.max_relative_distance,
            dropout=config.dropout,
            layer_norm_eps=config.layer_norm_eps,
        )
        self.encoder = TTNNSpeechT5Encoder(
            device,
            parameters["encoder"],
            encoder_config,
        )

        # Initialize decoder
        decoder_config = TTNNDecoderConfig(
            hidden_size=config.hidden_size,
            num_layers=config.decoder_num_layers,
            num_heads=config.decoder_num_heads,
            ffn_dim=config.decoder_ffn_dim,
            dropout=config.dropout,
            layer_norm_eps=config.layer_norm_eps,
            num_mel_bins=config.num_mel_bins,
            reduction_factor=config.reduction_factor,
            speech_decoder_prenet_units=config.speech_decoder_prenet_units,
            speech_decoder_prenet_layers=config.speech_decoder_prenet_layers,
            speaker_embedding_dim=config.speaker_embedding_dim,
        )
        self.decoder = TTNNSpeechT5Decoder(
            device,
            parameters["decoder"],
            decoder_config,
        )

        # Initialize post-net
        postnet_config = TTNNPostNetConfig(
            hidden_size=config.hidden_size,
            num_mel_bins=config.num_mel_bins,
            reduction_factor=config.reduction_factor,
            postnet_layers=config.postnet_num_layers,
            postnet_units=config.postnet_conv_dim,
            postnet_kernel=config.postnet_kernel_size,
            postnet_dropout=config.dropout,
        )
        self.postnet = TTNNSpeechT5SpeechDecoderPostnet(
            device,
            parameters["postnet"],
            postnet_config,
        )

    def __call__(
        self,
        input_ids: ttnn.Tensor,
        decoder_input_values: ttnn.Tensor,
        speaker_embeddings: Optional[ttnn.Tensor] = None,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        """
        Complete forward pass through encoder, decoder, and post-net.

        All operations are in TTNN - no torch ops!

        Args:
            input_ids: [batch, enc_seq_len] - text/phoneme tokens (uint32)
            decoder_input_values: [batch, dec_seq_len, num_mel_bins] - previous mel frames
            speaker_embeddings: [batch, speaker_embedding_dim] - optional speaker info

        Returns:
            feat_out: [batch, dec_seq_len * reduction_factor, num_mel_bins] - predicted mels
            outputs_after: [batch, dec_seq_len * reduction_factor, num_mel_bins] - with residual
            stop: [batch, dec_seq_len * reduction_factor] - stop token predictions
        """
        # Step 1: Encoder (text → encoder hidden states)
        encoder_output = self.encoder(input_ids)
        # Encoder returns tuple (hidden_states,), extract the tensor
        if isinstance(encoder_output, tuple):
            encoder_hidden_states = encoder_output[0]
        else:
            encoder_hidden_states = encoder_output

        # Step 2: Decoder (mel + encoder states → decoder hidden states)
        decoder_hidden_states = self.decoder(
            decoder_input_values=decoder_input_values,
            encoder_hidden_states=encoder_hidden_states,
            speaker_embeddings=speaker_embeddings,
        )

        # Step 3: Post-net (decoder states → mel + stop predictions)
        feat_out, outputs_after, stop = self.postnet(decoder_hidden_states)

        return feat_out, outputs_after, stop


def preprocess_speecht5_parameters(torch_model, config: TTNNSpeechT5Config, device):
    """
    Preprocess complete SpeechT5 model parameters for TTNN.

    Args:
        torch_model: PyTorch SpeechT5 model with encoder, decoder, postnet
        config: TTNNSpeechT5Config
        device: TTNN device

    Returns:
        parameters: Dict with encoder, decoder, postnet parameters
    """
    print("Preprocessing SpeechT5 parameters...")

    parameters = {}

    # Encoder
    print("  - Encoder parameters...")
    encoder_config = TTNNEncoderConfig(
        hidden_size=config.hidden_size,
        vocab_size=config.vocab_size,
        num_layers=config.encoder_num_layers,
        num_heads=config.encoder_num_heads,
        ffn_dim=config.encoder_ffn_dim,
        max_relative_distance=config.max_relative_distance,
        dropout=config.dropout,
        layer_norm_eps=config.layer_norm_eps,
    )
    parameters["encoder"] = preprocess_encoder_parameters(
        torch_model.encoder,
        encoder_config,
        device,
    )

    # Decoder
    print("  - Decoder parameters...")
    decoder_config = TTNNDecoderConfig(
        hidden_size=config.hidden_size,
        num_layers=config.decoder_num_layers,
        num_heads=config.decoder_num_heads,
        ffn_dim=config.decoder_ffn_dim,
        dropout=config.dropout,
        layer_norm_eps=config.layer_norm_eps,
        num_mel_bins=config.num_mel_bins,
        reduction_factor=config.reduction_factor,
        speech_decoder_prenet_units=config.speech_decoder_prenet_units,
        speech_decoder_prenet_layers=config.speech_decoder_prenet_layers,
        speaker_embedding_dim=config.speaker_embedding_dim,
    )
    parameters["decoder"] = preprocess_decoder_parameters(
        torch_model.decoder,
        decoder_config,
        device,
    )

    # Post-net
    print("  - Post-net parameters...")
    postnet_config = TTNNPostNetConfig(
        hidden_size=config.hidden_size,
        num_mel_bins=config.num_mel_bins,
        reduction_factor=config.reduction_factor,
        postnet_layers=config.postnet_num_layers,
        postnet_units=config.postnet_conv_dim,
        postnet_kernel=config.postnet_kernel_size,
        postnet_dropout=config.dropout,
    )
    parameters["postnet"] = preprocess_postnet_parameters(
        torch_model.postnet,
        postnet_config,
        device,
    )

    print("✓ All parameters preprocessed")

    return parameters
