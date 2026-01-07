# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
SpeechT5 Speech Decoder Post-Net - Op-by-Op PyTorch Implementation

Converts decoder hidden states to mel-spectrograms with refinement.
Based on HuggingFace transformers.models.speecht5.modeling_speecht5.SpeechT5SpeechDecoderPostnet

Following GUIDE_PYTORCH_REFERENCE_FROM_HF.md methodology.
Target: PCC = 1.0 vs HuggingFace
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Tuple


@dataclass
class SpeechT5PostNetConfig:
    """Configuration for Speech Decoder Post-Net."""

    hidden_size: int = 768
    num_mel_bins: int = 80
    reduction_factor: int = 2
    postnet_layers: int = 5
    postnet_units: int = 256
    postnet_kernel: int = 5
    postnet_dropout: float = 0.5


class SpeechT5BatchNormConvLayer(nn.Module):
    """
    Single Conv1D layer with BatchNorm, optional Tanh, and Dropout.

    Op-by-op implementation matching HuggingFace.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dropout: float,
        has_activation: bool,
    ):
        super().__init__()

        # Conv1d without bias (BatchNorm has bias)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,  # Same padding
            bias=False,
        )

        # BatchNorm
        self.batch_norm = nn.BatchNorm1d(out_channels)

        # Tanh activation (not on last layer)
        self.activation = nn.Tanh() if has_activation else None

        # Dropout
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            hidden_states: [batch, channels, time_steps]

        Returns:
            output: [batch, channels, time_steps]
        """
        # Op 1: Conv1d
        hidden_states = self.conv(hidden_states)

        # Op 2: BatchNorm
        hidden_states = self.batch_norm(hidden_states)

        # Op 3: Tanh activation (if present)
        if self.activation is not None:
            hidden_states = self.activation(hidden_states)

        # Op 4: Dropout
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class SpeechT5SpeechDecoderPostnet(nn.Module):
    """
    Speech Decoder Post-Net: Converts decoder hidden states to mel-spectrograms.

    Components:
    1. feat_out: Projects hidden states to mel features
    2. prob_out: Predicts stop tokens
    3. postnet: 5-layer convolutional network for mel refinement

    Op-by-op implementation for TTNN translation.
    """

    def __init__(self, config: SpeechT5PostNetConfig):
        super().__init__()

        self.config = config

        # Feature output projection
        # Projects hidden_size → num_mel_bins * reduction_factor
        self.feat_out = nn.Linear(config.hidden_size, config.num_mel_bins * config.reduction_factor)

        # Stop token prediction
        # Projects hidden_size → reduction_factor (stop prediction per frame)
        self.prob_out = nn.Linear(config.hidden_size, config.reduction_factor)

        # Convolutional post-net layers (5 layers)
        self.layers = nn.ModuleList()

        for layer_id in range(config.postnet_layers):
            # First layer: mel_bins → postnet_units
            if layer_id == 0:
                in_channels = config.num_mel_bins
            else:
                in_channels = config.postnet_units

            # Last layer: postnet_units → mel_bins
            if layer_id == config.postnet_layers - 1:
                out_channels = config.num_mel_bins
                has_activation = False  # No Tanh on last layer
            else:
                out_channels = config.postnet_units
                has_activation = True  # Tanh on all but last layer

            layer = SpeechT5BatchNormConvLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=config.postnet_kernel,
                dropout=config.postnet_dropout,
                has_activation=has_activation,
            )
            self.layers.append(layer)

    def postnet(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply convolutional post-net with residual connection.

        Args:
            hidden_states: [batch, time_steps, mel_bins]

        Returns:
            refined: [batch, time_steps, mel_bins]
        """
        # Op 1: Transpose for Conv1d ([B, L, C] → [B, C, L])
        layer_output = hidden_states.transpose(1, 2)

        # Op 2-6: Apply 5 conv layers
        for layer in self.layers:
            layer_output = layer(layer_output)

        # Op 7: Transpose back ([B, C, L] → [B, L, C])
        layer_output = layer_output.transpose(1, 2)

        # Op 8: Residual connection
        return hidden_states + layer_output

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with explicit operations.

        Args:
            hidden_states: [batch, decoder_seq_len, hidden_size]

        Returns:
            outputs_before_postnet: [batch, mel_seq_len, num_mel_bins]
            outputs_after_postnet: [batch, mel_seq_len, num_mel_bins]
            stop_logits: [batch, mel_seq_len]

        Operations:
            1. Project to mel features (with reduction factor)
            2. Reshape to separate mel frames
            3. Apply convolutional post-net
            4. Predict stop tokens
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Op 1: Project to mel features
        # [batch, seq_len, hidden_size] → [batch, seq_len, mel_bins * reduction_factor]
        feat_out = self.feat_out(hidden_states)

        # Op 2: Reshape to unfold reduction factor
        # [batch, seq_len, mel_bins * reduction_factor] → [batch, seq_len * reduction_factor, mel_bins]
        outputs_before_postnet = feat_out.view(batch_size, -1, self.config.num_mel_bins)  # seq_len * reduction_factor

        # Op 3: Apply convolutional post-net (with residual)
        outputs_after_postnet = self.postnet(outputs_before_postnet)

        # Op 4: Predict stop tokens
        # [batch, seq_len, hidden_size] → [batch, seq_len, reduction_factor]
        prob_out = self.prob_out(hidden_states)

        # Op 5: Reshape stop tokens
        # [batch, seq_len, reduction_factor] → [batch, seq_len * reduction_factor]
        stop_logits = prob_out.view(batch_size, -1)

        return outputs_before_postnet, outputs_after_postnet, stop_logits


def load_from_huggingface(model_name: str = "microsoft/speecht5_tts") -> SpeechT5SpeechDecoderPostnet:
    """
    Load post-net weights from HuggingFace checkpoint.

    Returns:
        postnet: SpeechT5SpeechDecoderPostnet with loaded weights
    """
    from transformers import SpeechT5ForTextToSpeech

    # Load HF model
    hf_model = SpeechT5ForTextToSpeech.from_pretrained(model_name)
    hf_postnet = hf_model.speech_decoder_postnet
    hf_config = hf_model.config

    # Create config
    config = SpeechT5PostNetConfig(
        hidden_size=hf_config.hidden_size,
        num_mel_bins=hf_config.num_mel_bins,
        reduction_factor=hf_config.reduction_factor,
        postnet_layers=hf_config.speech_decoder_postnet_layers,
        postnet_units=hf_config.speech_decoder_postnet_units,
        postnet_kernel=hf_config.speech_decoder_postnet_kernel,
        postnet_dropout=hf_config.speech_decoder_postnet_dropout,
    )

    # Create postnet
    postnet = SpeechT5SpeechDecoderPostnet(config)

    # Load feat_out weights
    postnet.feat_out.weight.data.copy_(hf_postnet.feat_out.weight.data)
    postnet.feat_out.bias.data.copy_(hf_postnet.feat_out.bias.data)

    # Load prob_out weights
    postnet.prob_out.weight.data.copy_(hf_postnet.prob_out.weight.data)
    postnet.prob_out.bias.data.copy_(hf_postnet.prob_out.bias.data)

    # Load convolutional layers
    for i, (our_layer, hf_layer) in enumerate(zip(postnet.layers, hf_postnet.layers)):
        # Conv weights (no bias)
        our_layer.conv.weight.data.copy_(hf_layer.conv.weight.data)

        # BatchNorm weights and running stats
        our_layer.batch_norm.weight.data.copy_(hf_layer.batch_norm.weight.data)
        our_layer.batch_norm.bias.data.copy_(hf_layer.batch_norm.bias.data)
        our_layer.batch_norm.running_mean.data.copy_(hf_layer.batch_norm.running_mean.data)
        our_layer.batch_norm.running_var.data.copy_(hf_layer.batch_norm.running_var.data)
        our_layer.batch_norm.num_batches_tracked.data.copy_(hf_layer.batch_norm.num_batches_tracked.data)

    postnet.eval()
    return postnet


if __name__ == "__main__":
    # Quick test
    print("Loading post-net from HuggingFace...")
    postnet = load_from_huggingface()

    print(f"\nPost-net config:")
    print(f"  Hidden size: {postnet.config.hidden_size}")
    print(f"  Mel bins: {postnet.config.num_mel_bins}")
    print(f"  Reduction factor: {postnet.config.reduction_factor}")
    print(f"  Postnet layers: {postnet.config.postnet_layers}")

    # Test forward pass
    batch_size = 1
    seq_len = 10
    hidden_size = 768

    test_input = torch.randn(batch_size, seq_len, hidden_size)

    with torch.no_grad():
        pre_mel, post_mel, stop = postnet(test_input)

    print(f"\nTest forward pass:")
    print(f"  Input: {test_input.shape}")
    print(f"  Pre-postnet mel: {pre_mel.shape}")
    print(f"  Post-postnet mel: {post_mel.shape}")
    print(f"  Stop logits: {stop.shape}")

    print("\n✓ Post-net loaded successfully!")
