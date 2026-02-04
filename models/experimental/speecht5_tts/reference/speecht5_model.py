# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
SpeechT5 Model - uses HuggingFace implementation as reference for PCC validation.
This serves as the ground truth for testing ttnn implementations.
"""

import torch
import torch.nn as nn
from transformers import SpeechT5ForTextToSpeech
from typing import Optional, Tuple


class SpeechT5ModelReference(nn.Module):
    """
    Wrapper around HuggingFace SpeechT5 for consistent interface.
    This is used as the PyTorch reference for ttnn PCC validation.
    """

    def __init__(self, model_name: str = "microsoft/speecht5_tts"):
        super().__init__()
        self.model = SpeechT5ForTextToSpeech.from_pretrained(model_name)
        self.config = self.model.config
        self.model.eval()

    def get_encoder(self):
        """Get the encoder module"""
        return self.model.speecht5.encoder

    def get_decoder(self):
        """Get the decoder module"""
        return self.model.speecht5.decoder

    def get_postnet(self):
        """Get the speech decoder postnet"""
        return self.model.speech_decoder_postnet

    def forward_encoder(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through encoder only.

        Args:
            input_ids: [batch, text_seq_len] - text token IDs
            attention_mask: [batch, text_seq_len] - attention mask

        Returns:
            encoder_output: [batch, text_seq_len, hidden_size]
        """
        encoder_output = self.model.speecht5.encoder(input_ids, attention_mask)
        return encoder_output.last_hidden_state

    def forward_decoder(
        self,
        decoder_input_values: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        speaker_embeddings: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through decoder only.

        Args:
            decoder_input_values: [batch, mel_seq_len, num_mel_bins] - mel-spectrogram input
            encoder_hidden_states: [batch, text_seq_len, hidden_size] - encoder output
            speaker_embeddings: [batch, speaker_embedding_dim] - speaker embeddings
            encoder_attention_mask: [batch, text_seq_len] - encoder attention mask

        Returns:
            outputs_before_postnet: [batch, mel_seq_len * reduction_factor, num_mel_bins]
            outputs_after_postnet: [batch, mel_seq_len * reduction_factor, num_mel_bins]
        """
        # Forward through decoder
        decoder_output = self.model.speecht5.decoder(
            input_values=decoder_input_values,
            encoder_hidden_states=encoder_hidden_states,
            speaker_embeddings=speaker_embeddings,
            encoder_attention_mask=encoder_attention_mask,
        )

        # Get mel-spectrogram predictions before and after postnet
        outputs_before_postnet = self.model.speech_decoder_postnet.feat_out(decoder_output.last_hidden_state)
        outputs_after_postnet = self.model.speech_decoder_postnet(outputs_before_postnet.transpose(1, 2))
        outputs_after_postnet = outputs_after_postnet.transpose(1, 2)

        return outputs_before_postnet, outputs_after_postnet

    def forward(
        self,
        input_ids: torch.Tensor,
        decoder_input_values: torch.Tensor,
        speaker_embeddings: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass through the model.

        Args:
            input_ids: [batch, text_seq_len] - text token IDs
            decoder_input_values: [batch, mel_seq_len, num_mel_bins] - mel-spectrogram input
            speaker_embeddings: [batch, speaker_embedding_dim] - speaker embeddings
            attention_mask: [batch, text_seq_len] - attention mask

        Returns:
            encoder_hidden_states: [batch, text_seq_len, hidden_size]
            outputs_before_postnet: [batch, mel_seq_len * reduction_factor, num_mel_bins]
            outputs_after_postnet: [batch, mel_seq_len * reduction_factor, num_mel_bins]
        """
        # Encoder
        encoder_hidden_states = self.forward_encoder(input_ids, attention_mask)

        # Decoder + Postnet
        outputs_before_postnet, outputs_after_postnet = self.forward_decoder(
            decoder_input_values,
            encoder_hidden_states,
            speaker_embeddings,
            attention_mask,
        )

        return encoder_hidden_states, outputs_before_postnet, outputs_after_postnet

    def generate_speech(
        self,
        input_ids: torch.Tensor,
        speaker_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 1000,
    ) -> torch.Tensor:
        """
        Generate speech autoregressively.

        Args:
            input_ids: [batch, text_seq_len] - text token IDs
            speaker_embeddings: [batch, speaker_embedding_dim] - speaker embeddings
            attention_mask: [batch, text_seq_len] - attention mask
            max_length: maximum number of mel frames to generate

        Returns:
            mel_spectrogram: [batch, max_length, num_mel_bins] - generated mel-spectrogram
        """
        return self.model.generate_speech(
            input_ids=input_ids,
            speaker_embeddings=speaker_embeddings,
            attention_mask=attention_mask,
            maxlength=max_length,
        )


def load_reference_model(model_name: str = "microsoft/speecht5_tts") -> SpeechT5ModelReference:
    """
    Load the reference SpeechT5 model.

    Args:
        model_name: HuggingFace model name or path

    Returns:
        model: SpeechT5ModelReference instance
    """
    return SpeechT5ModelReference(model_name)
