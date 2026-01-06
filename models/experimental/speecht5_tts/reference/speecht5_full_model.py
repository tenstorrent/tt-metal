# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Complete PyTorch Reference for SpeechT5 Text-to-Speech Model.

This combines the individual op-by-op PyTorch components:
- SpeechT5Encoder (pure PyTorch, PCC=1.0 vs HF)
- SpeechT5Decoder (pure PyTorch, PCC=1.0 vs HF)
- SpeechT5SpeechDecoderPostnet (pure PyTorch, PCC=1.0 vs HF)

Includes text processing, speaker embeddings, autoregressive generation, and vocoder.
This serves as the ground truth for TTNN implementation validation.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
from transformers import SpeechT5HifiGan

# Import individual components
from .speecht5_encoder import SpeechT5Encoder, SpeechT5Config, load_from_huggingface as load_encoder_from_huggingface
from .speecht5_decoder import (
    SpeechT5Decoder,
    SpeechT5DecoderConfig,
    load_from_huggingface as load_decoder_from_huggingface,
)
from .speecht5_postnet import (
    SpeechT5SpeechDecoderPostnet,
    SpeechT5PostNetConfig,
    load_from_huggingface as load_postnet_from_huggingface,
)


class SpeechT5FullReference(nn.Module):
    """
    Complete SpeechT5 Text-to-Speech Model using pure PyTorch components.

    This model combines:
    - Encoder: Text → Hidden states
    - Decoder: Autoregressive mel-spectrogram generation
    - Postnet: Mel-spectrogram refinement
    - Vocoder: Mel-spectrogram → Audio waveform

    All components are op-by-op PyTorch implementations validated against HuggingFace.
    """

    def __init__(self, config_dict: Dict[str, Any]):
        super().__init__()

        # Extract configurations
        encoder_config = SpeechT5Config(
            vocab_size=config_dict["vocab_size"],
            hidden_size=config_dict["hidden_size"],
            num_layers=config_dict["encoder_layers"],
            num_heads=config_dict["encoder_attention_heads"],
            ffn_dim=config_dict["encoder_ffn_dim"],
            dropout=config_dict["dropout"],
            layer_norm_eps=config_dict["layer_norm_eps"],
            max_position_embeddings=config_dict["max_position_embeddings"],
            max_relative_distance=config_dict["encoder_max_relative_position"],
        )

        decoder_config = SpeechT5DecoderConfig(
            hidden_size=config_dict["hidden_size"],
            num_layers=config_dict["decoder_layers"],
            num_heads=config_dict["decoder_attention_heads"],
            ffn_dim=config_dict["decoder_ffn_dim"],
            dropout=config_dict["dropout"],
            layer_norm_eps=config_dict["layer_norm_eps"],
            max_position_embeddings=config_dict["max_position_embeddings"],
            max_relative_distance=config_dict["decoder_max_relative_position"],
            num_mel_bins=config_dict["num_mel_bins"],
            reduction_factor=config_dict["reduction_factor"],
            speech_decoder_prenet_units=config_dict["speech_decoder_prenet_units"],
            speech_decoder_prenet_layers=config_dict["speech_decoder_prenet_layers"],
            speaker_embedding_dim=config_dict["speaker_embedding_dim"],
        )

        postnet_config = SpeechT5PostNetConfig(
            hidden_size=config_dict["hidden_size"],
            num_mel_bins=config_dict["num_mel_bins"],
            reduction_factor=config_dict["reduction_factor"],
            postnet_layers=config_dict["postnet_layers"],
            postnet_units=config_dict["postnet_units"],
            postnet_kernel=config_dict["postnet_kernel"],
            postnet_dropout=config_dict["postnet_dropout"],
        )

        # Initialize components
        self.encoder = SpeechT5Encoder(encoder_config)
        self.decoder = SpeechT5Decoder(decoder_config)
        self.postnet = SpeechT5SpeechDecoderPostnet(postnet_config)

        # Note: Speaker embeddings are handled by the decoder's prenet

        # Store configs
        self.config = config_dict
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config
        self.postnet_config = postnet_config

        # Vocoder (HiFiGAN) - loaded separately
        self.vocoder = None

    def load_vocoder(self, vocoder_model_name: str = "microsoft/speecht5_hifigan"):
        """Load the vocoder for audio generation."""
        if self.vocoder is None:
            print(f"Loading vocoder: {vocoder_model_name}")
            self.vocoder = SpeechT5HifiGan.from_pretrained(vocoder_model_name)
            self.vocoder.eval()
        return self.vocoder

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
        # Note: Our reference encoder returns a tuple, extract the hidden states
        return self.encoder(input_ids)[0]

    def forward_decoder_step(
        self,
        decoder_input: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        speaker_embeddings: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single decoder step.

        Args:
            decoder_input: [batch, mel_seq_len, num_mel_bins] - mel input for this step
            encoder_hidden_states: [batch, text_seq_len, hidden_size] - encoder output
            speaker_embeddings: [batch, speaker_embedding_dim] - speaker embeddings
            encoder_attention_mask: [batch, text_seq_len] - encoder attention mask

        Returns:
            mel_output: [batch, mel_seq_len * reduction_factor, num_mel_bins] - postnet output
            last_decoder_hidden: [batch, hidden_size] - last decoder hidden state for stop prediction
        """
        # Decoder forward pass (speaker embeddings handled by decoder prenet)
        decoder_hidden_states = self.decoder(
            decoder_input_values=decoder_input,
            encoder_hidden_states=encoder_hidden_states,
            speaker_embeddings=speaker_embeddings,
            attention_mask=encoder_attention_mask,
        )

        # PyTorch postnet
        mel_before_postnet, mel_after_postnet, stop_logits = self.postnet(decoder_hidden_states)

        # Return the last decoder hidden state (for stop prediction) and mel output
        last_decoder_hidden = decoder_hidden_states[:, -1, :]  # [batch, hidden_size]

        return mel_after_postnet, last_decoder_hidden

    def generate_speech(
        self,
        input_ids: torch.Tensor,
        speaker_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
        minlenratio: float = 0.0,
        maxlenratio: float = 20.0,
        vocoder: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """
        Generate speech autoregressively using stop token prediction.

        Args:
            input_ids: [batch, text_seq_len] - text token IDs
            speaker_embeddings: [batch, speaker_embedding_dim] - speaker embeddings
            attention_mask: [batch, text_seq_len] - attention mask
            threshold: stop token probability threshold
            minlenratio: minimum length ratio relative to input
            maxlenratio: maximum length ratio relative to input
            vocoder: optional vocoder to convert mel to audio

        Returns:
            mel_spectrogram or audio: Generated mel-spectrogram or audio waveform
        """
        batch_size = input_ids.shape[0]

        # Encode text
        encoder_hidden_states = self.forward_encoder(input_ids, attention_mask)

        # Calculate min/max lengths based on input length
        input_len = input_ids.shape[1]
        maxlen = int(input_len * maxlenratio / self.config["reduction_factor"])
        minlen = int(input_len * minlenratio / self.config["reduction_factor"])

        # Initialize decoder input (start with zeros)
        output_sequence = torch.zeros(batch_size, 1, self.config["num_mel_bins"]).to(  # Start with single frame
            input_ids.device
        )

        spectrogram = []
        idx = 0

        while True:
            idx += 1

            # Forward pass through decoder
            mel_output, last_decoder_hidden = self.forward_decoder_step(
                output_sequence,
                encoder_hidden_states,
                speaker_embeddings,
                attention_mask,
            )

            # Take the new spectrum for this step
            # mel_output shape: [batch, current_seq_len * reduction_factor, num_mel_bins]
            # We want the newly generated frames (last reduction_factor frames)
            start_idx = (output_sequence.shape[1] - 1) * self.config["reduction_factor"]
            new_spectrum = mel_output[:, start_idx : start_idx + self.config["reduction_factor"], :]
            spectrogram.append(new_spectrum)

            # Extend output sequence with the last frame of the new spectrum
            last_frame = new_spectrum[:, -1:, :]  # [batch, 1, num_mel_bins]
            output_sequence = torch.cat((output_sequence, last_frame), dim=1)

            # Predict stop probability using prob_out on decoder hidden state
            prob = torch.sigmoid(self.postnet.prob_out(last_decoder_hidden))

            # Check stopping conditions
            if idx < minlen:
                continue
            elif idx >= maxlen:
                # Hit maximum length, stop
                break
            else:
                # Check if stop probability meets threshold
                meet_threshold = torch.sum(prob, dim=-1) >= threshold
                if meet_threshold.any():
                    break

        # Concatenate all spectra
        mel_spectrogram = torch.cat(spectrogram, dim=1)

        # Apply vocoder if explicitly provided
        if vocoder is not None:
            with torch.no_grad():
                # Squeeze batch dimension for vocoder (expects [seq_len, num_mel_bins])
                audio = vocoder(mel_spectrogram.squeeze(0))
            return audio

        # Return mel spectrogram (squeeze batch dimension to match HF)
        return mel_spectrogram.squeeze(0)

    def forward(
        self,
        input_ids: torch.Tensor,
        decoder_input_values: torch.Tensor,
        speaker_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass (teacher forcing mode).

        Args:
            input_ids: [batch, text_seq_len] - text token IDs
            decoder_input_values: [batch, mel_seq_len, num_mel_bins] - mel input
            speaker_embeddings: [batch, speaker_embedding_dim] - speaker embeddings
            attention_mask: [batch, text_seq_len] - attention mask

        Returns:
            encoder_hidden_states: [batch, text_seq_len, hidden_size]
            mel_before_postnet: [batch, mel_seq_len * reduction_factor, num_mel_bins]
            mel_after_postnet: [batch, mel_seq_len * reduction_factor, num_mel_bins]
        """
        # Encoder
        encoder_hidden_states = self.forward_encoder(input_ids, attention_mask)

        # Decoder + Postnet
        mel_before_postnet, mel_after_postnet = self.forward_decoder_step(
            decoder_input_values,
            encoder_hidden_states,
            speaker_embeddings,
            attention_mask,
        )

        return encoder_hidden_states, mel_before_postnet, mel_after_postnet


def load_full_reference_from_huggingface(
    model_name: str = "microsoft/speecht5_tts", vocoder_name: str = "microsoft/speecht5_hifigan"
) -> SpeechT5FullReference:
    """
    Load complete reference model with weights from HuggingFace.

    Args:
        model_name: HuggingFace SpeechT5 model name
        vocoder_name: HuggingFace vocoder model name

    Returns:
        model: SpeechT5FullReference with weights loaded from HF
    """
    print(f"Loading full reference model from {model_name}...")

    # Load HF model to extract config and weights
    from transformers import SpeechT5ForTextToSpeech

    hf_model = SpeechT5ForTextToSpeech.from_pretrained(model_name)

    # Extract configuration
    config_dict = {
        "vocab_size": hf_model.config.vocab_size,
        "hidden_size": hf_model.config.hidden_size,
        "encoder_layers": hf_model.config.encoder_layers,
        "decoder_layers": hf_model.config.decoder_layers,
        "encoder_attention_heads": hf_model.config.encoder_attention_heads,
        "decoder_attention_heads": hf_model.config.decoder_attention_heads,
        "encoder_ffn_dim": hf_model.config.encoder_ffn_dim,
        "decoder_ffn_dim": hf_model.config.decoder_ffn_dim,
        "dropout": getattr(hf_model.config, "hidden_dropout", 0.1),  # Use hidden_dropout as default
        "layer_norm_eps": hf_model.config.layer_norm_eps,
        "max_position_embeddings": getattr(hf_model.config, "max_position_embeddings", 600),
        "encoder_max_relative_position": hf_model.config.encoder_max_relative_position,
        "decoder_max_relative_position": getattr(hf_model.config, "decoder_max_relative_position", 160),
        "num_mel_bins": hf_model.config.num_mel_bins,
        "reduction_factor": hf_model.config.reduction_factor,
        "speech_decoder_prenet_units": hf_model.config.speech_decoder_prenet_units,
        "speech_decoder_prenet_layers": hf_model.config.speech_decoder_prenet_layers,
        "speaker_embedding_dim": hf_model.config.speaker_embedding_dim,
        "postnet_layers": hf_model.config.speech_decoder_postnet_layers,
        "postnet_units": hf_model.config.speech_decoder_postnet_units,
        "postnet_kernel": hf_model.config.speech_decoder_postnet_kernel,
        "postnet_dropout": hf_model.config.speech_decoder_postnet_dropout,
    }

    # Load individual components with weights
    print("Loading encoder weights...")
    encoder = load_encoder_from_huggingface(model_name)

    print("Loading decoder weights...")
    decoder = load_decoder_from_huggingface(model_name)

    print("Loading postnet weights...")
    postnet = load_postnet_from_huggingface(model_name)

    # Create model with loaded components
    model = SpeechT5FullReference(config_dict)
    model.encoder = encoder
    model.decoder = decoder
    model.postnet = postnet

    # Note: Speaker embedding weights are loaded as part of the decoder

    # Load vocoder
    model.load_vocoder(vocoder_name)

    model.eval()
    print("✓ Full reference model loaded successfully!")
    return model
