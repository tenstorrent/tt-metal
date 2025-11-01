#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Clean TTNN SpeechT5 Demo - Streamlined Text-to-Speech Generation

This demo showcases the complete TTNN-optimized SpeechT5 pipeline with minimal code
and maximum performance. Generates high-quality speech from text input.

Features:
- Full TTNN implementation (Encoder + Decoder + Postnet)
- L1 memory optimizations for maximum performance
- HiFi4 compute kernels with optimized configurations
- Minimal torch ↔ ttnn conversions
- Clean, production-ready code
"""

import sys
import torch
import ttnn
import soundfile as sf
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset

sys.path.append("/home/ttuser/ssinghal/PR-fix/speecht5_tts/tt-metal")
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


def get_high_perf_compute_config():
    """Get optimized compute kernel configuration."""
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )


def l1_matmul(a, b, *args, **kwargs):
    """Optimized matrix multiplication with L1 memory."""
    if "compute_kernel_config" not in kwargs:
        kwargs["compute_kernel_config"] = get_high_perf_compute_config()
    if "memory_config" not in kwargs:
        kwargs["memory_config"] = ttnn.L1_MEMORY_CONFIG
    return ttnn.matmul(a, b, *args, **kwargs)


def l1_linear(input_tensor, weight, bias=None, *args, **kwargs):
    """Optimized linear layer with L1 memory."""
    if "compute_kernel_config" not in kwargs:
        kwargs["compute_kernel_config"] = get_high_perf_compute_config()
    if "memory_config" not in kwargs:
        kwargs["memory_config"] = ttnn.L1_MEMORY_CONFIG
    return ttnn.linear(input_tensor, weight, bias=bias, *args, **kwargs)


def l1_concat(tensors, *args, **kwargs):
    """Optimized concatenation with L1 memory."""
    return ttnn.concat(tensors, *args, memory_config=ttnn.L1_MEMORY_CONFIG, **kwargs)


def ensure_l1_memory(tensor):
    """Ensure tensor is in L1 memory for optimal performance."""
    return ttnn.to_memory_config(tensor, ttnn.L1_MEMORY_CONFIG)


def generate_speech_ttnn(
    text, speaker_embeddings, processor, vocoder, ttnn_encoder, ttnn_decoder, ttnn_postnet, device
):
    """
    Generate speech using optimized TTNN SpeechT5 pipeline.

    Args:
        text: Input text string
        speaker_embeddings: Speaker embedding tensor
        processor: SpeechT5Processor
        vocoder: SpeechT5HifiGan
        ttnn_encoder: TTNN encoder model
        ttnn_decoder: TTNN decoder model
        ttnn_postnet: TTNN postnet model
        device: TTNN device

    Returns:
        torch.Tensor: Generated audio waveform
    """

    # Process input text
    inputs = processor(text=text, return_tensors="pt")
    token_ids = inputs["input_ids"]

    # Convert inputs to TTNN with L1 memory
    ttnn_input_ids = ttnn.from_torch(
        token_ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    ttnn_speaker_embeddings = ttnn.from_torch(
        speaker_embeddings,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    # Encoder forward pass
    encoder_output = ttnn_encoder(ttnn_input_ids)[0]
    encoder_output = ensure_l1_memory(encoder_output)

    # Initialize decoder sequence
    batch_size = token_ids.shape[0]
    num_mel_bins = 80  # Standard for SpeechT5
    output_sequence_ttnn = ttnn.from_torch(
        torch.zeros(batch_size, 1, num_mel_bins),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    spectrogram = []
    max_steps = min(int(token_ids.shape[1] * 5.0), 50)  # Adaptive max steps

    # Autoregressive generation loop
    for step in range(max_steps):
        # Decoder step
        decoder_hidden_states = ttnn_decoder(
            decoder_input_values=output_sequence_ttnn,
            encoder_hidden_states=encoder_output,
            speaker_embeddings=ttnn_speaker_embeddings,
        )
        decoder_hidden_states = ensure_l1_memory(decoder_hidden_states)

        # Postnet
        postnet_output = ttnn_postnet(decoder_hidden_states)
        mel_before, mel_after, stop_logits = postnet_output
        mel_after = ensure_l1_memory(mel_after)
        stop_logits = ensure_l1_memory(stop_logits)

        # Check stopping condition
        stop_logits_torch = ttnn.to_torch(stop_logits)
        prob = torch.sigmoid(stop_logits_torch)
        if torch.sum(prob, dim=-1) >= 0.5:
            break

        # Extract new mel frame
        mel_after_torch = ttnn.to_torch(mel_after)
        current_seq_len = output_sequence_ttnn.shape[1]
        start_idx = (current_seq_len - 1) * 2  # reduction_factor = 2
        new_spectrum = mel_after_torch[:, start_idx : start_idx + 2, :]
        spectrogram.append(new_spectrum)

        # Extend sequence
        last_frame_ttnn = ttnn.from_torch(
            new_spectrum[:, -1:, :],
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        output_sequence_ttnn = l1_concat([output_sequence_ttnn, last_frame_ttnn], dim=1)

    # Concatenate spectrogram
    if spectrogram:
        final_spectrogram = torch.cat(spectrogram, dim=1)
    else:
        final_spectrogram = torch.zeros(batch_size, 1, num_mel_bins)

    # Generate audio
    speech = vocoder(final_spectrogram)

    # Cleanup TTNN tensors
    ttnn.deallocate(ttnn_input_ids)
    ttnn.deallocate(ttnn_speaker_embeddings)
    ttnn.deallocate(encoder_output)
    ttnn.deallocate(output_sequence_ttnn)

    return speech


def main():
    """Main demo function."""

    print("🎵 TTNN SpeechT5 Clean Demo")
    print("=" * 40)

    # Configuration
    text = "Hello, my dog is cute."
    output_file = "speech_ttnn_clean.wav"

    try:
        # Initialize device
        device = ttnn.open_device(device_id=0, l1_small_size=24576, trace_region_size=10000000)

        # Load models
        print("Loading models...")
        processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
        vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

        # Load speaker embeddings
        embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

        # Initialize TTNN models
        print("Initializing TTNN models...")

        # Encoder config
        encoder_config = TTNNEncoderConfig(
            vocab_size=model.config.vocab_size,
            hidden_size=model.config.hidden_size,
            num_layers=model.config.encoder_layers,
            num_heads=model.config.encoder_attention_heads,
            ffn_dim=model.config.encoder_ffn_dim,
            max_position_embeddings=model.config.max_length,
            layer_norm_eps=model.config.layer_norm_eps,
        )

        # Decoder config
        decoder_config = TTNNDecoderConfig(
            hidden_size=model.config.hidden_size,
            num_layers=model.config.decoder_layers,
            num_heads=model.config.decoder_attention_heads,
            ffn_dim=model.config.decoder_ffn_dim,
            max_position_embeddings=model.config.max_length,
            layer_norm_eps=model.config.layer_norm_eps,
            num_mel_bins=model.config.num_mel_bins,
            reduction_factor=model.config.reduction_factor,
            speech_decoder_prenet_units=model.config.speech_decoder_prenet_units,
            speech_decoder_prenet_layers=model.config.speech_decoder_prenet_layers,
            speech_decoder_prenet_dropout=model.config.speech_decoder_prenet_dropout,
            speaker_embedding_dim=model.config.speaker_embedding_dim,
        )

        # Postnet config
        postnet_config = TTNNPostNetConfig(
            hidden_size=model.config.hidden_size,
            num_mel_bins=model.config.num_mel_bins,
            reduction_factor=model.config.reduction_factor,
            postnet_layers=model.config.speech_decoder_postnet_layers,
            postnet_units=model.config.speech_decoder_postnet_units,
            postnet_kernel=model.config.speech_decoder_postnet_kernel,
        )

        # Create TTNN models
        ttnn_encoder = TTNNSpeechT5Encoder(
            device,
            preprocess_encoder_parameters(model.speecht5.encoder, encoder_config, device),
            encoder_config,
        )

        ttnn_decoder = TTNNSpeechT5Decoder(
            device,
            preprocess_decoder_parameters(model.speecht5.decoder, decoder_config, device),
            decoder_config,
        )

        ttnn_postnet = TTNNSpeechT5SpeechDecoderPostnet(
            device,
            preprocess_postnet_parameters(model.speech_decoder_postnet, postnet_config, device),
            postnet_config,
        )

        # Generate speech
        print(f"Generating speech for: '{text}'")
        speech = generate_speech_ttnn(
            text, speaker_embeddings, processor, vocoder, ttnn_encoder, ttnn_decoder, ttnn_postnet, device
        )

        # Save audio
        sf.write(output_file, speech.squeeze().detach().numpy(), samplerate=16000)
        audio_duration = len(speech.squeeze()) / 16000.0
        print(f"✅ Audio saved to {output_file}")
        print(".3f")
        # Cleanup
        ttnn.close_device(device)
        print("✅ Demo completed successfully!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
