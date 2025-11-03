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

    # TEST: Use KV CACHING approach (single-token autoregressive)
    # This is the new implementation that should be debugged
    print("🎵 Using KV CACHING approach (single-token autoregressive)...")

    # Pre-compute cross-attention K/V for all decoder layers (done once!)
    cross_attn_kv = ttnn_decoder.precompute_cross_attention_kv(encoder_output)
    print(f"✓ Pre-computed cross-attention K/V for {len(cross_attn_kv)} layers")

    # Reset KV cache for new text input
    for layer in ttnn_decoder.layers:
        layer.reset_kv_cache()
    print("✓ Reset KV cache for new input")

    # Initialize for single-token autoregressive generation
    batch_size = token_ids.shape[0]
    num_mel_bins = 80  # Standard for SpeechT5

    # 🔥 TTNN OPTIMIZATION: Use persistent tensors instead of Python lists
    max_frames = 50  # Pre-allocate for maximum expected frames (10 steps × 2 frames/step × 2.5x safety)
    spectrogram_tensor = ttnn.zeros(
        [batch_size, max_frames, num_mel_bins],
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    stop_logits_tensor = ttnn.zeros(
        [batch_size, max_frames],
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    frames_generated = 0
    stop_logits_generated = 0

    max_steps = 10  # Run more steps to see stopping behavior

    # Autoregressive generation loop with detailed timing
    import time

    total_decoder_time = 0.0
    total_postnet_time = 0.0
    total_conversion_time = 0.0
    total_concat_time = 0.0

    print("\n🎵 Starting autoregressive generation...")
    for step in range(max_steps):
        step_start = time.time()
        print(f"Step {step+1}/{max_steps}", end="", flush=True)
        # 🔥 TTNN OPTIMIZATION: Prepare input for this step (no host transfers)
        if step == 0:
            # First step: start with zeros (direct TTNN operation)
            current_input = ttnn.zeros(
                [batch_size, 1, num_mel_bins],
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
        else:
            # Subsequent steps: slice last generated frame from persistent tensor
            current_input = ttnn.slice(
                spectrogram_tensor,
                [0, frames_generated - 1, 0],  # Start: [batch, last_frame, mel_start]
                [batch_size, frames_generated, num_mel_bins],  # End: [batch, last_frame+1, mel_end]
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )

        # For now, skip attention mask to test basic KV caching functionality
        # TODO: Fix attention mask handling for KV caching
        causal_mask = None

        # Decoder step (single token with KV caching)
        decoder_start = time.time()
        decoder_hidden_states = ttnn_decoder(
            decoder_input_values=current_input,
            current_pos=step,
            cross_attn_kv=cross_attn_kv,
            speaker_embeddings=ttnn_speaker_embeddings,
            attention_mask=causal_mask,
        )
        decoder_hidden_states = ensure_l1_memory(decoder_hidden_states)
        decoder_time = time.time() - decoder_start
        total_decoder_time += decoder_time

        # Postnet
        postnet_start = time.time()
        postnet_output = ttnn_postnet(decoder_hidden_states)
        mel_before, mel_after, stop_logits = postnet_output
        mel_after = ensure_l1_memory(mel_after)
        stop_logits = ensure_l1_memory(stop_logits)
        postnet_time = time.time() - postnet_start
        total_postnet_time += postnet_time

        # 🔥 TTNN OPTIMIZATION: Check stopping condition on device (minimal host transfer)
        sigmoid_prob = ttnn.sigmoid(stop_logits)
        # Extract scalar using slice (more efficient than full transfer)
        last_prob_tensor = ttnn.slice(
            sigmoid_prob,
            [0, -1, 0],  # Last element: [batch, last_time, mel_dim]
            [1, 1, 1],  # Single scalar
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        # Only transfer the scalar value
        last_prob = ttnn.to_torch(last_prob_tensor).item()

        # Debug: print stop probs occasionally
        print(f" (stop_prob={last_prob:.3f})", end="", flush=True)

        # Accumulate stop logits on device
        stop_logits_tensor = ttnn.concat(
            [stop_logits_tensor[:, :stop_logits_generated, :], sigmoid_prob],
            dim=1,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        stop_logits_generated += sigmoid_prob.shape[1]

        # For autoregressive generation, check if the most recent stop probability > 0.5
        if last_prob > 0.5:  # Last element in the sequence
            print(f" (Early stop at prob={last_prob:.3f})", flush=True)
            break

        # 🔥 TTNN OPTIMIZATION: Accumulate mel frames on device (no host transfers)
        concat_start = time.time()

        # Extract new mel frames using TTNN slice (reduction_factor = 2, so we get 2 frames per step)
        new_frames = ttnn.slice(
            mel_after,
            [0, 0, 0],  # Start: [batch, time_start, mel_start]
            [batch_size, 2, num_mel_bins],  # End: [batch, time_start+2, mel_end] - first 2 frames
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # Accumulate on persistent tensor using TTNN concat
        spectrogram_tensor = ttnn.concat(
            [spectrogram_tensor[:, :frames_generated, :], new_frames],
            dim=1,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        frames_generated += new_frames.shape[1]
        concat_time = time.time() - concat_start
        total_concat_time += concat_time

    else:
        # FULL SEQUENCE approach (like test area but with causal masking)
        print("🎵 Using FULL SEQUENCE approach (with causal masking)...")

        # Initialize decoder sequence (like test area)
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
        stop_logits_list = []
        max_steps = 10  # Run more steps to see stopping behavior

        # Autoregressive generation loop (like test area)
        import time

        total_decoder_time = 0.0
        total_postnet_time = 0.0
        total_conversion_time = 0.0
        total_concat_time = 0.0

        print("\n🎵 Starting full sequence generation...")
        for step in range(max_steps):
            step_start = time.time()
            print(f"Step {step+1}/{max_steps}", end="", flush=True)

            # Create causal mask for autoregressive generation
            seq_len = output_sequence_ttnn.shape[1]
            causal_mask = ttnn_decoder._create_causal_mask(seq_len)

            # Decoder step (FULL SEQUENCE - with proper causal masking)
            decoder_start = time.time()
            decoder_hidden_states = ttnn_decoder(
                decoder_input_values=output_sequence_ttnn,
                encoder_hidden_states=encoder_output,
                speaker_embeddings=ttnn_speaker_embeddings,
                attention_mask=causal_mask,
            )
            decoder_hidden_states = ensure_l1_memory(decoder_hidden_states)
            decoder_time = time.time() - decoder_start
            total_decoder_time += decoder_time

            # Postnet
            postnet_start = time.time()
            postnet_output = ttnn_postnet(decoder_hidden_states)
            mel_before, mel_after, stop_logits = postnet_output
            mel_after = ensure_l1_memory(mel_after)
            stop_logits = ensure_l1_memory(stop_logits)
            postnet_time = time.time() - postnet_start
            total_postnet_time += postnet_time

            # Check stopping condition (like test area)
            stop_logits_torch = ttnn.to_torch(stop_logits)
            prob = torch.sigmoid(stop_logits_torch)
            stop_logits_list.append(stop_logits_torch)
            print(f" (stop_prob={prob[0, -1].item():.3f})", end="", flush=True)
            if torch.sum(prob, dim=-1) >= 0.5:
                print(f" (Early stop)", flush=True)
                break

            # ========== TIMING: Host operations ==========
            conversion_start = time.time()

            # Extract new mel frame (like test area)
            mel_to_torch_start = time.time()
            mel_after_torch = ttnn.to_torch(mel_after)
            current_seq_len = output_sequence_ttnn.shape[1]
            start_idx = (current_seq_len - 1) * 2  # reduction_factor = 2
            new_spectrum = mel_after_torch[:, start_idx : start_idx + 2, :]
            spectrogram.append(new_spectrum)

            # Extend sequence (like test area)
            concat_start = time.time()
            last_frame_ttnn = ttnn.from_torch(
                new_spectrum[:, -1:, :],
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            output_sequence_ttnn = l1_concat([output_sequence_ttnn, last_frame_ttnn], dim=1)
            concat_time = time.time() - concat_start
            total_concat_time += concat_time

    # 🔥 TTNN OPTIMIZATION: Single final transfer from persistent tensor
    if frames_generated > 0:
        # Extract generated frames from persistent tensor (single host transfer)
        final_spectrogram = ttnn.to_torch(
            ttnn.slice(spectrogram_tensor, [0, 0, 0], [batch_size, frames_generated, num_mel_bins])
        )
        print(f"✓ Generated {final_spectrogram.shape[1]} mel frames")

        # Extract stop logits if generated
        if stop_logits_generated > 0:
            all_stop_logits = ttnn.to_torch(ttnn.slice(stop_logits_tensor, [0, 0], [batch_size, stop_logits_generated]))
        else:
            all_stop_logits = torch.zeros(batch_size, 1)

        # Save TTNN mel spectrogram for comparison
        output_file = (
            "models/experimental/speecht5_tts/tests/kv_caching_mel.pt"
            if use_kv_caching
            else "models/experimental/speecht5_tts/tests/full_sequence_mel.pt"
        )
        torch.save({"mel_spectrogram": final_spectrogram, "text": text, "stop_logits": all_stop_logits}, output_file)
        print(f"✓ Saved mel to {output_file}")
    else:
        final_spectrogram = torch.zeros(batch_size, 1, num_mel_bins)

    # Performance Analysis
    print(f"\\n\\n🎯 Performance Analysis:")
    print(f"   Total steps completed: {len(spectrogram)}")
    print(f"   Total decoder time: {total_decoder_time:.3f}s ({total_decoder_time/max(1, len(spectrogram)):.3f}s/step)")
    print(f"   Total postnet time: {total_postnet_time:.3f}s ({total_postnet_time/max(1, len(spectrogram)):.3f}s/step)")
    print(f"   Total concat time: {total_concat_time:.3f}s ({total_concat_time/max(1, len(spectrogram)):.3f}s/step)")
    print(
        f"   Total conversion time: {total_conversion_time:.3f}s ({total_conversion_time/max(1, len(spectrogram)):.3f}s/step)"
    )
    total_time = total_decoder_time + total_postnet_time + total_conversion_time + total_concat_time
    print(f"   Total generation time: {total_time:.3f}s")
    print(f"   Tokens/sec: {len(spectrogram) / max(0.001, total_time):.2f}")

    # Generate audio
    print("\\n🎵 Generating final audio...")
    speech = vocoder(final_spectrogram)

    # Cleanup TTNN tensors
    ttnn.deallocate(ttnn_input_ids)
    ttnn.deallocate(ttnn_speaker_embeddings)
    ttnn.deallocate(encoder_output)
    if not use_kv_caching:
        ttnn.deallocate(output_sequence_ttnn)

    return speech


def main():
    """Main demo function."""

    print("🎵 TTNN SpeechT5 Clean Demo")
    print("=" * 40)

    # Configuration
    text = "Hello, my dog is cute."
    use_kv_caching = False  # Set to False to use full sequence mode
    output_file = "speech_ttnn_kv_caching.wav" if use_kv_caching else "speech_ttnn_full_sequence.wav"

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
