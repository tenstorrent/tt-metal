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
    text, speaker_embeddings, processor, vocoder, ttnn_encoder, ttnn_decoder, ttnn_postnet, device, max_steps=100
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
        max_steps: Maximum number of generation steps (default: 100)

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

    spectrogram_ttnn = None  # Will be built incrementally on device
    # Maximum steps for generation (default: 100)

    # Autoregressive generation loop with detailed timing
    import time

    total_decoder_time = 0.0
    total_postnet_time = 0.0
    total_conversion_time = 0.0
    total_concat_time = 0.0
    steps_completed = 0

    for step in range(max_steps):
        step_start = time.time()
        current_seq_len = output_sequence_ttnn.shape[1]
        print(f"Step {step+1}/{max_steps} (seq_len={current_seq_len})", end="", flush=True)
        # Decoder step (with detailed timing breakdown)
        decoder_start = time.time()

        # PHASE 1: Decoder inference (includes prenet + 6 transformer layers)
        decoder_inference_start = time.time()
        if step < 2:  # Only collect timing for first 2 steps to avoid overhead
            decoder_hidden_states, decoder_timing = ttnn_decoder(
                decoder_input_values=output_sequence_ttnn,
                encoder_hidden_states=encoder_output,
                speaker_embeddings=ttnn_speaker_embeddings,
                timing_details=True,
            )
        else:
            decoder_hidden_states = ttnn_decoder(
                decoder_input_values=output_sequence_ttnn,
                encoder_hidden_states=encoder_output,
                speaker_embeddings=ttnn_speaker_embeddings,
            )
        decoder_inference_time = time.time() - decoder_inference_start

        # PHASE 2: Memory management
        memory_mgmt_start = time.time()
        decoder_hidden_states = ensure_l1_memory(decoder_hidden_states)
        memory_mgmt_time = time.time() - memory_mgmt_start

        decoder_time = time.time() - decoder_start
        total_decoder_time += decoder_time

        # Log timing for steps (detailed for first 10 steps)
        if step < 10:
            print(
                f" [Decoder: {decoder_time:.3f}s = inference({decoder_inference_time:.3f}s) + mem_mgmt({memory_mgmt_time:.3f}s)]",
                end="",
                flush=True,
            )
        elif step == 9:  # Show final step timing
            print(f" [Decoder: {decoder_time:.3f}s]", end="", flush=True)

        # Postnet (with detailed timing)
        postnet_start = time.time()

        # PHASE 1: Postnet inference (conv layers + stop logits)
        postnet_inference_start = time.time()
        if step < 2:  # Only collect timing for first 2 steps to avoid overhead
            postnet_output, postnet_timing = ttnn_postnet(decoder_hidden_states, timing_details=True)
            mel_before, mel_after, stop_logits = postnet_output
        else:
            mel_before, mel_after, stop_logits = ttnn_postnet(decoder_hidden_states)
        postnet_inference_time = time.time() - postnet_inference_start

        # PHASE 2: Memory management
        postnet_memory_start = time.time()
        mel_after = ensure_l1_memory(mel_after)
        stop_logits = ensure_l1_memory(stop_logits)
        postnet_memory_time = time.time() - postnet_memory_start

        postnet_time = time.time() - postnet_start
        total_postnet_time += postnet_time

        # Log postnet timing (detailed for first 10 steps)
        if step < 10:
            print(
                f" [Postnet: {postnet_time:.3f}s = inference({postnet_inference_time:.3f}s) + mem_mgmt({postnet_memory_time:.3f}s)]",
                end="",
                flush=True,
            )
        elif step == 9:  # Show final step timing
            print(f" [Postnet: {postnet_time:.3f}s]", end="", flush=True)

        # Print detailed breakdown for first step only
        if step == 0:
            print("\n🔍 Detailed timing breakdown (Step 1):")
            print(f"  Decoder phases:")
            print(f"    Input memory: {decoder_timing['memory_input']:.4f}s")
            print(f"    Prenet: {decoder_timing['prenet']:.4f}s")
            print(f"    Causal mask: {decoder_timing['causal_mask']:.4f}s")
            print(f"    Decoder layers: {decoder_timing['decoder_layers']:.4f}s")
            for i, layer_time in enumerate(decoder_timing["layer_times"]):
                print(f"      Layer {i+1}: {layer_time:.4f}s")
            print(f"    Output memory: {decoder_timing['memory_output']:.4f}s")
            print(f"  Postnet phases:")
            print(f"    Input memory: {postnet_timing['memory_input']:.4f}s")
            print(f"    Mel projection: {postnet_timing['mel_projection']:.4f}s")
            print(f"    Mel reshape: {postnet_timing['mel_reshape']:.4f}s")
            print(f"    Conv postnet: {postnet_timing['conv_postnet']:.4f}s")
            print(f"    Stop projection: {postnet_timing['stop_projection']:.4f}s")
            print(f"    Stop reshape: {postnet_timing['stop_reshape']:.4f}s")
            print(f"    Output memory: {postnet_timing['memory_output']:.4f}s")
            print("  Continuing...", end="", flush=True)

        # Check stopping condition (fully device-side comparison)
        sigmoid_logits = ttnn.sigmoid(stop_logits, memory_config=ttnn.L1_MEMORY_CONFIG)
        sum_prob = ttnn.sum(sigmoid_logits, dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)
        should_stop = ttnn.ge(sum_prob, 0.5, memory_config=ttnn.L1_MEMORY_CONFIG)
        any_stop_scalar = ttnn.sum(should_stop)
        if ttnn.to_torch(any_stop_scalar).item() > 0:
            print(f" (Early stop at step {step+1})", flush=True)
            break

        # ========== Device-only operations ==========
        conversion_start = time.time()

        # Extract new mel frames (device-only)
        current_seq_len = output_sequence_ttnn.shape[1]
        start_idx = (current_seq_len - 1) * 2  # reduction_factor = 2
        end_idx = start_idx + 2

        # Slice the new frames from mel_after
        new_frames_ttnn = ttnn.slice(
            mel_after,
            [0, start_idx, 0],  # start indices [batch, seq, mel_bins]
            [batch_size, end_idx, num_mel_bins],  # end indices
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # Build spectrogram incrementally on device
        if spectrogram_ttnn is None:
            spectrogram_ttnn = new_frames_ttnn
        else:
            spectrogram_ttnn = l1_concat([spectrogram_ttnn, new_frames_ttnn], dim=1)

        # Extend sequence with last frame from new frames (directly from mel_after)
        last_frame_idx = start_idx + 1
        last_frame_ttnn = ttnn.slice(
            mel_after,
            [0, last_frame_idx, 0],  # start indices [batch, seq, mel_bins] - take frame at start_idx + 1
            [batch_size, last_frame_idx + 1, num_mel_bins],  # end indices
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        output_sequence_ttnn = l1_concat([output_sequence_ttnn, last_frame_ttnn], dim=1)

        steps_completed += 1

    # Transfer final spectrogram from device to host (only final transfer)
    if spectrogram_ttnn is not None:
        final_spectrogram = ttnn.to_torch(spectrogram_ttnn)
    else:
        final_spectrogram = torch.zeros(batch_size, 1, num_mel_bins)

    # Performance Analysis
    print(f"\\n\\n🎯 Performance Analysis:")
    print(f"   Total steps completed: {steps_completed}")
    print(f"   Total decoder time: {total_decoder_time:.3f}s ({total_decoder_time/steps_completed:.3f}s/step)")
    print(f"   Total postnet time: {total_postnet_time:.3f}s ({total_postnet_time/steps_completed:.3f}s/step)")
    print(f"   Total conversion time: {total_conversion_time:.3f}s ({total_conversion_time/steps_completed:.3f}s/step)")
    print(f"   Total concat time: {total_concat_time:.3f}s ({total_concat_time/steps_completed:.3f}s/step)")
    print(
        f"   Total generation time: {total_decoder_time + total_postnet_time + total_conversion_time + total_concat_time:.3f}s"
    )
    print(
        f"   Tokens/sec: {steps_completed / (total_decoder_time + total_postnet_time + total_conversion_time + total_concat_time):.2f}"
    )

    # Generate audio
    print("\\n🎵 Generating final audio...")
    speech = vocoder(final_spectrogram)

    # Cleanup TTNN tensors
    ttnn.deallocate(ttnn_input_ids)
    ttnn.deallocate(ttnn_speaker_embeddings)
    ttnn.deallocate(encoder_output)
    ttnn.deallocate(output_sequence_ttnn)
    if spectrogram_ttnn is not None:
        ttnn.deallocate(spectrogram_ttnn)

    return speech


def main():
    """Main demo function."""

    import argparse

    parser = argparse.ArgumentParser(description="TTNN SpeechT5 TTS Demo")
    parser.add_argument(
        "texts", nargs="+", help="Input text(s) to convert to speech. Each text will generate a separate audio file."
    )
    parser.add_argument(
        "--output_dir", default=".", help="Output directory for audio files (default: current directory)"
    )
    parser.add_argument("--max_steps", type=int, default=100, help="Maximum number of generation steps (default: 100)")

    args = parser.parse_args()

    print("🎵 TTNN SpeechT5 Clean Demo")
    print("=" * 40)
    print(f"📝 Processing {len(args.texts)} input text(s)")
    print(f"🎯 Max steps per generation: {args.max_steps}")
    print(f"📁 Output directory: {args.output_dir}")

    # Create output directory if it doesn't exist
    import os

    os.makedirs(args.output_dir, exist_ok=True)

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

        # Warm-up phase to compile TTNN operations (separate from timing)
        import time

        warmup_start_time = time.time()
        print("🔥 Warming up TTNN operations...")
        print("   This may take ~30-45 seconds as TTNN compiles kernels for optimal performance")
        print("   TTNN will optimize operations for the decoder, postnet, and memory management")
        warmup_text = "Hi there"
        # Reset KV cache before warm-up
        ttnn_decoder.reset_cache()
        warmup_speech = generate_speech_ttnn(
            warmup_text,
            speaker_embeddings,
            processor,
            vocoder,
            ttnn_encoder,
            ttnn_decoder,
            ttnn_postnet,
            device,
            max_steps=20,
        )
        warmup_duration = time.time() - warmup_start_time
        print(f"✅ Warm-up completed in {warmup_duration:.1f}s (generated {len(warmup_speech)} samples)")
        print("   TTNN kernels are now optimized - subsequent inference will be much faster!")

        # Generate speech for each input text
        for i, text in enumerate(args.texts, 1):
            print(f"\n🎵 [{i}/{len(args.texts)}] Generating speech for: '{text}'")

            # Reset KV cache for new sequence
            ttnn_decoder.reset_cache()

            # Generate filename from text (sanitize for filesystem)
            safe_text = "".join(c for c in text[:50] if c.isalnum() or c in (" ", "-", "_")).rstrip()
            if not safe_text:
                safe_text = f"speech_{i}"
            safe_text = safe_text.replace(" ", "_")
            output_file = os.path.join(args.output_dir, f"speech_ttnn_{safe_text}.wav")

            # Time the generation
            generation_start = time.time()
            speech = generate_speech_ttnn(
                text,
                speaker_embeddings,
                processor,
                vocoder,
                ttnn_encoder,
                ttnn_decoder,
                ttnn_postnet,
                device,
                max_steps=args.max_steps,
            )
            generation_time = time.time() - generation_start

            # Save audio
            sf.write(output_file, speech.squeeze().detach().numpy(), samplerate=16000)
            audio_duration = len(speech.squeeze()) / 16000.0

            print(f"✅ Audio saved to {output_file}")
            print(".3f")
            print(".2f")

        # Cleanup
        ttnn.close_device(device)
        print("\n✅ Demo completed successfully!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
