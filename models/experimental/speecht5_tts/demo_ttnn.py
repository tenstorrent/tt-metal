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
from models.experimental.speecht5_tts.tt.ttnn_speecht5_generator import SpeechT5Generator


def get_high_perf_compute_config(device):
    """Get optimized compute kernel configuration with FP32 accumulation."""
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,  # Enable FP32 destination accumulation for better accuracy
        packer_l1_acc=False,  # Disable L1 accumulation when using FP32 dest acc
    )


def generate_speech_ttnn(
    text,
    speaker_embeddings,
    processor,
    vocoder,
    ttnn_encoder,
    ttnn_decoder,
    ttnn_postnet,
    device,
    max_steps=100,
    return_stats=False,
    warmup_mode=False,
    enable_trace=False,
    generator=None,
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
        return_stats: If True, return statistics along with speech (default: False)
        warmup_mode: If True, skip vocoder and detailed timing for faster warm-up (default: False)
        enable_trace: If True, use trace execution for decoder and postnet (default: False)
        generator: SpeechT5Generator instance for trace support (required if enable_trace=True)

    Returns:
        torch.Tensor: Generated audio waveform (if return_stats=False)
        tuple: (speech, stats_dict) if return_stats=True
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

    # Import time for timing measurements
    import time

    # Encoder forward pass (with timing)
    encoder_start = time.time()
    if enable_trace and generator is not None:
        # Use trace execution for faster inference
        seq_len = ttnn_input_ids.shape[1]
        encoder_output = generator._execute_encoder_trace(seq_len, ttnn_input_ids)
    else:
        # Use regular execution
        encoder_output = ttnn_encoder(ttnn_input_ids)[0]
    encoder_time = time.time() - encoder_start

    # No KV cache for this demo

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
    total_decoder_time = 0.0
    total_postnet_time = 0.0
    total_conversion_time = 0.0
    total_concat_time = 0.0
    steps_completed = 0

    # Complete decoder loop timing
    decoder_loop_start = time.time()

    for step in range(max_steps):
        step_start = time.time()

        # Show progress every 20 steps
        if (step + 1) % 20 == 0 or step == 0:
            if warmup_mode:
                print(f"   Warm-up: Step {step+1}/{max_steps}", end="", flush=True)
            else:
                print(f"   Inference: Step {step+1}/{max_steps}", end="", flush=True)

        # Decoder step (with detailed timing breakdown)
        decoder_start = time.time()

        # PHASE 1: Decoder inference (includes prenet + 6 transformer layers)
        decoder_inference_start = time.time()
        # print("output_sequence_ttnn", output_sequence_ttnn.shape)
        if enable_trace and generator is not None:
            # Use trace execution for faster inference
            current_seq_len = output_sequence_ttnn.shape[1]
            decoder_hidden_states = generator._execute_decoder_trace(
                current_seq_len, output_sequence_ttnn, encoder_output, ttnn_speaker_embeddings
            )
            if step < 1 and warmup_mode:
                decoder_timing = {}  # Dummy timing dict for warmup mode
            else:
                decoder_timing = {}  # Trace execution doesn't provide detailed timing
        else:
            # Use regular execution
            if (
                step < 1 and not warmup_mode
            ):  # Collect timing for first 10 steps for detailed breakdown (skip in warmup mode)
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
                if step < 1 and warmup_mode:
                    decoder_timing = {}  # Dummy timing dict for warmup mode
        decoder_inference_time = time.time() - decoder_inference_start

        # PHASE 2: Memory management
        memory_mgmt_start = time.time()
        memory_mgmt_time = time.time() - memory_mgmt_start

        decoder_time = time.time() - decoder_start
        total_decoder_time += decoder_time

        # Postnet (with detailed timing)
        postnet_start = time.time()

        # PHASE 1: Postnet inference (conv layers + stop logits)
        postnet_inference_start = time.time()
        if enable_trace and generator is not None:
            # Use trace execution for faster inference
            mel_before, mel_after, stop_logits = generator._execute_postnet_trace(decoder_hidden_states)
            if step < 1 and warmup_mode:
                postnet_timing = {}  # Dummy timing dict for warmup mode
            else:
                postnet_timing = {}  # Trace execution doesn't provide detailed timing
        else:
            # Use regular execution
            if (
                step < 1 and not warmup_mode
            ):  # Collect timing for first 10 steps for detailed breakdown (skip in warmup mode)
                postnet_output, postnet_timing = ttnn_postnet(decoder_hidden_states, timing_details=True)
                mel_before, mel_after, stop_logits = postnet_output
            else:
                mel_before, mel_after, stop_logits = ttnn_postnet(decoder_hidden_states)
                if step < 1 and warmup_mode:
                    postnet_timing = {}  # Dummy timing dict for warmup mode
        postnet_inference_time = time.time() - postnet_inference_start

        # PHASE 2: Memory management
        postnet_memory_start = time.time()
        postnet_memory_time = time.time() - postnet_memory_start

        postnet_time = time.time() - postnet_start
        total_postnet_time += postnet_time

        # Check stopping condition (fully device-side comparison)
        sigmoid_logits = ttnn.sigmoid(stop_logits, memory_config=ttnn.L1_MEMORY_CONFIG)
        sum_prob = ttnn.sum(sigmoid_logits, dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)
        should_stop = ttnn.ge(sum_prob, 0.5, memory_config=ttnn.L1_MEMORY_CONFIG)
        any_stop_scalar = ttnn.sum(should_stop)
        if ttnn.to_torch(any_stop_scalar).item() > 0:
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
            spectrogram_ttnn = ttnn.concat(
                [spectrogram_ttnn, new_frames_ttnn], dim=1, memory_config=ttnn.L1_MEMORY_CONFIG
            )

        # Extend sequence with last frame from new frames (directly from mel_after)
        last_frame_idx = start_idx + 1
        last_frame_ttnn = ttnn.slice(
            mel_after,
            [0, last_frame_idx, 0],  # start indices [batch, seq, mel_bins] - take frame at start_idx + 1
            [batch_size, last_frame_idx + 1, num_mel_bins],  # end indices
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        output_sequence_ttnn = ttnn.concat(
            [output_sequence_ttnn, last_frame_ttnn], dim=1, memory_config=ttnn.L1_MEMORY_CONFIG
        )

        steps_completed += 1

        # Complete progress message
        if (step + 1) % 20 == 0:
            print(" ✓", flush=True)

    # End decoder loop timing
    decoder_loop_time = time.time() - decoder_loop_start

    # Transfer final spectrogram from device to host (only final transfer)
    if spectrogram_ttnn is not None:
        final_spectrogram = ttnn.to_torch(spectrogram_ttnn)
    else:
        final_spectrogram = torch.zeros(batch_size, 1, num_mel_bins)

    # Performance Analysis (silently collected for final summary)

    # Generate audio (skip in warmup mode)
    if not warmup_mode:
        print("\\n🎵 Generating final audio...")
        speech = vocoder(final_spectrogram)
    else:
        speech = torch.zeros(1, 16000)  # Dummy output for warmup mode

    # Cleanup TTNN tensors
    ttnn.deallocate(ttnn_input_ids)
    ttnn.deallocate(ttnn_speaker_embeddings)
    ttnn.deallocate(encoder_output)
    ttnn.deallocate(output_sequence_ttnn)
    if spectrogram_ttnn is not None:
        ttnn.deallocate(spectrogram_ttnn)

    if return_stats:
        # Calculate TTFT (Time To First Token) and token/sec
        ttft = encoder_time  # Encoder processes input to generate first context
        avg_token_time = decoder_loop_time / max(steps_completed, 1) if steps_completed > 0 else 0
        token_per_sec = 1.0 / avg_token_time if avg_token_time > 0 else 0

        stats = {
            "steps_completed": steps_completed,
            "final_seq_len": current_seq_len,
            "ttft": ttft,  # Time To First Token (encoder time)
            "avg_token_time": avg_token_time,  # Average time per token (decoder + postnet)
            "token_per_sec": token_per_sec,  # Tokens per second
            "encoder_time": encoder_time,
            "decoder_loop_time": decoder_loop_time,
            "total_decoder_time": total_decoder_time,
            "total_postnet_time": total_postnet_time,
            "total_conversion_time": total_conversion_time,
            "total_concat_time": total_concat_time,
        }
        return speech, stats
    else:
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
    parser.add_argument(
        "--enable_trace", action="store_true", help="Enable TTNN trace for faster inference (default: False)"
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    import os

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        # Enable persistent kernel cache for faster subsequent runs
        ttnn.device.EnablePersistentKernelCache()

        # Initialize device
        device = ttnn.open_device(device_id=0, l1_small_size=300000, trace_region_size=10000000)

        # Enable program cache for faster inference
        device.enable_program_cache()

        # Load models
        processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
        vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

        # Load speaker embeddings
        embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

        # Initialize TTNN models

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

        # Create individual TTNN models
        ttnn_encoder = TTNNSpeechT5Encoder(
            device,
            preprocess_encoder_parameters(model.speecht5.encoder, encoder_config, device),
            encoder_config,
        )

        ttnn_decoder = TTNNSpeechT5Decoder(
            device,
            preprocess_decoder_parameters(model.speecht5.decoder, decoder_config, device, speaker_embeddings),
            decoder_config,
            max_sequence_length=args.max_steps,  # Pass the max_steps value
        )

        ttnn_postnet = TTNNSpeechT5SpeechDecoderPostnet(
            device,
            preprocess_postnet_parameters(model.speech_decoder_postnet, postnet_config, device),
            postnet_config,
        )

        # Create generator wrapper for trace support
        generator = SpeechT5Generator(
            encoder=ttnn_encoder,
            decoder=ttnn_decoder,
            postnet=ttnn_postnet,
            device=device,
            max_steps=args.max_steps,
        )

        # Warm-up phase to compile TTNN operations (separate from timing)
        import time

        warmup_start_time = time.time()
        if args.enable_trace:
            print("🔥 Warming up TTNN operations with trace capture...")
            print("   This may take ~2-3 minutes as TTNN captures traces for encoder, decoder and postnet")
            print("   TTNN will pre-compile operations for all sequence lengths")
            generator.warmup_encoder_traces()
            generator.warmup_decode_traces()
            print("   Performing final trace validation run...")
        else:
            print("🔥 Warming up TTNN operations...")
            print("   This may take ~30-45 seconds as TTNN compiles kernels for optimal performance")
            print("   TTNN will optimize operations for the decoder, postnet, and memory management")

        # Use the first input text for warm-up to ensure encoder processes the actual input
        warmup_text = args.texts[0]
        warmup_speech = generate_speech_ttnn(
            warmup_text,
            speaker_embeddings,
            processor,
            vocoder,
            ttnn_encoder,
            ttnn_decoder,
            ttnn_postnet,
            device,
            max_steps=args.max_steps,
            warmup_mode=True,
            enable_trace=args.enable_trace,
            generator=generator,  # Pass generator for trace support
        )
        warmup_duration = time.time() - warmup_start_time
        print(f"✅ Warm-up completed in {warmup_duration:.1f}s (generated {len(warmup_speech)} samples)")
        if args.enable_trace:
            print("   TTNN traces are now captured - subsequent inference will be much faster!")
        else:
            print("   TTNN kernels are now optimized - subsequent inference will be much faster!")

        # Generate speech for each input text
        results = []
        for i, text in enumerate(args.texts, 1):
            # Generate filename from text (sanitize for filesystem)
            safe_text = "".join(c for c in text[:50] if c.isalnum() or c in (" ", "-", "_")).rstrip()
            if not safe_text:
                safe_text = f"speech_{i}"
            safe_text = safe_text.replace(" ", "_")
            output_file = os.path.join(args.output_dir, f"speech_ttnn_{safe_text}.wav")

            # Time the generation
            generation_start = time.time()
            speech, generation_stats = generate_speech_ttnn(
                text,
                speaker_embeddings,
                processor,
                vocoder,
                ttnn_encoder,
                ttnn_decoder,
                ttnn_postnet,
                device,
                max_steps=args.max_steps,
                return_stats=True,
                enable_trace=args.enable_trace,
                generator=generator,
            )
            generation_time = time.time() - generation_start

            # Save audio
            sf.write(output_file, speech.squeeze().detach().numpy(), samplerate=16000)
            audio_duration = len(speech.squeeze()) / 16000.0

            # Calculate tokens/sec
            tokens_generated = generation_stats.get("steps_completed", 0)
            tokens_per_sec = tokens_generated / generation_time if generation_time > 0 else 0

            # Store results
            result = {
                "text": text,
                "output_file": output_file,
                "generation_time": generation_time,
                "audio_duration": audio_duration,
                "tokens_generated": tokens_generated,
                "tokens_per_sec": tokens_per_sec,
                "sequence_length": generation_stats.get("final_seq_len", 0),
                "ttft": generation_stats.get("ttft", 0),
                "token_per_sec": generation_stats.get("token_per_sec", 0),
                "encoder_time": generation_stats.get("encoder_time", 0),
                "decoder_loop_time": generation_stats.get("decoder_loop_time", 0),
            }
            results.append(result)

        # Display summary table
        print("\n" + "=" * 140)
        print("📊 INFERENCE SUMMARY")
        print("=" * 140)
        print(f"{'#':<4} {'Text':<25} {'Tokens':<8} {'TTFT(ms)':<9} {'Token/s':<10} {'Time(s)':<10} {'Audio(s)':<8}")
        print("-" * 114)

        total_generation_time = 0
        total_tokens = 0
        total_audio_duration = 0

        for i, result in enumerate(results, 1):
            truncated_text = result["text"][:20] + "..." if len(result["text"]) > 23 else result["text"]
            print(
                f"{i:<4} {truncated_text:<25} {result['tokens_generated']:<8} {result.get('ttft', 0)*1000:<8.0f} {result.get('token_per_sec', 0):<10.2f} {result['generation_time']:<10.3f} {result['audio_duration']:<8.1f}"
            )
            total_generation_time += result["generation_time"]
            total_tokens += result["tokens_generated"]
            total_audio_duration += result["audio_duration"]

        print("-" * 114)
        print(
            f"{'TOTAL':<4} {'':<25} {total_tokens:<8} {'-':<9} {'-':<10} {total_generation_time:<10.3f} {total_audio_duration:<8.1f}"
        )

        print(f"\n📈 Overall Statistics:")
        print(f"   • Total inference time: {total_generation_time:.3f}s")
        print(f"   • Total tokens generated: {total_tokens}")
        print(f"   • Total audio duration: {total_audio_duration:.3f}s")
        print(f"   • Average tokens/sec: {total_tokens/total_generation_time:.2f} (across all texts)")
        print(f"   • Average audio duration: {total_audio_duration/len(results):.3f}s per text")

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
