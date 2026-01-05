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
- Multi-device support (N150, N300, etc.)
"""

import torch
import ttnn
import soundfile as sf
import pytest
import os
import time
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset

from models.experimental.speecht5_tts.tt.ttnn_speecht5_encoder import (
    TTNNSpeechT5Encoder,
    TTNNEncoderConfig,
    preprocess_encoder_parameters,
)
from models.experimental.speecht5_tts.tt.ttnn_speecht5_decoder import (
    TTNNSpeechT5Decoder,
    TTNNDecoderConfig,
    preprocess_decoder_parameters,
    init_kv_cache,
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
    generator=None,
    use_kv_cache=False,
    decoder_config=None,
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
        generator: SpeechT5Generator instance for trace support (enables trace when use_kv_cache=True)
        use_kv_cache: If True, use KV cache for faster autoregressive generation
        decoder_config: TTNNDecoderConfig (required if use_kv_cache=True)

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

    # Trace enabled when KV cache is enabled and generator is provided
    enable_trace = use_kv_cache and generator is not None

    # Start timing for TTFT (Time To First Token/Frame)
    generation_start = time.time()

    # Encoder forward pass (encoder doesn't benefit much from trace, runs only once)
    encoder_start = time.time()
    encoder_output = ttnn_encoder(ttnn_input_ids)[0]
    encoder_time = time.time() - encoder_start

    # If using trace with generator, copy encoder output to pre-allocated tensor
    if enable_trace:
        generator.copy_encoder_output(encoder_output)
        # Use generator's pre-allocated encoder_hidden_states for decoder
        encoder_output_for_decoder = generator.encoder_hidden_states
    else:
        encoder_output_for_decoder = encoder_output

    # Initialize decoder sequence
    batch_size = token_ids.shape[0]
    num_mel_bins = 80  # Standard for SpeechT5
    encoder_seq_len = encoder_output.shape[1]

    # Initialize KV cache if enabled
    kv_cache = None
    cross_attn_cache = None
    if use_kv_cache and decoder_config is not None:
        if enable_trace and generator is not None:
            # Use generator's pre-allocated KV cache for trace stability
            kv_cache = generator.kv_cache
            cross_attn_cache = generator.cross_attn_cache
            # Reset cross-attention cache validity for new generation
            generator._invalidate_cross_attn_cache()
        else:
            kv_cache, cross_attn_cache = init_kv_cache(
                decoder_config,
                device,
                max_batch_size=batch_size,
                max_seq_len=max_steps + 10,
                encoder_seq_len=encoder_seq_len,
            )

    # Initial mel frame (zeros)
    output_sequence_ttnn = ttnn.from_torch(
        torch.zeros(batch_size, 1, num_mel_bins),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    spectrogram_ttnn = None
    spectrogram_frames_cpu = []  # For trace mode: accumulate on CPU to avoid device allocations
    steps_completed = 0
    total_decoder_time = 0.0
    total_postnet_time = 0.0
    current_seq_len = 1  # Track sequence length for stats
    ttft = None  # Time To First Token/Frame - set after step 0 completes

    # For KV cache mode, track the current input frame
    current_input_ttnn = output_sequence_ttnn

    # Autoregressive generation loop
    decoder_loop_start = time.time()
    for step in range(max_steps):
        if (step + 1) % 20 == 0 or step == 0:
            if warmup_mode:
                print(f"   Warm-up: Step {step+1}/{max_steps}", end="", flush=True)
            else:
                print(f"   Inference: Step {step+1}/{max_steps}", end="", flush=True)

        # Decoder step
        decoder_start = time.time()

        if use_kv_cache and kv_cache is not None:
            # KV cache mode: pass only the current frame (seq_len=1 after step 0)

            if enable_trace and generator is not None:
                # WHISPER PATTERN: Preprocessing happens OUTSIDE trace capture
                # 1. Call preprocess_decoder_inputs with position (PE + dropout)
                # 2. Pass preprocessed hidden states to decoder (traced)

                # Update decode position for trace
                generator._reset_decode_pos(step, batch_size)

                # Preprocess: run prenet + PE addition OUTSIDE trace
                preprocessed_hidden_states = ttnn_decoder.preprocess_decoder_inputs(
                    decoder_input_values=current_input_ttnn,
                    position_offset=step,
                )

                # WHISPER PATTERN: Capture trace at step 0, execute starting step 1
                if step == 0:
                    # First iteration: non-traced to populate cross-attention cache
                    decoder_hidden_states = ttnn_decoder(
                        decoder_input_values=None,  # Not used when preprocessed_hidden_states provided
                        encoder_hidden_states=encoder_output_for_decoder,
                        speaker_embeddings=None,  # Pre-baked in prenet
                        kv_cache=kv_cache,
                        cross_attn_cache=cross_attn_cache,
                        cross_attn_cache_valid=False,
                        current_decode_pos=generator.current_decode_pos,
                        preprocessed_hidden_states=preprocessed_hidden_states,
                        encoder_attention_mask=generator.encoder_attention_mask,  # Mask for padding
                    )
                    # Cross-attention cache is now populated
                    generator.cross_attn_cache_valid = True

                    # Capture trace NOW (after first iteration), but use non-traced output
                    if not generator.trace_compiled:
                        generator._capture_decoder_trace(preprocessed_hidden_states)
                else:
                    # Step 1+: Execute trace (trace was captured at step 0)
                    decoder_hidden_states = generator._execute_decoder_trace(preprocessed_hidden_states)
            else:
                # KV cache without trace
                current_pos = ttnn.from_torch(
                    torch.tensor([step], dtype=torch.int32),
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )

                decoder_hidden_states = ttnn_decoder(
                    decoder_input_values=current_input_ttnn,
                    encoder_hidden_states=encoder_output_for_decoder,
                    speaker_embeddings=ttnn_speaker_embeddings,
                    kv_cache=kv_cache,
                    cross_attn_cache=cross_attn_cache,
                    cross_attn_cache_valid=(step > 0),  # Reuse cross-attn cache after first step
                    current_decode_pos=current_pos,
                    position_offset=step,  # Pass step as position for correct positional encoding
                )
        else:
            # Standard mode: pass full sequence
            decoder_hidden_states = ttnn_decoder(
                decoder_input_values=output_sequence_ttnn,
                encoder_hidden_states=encoder_output_for_decoder,
                speaker_embeddings=ttnn_speaker_embeddings,
            )

        decoder_time = time.time() - decoder_start
        total_decoder_time += decoder_time

        # Postnet (not traced - runs only once per step, less benefit from tracing)
        postnet_start = time.time()
        mel_before, mel_after, stop_logits = ttnn_postnet(decoder_hidden_states)

        postnet_time = time.time() - postnet_start
        total_postnet_time += postnet_time

        # Capture TTFT after step 0's postnet (first mel frame produced)
        if step == 0 and ttft is None:
            ttft = time.time() - generation_start

        # Check stopping condition (fully device-side comparison)
        sigmoid_logits = ttnn.sigmoid(stop_logits, memory_config=ttnn.L1_MEMORY_CONFIG)
        sum_prob = ttnn.sum(sigmoid_logits, dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)
        should_stop = ttnn.ge(sum_prob, 0.5, memory_config=ttnn.L1_MEMORY_CONFIG)
        any_stop_scalar = ttnn.sum(should_stop)
        if ttnn.to_torch(any_stop_scalar).item() > 0:
            break

        # Extract new mel frames
        if use_kv_cache and kv_cache is not None:
            # In KV cache mode, mel_after has shape [batch, reduction_factor, mel_bins]
            # since we only processed 1 input frame
            mel_after_shape = mel_after.shape
            if len(mel_after_shape) == 4:
                # Handle 4D output [B, 1, S, mel_bins]
                mel_frames = mel_after_shape[2]
            else:
                mel_frames = mel_after_shape[1]

            # Get both frames from the output (reduction_factor=2)
            new_frames_ttnn = mel_after
            if len(mel_after_shape) == 4:
                # Squeeze if 4D
                new_frames_ttnn = ttnn.reshape(mel_after, [batch_size, mel_frames, num_mel_bins])

            # Build spectrogram - use CPU accumulation when trace is enabled to avoid
            # device allocations that can corrupt trace memory
            if enable_trace and generator is not None:
                # Transfer to CPU immediately to avoid device allocations during trace
                spectrogram_frames_cpu.append(ttnn.to_torch(new_frames_ttnn).clone())
            else:
                # No trace: use device accumulation
                if spectrogram_ttnn is None:
                    spectrogram_ttnn = new_frames_ttnn
                else:
                    spectrogram_ttnn = ttnn.concat(
                        [spectrogram_ttnn, new_frames_ttnn], dim=1, memory_config=ttnn.L1_MEMORY_CONFIG
                    )

            # Get the last frame for next iteration input
            last_frame_ttnn = ttnn.slice(
                new_frames_ttnn,
                [0, mel_frames - 1, 0],
                [batch_size, mel_frames, num_mel_bins],
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            current_input_ttnn = last_frame_ttnn

            # Track sequence length
            current_seq_len += 1
        else:
            # Standard mode: extract frames from full mel_after
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

        if (step + 1) % 20 == 0:
            print(" ✓", flush=True)

    # End decoder loop timing
    decoder_loop_time = time.time() - decoder_loop_start

    # Transfer final spectrogram from device to host
    # For trace mode, we accumulated on CPU to avoid device allocations
    if enable_trace and generator is not None and spectrogram_frames_cpu:
        # Concatenate CPU frames (Whisper pattern - all accumulation on CPU)
        final_spectrogram = torch.cat(spectrogram_frames_cpu, dim=1)
    elif spectrogram_ttnn is not None:
        final_spectrogram = ttnn.to_torch(spectrogram_ttnn)
    else:
        final_spectrogram = torch.zeros(batch_size, 1, num_mel_bins)

    # Generate audio (skip in warmup mode)
    if not warmup_mode:
        print("\n🎵 Generating final audio...")
        speech = vocoder(final_spectrogram)
    else:
        speech = torch.zeros(1, 16000)

    # Cleanup TTNN tensors
    ttnn.deallocate(ttnn_input_ids)
    ttnn.deallocate(ttnn_speaker_embeddings)
    ttnn.deallocate(encoder_output)
    ttnn.deallocate(output_sequence_ttnn)
    if spectrogram_ttnn is not None:
        ttnn.deallocate(spectrogram_ttnn)

    if return_stats:
        # Calculate performance metrics
        # TTFT was captured after step 0's postnet (first mel frame produced)
        # If no steps completed, fall back to encoder_time
        if ttft is None:
            ttft = encoder_time
        avg_token_time = decoder_loop_time / max(steps_completed, 1) if steps_completed > 0 else 0
        token_per_sec = 1.0 / avg_token_time if avg_token_time > 0 else 0

        stats = {
            "steps_completed": steps_completed,
            "final_seq_len": current_seq_len,
            "ttft": ttft,
            "avg_token_time": avg_token_time,
            "token_per_sec": token_per_sec,
            "encoder_time": encoder_time,
            "decoder_loop_time": decoder_loop_time,
            "total_decoder_time": total_decoder_time,
            "total_postnet_time": total_postnet_time,
        }
        return speech, stats
    else:
        return speech


def run_demo(
    texts,
    output_dir=".",
    max_steps=100,
    use_kv_cache=True,
    speaker_id=0,
    mesh_device=None,
    model_location_generator=None,
):
    """Core demo function that can be called from pytest or main."""

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Enable persistent kernel cache for faster subsequent runs
        ttnn.device.EnablePersistentKernelCache()

        # Initialize device - use mesh_device if provided, otherwise default to single device
        if mesh_device is not None:
            device = mesh_device
            # For multi-device, get the first device
            if hasattr(mesh_device, "get_devices"):
                actual_device = mesh_device.get_devices()[0]
            else:
                actual_device = mesh_device
        else:
            # Default single device initialization
            device = ttnn.open_device(device_id=0, l1_small_size=300000, trace_region_size=10000000)
            actual_device = device

        # Enable program cache for faster inference
        if hasattr(device, "enable_program_cache"):
            device.enable_program_cache()
        elif hasattr(actual_device, "enable_program_cache"):
            actual_device.enable_program_cache()

        # Load models - use CIv2 cache if available
        is_ci_v2 = os.environ.get("TT_GH_CI_INFRA") is not None

        if is_ci_v2 and model_location_generator is not None:
            # CIv2: Use model_location_generator for cached models
            print("🔄 Loading models from CIv2 cache...")

            # SpeechT5 TTS model components
            tts_model_path = model_location_generator(
                "microsoft/speecht5_tts", model_subdir="", download_if_ci_v2=True, ci_v2_timeout_in_s=900
            )

            # HiFi-GAN vocoder
            vocoder_path = model_location_generator(
                "microsoft/speecht5_hifigan", model_subdir="", download_if_ci_v2=True, ci_v2_timeout_in_s=900
            )

            # Speaker embeddings dataset
            embeddings_path = model_location_generator(
                "microsoft/speecht5_tts/cmu-arctic-xvectors",
                model_subdir="",
                download_if_ci_v2=True,
                ci_v2_timeout_in_s=900,
            )

            # Load from local paths
            processor = SpeechT5Processor.from_pretrained(str(tts_model_path))
            model = SpeechT5ForTextToSpeech.from_pretrained(str(tts_model_path))
            vocoder = SpeechT5HifiGan.from_pretrained(str(vocoder_path))

            # Load speaker embeddings from local dataset
            embeddings_dataset = load_dataset(str(embeddings_path / "cmu-arctic-xvectors"), split="validation")
            speaker_embeddings = torch.tensor(embeddings_dataset[speaker_id]["xvector"]).unsqueeze(0)

        else:
            # Local/HuggingFace: Load directly from HuggingFace
            print("🔄 Loading models from HuggingFace...")
            processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
            model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
            vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

            # Load speaker embeddings from HuggingFace
            embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
            speaker_embeddings = torch.tensor(embeddings_dataset[speaker_id]["xvector"]).unsqueeze(0)

        # Initialize TTNN models

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
            actual_device,
            preprocess_decoder_parameters(model.speecht5.decoder, decoder_config, actual_device, speaker_embeddings),
            decoder_config,
            max_sequence_length=max_steps,  # Pass the max_steps value
        )

        ttnn_postnet = TTNNSpeechT5SpeechDecoderPostnet(
            actual_device,
            preprocess_postnet_parameters(model.speech_decoder_postnet, postnet_config, actual_device),
            postnet_config,
        )

        # Create generator wrapper for trace support (auto-enabled when use_kv_cache=True)
        # Generator is only created when KV cache is enabled for trace support
        generator = None
        if use_kv_cache:
            # Estimate encoder sequence length based on typical text length
            # Most texts will have < 100 tokens after processing
            estimated_encoder_seq_len = 128
            generator = SpeechT5Generator(
                encoder=ttnn_encoder,
                decoder=ttnn_decoder,
                postnet=ttnn_postnet,
                device=actual_device,
                decoder_config=decoder_config,
                max_steps=max_steps,
                max_batch_size=1,
                encoder_seq_len=estimated_encoder_seq_len,
            )

        # CRITICAL: Pre-compile postnet BEFORE any trace capture
        # This prevents conv2d kernel recompilation while trace is active (causes hangs)
        print("🔧 Pre-compiling postnet kernels...")
        dummy_decoder_output = ttnn.from_torch(
            torch.randn(1, 1, 1, decoder_config.hidden_size),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=actual_device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        _ = ttnn_postnet(dummy_decoder_output)
        ttnn.deallocate(dummy_decoder_output)
        print("   Postnet kernels compiled!")

        # Warm-up phase to compile TTNN operations (separate from timing)
        import time

        warmup_start_time = time.time()
        if use_kv_cache:
            print("🔥 Warming up TTNN operations with KV cache...")
            print("   This may take ~30-60 seconds as TTNN compiles kernels for optimal performance")
            print("   KV cache will enable faster autoregressive generation")
        else:
            print("🔥 Warming up TTNN operations...")
            print("   This may take ~30-45 seconds as TTNN compiles kernels for optimal performance")
            print("   TTNN will optimize operations for the decoder, postnet, and memory management")

        # Use the first input text for warm-up to ensure encoder processes the actual input
        warmup_text = texts[0]
        warmup_speech = generate_speech_ttnn(
            warmup_text,
            speaker_embeddings,
            processor,
            vocoder,
            ttnn_encoder,
            ttnn_decoder,
            ttnn_postnet,
            actual_device,
            max_steps=max_steps,
            warmup_mode=True,
            generator=generator,
            use_kv_cache=use_kv_cache,
            decoder_config=decoder_config,
        )
        warmup_duration = time.time() - warmup_start_time
        print(f"✅ Initial warm-up completed in {warmup_duration:.1f}s (generated {len(warmup_speech)} samples)")
        if use_kv_cache:
            print("   KV cache enabled - decoder reuses attention computations!")
        print("   TTNN kernels are now compiled - subsequent inference will be faster!")

        # Capture traces for ALL supported encoder sizes during warm-up
        # This ensures any input length will have a matching trace ready
        if generator is not None:
            from models.experimental.speecht5_tts.tt.ttnn_speecht5_generator import SUPPORTED_ENCODER_SEQ_LENS

            print(f"\n🔧 Capturing traces for all encoder sizes: {SUPPORTED_ENCODER_SEQ_LENS}")
            all_traces_start = time.time()
            generator.capture_all_traces(processor, batch_size=1)
            all_traces_duration = time.time() - all_traces_start
            print(f"✅ All traces captured in {all_traces_duration:.1f}s")

            # Report trace status
            compiled_sizes = [s for s in SUPPORTED_ENCODER_SEQ_LENS if generator.trace_compiled_per_size.get(s, False)]
            print(f"   Compiled traces for encoder sizes: {compiled_sizes}")

        # Keep traces after warm-up - trace captures decoder ops which are input-independent
        # Only reset KV caches to clear warm-up values, traces will be reused
        # Following simple_text_demo pattern: trace capture is "compile time", not inference time
        if generator is not None:
            # CRITICAL: Reset KV caches after warm-up to prevent stale values from corrupting inference
            generator._reset_kv_caches()
            print("   KV caches reset for fresh inference - ready for any input length!")

        # Generate speech for each input text
        results = []
        for i, text in enumerate(texts, 1):
            # Generate filename from text (sanitize for filesystem)
            safe_text = "".join(c for c in text[:50] if c.isalnum() or c in (" ", "-", "_")).rstrip()
            if not safe_text:
                safe_text = f"speech_{i}"
            safe_text = safe_text.replace(" ", "_")
            output_file = os.path.join(output_dir, f"speech_ttnn_{safe_text}.wav")

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
                max_steps=max_steps,
                return_stats=True,
                generator=generator,
                use_kv_cache=use_kv_cache,
                decoder_config=decoder_config,
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


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 300000, "trace_region_size": 10000000}],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {
            "N150": (1, 1),
            "N300": (1, 2),
            "T3K": (1, 8),
            "TG": (8, 4),
            "P150": (1, 1),
            "P300": (1, 2),
        }.get(os.environ.get("MESH_DEVICE"), (1, 1))
    ],
    indirect=True,
)
def test_demo(mesh_device, device_params, model_location_generator, request):
    """
    Pytest version of the demo for automated testing.

    Command-line options (defined in conftest.py):
        --input-text: Custom input text (default: "Hello world...")
        --max-steps: Override max generation steps (default: 24)
        --output-dir: Directory to save output audio (default: pytest_output)

    Usage:
        MESH_DEVICE=N150 pytest models/experimental/speecht5_tts/demo_ttnn.py::test_demo -v
        MESH_DEVICE=N150 pytest models/experimental/speecht5_tts/demo_ttnn.py::test_demo -v \
            --input-text "Hello world" --max-steps 500 --output-dir ./output
    """
    # Get command-line overrides
    input_text = (
        request.config.getoption("--input-text")
        or "Hello world, this is a test of text to speech synthesis on Tenstorrent hardware."
    )
    max_steps = request.config.getoption("--max-steps") or 24
    output_dir = request.config.getoption("--output-dir") or "pytest_output"

    run_demo(
        texts=[input_text],
        output_dir=output_dir,
        max_steps=max_steps,
        use_kv_cache=True,  # KV cache enabled by default (also auto-enables trace)
        speaker_id=0,
        mesh_device=mesh_device,
        model_location_generator=model_location_generator,
    )


def main(mesh_device=None):
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
        "--use_kv_cache",
        action="store_true",
        default=True,
        help="Use KV cache for faster autoregressive generation (default: True, also auto-enables trace)",
    )
    parser.add_argument(
        "--no_kv_cache",
        action="store_true",
        help="Disable KV cache (and trace)",
    )
    parser.add_argument("--speaker_id", type=int, default=0, help="Speaker ID from CMU ARCTIC dataset (0-7456)")

    args = parser.parse_args()

    # Handle no_kv_cache flag
    use_kv_cache = not args.no_kv_cache

    # Call the core demo function
    # For main(), we don't have model_location_generator, so pass None
    # The run_demo function will handle CIv2 detection internally
    return run_demo(
        texts=args.texts,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        use_kv_cache=use_kv_cache,  # KV cache enabled by default, also auto-enables trace
        speaker_id=args.speaker_id,
        mesh_device=mesh_device,
        model_location_generator=None,
    )


if __name__ == "__main__":
    exit(main())
