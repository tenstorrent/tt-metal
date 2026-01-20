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
- Long text support via automatic chunking (texts > 300 chars are split at sentence boundaries)
"""

import torch
import ttnn
import soundfile as sf
import pytest
import os
import time
import re
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset


# Default chunk size for long text processing (characters)
DEFAULT_CHUNK_SIZE = 300


def chunk_text(text, max_chunk_size=DEFAULT_CHUNK_SIZE):
    """
    Split long text into smaller chunks at sentence/word boundaries.

    Args:
        text: Input text string
        max_chunk_size: Maximum characters per chunk (default: 300)

    Returns:
        List of text chunks
    """
    # If text is short enough, return as single chunk
    if len(text) <= max_chunk_size:
        return [text]

    chunks = []
    remaining = text.strip()

    while remaining:
        if len(remaining) <= max_chunk_size:
            chunks.append(remaining)
            break

        # Try to find a sentence boundary within the chunk size
        # Look for sentence-ending punctuation followed by space
        chunk_candidate = remaining[:max_chunk_size]

        # Find the last sentence boundary (. ! ?) within the chunk
        sentence_end = -1
        for match in re.finditer(r"[.!?]\s+", chunk_candidate):
            sentence_end = match.end()

        if sentence_end > max_chunk_size // 3:
            # Found a good sentence boundary (at least 1/3 into the chunk)
            chunk = remaining[:sentence_end].strip()
            remaining = remaining[sentence_end:].strip()
        else:
            # No good sentence boundary, try comma or semicolon
            clause_end = -1
            for match in re.finditer(r"[,;]\s+", chunk_candidate):
                clause_end = match.end()

            if clause_end > max_chunk_size // 2:
                # Found a clause boundary (at least halfway)
                chunk = remaining[:clause_end].strip()
                remaining = remaining[clause_end:].strip()
            else:
                # Fall back to word boundary
                last_space = chunk_candidate.rfind(" ")
                if last_space > max_chunk_size // 2:
                    chunk = remaining[:last_space].strip()
                    remaining = remaining[last_space:].strip()
                else:
                    # No good boundary, force split at max_chunk_size
                    chunk = remaining[:max_chunk_size].strip()
                    remaining = remaining[max_chunk_size:].strip()

        if chunk:
            chunks.append(chunk)

    return chunks


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
    return_mel_only=False,
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
        return_mel_only: If True, return mel spectrogram instead of audio (for chunked processing)

    Returns:
        torch.Tensor: Generated audio waveform (if return_stats=False and return_mel_only=False)
        torch.Tensor: Mel spectrogram (if return_mel_only=True)
        tuple: (speech/mel, stats_dict) if return_stats=True
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

    # 2CQ: Events for overlapping position updates with CPU work
    # After postnet, we async update position for next iteration on CQ1
    # while CPU does stop check and mel extraction
    use_2cq = enable_trace and generator is not None
    op_event = None  # Signals CQ0 work (postnet) is done
    pos_event = None  # Signals CQ1 work (position update) is done

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

                # 2CQ: Wait for async position update from previous iteration
                if use_2cq and pos_event is not None:
                    ttnn.wait_for_event(0, pos_event)  # CQ0 waits for CQ1
                    pos_event = None
                else:
                    # Step 0 or non-2CQ: update position synchronously
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
                    # Step 1+: Execute trace (non-blocking for 2CQ overlap)
                    decoder_hidden_states = generator._execute_decoder_trace(preprocessed_hidden_states, blocking=False)
                    # Sync: create a copy to ensure trace output is ready for postnet
                    # This is required because postnet needs valid data from the trace
                    decoder_hidden_states = ttnn.to_memory_config(
                        decoder_hidden_states,
                        ttnn.L1_MEMORY_CONFIG,
                    )
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
        # Note: In 2CQ mode, postnet naturally waits for trace since both are on CQ0
        postnet_start = time.time()
        mel_before, mel_after, stop_logits = ttnn_postnet(decoder_hidden_states)

        postnet_time = time.time() - postnet_start
        total_postnet_time += postnet_time

        # 2CQ: Start async position update for next iteration on CQ1
        # This overlaps with the CPU-bound stop check and mel extraction below
        if use_2cq and step < max_steps - 1:
            op_event = ttnn.record_event(device, 0)  # Record on CQ0
            ttnn.wait_for_event(1, op_event)  # CQ1 waits for CQ0
            generator._reset_decode_pos(step + 1, batch_size, cq_id=1)  # Async on CQ1
            pos_event = ttnn.record_event(device, 1)  # Record on CQ1

        # Capture TTFT after step 0's postnet (first mel frame produced)
        if step == 0 and ttft is None:
            ttft = time.time() - generation_start

        # Check stopping condition
        # In KV cache mode: stop_logits is [batch, reduction_factor] (e.g., [1, 2])
        # In standard mode: stop_logits is [batch, total_mel_frames], need last reduction_factor
        # Reference uses: sigmoid(prob_out(hidden)).sum(dim=-1) >= threshold
        min_steps = 10  # Don't stop too early (following HF's minlen pattern)
        stop_threshold = 0.5  # Per-frame threshold (sum of 2 probs >= 0.5 means average >= 0.25)

        if step >= min_steps:
            # For KV cache mode (seq_len=1), stop_logits is already [batch, 2]
            # For standard mode, we need to slice the last reduction_factor frames
            stop_logits_shape = stop_logits.shape
            if use_kv_cache and kv_cache is not None:
                # KV cache mode: stop_logits is [batch, reduction_factor]
                current_stop_logits = stop_logits
            else:
                # Standard mode: slice last reduction_factor frames
                reduction_factor = 2
                total_mel_frames = stop_logits_shape[-1]
                current_stop_logits = ttnn.slice(
                    stop_logits,
                    [0, total_mel_frames - reduction_factor],
                    [batch_size, total_mel_frames],
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )

            sigmoid_logits = ttnn.sigmoid(current_stop_logits, memory_config=ttnn.L1_MEMORY_CONFIG)
            sum_prob = ttnn.sum(sigmoid_logits, dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)
            should_stop = ttnn.ge(sum_prob, stop_threshold, memory_config=ttnn.L1_MEMORY_CONFIG)
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

    # If returning mel only (for chunked processing), skip vocoder
    if return_mel_only:
        # Cleanup TTNN tensors
        ttnn.deallocate(ttnn_input_ids)
        ttnn.deallocate(ttnn_speaker_embeddings)
        ttnn.deallocate(encoder_output)
        ttnn.deallocate(output_sequence_ttnn)
        if spectrogram_ttnn is not None:
            ttnn.deallocate(spectrogram_ttnn)

        if return_stats:
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
            return final_spectrogram, stats
        return final_spectrogram

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


def generate_speech_long_text(
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
    max_chunk_size=DEFAULT_CHUNK_SIZE,
):
    """
    Generate speech for long text by chunking into smaller pieces.

    This function handles texts longer than the embedding limit by:
    1. Splitting text into chunks at sentence/word boundaries
    2. Processing each chunk to generate mel spectrograms
    3. Concatenating all mel spectrograms
    4. Running vocoder once on the combined spectrogram

    Args:
        text: Input text string (can be arbitrarily long)
        speaker_embeddings: Speaker embedding tensor
        processor: SpeechT5Processor
        vocoder: SpeechT5HifiGan
        ttnn_encoder: TTNN encoder model
        ttnn_decoder: TTNN decoder model
        ttnn_postnet: TTNN postnet model
        device: TTNN device
        max_steps: Maximum number of generation steps per chunk (default: 100)
        return_stats: If True, return statistics along with speech (default: False)
        warmup_mode: If True, skip vocoder and detailed timing for faster warm-up (default: False)
        generator: SpeechT5Generator instance for trace support
        use_kv_cache: If True, use KV cache for faster autoregressive generation
        decoder_config: TTNNDecoderConfig (required if use_kv_cache=True)
        max_chunk_size: Maximum characters per chunk (default: 300)

    Returns:
        torch.Tensor: Generated audio waveform (if return_stats=False)
        tuple: (speech, stats_dict) if return_stats=True
    """
    # Split text into chunks
    chunks = chunk_text(text, max_chunk_size)
    num_chunks = len(chunks)

    if num_chunks == 1:
        # Single chunk - use standard generation
        return generate_speech_ttnn(
            text=text,
            speaker_embeddings=speaker_embeddings,
            processor=processor,
            vocoder=vocoder,
            ttnn_encoder=ttnn_encoder,
            ttnn_decoder=ttnn_decoder,
            ttnn_postnet=ttnn_postnet,
            device=device,
            max_steps=max_steps,
            return_stats=return_stats,
            warmup_mode=warmup_mode,
            generator=generator,
            use_kv_cache=use_kv_cache,
            decoder_config=decoder_config,
        )

    print(f"\n📝 Long text detected ({len(text)} chars). Splitting into {num_chunks} chunks...")

    # Process each chunk and collect mel spectrograms
    mel_spectrograms = []
    total_stats = {
        "steps_completed": 0,
        "final_seq_len": 0,
        "ttft": 0,
        "encoder_time": 0,
        "decoder_loop_time": 0,
        "total_decoder_time": 0,
        "total_postnet_time": 0,
        "num_chunks": num_chunks,
        "chunk_stats": [],
    }

    generation_start = time.time()

    for i, chunk in enumerate(chunks):
        print(f"\n   Chunk {i+1}/{num_chunks}: '{chunk[:50]}{'...' if len(chunk) > 50 else ''}'")

        # Reset KV caches between chunks to start fresh
        if generator is not None:
            generator._reset_kv_caches()

        # Generate mel spectrogram for this chunk
        mel, chunk_stats = generate_speech_ttnn(
            text=chunk,
            speaker_embeddings=speaker_embeddings,
            processor=processor,
            vocoder=vocoder,
            ttnn_encoder=ttnn_encoder,
            ttnn_decoder=ttnn_decoder,
            ttnn_postnet=ttnn_postnet,
            device=device,
            max_steps=max_steps,
            return_stats=True,
            warmup_mode=warmup_mode,
            generator=generator,
            use_kv_cache=use_kv_cache,
            decoder_config=decoder_config,
            return_mel_only=True,  # Return mel instead of audio
        )

        mel_spectrograms.append(mel)

        # Accumulate stats
        total_stats["steps_completed"] += chunk_stats.get("steps_completed", 0)
        total_stats["final_seq_len"] += chunk_stats.get("final_seq_len", 0)
        total_stats["encoder_time"] += chunk_stats.get("encoder_time", 0)
        total_stats["decoder_loop_time"] += chunk_stats.get("decoder_loop_time", 0)
        total_stats["total_decoder_time"] += chunk_stats.get("total_decoder_time", 0)
        total_stats["total_postnet_time"] += chunk_stats.get("total_postnet_time", 0)

        # TTFT is from first chunk only
        if i == 0:
            total_stats["ttft"] = chunk_stats.get("ttft", 0)

        total_stats["chunk_stats"].append(
            {
                "chunk_index": i,
                "chunk_text": chunk[:50],
                "steps": chunk_stats.get("steps_completed", 0),
                "mel_frames": mel.shape[1] if mel is not None else 0,
            }
        )

    # Concatenate all mel spectrograms
    print(f"\n🔗 Concatenating {num_chunks} mel spectrograms...")
    combined_mel = torch.cat(mel_spectrograms, dim=1)
    print(f"   Combined mel shape: {combined_mel.shape}")

    # Generate audio from combined mel spectrogram
    if not warmup_mode:
        print("\n🎵 Generating final audio from combined spectrogram...")
        speech = vocoder(combined_mel)
    else:
        speech = torch.zeros(1, 16000)

    total_generation_time = time.time() - generation_start

    # Calculate aggregate stats
    total_stats["avg_token_time"] = (
        total_stats["decoder_loop_time"] / max(total_stats["steps_completed"], 1)
        if total_stats["steps_completed"] > 0
        else 0
    )
    total_stats["token_per_sec"] = 1.0 / total_stats["avg_token_time"] if total_stats["avg_token_time"] > 0 else 0
    total_stats["total_generation_time"] = total_generation_time

    if return_stats:
        return speech, total_stats
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
    max_chunk_size=DEFAULT_CHUNK_SIZE,
):
    """Core demo function that can be called from pytest or main.

    Args:
        texts: List of text strings to convert to speech
        output_dir: Directory to save output audio files
        max_steps: Maximum generation steps per chunk
        use_kv_cache: Enable KV cache for faster generation
        speaker_id: Speaker ID from CMU ARCTIC dataset
        mesh_device: TTNN mesh device (optional)
        model_location_generator: Model location generator for CIv2 (optional)
        max_chunk_size: Maximum characters per chunk for long texts (default: 300)
    """

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Initialize device - use mesh_device if provided, otherwise default to single device
        if mesh_device is not None:
            device = mesh_device
            # For multi-device, get the first device
            if hasattr(mesh_device, "get_devices"):
                actual_device = mesh_device.get_devices()[0]
            else:
                actual_device = mesh_device
        else:
            # Default single device initialization with 2 command queues
            device = ttnn.open_device(
                device_id=0, l1_small_size=300000, trace_region_size=10000000, num_command_queues=2
            )
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

        # Use the first chunk of the first input text for warm-up
        # This ensures warm-up doesn't OOM on long texts while still compiling relevant kernels
        warmup_chunks = chunk_text(texts[0], max_chunk_size)
        warmup_text = warmup_chunks[0]  # Use only the first chunk for warm-up
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
            # Use generate_speech_long_text which handles chunking for long texts
            generation_start = time.time()
            speech, generation_stats = generate_speech_long_text(
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
                max_chunk_size=max_chunk_size,
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
                "num_chunks": generation_stats.get("num_chunks", 1),
                "text_length": len(text),
            }
            results.append(result)

        # Display summary table
        print("\n" + "=" * 140)
        print("📊 INFERENCE SUMMARY")
        print("=" * 140)
        print(
            f"{'#':<4} {'Text':<25} {'Chars':<7} {'Chunks':<7} {'Tokens':<8} {'TTFT(ms)':<9} {'Token/s':<10} {'Time(s)':<10} {'Audio(s)':<8}"
        )
        print("-" * 140)

        total_generation_time = 0
        total_tokens = 0
        total_audio_duration = 0
        total_chunks = 0

        for i, result in enumerate(results, 1):
            truncated_text = result["text"][:20] + "..." if len(result["text"]) > 23 else result["text"]
            num_chunks = result.get("num_chunks", 1)
            text_length = result.get("text_length", len(result["text"]))
            print(
                f"{i:<4} {truncated_text:<25} {text_length:<7} {num_chunks:<7} {result['tokens_generated']:<8} {result.get('ttft', 0)*1000:<8.0f} {result.get('token_per_sec', 0):<10.2f} {result['generation_time']:<10.3f} {result['audio_duration']:<8.1f}"
            )
            total_generation_time += result["generation_time"]
            total_tokens += result["tokens_generated"]
            total_audio_duration += result["audio_duration"]
            total_chunks += num_chunks

        print("-" * 140)
        print(
            f"{'TOTAL':<4} {'':<25} {'-':<7} {total_chunks:<7} {total_tokens:<8} {'-':<9} {'-':<10} {total_generation_time:<10.3f} {total_audio_duration:<8.1f}"
        )

        print(f"\n📈 Overall Statistics:")
        print(f"   • Total inference time: {total_generation_time:.3f}s")
        print(f"   • Total tokens generated: {total_tokens}")
        print(f"   • Total chunks processed: {total_chunks}")
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
    [{"l1_small_size": 300000, "trace_region_size": 10000000, "num_command_queues": 2}],
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
    parser.add_argument(
        "--max_chunk_size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Maximum characters per chunk for long text processing (default: {DEFAULT_CHUNK_SIZE})",
    )

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
        max_chunk_size=args.max_chunk_size,
        mesh_device=mesh_device,
        model_location_generator=None,
    )


if __name__ == "__main__":
    exit(main())
