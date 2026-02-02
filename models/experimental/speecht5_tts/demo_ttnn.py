#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTNN SpeechT5 Demo - Hybrid FP32+BF16 SDPA Edition with Trace Support

This demo uses a hybrid approach for optimal precision and performance:
- FP32 precision for most operations (prenet, FFN, layer norms, postnet)
- BF16 SDPA for attention (trace-compatible, high performance)

This combines the benefits of both:
- High precision from FP32 in critical paths
- Trace support and performance from BF16 SDPA
- Compatible with long sequences

Features:
- Hybrid FP32+BF16 precision
- SDPA-based attention (trace-compatible)
- Full TTNN implementation (Encoder + Decoder + Postnet)
- L1 memory optimizations
- KV cache support for autoregressive generation
- Trace capture and execution for maximum performance
- 2-command-queue (2CQ) support for async operations
- Warm-up phase for kernel compilation
- Long text support via automatic chunking (texts > 300 chars split at sentence boundaries)

Usage:
    # Basic usage with short text:
    python models/experimental/speecht5_tts/demo_ttnn_fp32.py \\
        --text "Hello world!" \\
        --output speech.wav

    # Long text (automatic chunking):
    python models/experimental/speecht5_tts/demo_ttnn_fp32.py \\
        --text "Very long text here..." \\
        --output speech.wav \\
        --max_chunk_size 300

    # Multiple texts:
    python models/experimental/speecht5_tts/demo_ttnn_fp32.py \\
        --texts "First text." "Second text." \\
        --output_dir ./outputs

    # Custom max steps per chunk:
    python models/experimental/speecht5_tts/demo_ttnn_fp32.py \\
        --text "Your text" \\
        --output speech.wav \\
        --max_steps 500
"""

import torch
import ttnn
import soundfile as sf
import os
import time
import re
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset

DEFAULT_CHUNK_SIZE = 300


def chunk_text(text, max_chunk_size=DEFAULT_CHUNK_SIZE):
    """Split long text into smaller chunks at sentence/word boundaries."""
    if len(text) <= max_chunk_size:
        return [text]

    chunks = []
    remaining = text.strip()

    while remaining:
        if len(remaining) <= max_chunk_size:
            chunks.append(remaining)
            break

        chunk_candidate = remaining[:max_chunk_size]
        sentence_end = -1
        for match in re.finditer(r"[.!?]\s+", chunk_candidate):
            sentence_end = match.end()

        if sentence_end > max_chunk_size // 3:
            chunk = remaining[:sentence_end].strip()
            remaining = remaining[sentence_end:].strip()
        else:
            clause_end = -1
            for match in re.finditer(r"[,;]\s+", chunk_candidate):
                clause_end = match.end()

            if clause_end > max_chunk_size // 2:
                chunk = remaining[:clause_end].strip()
                remaining = remaining[clause_end:].strip()
            else:
                last_space = chunk_candidate.rfind(" ")
                if last_space > max_chunk_size // 2:
                    chunk = remaining[:last_space].strip()
                    remaining = remaining[last_space:].strip()
                else:
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
from models.experimental.speecht5_tts.tt.ttnn_speecht5_decoder_manual_attn import (
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
from models.experimental.speecht5_tts.tt.ttnn_speecht5_generator import (
    SpeechT5Generator,
    SUPPORTED_ENCODER_SEQ_LENS,
)


def generate_speech_fp32(
    text,
    speaker_embeddings,
    processor,
    vocoder,
    ttnn_encoder,
    ttnn_decoder,
    ttnn_postnet,
    device,
    max_steps=100,
    use_kv_cache=True,
    decoder_config=None,
    return_stats=False,
    warmup_mode=False,
    generator=None,
    return_mel_only=False,
):
    """
    Generate speech using FP32 TTNN SpeechT5 pipeline.

    Args:
        text: Input text string
        speaker_embeddings: Speaker embedding tensor
        processor: SpeechT5Processor
        vocoder: SpeechT5HifiGan
        ttnn_encoder: TTNN encoder model
        ttnn_decoder: TTNN FP32 decoder model
        ttnn_postnet: TTNN postnet model
        device: TTNN device
        max_steps: Maximum generation steps (default: 100)
        use_kv_cache: Use KV cache for faster generation
        decoder_config: TTNNDecoderConfig (required if use_kv_cache=True)
        return_stats: If True, return (speech, stats) tuple
        warmup_mode: If True, skip vocoder and detailed timing for faster warm-up
        generator: SpeechT5Generator instance for trace support (enables trace when use_kv_cache=True)

    Returns:
        torch.Tensor: Generated audio waveform
        tuple: (speech, stats) if return_stats=True
    """

    # Start timing
    generation_start = time.time()

    # Process input
    inputs = processor(text=text, return_tensors="pt")
    token_ids = inputs["input_ids"]

    # Convert to TTNN
    ttnn_input_ids = ttnn.from_torch(token_ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    # Trace enabled when KV cache is enabled and generator is provided
    enable_trace = use_kv_cache and generator is not None
    batch_size = 1

    # Encoder forward pass
    if not warmup_mode:
        print("🔄 Encoding text...")
    encoder_start = time.time()
    encoder_output = ttnn_encoder(ttnn_input_ids)[0]
    encoder_time = time.time() - encoder_start
    encoder_output = ttnn.unsqueeze(encoder_output, dim=1)  # Add batch dimension for decoder

    # If using trace with generator, copy encoder output to pre-allocated tensor
    if enable_trace:
        generator.copy_encoder_output(encoder_output)
        # Use generator's pre-allocated encoder_hidden_states for decoder
        encoder_output_for_decoder = generator.encoder_hidden_states
    else:
        encoder_output_for_decoder = encoder_output

    # Initialize KV cache if enabled
    if use_kv_cache:
        encoder_seq_len = token_ids.shape[1]
        if enable_trace and generator is not None:
            # Use generator's pre-allocated KV cache for trace stability
            kv_cache = generator.kv_cache
            cross_attn_cache = generator.cross_attn_cache
        else:
            # Regular KV cache (no trace)
            kv_cache, cross_attn_cache = init_kv_cache(
                decoder_config, device, batch_size, max_steps + 10, encoder_seq_len
            )
    else:
        kv_cache = None
        cross_attn_cache = None

    # Autoregressive generation
    decoder_input = torch.zeros(1, 1, decoder_config.num_mel_bins)
    all_mel_outputs = []
    spectrogram_frames_cpu = []  # For trace mode: accumulate on CPU to avoid device allocations

    # Performance tracking
    steps_completed = 0
    total_decoder_time = 0.0
    total_postnet_time = 0.0
    ttft = None

    # 2CQ support: use 2 command queues when trace is enabled
    use_2cq = enable_trace and generator is not None
    pos_event = None

    if not warmup_mode:
        print("🎙️  Generating speech autoregressively...")
    elif warmup_mode:
        print(f"   Warm-up: Running {max_steps} steps...", end="", flush=True)

    decoder_loop_start = time.time()

    for step in range(max_steps):
        if not warmup_mode and ((step + 1) % 20 == 0 or step == 0):
            print(f"   Step {step+1}/{max_steps}", end="", flush=True)

        # Decoder step
        decoder_start = time.time()

        # Convert current input to TTNN (FP32)
        ttnn_decoder_input = ttnn.from_torch(decoder_input, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

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
                    decoder_input_values=ttnn_decoder_input,
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
                    decoder_hidden_states = ttnn.to_memory_config(
                        decoder_hidden_states,
                        ttnn.L1_MEMORY_CONFIG,
                    )
            else:
                # KV cache without trace
                current_pos_tensor = ttnn.full(
                    (1, 1),
                    step,
                    dtype=ttnn.int32,
                    device=device,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )

                decoder_hidden_states = ttnn_decoder(
                    decoder_input_values=ttnn_decoder_input,
                    encoder_hidden_states=encoder_output_for_decoder,
                    kv_cache=kv_cache,
                    cross_attn_cache=cross_attn_cache,
                    cross_attn_cache_valid=(step > 0),
                    current_decode_pos=current_pos_tensor,
                    position_offset=step,
                )
        else:
            # Prefill mode: process full sequence (no KV cache)
            decoder_hidden_states = ttnn_decoder(
                decoder_input_values=ttnn_decoder_input,
                encoder_hidden_states=encoder_output_for_decoder,
                kv_cache=None,
                cross_attn_cache=None,
                cross_attn_cache_valid=False,
                current_decode_pos=None,
                position_offset=0,
            )

        total_decoder_time += time.time() - decoder_start

        # Postnet (not traced - runs only once per step, less benefit from tracing)
        # Note: In 2CQ mode, postnet naturally waits for trace since both are on CQ0
        postnet_start = time.time()
        outputs_before, outputs_after, stop_logits = ttnn_postnet(decoder_hidden_states)
        total_postnet_time += time.time() - postnet_start

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

        # Convert outputs to torch for processing
        # Note: mel_output shape is [reduction_factor, num_mel_bins] after squeeze
        mel_output_ttnn = ttnn.squeeze(outputs_after, 0)  # Remove batch dimension in ttnn
        stop_logits_ttnn = stop_logits  # Keep on device for stop condition check

        # Get config values needed for processing
        reduction_factor = decoder_config.reduction_factor
        num_mel_bins = decoder_config.num_mel_bins

        # For trace mode: accumulate on CPU immediately to avoid device allocations during trace
        if enable_trace and generator is not None:
            # Transfer to CPU immediately
            mel_output_torch = ttnn.to_torch(mel_output_ttnn)
            spectrogram_frames_cpu.append(mel_output_torch)
        else:
            # No trace: keep on device
            mel_output_torch = ttnn.to_torch(mel_output_ttnn)
            all_mel_outputs.append(mel_output_torch)

        # Check stop condition (same logic as demo_ttnn.py)
        min_steps = 10
        stop_threshold = 0.5
        if step >= min_steps:
            # In KV cache mode, stop_logits is already [batch, reduction_factor] for current step
            # In non-KV-cache mode, we'd need to slice, but we always use KV cache here
            if use_kv_cache:
                current_stop_logits = stop_logits_ttnn
            else:
                # Standard mode: slice last reduction_factor frames (not used in this demo)
                total_mel_frames = (step + 1) * reduction_factor
                current_stop_logits = ttnn.slice(
                    stop_logits_ttnn,
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

        # For next input, extract LAST frame using ttnn slice operation
        # mel_output_ttnn shape: [reduction_factor, num_mel_bins]
        # We want the last frame: shape [1, num_mel_bins]
        current_mel_ttnn = ttnn.slice(
            mel_output_ttnn,
            [reduction_factor - 1, 0],
            [reduction_factor, num_mel_bins],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # Convert to torch for next iteration (needed for ttnn.from_torch at line 181)
        current_mel = ttnn.to_torch(current_mel_ttnn)

        # Prepare next input
        if use_kv_cache:
            # Unsqueeze in ttnn to add batch dimension: [1, num_mel_bins] -> [1, 1, num_mel_bins]
            decoder_input_ttnn = ttnn.unsqueeze(current_mel_ttnn, 0)
            # Convert to torch for next iteration
            decoder_input = ttnn.to_torch(decoder_input_ttnn)
        else:
            decoder_input = torch.cat([decoder_input, current_mel.unsqueeze(0)], dim=1)

        steps_completed += 1

        if not warmup_mode and (step + 1) % 20 == 0:
            print(" ✓", flush=True)

    decoder_loop_time = time.time() - decoder_loop_start

    # Combine mel outputs
    # For trace mode, we accumulated on CPU to avoid device allocations
    if enable_trace and generator is not None and spectrogram_frames_cpu:
        mel_spectrogram = torch.cat(spectrogram_frames_cpu, dim=0).unsqueeze(0)
    else:
        mel_spectrogram = torch.cat(all_mel_outputs, dim=0).unsqueeze(0)

    # Generate audio with vocoder (skip if warmup_mode or return_mel_only)
    if not warmup_mode and not return_mel_only:
        print("\n🎵 Generating final audio...")
        vocoder_start = time.time()
        with torch.no_grad():
            speech = vocoder(mel_spectrogram)
        vocoder_time = time.time() - vocoder_start
    else:
        # Warmup/mel-only mode: skip vocoder
        speech = torch.zeros(mel_spectrogram.shape[1] * 256)  # Dummy audio
        vocoder_time = 0.0

    # Calculate performance metrics
    total_time = time.time() - generation_start
    if ttft is None:
        ttft = encoder_time
    avg_token_time = decoder_loop_time / max(steps_completed, 1) if steps_completed > 0 else 0
    token_per_sec = 1.0 / avg_token_time if avg_token_time > 0 else 0

    if return_stats:
        stats = {
            "steps_completed": steps_completed,
            "mel_frames": mel_spectrogram.shape[1],
            "ttft": ttft,
            "avg_token_time": avg_token_time,
            "token_per_sec": token_per_sec,
            "encoder_time": encoder_time,
            "decoder_loop_time": decoder_loop_time,
            "total_decoder_time": total_decoder_time,
            "total_postnet_time": total_postnet_time,
            "vocoder_time": vocoder_time,
            "total_time": total_time,
        }
        if return_mel_only:
            return mel_spectrogram, stats
        return speech.squeeze(), stats

    if return_mel_only:
        return mel_spectrogram
    return speech.squeeze()


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

    Handles texts longer than the embedding limit by:
    1. Splitting text into chunks at sentence/word boundaries
    2. Processing each chunk to generate mel spectrograms
    3. Concatenating all mel spectrograms
    4. Running vocoder once on the combined spectrogram

    Args:
        text: Input text string
        speaker_embeddings: Speaker embeddings tensor
        processor: HuggingFace processor
        vocoder: HiFiGAN vocoder model
        ttnn_encoder: TTNN encoder model
        ttnn_decoder: TTNN decoder model
        ttnn_postnet: TTNN postnet model
        device: TTNN device
        max_steps: Maximum generation steps per chunk
        return_stats: If True, return (speech, stats) tuple
        warmup_mode: If True, skip vocoder (warmup only)
        generator: SpeechT5Generator instance for trace support
        use_kv_cache: If True, use KV cache for faster generation
        decoder_config: TTNNDecoderConfig (required if use_kv_cache=True)
        max_chunk_size: Maximum characters per chunk

    Returns:
        torch.Tensor: Generated audio waveform (if return_stats=False)
        tuple: (speech, stats_dict) if return_stats=True
    """
    # Split text into chunks
    chunks = chunk_text(text, max_chunk_size)
    num_chunks = len(chunks)

    if num_chunks == 1:
        # Single chunk - use standard generation
        return generate_speech_fp32(
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

    # Multiple chunks - process each and concatenate
    print(f"\n📚 Long text detected ({len(text)} chars)")
    print(f"   Splitting into {num_chunks} chunks for processing...")

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

        # Reset KV caches between chunks
        if generator is not None:
            generator._reset_kv_caches()

        # Generate mel for this chunk (skip vocoder)
        mel, chunk_stats = generate_speech_fp32(
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
            return_mel_only=True,  # Return mel instead of audio
            generator=generator,
            use_kv_cache=use_kv_cache,
            decoder_config=decoder_config,
        )

        mel_spectrograms.append(mel)

        # Aggregate stats
        total_stats["steps_completed"] += chunk_stats.get("steps_completed", 0)
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
                "mel_frames": chunk_stats.get("mel_frames", 0),
            }
        )

    # Concatenate all mel spectrograms
    print(f"\n🔗 Concatenating {num_chunks} mel spectrograms...")
    # Debug: check shapes
    print(f"   Mel shapes: {[mel.shape for mel in mel_spectrograms]}")
    # mel_spectrograms should be 3D: [batch=1, seq_len, mel_bins]
    # If they're 2D [seq_len, mel_bins], add batch dimension
    if len(mel_spectrograms[0].shape) == 2:
        mel_spectrograms = [mel.unsqueeze(0) for mel in mel_spectrograms]
    # Concatenate along the sequence dimension (dim=1)
    combined_mel = torch.cat(mel_spectrograms, dim=1)
    print(f"   Combined mel shape: {combined_mel.shape}")

    total_stats["mel_frames"] = combined_mel.shape[1]
    total_stats["avg_token_time"] = total_stats["decoder_loop_time"] / max(total_stats["steps_completed"], 1)
    total_stats["token_per_sec"] = 1.0 / total_stats["avg_token_time"] if total_stats["avg_token_time"] > 0 else 0
    total_stats["total_generation_time"] = time.time() - generation_start

    if warmup_mode:
        # Warmup mode: return dummy audio
        speech = torch.zeros(combined_mel.shape[1] * 256)
        if return_stats:
            return speech, total_stats
        return speech

    # Run vocoder on combined mel
    print(f"\n🎵 Generating final audio from {num_chunks}-chunk mel...")
    vocoder_start = time.time()
    with torch.no_grad():
        speech = vocoder(combined_mel)
    total_stats["vocoder_time"] = time.time() - vocoder_start

    if return_stats:
        return speech.squeeze(), total_stats

    return speech.squeeze()


def main():
    """Main demo function."""
    import argparse

    parser = argparse.ArgumentParser(description="TTNN SpeechT5 TTS Demo - Hybrid FP32+BF16 SDPA Edition")
    parser.add_argument("--text", type=str, default=None, help="Single text to synthesize")
    parser.add_argument("--texts", type=str, nargs="+", default=None, help="Multiple texts to synthesize")
    parser.add_argument(
        "--output", type=str, default="speech_fp32.wav", help="Output audio file path (for single text)"
    )
    parser.add_argument("--output_dir", type=str, default=".", help="Output directory for multiple texts")
    parser.add_argument("--max_steps", type=int, default=200, help="Maximum generation steps per chunk")
    parser.add_argument(
        "--max_chunk_size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Maximum characters per chunk for long text (default: {DEFAULT_CHUNK_SIZE})",
    )
    args = parser.parse_args()

    # Determine which mode: single text or multiple texts
    if args.texts:
        texts = args.texts
        output_dir = args.output_dir
        single_text_mode = False
    elif args.text:
        texts = [args.text]
        output_dir = None
        single_text_mode = True
    else:
        # Default text if none provided
        texts = ["Hello world! This is a test of the hybrid FP32+BF16 SDPA decoder."]
        single_text_mode = True

    print("=" * 80)
    print("TTNN SpeechT5 TTS Demo - Hybrid FP32+BF16 SDPA Edition")
    print("=" * 80)

    # Configuration
    text = texts[0] if single_text_mode else None
    output_path = args.output
    max_steps = args.max_steps
    max_chunk_size = args.max_chunk_size
    device_id = 0

    if not single_text_mode:
        print(f"\n📝 Processing {len(texts)} texts")
        print(f"   Output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
    else:
        print(f"\n📝 Single text mode")
        print(f"   Input text: {texts[0][:80]}{'...' if len(texts[0]) > 80 else ''}")
        print(f"   Output path: {output_path}")
    print(f"Precision: FP32 with manual matmul attention\n")

    # Load HuggingFace models
    print("Loading models...")
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    hf_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
    hf_model.eval()

    # Load speaker embeddings
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

    # Initialize TTNN device
    print("Initializing TTNN device...")
    device = ttnn.open_device(
        device_id=device_id, l1_small_size=300000, trace_region_size=10000000, num_command_queues=2
    )
    device.enable_program_cache()

    # Create configs
    encoder_config = TTNNEncoderConfig(
        vocab_size=hf_model.config.vocab_size,
        hidden_size=hf_model.config.hidden_size,
        num_layers=hf_model.config.encoder_layers,
        num_heads=hf_model.config.encoder_attention_heads,
        ffn_dim=hf_model.config.encoder_ffn_dim,
        max_position_embeddings=hf_model.config.max_length,
        layer_norm_eps=hf_model.config.layer_norm_eps,
    )

    decoder_config = TTNNDecoderConfig(
        hidden_size=hf_model.config.hidden_size,
        num_layers=hf_model.config.decoder_layers,
        num_heads=hf_model.config.decoder_attention_heads,
        ffn_dim=hf_model.config.decoder_ffn_dim,
        max_position_embeddings=hf_model.config.max_length,
        layer_norm_eps=hf_model.config.layer_norm_eps,
        num_mel_bins=hf_model.config.num_mel_bins,
        reduction_factor=hf_model.config.reduction_factor,
        speech_decoder_prenet_units=hf_model.config.speech_decoder_prenet_units,
        speech_decoder_prenet_layers=hf_model.config.speech_decoder_prenet_layers,
        speech_decoder_prenet_dropout=0.5,
        speaker_embedding_dim=hf_model.config.speaker_embedding_dim,
        use_fp32=True,  # Enable FP32
    )

    postnet_config = TTNNPostNetConfig(
        postnet_units=hf_model.config.speech_decoder_postnet_units,
        postnet_layers=hf_model.config.speech_decoder_postnet_layers,
        postnet_kernel=hf_model.config.speech_decoder_postnet_kernel,
        postnet_dropout=0.5,
        num_mel_bins=hf_model.config.num_mel_bins,
        reduction_factor=hf_model.config.reduction_factor,
    )

    # Create TTNN models
    print("Creating TTNN models with FP32 decoder...")
    encoder_params = preprocess_encoder_parameters(hf_model.speecht5.encoder, encoder_config, device)
    ttnn_encoder = TTNNSpeechT5Encoder(device, encoder_params, encoder_config)

    decoder_params = preprocess_decoder_parameters(
        hf_model.speecht5.decoder, decoder_config, device, speaker_embeddings
    )
    ttnn_decoder = TTNNSpeechT5Decoder(device, decoder_params, decoder_config, max_sequence_length=512)

    postnet_params = preprocess_postnet_parameters(hf_model.speech_decoder_postnet, postnet_config, device)
    ttnn_postnet = TTNNSpeechT5SpeechDecoderPostnet(device, postnet_params, postnet_config)

    # Create generator wrapper for trace support
    print("🔧 Creating trace-enabled generator...")
    estimated_encoder_seq_len = 128
    generator = SpeechT5Generator(
        encoder=ttnn_encoder,
        decoder=ttnn_decoder,
        postnet=ttnn_postnet,
        device=device,
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
        dtype=ttnn.float32,  # FP32 for FP32 decoder
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    _ = ttnn_postnet(dummy_decoder_output)
    ttnn.deallocate(dummy_decoder_output)
    print("   Postnet kernels compiled!")

    # Warm-up phase to compile TTNN operations
    print("\n🔥 Warming up TTNN operations with FP32 + KV cache + Trace...")
    print("   This may take ~30-60 seconds as TTNN compiles kernels for optimal performance")
    warmup_start_time = time.time()

    # Use a short warmup text
    warmup_chunks = chunk_text(texts[0], max_chunk_size=max_chunk_size)
    warmup_text = warmup_chunks[0]  # Use only the first chunk for warm-up
    print(f"   Warm-up: Running {max_steps} steps...")
    warmup_speech = generate_speech_fp32(
        warmup_text,
        speaker_embeddings,
        processor,
        vocoder,
        ttnn_encoder,
        ttnn_decoder,
        ttnn_postnet,
        device,
        max_steps=max_steps,
        warmup_mode=True,
        generator=generator,
        use_kv_cache=True,
        decoder_config=decoder_config,
    )
    warmup_duration = time.time() - warmup_start_time
    print(f"✅ Initial warm-up completed in {warmup_duration:.1f}s (generated {len(warmup_speech)} samples)")
    print("   FP32 + KV cache + Trace enabled - optimal precision & performance!")

    # Capture traces for size 128 only (FP32 compatibility)
    # NOTE: Multi-size trace capture disabled for FP32 due to dtype mixing
    # (FP32 decoder states + BF16 speaker embeddings cause concat errors)
    print(f"\n🔧 Trace already captured for encoder_seq_len=128 during warm-up")
    # generator.capture_all_traces(processor, batch_size=1)  # Disabled for FP32
    print(f"✅ Trace ready! (encoder_seq_len=128 only)")

    # Report trace status
    compiled_sizes = [s for s in SUPPORTED_ENCODER_SEQ_LENS if generator.trace_compiled_per_size.get(s, False)]
    print(f"   Compiled traces for encoder sizes: {compiled_sizes}")

    # Reset KV caches after warm-up
    generator._reset_kv_caches()
    print("   KV caches reset for fresh inference - ready for any input length!")

    # Generate speech (INFERENCE - this is what we measure for performance)
    print("\n" + "=" * 80)
    print("🎙️  INFERENCE (excluding compile/warm-up)")
    print("=" * 80)

    results = []
    for i, text in enumerate(texts, 1):
        if not single_text_mode:
            # Generate filename for multi-text mode
            safe_text = "".join(c for c in text[:50] if c.isalnum() or c in (" ", "-", "_")).rstrip()
            if not safe_text:
                safe_text = f"speech_{i}"
            safe_text = safe_text.replace(" ", "_")
            output_file = os.path.join(output_dir, f"speech_fp32_{safe_text}.wav")
        else:
            output_file = output_path

        inference_start_time = time.time()

        # Use generate_speech_long_text which handles chunking
        speech, stats = generate_speech_long_text(
            text=text,
            speaker_embeddings=speaker_embeddings,
            processor=processor,
            vocoder=vocoder,
            ttnn_encoder=ttnn_encoder,
            ttnn_decoder=ttnn_decoder,
            ttnn_postnet=ttnn_postnet,
            device=device,
            max_steps=max_steps,
            use_kv_cache=True,
            decoder_config=decoder_config,
            return_stats=True,
            generator=generator,
            max_chunk_size=max_chunk_size,
        )

        inference_time = time.time() - inference_start_time

        # Save audio
        sf.write(output_file, speech.cpu().numpy(), samplerate=16000)
        audio_duration = len(speech) / 16000.0

        # Store results
        results.append(
            {
                "text": text,
                "output_file": output_file,
                "generation_time": inference_time,
                "audio_duration": audio_duration,
                "tokens_generated": stats["steps_completed"],
                "token_per_sec": stats.get("token_per_sec", 0),
                "ttft": stats.get("ttft", 0),
                "num_chunks": stats.get("num_chunks", 1),
                "stats": stats,
            }
        )

    # Display summary
    print(f"\n{'='*80}")
    print(f"✅ Generation complete!")
    print(f"{'='*80}")

    # Display INFERENCE SUMMARY (matching demo_ttnn.py format exactly)
    print("\n" + "=" * 140)
    print("📊 INFERENCE SUMMARY (FP32 + Trace + KV Cache)")
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
        print(
            f"{i:<4} {truncated_text:<25} {len(result['text']):<7} {result['num_chunks']:<7} {result['tokens_generated']:<8} {result['ttft']*1000:<8.0f} {result['token_per_sec']:<10.2f} {result['generation_time']:<10.3f} {result['audio_duration']:<8.1f}"
        )
        total_generation_time += result["generation_time"]
        total_tokens += result["tokens_generated"]
        total_audio_duration += result["audio_duration"]
        total_chunks += result["num_chunks"]

    print("-" * 140)
    print(
        f"{'TOTAL':<4} {'':<25} {'-':<7} {total_chunks:<7} {total_tokens:<8} {'-':<9} {'-':<10} {total_generation_time:<10.3f} {total_audio_duration:<8.1f}"
    )

    print(f"{'='*80}")

    # Cleanup
    ttnn.close_device(device)
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()
