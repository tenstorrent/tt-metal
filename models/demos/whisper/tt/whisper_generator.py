# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Whisper generation functions using the functional whisper implementation from ttnn_optimized_functional_whisper.
"""

import time
import zlib
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
from loguru import logger
from tqdm import tqdm

import ttnn
from models.common.generation_utils import get_logits_processor

from . import ttnn_optimized_functional_whisper


@dataclass
class GenerationParams:
    """Dataclass for Whisper generation parameters."""

    temperatures: Union[float, Tuple[float, ...]] = 0.0
    compression_ratio_threshold: Optional[float] = 2.4
    logprob_threshold: Optional[float] = -2.0
    no_speech_threshold: Optional[float] = 0.6
    return_timestamps: bool = False
    language: str = "en"
    task: str = "transcribe"


# Default values for quality metrics
DEFAULT_AVG_LOGPROB = -0.5
DEFAULT_NO_SPEECH_PROB = 0.0

# Control timestamp generation via the <|notimestamps|> token
NOTIMESTAMPS_TOKEN_ID = 50364
# <|nospeech|> token
NO_SPEECH_TOKEN_ID = 50363

# Whisper timestamp tokens: 50365-51864 represent time intervals
# Timestamp tokens start at 50365 and represent 0.02 second intervals
EOS_TOKEN_ID = 50257  # <|endoftext|> token
TIMESTAMP_TOKEN_START = 50365
TIMESTAMP_TOKEN_END = 51864  # 1500 tokens = 30 seconds max


def generate(
    config,
    device,
    mesh_mappers,
    current_batch,
    feature_extractor,
    parameters,
    processor,
    ttnn_linear_weight,
    mesh_device,
    generation_config,
    input_mesh_mapper,
    output_mesh_composer,
    weights_mesh_mapper,
    kv_cache=None,
    generation_params: Optional[GenerationParams] = None,
    stream_generation=False,
    return_perf_metrics=False,
):
    # Unpack generation parameters
    if generation_params is None:
        generation_params = GenerationParams()

    temperatures = generation_params.temperatures
    compression_ratio_threshold = generation_params.compression_ratio_threshold
    logprob_threshold = generation_params.logprob_threshold
    no_speech_threshold = generation_params.no_speech_threshold
    return_timestamps = generation_params.return_timestamps
    language = generation_params.language
    task = generation_params.task

    # Process input features
    all_input_features = []
    start_encode = time.time()
    for sampling_rate, audio_array in current_batch:
        inputs = feature_extractor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        )
        all_input_features.append(inputs.input_features)

    input_features = torch.cat(all_input_features, dim=0)  # [B, x, y]
    del all_input_features
    unpadded_batch_size = input_features.shape[0]
    assert (
        unpadded_batch_size == 1 * mesh_device.get_num_devices()
    ), "Only batch size (per device) 1 is supported for inference"

    # Calculate audio durations for timestamp capping
    audio_durations = _calculate_audio_duration(current_batch) if return_timestamps else None

    # Compute embeddings
    input_embeds = ttnn_optimized_functional_whisper.preprocess_encoder_inputs(
        config=config,
        input_features=input_features,
        parameters=parameters.encoder,
        device=mesh_device,
        weights_mesh_mapper=weights_mesh_mapper,
        input_mesh_mapper=input_mesh_mapper,
    )

    # Run encoder
    encoder_hidden_states = ttnn_optimized_functional_whisper.encoder(
        config=config,
        inputs_embeds=input_embeds,
        parameters=parameters.encoder,
    )
    ttnn.synchronize_device(mesh_device)
    logger.info(f"Time to encoder states: {(time.time() - start_encode)*1000:.3f}ms")

    # Handle both single temperature and temperature list/tuple
    if isinstance(temperatures, (int, float)):
        temperatures = [temperatures]

    # For streaming mode, skip temperature fallback and quality checks
    # Use only the first temperature and yield tokens immediately
    if stream_generation:
        temperature = temperatures[0]
        logger.info(f"Streaming mode: using temperature {temperature}, skipping quality checks")

        return _generate_with_temperature(
            temperature=temperature,
            config=config,
            device=device,
            mesh_mappers=mesh_mappers,
            start_encode=start_encode,
            generation_config=generation_config,
            encoder_hidden_states=encoder_hidden_states,
            input_features=input_features,
            parameters=parameters,
            processor=processor,
            ttnn_linear_weight=ttnn_linear_weight,
            mesh_device=mesh_device,
            input_mesh_mapper=input_mesh_mapper,
            output_mesh_composer=output_mesh_composer,
            kv_cache=kv_cache,
            unpadded_batch_size=unpadded_batch_size,
            return_perf_metrics=return_perf_metrics,
            return_timestamps=return_timestamps,
            audio_durations=audio_durations,
            language=language,
            task=task,
            streaming=True,
        )

    # Non-streaming mode: Try generation with different temperatures
    best_output = None
    best_quality_score = float("inf")

    for temperature in temperatures:
        logger.info(f"Trying generation with temperature: {temperature}")

        try:
            output = _generate_with_temperature(
                temperature=temperature,
                config=config,
                device=device,
                mesh_mappers=mesh_mappers,
                start_encode=start_encode,
                generation_config=generation_config,
                encoder_hidden_states=encoder_hidden_states,
                input_features=input_features,
                parameters=parameters,
                processor=processor,
                ttnn_linear_weight=ttnn_linear_weight,
                mesh_device=mesh_device,
                input_mesh_mapper=input_mesh_mapper,
                output_mesh_composer=output_mesh_composer,
                kv_cache=kv_cache,
                unpadded_batch_size=unpadded_batch_size,
                return_perf_metrics=return_perf_metrics,
                return_timestamps=return_timestamps,
                audio_durations=audio_durations,
                language=language,
                task=task,
                streaming=False,  # Non-streaming mode for quality checks
            )

            # Non-streaming generation - consume the generator to get the single result
            if return_perf_metrics:
                result_data, avg_logprobs, no_speech_probs, ttft, throughput = next(output)
            else:
                result_data, avg_logprobs, no_speech_probs = next(output)

            # Check quality for each result
            all_good = True
            for idx, data in enumerate(result_data):
                if return_timestamps:
                    # For timestamps, extract text from segments for quality check
                    text = " ".join([segment["text"] for segment in data])
                else:
                    text = data

                compression_ratio = _calculate_compression_ratio(text)
                # Extract per-batch-item metrics
                avg_logprob = avg_logprobs[idx].item() if idx < len(avg_logprobs) else DEFAULT_AVG_LOGPROB
                no_speech_prob = no_speech_probs[idx].item() if idx < len(no_speech_probs) else DEFAULT_NO_SPEECH_PROB

                is_good, reason = _check_generation_quality(
                    text,
                    avg_logprob,
                    no_speech_prob,
                    compression_ratio,
                    logprob_threshold,
                    compression_ratio_threshold,
                    no_speech_threshold,
                )

                if not is_good:
                    logger.info(f"Quality check failed with temperature {temperature}: {reason}")
                    all_good = False
                    break

            if all_good:
                logger.info(f"Generation successful with temperature {temperature}")
                if return_perf_metrics:
                    return (result_data, avg_logprobs, no_speech_probs, ttft, throughput)
                else:
                    return (result_data, avg_logprobs, no_speech_probs)

            # Track best attempt
            avg_compression = sum(
                _calculate_compression_ratio(
                    " ".join([segment["text"] for segment in data]) if return_timestamps else data
                )
                for data in result_data
            ) / len(result_data)
            if avg_compression < best_quality_score:
                best_quality_score = avg_compression
                if return_perf_metrics:
                    best_output = (result_data, avg_logprobs, no_speech_probs, ttft, throughput)
                else:
                    best_output = (result_data, avg_logprobs, no_speech_probs)

        except Exception as e:
            logger.warning(f"Generation failed with temperature {temperature}: {e}")
            continue

    # If all temperatures failed, return best attempt
    logger.warning("All temperature attempts failed quality checks, returning best attempt")
    if best_output is not None:
        return best_output
    else:
        # Return appropriate format based on return_perf_metrics and return_timestamps
        if return_timestamps:
            empty_segments = [[] for _ in range(unpadded_batch_size)]
            if return_perf_metrics:
                return (
                    empty_segments,
                    torch.zeros(unpadded_batch_size),
                    torch.zeros(unpadded_batch_size),
                    0.0,
                    0.0,
                )
            else:
                return (empty_segments, torch.zeros(unpadded_batch_size), torch.zeros(unpadded_batch_size))
        else:
            if return_perf_metrics:
                return (
                    [""] * unpadded_batch_size,
                    torch.zeros(unpadded_batch_size),
                    torch.zeros(unpadded_batch_size),
                    0.0,
                    0.0,
                )
            else:
                return (
                    [""] * unpadded_batch_size,
                    torch.zeros(unpadded_batch_size),
                    torch.zeros(unpadded_batch_size),
                )


def _generate_with_temperature(
    temperature,
    config,
    device,
    mesh_mappers,
    start_encode,
    generation_config,
    encoder_hidden_states,
    input_features,
    parameters,
    processor,
    ttnn_linear_weight,
    mesh_device,
    input_mesh_mapper,
    output_mesh_composer,
    kv_cache,
    unpadded_batch_size,
    return_perf_metrics=False,
    return_timestamps=False,
    audio_durations=None,
    language="en",
    task="transcribe",
    streaming=False,
):
    """
    Generate text with a specific temperature.
    Supports both streaming and non-streaming modes.
    """
    # Input ids - use forced decoder IDs for translation
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)
    logger.debug(f"forced_decoder_ids from processor: {forced_decoder_ids}")

    # Keep forced_decoder_ids as tuples with positions
    # When return_timestamps=True, remove <|notimestamps|> to allow timestamp generation
    if return_timestamps:
        # Remove notimestamps token if present
        forced_decoder_ids = [(pos, tok) for pos, tok in forced_decoder_ids if tok != NOTIMESTAMPS_TOKEN_ID]

    # When return_timestamps=False, add <|notimestamps|> to disable timestamps
    else:
        # Add notimestamps token if not present (at the appropriate position)
        if not any(tok == NOTIMESTAMPS_TOKEN_ID for _, tok in forced_decoder_ids):
            # Find the last position and add notimestamps after it
            max_pos = max((pos for pos, _ in forced_decoder_ids), default=0)
            forced_decoder_ids.append((max_pos + 1, NOTIMESTAMPS_TOKEN_ID))

    # Create a position-to-token mapping instead of a simple list
    # This preserves the actual positions (which may be 1-indexed or have gaps)
    forced_tokens_dict = {pos: token_id for pos, token_id in forced_decoder_ids}
    logger.debug(f"forced_tokens_dict after manipulation: {forced_tokens_dict}")

    # Start with simple start token
    input_ids = torch.tensor([[1]]) * config.decoder_start_token_id
    input_ids = input_ids.repeat(input_features.shape[0], 1)
    logits_processor = get_logits_processor(input_ids, config)

    if not kv_cache:
        input_ids = _pad_input_32(input_ids, config.pad_token_id).to(torch.long)
        decoder_start_values = generation_config.pad_token_id * torch.ones(1, 32).to(torch.long)

    # Initial decode position
    current_decode_pos = (
        ttnn.from_torch(
            torch.zeros(unpadded_batch_size), device=mesh_device, dtype=ttnn.int32, mesh_mapper=input_mesh_mapper
        )
        if kv_cache
        else None
    )

    MAX_GEN_LEN = config.max_length
    output_ids = []
    total_decode_time = 0
    prompt_is_done = [False for _ in range(unpadded_batch_size)]
    log_probs = []  # Track log probabilities
    no_speech_probs = None  # Will be extracted from first frame

    # Track full token sequences for timestamp extraction
    full_token_sequences = [[] for _ in range(unpadded_batch_size)] if return_timestamps else None

    # Non-streaming mode: collect all results in a list
    if not streaming:
        output = [[] for _ in range(input_features.shape[0])]
    ttft = 0.0
    avg_decode_throughput = 0.0

    # Generation loop
    for i in tqdm(range(MAX_GEN_LEN), desc=f"Decode inference iterations (temp={temperature})"):
        start_iter = time.time()

        decoder_hidden_states, decoder_attention_mask = ttnn_optimized_functional_whisper.preprocess_decoder_inputs(
            config=config,
            input_ids=input_ids,
            attention_mask=None,
            parameters=parameters.decoder,
            device=mesh_device,
            decode_pos=i if kv_cache else None,
            create_attention_mask=(not kv_cache),
            input_mesh_mapper=input_mesh_mapper,
        )

        decoder_output = ttnn_optimized_functional_whisper.decoder(
            config,
            decoder_hidden_states,
            decoder_attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            kv_cache=kv_cache,
            current_decode_pos=current_decode_pos,
            parameters=parameters.decoder,
        )

        if not kv_cache:
            # Note: if not using a kv cache, the entire sequence is recomputed at each step
            # Only run the lm head on the last tile to fix bad outputs and reduce redundant computation
            last_tile_start_idx = i // 32 * 32
            output_idx = i % 32
            decoder_output = decoder_output[:, last_tile_start_idx : last_tile_start_idx + 32, :]
        else:
            output_idx = 0

        decoder_output = decoder_output @ ttnn_linear_weight
        logits_to_torch = ttnn.to_torch(decoder_output, mesh_composer=output_mesh_composer)
        next_token_logits = logits_to_torch[:, output_idx, :]
        next_tokens_scores = logits_processor(input_features, next_token_logits)

        # Force tokens at specific positions based on forced_tokens_dict
        if i in forced_tokens_dict:
            next_tokens = torch.tensor([forced_tokens_dict[i]]).repeat(input_features.shape[0])
        else:
            next_tokens = _sample_token(next_tokens_scores, temperature)

        # Track log probabilities
        with torch.no_grad():
            log_probs.append(
                torch.log_softmax(next_tokens_scores, dim=-1).gather(1, next_tokens.unsqueeze(1)).squeeze(1)
            )

        output_ids.append(next_tokens)

        # Track full token sequences for timestamp extraction
        if return_timestamps:
            for batch_idx in range(unpadded_batch_size):
                full_token_sequences[batch_idx].append(next_tokens[batch_idx].item())

        if i == 0:
            first_token_time = time.time()
            ttft = first_token_time - start_encode
            # Extract no_speech probability from first frame logits
            with torch.no_grad():
                probs = torch.softmax(next_token_logits, dim=-1)
                no_speech_probs = probs[:, NO_SPEECH_TOKEN_ID]  # Per-batch probabilities

        # Update input_ids and current_decode_pos
        if not kv_cache:
            if (i + 1) % 32 == 0:
                input_ids = torch.cat([input_ids, decoder_start_values], dim=1)
            input_ids[:, i + 1] = next_tokens[:, None]
        else:
            input_ids = next_tokens[:, None]
            ttnn.plus_one(current_decode_pos)

        total_decode_time += time.time() - start_iter
        avg_decode_throughput = (i + 1) / total_decode_time

        for user_id, user_decode_id in enumerate(next_tokens[:unpadded_batch_size]):
            if user_decode_id == config.eos_token_id:
                prompt_is_done[user_id] = True
            if prompt_is_done[user_id]:
                next_tokens[user_id] = config.eos_token_id

        ttnn_transcription = processor.batch_decode(next_tokens.unsqueeze(dim=1), skip_special_tokens=True)

        # Streaming mode: yield incremental results
        if streaming:
            # Calculate current average log probability for each batch item
            if log_probs:
                current_avg_logprob = torch.stack(log_probs, dim=1).mean(dim=1)
            else:
                current_avg_logprob = torch.zeros(input_features.shape[0])

            # Use zeros for no_speech_probs if not yet calculated
            if no_speech_probs is None:
                current_no_speech_probs = torch.zeros(input_features.shape[0])
            else:
                current_no_speech_probs = no_speech_probs

            # For streaming, we yield the current transcription without timestamps
            # Timestamps will be processed at the end if return_timestamps=True
            if return_perf_metrics:
                yield ttnn_transcription, current_avg_logprob, current_no_speech_probs, ttft, avg_decode_throughput
            else:
                yield ttnn_transcription, current_avg_logprob, current_no_speech_probs
        else:
            # Non-streaming mode: collect results
            for idx in range(input_features.shape[0]):
                output[idx].append(ttnn_transcription[idx])

        if all(prompt_is_done):
            break

    total_generate_time = time.time() - start_encode
    logger.info(f"Time to first token: {(ttft*1000):.3f}ms")
    logger.info(f"Total decode time: {total_decode_time:.3f}s")
    logger.info(f"Total generate time: {total_generate_time:.3f}s")
    logger.info(f"Average decode throughput (per user): {avg_decode_throughput:.3f} t/s/u")
    logger.info(f"Average decode throughput (total batch): {(avg_decode_throughput * unpadded_batch_size):.3f} t/s")

    # Calculate average log probability for each batch item
    if log_probs:
        avg_logprob = torch.stack(log_probs, dim=1).mean(dim=1)
    else:
        avg_logprob = torch.zeros(input_features.shape[0])

    # Use zeros for no_speech_probs if not calculated
    if no_speech_probs is None:
        no_speech_probs = torch.zeros(input_features.shape[0])

    # Process timestamps if requested
    if return_timestamps and full_token_sequences:
        # Extract timestamps for each batch item
        segments_with_timestamps = []
        for batch_idx in range(unpadded_batch_size):
            if full_token_sequences[batch_idx]:
                token_sequence = torch.tensor(full_token_sequences[batch_idx])
                audio_duration = audio_durations[batch_idx] if audio_durations else None
                segments = _extract_timestamps_from_tokens(token_sequence, processor, audio_duration)
                segments_with_timestamps.append(segments)
            else:
                segments_with_timestamps.append([])

        # Yield final result with timestamps (works for both streaming and non-streaming)
        if return_perf_metrics:
            yield segments_with_timestamps, avg_logprob, no_speech_probs, ttft, avg_decode_throughput
        else:
            yield segments_with_timestamps, avg_logprob, no_speech_probs
    else:
        if streaming:
            # For streaming without timestamps, yield final accumulated result
            # Accumulate all tokens from output_ids
            final_output = []
            for batch_idx in range(unpadded_batch_size):
                # Collect all tokens for this batch item
                batch_tokens = [output_ids[i][batch_idx] for i in range(len(output_ids))]
                # Decode the full sequence
                decoded_text = processor.batch_decode(
                    torch.tensor(batch_tokens).unsqueeze(0), skip_special_tokens=True
                )[0]
                final_output.append(decoded_text)

            if return_perf_metrics:
                yield final_output, avg_logprob, no_speech_probs, ttft, avg_decode_throughput
            else:
                yield final_output, avg_logprob, no_speech_probs
        else:
            # Join the collected tokens into final text
            output = ["".join(tokens) for tokens in output]
            if return_perf_metrics:
                yield (output, avg_logprob, no_speech_probs, ttft, avg_decode_throughput)
            else:
                yield (output, avg_logprob, no_speech_probs)


def _pad_input_32(tensor, value):
    """Pad input to multiple of 32."""
    len_tensor = tensor.shape[1]
    if len_tensor % 32 == 0:
        return tensor
    padded_len = ((len_tensor // 32) + 1) * 32
    pad_tensor = (value * torch.ones(tensor.shape[0], padded_len - len_tensor)).to(torch.long)
    tensor = torch.cat([tensor, pad_tensor], dim=1)
    return tensor


def _sample_token(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Sample token from logits with temperature.
    """
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1)
    else:
        # Apply temperature scaling before softmax
        scaled_logits = logits / temperature
        probs = torch.softmax(scaled_logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)


def _calculate_compression_ratio(text: str) -> float:
    """
    Calculate compression ratio of text using zlib.
    """
    if not text:
        return 0.0
    text_bytes = text.encode("utf-8")
    compressed_bytes = zlib.compress(text_bytes)
    return len(text_bytes) / len(compressed_bytes)


def _check_generation_quality(
    text: str,
    avg_logprob: float,
    no_speech_prob: float,
    compression_ratio: float,
    logprob_threshold: Optional[float],
    compression_ratio_threshold: Optional[float],
    no_speech_threshold: Optional[float],
) -> Tuple[bool, str]:
    """
    Check if generation passes quality thresholds.
    """
    # Check for silence
    if no_speech_threshold is not None and logprob_threshold is not None:
        if no_speech_prob > no_speech_threshold and avg_logprob < logprob_threshold:
            return False, "silence_detected"

    # Check compression ratio (high ratio indicates repetitive text)
    if compression_ratio_threshold is not None:
        if compression_ratio > compression_ratio_threshold:
            return False, "high_compression_ratio"

    return True, "quality_good"


def _extract_timestamps_from_tokens(token_ids: torch.Tensor, processor, audio_duration: float = None) -> List[dict]:
    """
    Extract timestamps from generated token IDs and segment the text.
    """
    # Convert to list for easier processing
    tokens = token_ids.tolist()
    segments = []
    current_text_tokens = []
    current_start_time = 0.0

    for i, token in enumerate(tokens):
        # Stop processing at EOS token to avoid repetitive tokens
        if token == EOS_TOKEN_ID:
            break

        if TIMESTAMP_TOKEN_START <= token <= TIMESTAMP_TOKEN_END:
            # This is a timestamp token
            timestamp_seconds = (token - TIMESTAMP_TOKEN_START) * 0.02

            # If we have accumulated text tokens, create a segment
            if current_text_tokens:
                # Decode the text tokens
                text_tokens_tensor = torch.tensor([current_text_tokens])
                segment_text = processor.batch_decode(text_tokens_tensor, skip_special_tokens=True)[0]

                # Only add non-empty segments
                if segment_text.strip():
                    segments.append(
                        {"text": segment_text.strip(), "start": current_start_time, "end": timestamp_seconds}
                    )

            # Start new segment
            current_text_tokens = []
            current_start_time = timestamp_seconds
        else:
            # Regular text token
            current_text_tokens.append(token)

    # Handle any remaining text tokens (final segment)
    if current_text_tokens:
        # Skip creating segment if it starts beyond the audio duration
        if audio_duration is not None and current_start_time >= audio_duration:
            # This segment is beyond the audio duration, likely repetitive tokens
            return segments

        text_tokens_tensor = torch.tensor([current_text_tokens])
        segment_text = processor.batch_decode(text_tokens_tensor, skip_special_tokens=True)[0]
        if segment_text.strip():
            # For the final segment, we don't have an end timestamp, so we'll use a reasonable estimate
            # Use audio duration to cap the estimation if available
            if audio_duration is not None:
                estimated_duration = min(
                    max(0.5, len(segment_text) * 0.1),  # Current estimation
                    audio_duration - current_start_time,  # Don't exceed audio length
                )
            else:
                estimated_duration = max(0.5, len(segment_text) * 0.1)  # Fallback to original logic

            segments.append(
                {
                    "text": segment_text.strip(),
                    "start": current_start_time,
                    "end": current_start_time + estimated_duration,
                }
            )

    return segments


def _calculate_audio_duration(current_batch) -> List[float]:
    """
    Calculate audio duration for each item in the batch.
    """
    durations = []
    for sampling_rate, audio_array in current_batch:
        duration = len(audio_array) / sampling_rate
        durations.append(duration)
    return durations
