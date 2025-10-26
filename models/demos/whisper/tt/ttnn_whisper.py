# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
TtnnWhisper wrapper class for the functional whisper implementation.
"""

import time
import zlib
from typing import List, Optional, Tuple

import torch
from loguru import logger
from tqdm import tqdm

import ttnn
from models.common.generation_utils import get_logits_processor

from . import ttnn_optimized_functional_whisper


class TtnnWhisper:
    def __init__(self, config=None, device=None, mesh_mappers=None):
        self.config = config
        self.device = device
        self.mesh_mappers = mesh_mappers

    def preprocess_encoder_inputs(
        self,
        config,
        input_features,
        *,
        parameters,
        device=None,
        input_mesh_mapper=None,
        weights_mesh_mapper=None,
    ):
        device = device or self.device
        if self.mesh_mappers and not input_mesh_mapper:
            input_mesh_mapper = self.mesh_mappers[0]
        if self.mesh_mappers and not weights_mesh_mapper:
            weights_mesh_mapper = self.mesh_mappers[1]

        return ttnn_optimized_functional_whisper.preprocess_encoder_inputs(
            config=config,
            input_features=input_features,
            parameters=parameters,
            device=device,
            input_mesh_mapper=input_mesh_mapper,
            weights_mesh_mapper=weights_mesh_mapper,
        )

    def encoder(self, config, inputs_embeds, *, parameters):
        return ttnn_optimized_functional_whisper.encoder(
            config=config,
            inputs_embeds=inputs_embeds,
            parameters=parameters,
        )

    def preprocess_decoder_inputs(
        self,
        config,
        input_ids,
        attention_mask,
        *,
        parameters,
        device=None,
        input_mesh_mapper=None,
        decode_pos=None,
        create_attention_mask=True,
    ):
        device = device or self.device
        if self.mesh_mappers and not input_mesh_mapper:
            input_mesh_mapper = self.mesh_mappers[0]

        return ttnn_optimized_functional_whisper.preprocess_decoder_inputs(
            config=config,
            input_ids=input_ids,
            attention_mask=attention_mask,
            parameters=parameters,
            device=device,
            input_mesh_mapper=input_mesh_mapper,
            decode_pos=decode_pos,
            create_attention_mask=create_attention_mask,
        )

    def decoder(
        self,
        config,
        hidden_states,
        decoder_attention_mask,
        encoder_hidden_states,
        kv_cache=None,
        current_decode_pos=None,
        *,
        parameters,
    ):
        return ttnn_optimized_functional_whisper.decoder(
            config=config,
            hidden_states=hidden_states,
            decoder_attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            kv_cache=kv_cache,
            current_decode_pos=current_decode_pos,
            parameters=parameters,
        )

    def preprocess_inputs(
        self,
        *,
        config,
        input_features,
        input_ids,
        attention_mask,
        parameters,
        device=None,
        create_attention_mask=True,
        input_mesh_mapper=None,
        weights_mesh_mapper=None,
    ):
        device = device or self.device
        if self.mesh_mappers and not input_mesh_mapper:
            input_mesh_mapper = self.mesh_mappers[0]
        if self.mesh_mappers and not weights_mesh_mapper:
            weights_mesh_mapper = self.mesh_mappers[1]

        return ttnn_optimized_functional_whisper.preprocess_inputs(
            config=config,
            input_features=input_features,
            input_ids=input_ids,
            attention_mask=attention_mask,
            parameters=parameters,
            device=device,
            create_attention_mask=create_attention_mask,
            input_mesh_mapper=input_mesh_mapper,
            weights_mesh_mapper=weights_mesh_mapper,
        )

    @staticmethod
    def convert_to_ttnn(model, name):
        return ttnn_optimized_functional_whisper.convert_to_ttnn(model, name)

    @staticmethod
    def create_custom_mesh_preprocessor(weights_mesh_mapper):
        return ttnn_optimized_functional_whisper.create_custom_mesh_preprocessor(weights_mesh_mapper)

    @staticmethod
    def init_kv_cache(config, device, max_batch_size, max_seq_len, weights_mesh_mapper, n_layers=None):
        return ttnn_optimized_functional_whisper.init_kv_cache(
            config=config,
            device=device,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            weights_mesh_mapper=weights_mesh_mapper,
            n_layers=n_layers,
        )

    def generate(
        self,
        config,
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
        temperatures=(0.0,),
        compression_ratio_threshold=None,
        logprob_threshold=None,
        no_speech_threshold=None,
        return_timestamps=False,
        stream_generation=False,
        return_perf_metrics=False,
        language="en",
        task="transcribe",
    ):
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
        audio_durations = self._calculate_audio_duration(current_batch) if return_timestamps else None

        # Compute embeddings
        input_embeds = self.preprocess_encoder_inputs(
            config=config,
            input_features=input_features,
            parameters=parameters.encoder,
            device=mesh_device,
            weights_mesh_mapper=weights_mesh_mapper,
            input_mesh_mapper=input_mesh_mapper,
        )

        logger.info(f"input_embeds.shape: {input_embeds.shape}")

        # Run encoder
        encoder_hidden_states = self.encoder(
            config=config,
            inputs_embeds=input_embeds,
            parameters=parameters.encoder,
        )
        ttnn.synchronize_device(mesh_device)
        logger.info(f"Time to encoder states: {(time.time() - start_encode)*1000:.3f}ms")

        logger.info(f"encoder_hidden_states.shape: {encoder_hidden_states.shape}")
        # Try generation with different temperatures
        best_output = None
        best_quality_score = float("inf")

        # Handle both single temperature and temperature list/tuple
        if isinstance(temperatures, (int, float)):
            temperatures = [temperatures]

        for temperature in temperatures:
            logger.info(f"Trying generation with temperature: {temperature}")

            try:
                if stream_generation:
                    output = self._generate_with_temperature_streaming(
                        temperature=temperature,
                        config=config,
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
                    )
                else:
                    output = self._generate_with_temperature_non_streaming(
                        temperature=temperature,
                        config=config,
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
                    )

                if stream_generation:
                    # For streaming, collect all results to check quality
                    outputs = list(output)
                    if return_perf_metrics:
                        # outputs = [(texts/segments, avg_logprob, no_speech_prob, ttft, throughput), ...]
                        result_data = [item[0] for item in outputs]
                        # Use final avg_logprobs
                        avg_logprobs = outputs[-1][1] if outputs else torch.zeros(unpadded_batch_size)
                        # Use final no_speech_probs
                        no_speech_probs = outputs[-1][2] if outputs else torch.zeros(unpadded_batch_size)
                        ttft = outputs[0][3] if outputs else 0
                        throughput = outputs[0][4] if outputs else 0
                    else:
                        # outputs = [(texts/segments, avg_logprob, no_speech_prob), ...]
                        result_data = [item[0] for item in outputs]
                        # Use final avg_logprobs
                        avg_logprobs = outputs[-1][1] if outputs else torch.zeros(unpadded_batch_size)
                        # Use final no_speech_probs
                        no_speech_probs = outputs[-1][2] if outputs else torch.zeros(unpadded_batch_size)

                    # Check quality for each result
                    quality_scores = []
                    for idx, data in enumerate(result_data):
                        if return_timestamps:
                            # For timestamps, extract text from segments for quality check
                            text = " ".join([segment["text"] for segment in data])
                        else:
                            text = data

                        compression_ratio = self._calculate_compression_ratio(text)
                        # Extract per-batch-item metrics
                        avg_logprob = avg_logprobs[idx].item() if idx < len(avg_logprobs) else -0.5
                        no_speech_prob = no_speech_probs[idx].item() if idx < len(no_speech_probs) else 0.0

                        is_good, reason = self._check_generation_quality(
                            text,
                            avg_logprob,
                            no_speech_prob,
                            compression_ratio,
                            logprob_threshold,
                            compression_ratio_threshold,
                            no_speech_threshold,
                        )

                        if is_good:
                            logger.info(f"Generation successful with temperature {temperature}")
                            if return_perf_metrics:
                                return (result_data, ttft, throughput)
                            else:
                                return result_data

                        quality_scores.append(compression_ratio)  # Lower is better

                    # Track best attempt
                    best_quality = min(quality_scores) if quality_scores else float("inf")
                    if best_quality < best_quality_score:
                        best_quality_score = best_quality
                        best_output = outputs

                else:
                    # Non-streaming generation
                    if return_perf_metrics:
                        result_data, avg_logprobs, no_speech_probs, ttft, throughput = output
                    else:
                        result_data, avg_logprobs, no_speech_probs = output

                    # Check quality for each result
                    all_good = True
                    for idx, data in enumerate(result_data):
                        if return_timestamps:
                            # For timestamps, extract text from segments for quality check
                            text = " ".join([segment["text"] for segment in data])
                        else:
                            text = data

                        compression_ratio = self._calculate_compression_ratio(text)
                        # Extract per-batch-item metrics
                        avg_logprob = avg_logprobs[idx].item() if idx < len(avg_logprobs) else -0.5
                        no_speech_prob = no_speech_probs[idx].item() if idx < len(no_speech_probs) else 0.0

                        is_good, reason = self._check_generation_quality(
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
                        return output

                    # Track best attempt
                    avg_compression = sum(
                        self._calculate_compression_ratio(
                            " ".join([segment["text"] for segment in data]) if return_timestamps else data
                        )
                        for data in result_data
                    ) / len(result_data)
                    if avg_compression < best_quality_score:
                        best_quality_score = avg_compression
                        best_output = output

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

    def _generate_with_temperature_streaming(
        self,
        temperature,
        config,
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
    ):
        """
        Generate text with a specific temperature (streaming mode).
        """
        # Input ids - use forced decoder IDs for translation
        forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)
        logger.debug(f"forced_decoder_ids from processor: {forced_decoder_ids}")

        # Control timestamp generation via the <|notimestamps|> token
        # In Whisper vocab: 50363 = <|nospeech|>, 50364 = <|notimestamps|>
        # When return_timestamps=True, remove <|notimestamps|> to allow timestamp generation
        # When return_timestamps=False, add <|notimestamps|> to disable timestamps
        NOTIMESTAMPS_TOKEN_ID = 50364

        # Keep forced_decoder_ids as tuples with positions
        if return_timestamps:
            # Remove notimestamps token if present
            forced_decoder_ids = [(pos, tok) for pos, tok in forced_decoder_ids if tok != NOTIMESTAMPS_TOKEN_ID]
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
            input_ids = self._pad_input_32(input_ids, config.pad_token_id).to(torch.long)
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

        # Streaming generation - yield results as they're generated
        for i in tqdm(range(MAX_GEN_LEN), desc=f"Decode inference iterations (temp={temperature})"):
            start_iter = time.time()

            decoder_hidden_states, decoder_attention_mask = self.preprocess_decoder_inputs(
                config=config,
                input_ids=input_ids,
                attention_mask=None,
                parameters=parameters.decoder,
                device=mesh_device,
                decode_pos=i if kv_cache else None,
                create_attention_mask=(not kv_cache),
                input_mesh_mapper=input_mesh_mapper,
            )

            output = self.decoder(
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
                output = output[:, last_tile_start_idx : last_tile_start_idx + 32, :]
            else:
                output_idx = 0

            output = output @ ttnn_linear_weight
            logits_to_torch = ttnn.to_torch(output, mesh_composer=output_mesh_composer)
            next_token_logits = logits_to_torch[:, output_idx, :]
            next_tokens_scores = logits_processor(input_features, next_token_logits)

            # Force tokens at specific positions based on forced_tokens_dict
            if i in forced_tokens_dict:
                next_tokens = torch.tensor([forced_tokens_dict[i]]).repeat(input_features.shape[0])
            else:
                next_tokens = self._sample_token(next_tokens_scores, temperature)

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
                no_speech_token_id = 50363  # <|nospeech|> token
                with torch.no_grad():
                    probs = torch.softmax(next_token_logits, dim=-1)
                    no_speech_probs = probs[:, no_speech_token_id]  # Per-batch probabilities

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

            if all(prompt_is_done):
                break
        total_generate_time = time.time() - start_encode
        logger.info(f"Time to first token: {(ttft*1000):.3f}ms")
        logger.info(f"Total decode time: {total_decode_time:.3f}s")
        logger.info(f"Total generate time: {total_generate_time:.3f}s")
        logger.info(f"Average decode throughput (per user): {avg_decode_throughput:.3f} t/s/u")
        logger.info(f"Average decode throughput (total batch): {(avg_decode_throughput * unpadded_batch_size):.3f} t/s")

        # Process timestamps if requested
        if return_timestamps and full_token_sequences:
            # Extract timestamps for each batch item
            segments_with_timestamps = []
            for batch_idx in range(unpadded_batch_size):
                if full_token_sequences[batch_idx]:
                    token_sequence = torch.tensor(full_token_sequences[batch_idx])
                    audio_duration = audio_durations[batch_idx] if audio_durations else None
                    segments = self._extract_timestamps_from_tokens(token_sequence, processor, audio_duration)
                    segments_with_timestamps.append(segments)
                else:
                    segments_with_timestamps.append([])

            # Yield final result with timestamps
            if return_perf_metrics:
                yield segments_with_timestamps, current_avg_logprob, current_no_speech_probs, ttft, avg_decode_throughput
            else:
                yield segments_with_timestamps, current_avg_logprob, current_no_speech_probs

    def _generate_with_temperature_non_streaming(
        self,
        temperature,
        config,
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
    ):
        """
        Generate text with a specific temperature (non-streaming mode).
        """
        # Input ids - use forced decoder IDs for translation
        forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)
        logger.debug(f"forced_decoder_ids from processor: {forced_decoder_ids}")

        # Control timestamp generation via the <|notimestamps|> token
        # In Whisper vocab: 50363 = <|nospeech|>, 50364 = <|notimestamps|>
        # When return_timestamps=True, remove <|notimestamps|> to allow timestamp generation
        # When return_timestamps=False, add <|notimestamps|> to disable timestamps
        NOTIMESTAMPS_TOKEN_ID = 50364

        # Keep forced_decoder_ids as tuples with positions
        if return_timestamps:
            # Remove notimestamps token if present
            forced_decoder_ids = [(pos, tok) for pos, tok in forced_decoder_ids if tok != NOTIMESTAMPS_TOKEN_ID]
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
            input_ids = self._pad_input_32(input_ids, config.pad_token_id).to(torch.long)
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

        # Non-streaming generation - collect all results and return final output
        output = [[] for _ in range(input_features.shape[0])]
        ttft = 0.0
        avg_decode_throughput = 0.0

        # Process the generation loop directly
        for i in tqdm(range(MAX_GEN_LEN), desc=f"Decode inference iterations (temp={temperature})"):
            start_iter = time.time()

            decoder_hidden_states, decoder_attention_mask = self.preprocess_decoder_inputs(
                config=config,
                input_ids=input_ids,
                attention_mask=None,
                parameters=parameters.decoder,
                device=mesh_device,
                decode_pos=i if kv_cache else None,
                create_attention_mask=(not kv_cache),
                input_mesh_mapper=input_mesh_mapper,
            )

            decoder_output = self.decoder(
                config,
                decoder_hidden_states,
                decoder_attention_mask=decoder_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                kv_cache=kv_cache,
                current_decode_pos=current_decode_pos,
                parameters=parameters.decoder,
            )

            if not kv_cache:
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
                # Apply temperature and sample
                next_tokens = self._sample_token(next_tokens_scores, temperature)

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
                no_speech_token_id = 50363  # <|nospeech|> token
                with torch.no_grad():
                    probs = torch.softmax(next_token_logits, dim=-1)
                    no_speech_probs = probs[:, no_speech_token_id]  # Per-batch probabilities

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

            # Collect results
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
            for batch_idx in range(input_features.shape[0]):
                if full_token_sequences[batch_idx]:
                    token_sequence = torch.tensor(full_token_sequences[batch_idx])
                    audio_duration = audio_durations[batch_idx] if audio_durations else None
                    segments = self._extract_timestamps_from_tokens(token_sequence, processor, audio_duration)
                    segments_with_timestamps.append(segments)
                else:
                    segments_with_timestamps.append([])

            if return_perf_metrics:
                return (segments_with_timestamps, avg_logprob, no_speech_probs, ttft, avg_decode_throughput)
            else:
                return (segments_with_timestamps, avg_logprob, no_speech_probs)
        else:
            # Join the collected tokens into final text
            output = ["".join(tokens) for tokens in output]
            if return_perf_metrics:
                return (output, avg_logprob, no_speech_probs, ttft, avg_decode_throughput)
            else:
                return (output, avg_logprob, no_speech_probs)

    def _pad_input_32(self, tensor, value):
        """Pad input to multiple of 32."""
        len_tensor = tensor.shape[1]
        if len_tensor % 32 == 0:
            return tensor
        padded_len = ((len_tensor // 32) + 1) * 32
        pad_tensor = (value * torch.ones(tensor.shape[0], padded_len - len_tensor)).to(torch.long)
        tensor = torch.cat([tensor, pad_tensor], dim=1)
        return tensor

    def _sample_token(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
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

    def _calculate_compression_ratio(self, text: str) -> float:
        """
        Calculate compression ratio of text using zlib.
        """
        if not text:
            return 0.0
        text_bytes = text.encode("utf-8")
        compressed_bytes = zlib.compress(text_bytes)
        return len(text_bytes) / len(compressed_bytes)

    def _check_generation_quality(
        self,
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

    def _extract_timestamps_from_tokens(
        self, token_ids: torch.Tensor, processor, audio_duration: float = None
    ) -> List[dict]:
        """
        Extract timestamps from generated token IDs and segment the text.
        """
        # Whisper timestamp tokens: 50365-51864 represent time intervals
        # Token IDs: 50257 = <|endoftext|>, 50362 = <|startofprev|>, 50363 = <|nospeech|>, 50364 = <|notimestamps|>
        # Timestamp tokens start at 50365 and represent 0.02 second intervals
        EOS_TOKEN_ID = 50257  # <|endoftext|> token
        TIMESTAMP_TOKEN_START = 50365
        TIMESTAMP_TOKEN_END = 51864  # 1500 tokens = 30 seconds max

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

    def _calculate_audio_duration(self, current_batch) -> List[float]:
        """
        Calculate audio duration for each item in the batch.
        """
        durations = []
        for sampling_rate, audio_array in current_batch:
            duration = len(audio_array) / sampling_rate
            durations.append(duration)
        return durations


# Re-export constants and functions for backward compatibility
WHISPER_L1_SMALL_SIZE = ttnn_optimized_functional_whisper.WHISPER_L1_SMALL_SIZE
WHISPER_MEMORY_CONFIG = ttnn_optimized_functional_whisper.WHISPER_MEMORY_CONFIG

# Re-export all functions for backward compatibility
gelu = ttnn_optimized_functional_whisper.gelu
dropout = ttnn_optimized_functional_whisper.dropout
calculate_key_values = ttnn_optimized_functional_whisper.calculate_key_values
get_decode_sdpa_configs = ttnn_optimized_functional_whisper.get_decode_sdpa_configs
functional_sdpa = ttnn_optimized_functional_whisper.functional_sdpa
whisper_attention = ttnn_optimized_functional_whisper.whisper_attention
encoder_layer = ttnn_optimized_functional_whisper.encoder_layer
encoder = ttnn_optimized_functional_whisper.encoder
make_causal_mask = ttnn_optimized_functional_whisper.make_causal_mask
expand_mask = ttnn_optimized_functional_whisper.expand_mask
decoder_layer = ttnn_optimized_functional_whisper.decoder_layer
prepare_decoder_attention_mask = ttnn_optimized_functional_whisper.prepare_decoder_attention_mask
decoder = ttnn_optimized_functional_whisper.decoder
get_conv_configs = ttnn_optimized_functional_whisper.get_conv_configs
prepare_conv_weights = ttnn_optimized_functional_whisper.prepare_conv_weights
preprocess_encoder_inputs = ttnn_optimized_functional_whisper.preprocess_encoder_inputs
preprocess_decoder_inputs = ttnn_optimized_functional_whisper.preprocess_decoder_inputs
preprocess_inputs = ttnn_optimized_functional_whisper.preprocess_inputs
custom_preprocessor = ttnn_optimized_functional_whisper.custom_preprocessor
