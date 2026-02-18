# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Whisper generation functions using the functional whisper implementation from ttnn_optimized_functional_whisper.
"""

import time
import zlib
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
from loguru import logger
from tqdm import tqdm

import ttnn
from models.common.generation_utils import get_logits_processor

from . import ttnn_optimized_functional_whisper
from .ttnn_optimized_functional_whisper import WHISPER_BATCH_SIZE


@dataclass
class GenerationParams:
    """Dataclass for Whisper generation parameters.

    Note: language, task, prompt, and use_trace are batch-homogeneous and must be
    passed as separate parameters to generate().
    """

    temperatures: Union[float, Tuple[float, ...]] = 0.0
    compression_ratio_threshold: Optional[float] = 2.4
    logprob_threshold: Optional[float] = -2.0
    no_speech_threshold: Optional[float] = 0.6
    return_timestamps: bool = False


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

# Prompt-related tokens
STARTOFPREV_TOKEN_ID = 50362  # <|startofprev|> token for prompt conditioning
STARTOFTRANSCRIPT_TOKEN_ID = 50258  # <|startoftranscript|> token
MAX_PROMPT_TOKENS = 224  # Maximum number of tokens allowed in prompt


class WhisperGenerator:
    """
    Whisper generator with fully persistent trace support across ALL generations.

    This class maintains trace artifacts as instance variables, enabling trace reuse
    across multiple audio generations. The decoder trace is captured once on the first
    generation and reused for ALL subsequent generations.

    Key insight: The decoder trace can be fully persistent because:
    1. cross_attn_cache is pre-allocated with stable memory addresses
    2. encoder_hidden_states is pre-allocated with stable memory addresses
    3. First decoder iteration (non-traced) copies new K/V into pre-allocated cache
    4. Subsequent iterations (traced) use the pre-allocated cache at same addresses
    5. The trace references these stable addresses across all generations
    """

    def __init__(
        self,
        config,
        mesh_device,
        parameters,
        processor,
        feature_extractor,
        ttnn_linear_weight,
        generation_config,
        input_mesh_mapper,
        output_mesh_composer,
        weights_mesh_mapper,
        kv_cache_per_batch_size=None,
        cross_attn_cache_per_batch_size=None,
        max_batch_size=2,
    ):
        """
        Initialize the WhisperGenerator.

        Args:
            config: Whisper model configuration
            mesh_device: TTNN mesh device
            parameters: Preprocessed model parameters
            processor: Whisper processor for tokenization
            feature_extractor: Feature extractor for audio preprocessing
            ttnn_linear_weight: Language model head weights
            generation_config: HuggingFace generation config
            input_mesh_mapper: Mesh mapper for inputs
            output_mesh_composer: Mesh composer for outputs
            weights_mesh_mapper: Mesh mapper for weights
            kv_cache: Self-attention KV cache (optional)
            cross_attn_cache: Cross-attention cache (pre-allocated, optional)
        """
        self.config = config
        self.mesh_device = mesh_device
        self.parameters = parameters
        self.processor = processor
        self.feature_extractor = feature_extractor
        self.ttnn_linear_weight = ttnn_linear_weight
        self.generation_config = generation_config
        self.input_mesh_mapper = input_mesh_mapper
        self.output_mesh_composer = output_mesh_composer
        self.weights_mesh_mapper = weights_mesh_mapper
        self.kv_cache_per_batch_size = kv_cache_per_batch_size
        self.cross_attn_cache_per_batch_size = cross_attn_cache_per_batch_size

        # Cross-attention cache validity flag
        self.cross_attn_cache_valid = False

        self.trace_id_decoder = defaultdict(lambda: None)
        self.trace_input_decoder = defaultdict(lambda: None)
        self.trace_output_decoder = defaultdict(lambda: None)
        # self.trace_compiled = False

        # Pre-allocated encoder_hidden_states tensor
        # encoder_seq_len = 1500 for Whisper (30s max audio / 20ms per frame)
        encoder_seq_len = 1500
        self.encoder_hidden_states_per_size = defaultdict(lambda: None)
        for batch_size in [1, WHISPER_BATCH_SIZE]:
            self.encoder_hidden_states_per_size[batch_size] = ttnn.allocate_tensor_on_device(
                ttnn.Shape([batch_size, 1, encoder_seq_len, config.d_model]),
                ttnn.bfloat16,
                ttnn.TILE_LAYOUT,
                mesh_device,
                ttnn.DRAM_MEMORY_CONFIG,
            )

        # Pre-allocated device tensor for current decode position
        self.current_decode_pos_per_size = defaultdict(lambda: None)
        for batch_size in [1, WHISPER_BATCH_SIZE]:
            self.current_decode_pos_per_size[batch_size] = ttnn.allocate_tensor_on_device(
                ttnn.Shape([batch_size]),
                ttnn.int32,
                ttnn.ROW_MAJOR_LAYOUT,
                self.mesh_device,
                ttnn.DRAM_MEMORY_CONFIG,
            )

    def _get_batch_size_per_device(self, unpadded_batch_size):
        if unpadded_batch_size % self.mesh_device.get_num_devices() != 0:
            raise ValueError(
                f"Unpadded batch size {unpadded_batch_size} must be divisible by the number of devices {self.mesh_device.get_num_devices()}"
            )
        return unpadded_batch_size // self.mesh_device.get_num_devices()

    def _invalidate_cross_attn_cache(self):
        """Invalidate cross-attention cache for new generation."""
        self.cross_attn_cache_valid = False

    def _reset_decode_pos(self, value, global_batch_size):
        """Reset current_decode_pos to a specific value in-place

        Args:
            value: The position value to set (integer)
            global_batch_size: Total batch size across all devices
        """
        pos_host = torch.full((global_batch_size,), value, dtype=torch.int32)
        pos_tensor_host = ttnn.from_torch(pos_host, dtype=ttnn.int32, mesh_mapper=self.input_mesh_mapper)
        trace_key = self._get_batch_size_per_device(global_batch_size)
        ttnn.copy_host_to_device_tensor(pos_tensor_host, self.current_decode_pos_per_size[trace_key])

    def _release_decoder_trace(self):
        """Release the decoder trace resources (for cleanup)."""
        for trace_key in self.trace_id_decoder.keys():
            if self.trace_id_decoder[trace_key] is not None:
                ttnn.release_trace(self.mesh_device, self.trace_id_decoder[trace_key])
                self.trace_id_decoder[trace_key] = None
                self.trace_input_decoder[trace_key] = None
                self.trace_output_decoder[trace_key] = None
                logger.debug(f"Released decoder trace for batch size per device {trace_key}")

    def _capture_decoder_trace(self, trace_key, sample_decoder_hidden_states):
        """
        Capture decoder trace once, reuse for ALL subsequent generations.

        The trace is captured after the first decoder iteration when cross_attn_cache
        is already populated. The trace uses the pre-allocated encoder_hidden_states
        tensor which has a stable memory address.

        Args:
            sample_decoder_hidden_states: Sample input tensor for trace capture
        """
        if self.trace_id_decoder[trace_key] is not None:
            return  # Already captured

        # Create decoder function that will be traced
        # cross_attn_cache_valid=True because cache was just populated in iteration 0
        def traced_decoder_fn(trace_key, hidden_states):
            return ttnn_optimized_functional_whisper.decoder(
                self.config,
                hidden_states,
                decoder_attention_mask=None,
                encoder_hidden_states=self.encoder_hidden_states_per_size[trace_key],  # Pre-allocated, stable address
                kv_cache=self.kv_cache_per_batch_size[trace_key],
                cross_attn_cache=self.cross_attn_cache_per_batch_size[trace_key],
                cross_attn_cache_valid=True,
                current_decode_pos=self.current_decode_pos_per_size[trace_key],
                parameters=self.parameters.decoder,
            )

        # Move input to L1 for trace capture
        l1_memory_config = ttnn.L1_MEMORY_CONFIG
        l1_input = ttnn.to_memory_config(sample_decoder_hidden_states, l1_memory_config)

        # Compile run
        compile_output = traced_decoder_fn(trace_key, l1_input)
        ttnn.deallocate(compile_output, force=True)
        ttnn.deallocate(l1_input)
        logger.info("Decoder trace compile run complete")

        # Allocate L1 input for trace capture
        self.trace_input_decoder[trace_key] = ttnn.to_memory_config(sample_decoder_hidden_states, l1_memory_config)

        # Capture trace
        self.trace_id_decoder[trace_key] = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        self.trace_output_decoder[trace_key] = traced_decoder_fn(trace_key, self.trace_input_decoder[trace_key])

        ttnn.end_trace_capture(self.mesh_device, self.trace_id_decoder[trace_key], cq_id=0)
        ttnn.synchronize_device(self.mesh_device)

        logger.info(f"Persistent decoder trace capture complete for batch size per device {trace_key}")

    def _execute_decoder_trace(self, trace_key, decoder_hidden_states):
        """
        Execute the captured decoder trace with new input.

        Args:
            decoder_hidden_states: New decoder hidden states (on device)

        Returns:
            Decoder output tensor
        """
        if self.trace_id_decoder[trace_key] is None:
            raise RuntimeError("Decoder trace not captured. Call _capture_decoder_trace first.")

        # Copy new input to persistent L1 tensor
        self.trace_input_decoder[trace_key] = ttnn.to_memory_config(
            decoder_hidden_states,
            ttnn.L1_MEMORY_CONFIG,
            output_tensor=self.trace_input_decoder[trace_key],
        )

        # Execute trace
        ttnn.execute_trace(self.mesh_device, self.trace_id_decoder[trace_key], cq_id=0, blocking=True)

        return self.trace_output_decoder[trace_key]

    def generate(
        self,
        current_batch,
        generation_params: Optional[Union[GenerationParams, List[GenerationParams]]] = None,
        language: str = "en",
        task: str = "transcribe",
        prompt: Optional[str] = None,
        use_trace: bool = True,
        stream_generation=False,
        return_perf_metrics=False,
    ):
        """
        Generate transcription for audio batch with persistent trace support.

        Args:
            current_batch: List of (sampling_rate, audio_array) tuples
            generation_params: Single GenerationParams (broadcast to all) or list of per-request generation parameters
            language: Language code for transcription (batch-homogeneous)
            task: Task type ("transcribe" or "translate") (batch-homogeneous)
            prompt: Optional prompt to guide style/spelling (batch-homogeneous)
            use_trace: Whether to use traced execution for decoder (batch-homogeneous)
            stream_generation: Whether to stream tokens
            return_perf_metrics: Whether to return performance metrics

        Returns:
            Generated transcription and metrics
        """
        if generation_params is None:
            generation_params = [GenerationParams() for _ in range(len(current_batch))]
        elif isinstance(generation_params, list):
            if len(generation_params) < len(current_batch):
                for _ in range(len(generation_params), len(current_batch)):
                    generation_params.append(GenerationParams())
            elif len(generation_params) > len(current_batch):
                raise ValueError(
                    f"Generation parameters list cannot be longer than the current batch: {len(generation_params)} > {len(current_batch)}"
                )
        else:
            generation_params = [generation_params] * len(current_batch)

        temperatures = []
        compression_ratio_threshold = []
        logprob_threshold = []
        no_speech_threshold = []
        return_timestamps = []
        for params in generation_params:
            temperatures.append(params.temperatures)
            compression_ratio_threshold.append(params.compression_ratio_threshold)
            logprob_threshold.append(params.logprob_threshold)
            no_speech_threshold.append(params.no_speech_threshold)
            return_timestamps.append(params.return_timestamps)

        # Invalidate cross-attention cache for new generation
        self._invalidate_cross_attn_cache()

        # Process input features
        all_input_features = []
        start_encode = time.time()
        for sampling_rate, audio_array in current_batch:
            inputs = self.feature_extractor(
                audio_array,
                sampling_rate=sampling_rate,
                return_tensors="pt",
            )
            all_input_features.append(inputs.input_features)

        input_features = torch.cat(all_input_features, dim=0)
        del all_input_features
        unpadded_batch_size = input_features.shape[0]
        assert (
            unpadded_batch_size <= 2 * self.mesh_device.get_num_devices()
        ), "Only batch size (per device) 1 or 2 is supported for inference"

        # Calculate audio durations for timestamp capping
        audio_durations = self._calculate_audio_duration(current_batch) if any(return_timestamps) else None

        # Compute encoder embeddings
        input_embeds = ttnn_optimized_functional_whisper.preprocess_encoder_inputs(
            config=self.config,
            input_features=input_features.unsqueeze(1),
            parameters=self.parameters.encoder,
            device=self.mesh_device,
            weights_mesh_mapper=self.weights_mesh_mapper,
            input_mesh_mapper=self.input_mesh_mapper,
        )

        # Run encoder
        encoder_output = ttnn_optimized_functional_whisper.encoder(
            config=self.config,
            inputs_embeds=input_embeds,
            parameters=self.parameters.encoder,
        )

        # Copy encoder output to pre-allocated tensor
        ttnn.copy(
            encoder_output, self.encoder_hidden_states_per_size[self._get_batch_size_per_device(unpadded_batch_size)]
        )
        ttnn.synchronize_device(self.mesh_device)
        logger.info(f"Time to encoder states: {(time.time() - start_encode)*1000:.3f}ms")

        # Collect temperatures to try: flatten per-request temps to unique sequence
        temps_to_try = []
        for t in temperatures:
            temps_to_try.extend((t,) if isinstance(t, (int, float)) else t)
        temps_to_try = list(dict.fromkeys(temps_to_try))  # unique, preserve order

        # For streaming mode, skip temperature fallback and quality checks
        # Use only the first temperature and yield tokens immediately
        if stream_generation:
            temperature = temps_to_try[0] if temps_to_try else 0.0
            logger.info(f"Streaming mode: using temperature {temperature}, skipping quality checks")

            return self._generate_with_temperature(
                temperature=temperature,
                start_encode=start_encode,
                input_features=input_features.unsqueeze(1),
                unpadded_batch_size=unpadded_batch_size,
                return_perf_metrics=return_perf_metrics,
                return_timestamps=return_timestamps,
                audio_durations=audio_durations,
                language=language,
                task=task,
                prompt=prompt,
                streaming=True,
                use_trace=use_trace,
            )

        # Non-streaming mode: Try generation with different temperatures
        best_output = None
        best_quality_score = float("inf")

        for temperature in temps_to_try:
            logger.info(f"Trying generation with temperature: {temperature}")

            try:
                output = self._generate_with_temperature(
                    temperature=temperature,
                    start_encode=start_encode,
                    input_features=input_features.unsqueeze(1),
                    unpadded_batch_size=unpadded_batch_size,
                    return_perf_metrics=return_perf_metrics,
                    return_timestamps=return_timestamps,
                    audio_durations=audio_durations,
                    language=language,
                    task=task,
                    prompt=prompt,
                    streaming=False,
                    use_trace=use_trace,
                )

                # Non-streaming generation - consume the generator
                if return_perf_metrics:
                    result_data, avg_logprobs, no_speech_probs, ttft, throughput = next(output)
                else:
                    result_data, avg_logprobs, no_speech_probs = next(output)

                # Check quality for each result
                all_good = True
                for idx, data in enumerate(result_data):
                    wants_timestamps = (
                        return_timestamps[idx] if isinstance(return_timestamps, list) else return_timestamps
                    )
                    if wants_timestamps:
                        # For timestamps, extract text from segments for quality check
                        text = " ".join([segment["text"] for segment in data])
                    else:
                        text = data

                    compression_ratio = self._calculate_compression_ratio(text)
                    # Extract per-batch-item metrics
                    avg_logprob = avg_logprobs[idx].item() if idx < len(avg_logprobs) else DEFAULT_AVG_LOGPROB
                    no_speech_prob = (
                        no_speech_probs[idx].item() if idx < len(no_speech_probs) else DEFAULT_NO_SPEECH_PROB
                    )

                    is_good, reason = self._check_generation_quality(
                        text,
                        avg_logprob,
                        no_speech_prob,
                        compression_ratio,
                        logprob_threshold[idx],
                        compression_ratio_threshold[idx],
                        no_speech_threshold[idx],
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
                    self._calculate_compression_ratio(
                        " ".join([segment["text"] for segment in data])
                        if (return_timestamps[idx] if isinstance(return_timestamps, list) else return_timestamps)
                        else data
                    )
                    for idx, data in enumerate(result_data)
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
            if any(return_timestamps) if isinstance(return_timestamps, list) else return_timestamps:
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
        self,
        temperature,
        start_encode,
        input_features,
        unpadded_batch_size,
        return_perf_metrics=False,
        return_timestamps=False,
        audio_durations=None,
        language="en",
        task="transcribe",
        prompt=None,
        streaming=False,
        use_trace=True,
    ):
        """
        Generate text with a specific temperature using fully persistent traces.

        Uses pre-allocated self.encoder_hidden_states instead of a passed encoder_hidden_states parameter.
        """
        return_timestamps_for_prefix = (
            any(return_timestamps) if isinstance(return_timestamps, list) else return_timestamps
        )

        trace_key = self._get_batch_size_per_device(unpadded_batch_size)

        # Input ids - use forced decoder IDs for translation
        forced_decoder_ids = self.processor.get_decoder_prompt_ids(language=language, task=task)

        # Keep forced_decoder_ids as tuples with positions
        # When return_timestamps=True, remove <|notimestamps|> to allow timestamp generation
        if return_timestamps_for_prefix:
            # Remove notimestamps token if present
            forced_decoder_ids = [(pos, tok) for pos, tok in forced_decoder_ids if tok != NOTIMESTAMPS_TOKEN_ID]
        else:
            # Add notimestamps token if not present (at the appropriate position)
            if not any(tok == NOTIMESTAMPS_TOKEN_ID for _, tok in forced_decoder_ids):
                # Find the last position and add notimestamps after it
                max_pos = max((pos for pos, _ in forced_decoder_ids), default=0)
                forced_decoder_ids.append((max_pos + 1, NOTIMESTAMPS_TOKEN_ID))

        # When prompt is provided, the sequence becomes:
        # <|startofprev|> -> [prompt tokens] -> <|startoftranscript|> -> <|language|> -> <|task|> -> ...
        prompt_offset = 0
        if prompt is not None:
            # Tokenize the prompt
            prompt_tokens = self.processor.tokenizer.encode(prompt, add_special_tokens=False)

            # Truncate prompt to MAX_PROMPT_TOKENS if needed
            if len(prompt_tokens) > MAX_PROMPT_TOKENS:
                logger.warning(f"Prompt has {len(prompt_tokens)} tokens, truncating to {MAX_PROMPT_TOKENS} tokens")
                prompt_tokens = prompt_tokens[:MAX_PROMPT_TOKENS]

            # Build forced tokens dict
            forced_tokens_dict = {0: STARTOFPREV_TOKEN_ID}
            for i, token in enumerate(prompt_tokens):
                forced_tokens_dict[i + 1] = token
            # Add <|startoftranscript|> after prompt
            forced_tokens_dict[len(prompt_tokens) + 1] = STARTOFTRANSCRIPT_TOKEN_ID
            # Offset for the rest of the forced decoder ids
            prompt_offset = len(prompt_tokens) + 1  # +1 for <|startofprev|>

            # Shift forced_decoder_ids positions to account for prompt prefix
            for pos, tok in forced_decoder_ids:
                forced_tokens_dict[pos + prompt_offset] = tok

        else:
            # Create a position-to-token mapping
            forced_tokens_dict = {pos: token_id for pos, token_id in forced_decoder_ids}
            # Add decoder_start_token at position 0
            forced_tokens_dict[0] = self.config.decoder_start_token_id

        # Calculate where actual transcription starts (after all forced tokens including prompt)
        # This is the first position that is NOT a forced token
        transcription_start_pos = max(forced_tokens_dict.keys()) + 1 if forced_tokens_dict else 0

        # Build the full prefix sequence from forced_tokens_dict
        prefix_sequence = [forced_tokens_dict[pos] for pos in sorted(forced_tokens_dict.keys())]
        prefix_len = len(prefix_sequence)

        # Initialize input_ids with the full prefix sequence for proper conditioning
        input_ids = torch.tensor([prefix_sequence]).repeat(input_features.shape[0], 1).to(torch.long)
        logits_processor = get_logits_processor(input_ids, self.config)

        if not self.kv_cache_per_batch_size[trace_key]:
            input_ids = self._pad_input_32(input_ids, self.config.pad_token_id).to(torch.long)
            decoder_start_values = self.generation_config.pad_token_id * torch.ones(1, 32).to(torch.long)

        MAX_GEN_LEN = self.config.max_length
        output_ids = []
        total_decode_time = 0
        prompt_is_done = [False for _ in range(unpadded_batch_size)]
        log_probs = []  # Track log probabilities
        no_speech_probs = None  # Will be extracted from first frame logits

        # Track full token sequences for timestamp extraction
        full_token_sequences = [[] for _ in range(unpadded_batch_size)] if return_timestamps_for_prefix else None

        # Non-streaming mode: collect all results in a list
        if not streaming:
            output = [[] for _ in range(input_features.shape[0])]
        ttft = 0.0
        avg_decode_throughput = 0.0

        # Run prefill pass for KV cache mode to populate cache with prompt context
        # Process all prefix tokens; the last iteration samples the first transcription token
        # NOTE: This is sub-optimal - processing tokens one at a time (decode-style prefill)
        # rather than a single batched prefill forward pass. A true prefill implementation
        # would process all prefix tokens in one forward pass for better performance.
        if self.kv_cache_per_batch_size[trace_key] and prompt is not None and prefix_len > 1:
            logger.debug(f"Running prefill pass for {prefix_len} prefix tokens")
            first_transcription_token = None

            for prefill_pos in range(prefix_len):
                prefill_input = input_ids[:, prefill_pos : prefill_pos + 1]
                self._reset_decode_pos(prefill_pos, unpadded_batch_size)

                (
                    decoder_hidden_states,
                    decoder_attention_mask,
                ) = ttnn_optimized_functional_whisper.preprocess_decoder_inputs(
                    config=self.config,
                    input_ids=prefill_input,
                    attention_mask=None,
                    parameters=self.parameters.decoder,
                    device=self.mesh_device,
                    decode_pos=prefill_pos,
                    create_attention_mask=False,
                    input_mesh_mapper=self.input_mesh_mapper,
                )

                if (
                    use_trace
                    and self.kv_cache_per_batch_size[trace_key]
                    and self.trace_id_decoder[trace_key]
                    and prefill_pos > 0
                ):
                    decoder_output = self._execute_decoder_trace(trace_key, decoder_hidden_states)
                else:
                    decoder_output = ttnn_optimized_functional_whisper.decoder(
                        self.config,
                        decoder_hidden_states,
                        decoder_attention_mask=decoder_attention_mask,
                        encoder_hidden_states=self.encoder_hidden_states_per_size[trace_key],
                        kv_cache=self.kv_cache_per_batch_size[trace_key],
                        cross_attn_cache=self.cross_attn_cache_per_batch_size[trace_key],
                        cross_attn_cache_valid=self.cross_attn_cache_valid,
                        current_decode_pos=self.current_decode_pos_per_size[trace_key],
                        parameters=self.parameters.decoder,
                    )

                    # After first prefill iteration, cross_attn_cache is populated
                    if prefill_pos == 0:
                        self.cross_attn_cache_valid = True
                        # Capture trace for reuse in subsequent prefill and decode iterations
                        if (
                            use_trace
                            and self.kv_cache_per_batch_size[trace_key]
                            and not self.trace_id_decoder[trace_key]
                        ):
                            self._capture_decoder_trace(trace_key, decoder_hidden_states)

                # On last prefill iteration, sample the first transcription token
                if prefill_pos == prefix_len - 1:
                    # Squeeze extra dimension from 4D [batch, 1, seq, hidden] to 3D [batch, seq, hidden]
                    decoder_output = ttnn.squeeze(decoder_output, 1)
                    decoder_output = decoder_output @ self.ttnn_linear_weight
                    logits_to_torch = ttnn.to_torch(decoder_output, mesh_composer=self.output_mesh_composer)
                    next_token_logits = logits_to_torch[:, 0, :]
                    next_tokens_scores = logits_processor(input_ids, next_token_logits)
                    first_transcription_token = self._sample_token(next_tokens_scores, temperature)

                    # Record TTFT
                    ttft = time.time() - start_encode

                    # Extract no_speech probability from first frame logits
                    with torch.no_grad():
                        probs = torch.softmax(next_token_logits, dim=-1)
                        no_speech_probs = probs[:, NO_SPEECH_TOKEN_ID]

                        # Track log probabilities for first transcription token
                        log_probs.append(
                            torch.log_softmax(next_tokens_scores, dim=-1)
                            .gather(1, first_transcription_token.unsqueeze(1))
                            .squeeze(1)
                        )

                    output_ids.append(first_transcription_token)

                    # Track full token sequences for timestamp extraction
                    if return_timestamps_for_prefix:
                        for batch_idx in range(unpadded_batch_size):
                            full_token_sequences[batch_idx].append(first_transcription_token[batch_idx].item())

                    for user_id, user_decode_id in enumerate(first_transcription_token[:unpadded_batch_size]):
                        if user_decode_id == self.config.eos_token_id:
                            prompt_is_done[user_id] = True

                    # If streaming, yield the first token
                    if streaming:
                        ttnn_transcription = self.processor.batch_decode(
                            first_transcription_token.unsqueeze(dim=1), skip_special_tokens=True
                        )
                        current_avg_logprob = (
                            log_probs[0].unsqueeze(0) if log_probs else torch.zeros(input_features.shape[0])
                        )
                        if len(log_probs) > 1:
                            current_avg_logprob = torch.stack(log_probs, dim=1).mean(dim=1)

                        if return_perf_metrics:
                            yield ttnn_transcription, current_avg_logprob, no_speech_probs, ttft, 0.0, False
                        else:
                            yield ttnn_transcription, current_avg_logprob, no_speech_probs, False
                    else:
                        # Non-streaming mode: collect the first token
                        ttnn_transcription = self.processor.batch_decode(
                            first_transcription_token.unsqueeze(dim=1), skip_special_tokens=True
                        )
                        for idx in range(input_features.shape[0]):
                            output[idx].append(ttnn_transcription[idx])

            # Set decode position to prefix_len for generation to continue
            self._reset_decode_pos(prefix_len, unpadded_batch_size)
            # Set input_ids to the first transcription token (sampled during prefill)
            input_ids = first_transcription_token[:, None]
        else:
            # For KV cache mode without prefill: reset decode position to 0 and set input_ids to the first transcription token
            if self.kv_cache_per_batch_size[trace_key]:
                self._reset_decode_pos(0, unpadded_batch_size)
                # For KV cache mode without prefill, start with just the first token
                input_ids = input_ids[:, :1]

        # Generation loop start: if prefill ran, first token was already sampled, so start from transcription_start_pos + 1
        # Otherwise start from 0
        if self.kv_cache_per_batch_size[trace_key] and prompt is not None:
            generation_start = transcription_start_pos + 1
        else:
            generation_start = 0

        # Skip decode loop if all prompts finished during prefill (e.g., first token was EOS)
        if all(prompt_is_done):
            generation_start = MAX_GEN_LEN

        for i in tqdm(range(generation_start, MAX_GEN_LEN), desc=f"Decode inference iterations (temp={temperature})"):
            start_iter = time.time()

            decoder_hidden_states, decoder_attention_mask = ttnn_optimized_functional_whisper.preprocess_decoder_inputs(
                config=self.config,
                input_ids=input_ids,
                attention_mask=None,
                parameters=self.parameters.decoder,
                device=self.mesh_device,
                decode_pos=i if self.kv_cache_per_batch_size[trace_key] else None,
                create_attention_mask=(not self.kv_cache_per_batch_size[trace_key]),
                input_mesh_mapper=self.input_mesh_mapper,
            )

            # Use persistent trace for iterations after the first
            if use_trace and self.kv_cache_per_batch_size[trace_key] and self.trace_id_decoder[trace_key] and i > 0:
                # Execute persistent trace (reused across ALL generations)
                decoder_output = self._execute_decoder_trace(trace_key, decoder_hidden_states)
            else:
                # Regular decoder execution (first iteration or trace disabled)
                decoder_output = ttnn_optimized_functional_whisper.decoder(
                    self.config,
                    decoder_hidden_states,
                    decoder_attention_mask=decoder_attention_mask,
                    encoder_hidden_states=self.encoder_hidden_states_per_size[trace_key],
                    kv_cache=self.kv_cache_per_batch_size[trace_key],
                    current_decode_pos=self.current_decode_pos_per_size[trace_key],
                    cross_attn_cache=self.cross_attn_cache_per_batch_size[trace_key],
                    cross_attn_cache_valid=self.cross_attn_cache_valid,
                    parameters=self.parameters.decoder,
                )

                # After first iteration, cross_attn_cache is populated with valid K/V
                if i == generation_start:
                    self.cross_attn_cache_valid = True

                # Capture trace after first iteration (cross-attention cache is now populated)
                if (
                    use_trace
                    and self.kv_cache_per_batch_size[trace_key]
                    and i == generation_start
                    and not self.trace_id_decoder[trace_key]
                ):
                    logger.info(f"Capturing fully persistent decoder trace for batch size per device {trace_key}")
                    self._capture_decoder_trace(trace_key, decoder_hidden_states)

            if not self.kv_cache_per_batch_size[trace_key]:
                # Note: if not using a kv cache, the entire sequence is recomputed at each step
                # Only run the lm head on the last tile to fix bad outputs and reduce redundant computation
                last_tile_start_idx = i // 32 * 32
                output_idx = i % 32
                decoder_output = decoder_output[:, last_tile_start_idx : last_tile_start_idx + 32, :]
            else:
                output_idx = 0

            # Squeeze extra dimension from 4D [batch, 1, seq, hidden] to 3D [batch, seq, hidden]
            decoder_output = ttnn.squeeze(decoder_output, 1)
            decoder_output = decoder_output @ self.ttnn_linear_weight
            logits_to_torch = ttnn.to_torch(decoder_output, mesh_composer=self.output_mesh_composer)
            next_token_logits = logits_to_torch[:, output_idx, :]
            next_tokens_scores = logits_processor(input_ids, next_token_logits)

            # Force tokens at specific positions based on forced_tokens_dict
            if i in forced_tokens_dict:
                next_tokens = torch.tensor([forced_tokens_dict[i]]).repeat(input_features.shape[0])
            else:
                next_tokens = self._sample_token(next_tokens_scores, temperature)

            # Only collect output tokens after the forced prefix (prompt + special tokens)
            # This ensures prompt text doesn't appear in the transcription
            if i >= transcription_start_pos:
                # Track log probabilities for actual transcription tokens only
                with torch.no_grad():
                    log_probs.append(
                        torch.log_softmax(next_tokens_scores, dim=-1).gather(1, next_tokens.unsqueeze(1)).squeeze(1)
                    )

                output_ids.append(next_tokens)

                # Track full token sequences for timestamp extraction
                if return_timestamps_for_prefix:
                    for batch_idx in range(unpadded_batch_size):
                        full_token_sequences[batch_idx].append(next_tokens[batch_idx].item())

            # Record TTFT on first decode iteration (only if not already set by prefill)
            if i == generation_start and ttft == 0.0:
                first_token_time = time.time()
                ttft = first_token_time - start_encode
                # Extract no_speech probability from first frame logits
                with torch.no_grad():
                    probs = torch.softmax(next_token_logits, dim=-1)
                    no_speech_probs = probs[:, NO_SPEECH_TOKEN_ID]

            # Update input_ids and current_decode_pos
            if not self.kv_cache_per_batch_size[trace_key]:
                if (i + 1) % 32 == 0:
                    input_ids = torch.cat([input_ids, decoder_start_values], dim=1)
                input_ids[:, i + 1] = next_tokens[:, None]
            else:
                input_ids = next_tokens[:, None]
                ttnn.plus_one(self.current_decode_pos_per_size[trace_key])

            total_decode_time += time.time() - start_iter
            # Calculate throughput based on tokens generated (not including prefix)
            tokens_generated = i - generation_start + 1
            avg_decode_throughput = tokens_generated / total_decode_time

            for user_id, user_decode_id in enumerate(next_tokens[:unpadded_batch_size]):
                if user_decode_id == self.config.eos_token_id:
                    prompt_is_done[user_id] = True
                if prompt_is_done[user_id]:
                    next_tokens[user_id] = self.config.eos_token_id

            # Only output transcription tokens (skip prompt and forced prefix tokens)
            if i >= transcription_start_pos:
                ttnn_transcription = self.processor.batch_decode(next_tokens.unsqueeze(dim=1), skip_special_tokens=True)

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
                    # is_final=False indicates this is an intermediate token, not the final result
                    if return_perf_metrics:
                        yield ttnn_transcription, current_avg_logprob, current_no_speech_probs, ttft, avg_decode_throughput, False
                    else:
                        yield ttnn_transcription, current_avg_logprob, current_no_speech_probs, False
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
        if return_timestamps_for_prefix and full_token_sequences:
            # Extract timestamps for each batch item
            segments_with_timestamps = []
            for batch_idx in range(unpadded_batch_size):
                if full_token_sequences[batch_idx]:
                    token_sequence = torch.tensor(full_token_sequences[batch_idx])
                    audio_duration = audio_durations[batch_idx] if audio_durations else None
                    segments = self._extract_timestamps_from_tokens(token_sequence, self.processor, audio_duration)
                    segments_with_timestamps.append(segments)
                else:
                    segments_with_timestamps.append([])

            # Per-item format: segments for items that want timestamps, plain text for others
            return_timestamps_per_item = (
                return_timestamps if isinstance(return_timestamps, list) else [return_timestamps] * unpadded_batch_size
            )
            final_result = []
            for batch_idx in range(unpadded_batch_size):
                if return_timestamps_per_item[batch_idx]:
                    final_result.append(segments_with_timestamps[batch_idx])
                else:
                    final_result.append(" ".join(seg["text"] for seg in segments_with_timestamps[batch_idx]).strip())

            # Yield final result (works for both streaming and non-streaming)
            # For streaming mode, include is_final=True to mark this as the final result
            if return_perf_metrics:
                if streaming:
                    yield final_result, avg_logprob, no_speech_probs, ttft, avg_decode_throughput, True
                else:
                    yield final_result, avg_logprob, no_speech_probs, ttft, avg_decode_throughput
            else:
                if streaming:
                    yield final_result, avg_logprob, no_speech_probs, True
                else:
                    yield final_result, avg_logprob, no_speech_probs
        else:
            if streaming:
                # For streaming without timestamps, yield final accumulated result
                # Accumulate all tokens from output_ids
                final_output = []
                for batch_idx in range(unpadded_batch_size):
                    # Collect all tokens for this batch item
                    batch_tokens = [output_ids[i][batch_idx] for i in range(len(output_ids))]
                    # Decode the full sequence
                    decoded_text = self.processor.batch_decode(
                        torch.tensor(batch_tokens).unsqueeze(0), skip_special_tokens=True
                    )[0]
                    final_output.append(decoded_text.strip())

                # is_final=True indicates this is the final batch-decoded result
                if return_perf_metrics:
                    yield final_output, avg_logprob, no_speech_probs, ttft, avg_decode_throughput, True
                else:
                    yield final_output, avg_logprob, no_speech_probs, True
            else:
                # Join the collected tokens into final text and strip leading/trailing whitespace
                output = ["".join(tokens).strip() for tokens in output]
                if return_perf_metrics:
                    yield (output, avg_logprob, no_speech_probs, ttft, avg_decode_throughput)
                else:
                    yield (output, avg_logprob, no_speech_probs)

    def cleanup(self):
        """Release trace resources."""
        if self.trace_id_decoder:
            self._release_decoder_trace()
            logger.info("Released decoder trace resources")

    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.cleanup()
        except Exception:
            pass  # Ignore errors during cleanup

    @staticmethod
    def _pad_input_32(tensor, value):
        """Pad input to multiple of 32."""
        len_tensor = tensor.shape[1]
        if len_tensor % 32 == 0:
            return tensor
        padded_len = ((len_tensor // 32) + 1) * 32
        pad_tensor = (value * torch.ones(tensor.shape[0], padded_len - len_tensor)).to(torch.long)
        tensor = torch.cat([tensor, pad_tensor], dim=1)
        return tensor

    @staticmethod
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

    @staticmethod
    def _calculate_compression_ratio(text: str) -> float:
        """
        Calculate compression ratio of text using zlib.
        """
        if not text:
            return 0.0
        text_bytes = text.encode("utf-8")
        compressed_bytes = zlib.compress(text_bytes)
        return len(text_bytes) / len(compressed_bytes)

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def _calculate_audio_duration(current_batch) -> List[float]:
        """
        Calculate audio duration for each item in the batch.
        """
        durations = []
        for sampling_rate, audio_array in current_batch:
            duration = len(audio_array) / sampling_rate
            durations.append(duration)
        return durations
