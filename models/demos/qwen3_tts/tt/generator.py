# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
2-Command-Queue Generator for Qwen3-TTS with Streaming Audio Support.

This module provides:
- Trace capture for prefill and decode modes
- 2CQ support for async token transfer to host
- Streaming audio decoding while generation continues
- Pre-allocated tensors for efficient execution

The 2CQ pattern overlaps:
- CQ0: Model execution (traced)
- CQ1: Async token transfer to host + position updates
- CPU: Audio decoding in parallel
"""

import queue as queue_module
import threading
from typing import Callable, List, Optional, Tuple

import torch
from loguru import logger

import ttnn
from models.demos.qwen3_tts.tt.kv_cache import create_kv_cache
from models.demos.qwen3_tts.tt.model_config import Qwen3TTSCodePredictorConfig, Qwen3TTSTalkerConfig
from models.demos.qwen3_tts.tt.qwen3_tts import Qwen3TTS
from models.demos.qwen3_tts.tt.rope import get_rope_tensors, get_transformation_mat


class StreamingAudioDecoder:
    """
    Background thread for streaming audio decoding.

    Receives token chunks and decodes them to audio in parallel
    with token generation.
    """

    def __init__(
        self,
        decoder_fn: Callable[[torch.Tensor], torch.Tensor],
        chunk_size: int = 50,
        sample_rate: int = 24000,
    ):
        """
        Initialize streaming decoder.

        Args:
            decoder_fn: Function that takes [batch, num_quantizers, seq_len] tokens
                       and returns [batch, 1, num_samples] audio
            chunk_size: Number of tokens per audio chunk
            sample_rate: Audio sample rate for callback
        """
        self.decoder_fn = decoder_fn
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate

        self.token_queue = queue_module.Queue()
        self.audio_queue = queue_module.Queue()
        self.stop_event = threading.Event()
        self.decoder_thread = None

        # Accumulated tokens
        self.all_tokens = []
        self.decoded_up_to = 0

    def start(self):
        """Start the decoder thread."""
        self.stop_event.clear()
        self.all_tokens = []
        self.decoded_up_to = 0
        self.decoder_thread = threading.Thread(target=self._decode_loop, daemon=True)
        self.decoder_thread.start()
        logger.info("Streaming audio decoder started")

    def stop(self):
        """Stop the decoder thread and decode remaining tokens."""
        self.stop_event.set()
        self.token_queue.put(None)  # Signal to stop
        if self.decoder_thread is not None:
            self.decoder_thread.join(timeout=5.0)
        logger.info("Streaming audio decoder stopped")

    def add_tokens(self, tokens: torch.Tensor):
        """
        Add generated tokens to the queue.

        Args:
            tokens: Token tensor [num_quantizers] for single step
        """
        self.token_queue.put(tokens.clone())

    def get_audio_chunk(self, timeout: float = 0.1) -> Optional[torch.Tensor]:
        """
        Get next decoded audio chunk (non-blocking).

        Returns:
            Audio chunk [num_samples] or None if not available
        """
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue_module.Empty:
            return None

    def get_all_audio(self) -> torch.Tensor:
        """
        Get all decoded audio after generation completes.

        Returns:
            Full audio tensor [batch, 1, num_samples]
        """
        # Collect any remaining chunks
        chunks = []
        while True:
            try:
                chunk = self.audio_queue.get_nowait()
                if chunk is not None:
                    chunks.append(chunk)
            except queue_module.Empty:
                break

        # Decode any remaining tokens
        if self.all_tokens and self.decoded_up_to < len(self.all_tokens):
            remaining = torch.stack(self.all_tokens[self.decoded_up_to :], dim=-1)  # [16, remaining]
            remaining = remaining.unsqueeze(0)  # [1, 16, remaining]
            audio = self.decoder_fn(remaining)
            chunks.append(audio.squeeze())

        if chunks:
            return torch.cat(chunks, dim=-1)
        return torch.tensor([])

    def _decode_loop(self):
        """Background decode loop."""
        while not self.stop_event.is_set():
            try:
                tokens = self.token_queue.get(timeout=0.1)
                if tokens is None:
                    break

                self.all_tokens.append(tokens)

                # Check if we have enough tokens for a chunk
                num_tokens = len(self.all_tokens)
                if num_tokens - self.decoded_up_to >= self.chunk_size:
                    # Decode chunk
                    chunk_tokens = self.all_tokens[self.decoded_up_to : self.decoded_up_to + self.chunk_size]
                    chunk_tensor = torch.stack(chunk_tokens, dim=-1)  # [16, chunk_size]
                    chunk_tensor = chunk_tensor.unsqueeze(0)  # [1, 16, chunk_size]

                    try:
                        audio_chunk = self.decoder_fn(chunk_tensor)
                        self.audio_queue.put(audio_chunk.squeeze())
                        self.decoded_up_to += self.chunk_size
                        logger.debug(
                            f"Decoded chunk: tokens {self.decoded_up_to - self.chunk_size}-{self.decoded_up_to}"
                        )
                    except Exception as e:
                        logger.error(f"Error decoding chunk: {e}")

            except queue_module.Empty:
                continue


class Qwen3TTSGenerator2CQ:
    """
    2-Command-Queue Generator for Qwen3-TTS with streaming support.

    Features:
    - Trace capture for prefill and decode modes
    - 2CQ for async token transfer to host
    - Streaming audio callback while generating
    - Pre-allocated tensors to avoid allocation during trace
    """

    def __init__(
        self,
        model: Qwen3TTS,
        device,
        talker_config: Qwen3TTSTalkerConfig,
        code_predictor_config: Qwen3TTSCodePredictorConfig,
        max_batch_size: int = 1,
        max_seq_len: int = 2048,
        use_2cq: bool = True,
    ):
        """
        Initialize the 2CQ generator.

        Args:
            model: Qwen3TTS model
            device: TTNN device
            talker_config: Talker configuration
            code_predictor_config: CodePredictor configuration
            max_batch_size: Maximum batch size
            max_seq_len: Maximum sequence length
            use_2cq: Enable 2-command-queue mode for async transfers
        """
        self.model = model
        self.device = device
        self.talker_config = talker_config
        self.code_predictor_config = code_predictor_config
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.use_2cq = use_2cq

        # Trace IDs
        self.prefill_trace_id = None
        self.decode_trace_id = None

        # Pre-allocated tensors for trace execution
        self.prefill_inputs = None
        self.prefill_output = None
        self.decode_inputs = None
        self.decode_output = None

        # RoPE tensors
        self.talker_trans_mat = None
        self.cp_trans_mat = None

        # KV caches
        self.talker_kv_cache = None
        self.cp_kv_cache = None

        # Trace state
        self.prefill_trace_captured = False
        self.decode_trace_captured = False

        # 2CQ event tracking
        self.cq0_event = None  # Event recorded on CQ0
        self.cq1_event = None  # Event recorded on CQ1

        # Pre-allocated position tensor for CQ1 updates
        self.position_tensor = None

        # Streaming decoder
        self.streaming_decoder = None

    def setup(self):
        """Setup generator with pre-allocated tensors."""
        logger.info("Setting up Qwen3-TTS 2CQ generator...")

        # Pre-compute transformation matrices
        self.talker_trans_mat = get_transformation_mat(self.talker_config.head_dim, self.device)
        self.cp_trans_mat = get_transformation_mat(self.code_predictor_config.head_dim, self.device)

        # Create KV caches
        self.talker_kv_cache = create_kv_cache(
            self.device,
            self.talker_config,
            self.max_batch_size,
            self.max_seq_len,
        )
        self.cp_kv_cache = create_kv_cache(
            self.device,
            self.code_predictor_config,
            self.max_batch_size,
            self.max_seq_len,
        )

        # Pre-allocate position tensor for 2CQ updates
        self.position_tensor = ttnn.from_torch(
            torch.zeros(self.max_batch_size, dtype=torch.int32),
            dtype=ttnn.int32,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        logger.info("2CQ Generator setup complete")

    def set_streaming_decoder(self, decoder_fn: Callable, chunk_size: int = 50):
        """
        Set up streaming audio decoder.

        Args:
            decoder_fn: Function to decode tokens to audio
            chunk_size: Tokens per audio chunk
        """
        self.streaming_decoder = StreamingAudioDecoder(
            decoder_fn=decoder_fn,
            chunk_size=chunk_size,
        )
        logger.info(f"Streaming decoder configured with chunk_size={chunk_size}")

    def _update_position_async(self, position: int, cq_id: int = 1):
        """
        Update position tensor asynchronously on specified command queue.

        Args:
            position: New position value
            cq_id: Command queue ID (default 1 for async)
        """
        pos_host = torch.full((self.max_batch_size,), position, dtype=torch.int32)
        pos_tensor_host = ttnn.from_torch(pos_host, dtype=ttnn.int32)
        ttnn.copy_host_to_device_tensor(pos_tensor_host, self.position_tensor, cq_id=cq_id)

    def warmup_decode(self):
        """Warmup decode forward pass."""
        logger.info("Warming up decode...")

        seq_len = 1
        warmup_input = torch.zeros(self.max_batch_size, seq_len, dtype=torch.long)
        warmup_input_tt = ttnn.from_torch(
            warmup_input,
            device=self.device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        position_ids = torch.tensor([128])
        talker_cos, talker_sin = get_rope_tensors(
            self.device,
            self.talker_config.head_dim,
            1,
            position_ids,
            self.talker_config.rope_theta,
        )
        cp_cos, cp_sin = get_rope_tensors(
            self.device,
            self.code_predictor_config.head_dim,
            1,
            position_ids,
            self.code_predictor_config.rope_theta,
        )

        _ = self.model.forward(
            warmup_input_tt,
            talker_cos,
            talker_sin,
            self.talker_trans_mat,
            cp_cos,
            cp_sin,
            self.cp_trans_mat,
        )

        ttnn.synchronize_device(self.device)
        logger.info("Decode warmup complete")

    def capture_decode_trace(self, start_pos: int = 128):
        """Capture trace for decode mode."""
        logger.info(f"Capturing decode trace at position {start_pos}...")

        seq_len = 1

        # Pre-allocate input tensor
        input_tensor = ttnn.from_torch(
            torch.zeros(self.max_batch_size, seq_len, dtype=torch.long),
            device=self.device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        position_ids = torch.tensor([start_pos])
        talker_cos, talker_sin = get_rope_tensors(
            self.device,
            self.talker_config.head_dim,
            1,
            position_ids,
            self.talker_config.rope_theta,
        )
        cp_cos, cp_sin = get_rope_tensors(
            self.device,
            self.code_predictor_config.head_dim,
            1,
            position_ids,
            self.code_predictor_config.rope_theta,
        )

        # Begin trace capture on CQ0
        self.decode_trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)

        output = self.model.forward(
            input_tensor,
            talker_cos,
            talker_sin,
            self.talker_trans_mat,
            cp_cos,
            cp_sin,
            self.cp_trans_mat,
        )

        ttnn.end_trace_capture(self.device, self.decode_trace_id, cq_id=0)

        self.decode_inputs = {
            "input_ids": input_tensor,
            "talker_cos": talker_cos,
            "talker_sin": talker_sin,
            "cp_cos": cp_cos,
            "cp_sin": cp_sin,
        }
        self.decode_output = output

        self.decode_trace_captured = True
        ttnn.synchronize_device(self.device)
        logger.info("Decode trace captured")

    def generate_streaming(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        prefill_output: Optional[List[ttnn.Tensor]] = None,
        start_pos: int = 0,
        audio_callback: Optional[Callable[[torch.Tensor], None]] = None,
        return_stats: bool = True,
    ) -> Tuple[List[torch.Tensor], dict]:
        """
        Generate tokens with 2CQ streaming.

        This method:
        1. Executes decode trace on CQ0 (non-blocking)
        2. Transfers tokens to host on CQ1 (async)
        3. Calls audio_callback with decoded audio chunks (parallel)

        Args:
            input_ids: Initial input tokens [batch, seq_len]
            max_new_tokens: Maximum tokens to generate
            prefill_output: Optional prefill output to continue from
            start_pos: Starting position in sequence
            audio_callback: Optional callback for streaming audio chunks
            return_stats: Whether to return detailed timing stats

        Returns:
            Tuple of (generated_tokens_list, stats_dict)
        """
        import time

        if not self.decode_trace_captured:
            raise RuntimeError("Decode trace not captured. Call capture_decode_trace first.")

        generated_tokens = []
        current_pos = start_pos + input_ids.shape[1]

        # Performance tracking
        generation_start = time.time()
        ttft = None  # Time to first token
        decode_times = []
        first_audio_chunk_time = None

        # Start streaming decoder if configured
        if self.streaming_decoder is not None:
            self.streaming_decoder.start()

        # Initialize 2CQ events
        self.cq0_event = None
        self.cq1_event = None

        logger.info(f"Starting 2CQ generation: max_tokens={max_new_tokens}, use_2cq={self.use_2cq}")
        decode_loop_start = time.time()

        try:
            for step in range(max_new_tokens):
                step_start = time.time()

                # 2CQ: Wait for previous CQ1 async work to complete
                if self.use_2cq and self.cq1_event is not None:
                    ttnn.wait_for_event(0, self.cq1_event)
                    self.cq1_event = None

                # Prepare input for this step
                if step == 0 and prefill_output is not None:
                    # Use last token from prefill
                    current_input = input_ids[:, -1:]
                else:
                    # Use previously generated token
                    current_input = generated_tokens[-1] if generated_tokens else input_ids[:, -1:]
                    if isinstance(current_input, torch.Tensor) and current_input.dim() == 1:
                        current_input = current_input.unsqueeze(0)

                # Copy input to device
                input_tt = ttnn.from_torch(
                    current_input,
                    device=None,
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
                ttnn.copy_host_to_device_tensor(input_tt, self.decode_inputs["input_ids"])

                # Execute decode trace (non-blocking for 2CQ overlap)
                blocking = not self.use_2cq
                ttnn.execute_trace(self.device, self.decode_trace_id, cq_id=0, blocking=blocking)

                # Record event on CQ0 for synchronization
                if self.use_2cq:
                    self.cq0_event = ttnn.record_event(self.device, 0)

                # Get output logits and sample token
                # For 2CQ, this happens while CQ1 does async work
                # decode_output is (codec_logits, cp_logits_list, talker_kv, cp_kv)
                codec_logits, cp_logits_list, _, _ = self.decode_output

                # Sample from all 16 code groups
                tokens_per_group = []

                # Code 0 from codec_head
                codec_logits_torch = ttnn.to_torch(codec_logits)
                token_0 = torch.argmax(codec_logits_torch, dim=-1)
                tokens_per_group.append(token_0)

                # Codes 1-15 from CodePredictor
                for logits in cp_logits_list:
                    logits_torch = ttnn.to_torch(logits)
                    token = torch.argmax(logits_torch, dim=-1)
                    tokens_per_group.append(token)

                generated_token = torch.stack(tokens_per_group, dim=-1)  # [batch, 16]

                generated_tokens.append(generated_token)

                # Capture TTFT after first token is generated
                if step == 0 and ttft is None:
                    ttft = time.time() - generation_start

                # 2CQ: Start async work on CQ1
                if self.use_2cq and step < max_new_tokens - 1:
                    # CQ1 waits for CQ0
                    ttnn.wait_for_event(1, self.cq0_event)

                    # Async position update for next iteration
                    self._update_position_async(current_pos + 1, cq_id=1)

                    # Record event on CQ1
                    self.cq1_event = ttnn.record_event(self.device, 1)

                # Send tokens to streaming decoder
                if self.streaming_decoder is not None:
                    self.streaming_decoder.add_tokens(generated_token.squeeze())

                    # Check for audio chunk and call callback
                    if audio_callback is not None:
                        audio_chunk = self.streaming_decoder.get_audio_chunk(timeout=0.001)
                        if audio_chunk is not None:
                            if first_audio_chunk_time is None:
                                first_audio_chunk_time = time.time() - generation_start
                            audio_callback(audio_chunk)

                current_pos += 1

                # Track decode time
                step_time = time.time() - step_start
                decode_times.append(step_time)

                # Progress logging
                if (step + 1) % 20 == 0:
                    avg_time = sum(decode_times[-20:]) / min(20, len(decode_times))
                    logger.info(f"Step {step + 1}/{max_new_tokens}, avg decode: {avg_time*1000:.1f}ms")

        finally:
            # Stop streaming decoder
            if self.streaming_decoder is not None:
                self.streaming_decoder.stop()

        # Calculate final stats
        decode_loop_time = time.time() - decode_loop_start
        total_time = time.time() - generation_start
        num_tokens = len(generated_tokens)

        avg_decode_time = sum(decode_times) / len(decode_times) if decode_times else 0
        tokens_per_sec = num_tokens / decode_loop_time if decode_loop_time > 0 else 0

        stats = {
            "tokens_generated": num_tokens,
            "ttft": ttft,
            "ttft_ms": ttft * 1000 if ttft else 0,
            "avg_decode_time": avg_decode_time,
            "avg_decode_time_ms": avg_decode_time * 1000,
            "tokens_per_sec": tokens_per_sec,
            "decode_loop_time": decode_loop_time,
            "total_time": total_time,
            "first_audio_chunk_time": first_audio_chunk_time,
            "first_audio_chunk_time_ms": first_audio_chunk_time * 1000 if first_audio_chunk_time else None,
            "use_2cq": self.use_2cq,
        }

        logger.info(f"Generation complete: {num_tokens} tokens in {decode_loop_time:.2f}s")
        logger.info(f"  TTFT: {stats['ttft_ms']:.1f}ms")
        logger.info(f"  Throughput: {tokens_per_sec:.2f} tok/s")
        logger.info(f"  Avg decode: {stats['avg_decode_time_ms']:.1f}ms/token")

        return generated_tokens, stats

    def get_final_audio(self) -> torch.Tensor:
        """
        Get final decoded audio after generation.

        Returns:
            Full audio tensor
        """
        if self.streaming_decoder is not None:
            return self.streaming_decoder.get_all_audio()
        return torch.tensor([])

    def release_traces(self):
        """Release all captured traces."""
        if self.prefill_trace_id is not None:
            ttnn.release_trace(self.device, self.prefill_trace_id)
            self.prefill_trace_id = None
            self.prefill_trace_captured = False

        if self.decode_trace_id is not None:
            ttnn.release_trace(self.device, self.decode_trace_id)
            self.decode_trace_id = None
            self.decode_trace_captured = False

        logger.info("Traces released")


def create_generator_2cq(
    model: Qwen3TTS,
    device,
    max_batch_size: int = 1,
    max_seq_len: int = 2048,
    use_2cq: bool = True,
) -> Qwen3TTSGenerator2CQ:
    """
    Factory function to create a 2CQ generator.

    Args:
        model: Qwen3TTS model
        device: TTNN device
        max_batch_size: Maximum batch size
        max_seq_len: Maximum sequence length
        use_2cq: Enable 2-command-queue mode

    Returns:
        Initialized 2CQ generator
    """
    return Qwen3TTSGenerator2CQ(
        model=model,
        device=device,
        talker_config=model.talker_config,
        code_predictor_config=model.code_predictor_config,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        use_2cq=use_2cq,
    )
