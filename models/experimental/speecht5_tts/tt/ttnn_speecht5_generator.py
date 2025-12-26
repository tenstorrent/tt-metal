# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
SpeechT5 TTS Generator with Trace Support.

Manages trace capture and execution for SpeechT5 TTS models.
Supports encoder, decoder, and postnet traces for optimal performance.
"""

import torch
import ttnn
from collections import defaultdict
from loguru import logger

from models.tt_transformers.tt.common import copy_host_to_device


class SpeechT5Generator:
    """
    Generator wrapper for SpeechT5 TTS models with trace support.

    Manages trace capture and execution for encoder, decoder, and postnet components.
    Follows the pattern established in Galaxy LLM implementation.
    """

    def __init__(self, encoder, decoder, postnet, device, max_steps: int = 100):
        """
        Initialize SpeechT5Generator with trace support.

        Args:
            encoder: TTNNSpeechT5Encoder instance
            decoder: TTNNSpeechT5Decoder instance
            postnet: TTNNSpeechT5SpeechDecoderPostnet instance
            device: TTNN device
            max_steps: Maximum number of generation steps (for decoder trace prewarming)
        """
        self.encoder = encoder
        self.decoder = decoder
        self.postnet = postnet
        self.device = device
        self.max_steps = max_steps

        # Decoder trace storage (keyed by sequence length)
        self.trace_id_decoder = defaultdict(lambda: None)
        self.trace_inputs_decoder = defaultdict(lambda: None)
        self.trace_output_decoder = defaultdict(lambda: None)

        # Postnet trace storage (fixed input shape)
        self.trace_id_postnet = None
        self.trace_inputs_postnet = None
        self.trace_output_postnet = None

        # Encoder trace storage (keyed by sequence length)
        self.trace_id_encoder = defaultdict(lambda: None)
        self.trace_inputs_encoder = defaultdict(lambda: None)
        self.trace_output_encoder = defaultdict(lambda: None)

        # Trace warmup flags
        self.decode_traces_warmup = False
        self.encoder_traces_warmup = False

        # Common encoder lengths to pre-warm traces for
        self.common_encoder_lengths = [10, 20, 50, 100, 200]

    def warmup_decode_traces(self):
        """
        Pre-warm decoder and postnet traces for all sequence lengths from 1 to max_steps.

        This ensures traces are captured during initialization, avoiding compilation
        overhead during inference.
        """
        if self.decode_traces_warmup:
            return

        self.decode_traces_warmup = True
        logger.info(f"Warming up decoder and postnet traces for sequence lengths 1 to {self.max_steps}")

        # Create dummy inputs for trace capture
        batch_size = 1
        num_mel_bins = 80
        hidden_size = 768
        encoder_seq_len = 50  # Dummy encoder sequence length

        # Dummy encoder output for decoder input
        encoder_hidden_states = ttnn.from_torch(
            torch.randn(batch_size, encoder_seq_len, hidden_size),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # Dummy speaker embeddings
        speaker_embeddings = ttnn.from_torch(
            torch.randn(batch_size, 512),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        try:
            # Capture postnet trace first (fixed input shape)
            logger.info("Capturing postnet trace...")
            self._capture_postnet_trace()

            # Capture decoder traces for each sequence length
            for seq_len in range(1, self.max_steps + 1):
                if seq_len % 20 == 0 or seq_len == 1:
                    logger.info(f"Capturing decoder trace for sequence length {seq_len}...")

                self._capture_decoder_trace(seq_len)

                # Clean up dummy inputs for next iteration
                if seq_len < self.max_steps:
                    ttnn.deallocate(encoder_hidden_states)
                    ttnn.deallocate(speaker_embeddings)

                    encoder_hidden_states = ttnn.from_torch(
                        torch.randn(batch_size, encoder_seq_len, hidden_size),
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=self.device,
                        memory_config=ttnn.L1_MEMORY_CONFIG,
                    )
                    speaker_embeddings = ttnn.from_torch(
                        torch.randn(batch_size, 512),
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=self.device,
                        memory_config=ttnn.L1_MEMORY_CONFIG,
                    )

        except Exception as e:
            logger.error(f"Error during decode trace warmup: {e}")
            raise
        finally:
            # Clean up dummy tensors
            ttnn.deallocate(encoder_hidden_states)
            ttnn.deallocate(speaker_embeddings)

        logger.info("Decoder and postnet trace warmup completed")

    def warmup_encoder_traces(self, common_lengths=None):
        """
        Pre-warm encoder traces for common input sequence lengths.

        Args:
            common_lengths: List of sequence lengths to pre-warm traces for.
                           If None, uses self.common_encoder_lengths.
        """
        if self.encoder_traces_warmup:
            return

        if common_lengths is None:
            common_lengths = self.common_encoder_lengths

        self.encoder_traces_warmup = True
        logger.info(f"Warming up encoder traces for sequence lengths: {common_lengths}")

        # Create dummy input for trace capture
        batch_size = 1
        vocab_size = 81  # SpeechT5 vocab size

        try:
            for seq_len in common_lengths:
                logger.info(f"Capturing encoder trace for sequence length {seq_len}...")

                self._capture_encoder_trace(seq_len)

        except Exception as e:
            logger.error(f"Error during encoder trace warmup: {e}")
            raise

        logger.info("Encoder trace warmup completed")

    def _capture_encoder_trace(self, seq_len: int):
        """
        Capture encoder trace for a specific sequence length.

        Args:
            seq_len: Sequence length for encoder input
        """
        batch_size = 1

        # Create dummy input tokens (using a range for deterministic testing)
        input_ids = ttnn.from_torch(
            torch.arange(seq_len, dtype=torch.int32).unsqueeze(0),  # [1, seq_len]
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # First run to compile operations
        encoder_output = self.encoder(input_ids)[0]
        ttnn.synchronize_device(self.device)

        # Prepare inputs for trace capture
        host_inputs = self.encoder.prepare_encoder_inputs(input_ids)
        device_inputs = copy_host_to_device(host_inputs, mesh_device=self.device)

        # Capture trace
        trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
        transformed_inputs = self.encoder.prepare_encoder_inputs(input_ids)
        trace_output = self.encoder(input_ids)[0]
        ttnn.end_trace_capture(self.device, trace_id, cq_id=0)
        ttnn.synchronize_device(self.device)

        # Store trace information
        self.trace_id_encoder[seq_len] = trace_id
        self.trace_inputs_encoder[seq_len] = device_inputs
        self.trace_output_encoder[seq_len] = trace_output

        # Clean up
        ttnn.deallocate(input_ids)
        ttnn.deallocate(encoder_output)

    def _execute_encoder_trace(self, seq_len: int, input_ids):
        """
        Execute encoder trace for given sequence length.

        Args:
            seq_len: Sequence length key for trace lookup
            input_ids: Input token IDs

        Returns:
            Encoder output hidden states
        """
        # Find the nearest pre-warmed trace length
        trace_seq_len = self._get_encoder_trace_length(seq_len)

        if trace_seq_len not in self.trace_id_encoder or self.trace_id_encoder[trace_seq_len] is None:
            logger.warning(
                f"No encoder trace found for sequence length {trace_seq_len}, falling back to non-traced execution"
            )
            return self.encoder(input_ids)[0]

        # Prepare inputs and copy to device
        host_inputs = self.encoder.prepare_encoder_inputs(input_ids)
        copy_host_to_device(
            host_tensors=host_inputs,
            device_tensors=self.trace_inputs_encoder[trace_seq_len],
        )

        # Execute trace
        ttnn.execute_trace(self.device, self.trace_id_encoder[trace_seq_len], cq_id=0, blocking=False)

        return self.trace_output_encoder[trace_seq_len]

    def _get_encoder_trace_length(self, actual_len: int):
        """
        Get the nearest pre-warmed trace length for a given actual length.

        Args:
            actual_len: Actual input sequence length

        Returns:
            Nearest pre-warmed trace length
        """
        # Find the smallest pre-warmed length that is >= actual_len
        for trace_len in sorted(self.common_encoder_lengths):
            if trace_len >= actual_len:
                return trace_len

        # If no length is large enough, use the largest available
        return max(self.common_encoder_lengths)

    def _capture_decoder_trace(self, seq_len: int):
        """
        Capture decoder trace for a specific sequence length.

        Args:
            seq_len: Sequence length for decoder input
        """
        batch_size = 1
        num_mel_bins = 80
        hidden_size = 768
        encoder_seq_len = 50  # Fixed for dummy input

        # Create dummy inputs
        decoder_input_values = ttnn.from_torch(
            torch.randn(batch_size, seq_len, num_mel_bins),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        encoder_hidden_states = ttnn.from_torch(
            torch.randn(batch_size, encoder_seq_len, hidden_size),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        speaker_embeddings = ttnn.from_torch(
            torch.randn(batch_size, 512),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # First run to compile operations
        decoder_output = self.decoder(
            decoder_input_values=decoder_input_values,
            encoder_hidden_states=encoder_hidden_states,
            speaker_embeddings=speaker_embeddings,
        )
        ttnn.synchronize_device(self.device)

        # Prepare inputs for trace capture
        host_inputs = self.decoder.prepare_decode_inputs(
            decoder_input_values, encoder_hidden_states, speaker_embeddings
        )
        device_inputs = copy_host_to_device(host_inputs, mesh_device=self.device)

        # Capture trace
        trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
        transformed_inputs = self.decoder.prepare_decode_inputs(
            decoder_input_values, encoder_hidden_states, speaker_embeddings
        )
        trace_output = self.decoder(
            decoder_input_values=decoder_input_values,
            encoder_hidden_states=encoder_hidden_states,
            speaker_embeddings=speaker_embeddings,
        )
        ttnn.end_trace_capture(self.device, trace_id, cq_id=0)
        ttnn.synchronize_device(self.device)

        # Store trace information
        self.trace_id_decoder[seq_len] = trace_id
        self.trace_inputs_decoder[seq_len] = device_inputs
        self.trace_output_decoder[seq_len] = trace_output

        # Clean up
        ttnn.deallocate(decoder_input_values)
        ttnn.deallocate(encoder_hidden_states)
        ttnn.deallocate(speaker_embeddings)
        if seq_len > 1:  # Keep the first decoder output for postnet trace
            ttnn.deallocate(decoder_output)

    def _capture_postnet_trace(self):
        """
        Capture postnet trace. Postnet has fixed input shape [batch, 1, hidden_size].
        """
        batch_size = 1
        hidden_size = 768

        # Create dummy decoder output (postnet input)
        decoder_hidden_states = ttnn.from_torch(
            torch.randn(batch_size, 1, hidden_size),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # First run to compile operations
        mel_before, mel_after, stop_logits = self.postnet(decoder_hidden_states)
        ttnn.synchronize_device(self.device)

        # Prepare inputs for trace capture
        host_inputs = self.postnet.prepare_postnet_inputs(decoder_hidden_states)
        device_inputs = copy_host_to_device(host_inputs, mesh_device=self.device)

        # Capture trace
        trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
        transformed_inputs = self.postnet.prepare_postnet_inputs(decoder_hidden_states)
        trace_mel_before, trace_mel_after, trace_stop_logits = self.postnet(decoder_hidden_states)
        ttnn.end_trace_capture(self.device, trace_id, cq_id=0)
        ttnn.synchronize_device(self.device)

        # Store trace information
        self.trace_id_postnet = trace_id
        self.trace_inputs_postnet = device_inputs
        self.trace_output_postnet = (trace_mel_before, trace_mel_after, trace_stop_logits)

        # Clean up
        ttnn.deallocate(decoder_hidden_states)
        ttnn.deallocate(mel_before)
        ttnn.deallocate(mel_after)
        ttnn.deallocate(stop_logits)

    def _execute_decoder_trace(self, seq_len: int, decoder_input_values, encoder_hidden_states, speaker_embeddings):
        """
        Execute decoder trace for given sequence length.

        Args:
            seq_len: Sequence length key for trace lookup
            decoder_input_values: Input mel spectrogram values
            encoder_hidden_states: Encoder output hidden states
            speaker_embeddings: Speaker embeddings

        Returns:
            Decoder output hidden states
        """
        if seq_len not in self.trace_id_decoder or self.trace_id_decoder[seq_len] is None:
            logger.warning(
                f"No decoder trace found for sequence length {seq_len}, falling back to non-traced execution"
            )
            return self.decoder(
                decoder_input_values=decoder_input_values,
                encoder_hidden_states=encoder_hidden_states,
                speaker_embeddings=speaker_embeddings,
            )

        # Prepare inputs and copy to device
        host_inputs = self.decoder.prepare_decode_inputs(
            decoder_input_values, encoder_hidden_states, speaker_embeddings
        )
        copy_host_to_device(
            host_tensors=host_inputs,
            device_tensors=self.trace_inputs_decoder[seq_len],
        )

        # Execute trace
        ttnn.execute_trace(self.device, self.trace_id_decoder[seq_len], cq_id=0, blocking=False)

        return self.trace_output_decoder[seq_len]

    def _execute_postnet_trace(self, decoder_hidden_states):
        """
        Execute postnet trace.

        Args:
            decoder_hidden_states: Decoder output hidden states

        Returns:
            Tuple of (mel_before, mel_after, stop_logits)
        """
        if self.trace_id_postnet is None:
            logger.warning("No postnet trace found, falling back to non-traced execution")
            return self.postnet(decoder_hidden_states)

        # Prepare inputs and copy to device
        host_inputs = self.postnet.prepare_postnet_inputs(decoder_hidden_states)
        copy_host_to_device(
            host_tensors=host_inputs,
            device_tensors=self.trace_inputs_postnet,
        )

        # Execute trace
        ttnn.execute_trace(self.device, self.trace_id_postnet, cq_id=0, blocking=False)

        return self.trace_output_postnet
