# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
SpeechT5 TTS Generator with Trace Support.

Manages trace capture and execution for SpeechT5 TTS models.
Follows the Whisper generator pattern for fully persistent traces.

Key design (following Whisper pattern):
1. Pre-allocate tensors with stable memory addresses at initialization
2. Call preprocess_decoder_inputs OUTSIDE trace capture (handles PE + dropout)
3. Capture trace on decoder layers only (no position-dependent operations)
4. Execute trace by copying preprocessed hidden states to pre-allocated L1 tensor
5. Cross-attention cache is populated on first iteration, reused in trace

Multi-size trace support:
- Pre-allocate tensors for ALL supported encoder sizes (128, 256, 384, 512, 768)
- Capture traces for all sizes during warm-up
- Select appropriate trace based on padded encoder size during inference
"""

import torch
import ttnn
from loguru import logger

# Supported encoder sequence lengths (padded sizes)
# Similar to Whisper's chunked input approach
SUPPORTED_ENCODER_SEQ_LENS = [128, 256, 384, 512, 768]


def get_padded_encoder_seq_len(seq_len: int) -> int:
    """Get the smallest supported encoder sequence length >= seq_len."""
    for supported_len in SUPPORTED_ENCODER_SEQ_LENS:
        if seq_len <= supported_len:
            return supported_len
    # If larger than max, return the largest supported
    logger.warning(f"Encoder seq_len {seq_len} > max supported {SUPPORTED_ENCODER_SEQ_LENS[-1]}")
    return SUPPORTED_ENCODER_SEQ_LENS[-1]


class SpeechT5Generator:
    """
    Generator wrapper for SpeechT5 TTS models with fully persistent trace support.

    This class maintains trace artifacts as instance variables, enabling trace reuse
    across multiple generations. Traces are pre-captured for ALL supported encoder
    sizes (128, 256, 384, 512, 768) during warm-up, so any input length will have
    a matching trace ready.

    Key insight (following Whisper pattern):
    1. preprocess_decoder_inputs runs OUTSIDE trace - handles PE slicing, dropout
    2. encoder_hidden_states is pre-allocated with stable memory addresses (per size)
    3. cross_attn_cache is pre-allocated with stable memory addresses (per size)
    4. First decoder iteration (non-traced) copies new K/V into pre-allocated cache
    5. Subsequent iterations (traced) use the pre-allocated cache at same addresses
    6. The trace captures only decoder layers - NO position-dependent operations
    """

    def __init__(
        self,
        encoder,
        decoder,
        postnet,
        device,
        decoder_config,
        max_steps: int = 100,
        max_batch_size: int = 1,
        encoder_seq_len: int = 128,
    ):
        """
        Initialize SpeechT5Generator with pre-allocated tensors for trace support.

        Pre-allocates tensors for ALL supported encoder sizes to enable trace reuse
        across different input lengths.

        Args:
            encoder: TTNNSpeechT5Encoder instance
            decoder: TTNNSpeechT5Decoder instance
            postnet: TTNNSpeechT5SpeechDecoderPostnet instance
            device: TTNN device
            decoder_config: TTNNDecoderConfig for decoder parameters
            max_steps: Maximum number of generation steps
            max_batch_size: Maximum batch size
            encoder_seq_len: Initial encoder sequence length (for backward compatibility)
        """
        self.encoder = encoder
        self.decoder = decoder
        self.postnet = postnet
        self.device = device
        self.decoder_config = decoder_config
        self.max_steps = max_steps
        self.max_batch_size = max_batch_size

        # Model dimensions
        self.hidden_size = decoder_config.hidden_size
        self.num_mel_bins = decoder_config.num_mel_bins

        # Current active encoder sequence length (set by copy_encoder_output)
        self.current_encoder_seq_len = None

        # Pre-allocated tensors PER encoder size
        # These are indexed by encoder_seq_len
        self.encoder_hidden_states_per_size = {}
        self.encoder_attention_mask_per_size = {}
        self.cross_attn_cache_per_size = {}

        # Trace state PER encoder size
        self.trace_id_per_size = {}
        self.trace_input_per_size = {}
        self.trace_output_per_size = {}
        self.trace_compiled_per_size = {}

        # Shared self-attention KV cache (independent of encoder size)
        from models.experimental.speecht5_tts.tt.ttnn_speecht5_decoder import init_kv_cache

        # Initialize self-attention KV cache only (cross-attention cache is per-size)
        self.kv_cache, _ = init_kv_cache(
            decoder_config,
            device,
            max_batch_size=max_batch_size,
            max_seq_len=max_steps + 10,
            encoder_seq_len=128,  # Dummy, we only need kv_cache
        )

        # Pre-allocated current_decode_pos tensor (stable memory address, shared across sizes)
        self.current_decode_pos = ttnn.allocate_tensor_on_device(
            ttnn.Shape([max_batch_size]),
            ttnn.int32,
            ttnn.ROW_MAJOR_LAYOUT,
            device,
            ttnn.DRAM_MEMORY_CONFIG,
        )

        # Pre-allocated decoder input for trace (shape [batch, 1, hidden_size])
        # This holds the preprocessed hidden states (after prenet + PE) for traced execution
        # Shared across all encoder sizes (decoder input shape is independent of encoder size)
        self.decoder_input_preallocated = ttnn.allocate_tensor_on_device(
            ttnn.Shape([max_batch_size, 1, self.hidden_size]),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            device,
            ttnn.L1_MEMORY_CONFIG,
        )

        # Pre-allocate tensors for ALL supported encoder sizes
        logger.info(f"Pre-allocating tensors for encoder sizes: {SUPPORTED_ENCODER_SEQ_LENS}")
        for enc_seq_len in SUPPORTED_ENCODER_SEQ_LENS:
            self._allocate_for_encoder_size(enc_seq_len)

        # Set initial encoder_seq_len (for backward compatibility)
        self.encoder_seq_len = get_padded_encoder_seq_len(encoder_seq_len)

        # Cross-attention cache validity flag (per-size, managed via current_encoder_seq_len)
        self.cross_attn_cache_valid = False

        # Legacy references for backward compatibility
        # These point to the current active size's tensors
        self.encoder_hidden_states = self.encoder_hidden_states_per_size[self.encoder_seq_len]
        self.encoder_attention_mask = self.encoder_attention_mask_per_size[self.encoder_seq_len]
        self.cross_attn_cache = self.cross_attn_cache_per_size[self.encoder_seq_len]
        self.trace_id_decoder = self.trace_id_per_size.get(self.encoder_seq_len)
        self.trace_input_decoder = self.trace_input_per_size.get(self.encoder_seq_len)
        self.trace_output_decoder = self.trace_output_per_size.get(self.encoder_seq_len)
        self.trace_compiled = self.trace_compiled_per_size.get(self.encoder_seq_len, False)

        logger.info(
            f"SpeechT5Generator initialized with max_steps={max_steps}, "
            f"max_batch_size={max_batch_size}, supported_encoder_sizes={SUPPORTED_ENCODER_SEQ_LENS}"
        )

    def _allocate_for_encoder_size(self, enc_seq_len: int):
        """
        Allocate tensors for a specific encoder sequence length.

        Args:
            enc_seq_len: Encoder sequence length (should be one of SUPPORTED_ENCODER_SEQ_LENS)
        """
        from models.experimental.speecht5_tts.tt.ttnn_speecht5_decoder import init_kv_cache

        # Pre-allocate encoder_hidden_states tensor
        # Shape: [batch, 1, encoder_seq_len, hidden_size] for 4D tensor compatibility
        self.encoder_hidden_states_per_size[enc_seq_len] = ttnn.allocate_tensor_on_device(
            ttnn.Shape([self.max_batch_size, 1, enc_seq_len, self.hidden_size]),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            self.device,
            ttnn.DRAM_MEMORY_CONFIG,
        )

        # Pre-allocate encoder attention mask for cross-attention
        # Shape: [batch, 1, 1, encoder_seq_len] - broadcastable to attention weights
        self.encoder_attention_mask_per_size[enc_seq_len] = ttnn.allocate_tensor_on_device(
            ttnn.Shape([self.max_batch_size, 1, 1, enc_seq_len]),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            self.device,
            ttnn.DRAM_MEMORY_CONFIG,
        )

        # Initialize mask to all zeros (no masking by default)
        zeros_mask = ttnn.zeros(
            [self.max_batch_size, 1, 1, enc_seq_len],
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.copy(zeros_mask, self.encoder_attention_mask_per_size[enc_seq_len])
        ttnn.deallocate(zeros_mask)

        # Pre-allocate cross-attention cache for this encoder size
        _, cross_attn_cache = init_kv_cache(
            self.decoder_config,
            self.device,
            max_batch_size=self.max_batch_size,
            max_seq_len=self.max_steps + 10,
            encoder_seq_len=enc_seq_len,
        )
        self.cross_attn_cache_per_size[enc_seq_len] = cross_attn_cache

        # Initialize trace state for this size
        self.trace_id_per_size[enc_seq_len] = None
        self.trace_input_per_size[enc_seq_len] = None
        self.trace_output_per_size[enc_seq_len] = None
        self.trace_compiled_per_size[enc_seq_len] = False

        logger.debug(f"Allocated tensors for encoder_seq_len={enc_seq_len}")

    def _switch_to_encoder_size(self, enc_seq_len: int):
        """
        Switch the active encoder size and update legacy references.

        Args:
            enc_seq_len: Encoder sequence length to switch to
        """
        if enc_seq_len not in self.encoder_hidden_states_per_size:
            raise ValueError(
                f"Encoder size {enc_seq_len} not pre-allocated. Supported sizes: {SUPPORTED_ENCODER_SEQ_LENS}"
            )

        self.current_encoder_seq_len = enc_seq_len
        self.encoder_seq_len = enc_seq_len

        # Update legacy references
        self.encoder_hidden_states = self.encoder_hidden_states_per_size[enc_seq_len]
        self.encoder_attention_mask = self.encoder_attention_mask_per_size[enc_seq_len]
        self.cross_attn_cache = self.cross_attn_cache_per_size[enc_seq_len]
        self.trace_id_decoder = self.trace_id_per_size.get(enc_seq_len)
        self.trace_input_decoder = self.trace_input_per_size.get(enc_seq_len)
        self.trace_output_decoder = self.trace_output_per_size.get(enc_seq_len)
        self.trace_compiled = self.trace_compiled_per_size.get(enc_seq_len, False)

    def _invalidate_cross_attn_cache(self):
        """Invalidate cross-attention cache for new generation."""
        self.cross_attn_cache_valid = False

    def _reset_kv_caches(self):
        """
        Reset KV caches to zeros for a fresh generation.

        This is CRITICAL when reusing generator between warm-up and inference,
        as stale KV cache values from warm-up will corrupt inference output.
        """
        # Reset self-attention KV cache (shared across all encoder sizes)
        for layer_idx, (k_cache, v_cache) in enumerate(self.kv_cache):
            zeros_k = ttnn.zeros_like(k_cache, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            zeros_v = ttnn.zeros_like(v_cache, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.copy(zeros_k, k_cache)
            ttnn.copy(zeros_v, v_cache)
            ttnn.deallocate(zeros_k)
            ttnn.deallocate(zeros_v)

        # Reset cross-attention KV caches for ALL sizes
        for enc_seq_len, cross_attn_cache in self.cross_attn_cache_per_size.items():
            for layer_idx, (k_cache, v_cache) in enumerate(cross_attn_cache):
                zeros_k = ttnn.zeros_like(k_cache, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                zeros_v = ttnn.zeros_like(v_cache, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                ttnn.copy(zeros_k, k_cache)
                ttnn.copy(zeros_v, v_cache)
                ttnn.deallocate(zeros_k)
                ttnn.deallocate(zeros_v)

        # Invalidate cross-attention cache validity flag
        self.cross_attn_cache_valid = False
        logger.debug("Reset KV caches to zeros for all sizes")

    def _reset_decode_pos(self, value: int, batch_size: int):
        """
        Reset current_decode_pos to a specific value in-place.

        Args:
            value: The position value to set (integer)
            batch_size: Number of batch items
        """
        pos_host = torch.full((batch_size,), value, dtype=torch.int32)
        pos_tensor_host = ttnn.from_torch(pos_host, dtype=ttnn.int32)
        ttnn.copy_host_to_device_tensor(pos_tensor_host, self.current_decode_pos)

    def _release_decoder_trace(self, enc_seq_len: int = None):
        """
        Release the decoder trace resources for a specific size (or current size).

        Args:
            enc_seq_len: Encoder sequence length. If None, uses current_encoder_seq_len.
        """
        if enc_seq_len is None:
            enc_seq_len = self.current_encoder_seq_len or self.encoder_seq_len

        trace_id = self.trace_id_per_size.get(enc_seq_len)
        if trace_id is not None:
            ttnn.release_trace(self.device, trace_id)
            self.trace_id_per_size[enc_seq_len] = None
            self.trace_input_per_size[enc_seq_len] = None
            self.trace_output_per_size[enc_seq_len] = None
            self.trace_compiled_per_size[enc_seq_len] = False

            # Update legacy references if this is the current size
            if enc_seq_len == self.current_encoder_seq_len:
                self.trace_id_decoder = None
                self.trace_input_decoder = None
                self.trace_output_decoder = None
                self.trace_compiled = False

            logger.debug(f"Released decoder trace for encoder_seq_len={enc_seq_len}")

    def _capture_decoder_trace(self, sample_preprocessed_hidden_states, enc_seq_len: int = None):
        """
        Capture decoder trace for a specific encoder size.

        The trace is captured after the first decoder iteration when cross_attn_cache
        is already populated. The trace uses:
        - Pre-allocated encoder_hidden_states (stable memory address)
        - Pre-allocated cross_attn_cache (stable memory address)
        - Preprocessed hidden states (prenet + PE already applied OUTSIDE trace)

        Args:
            sample_preprocessed_hidden_states: Sample preprocessed hidden states for trace capture
            enc_seq_len: Encoder sequence length. If None, uses current_encoder_seq_len.
        """
        if enc_seq_len is None:
            enc_seq_len = self.current_encoder_seq_len or self.encoder_seq_len

        if self.trace_id_per_size.get(enc_seq_len) is not None:
            return  # Already captured for this size

        logger.info(f"Capturing decoder trace for encoder_seq_len={enc_seq_len}")

        # Ensure we're using the correct tensors for this size
        encoder_hidden_states = self.encoder_hidden_states_per_size[enc_seq_len]
        encoder_attention_mask = self.encoder_attention_mask_per_size[enc_seq_len]
        cross_attn_cache = self.cross_attn_cache_per_size[enc_seq_len]

        # Allocate L1 input for trace capture
        trace_input = ttnn.to_memory_config(sample_preprocessed_hidden_states, ttnn.L1_MEMORY_CONFIG)

        # Capture trace - decoder receives preprocessed hidden states (prenet + PE already done)
        trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
        trace_output = self.decoder(
            decoder_input_values=None,
            encoder_hidden_states=encoder_hidden_states,
            speaker_embeddings=None,
            kv_cache=self.kv_cache,
            cross_attn_cache=cross_attn_cache,
            cross_attn_cache_valid=True,
            current_decode_pos=self.current_decode_pos,
            preprocessed_hidden_states=trace_input,
            encoder_attention_mask=encoder_attention_mask,
        )
        ttnn.end_trace_capture(self.device, trace_id, cq_id=0)
        ttnn.synchronize_device(self.device)

        # Store trace artifacts for this size
        self.trace_id_per_size[enc_seq_len] = trace_id
        self.trace_input_per_size[enc_seq_len] = trace_input
        self.trace_output_per_size[enc_seq_len] = trace_output
        self.trace_compiled_per_size[enc_seq_len] = True

        # Update legacy references if this is the current size
        if enc_seq_len == self.current_encoder_seq_len:
            self.trace_id_decoder = trace_id
            self.trace_input_decoder = trace_input
            self.trace_output_decoder = trace_output
            self.trace_compiled = True

        logger.info(f"Decoder trace captured for encoder_seq_len={enc_seq_len}")

    def _execute_decoder_trace(self, preprocessed_hidden_states, enc_seq_len: int = None):
        """
        Execute the captured decoder trace with new preprocessed hidden states.

        Args:
            preprocessed_hidden_states: New preprocessed hidden states (prenet + PE already applied)
            enc_seq_len: Encoder sequence length. If None, uses current_encoder_seq_len.

        Returns:
            Decoder output tensor
        """
        if enc_seq_len is None:
            enc_seq_len = self.current_encoder_seq_len or self.encoder_seq_len

        trace_id = self.trace_id_per_size.get(enc_seq_len)
        if trace_id is None:
            raise RuntimeError(
                f"Decoder trace not captured for encoder_seq_len={enc_seq_len}. Call _capture_decoder_trace first."
            )

        trace_input = self.trace_input_per_size[enc_seq_len]
        trace_output = self.trace_output_per_size[enc_seq_len]

        # Copy new preprocessed input to persistent L1 tensor (KEY: output_tensor parameter)
        self.trace_input_per_size[enc_seq_len] = ttnn.to_memory_config(
            preprocessed_hidden_states,
            ttnn.L1_MEMORY_CONFIG,
            output_tensor=trace_input,  # Overwrite, don't create new
        )

        # Execute trace
        ttnn.execute_trace(self.device, trace_id, cq_id=0, blocking=True)

        return trace_output

    def capture_all_traces(self, processor, batch_size: int = 1):
        """
        Capture traces for ALL supported encoder sizes during warm-up.

        This method creates dummy inputs for each supported encoder size,
        runs a compile pass, and captures a trace. After calling this method,
        any input length will have a matching trace ready.

        Args:
            processor: SpeechT5Processor for tokenizing dummy texts
            batch_size: Batch size to use for trace capture
        """
        logger.info(f"Capturing traces for all supported encoder sizes: {SUPPORTED_ENCODER_SEQ_LENS}")

        # Create dummy texts that will result in different padded sizes
        # We need to create encoder outputs of specific lengths, then pad
        dummy_texts = {
            128: "Hello",  # Short text -> padded to 128
            256: "A" * 200,  # Medium text -> padded to 256
            384: "B" * 300,  # Longer text -> padded to 384
            512: "C" * 450,  # Even longer -> padded to 512
            768: "D" * 700,  # Very long -> padded to 768
        }

        for target_size in SUPPORTED_ENCODER_SEQ_LENS:
            if self.trace_compiled_per_size.get(target_size, False):
                logger.info(f"  Trace for encoder_seq_len={target_size} already compiled, skipping")
                continue

            logger.info(f"  Capturing trace for encoder_seq_len={target_size}...")

            # Switch to this size
            self._switch_to_encoder_size(target_size)

            # Create dummy encoder output directly
            # Shape: [batch, target_size, hidden_size]
            dummy_encoder_output = torch.randn(batch_size, target_size, self.hidden_size)
            dummy_encoder_output_ttnn = ttnn.from_torch(
                dummy_encoder_output.unsqueeze(1),  # [B, 1, S, H]
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            # Copy to pre-allocated tensor (sets up encoder_hidden_states)
            ttnn.copy(dummy_encoder_output_ttnn, self.encoder_hidden_states_per_size[target_size])
            ttnn.deallocate(dummy_encoder_output_ttnn)

            # Set up mask (all zeros since this is a dummy with no padding)
            zeros_mask = ttnn.zeros(
                [batch_size, 1, 1, target_size],
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.copy(zeros_mask, self.encoder_attention_mask_per_size[target_size])
            ttnn.deallocate(zeros_mask)

            # Create dummy decoder input (mel frame)
            dummy_mel = torch.zeros(batch_size, 1, self.num_mel_bins)
            dummy_mel_ttnn = ttnn.from_torch(
                dummy_mel,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )

            # Reset decode position
            self._reset_decode_pos(0, batch_size)

            # Run step 0 (non-traced) to populate cross-attention cache and compile
            preprocessed = self.decoder.preprocess_decoder_inputs(
                decoder_input_values=dummy_mel_ttnn,
                position_offset=0,
            )

            # Run decoder to populate cross-attention cache
            _ = self.decoder(
                decoder_input_values=None,
                encoder_hidden_states=self.encoder_hidden_states_per_size[target_size],
                speaker_embeddings=None,
                kv_cache=self.kv_cache,
                cross_attn_cache=self.cross_attn_cache_per_size[target_size],
                cross_attn_cache_valid=False,
                current_decode_pos=self.current_decode_pos,
                preprocessed_hidden_states=preprocessed,
                encoder_attention_mask=self.encoder_attention_mask_per_size[target_size],
            )

            # Reset decode position for step 1 (for trace capture)
            self._reset_decode_pos(1, batch_size)

            # Create new preprocessed input for step 1
            preprocessed_step1 = self.decoder.preprocess_decoder_inputs(
                decoder_input_values=dummy_mel_ttnn,
                position_offset=1,
            )

            # Capture trace for this size
            self._capture_decoder_trace(preprocessed_step1, target_size)

            # Cleanup dummy tensors
            ttnn.deallocate(dummy_mel_ttnn)

            # Reset KV caches for this size to clear dummy values
            for layer_idx, (k_cache, v_cache) in enumerate(self.kv_cache):
                zeros_k = ttnn.zeros_like(k_cache, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                zeros_v = ttnn.zeros_like(v_cache, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                ttnn.copy(zeros_k, k_cache)
                ttnn.copy(zeros_v, v_cache)
                ttnn.deallocate(zeros_k)
                ttnn.deallocate(zeros_v)

            for layer_idx, (k_cache, v_cache) in enumerate(self.cross_attn_cache_per_size[target_size]):
                zeros_k = ttnn.zeros_like(k_cache, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                zeros_v = ttnn.zeros_like(v_cache, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                ttnn.copy(zeros_k, k_cache)
                ttnn.copy(zeros_v, v_cache)
                ttnn.deallocate(zeros_k)
                ttnn.deallocate(zeros_v)

        logger.info("All traces captured successfully!")

        # Report status
        for size in SUPPORTED_ENCODER_SEQ_LENS:
            status = "compiled" if self.trace_compiled_per_size.get(size, False) else "NOT compiled"
            logger.info(f"  encoder_seq_len={size}: {status}")

    def copy_encoder_output(self, encoder_output):
        """
        Store encoder output for trace stability with padding.

        Pads encoder output to the nearest supported size (128, 256, 384, 512, 768)
        and switches to the appropriate pre-allocated tensors. Uses attention mask
        to ignore padded positions during cross-attention.

        Args:
            encoder_output: Encoder output tensor to copy
        """
        # Get encoder output shape
        if len(encoder_output.shape) == 3:
            batch, seq_len, hidden = encoder_output.shape
        else:
            batch, _, seq_len, hidden = encoder_output.shape

        # Store actual sequence length for reference
        self.actual_encoder_seq_len = seq_len

        # Get padded size (ceiling to nearest supported size)
        padded_seq_len = get_padded_encoder_seq_len(seq_len)

        # Switch to the appropriate pre-allocated tensors for this size
        self._switch_to_encoder_size(padded_seq_len)

        # Convert to torch for padding, then back to TTNN
        encoder_torch = ttnn.to_torch(encoder_output)
        if len(encoder_torch.shape) == 4:
            encoder_torch = encoder_torch.squeeze(1)  # [B, 1, S, H] -> [B, S, H]

        # Pad encoder output to fixed size [B, padded_seq_len, H]
        if seq_len < padded_seq_len:
            padding = torch.zeros(batch, padded_seq_len - seq_len, hidden, dtype=encoder_torch.dtype)
            encoder_padded = torch.cat([encoder_torch, padding], dim=1)
        else:
            encoder_padded = encoder_torch[:, :padded_seq_len, :]  # Truncate if needed

        # Reshape to 4D [B, 1, padded_seq_len, H]
        encoder_padded = encoder_padded.unsqueeze(1)

        # Copy to pre-allocated tensor (stable memory address for trace)
        encoder_padded_ttnn = ttnn.from_torch(
            encoder_padded,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.copy(encoder_padded_ttnn, self.encoder_hidden_states)
        ttnn.deallocate(encoder_padded_ttnn)

        # Create attention mask: 0 for actual positions, -1e9 for padded positions
        # Shape: [B, 1, 1, padded_seq_len]
        mask = torch.zeros(batch, 1, 1, padded_seq_len, dtype=torch.float32)
        if seq_len < padded_seq_len:
            mask[:, :, :, seq_len:] = -1e9  # Mask out padded positions

        mask_ttnn = ttnn.from_torch(
            mask,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.copy(mask_ttnn, self.encoder_attention_mask)
        ttnn.deallocate(mask_ttnn)

        logger.debug(f"Stored encoder output: actual_len={seq_len}, padded_len={padded_seq_len}")

    def cleanup(self):
        """Release trace resources for all sizes. Call before closing device."""
        for enc_seq_len in SUPPORTED_ENCODER_SEQ_LENS:
            trace_id = self.trace_id_per_size.get(enc_seq_len)
            if trace_id is not None:
                try:
                    ttnn.release_trace(self.device, trace_id)
                    self.trace_id_per_size[enc_seq_len] = None
                    self.trace_input_per_size[enc_seq_len] = None
                    self.trace_output_per_size[enc_seq_len] = None
                    self.trace_compiled_per_size[enc_seq_len] = False
                    logger.debug(f"Released decoder trace for encoder_seq_len={enc_seq_len}")
                except Exception as e:
                    logger.warning(f"Error releasing trace for encoder_seq_len={enc_seq_len}: {e}")

        # Reset legacy references
        self.trace_id_decoder = None
        self.trace_input_decoder = None
        self.trace_output_decoder = None
        self.trace_compiled = False

        logger.info("Released all decoder trace resources")

    def __del__(self):
        """Cleanup on destruction - do NOT release trace here as device may be closed."""
        # Don't call cleanup() in destructor - the device might already be closed
        # The trace resources will be cleaned up when the device is closed
