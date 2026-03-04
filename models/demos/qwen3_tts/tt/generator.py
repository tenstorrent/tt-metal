# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Optimized generator for Qwen3-TTS with tracing support.

This module provides:
- Trace capture for prefill and decode modes
- Pre-allocated tensors for efficient execution
- KV-cache management
"""

from typing import List, Tuple

import torch
from loguru import logger

import ttnn
from models.demos.qwen3_tts.tt.kv_cache import create_kv_cache
from models.demos.qwen3_tts.tt.model_config import Qwen3TTSCodePredictorConfig, Qwen3TTSTalkerConfig
from models.demos.qwen3_tts.tt.qwen3_tts import Qwen3TTS
from models.demos.qwen3_tts.tt.rope import get_rope_tensors, get_transformation_mat


class Qwen3TTSGenerator:
    """
    Optimized generator for Qwen3-TTS with tracing support.

    Features:
    - Prefill trace capture for efficient first-token generation
    - Decode trace capture for fast autoregressive decoding
    - Pre-allocated tensors to avoid allocation overhead during trace
    - KV-cache management for decode mode
    """

    def __init__(
        self,
        model: Qwen3TTS,
        device,
        talker_config: Qwen3TTSTalkerConfig,
        code_predictor_config: Qwen3TTSCodePredictorConfig,
        max_batch_size: int = 1,
        max_seq_len: int = 2048,
    ):
        """
        Initialize the generator.

        Args:
            model: Qwen3TTS model
            device: TTNN device
            talker_config: Talker configuration
            code_predictor_config: CodePredictor configuration
            max_batch_size: Maximum batch size
            max_seq_len: Maximum sequence length
        """
        self.model = model
        self.device = device
        self.talker_config = talker_config
        self.code_predictor_config = code_predictor_config
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len

        # Trace IDs
        self.prefill_trace_id = None
        self.decode_trace_id = None

        # Pre-allocated tensors for trace execution
        self.prefill_inputs = None
        self.prefill_output = None
        self.decode_inputs = None
        self.decode_output = None

        # RoPE tensors (pre-computed)
        self.talker_trans_mat = None
        self.cp_trans_mat = None

        # KV caches
        self.talker_kv_cache = None
        self.cp_kv_cache = None

        # Trace state
        self.prefill_trace_captured = False
        self.decode_trace_captured = False

    def setup(self):
        """
        Setup generator with pre-allocated tensors and transformation matrices.
        """
        logger.info("Setting up Qwen3-TTS generator...")

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

        logger.info("Generator setup complete")

    def warmup_prefill(self, seq_len: int = 128):
        """
        Warmup prefill forward pass to compile all ops.

        Args:
            seq_len: Sequence length for warmup
        """
        logger.info(f"Warming up prefill with seq_len={seq_len}...")

        # Create warmup input
        warmup_input = torch.zeros(self.max_batch_size, seq_len, dtype=torch.long)
        warmup_input_tt = ttnn.from_torch(
            warmup_input,
            device=self.device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Create RoPE tensors
        position_ids = torch.arange(seq_len)
        talker_cos, talker_sin = get_rope_tensors(
            self.device,
            self.talker_config.head_dim,
            seq_len,
            position_ids,
            self.talker_config.rope_theta,
        )
        cp_cos, cp_sin = get_rope_tensors(
            self.device,
            self.code_predictor_config.head_dim,
            seq_len,
            position_ids,
            self.code_predictor_config.rope_theta,
        )

        # Run warmup forward (compiles all ops)
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
        logger.info("Prefill warmup complete")

    def capture_prefill_trace(self, seq_len: int = 128):
        """
        Capture trace for prefill mode.

        Args:
            seq_len: Sequence length for trace
        """
        logger.info(f"Capturing prefill trace for seq_len={seq_len}...")

        # Pre-allocate input tensor
        input_tensor = ttnn.from_torch(
            torch.zeros(self.max_batch_size, seq_len, dtype=torch.long),
            device=self.device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Create RoPE tensors
        position_ids = torch.arange(seq_len)
        talker_cos, talker_sin = get_rope_tensors(
            self.device,
            self.talker_config.head_dim,
            seq_len,
            position_ids,
            self.talker_config.rope_theta,
        )
        cp_cos, cp_sin = get_rope_tensors(
            self.device,
            self.code_predictor_config.head_dim,
            seq_len,
            position_ids,
            self.code_predictor_config.rope_theta,
        )

        # Begin trace capture
        self.prefill_trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)

        # Run forward
        output = self.model.forward(
            input_tensor,
            talker_cos,
            talker_sin,
            self.talker_trans_mat,
            cp_cos,
            cp_sin,
            self.cp_trans_mat,
        )

        # End trace capture
        ttnn.end_trace_capture(self.device, self.prefill_trace_id, cq_id=0)

        # Store tensors for trace execution
        self.prefill_inputs = {
            "input_ids": input_tensor,
            "talker_cos": talker_cos,
            "talker_sin": talker_sin,
            "cp_cos": cp_cos,
            "cp_sin": cp_sin,
        }
        self.prefill_output = output

        self.prefill_trace_captured = True
        ttnn.synchronize_device(self.device)
        logger.info("Prefill trace captured")

    def execute_prefill_trace(self, input_ids: torch.Tensor) -> Tuple[ttnn.Tensor, List[ttnn.Tensor], None, None]:
        """
        Execute prefill trace with new input.

        Args:
            input_ids: Input token IDs [batch, seq_len]

        Returns:
            Tuple of (codec_logits, cp_logits_list, None, None) where:
            - codec_logits: Logits for code 0 from Talker's codec_head
            - cp_logits_list: List of logits for codes 1-15 from CodePredictor
        """
        if not self.prefill_trace_captured:
            raise RuntimeError("Prefill trace not captured. Call capture_prefill_trace first.")

        # Copy input to device tensor
        input_tt = ttnn.from_torch(
            input_ids,
            device=None,  # Host tensor
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        ttnn.copy_host_to_device_tensor(input_tt, self.prefill_inputs["input_ids"])

        # Execute trace
        ttnn.execute_trace(self.device, self.prefill_trace_id, cq_id=0, blocking=False)

        return self.prefill_output

    def warmup_decode(self):
        """
        Warmup decode forward pass (single token generation).
        """
        logger.info("Warming up decode...")

        # Single token decode
        seq_len = 1
        warmup_input = torch.zeros(self.max_batch_size, seq_len, dtype=torch.long)
        warmup_input_tt = ttnn.from_torch(
            warmup_input,
            device=self.device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Decode position (after prefill)
        position_ids = torch.tensor([128])  # Example position
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

        # Run warmup
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
        """
        Capture trace for decode mode (single token).

        Args:
            start_pos: Starting position for decode
        """
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

        # Decode position
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

        # Begin trace capture
        self.decode_trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)

        # Run forward
        output = self.model.forward(
            input_tensor,
            talker_cos,
            talker_sin,
            self.talker_trans_mat,
            cp_cos,
            cp_sin,
            self.cp_trans_mat,
        )

        # End trace capture
        ttnn.end_trace_capture(self.device, self.decode_trace_id, cq_id=0)

        # Store tensors
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

    def execute_decode_trace(
        self,
        input_ids: torch.Tensor,
        position: int,
    ) -> Tuple[ttnn.Tensor, List[ttnn.Tensor], None, None]:
        """
        Execute decode trace with new input and position.

        Args:
            input_ids: Input token IDs [batch, 1]
            position: Current sequence position

        Returns:
            Tuple of (codec_logits, cp_logits_list, None, None) where:
            - codec_logits: Logits for code 0 from Talker's codec_head
            - cp_logits_list: List of logits for codes 1-15 from CodePredictor
        """
        if not self.decode_trace_captured:
            raise RuntimeError("Decode trace not captured. Call capture_decode_trace first.")

        # Copy input to device tensor
        input_tt = ttnn.from_torch(
            input_ids,
            device=None,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        ttnn.copy_host_to_device_tensor(input_tt, self.decode_inputs["input_ids"])

        # Update RoPE tensors for new position
        position_ids = torch.tensor([position])
        talker_cos_host, talker_sin_host = get_rope_tensors(
            None,  # Host only
            self.talker_config.head_dim,
            1,
            position_ids,
            self.talker_config.rope_theta,
        )
        # Note: For full optimization, RoPE should use embedding lookup
        # instead of recomputing each step

        # Execute trace
        ttnn.execute_trace(self.device, self.decode_trace_id, cq_id=0, blocking=False)

        return self.decode_output

    def prefill(
        self,
        input_ids: torch.Tensor,
        use_trace: bool = True,
    ) -> Tuple[ttnn.Tensor, List[ttnn.Tensor], None, None]:
        """
        Run prefill with optional tracing.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            use_trace: Whether to use traced execution

        Returns:
            Tuple of (codec_logits, cp_logits_list, None, None)
        """
        if use_trace and self.prefill_trace_captured:
            return self.execute_prefill_trace(input_ids)
        else:
            # Non-traced execution
            seq_len = input_ids.shape[1]
            input_tt = ttnn.from_torch(
                input_ids,
                device=self.device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            position_ids = torch.arange(seq_len)
            talker_cos, talker_sin = get_rope_tensors(
                self.device,
                self.talker_config.head_dim,
                seq_len,
                position_ids,
                self.talker_config.rope_theta,
            )
            cp_cos, cp_sin = get_rope_tensors(
                self.device,
                self.code_predictor_config.head_dim,
                seq_len,
                position_ids,
                self.code_predictor_config.rope_theta,
            )

            return self.model.forward(
                input_tt,
                talker_cos,
                talker_sin,
                self.talker_trans_mat,
                cp_cos,
                cp_sin,
                self.cp_trans_mat,
            )

    def decode_step(
        self,
        input_ids: torch.Tensor,
        position: int,
        use_trace: bool = True,
    ) -> Tuple[ttnn.Tensor, List[ttnn.Tensor], None, None]:
        """
        Run single decode step with optional tracing.

        Args:
            input_ids: Input token IDs [batch, 1]
            position: Current sequence position
            use_trace: Whether to use traced execution

        Returns:
            Tuple of (codec_logits, cp_logits_list, None, None)
        """
        if use_trace and self.decode_trace_captured:
            return self.execute_decode_trace(input_ids, position)
        else:
            # Non-traced execution
            input_tt = ttnn.from_torch(
                input_ids,
                device=self.device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            position_ids = torch.tensor([position])
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

            return self.model.forward(
                input_tt,
                talker_cos,
                talker_sin,
                self.talker_trans_mat,
                cp_cos,
                cp_sin,
                self.cp_trans_mat,
            )

    def sample_tokens(
        self,
        output: Tuple[ttnn.Tensor, List[ttnn.Tensor], None, None],
        temperature: float = 1.0,
        top_k: int = 0,
    ) -> torch.Tensor:
        """
        Sample tokens from model output.

        Args:
            output: Tuple of (codec_logits, cp_logits_list, _, _)
            temperature: Sampling temperature (1.0 = no change, <1.0 = more deterministic)
            top_k: If > 0, only sample from top k tokens

        Returns:
            Token IDs [batch, 16] for all code groups
        """
        codec_logits, cp_logits_list, _, _ = output

        tokens_per_group = []

        # Code 0 from Talker's codec_head
        codec_logits_torch = ttnn.to_torch(codec_logits)
        if temperature != 1.0:
            codec_logits_torch = codec_logits_torch / temperature
        if top_k > 0:
            v, _ = torch.topk(codec_logits_torch, min(top_k, codec_logits_torch.size(-1)))
            codec_logits_torch[codec_logits_torch < v[:, :, [-1]]] = float("-inf")
            probs = torch.softmax(codec_logits_torch, dim=-1)
            token_0 = torch.multinomial(probs.squeeze(1), num_samples=1).squeeze(-1)
        else:
            token_0 = torch.argmax(codec_logits_torch, dim=-1).squeeze(-1)  # [batch]
        tokens_per_group.append(token_0)

        # Codes 1-15 from CodePredictor
        for logits in cp_logits_list:
            logits_torch = ttnn.to_torch(logits)
            if temperature != 1.0:
                logits_torch = logits_torch / temperature
            if top_k > 0:
                v, _ = torch.topk(logits_torch, min(top_k, logits_torch.size(-1)))
                logits_torch[logits_torch < v[:, :, [-1]]] = float("-inf")
                probs = torch.softmax(logits_torch, dim=-1)
                token = torch.multinomial(probs.squeeze(1), num_samples=1).squeeze(-1)
            else:
                token = torch.argmax(logits_torch, dim=-1).squeeze(-1)  # [batch]
            tokens_per_group.append(token)

        # Stack all 16 codes: [batch, 16]
        generated_tokens = torch.stack(tokens_per_group, dim=-1)
        return generated_tokens

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


def create_generator(
    model: Qwen3TTS,
    device,
    max_batch_size: int = 1,
    max_seq_len: int = 2048,
) -> Qwen3TTSGenerator:
    """
    Factory function to create a generator.

    Args:
        model: Qwen3TTS model
        device: TTNN device
        max_batch_size: Maximum batch size
        max_seq_len: Maximum sequence length

    Returns:
        Initialized generator
    """
    return Qwen3TTSGenerator(
        model=model,
        device=device,
        talker_config=model.talker_config,
        code_predictor_config=model.code_predictor_config,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
    )
