# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTNN Implementation of SpeechT5 Encoder.

Translates from: reference/speecht5_encoder.py
Target PCC: > 0.94 vs PyTorch reference

Architecture:
1. Embedding + Scaled Positional Encoding
2. Pre-encoder Layer Norm + Dropout
3. Compute Relative Position Bias (once)
4. N x Encoder Blocks (Multi-Head Attention + Feed-Forward)
5. Return hidden states

Memory Configuration:
- Weights: DRAM (ttnn.DRAM_MEMORY_CONFIG)
- Activations: L1 (default)
"""

import math
from dataclasses import dataclass
from typing import Tuple

import torch
import ttnn

from models.experimental.speecht5_tts.reference.speecht5_encoder import create_sinusoidal_positions


@dataclass
class TTNNEncoderConfig:
    """Configuration for TTNN SpeechT5 Encoder."""

    vocab_size: int = 81
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    ffn_dim: int = 3072
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    max_position_embeddings: int = 600
    max_relative_distance: int = 160

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads


def create_sinusoidal_positions_torch(max_length: int, hidden_size: int) -> torch.Tensor:
    """Create sinusoidal positional encoding buffer (PyTorch)."""
    position = torch.arange(max_length).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, hidden_size, 2) * (-math.log(10000.0) / hidden_size))

    pe = torch.zeros(1, max_length, hidden_size)
    pe[0, :, 0::2] = torch.sin(position * div_term)
    pe[0, :, 1::2] = torch.cos(position * div_term)

    return pe


class TTNNSpeechT5FeedForward:
    """
    TTNN implementation of SpeechT5 Feed-Forward Network.

    Operations:
    1. Linear: hidden_size -> ffn_dim (768 -> 3072)
    2. GELU activation
    3. Dropout (inference: skip)
    4. Linear: ffn_dim -> hidden_size (3072 -> 768)
    5. Dropout (inference: skip)

    Specialized configs based on tensor shapes:
    - Intermediate: [seq_len, 768] @ [768, 3072] -> [seq_len, 3072]
    - Output: [seq_len, 3072] @ [3072, 768] -> [seq_len, 768]
    """

    def __init__(self, device, parameters, config: TTNNEncoderConfig):
        self.device = device
        self.parameters = parameters
        self.config = config
        self.training = False  # Inference mode

        # Memory configs for operations
        self.L1_MEMCFG = ttnn.L1_MEMORY_CONFIG

        # Pre-compute specialized configs
        self.core_grid = get_encoder_core_grid(device)

        # FFN intermediate: [seq_len, 768] @ [768, 3072]
        # FFN output: [seq_len, 3072] @ [3072, 768]
        self.ffn_compute_config = get_high_perf_compute_config(device)

    def __call__(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        """
        Feed-forward network with comprehensive L1 memory management.

        Args:
            hidden_states: [batch, seq_len, hidden_size]

        Returns:
            output: [batch, seq_len, hidden_size]
        """
        # Op 1: Intermediate dense (768 -> 3072) with specialized config
        hidden_states = ttnn.linear(
            hidden_states,
            self.parameters["intermediate_dense"]["weight"],
            bias=self.parameters["intermediate_dense"]["bias"],
            memory_config=ttnn.L1_MEMORY_CONFIG,
            core_grid=self.core_grid,
            compute_kernel_config=self.ffn_compute_config,
        )

        # Op 2: GELU activation (L1 output)
        hidden_states = ttnn.gelu(hidden_states, memory_config=ttnn.L1_MEMORY_CONFIG)

        # Op 3: Dropout (skip in inference)
        # if self.training:
        #     hidden_states = ttnn.dropout(hidden_states, p=self.config.dropout)

        # Op 4: Output dense (3072 -> 768) with specialized config
        hidden_states = ttnn.linear(
            hidden_states,
            self.parameters["output_dense"]["weight"],
            bias=self.parameters["output_dense"]["bias"],
            memory_config=ttnn.L1_MEMORY_CONFIG,
            core_grid=self.core_grid,
            compute_kernel_config=self.ffn_compute_config,
        )

        # Op 5: Dropout (skip in inference)
        # if self.training:
        #     hidden_states = ttnn.dropout(hidden_states, p=self.config.dropout)

        return hidden_states


class TTNNSpeechT5Attention:
    """
    TTNN implementation of SpeechT5 Multi-Head Self-Attention.

    Key features:
    - Query scaling: 1/sqrt(head_dim)
    - Relative position bias via Q projection
    - Reshape pattern: [B, S, H] -> [B*NH, S, HD]
    - Specialized configs based on tensor shapes

    Operations:
    1-3. Q, K, V projections with query scaling
    4-6. Reshape for multi-head
    7. Compute attention scores (Q @ K^T)
    8-12. Add relative position bias
    13. Softmax
    14. Apply attention to values
    15-16. Reshape back
    17. Output projection
    """

    def __init__(self, device, parameters, config: TTNNEncoderConfig):
        self.device = device
        self.parameters = parameters
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.scaling = 1.0 / math.sqrt(self.head_dim)

        # Memory configs for operations
        self.L1_MEMCFG = ttnn.L1_MEMORY_CONFIG

        # Pre-compute specialized configs for Q/K/V/out projections
        # Shape: [seq_len, 768] @ [768, 768] -> [seq_len, 768]
        self.core_grid = get_encoder_core_grid(device)
        self.linear_compute_config = get_high_perf_compute_config(device)

        # Attention matmul config (Q@K^T, attn@V)
        # Will be computed dynamically based on seq_len
        _, self.attn_compute_config = get_encoder_matmul_config(
            device, M=1, K=self.head_dim, N=1  # Placeholder, actual shapes vary
        )

    def __call__(self, hidden_states: ttnn.Tensor, position_bias: ttnn.Tensor) -> ttnn.Tensor:
        """
        Forward pass with relative position bias.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            position_bias: [seq_len, seq_len, 64]

        Returns:
            output: [batch, seq_len, hidden_size]
        """
        # Get shapes
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]

        # Ops 1-3: Project to Q, K, V with query scaling (specialized configs)
        query_states = ttnn.linear(
            hidden_states,
            self.parameters["q_proj"]["weight"],
            bias=self.parameters["q_proj"]["bias"],
            memory_config=ttnn.L1_MEMORY_CONFIG,
            core_grid=self.core_grid,
            compute_kernel_config=self.linear_compute_config,
        )

        # Scale query
        query_states = ttnn.multiply(query_states, self.scaling, memory_config=ttnn.L1_MEMORY_CONFIG)

        key_states = ttnn.linear(
            hidden_states,
            self.parameters["k_proj"]["weight"],
            bias=self.parameters["k_proj"]["bias"],
            memory_config=ttnn.L1_MEMORY_CONFIG,
            core_grid=self.core_grid,
            compute_kernel_config=self.linear_compute_config,
        )

        value_states = ttnn.linear(
            hidden_states,
            self.parameters["v_proj"]["weight"],
            bias=self.parameters["v_proj"]["bias"],
            memory_config=ttnn.L1_MEMORY_CONFIG,
            core_grid=self.core_grid,
            compute_kernel_config=self.linear_compute_config,
        )

        # PHASE 2: Reshape for multi-head attention (L1 outputs)
        # [batch, seq_len, hidden] -> [batch*num_heads, seq_len, head_dim]

        # Reshape to [batch, seq, num_heads, head_dim] (L1)
        query_states = ttnn.reshape(
            query_states, [batch_size, seq_len, self.num_heads, self.head_dim], memory_config=ttnn.L1_MEMORY_CONFIG
        )
        key_states = ttnn.reshape(
            key_states, [batch_size, seq_len, self.num_heads, self.head_dim], memory_config=ttnn.L1_MEMORY_CONFIG
        )
        value_states = ttnn.reshape(
            value_states, [batch_size, seq_len, self.num_heads, self.head_dim], memory_config=ttnn.L1_MEMORY_CONFIG
        )

        # Transpose to [batch, num_heads, seq, head_dim] (L1)
        query_states = ttnn.permute(query_states, [0, 2, 1, 3], memory_config=ttnn.L1_MEMORY_CONFIG)
        key_states = ttnn.permute(key_states, [0, 2, 1, 3], memory_config=ttnn.L1_MEMORY_CONFIG)
        value_states = ttnn.permute(value_states, [0, 2, 1, 3], memory_config=ttnn.L1_MEMORY_CONFIG)

        # Reshape to [batch*num_heads, seq, head_dim] (L1)
        query_states = ttnn.reshape(
            query_states, [batch_size * self.num_heads, seq_len, self.head_dim], memory_config=ttnn.L1_MEMORY_CONFIG
        )
        key_states = ttnn.reshape(
            key_states, [batch_size * self.num_heads, seq_len, self.head_dim], memory_config=ttnn.L1_MEMORY_CONFIG
        )
        value_states = ttnn.reshape(
            value_states, [batch_size * self.num_heads, seq_len, self.head_dim], memory_config=ttnn.L1_MEMORY_CONFIG
        )

        # PHASE 3: Attention computation with specialized configs
        # Op 7: Compute attention scores: Q @ K^T
        # Need to transpose key_states: [batch*heads, seq, head_dim] -> [batch*heads, head_dim, seq]
        key_states_t = ttnn.permute(key_states, [0, 2, 1], memory_config=ttnn.L1_MEMORY_CONFIG)
        attn_weights = ttnn.matmul(
            query_states,
            key_states_t,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            core_grid=self.core_grid,
            compute_kernel_config=self.attn_compute_config,
        )

        # PHASE 4: Add relative position bias (SpeechT5-specific)
        if position_bias is not None:
            # Op 8: Transpose query for position bias matmul
            # [batch*heads, seq, head_dim] -> [seq, batch*heads, head_dim]
            reshape_q = ttnn.permute(query_states, [1, 0, 2], memory_config=ttnn.L1_MEMORY_CONFIG)

            # Op 9: Transpose position_bias
            # [seq, seq, 64] -> [seq, 64, seq]
            position_bias_t = ttnn.permute(position_bias, [0, 2, 1], memory_config=ttnn.L1_MEMORY_CONFIG)

            # Op 10: Matmul for relative position bias
            # [seq, batch*heads, head_dim] @ [seq, 64, seq] -> [seq, batch*heads, seq]
            # Note: head_dim must be 64 for this to work!
            rel_pos_bias = ttnn.matmul(
                reshape_q,
                position_bias_t,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                core_grid=self.core_grid,
                compute_kernel_config=self.attn_compute_config,
            )

            # Op 11: Transpose back
            # [seq, batch*heads, seq] -> [batch*heads, seq, seq]
            rel_pos_bias = ttnn.permute(rel_pos_bias, [1, 0, 2], memory_config=ttnn.L1_MEMORY_CONFIG)

            # Op 12: Add to attention weights
            attn_weights = ttnn.add(attn_weights, rel_pos_bias, memory_config=ttnn.L1_MEMORY_CONFIG)

        # Op 13: Softmax
        attn_weights = ttnn.softmax(attn_weights, dim=-1)

        # Op 14: Apply attention to values
        # [batch*heads, seq, seq] @ [batch*heads, seq, head_dim]
        attn_output = ttnn.matmul(
            attn_weights,
            value_states,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            core_grid=self.core_grid,
            compute_kernel_config=self.attn_compute_config,
        )

        # PHASE 5: Reshape back to [batch, seq, hidden]
        # [batch*heads, seq, head_dim] -> [batch, heads, seq, head_dim]
        attn_output = ttnn.reshape(
            attn_output, [batch_size, self.num_heads, seq_len, self.head_dim], memory_config=ttnn.L1_MEMORY_CONFIG
        )

        # [batch, heads, seq, head_dim] -> [batch, seq, heads, head_dim]
        attn_output = ttnn.permute(attn_output, [0, 2, 1, 3], memory_config=ttnn.L1_MEMORY_CONFIG)

        # [batch, seq, heads, head_dim] -> [batch, seq, hidden]
        attn_output = ttnn.reshape(
            attn_output, [batch_size, seq_len, self.hidden_size], memory_config=ttnn.L1_MEMORY_CONFIG
        )

        # Op 17: Final linear projection with specialized configs
        output = ttnn.linear(
            attn_output,
            self.parameters["out_proj"]["weight"],
            bias=self.parameters["out_proj"]["bias"],
            memory_config=ttnn.L1_MEMORY_CONFIG,
            core_grid=self.core_grid,
            compute_kernel_config=self.linear_compute_config,
        )

        return output


class TTNNSpeechT5EncoderBlock:
    """
    TTNN implementation of single encoder block.

    Pattern: POST-NORM
    1. Attention + Dropout + Residual + LayerNorm
    2. FFN + Residual + LayerNorm
    """

    def __init__(self, device, parameters, config: TTNNEncoderConfig):
        self.device = device
        self.config = config
        self.training = False  # Inference mode

        # Sub-modules
        self.attention = TTNNSpeechT5Attention(
            device,
            parameters["attention"],
            config,
        )
        self.feed_forward = TTNNSpeechT5FeedForward(
            device,
            parameters["feed_forward"],
            config,
        )

        # Layer norm parameters
        self.layer_norm_params = parameters["layer_norm"]
        self.final_layer_norm_params = parameters["final_layer_norm"]

    def __call__(self, hidden_states: ttnn.Tensor, position_bias: ttnn.Tensor) -> ttnn.Tensor:
        """
        Encoder block with comprehensive L1 memory management.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            position_bias: [seq_len, seq_len, 64]

        Returns:
            hidden_states: [batch, seq_len, hidden_size]
        """

        # Attention sub-layer (POST-NORM pattern)
        residual = hidden_states

        # Attention - rely on internal L1 memory configs
        hidden_states = self.attention(hidden_states, position_bias=position_bias)

        # Dropout intentionally omitted in this TTNN encoder implementation

        # Residual connection
        hidden_states = ttnn.add(residual, hidden_states, memory_config=ttnn.L1_MEMORY_CONFIG)

        # Layer norm
        hidden_states = ttnn.layer_norm(
            hidden_states,
            weight=self.layer_norm_params["weight"],
            bias=self.layer_norm_params["bias"],
            epsilon=self.config.layer_norm_eps,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # PHASE 2: Feed-forward sub-layer
        residual_ffn = hidden_states

        # Feed-forward - rely on internal L1 memory configs
        ffn_output = self.feed_forward(hidden_states)

        # Residual connection
        hidden_states = ttnn.add(residual_ffn, ffn_output, memory_config=ttnn.L1_MEMORY_CONFIG)

        # Final layer norm
        hidden_states = ttnn.layer_norm(
            hidden_states,
            weight=self.final_layer_norm_params["weight"],
            bias=self.final_layer_norm_params["bias"],
            epsilon=self.config.layer_norm_eps,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        return hidden_states


class TTNNSpeechT5Encoder:
    """
    TTNN implementation of complete SpeechT5 Encoder.

    Flow:
    1. Embedding lookup
    2. Add scaled positional encoding
    3. Pre-encoder layer norm
    4. Pre-encoder dropout (skip in inference)
    5. Compute relative position bias (once)
    6. Pass through N encoder blocks
    7. Return hidden states
    """

    def __init__(self, device, parameters, config: TTNNEncoderConfig):
        self.device = device
        self.parameters = parameters
        self.config = config
        self.training = False  # Inference mode
        self.max_relative_distance = getattr(
            config, "max_relative_distance", getattr(config, "encoder_max_relative_position", 160)
        )  # Use config value (160 for HF)

        # Memory configs for operations
        self.L1_MEMCFG = ttnn.L1_MEMORY_CONFIG

        # Encoder blocks
        self.layers = [
            TTNNSpeechT5EncoderBlock(
                device,
                parameters["layers"][i],
                config,
            )
            for i in range(config.num_layers)
        ]

        # Pre-compute position bias tensors for common sequence lengths
        # This avoids dynamic tensor creation during forward pass (required for trace)
        self.position_bias_cache = {}
        self._precompute_position_bias_for_common_lengths()

    def __call__(self, input_ids: ttnn.Tensor) -> Tuple[ttnn.Tensor]:
        """
        Forward pass with optimized L1 memory management.

        Args:
            input_ids: [batch, seq_len] - token IDs

        Returns:
            tuple: (hidden_states,) where hidden_states is [batch, seq_len, hidden_size]
        """
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]

        # Op 1: Embedding lookup (L1 output)
        hidden_states = ttnn.embedding(
            input_ids,
            self.parameters["embed_tokens"]["weight"],
            memory_config=self.L1_MEMCFG,
        )

        pe_slice = self.parameters["positional_encoding"][:, :seq_len, :]

        hidden_states = hidden_states + self.parameters["encode_positions_alpha"] * pe_slice

        # Op 2: Pre-encoder layer norm
        hidden_states = ttnn.layer_norm(
            hidden_states,
            weight=self.parameters["layer_norm"]["weight"],
            bias=self.parameters["layer_norm"]["bias"],
            epsilon=self.config.layer_norm_eps,
            memory_config=self.L1_MEMCFG,
        )

        # Op 4: Pre-encoder dropout is omitted in this TTNN implementation (inference-focused).

        # Op 3: Encoder blocks
        position_bias = self._compute_position_bias(seq_len)
        for block in self.layers:
            hidden_states = block(hidden_states, position_bias=position_bias)

        return (hidden_states,)

    def prepare_encoder_inputs(self, input_ids: ttnn.Tensor):
        """
        Prepare inputs for trace execution by ensuring proper memory config.

        This method separates input preparation from the forward pass to support trace capture.

        Args:
            input_ids: [batch, seq_len] - token IDs

        Returns:
            List of prepared input tensors
        """
        # Ensure input is in L1 memory config for trace compatibility
        prepared_input_ids = ttnn.to_memory_config(input_ids, ttnn.L1_MEMORY_CONFIG)
        return [prepared_input_ids]

    def _precompute_position_bias_for_common_lengths(self):
        """
        Pre-compute position bias tensors for common sequence lengths.
        This avoids dynamic tensor creation during forward pass, which is
        required for trace support.

        Pre-computes for lengths: 13 (Hello world), 20 (typical)
        Note: Add more lengths as needed for your use case.
        """
        common_lengths = [13, 20]

        for seq_len in common_lengths:
            # Create position sequence
            pos_seq = torch.arange(0, seq_len, dtype=torch.long)

            # Compute relative positions [seq_len, seq_len]
            relative_positions = pos_seq[:, None] - pos_seq[None, :]

            # Clamp to max distance
            relative_positions = torch.clamp(
                relative_positions,
                -self.max_relative_distance,
                self.max_relative_distance - 1,
            )

            # Shift to positive indices
            relative_positions = relative_positions + self.max_relative_distance

            # Convert to int32 first, then to uint32 for TTNN embedding
            relative_positions = relative_positions.to(torch.int32)

            # Convert to TTNN tensor on device
            relative_positions_ttnn = ttnn.from_torch(
                relative_positions,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            # Embedding lookup
            # Note: Do NOT specify layout for embedding - let it determine output layout automatically
            position_embeddings = ttnn.embedding(
                relative_positions_ttnn,
                self.parameters["relative_pe_k"]["weight"],
                memory_config=self.L1_MEMCFG,
            )

            # Cache the result
            self.position_bias_cache[seq_len] = position_embeddings

    def _compute_position_bias(self, seq_len: int) -> ttnn.Tensor:
        """
        Get position bias for the given sequence length.
        Uses pre-computed cache when available, otherwise computes on-the-fly.

        Args:
            seq_len: Sequence length

        Returns:
            position_bias: [seq_len, seq_len, 64]

        Note:
            For trace support, the sequence length must be pre-computed in the cache.
            On-the-fly computation will fail during trace capture.
        """
        # Check cache first (required for trace support)
        if seq_len in self.position_bias_cache:
            return self.position_bias_cache[seq_len]

        # If not in cache, compute on-the-fly (this won't work during trace!)
        # This path is only for non-traced execution with unusual sequence lengths
        pos_seq = torch.arange(0, seq_len, dtype=torch.long)
        relative_positions = pos_seq[:, None] - pos_seq[None, :]
        relative_positions = torch.clamp(
            relative_positions,
            -self.max_relative_distance,
            self.max_relative_distance - 1,
        )
        relative_positions = relative_positions + self.max_relative_distance

        # Convert indices to TTNN tensor
        relative_positions_ttnn = ttnn.from_torch(
            relative_positions,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
        )

        # Do embedding lookup using ttnn.embedding
        # We've confirmed this works correctly (PCC=0.999999 vs HF)
        position_embeddings = ttnn.embedding(
            relative_positions_ttnn,
            self.parameters["relative_pe_k"]["weight"],
            layout=ttnn.TILE_LAYOUT,
            memory_config=self.L1_MEMCFG,
        )

        return position_embeddings


# ============================================================================
# Memory Management Utilities - Comprehensive L1 Optimization
# ============================================================================


def get_encoder_core_grid(device):
    """
    Get optimal core grid for encoder operations based on device.
    N150 supports 8x8 grid.
    """
    compute_grid = device.compute_with_storage_grid_size()
    return ttnn.CoreGrid(y=compute_grid.y, x=compute_grid.x)


def get_encoder_linear_config(device, M, K, N):
    """
    Get specialized program config for encoder linear operations.

    Args:
        device: TTNN device
        M: Number of rows in input (seq_len for encoder)
        K: Input dimension (hidden_size or ffn_dim)
        N: Output dimension (hidden_size or ffn_dim)

    Returns:
        tuple: (core_grid, compute_kernel_config) for the linear operation

    Tensor shapes in encoder:
        Q/K/V projections: [seq_len, 768] @ [768, 768] (M=seq_len, K=768, N=768)
        FFN intermediate: [seq_len, 768] @ [768, 3072] (M=seq_len, K=768, N=3072)
        FFN output: [seq_len, 3072] @ [3072, 768] (M=seq_len, K=3072, N=768)
    """
    compute_grid = device.compute_with_storage_grid_size()
    core_grid = ttnn.CoreGrid(y=compute_grid.y, x=compute_grid.x)

    # Compute kernel config - use HiFi4 for accuracy, enable packer L1 for bandwidth
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,  # FP32 accumulation for better accuracy
        packer_l1_acc=False,  # Disable when using fp32_dest_acc
    )

    return core_grid, compute_kernel_config


def get_encoder_matmul_config(device, M, K, N):
    """
    Get specialized config for encoder matmul operations (attention).

    Args:
        device: TTNN device
        M: Rows in first matrix (typically num_heads * seq_len or seq_len)
        K: Shared dimension
        N: Columns in second matrix

    Returns:
        tuple: (core_grid, compute_kernel_config) for the matmul operation

    Attention shapes:
        Q @ K^T: [batch*heads, seq, head_dim] @ [batch*heads, head_dim, seq]
        attn @ V: [batch*heads, seq, seq] @ [batch*heads, seq, head_dim]
    """
    compute_grid = device.compute_with_storage_grid_size()
    core_grid = ttnn.CoreGrid(y=compute_grid.y, x=compute_grid.x)

    # For attention matmuls, use HiFi2 for speed (attention is less sensitive)
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,  # Enable L1 accumulation for bandwidth
    )

    return core_grid, compute_kernel_config


# ============================================================================
# High-Performance Compute Kernel Configs - Maximum Core Utilization
# ============================================================================


def get_high_perf_compute_config(device=None):
    """
    Get compute kernel config optimized for maximum core utilization and performance.
    Uses HiFi4 with default settings for accuracy while maintaining L1 memory optimizations.

    Args:
        device: Optional TTNN device. If provided, uses device.arch() for config.
    """
    if device is not None:
        return ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
    # Fallback for backward compatibility
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )


def preprocess_encoder_parameters(torch_encoder, config: TTNNEncoderConfig, device):
    """
    Preprocess PyTorch encoder parameters for TTNN.

    Args:
        torch_encoder: PyTorch SpeechT5Encoder
        config: Encoder configuration
        device: TTNN device

    Returns:
        parameters: Dictionary of TTNN parameters

    Note:
        - Weights stored in DRAM
        - Linear weights transposed
        - Layer norm parameters kept as-is
    """
    parameters = {}

    # Memory configs
    DRAM_MEMCFG = ttnn.DRAM_MEMORY_CONFIG
    L1_MEMCFG = ttnn.L1_MEMORY_CONFIG

    # 1. Embedding weights (from prenet in composite encoder)
    parameters["embed_tokens"] = {
        "weight": ttnn.from_torch(
            torch_encoder.prenet.embed_tokens.weight.data,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=DRAM_MEMCFG,
        )
    }

    # Get core encoder for other attributes
    core_encoder = torch_encoder.wrapped_encoder

    # 2. Relative positional encoding weights (from embed_positions)
    parameters["relative_pe_k"] = {
        "weight": ttnn.from_torch(
            core_encoder.embed_positions.pe_k.weight.data,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=DRAM_MEMCFG,
        )
    }

    # 3. Create positional encoding tensor
    pe = create_sinusoidal_positions(config.max_position_embeddings, config.hidden_size)
    parameters["positional_encoding"] = ttnn.from_torch(
        pe,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=DRAM_MEMCFG,
    )

    # 4. Positional encoding alpha
    parameters["encode_positions_alpha"] = ttnn.from_torch(
        torch_encoder.prenet.encode_positions.alpha.data.unsqueeze(0),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=DRAM_MEMCFG,
    )

    # 5. Pre-encoder layer norm
    parameters["layer_norm"] = {
        "weight": ttnn.from_torch(
            core_encoder.layer_norm.weight.data,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=DRAM_MEMCFG,
        ),
        "bias": ttnn.from_torch(
            core_encoder.layer_norm.bias.data,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=DRAM_MEMCFG,
        ),
    }

    # 6. Encoder layers
    parameters["layers"] = []
    for i, torch_block in enumerate(core_encoder.layers):
        block_params = {}

        # Attention parameters
        block_params["attention"] = {
            "q_proj": {
                "weight": ttnn.from_torch(
                    torch_block.attention.q_proj.weight.data.T,  # Transpose!
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=DRAM_MEMCFG,
                ),
                "bias": ttnn.from_torch(
                    torch_block.attention.q_proj.bias.data,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=DRAM_MEMCFG,
                ),
            },
            "k_proj": {
                "weight": ttnn.from_torch(
                    torch_block.attention.k_proj.weight.data.T,  # Transpose!
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=DRAM_MEMCFG,
                ),
                "bias": ttnn.from_torch(
                    torch_block.attention.k_proj.bias.data,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=DRAM_MEMCFG,
                ),
            },
            "v_proj": {
                "weight": ttnn.from_torch(
                    torch_block.attention.v_proj.weight.data.T,  # Transpose!
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=DRAM_MEMCFG,
                ),
                "bias": ttnn.from_torch(
                    torch_block.attention.v_proj.bias.data,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=DRAM_MEMCFG,
                ),
            },
            "out_proj": {
                "weight": ttnn.from_torch(
                    torch_block.attention.out_proj.weight.data.T,  # Transpose!
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=DRAM_MEMCFG,
                ),
                "bias": ttnn.from_torch(
                    torch_block.attention.out_proj.bias.data,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=DRAM_MEMCFG,
                ),
            },
        }

        # Feed-forward parameters
        block_params["feed_forward"] = {
            "intermediate_dense": {
                "weight": ttnn.from_torch(
                    torch_block.feed_forward.intermediate_dense.weight.data.T,  # Transpose!
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=DRAM_MEMCFG,
                ),
                "bias": ttnn.from_torch(
                    torch_block.feed_forward.intermediate_dense.bias.data,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=DRAM_MEMCFG,
                ),
            },
            "output_dense": {
                "weight": ttnn.from_torch(
                    torch_block.feed_forward.output_dense.weight.data.T,  # Transpose!
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=DRAM_MEMCFG,
                ),
                "bias": ttnn.from_torch(
                    torch_block.feed_forward.output_dense.bias.data,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=DRAM_MEMCFG,
                ),
            },
        }

        # Layer norm parameters
        block_params["layer_norm"] = {
            "weight": ttnn.from_torch(
                torch_block.layer_norm.weight.data,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=DRAM_MEMCFG,
            ),
            "bias": ttnn.from_torch(
                torch_block.layer_norm.bias.data,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=DRAM_MEMCFG,
            ),
        }

        block_params["final_layer_norm"] = {
            "weight": ttnn.from_torch(
                torch_block.final_layer_norm.weight.data,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=DRAM_MEMCFG,
            ),
            "bias": ttnn.from_torch(
                torch_block.final_layer_norm.bias.data,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=DRAM_MEMCFG,
            ),
        }

        parameters["layers"].append(block_params)

    return parameters


if __name__ == "__main__":
    from models.experimental.speecht5_tts.reference.speecht5_encoder import (
        load_from_huggingface,
        SpeechT5Config as PyTorchConfig,
    )

    print("=" * 80)
    print("TTNN SpeechT5 Encoder Test")
    print("=" * 80)

    # Load PyTorch model
    print("\n1. Loading PyTorch encoder...")
    torch_encoder = load_from_huggingface()
    torch_encoder.eval()

    torch_config = PyTorchConfig()
    ttnn_config = TTNNEncoderConfig(
        vocab_size=torch_config.vocab_size,
        hidden_size=torch_config.hidden_size,
        num_layers=torch_config.num_layers,
        num_heads=torch_config.num_heads,
        ffn_dim=torch_config.ffn_dim,
        dropout=torch_config.dropout,
        layer_norm_eps=torch_config.layer_norm_eps,
        max_position_embeddings=torch_config.max_position_embeddings,
        max_relative_distance=torch_config.max_relative_distance,
    )

    # Initialize TTNN device
    print("\n2. Initializing TTNN device...")
    device = ttnn.open_device(device_id=0)

    # Preprocess parameters
    print("\n3. Preprocessing parameters...")
    parameters = preprocess_encoder_parameters(torch_encoder, ttnn_config, device)

    # Create TTNN encoder
    print("\n4. Creating TTNN encoder...")
    ttnn_encoder = TTNNSpeechT5Encoder(
        device=device,
        parameters=parameters,
        config=ttnn_config,
    )

    # Test forward pass
    print("\n5. Testing forward pass...")
    torch.manual_seed(42)
    input_ids = torch.randint(0, torch_config.vocab_size, (1, 15))

    # PyTorch forward
    with torch.no_grad():
        torch_output = torch_encoder(input_ids)[0]

    # TTNN forward
    ttnn_input = ttnn.from_torch(
        input_ids,
        dtype=ttnn.uint32,  # Must be UINT32 for embedding
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    ttnn_output = ttnn_encoder(ttnn_input)[0]
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    # Compute PCC
    def compute_pcc(a, b):
        a_flat = a.flatten().float()
        b_flat = b.flatten().float()
        return torch.corrcoef(torch.stack([a_flat, b_flat]))[0, 1].item()

    pcc = compute_pcc(torch_output, ttnn_output_torch)
    print(f"\nPCC: {pcc:.6f}")

    if pcc > 0.94:
        print("✅ PASS: PCC > 0.94")
    else:
        print("❌ FAIL: PCC < 0.94")

    # Cleanup
    ttnn.close_device(device)

    print("\n" + "=" * 80)
