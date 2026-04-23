# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Configuration classes for Qwen3-TTS TTNN implementation.
"""

from dataclasses import dataclass
from typing import Tuple

import ttnn


@dataclass
class Qwen3TTSTalkerConfig:
    """Configuration for the Qwen3-TTS Talker model."""

    hidden_size: int = 2048
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 6144
    text_vocab_size: int = 151936
    audio_vocab_size: int = 3072
    max_position_embeddings: int = 32768
    rope_theta: float = 1000000.0
    rms_norm_eps: float = 1e-6
    num_code_groups: int = 16
    mrope_section: Tuple[int, int, int] = (24, 20, 20)
    mrope_interleaved: bool = True

    # TTNN specific settings
    tile_size: int = 32

    @property
    def qkv_size(self) -> int:
        """Combined size of Q, K, V projections."""
        return (self.num_attention_heads + 2 * self.num_key_value_heads) * self.head_dim


@dataclass
class Qwen3TTSCodePredictorConfig:
    """Configuration for the Qwen3-TTS Code Predictor model."""

    hidden_size: int = 1024
    num_hidden_layers: int = 5
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 3072
    vocab_size: int = 2048
    max_position_embeddings: int = 65536
    rope_theta: float = 1000000.0
    rms_norm_eps: float = 1e-6
    num_code_groups: int = 16

    # TTNN specific settings
    tile_size: int = 32

    @property
    def qkv_size(self) -> int:
        """Combined size of Q, K, V projections."""
        return (self.num_attention_heads + 2 * self.num_key_value_heads) * self.head_dim


def get_device_core_grid(device) -> ttnn.CoreGrid:
    """Full compute grid for ``ttnn.linear`` / matmul (improves utilization on small-M decode)."""
    gs = device.compute_with_storage_grid_size()
    return ttnn.CoreGrid(y=gs.y, x=gs.x)


# L1 interleaved activations for ops that follow decode matmuls (residual ``add``, norms).
_L1_INTERLEAVED = ttnn.MemoryConfig(
    memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
    buffer_type=ttnn.BufferType.L1,
)


def mlp_decode_linear_output_memory_config(mode: str) -> ttnn.MemoryConfig:
    """
    Decode SwiGLU ``ttnn.linear`` outputs: L1 width-sharded instead of L1 interleaved
    (``tech_reports/LLMs/llms.md`` §4.3.1 — less DRAM traffic on small-M).
    Prefill keeps DRAM interleaved.
    """
    if mode == "decode":
        return ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG
    return ttnn.DRAM_MEMORY_CONFIG


def restore_mlp_decode_output_to_interleaved_l1(x: ttnn.Tensor) -> ttnn.Tensor:
    """Undo width sharding after the MLP so residuals match attention output layout."""
    return ttnn.to_memory_config(x, _L1_INTERLEAVED)


def code_predictor_decode_linear_output_memory_config(mode: str) -> ttnn.MemoryConfig:
    """Decode linears (input proj / LM head): width-sharded L1; prefill stays DRAM."""
    if mode == "decode":
        return ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG
    return ttnn.DRAM_MEMORY_CONFIG


def restore_code_predictor_linear_output_to_dram(x: ttnn.Tensor) -> ttnn.Tensor:
    """Restore CP linear outputs to DRAM interleaved for the rest of the CP graph / D2H."""
    return ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)


def talker_codec_decode_one_token_linear_output_memory_config(seq_len: int) -> ttnn.MemoryConfig:
    """Single-token Talker codec_head decode: width-sharded L1; multi-token prefill uses L1 interleaved."""
    if seq_len == 1:
        return ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG
    return ttnn.L1_MEMORY_CONFIG


def restore_talker_codec_logits_memory(x: ttnn.Tensor, *, seq_len: int) -> ttnn.Tensor:
    """Match prior layout: L1 interleaved for seq==1 after width-sharded linear, else unchanged path."""
    if seq_len == 1:
        return ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
    return x


def get_compute_kernel_config():
    """Returns compute kernel config for high-fidelity computations."""
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def get_compute_kernel_config_hifi4():
    """Returns compute kernel config for highest fidelity computations."""
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
