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
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
