# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""GQA configuration dataclass and model presets.

Provides a flexible GQAConfig that captures all parameters needed to instantiate
a Grouped-Query Attention block, plus factory functions for each target model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class GQAConfig:
    """Configuration for a single Grouped-Query Attention block.

    Parameters
    ----------
    hidden_size : int
        Model / residual-stream width (d_model).
    num_attention_heads : int
        Number of query heads.
    num_key_value_heads : int
        Number of key/value heads (< num_attention_heads for GQA).
    head_dim : int
        Dimension per head.  When ``num_attention_heads * head_dim != hidden_size``
        the projection matrices are non-square (e.g. GLM-4.7).
    attention_bias : bool
        Whether Q/K/V/O projections use bias terms.
    rope_theta : float
        Base frequency for Rotary Position Embedding.
    max_position_embeddings : int
        Maximum sequence length for the RoPE cache.
    rms_norm_eps : float
        Epsilon for pre-attention RMSNorm.
    rope_partial_factor : float
        Fraction of head_dim that receives RoPE (1.0 = full).
        GLM-4 uses 0.5 (partial_rotary_factor).
    use_qk_norm : bool
        Apply RMSNorm to Q and K after projection (used by GLM-4).
    attn_logit_softcapping : Optional[float]
        If set, softcap attention logits to this value (used by Grok-2).
    scaling : Optional[float]
        Custom Q·K scaling factor.  Defaults to 1/sqrt(head_dim).
    model_name : str
        Human-readable identifier for this configuration.
    """

    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    attention_bias: bool = False
    rope_theta: float = 10000.0
    max_position_embeddings: int = 8192
    rms_norm_eps: float = 1e-5
    rope_partial_factor: float = 1.0
    use_qk_norm: bool = False
    attn_logit_softcapping: Optional[float] = None
    scaling: Optional[float] = None
    model_name: str = ""

    @property
    def num_kv_groups(self) -> int:
        """Number of query heads per KV head."""
        assert self.num_attention_heads % self.num_key_value_heads == 0
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def q_proj_size(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_proj_size(self) -> int:
        return self.num_key_value_heads * self.head_dim

    @property
    def effective_scaling(self) -> float:
        if self.scaling is not None:
            return self.scaling
        return self.head_dim**-0.5


# ---------------------------------------------------------------------------
# Model presets
# ---------------------------------------------------------------------------

GLM_4_7_355B = GQAConfig(
    hidden_size=5120,
    num_attention_heads=96,
    num_key_value_heads=8,
    head_dim=128,
    attention_bias=True,
    rope_theta=1_000_000,
    max_position_embeddings=131072,
    rms_norm_eps=1e-5,
    rope_partial_factor=0.5,
    use_qk_norm=False,
    model_name="glm_4_7_355b",
)

GPT_OSS_20B = GQAConfig(
    hidden_size=2880,
    num_attention_heads=64,
    num_key_value_heads=8,
    head_dim=64,
    attention_bias=True,
    rope_theta=150_000,
    max_position_embeddings=131072,
    rms_norm_eps=1e-5,
    model_name="gpt_oss_20b",
)

GPT_OSS_120B = GQAConfig(
    hidden_size=2880,
    num_attention_heads=64,
    num_key_value_heads=8,
    head_dim=64,
    attention_bias=True,
    rope_theta=150_000,
    max_position_embeddings=131072,
    rms_norm_eps=1e-5,
    model_name="gpt_oss_120b",
)

GROK_2_270B = GQAConfig(
    hidden_size=8192,
    num_attention_heads=64,
    num_key_value_heads=8,
    head_dim=128,
    attention_bias=False,
    rope_theta=208_533_496,
    max_position_embeddings=131072,
    rms_norm_eps=1e-5,
    attn_logit_softcapping=30.0,
    model_name="grok_2_270b",
)

LLAMA_GUARD_4 = GQAConfig(
    hidden_size=5120,
    num_attention_heads=40,
    num_key_value_heads=8,
    head_dim=128,
    attention_bias=False,
    rope_theta=500_000,
    max_position_embeddings=131072,
    rms_norm_eps=1e-5,
    model_name="llama_guard_4",
)

QWEN3_235B = GQAConfig(
    hidden_size=4096,
    num_attention_heads=64,
    num_key_value_heads=4,
    head_dim=128,
    attention_bias=False,
    rope_theta=1_000_000,
    max_position_embeddings=40960,
    rms_norm_eps=1e-6,
    model_name="qwen3_235b",
)

ALL_MODEL_CONFIGS = [
    GLM_4_7_355B,
    GPT_OSS_20B,
    GPT_OSS_120B,
    GROK_2_270B,
    LLAMA_GUARD_4,
    QWEN3_235B,
]
