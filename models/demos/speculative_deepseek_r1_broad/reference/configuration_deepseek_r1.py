# SPDX-FileCopyrightText: © 2023 DeepSeek-AI and The HuggingFace Inc. team.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class DeepseekR1Config(PretrainedConfig):
    r"""
    Configuration class for DeepSeek-R1-0528 models.

    DeepSeek-R1-0528 shares the DeepSeek-V3 architecture (MLA attention with
    compressed KV, Mixture-of-Experts with grouped routing, YaRN RoPE scaling).
    This config mirrors ``DeepseekV3Config`` with defaults matching the
    R1-0528 release.

    Args:
        vocab_size (`int`, *optional*, defaults to 129280):
            Vocabulary size of the model.
        hidden_size (`int`, *optional*, defaults to 7168):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 18432):
            Dimension of the dense MLP representations.
        moe_intermediate_size (`int`, *optional*, defaults to 2048):
            Dimension of the MoE expert MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 61):
            Number of transformer decoder layers.
        num_nextn_predict_layers (`int`, *optional*, defaults to 1):
            Number of next-n prediction layers.
        num_attention_heads (`int`, *optional*, defaults to 128):
            Number of attention heads.
        num_key_value_heads (`int`, *optional*):
            Number of key/value heads for GQA.  Defaults to ``num_attention_heads``.
        n_shared_experts (`int`, *optional*, defaults to 1):
            Number of shared (always-active) MoE experts.
        n_routed_experts (`int`, *optional*, defaults to 256):
            Total number of routed MoE experts.
        ep_size (`int`, *optional*, defaults to 1):
            Expert-parallel world size.
        routed_scaling_factor (`float`, *optional*, defaults to 2.5):
            Scaling factor applied to routed expert weights.
        kv_lora_rank (`int`, *optional*, defaults to 512):
            Rank of the low-rank KV projection in MLA.
        q_lora_rank (`int`, *optional*, defaults to 1536):
            Rank of the low-rank Q projection in MLA.
        qk_rope_head_dim (`int`, *optional*, defaults to 64):
            Head dimension for the RoPE portion of Q/K.
        v_head_dim (`int`, *optional*, defaults to 128):
            Head dimension for values.
        qk_nope_head_dim (`int`, *optional*, defaults to 128):
            Head dimension for the non-positional portion of Q/K.
        topk_method (`str`, *optional*, defaults to ``"noaux_tc"``):
            Top-k selection method for MoE routing.
        n_group (`int`, *optional*, defaults to 8):
            Number of expert groups for grouped routing.
        topk_group (`int`, *optional*, defaults to 4):
            Number of groups selected per token.
        num_experts_per_tok (`int`, *optional*, defaults to 8):
            Number of routed experts activated per token.
        moe_layer_freq (`int`, *optional*, defaults to 1):
            Frequency of MoE layers (every ``moe_layer_freq`` layers).
        first_k_dense_replace (`int`, *optional*, defaults to 3):
            Number of initial dense layers before MoE layers begin.
        norm_topk_prob (`bool`, *optional*, defaults to ``True``):
            Whether to normalize routed expert probabilities.
        scoring_func (`str`, *optional*, defaults to ``"sigmoid"``):
            Scoring function for MoE gating.
        hidden_act (`str`, *optional*, defaults to ``"silu"``):
            Activation function in MLP layers.
        max_position_embeddings (`int`, *optional*, defaults to 163840):
            Maximum sequence length the model supports.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Standard deviation for weight initialization.
        rms_norm_eps (`float`, *optional*, defaults to 1e-6):
            Epsilon for RMS normalization.
        use_cache (`bool`, *optional*, defaults to ``True``):
            Whether to return KV cache for incremental decoding.
        pad_token_id (`int`, *optional*):
            Padding token ID.
        bos_token_id (`int`, *optional*, defaults to 0):
            Beginning-of-sequence token ID.
        eos_token_id (`int`, *optional*, defaults to 1):
            End-of-sequence token ID.
        tie_word_embeddings (`bool`, *optional*, defaults to ``False``):
            Whether to tie input/output embeddings.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            Base frequency for RoPE.
        rope_scaling (`Dict`, *optional*):
            RoPE scaling configuration (YaRN, linear, dynamic).
        attention_bias (`bool`, *optional*, defaults to ``False``):
            Whether to use bias in attention projections.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Dropout probability for attention weights.

    ```python
    >>> from models.demos.speculative_deepseek_r1_broad.reference.configuration_deepseek_r1 import DeepseekR1Config
    >>> config = DeepseekR1Config()
    >>> config.hidden_size
    7168
    ```
    """

    model_type = "deepseek_v3"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=129280,
        hidden_size=7168,
        intermediate_size=18432,
        moe_intermediate_size=2048,
        num_hidden_layers=61,
        num_nextn_predict_layers=1,
        num_attention_heads=128,
        num_key_value_heads=128,
        n_shared_experts=1,
        n_routed_experts=256,
        ep_size=1,
        routed_scaling_factor=2.5,
        kv_lora_rank=512,
        q_lora_rank=1536,
        qk_rope_head_dim=64,
        v_head_dim=128,
        qk_nope_head_dim=128,
        topk_method="noaux_tc",
        n_group=8,
        topk_group=4,
        num_experts_per_tok=8,
        moe_layer_freq=1,
        first_k_dense_replace=3,
        norm_topk_prob=True,
        scoring_func="sigmoid",
        hidden_act="silu",
        max_position_embeddings=163840,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=0,
        eos_token_id=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.num_attention_heads = num_attention_heads
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.ep_size = ep_size
        self.routed_scaling_factor = routed_scaling_factor
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.topk_method = topk_method
        self.n_group = n_group
        self.topk_group = topk_group
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_layer_freq = moe_layer_freq
        self.first_k_dense_replace = first_k_dense_replace
        self.norm_topk_prob = norm_topk_prob
        self.scoring_func = scoring_func
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Backward-compatible summary config used by the eagle3 demo scripts
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DeepSeekR1ReferenceConfig:
    """Lightweight summary extracted from any HF-compatible checkpoint."""

    model_id: str
    architecture: str
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    max_position_embeddings: int
    rope_theta: float
    torch_dtype: str
    intermediate_size: int
    moe_intermediate_size: int
    n_routed_experts: int
    n_shared_experts: int
    num_experts_per_tok: int
    kv_lora_rank: int
    q_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    rms_norm_eps: float
    first_k_dense_replace: int
    moe_layer_freq: int


def _get_first_architecture(cfg: object) -> str:
    architectures = getattr(cfg, "architectures", None) or []
    return str(architectures[0]) if architectures else cfg.__class__.__name__


def load_reference_config(model_id: str, *, trust_remote_code: bool = False) -> DeepSeekR1ReferenceConfig:
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    max_pos = int(getattr(cfg, "max_position_embeddings", getattr(cfg, "max_sequence_length", 0)))
    return DeepSeekR1ReferenceConfig(
        model_id=model_id,
        architecture=_get_first_architecture(cfg),
        hidden_size=int(getattr(cfg, "hidden_size", 0)),
        num_hidden_layers=int(getattr(cfg, "num_hidden_layers", 0)),
        num_attention_heads=int(getattr(cfg, "num_attention_heads", 0)),
        num_key_value_heads=int(getattr(cfg, "num_key_value_heads", getattr(cfg, "num_attention_heads", 0))),
        vocab_size=int(getattr(cfg, "vocab_size", 0)),
        max_position_embeddings=max_pos,
        rope_theta=float(getattr(cfg, "rope_theta", 0.0)),
        torch_dtype=str(getattr(cfg, "torch_dtype", "unknown")),
        intermediate_size=int(getattr(cfg, "intermediate_size", 0)),
        moe_intermediate_size=int(getattr(cfg, "moe_intermediate_size", 0)),
        n_routed_experts=int(getattr(cfg, "n_routed_experts", 0)),
        n_shared_experts=int(getattr(cfg, "n_shared_experts", 0)),
        num_experts_per_tok=int(getattr(cfg, "num_experts_per_tok", 0)),
        kv_lora_rank=int(getattr(cfg, "kv_lora_rank", 0)),
        q_lora_rank=int(getattr(cfg, "q_lora_rank", 0)),
        qk_nope_head_dim=int(getattr(cfg, "qk_nope_head_dim", 0)),
        qk_rope_head_dim=int(getattr(cfg, "qk_rope_head_dim", 0)),
        v_head_dim=int(getattr(cfg, "v_head_dim", 0)),
        rms_norm_eps=float(getattr(cfg, "rms_norm_eps", 1e-6)),
        first_k_dense_replace=int(getattr(cfg, "first_k_dense_replace", 0)),
        moe_layer_freq=int(getattr(cfg, "moe_layer_freq", 1)),
    )


def save_reference_config_json(config: DeepSeekR1ReferenceConfig, path: str | Path) -> None:
    import json

    with open(path, "w", encoding="utf-8") as f:
        json.dump(config.__dict__, f, indent=2)
