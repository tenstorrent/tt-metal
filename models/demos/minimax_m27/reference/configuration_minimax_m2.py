# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

from transformers.configuration_utils import PretrainedConfig


class MiniMaxM2Config(PretrainedConfig):
    """Lightweight MiniMax M2.7 configuration for local reference modeling."""

    model_type = "minimax_m2"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 200064,
        hidden_size: int = 3072,
        intermediate_size: int = 1536,
        num_hidden_layers: int = 62,
        num_attention_heads: int = 48,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 204800,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 5_000_000,
        attention_dropout: float = 0.0,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        num_experts_per_tok: int = 8,
        num_local_experts: int = 256,
        scoring_func: str = "sigmoid",
        use_qk_norm: bool = True,
        use_routing_bias: bool = True,
        qk_norm_type: str = "per_layer",
        rotary_dim: int = 64,
        partial_rotary_factor: float = 1.0,
        output_router_logits: bool = False,
        router_aux_loss_coef: float = 0.001,
        router_jitter_noise: float = 0.0,
        pad_token_id: int | None = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings

        self.num_experts_per_tok = num_experts_per_tok
        self.num_local_experts = num_local_experts
        self.scoring_func = scoring_func
        self.use_qk_norm = use_qk_norm
        self.use_routing_bias = use_routing_bias
        self.qk_norm_type = qk_norm_type

        self.rotary_dim = rotary_dim
        self.partial_rotary_factor = partial_rotary_factor
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.router_jitter_noise = router_jitter_noise

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ["MiniMaxM2Config"]
