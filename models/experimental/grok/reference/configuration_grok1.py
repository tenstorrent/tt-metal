# SPDX-FileCopyrightText: © 2024 HPC-AI Tech

# SPDX-License-Identifier: Apache-2.0
from transformers.configuration_utils import PretrainedConfig


class Grok1Config(PretrainedConfig):
    model_type = "grok-1"
    keys_to_ignore_at_inference = ["past_key_values"]


def __init__(
    self,
    vocab_size=131072,
    hidden_size=6144,
    intermediate_size=32768,
    num_hidden_layers=64,
    num_attention_heads=48,
    num_key_value_heads=8,
    attn_output_multiplier=0.08838834764831845,
    max_attn_value=30.0,
    max_position_embeddings=8192,
    embedding_multiplier_scale: float = 78.38367176906169,
    output_multiplier_scale: float = 0.5773502691896257,
    rms_norm_eps=1e-5,
    use_cache=True,
    pad_token_id=0,
    bos_token_id=1,
    eos_token_id=2,
    tie_word_embeddings=True,
    num_experts_per_tok=2,
    num_experts=8,
    output_router_logits=False,
    router_aux_loss_coef=0.001,
    **kwargs,
):
    super().__init__(
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        tie_word_embeddings=tie_word_embeddings,
        **kwargs,
    )
    self.vocab_size = vocab_size
    self.attn_output_multiplier = attn_output_multiplier
    self.max_attn_value = max_attn_value
    self.max_position_embeddings = max_position_embeddings
    self.embedding_multiplier_scale = embedding_multiplier_scale
    self.output_multiplier_scale = output_multiplier_scale
    self.hidden_size = hidden_size
    self.intermediate_size = intermediate_size
    self.num_hidden_layers = num_hidden_layers
    self.num_attention_heads = num_attention_heads

    # for backward compatibility
    if num_key_value_heads is None:
        num_key_value_heads = num_attention_heads

    self.num_key_value_heads = num_key_value_heads
    self.rms_norm_eps = rms_norm_eps
    self.use_cache = use_cache

    self.num_experts_per_tok = num_experts_per_tok
    self.num_experts = num_experts
    self.output_router_logits = output_router_logits
    self.router_aux_loss_coef = router_aux_loss_coef
