# SPDX-FileCopyrightText: © 2025-2026 The Moonshot AI Team.
# SPDX-License-Identifier: Apache-2.0
#
# Copy from https://huggingface.co/moonshotai/Kimi-K3/blob/main/configuration_kimi_k3.py
#
# TRIMMED: only ``KimiLinearConfig`` (the text-tower config) is vendored. The upstream file also
# defines ``KimiK3VisionConfig`` and the multimodal wrapper ``KimiK3Config``; neither is needed for
# the MLA work, and the wrapper's only role -- descending to ``text_config`` -- is already handled
# by ``unwrap_multimodal_config`` / ``_unwrap_multimodal_config`` in the TT stack.
#
# Kept verbatim otherwise so the K3 field names and defaults are auditable against upstream.
#
# NOTE for readers chasing the softmax-scale bug: ``rope_scaling`` defaults to None here and is
# assigned as-is, BUT ``PretrainedConfig.__init__`` on transformers >= 5 normalizes it into
# ``{"rope_theta": ..., "rope_type": "default"}`` when the checkpoint config omits it. So on a real
# K3 config ``config.rope_scaling`` is a truthy dict WITHOUT a "factor" key -- an
# ``is not None`` guard is not enough, callers must check for the key.

from typing import Optional

from transformers.configuration_utils import PretrainedConfig


class KimiLinearConfig(PretrainedConfig):
    model_type = "kimi_linear"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        model_type="kimi_linear",
        vocab_size=163840,
        hidden_size=4096,
        head_dim=None,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        rope_theta=10000.0,
        rope_scaling=None,
        tie_word_embeddings=False,
        moe_intermediate_size: Optional[int] = None,
        moe_renormalize: bool = True,
        moe_router_activation_func: str = "sigmoid",
        num_experts: Optional[int] = None,
        num_experts_per_token: Optional[int] = None,
        num_shared_experts: int = 0,
        routed_scaling_factor: float = 1.0,
        first_k_dense_replace: int = 0,
        moe_layer_freq: int = 1,
        use_grouped_topk: bool = True,
        num_expert_group: int = 1,
        topk_group: int = 1,
        q_lora_rank: Optional[int] = None,
        kv_lora_rank: Optional[int] = None,
        qk_nope_head_dim: Optional[int] = None,
        qk_rope_head_dim: Optional[int] = None,
        v_head_dim: Optional[int] = None,
        mla_use_nope: Optional[bool] = False,
        mla_use_output_gate: Optional[bool] = False,
        num_nextn_predict_layers: int = 0,
        linear_attn_config: Optional[dict] = None,
        attn_res_block_size: Optional[int] = None,
        latent_moe_use_norm: bool = False,
        activation_situ_beta: Optional[float] = None,
        activation_situ_linear_beta: Optional[float] = None,
        max_position_embeddings: int = 4096,
        routed_expert_hidden_size: Optional[int] = None,
        topk_method: str = "noaux_tc",
        **kwargs,
    ):
        self.model_type = model_type
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.mla_use_nope = mla_use_nope
        self.mla_use_output_gate = mla_use_output_gate
        # moe config
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.moe_renormalize = moe_renormalize
        self.num_shared_experts = num_shared_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.moe_router_activation_func = moe_router_activation_func
        assert self.moe_router_activation_func in ("softmax", "sigmoid")
        self.moe_intermediate_size = moe_intermediate_size
        self.first_k_dense_replace = first_k_dense_replace
        self.moe_layer_freq = moe_layer_freq
        self.use_grouped_topk = use_grouped_topk
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.num_nextn_predict_layers = num_nextn_predict_layers

        self.attn_res_block_size = attn_res_block_size
        self.latent_moe_use_norm = latent_moe_use_norm
        self.activation_situ_beta = activation_situ_beta
        self.activation_situ_linear_beta = activation_situ_linear_beta
        self.max_position_embeddings = max_position_embeddings
        self.routed_expert_hidden_size = routed_expert_hidden_size
        self.topk_method = topk_method

        if linear_attn_config is not None:
            assert linear_attn_config["kda_layers"] is not None
            assert linear_attn_config["full_attn_layers"] is not None
        self.linear_attn_config = linear_attn_config

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def is_mla(self):
        return (
            self.q_lora_rank is not None
            or self.kv_lora_rank is not None
            or self.qk_nope_head_dim is not None
            or self.qk_rope_head_dim is not None
            or self.v_head_dim is not None
            or self.mla_use_nope is True
        )

    @property
    def is_moe(self):
        return self.num_experts is not None

    @property
    def is_linear_attn(self) -> bool:
        return not (
            self.linear_attn_config is None
            or (
                isinstance(self.linear_attn_config, dict)
                and self.linear_attn_config["kda_layers"] is not None
                and len(self.linear_attn_config["kda_layers"]) == 0
            )
        )

    def is_kda_layer(self, layer_idx: int):
        """NOTE: ``kda_layers`` / ``full_attn_layers`` are **1-indexed** -- hence the ``+ 1``."""
        return self.linear_attn_config is not None and (layer_idx + 1) in self.linear_attn_config["kda_layers"]
