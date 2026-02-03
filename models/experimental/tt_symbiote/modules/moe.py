# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Mixture of Experts implementations for TTNN."""


import torch
from torch import nn
import ttnn
from transformers.configuration_utils import PretrainedConfig
from torch.nn import functional as F
from models.experimental.tt_symbiote.core.module import TTNNModule
from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor
from ttnn.model_preprocessing import preprocess_linear_weight
from models.experimental.tt_symbiote.core.module import TTNNModule, deallocate_weights_after
from models.experimental.tt_symbiote.modules.linear import TTNNLinear, TTNNLinearSilu, TTNNLinearLLama


TOPK_MIN_WIDTH = 64  # Minimum width of the topk input tensor
SPARSITY_BLOCK_SIZE = 32


# Helper math functions
def even_int_div(a: int, b: int) -> int:
    """Integer division that raises an error if b does not divide a without a remainder."""
    assert a % b == 0
    return a // b


class Glm4MoeConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Glm4MoeModel`]. It is used to instantiate a
    Glm4Moe model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of [THUDM/GLM-4-100B-A10B](https://huggingface.co/THUDM/GLM-4-100B-A10B).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 151552):
            Vocabulary size of the Glm4Moe model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Glm4MoeModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 10944):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 46):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 96):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details, check out [this
            paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to `32`.

        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 131072):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        rope_parameters (`RopeParameters`, *optional*):
            Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain
            a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE
            with longer `max_position_embeddings`.
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        moe_intermediate_size (`int`, *optional*, defaults to 1408):
            Intermediate size of the routed expert.
        num_experts_per_tok (`int`, *optional*, defaults to 8):
            number of experts per token.
        n_shared_experts (`int`, *optional*, defaults to 1):
            Number of shared experts.
        n_routed_experts (`int`, *optional*, defaults to 128):
            Number of routed experts.
        routed_scaling_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor or routed experts.
        n_group (`int`, *optional*, defaults to 1):
            Number of groups for routed experts.
        topk_group (`int`, *optional*, defaults to 1):
            Number of selected groups for each token(for each token, ensuring the selected experts is only within `topk_group` groups).
        first_k_dense_replace (`int`, *optional*, defaults to 1):
            Number of dense layers in shallow layers(embed->dense->dense->...->dense->moe->moe...->lm_head).
                                                            \--k dense layers--/
        norm_topk_prob (`bool`, *optional*, defaults to `True`):
            Whether to normalize the topk probabilities.
        use_qk_norm (`bool`, *optional*, defaults to `False`):
            Whether to use query-key normalization in the attention
        bos_token_id (`int`, *optional*):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*):
            End of stream token id.
        pad_token_id (`int`, *optional*):
            Padding token id.

    ```python
    >>> from transformers import Glm4MoeModel, Glm4MoeConfig

    >>> # Initializing a Glm4Moe style configuration
    >>> configuration = Glm4MoeConfig()

    >>> # Initializing a model from the GLM-4-MOE-100B-A10B style configuration
    >>> model = Glm4MoeModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "glm4_moe"
    keys_to_ignore_at_inference = ["past_key_values"]

    # Default tensor parallel plan for base model `Glm4Moe`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.experts.gate_up_proj": "local_rowwise",
        "layers.*.mlp.experts.down_proj": "local_rowwise",
        "layers.*.mlp.experts": "gather",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }
    attribute_map = {
        "num_local_experts": "n_routed_experts",
    }

    def __init__(
        self,
        vocab_size: int | None = 151552,
        hidden_size: int | None = 4096,
        intermediate_size: int | None = 10944,
        num_hidden_layers: int | None = 46,
        num_attention_heads: int | None = 96,
        num_key_value_heads: int | None = 8,
        hidden_act: str | None = "silu",
        max_position_embeddings: int | None = 131072,
        initializer_range: float | None = 0.02,
        rms_norm_eps: int | None = 1e-5,
        use_cache: bool | None = True,
        tie_word_embeddings: bool | None = False,
        rope_parameters=None,
        attention_bias: bool | None = False,
        attention_dropout: float | None = 0.0,
        moe_intermediate_size: int | None = 1408,
        num_experts_per_tok: int | None = 8,
        n_shared_experts: int | None = 1,
        n_routed_experts: int | None = 128,
        routed_scaling_factor: float | None = 1.0,
        n_group: int | None = 1,
        topk_group: int | None = 1,
        first_k_dense_replace: int | None = 1,
        norm_topk_prob: bool | None = True,
        use_qk_norm: bool | None = False,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        pad_token_id: int | None = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        kwargs.setdefault("partial_rotary_factor", 0.5)  # assign default for BC

        # MoE arguments
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.n_group = n_group
        self.topk_group = topk_group
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.first_k_dense_replace = first_k_dense_replace
        self.norm_topk_prob = norm_topk_prob
        self.use_qk_norm = use_qk_norm
        self.tie_word_embeddings = tie_word_embeddings
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

        super().__init__(**kwargs)


class Glm4MoeRouteTokenToExperts(nn.Module):
    def __init__(
        self,
        e_score_correction_bias,
        n_routed_experts,
        n_group,
        topk_group,
        top_k,
        norm_topk_prob,
        routed_scaling_factor,
    ):
        super().__init__()
        self.e_score_correction_bias = e_score_correction_bias
        self.n_routed_experts = n_routed_experts
        self.n_group = n_group
        self.topk_group = topk_group
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor

    def forward(self, router_logits):
        router_logits = router_logits.sigmoid()
        router_logits_for_choice = router_logits + self.e_score_correction_bias
        group_scores = (
            router_logits_for_choice.view(-1, self.n_group, self.n_routed_experts // self.n_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )
        scores_for_choice = router_logits_for_choice.masked_fill(~score_mask.bool(), 0.0)
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        topk_weights = router_logits.gather(1, topk_indices)
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights /= denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights


class Glm4MoeNaiveMoe(nn.Module):
    """Collection of expert weights stored as 3D tensors."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.act_fn = nn.SiLU()
        torch.nn.init.normal_(self.gate_up_proj, mean=0.0, std=config.initializer_range)
        torch.nn.init.normal_(self.down_proj, mean=0.0, std=config.initializer_range)

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = nn.functional.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states


class Glm4MoeTopkRouter(nn.Module):
    def __init__(self, config: Glm4MoeConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob

        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, config.hidden_size)))
        self.register_buffer("e_score_correction_bias", torch.zeros((self.n_routed_experts), dtype=torch.float32))
        torch.nn.init.normal_(self.weight, mean=0.0, std=self.config.initializer_range)
        torch.nn.init.zeros_(self.e_score_correction_bias)

    def forward(self, hidden_states):
        hidden_states = hidden_states.view(-1, self.config.hidden_size)
        router_logits = F.linear(hidden_states, self.weight)
        return router_logits


class Glm4MoeMLP(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class Glm4MoeRouteTokenToExperts(nn.Module):
    def __init__(
        self,
        e_score_correction_bias,
        n_routed_experts,
        n_group,
        topk_group,
        top_k,
        norm_topk_prob,
        routed_scaling_factor,
    ):
        super().__init__()
        self.e_score_correction_bias = e_score_correction_bias
        self.n_routed_experts = n_routed_experts
        self.n_group = n_group
        self.topk_group = topk_group
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor

    def forward(self, router_logits):
        router_logits = router_logits.sigmoid()
        router_logits_for_choice = router_logits + self.e_score_correction_bias
        group_scores = (
            router_logits_for_choice.view(-1, self.n_group, self.n_routed_experts // self.n_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )
        scores_for_choice = router_logits_for_choice.masked_fill(~score_mask.bool(), 0.0)
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        topk_weights = router_logits.gather(1, topk_indices)
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights /= denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights


class Glm4MoeMoE(torch.nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.experts = Glm4MoeNaiveMoe(config)
        self.gate = Glm4MoeTopkRouter(config)
        self.shared_experts = Glm4MoeMLP(
            config=config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts
        )
        self.n_routed_experts = config.n_routed_experts
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.top_k = config.num_experts_per_tok
        self.route_tokens_to_experts = Glm4MoeRouteTokenToExperts(
            self.gate.e_score_correction_bias,
            self.n_routed_experts,
            self.n_group,
            self.topk_group,
            self.top_k,
            self.norm_topk_prob,
            self.routed_scaling_factor,
        )

    def forward(self, hidden_states):
        residuals = hidden_states
        orig_shape = hidden_states.shape
        router_logits = self.gate(hidden_states)
        topk_indices, topk_weights = self.route_tokens_to_experts(router_logits)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = self.experts(hidden_states, topk_indices, topk_weights).view(*orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states


class Glm4MoeNaiveMoeHybrid(nn.Module):
    """Collection of expert weights stored as 3D tensors."""

    def __init__(self, old_layer):
        super().__init__()
        self.num_experts = old_layer.num_experts
        self.hidden_dim = old_layer.hidden_dim
        self.intermediate_dim = old_layer.intermediate_dim
        self.gate_layers = {
            i: TTNNLinearSilu.from_parameters(
                old_layer.gate_up_proj[i, : self.intermediate_dim, :], linear_class=TTNNLinearLLama
            )
            for i in range(self.num_experts)
        }
        self.up_layers = {
            i: TTNNLinearLLama.from_parameters(old_layer.gate_up_proj[i, self.intermediate_dim :, :])
            for i in range(self.num_experts)
        }
        del old_layer.gate_up_proj
        self.down_layers = {
            i: TTNNLinearLLama.from_parameters(old_layer.down_proj[i, :, :]) for i in range(self.num_experts)
        }
        del old_layer.down_proj
        assert old_layer.config.hidden_act == "silu", "Only SiLU activation is supported in naive MoE."

    @classmethod
    def from_torch(cls, moe_module: Glm4MoeNaiveMoe) -> "TTNNGlm4MoeNaiveMoe":
        """Create TTNNGlm4MoeNaiveMoe from PyTorch Glm4MoeNaiveMoe layer."""
        new_moe = cls(moe_module)
        return new_moe

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            int_expert = expert_idx.item()
            gate = self.gate_layers[int_expert](current_state)
            up = self.up_layers[int_expert](current_state)
            current_hidden_states = gate * up
            current_hidden_states = self.down_layers[int_expert](current_hidden_states)
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states


class TTNNGlm4MoeNaiveMoe(TTNNModule):
    def preprocess_weights_impl(self):
        self.tt_gate_up_proj = preprocess_linear_weight(
            self.torch_layer.gate_up_proj, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        self.tt_down_proj = preprocess_linear_weight(
            self.torch_layer.down_proj, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )

    def move_weights_to_device_impl(self):
        self.tt_gate_up_proj = ttnn.to_device(self.tt_gate_up_proj, self.device)
        self.tt_down_proj = ttnn.to_device(self.tt_down_proj, self.device)
        self.num_experts_per_device = even_int_div(self.torch_layer.num_experts, self.device.get_num_devices())
        self.expert_mapping_tensors = ttnn.from_torch(
            torch.eye(self.device.get_num_devices(), dtype=torch.int32)
            .repeat_interleave(self.num_experts_per_device, dim=0)
            .unsqueeze(0)
            .unsqueeze(0),
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
            dtype=ttnn.uint16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        self.remap_topk_mask = ttnn.from_torch(
            torch.ones((1, self.device.shape[0], 1, self.torch_layer.num_experts), dtype=torch.bfloat16),
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

    def forward(self, x, topk_experts_indices, topk_experts_weights):
        return self.torch_layer(
            TorchTTNNTensor(x),
            TorchTTNNTensor(topk_experts_indices, dtype=torch.int64),
            TorchTTNNTensor(topk_experts_weights),
        ).to_ttnn

    def forward_mesh(self, x, topk_experts_indices, topk_experts_weights):
        ccl = self.device_state.ccl_manager
        hidden_size = self.torch_layer.hidden_dim
        moe_intermediate_size = self.torch_layer.intermediate_dim
        assert len(x.shape) == 2  # Shape: [num_tokens, hidden_size]
        x = ttnn.reshape(x, shape=(1, 1, x.shape[-2], x.shape[-1]))
        x = ttnn.experimental.all_gather_async(
            x,
            **ccl.populate_all_gather_runtime_args(
                {
                    "mesh_device": self.device,
                    "dim": -1,  # Last dimension
                    "memory_config": ttnn.DRAM_MEMORY_CONFIG,
                    "cluster_axis": 1,
                    "topology": ttnn.Topology.Linear,
                }
            ),
        )

        seq_len = 1  # a2a dispatch and combine require DP=num_dispatch_devices, hence in prefill for bs=1, we interchange the seq_len with batch_size dimensions
        batch_size_per_device = x.shape[
            -2
        ]  # Input is expected to be DP. In prefill, this is equivalent to seq_len_per_device
        batch_size = batch_size_per_device * self.device.shape[0]  # Global batch size

        x_rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x_rm = ttnn.reshape(x_rm, shape=(batch_size_per_device, 1, seq_len, hidden_size))
        topk_experts_indices = ttnn.to_layout(topk_experts_indices, ttnn.TILE_LAYOUT)
        topk_experts_indices = ttnn.typecast(topk_experts_indices, ttnn.uint16)
        topk_experts_indices_rm = ttnn.to_layout(topk_experts_indices, ttnn.ROW_MAJOR_LAYOUT)
        topk_experts_indices_rm = ttnn.reshape(
            topk_experts_indices_rm,
            shape=(batch_size_per_device, 1, seq_len, self.torch_layer.config.num_experts_per_tok),
        )
        all_to_all_dispatch_output_tensors, all_to_all_dispatch_metadata_tensors = ttnn.all_to_all_dispatch(
            x_rm, topk_experts_indices_rm, self.expert_mapping_tensors, num_links=1
        )
        post_all_to_all_dispatch_output = ttnn.reshape(
            all_to_all_dispatch_output_tensors, shape=(1, 1, batch_size * seq_len, hidden_size)
        )
        post_all_to_all_dispatch_output = ttnn.to_layout(post_all_to_all_dispatch_output, ttnn.TILE_LAYOUT)
        # repeat remap_topk_mask for the num_tokens known at runtime
        remap_topk_mask = ttnn.repeat(self.remap_topk_mask, ttnn.Shape((1, batch_size_per_device, 1, 1)))
        _, sparsity_t = ttnn.moe_expert_token_remap(
            remap_topk_mask,
            self.expert_mapping_tensors,
            all_to_all_dispatch_metadata_tensors,
            reduction_size=SPARSITY_BLOCK_SIZE,
        )

        ### START Expert computation
        _, _, num_tokens, hidden_size = post_all_to_all_dispatch_output.shape
        num_sparse_blocks = num_tokens // SPARSITY_BLOCK_SIZE
        post_all_to_all_dispatch_output = ttnn.reshape(
            post_all_to_all_dispatch_output, shape=(1, num_sparse_blocks, SPARSITY_BLOCK_SIZE, hidden_size)
        )

        # Gate and up projections
        w1_out = ttnn.sparse_matmul(
            post_all_to_all_dispatch_output,
            sparsity=sparsity_t,
            is_input_a_sparse=False,
            is_input_b_sparse=True,
            # output_tile=ttnn.Tile([SPARSITY_BLOCK_SIZE, ttnn.TILE_SIZE]),
        )
        w3_out = ttnn.sparse_matmul(
            post_all_to_all_dispatch_output,
            sparsity=sparsity_t,
            is_input_a_sparse=False,
            is_input_b_sparse=True,
        )

        # Apply activation and multiply
        activated = ttnn.mul(
            w1_out,
            w3_out,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
        )
        ttnn.deallocate(w1_out)
        ttnn.deallocate(w3_out)

        # Reshape for down projection
        # activated.shape = Shape([1, 4, 1, 8, 32, 2048])
        activated = ttnn.squeeze(activated, 0)
        activated = ttnn.squeeze(activated, 1)

        # Down projection
        output = ttnn.sparse_matmul(
            activated,
            sparsity=sparsity_t,
            is_input_a_sparse=True,
            is_input_b_sparse=False,
        )
        ttnn.deallocate(activated)

        # Reshape for output
        output = ttnn.permute(output, (1, 0, 2, 3))
        experts_output = ttnn.reshape(output, shape=(1, self.num_experts_per_device, num_tokens, hidden_size))

        ### END Expert computation

        ttnn.deallocate(post_all_to_all_dispatch_output)
        experts_output = ttnn.to_layout(experts_output, ttnn.ROW_MAJOR_LAYOUT)
        experts_output = ttnn.reshape(
            experts_output, shape=(self.num_experts_per_device, batch_size, seq_len, hidden_size)
        )
        all_to_all_combine_output_tensors = ttnn.all_to_all_combine(
            experts_output,
            all_to_all_dispatch_metadata_tensors,
            self.expert_mapping_tensors,
            num_links=1,
        )
        post_combine_output_tensor = ttnn.reshape(
            all_to_all_combine_output_tensors,
            shape=(self.torch_layer.config.num_experts_per_tok, 1, batch_size_per_device * seq_len, hidden_size),
        )
        post_combine_output_tensor = ttnn.to_layout(post_combine_output_tensor, ttnn.TILE_LAYOUT)
        topk_experts_weights_rm = ttnn.to_layout(topk_experts_weights, ttnn.ROW_MAJOR_LAYOUT)
        topk_experts_weights_rm = ttnn.repeat(topk_experts_weights_rm, ttnn.Shape((hidden_size, 1, 1, 1)))
        topk_experts_weights_rm = ttnn.permute(topk_experts_weights_rm, (3, 1, 2, 0))
        topk_experts_weights = ttnn.to_layout(topk_experts_weights_rm, ttnn.TILE_LAYOUT)
        ttnn.deallocate(topk_experts_weights_rm)
        post_combine_output_tensor = ttnn.mul(
            post_combine_output_tensor,
            topk_experts_weights,
        )
        post_combine_output_tensor = ttnn.sum(post_combine_output_tensor, dim=0, keepdim=True)
        post_combine_output_tensor = ttnn.experimental.reduce_scatter_minimal_async(
            post_combine_output_tensor,
            **ccl.populate_reduce_scatter_runtime_args(
                {
                    "dim": 3,  # Last dimension
                    "memory_config": ttnn.DRAM_MEMORY_CONFIG,
                    "cluster_axis": 1,
                    "topology": ttnn.Topology.Linear,
                }
            ),
        )
        return post_combine_output_tensor


class TTNNGlm4MoeTopkRouter(TTNNModule):
    def preprocess_weights_impl(self):
        """Preprocess linear weights for TTNN."""
        self.tt_weight_host = preprocess_linear_weight(
            self.torch_layer.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        self.e_score_correction_bias = self.torch_layer.e_score_correction_bias

    def move_weights_to_device_impl(self):
        """Move weights to TTNN device."""
        self.tt_weight = ttnn.to_device(self.tt_weight_host, self.device)

    def deallocate_weights_impl(self):
        """Deallocate weights from device."""
        ttnn.deallocate(self.tt_weight)
        super().deallocate_weights_impl()

    @deallocate_weights_after
    def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        """Forward pass through linear layer."""
        if input_tensor.layout != ttnn.TILE_LAYOUT:
            input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        input_tensor_shape = list(input_tensor.shape)
        input_shape = list(input_tensor_shape)
        while len(input_shape) < 4:
            input_shape.insert(0, 1)  # Add batch dimensions if needed
        input_tensor = ttnn.reshape(input_tensor, input_shape)
        tt_output = ttnn.linear(input_tensor, self.tt_weight, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        tt_output = ttnn.reshape(tt_output, [-1] + [tt_output.shape[-1]])
        return tt_output


class TTNNGlm4MoeMLP(TTNNModule):
    @classmethod
    def from_torch(cls, torch_layer: Glm4MoeMLP):
        """Create a TTNNGlm4MoeMLP from a PyTorch Glm4MoeMLP layer."""
        tt_module = cls()
        tt_module._fallback_torch_layer = torch_layer
        tt_module.gate_proj = TTNNLinearSilu.from_torch(torch_layer.gate_proj)
        tt_module.up_proj = TTNNLinear.from_torch(torch_layer.up_proj)
        tt_module.down_proj = TTNNLinear.from_torch(torch_layer.down_proj)
        return tt_module

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x_gate = self.gate_proj(x)
        x_up = self.up_proj(x)
        x = ttnn.mul(
            x_gate.to_ttnn,
            x_up.to_ttnn,
        )
        ttnn.deallocate(x_gate.to_ttnn)
        ttnn.deallocate(x_up.to_ttnn)
        x = self.down_proj(x)
        return x


class TTNNGlm4MoeRouteTokenToExperts(TTNNModule):
    def preprocess_weights_impl(self):
        self.e_score_correction_bias = ttnn.from_torch(
            self.torch_layer.e_score_correction_bias.unsqueeze(0).unsqueeze(0).unsqueeze(0).to(torch.bfloat16)
        )
        self.e_score_correction_bias = ttnn.to_layout(
            self.e_score_correction_bias,
            ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # scatter input mask: (1,1,1,n_group) filled with -inf
        self.scatter_input = ttnn.from_torch(
            torch.full((1, 1, 1, self.torch_layer.n_group), -float("inf")).to(torch.bfloat16)
        )

        # scatter src: (1,1,1,topk_group) filled with ones
        self.scatter_src = ttnn.from_torch(torch.ones((1, 1, 1, self.torch_layer.topk_group), dtype=torch.bfloat16))

        # expert scale: (1,1,1,top_k)
        self.expert_scale = ttnn.from_torch(
            torch.tensor([self.torch_layer.routed_scaling_factor])
            .to(torch.bfloat16)
            .repeat(1, self.torch_layer.top_k)
            .unsqueeze(0)
            .unsqueeze(0)
        )

    def move_weights_to_device_impl(self):
        self.e_score_correction_bias = ttnn.to_device(self.e_score_correction_bias, self.device)
        self.scatter_input = ttnn.to_device(self.scatter_input, self.device)
        self.scatter_src = ttnn.to_device(self.scatter_src, self.device)
        self.expert_scale = ttnn.to_device(self.expert_scale, self.device)

    def forward(self, router_logits: ttnn.Tensor):
        if router_logits.layout != ttnn.TILE_LAYOUT:
            router_logits = ttnn.to_layout(
                router_logits,
                ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        router_logits = ttnn.reshape(router_logits, ttnn.Shape((1, 1, router_logits.shape[0], router_logits.shape[1])))
        scores = ttnn.sigmoid(router_logits)
        ttnn.deallocate(router_logits)

        T = scores.shape[2]
        n_experts = scores.shape[3]
        experts_per_group = n_experts // self.torch_layer.n_group

        # ------------------------------------------------------------
        # Add correction bias
        # ------------------------------------------------------------
        bias = ttnn.repeat(self.e_score_correction_bias, ttnn.Shape((1, 1, T, 1)))
        bias = ttnn.to_layout(bias, ttnn.TILE_LAYOUT)

        scores_with_bias = ttnn.add(scores, bias)
        ttnn.deallocate(bias)

        # ------------------------------------------------------------
        # Reshape into groups: (1,T,n_group,experts_per_group)
        # ------------------------------------------------------------
        grouped = ttnn.reshape(scores_with_bias, ttnn.Shape((1, T, self.torch_layer.n_group, experts_per_group)))

        # ------------------------------------------------------------
        # Top-2 experts per group
        # ------------------------------------------------------------
        top2_scores, _ = ttnn.topk(grouped, k=2, dim=3)
        ttnn.deallocate(grouped)

        # group_scores: (1,T,n_group)
        group_scores = ttnn.sum(top2_scores, dim=3)
        ttnn.deallocate(top2_scores)

        # ------------------------------------------------------------
        # Top-k groups
        # ------------------------------------------------------------
        _, topk_group_idx = ttnn.topk(group_scores, k=self.torch_layer.topk_group, dim=2)
        ttnn.deallocate(group_scores)

        # ------------------------------------------------------------
        # Build group mask via scatter
        # ------------------------------------------------------------
        input_mask = ttnn.repeat(self.scatter_input, ttnn.Shape((1, 1, T, 1)))

        src_tensor = ttnn.repeat(self.scatter_src, ttnn.Shape((1, 1, T, 1)))
        topk_group_idx = ttnn.unsqueeze(topk_group_idx, dim=1)
        active_groups_mask = ttnn.scatter(input=input_mask, index=topk_group_idx, src=src_tensor, dim=3)
        ttnn.deallocate(input_mask)
        ttnn.deallocate(src_tensor)
        ttnn.deallocate(topk_group_idx)

        # reshape: (1,T,n_group,1)
        active_groups_mask = ttnn.reshape(active_groups_mask, ttnn.Shape((1, T, self.torch_layer.n_group, 1)))

        # ------------------------------------------------------------
        # Expand group mask → expert mask
        # ------------------------------------------------------------
        active_experts_mask = ttnn.repeat(active_groups_mask, ttnn.Shape((1, 1, 1, experts_per_group)))
        ttnn.deallocate(active_groups_mask)

        # reshape back: (1,1,T,16)
        active_experts_mask = ttnn.reshape(active_experts_mask, ttnn.Shape((1, 1, T, n_experts)))

        # ------------------------------------------------------------
        # Zero out inactive experts
        # ------------------------------------------------------------
        masked_scores = ttnn.mul(scores_with_bias, active_experts_mask)
        ttnn.deallocate(active_experts_mask)

        # ------------------------------------------------------------
        # Top-k experts from active experts
        # ------------------------------------------------------------
        _, topk_expert_idx = ttnn.topk(masked_scores, k=self.torch_layer.top_k, dim=3)
        ttnn.deallocate(masked_scores)

        # ------------------------------------------------------------
        # Gather original sigmoid scores (NO bias)
        # ------------------------------------------------------------
        topk_weights = ttnn.gather(scores, dim=3, index=topk_expert_idx)
        ttnn.deallocate(scores)

        # ------------------------------------------------------------
        # Normalize weights
        # ------------------------------------------------------------
        denom = ttnn.sum(topk_weights, dim=3, keepdim=True) + 1e-20
        topk_weights = ttnn.div(topk_weights, denom)
        ttnn.deallocate(denom)

        # ------------------------------------------------------------
        # Apply scaling factor
        # ------------------------------------------------------------
        scale = ttnn.repeat(self.expert_scale, ttnn.Shape((1, 1, T, 1)))
        scale = ttnn.to_layout(scale, ttnn.TILE_LAYOUT)

        topk_weights = ttnn.mul(topk_weights, scale)
        ttnn.deallocate(scale)
        T = topk_weights.shape[2]

        topk_expert_idx = ttnn.reshape(topk_expert_idx, ttnn.Shape((T, self.torch_layer.top_k)))

        topk_weights = ttnn.reshape(topk_weights, ttnn.Shape((T, self.torch_layer.top_k)))
        return topk_expert_idx, topk_weights


class TTNNGlm4MoeMoE(TTNNModule):
    @classmethod
    def from_torch(cls, torch_module: Glm4MoeMoE) -> "TTNNGlm4MoeMoE":
        ttnn_module = cls()
        ttnn_module._fallback_torch_layer = torch_module
        ttnn_module.experts = Glm4MoeNaiveMoeHybrid.from_torch(torch_module.experts)
        ttnn_module.gate = TTNNGlm4MoeTopkRouter.from_torch(torch_module.gate)
        ttnn_module.shared_experts = TTNNGlm4MoeMLP.from_torch(torch_module.shared_experts)
        ttnn_module.route_tokens_to_experts = TTNNGlm4MoeRouteTokenToExperts.from_torch(
            Glm4MoeRouteTokenToExperts(
                torch_module.gate.e_score_correction_bias,
                torch_module.n_routed_experts,
                torch_module.n_group,
                torch_module.topk_group,
                torch_module.top_k,
                torch_module.norm_topk_prob,
                torch_module.routed_scaling_factor,
            )
        )
        return ttnn_module

    def forward(self, hidden_states):
        hidden_states = TorchTTNNTensor(hidden_states)
        residuals = hidden_states
        orig_shape = hidden_states.shape
        router_logits = self.gate(hidden_states)
        topk_indices, topk_weights = self.torch_layer.route_tokens_to_experts(router_logits)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        if self.device_state is not None:
            hidden_states.set_distributed_config(self.device_state.tensor_config)
        hidden_states = self.experts(hidden_states, topk_indices.to(dtype=torch.int64), topk_weights).view(*orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states.to_ttnn
