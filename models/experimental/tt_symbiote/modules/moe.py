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
from models.experimental.tt_symbiote.core.module import TTNNModule, run_on_devices, DeviceArch
from models.experimental.tt_symbiote.modules.linear import (
    TTNNLinearSilu,
    TTNNLinearLLamaIColShardedWRowSharded,
    TTNNLinearIColShardedWRowSharded,
)
from models.experimental.tt_symbiote.core.run_config import disable_trace

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


class TTNNGlm4MoeExpertLayers(TTNNModule):
    """TTNN module that handles expert layer execution."""

    def __init__(self, num_experts: int, hidden_dim: int, intermediate_dim: int, num_experts_off_chip: int = 20):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_experts_off_chip = num_experts_off_chip
        self.gate_layers = {}
        self.up_layers = {}
        self.down_layers = {}

    @classmethod
    def from_parameters(cls, gate_up_proj: torch.Tensor, down_proj: torch.Tensor, num_experts_off_chip: int = 20):
        """Create from expert weight tensors."""
        num_experts, _, hidden_dim = gate_up_proj.shape
        intermediate_dim = down_proj.shape[-1]

        module = cls(num_experts, hidden_dim, intermediate_dim, num_experts_off_chip)

        for i in range(num_experts):
            linear_class = (
                TTNNLinearLLamaIColShardedWRowSharded if i < num_experts_off_chip else TTNNLinearIColShardedWRowSharded
            )

            module.gate_layers[i] = TTNNLinearSilu.from_parameters(
                gate_up_proj[i, :intermediate_dim, :], linear_class=linear_class
            )

            up_linear_class = (
                TTNNLinearLLamaIColShardedWRowSharded if i < num_experts_off_chip else TTNNLinearIColShardedWRowSharded
            )
            module.up_layers[i] = up_linear_class.from_parameters(gate_up_proj[i, intermediate_dim:, :])

            down_linear_class = (
                TTNNLinearLLamaIColShardedWRowSharded if i < num_experts_off_chip else TTNNLinearIColShardedWRowSharded
            )
            module.down_layers[i] = down_linear_class.from_parameters(down_proj[i, :, :])

        return module

    @disable_trace
    def forward(self, current_state: torch.Tensor, expert_idx: int) -> torch.Tensor:
        """Execute single expert forward pass."""
        gate = self.gate_layers[expert_idx](current_state)
        up = self.up_layers[expert_idx](current_state)
        current_hidden_states = gate.to_ttnn * up.to_ttnn
        current_hidden_states = self.down_layers[expert_idx](current_hidden_states)
        return current_hidden_states


class Glm4MoeExpertLayersTorch(nn.Module):
    """Collection of expert layers stored as separate linear modules."""

    def __init__(self, gate_up_proj: torch.Tensor, down_proj: torch.Tensor):
        super().__init__()
        self.gate_up_proj = nn.Parameter(gate_up_proj)
        self.down_proj = nn.Parameter(down_proj)
        self.act_fn = nn.SiLU()

    def forward(self, current_state: torch.Tensor, expert_idx: int) -> torch.Tensor:
        """Execute single expert forward pass."""
        gate, up = nn.functional.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
        current_hidden_states = self.act_fn(gate) * up
        current_hidden_states = nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
        return current_hidden_states


class Glm4MoeNaiveMoeHybrid(nn.Module):
    """Collection of expert weights with hybrid TTNN execution."""

    def __init__(self, old_layer, num_experts_off_chip: int = 20):
        super().__init__()
        self.num_experts = old_layer.num_experts
        self.hidden_dim = old_layer.hidden_dim
        self.intermediate_dim = old_layer.intermediate_dim

        # Create TTNN expert layers module
        ttnn = False
        if ttnn:
            self.expert_layers = TTNNGlm4MoeExpertLayers.from_parameters(
                old_layer.gate_up_proj, old_layer.down_proj, num_experts_off_chip=num_experts_off_chip
            )

            # Clean up old layer weights
            del old_layer.gate_up_proj
            del old_layer.down_proj
        else:
            self.expert_layers = Glm4MoeExpertLayersTorch(old_layer.gate_up_proj, old_layer.down_proj)

        assert old_layer.config.hidden_act == "silu", "Only SiLU activation is supported in naive MoE."

    @classmethod
    def from_torch(cls, moe_module: Glm4MoeNaiveMoe, num_experts_off_chip: int = 20) -> "Glm4MoeNaiveMoeHybrid":
        """Create Glm4MoeNaiveMoeHybrid from PyTorch Glm4MoeNaiveMoe layer."""
        return cls(moe_module, num_experts_off_chip=num_experts_off_chip)

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

            # Use TTNN expert layers
            current_hidden_states = self.expert_layers(current_state, expert_idx.item())

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


class TTNNGlm4MoeTopkRouter(TTNNLinearIColShardedWRowSharded):
    def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        tt_output = super().forward(input_tensor)
        tt_output = ttnn.reshape(tt_output, [-1] + [tt_output.shape[-1]])
        return tt_output


class TTNNGlm4MoeMLP(TTNNModule):
    @classmethod
    def from_torch(cls, torch_layer: Glm4MoeMLP):
        """Create a TTNNGlm4MoeMLP from a PyTorch Glm4MoeMLP layer."""
        tt_module = cls()
        tt_module._fallback_torch_layer = torch_layer
        tt_module.gate_proj = TTNNLinearSilu.from_torch(
            torch_layer.gate_proj, linear_class=TTNNLinearIColShardedWRowSharded
        )
        tt_module.up_proj = TTNNLinearIColShardedWRowSharded.from_torch(torch_layer.up_proj)
        tt_module.down_proj = TTNNLinearIColShardedWRowSharded.from_torch(torch_layer.down_proj)
        return tt_module

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x_gate = self.gate_proj(x)
        x_up = self.up_proj(x)
        x = ttnn.mul(
            x_gate.to_ttnn,
            x_up.to_ttnn,
        )
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
        ttnn_module.experts = Glm4MoeNaiveMoeHybrid.from_torch(torch_module.experts, num_experts_off_chip=32)
        ttnn_module.gate = TTNNGlm4MoeTopkRouter.from_parameters(
            torch_module.gate.weight, torch_module.gate.e_score_correction_bias
        )
        ttnn_module.gate._fallback_torch_layer = torch_module.gate
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

    @run_on_devices(DeviceArch.T3K)
    def forward(self, hidden_states):
        hidden_states = TorchTTNNTensor(hidden_states)
        residuals = hidden_states
        orig_shape = list(hidden_states.shape)
        router_logits = self.gate(hidden_states)
        topk_indices, topk_weights = self.torch_layer.route_tokens_to_experts(router_logits)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = self.experts(hidden_states, topk_indices.to(dtype=torch.int64), topk_weights).view(*orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states.to_ttnn
