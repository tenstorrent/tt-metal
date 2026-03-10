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
import math

TOPK_MIN_WIDTH = 64  # Minimum width of the topk input tensor
SPARSITY_BLOCK_SIZE = 32


# Helper math functions
def even_int_div(a: int, b: int) -> int:
    """Integer division that raises an error if b does not divide a without a remainder."""
    assert a % b == 0
    return a // b


def _make_sparse_matmul_program_config(
    device,
    out_features: int,
    in0_block_w: int,
    out_subblock_h: int = 1,
    out_subblock_w: int = 1,
    per_core_M: int = 1,
    override: bool = False,
):
    grid = device.compute_with_storage_grid_size()
    core_x = int(getattr(grid, "x"))
    core_y = int(getattr(grid, "y"))
    n_tiles = (int(out_features) + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE
    # The sparse matmul 1D program assigns 2D blocks across a 2D core grid and requires the
    # number of blocks not to exceed the number of available cores. Use a conservative
    # per_core_N based on ceil-div to keep num_blocks_x small when out_features > num_cores.
    num_cores = max(1, core_x * core_y)
    per_core_N = max(1, int(math.ceil(n_tiles / num_cores)))
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(7, 4) if override else ttnn.CoreCoord(core_x, core_y),
        in0_block_w=int(in0_block_w),
        out_subblock_h=int(out_subblock_h),
        out_subblock_w=int(out_subblock_w),
        out_block_h=1,
        out_block_w=1,
        per_core_M=int(per_core_M),
        per_core_N=int(per_core_N),
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )


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


class TTNNMoERouterDecode(TTNNModule):
    """TTNN-accelerated router for decode mode."""

    @classmethod
    def from_torch(cls, torch_module: Glm4MoeRouteTokenToExperts):
        instance = cls()
        instance._fallback_torch_layer = torch_module
        return instance

    def move_weights_to_device_impl(self):
        self.tt_bias = ttnn.from_torch(
            self._fallback_torch_layer.e_score_correction_bias.reshape(1, 1, 1, -1).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(self.device, -1)
            if self.device.__class__.__name__ == "MeshDevice"
            else None,
        )
        self.tt_bias = ttnn.to_device(self.tt_bias, self.device)
        self.tt_bias = ttnn.to_layout(self.tt_bias, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def forward(self, logits: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """TTNN implementation of expert selection.

        Args:
            logits: [num_tokens, hidden_dim] TILE

        Returns:
            expert_ids: [num_tokens, experts_per_token] TILE uint16
            expert_weights: [num_tokens, experts_per_token] TILE bf16
        """

        # Sigmoid
        probabilities = ttnn.sigmoid(logits)
        probabilities = ttnn.unsqueeze(probabilities, 0)
        probabilities = ttnn.unsqueeze(probabilities, 0)
        # Add bias (broadcast over tokens)
        if probabilities.shape[2] > 1:
            bias_expanded = ttnn.repeat(self.tt_bias, ttnn.Shape((1, 1, probabilities.shape[2], 1)))
        else:
            bias_expanded = self.tt_bias
        adjusted = ttnn.add(probabilities, bias_expanded)

        # Group-based top-k selection
        k = self.torch_layer.top_k

        # Select top-k (simplified - full group logic would be more complex)
        topk_values, topk_indices = ttnn.topk(adjusted, k=k, dim=-1, largest=True, sorted=False)
        ttnn.deallocate(topk_values, force=False)

        # Gather weights from unbiased probabilities
        expert_weights = ttnn.gather(probabilities, dim=3, index=topk_indices)
        ttnn.deallocate(probabilities, force=False)

        # Normalize if configured
        if self.torch_layer.norm_topk_prob:
            denom = ttnn.sum(expert_weights, dim=3, keepdim=True)
            denom = ttnn.add(denom, 1e-20, output_tensor=denom)
            expert_weights = ttnn.div(expert_weights, denom)
            ttnn.deallocate(denom, force=False)

        # Apply scaling
        if self.torch_layer.routed_scaling_factor != 1.0:
            expert_weights = ttnn.mul(expert_weights, self.torch_layer.routed_scaling_factor)

        # Reshape for expert dispatch
        num_tokens = topk_indices.shape[2]
        expert_ids = ttnn.reshape(topk_indices, ttnn.Shape((num_tokens, k)))
        expert_weights = ttnn.reshape(expert_weights, ttnn.Shape((num_tokens, k)))

        return expert_ids, expert_weights


class TTNNExperts(TTNNModule):
    """
    Baseline experts module for DeepSeek V3.

    Executes expert computations with basic matmul (no sparsity optimization in baseline).
    Each expert has gate_proj, up_proj, and down_proj.
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        self.num_experts = config.n_routed_experts
        self.num_experts_per_tok = config.num_experts_per_tok

        # Will be set in preprocess_weights_impl
        # Separate weights for sparse matmul: w1 (gate), w3 (up), w2 (down)
        self.tt_w1_proj = None
        self.tt_w3_proj = None
        self.tt_w2_proj = None
        # Keep fused version for dense fallback
        self.tt_gate_up_proj = None

        # Will be set in move_weights_to_device_impl
        self.expert_mapping_tensors = None
        self.remap_topk_mask = None

        # Control flags
        self.use_sparsity = True

    @staticmethod
    def _get_num_experts_per_device(config: PretrainedConfig, mesh_device: ttnn.Device) -> int:
        """Calculate number of experts per device."""
        num_devices = mesh_device.get_num_devices()
        return even_int_div(config.n_routed_experts, num_devices)

    @classmethod
    def from_torch(cls, torch_experts):
        """
        Create TTNNExperts from PyTorch experts module.

        Args:
            torch_experts: PyTorch DeepseekV3NaiveMoe module
        """
        module = cls(torch_experts.config)
        module._fallback_torch_layer = torch_experts

        # Extract expert weights from module
        # Shape: (num_experts, 2*intermediate_size, hidden_size) for gate_up
        # Shape: (num_experts, hidden_size, intermediate_size) for down

        module.torch_gate_up_proj = torch_experts.gate_up_proj
        # Split into w1 (gate) and w3 (up) for sparse matmul
        module.torch_w1_proj = module.torch_gate_up_proj[:, : torch_experts.config.moe_intermediate_size, :].permute(
            [0, 2, 1]
        )
        module.torch_w3_proj = module.torch_gate_up_proj[:, torch_experts.config.moe_intermediate_size :, :].permute(
            [0, 2, 1]
        )

        module.torch_w2_proj = torch_experts.down_proj.permute([0, 2, 1])

        return module

    def preprocess_weights_impl(self):
        """Preprocess expert weights: convert to bfloat16 and TILE_LAYOUT."""
        # w1 (gate): (num_experts, intermediate_size, hidden_size)
        self.tt_w1_proj = ttnn.from_torch(
            self.torch_w1_proj.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(self.device, dim=0),
        )

        # w3 (up): (num_experts, intermediate_size, hidden_size)
        self.tt_w3_proj = ttnn.from_torch(
            self.torch_w3_proj.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(self.device, dim=0),
        )

        # w2 (down): (num_experts, hidden_size, intermediate_size)
        self.tt_w2_proj = ttnn.from_torch(
            self.torch_w2_proj.to(torch.bfloat16).permute(0, 2, 1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(self.device, dim=0),
        )

        # Also keep fused version for dense fallback
        self.tt_gate_up_proj = ttnn.from_torch(
            self.torch_gate_up_proj.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(self.device, dim=0),
        )

        # Clean up torch weights
        del self.torch_w1_proj
        del self.torch_w3_proj
        del self.torch_w2_proj
        del self.torch_gate_up_proj

    def move_weights_to_device_impl(self):
        """Move preprocessed weights to device and create mapping tensors."""

        self.num_experts_per_device = self._get_num_experts_per_device(self.config, self.device)
        self.num_devices = self.device.get_num_devices()
        self.num_dispatch_devices = self.device.shape[1]

        self.tt_w1_proj = ttnn.to_device(self.tt_w1_proj, self.device)
        self.tt_w3_proj = ttnn.to_device(self.tt_w3_proj, self.device)
        self.tt_w2_proj = ttnn.to_device(self.tt_w2_proj, self.device)
        self.tt_gate_up_proj = ttnn.to_device(self.tt_gate_up_proj, self.device)

        # Create expert mapping tensors for all-to-all ops
        self.expert_mapping_tensors = ttnn.from_torch(
            torch.eye(self.num_devices, dtype=torch.int32)
            .repeat_interleave(self.num_experts_per_device, dim=0)
            .unsqueeze(0)
            .unsqueeze(0),
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
            dtype=ttnn.uint16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        # Create remap topk mask for expert token remap
        self.remap_topk_mask = ttnn.from_torch(
            torch.ones((1, self.num_dispatch_devices, 1, self.num_experts), dtype=torch.bfloat16),
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

    @run_on_devices(DeviceArch.T3K)
    def forward(
        self, x: ttnn.Tensor, topk_experts_indices: ttnn.Tensor, topk_experts_weights: ttnn.Tensor
    ) -> ttnn.Tensor:
        """
        Execute full expert pipeline: dispatch → compute → combine → weight.

        Args:
            x: Input tensor of shape (batch_size_per_device, 1, seq_len, hidden_size)
            topk_experts_indices: Expert indices of shape (batch_size_per_device*seq_len, num_experts_per_tok)
            topk_experts_weights: Expert weights of shape (batch_size_per_device*seq_len, num_experts_per_tok)

        Returns:
            Output tensor of shape (1, 1, batch_size_per_device*seq_len, hidden_size)
        """

        # Extract dimensions
        batch_size_per_device = x.shape[0]
        seq_len = x.shape[2]
        batch_size = batch_size_per_device * self.num_dispatch_devices

        # Store original num_tokens for unpadding later
        original_num_tokens = batch_size_per_device * seq_len

        # Pad to nearest multiple of SPARSITY_BLOCK_SIZE if needed
        num_tokens = original_num_tokens
        pad_amount = 0
        if num_tokens % SPARSITY_BLOCK_SIZE != 0:
            pad_amount = SPARSITY_BLOCK_SIZE - (num_tokens % SPARSITY_BLOCK_SIZE)
            num_tokens += pad_amount

            # Pad x in seq_len dimension
            x = ttnn.pad(x, padding=((0, 0), (0, 0), (0, pad_amount), (0, 0)), value=0.0)

            # Pad topk_experts_indices
            topk_experts_indices = ttnn.pad(
                topk_experts_indices, padding=((0, pad_amount), (0, 0)), value=0  # Pad with expert 0
            )

            # Pad topk_experts_weights
            topk_experts_weights = ttnn.pad(
                topk_experts_weights, padding=((0, pad_amount), (0, 0)), value=0.0  # Pad with zero weight
            )

            # Update seq_len to include padding
            seq_len = num_tokens // batch_size_per_device

        # 1. Prepare tensors for all-to-all dispatch (convert to ROW_MAJOR)
        x_rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)

        x_rm = ttnn.reshape(x_rm, shape=(batch_size_per_device, 1, seq_len, self.hidden_size))

        topk_experts_indices_rm = ttnn.to_layout(topk_experts_indices, ttnn.ROW_MAJOR_LAYOUT)

        topk_experts_indices_rm = ttnn.reshape(
            topk_experts_indices_rm, shape=(batch_size_per_device, 1, seq_len, self.num_experts_per_tok)
        )
        # 2. All-to-all dispatch: distribute tokens to expert devices
        all_to_all_dispatch_output, all_to_all_dispatch_metadata = ttnn.all_to_all_dispatch(
            x_rm,
            topk_experts_indices_rm,
            self.expert_mapping_tensors,
            cluster_axis=1,
        )

        # 3. Reshape for expert computation
        post_dispatch = ttnn.reshape(all_to_all_dispatch_output, shape=(1, 1, batch_size * seq_len, self.hidden_size))
        post_dispatch = ttnn.to_layout(post_dispatch, ttnn.TILE_LAYOUT)

        num_tokens = batch_size * seq_len

        # 4. Generate sparsity tensor if using sparse matmul
        remap_topk_mask_expanded = ttnn.repeat(self.remap_topk_mask, ttnn.Shape((1, batch_size_per_device, 1, 1)))
        _, sparsity_t = ttnn.moe_expert_token_remap(
            remap_topk_mask_expanded,
            self.expert_mapping_tensors,
            all_to_all_dispatch_metadata,
            reduction_size=SPARSITY_BLOCK_SIZE,
        )

        # Sparse path
        num_sparse_blocks = num_tokens // SPARSITY_BLOCK_SIZE
        x_sparse = ttnn.reshape(post_dispatch, shape=(1, num_sparse_blocks, SPARSITY_BLOCK_SIZE, self.hidden_size))

        gate_up_program_config = _make_sparse_matmul_program_config(
            device=self.device,
            out_features=int(self.intermediate_size),
            in0_block_w=1,
            per_core_M=1,
            override=True,
        )
        down_program_config = _make_sparse_matmul_program_config(
            device=self.device,
            out_features=int(self.hidden_size),
            in0_block_w=1,
            per_core_M=1,
            # override=True,
        )

        # w1 and w3 projections
        w1_out = ttnn.sparse_matmul(
            x_sparse,
            self.tt_w1_proj,
            sparsity=sparsity_t,
            output_tile=ttnn.Tile([SPARSITY_BLOCK_SIZE, ttnn.TILE_SIZE]),
            program_config=gate_up_program_config,
            is_input_a_sparse=False,
            is_input_b_sparse=True,
        )
        w3_out = ttnn.sparse_matmul(
            x_sparse,
            self.tt_w3_proj,
            sparsity=sparsity_t,
            output_tile=ttnn.Tile([SPARSITY_BLOCK_SIZE, ttnn.TILE_SIZE]),
            program_config=gate_up_program_config,
            is_input_a_sparse=False,
            is_input_b_sparse=True,
        )

        # Activation and multiply
        w1_activated = ttnn.silu(w1_out)
        intermediate = ttnn.mul(w1_activated, w3_out)
        ttnn.deallocate(w1_out)
        ttnn.deallocate(w3_out)

        # Reshape for w2
        intermediate = ttnn.squeeze(intermediate, 0)
        intermediate = ttnn.squeeze(intermediate, 1)

        # w2 projection
        expert_output = ttnn.sparse_matmul(
            intermediate,
            self.tt_w2_proj,
            sparsity=sparsity_t,
            output_tile=ttnn.Tile([SPARSITY_BLOCK_SIZE, ttnn.TILE_SIZE]),
            program_config=down_program_config,
            is_input_a_sparse=True,
            is_input_b_sparse=False,
        )
        ttnn.deallocate(intermediate)

        # Reshape to expected format
        expert_output = ttnn.permute(expert_output, (1, 0, 2, 3))
        expert_output = ttnn.reshape(
            expert_output, shape=(1, self.num_experts_per_device, num_tokens, self.hidden_size)
        )

        ttnn.deallocate(post_dispatch)

        # 6. Prepare for all-to-all combine (convert to ROW_MAJOR)
        expert_output = ttnn.to_layout(expert_output, ttnn.ROW_MAJOR_LAYOUT)
        expert_output = ttnn.reshape(
            expert_output, shape=(self.num_experts_per_device, batch_size, seq_len, self.hidden_size)
        )

        # 7. All-to-all combine: gather results back
        combined_output = ttnn.all_to_all_combine(
            expert_output,
            all_to_all_dispatch_metadata,
            self.expert_mapping_tensors,
            cluster_axis=1,
        )

        # 8. Reshape combined output
        combined_output = ttnn.reshape(
            combined_output, shape=(self.num_experts_per_tok, 1, batch_size_per_device * seq_len, self.hidden_size)
        )
        print("combined_output.shape : ", combined_output.shape)
        combined_output = ttnn.to_layout(combined_output, ttnn.TILE_LAYOUT)

        # 9. Apply expert weights
        print("topk_experts_weights.shape : ", topk_experts_weights.shape)
        topk_experts_weights_rm = ttnn.to_layout(topk_experts_weights, ttnn.ROW_MAJOR_LAYOUT)
        topk_experts_weights_rm = ttnn.unsqueeze(topk_experts_weights_rm, 0)
        print("topk_experts_weights_rm unsqz 1.shape : ", topk_experts_weights_rm.shape)
        topk_experts_weights_rm = ttnn.unsqueeze(topk_experts_weights_rm, 0)
        print("topk_experts_weights_rm unsqz 2.shape : ", topk_experts_weights_rm.shape)
        # topk_experts_weights_rm = ttnn.repeat(topk_experts_weights_rm, repeat_dims=(self.hidden_size, 1, 1, 1))
        print("topk_experts_weights_rm repeat.shape : ", topk_experts_weights_rm.shape)
        topk_experts_weights_rm = ttnn.reshape(
            topk_experts_weights_rm,
            shape=(
                topk_experts_weights_rm.shape[0],
                topk_experts_weights_rm.shape[1],
                topk_experts_weights_rm.shape[3],
                topk_experts_weights_rm.shape[4],
            ),
        )
        print("topk_experts_weights_rm reshape.shape : ", topk_experts_weights_rm.shape)
        topk_experts_weights_rm = ttnn.permute(topk_experts_weights_rm, (3, 1, 2, 0))
        print("topk_experts_weights_rm permute.shape : ", topk_experts_weights_rm.shape)
        topk_experts_weights_tile = ttnn.to_layout(topk_experts_weights_rm, ttnn.TILE_LAYOUT)
        print("topk_experts_weights_tile.shape : ", topk_experts_weights_tile.shape)
        ttnn.deallocate(topk_experts_weights_rm)

        weighted_output = ttnn.mul(
            combined_output,
            topk_experts_weights_tile,
        )
        print("weighted_output.shape : ", weighted_output.shape)

        # 10. Sum over experts dimension
        final_output = ttnn.sum(weighted_output, dim=0, keepdim=True)

        # 11. Remove padding if it was added
        if pad_amount > 0:
            # Slice to remove padding: final_output shape is (1, 1, batch_size_per_device*seq_len, hidden_size)
            # We need to slice the seq dimension from [0:original_num_tokens]
            final_output = ttnn.slice(final_output, (0, 0, 0, 0), (1, 1, original_num_tokens, self.hidden_size))

        return final_output


class TTNNMoE(TTNNModule):
    """
    Baseline MoE module for DeepSeek V3.

    Forward pass:
    1. All-gather to revert tensor parallelism
    2. MoE gate routing
    3. Experts module handles: dispatch → compute → combine → weight
    4. Reduce-scatter final output
    5. Add shared experts output
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_experts_per_tok = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts

    @classmethod
    def from_torch(cls, torch_moe):
        """
        Create TTNNMoE from PyTorch MoE module.

        Args:
            torch_moe: PyTorch DeepseekV3MoE module
        """
        module = cls(torch_moe.config)
        module._fallback_torch_layer = torch_moe

        # Create submodules
        module.gate = TTNNGlm4MoeTopkRouter.from_parameters(
            torch_moe.gate.weight, torch_moe.gate.e_score_correction_bias
        )
        # Convert router to TTNN
        module.route_tokens_to_experts = TTNNMoERouterDecode.from_torch(
            Glm4MoeRouteTokenToExperts(
                torch_moe.gate.e_score_correction_bias,
                torch_moe.n_routed_experts,
                torch_moe.n_group,
                torch_moe.topk_group,
                torch_moe.top_k,
                torch_moe.norm_topk_prob,
                torch_moe.routed_scaling_factor,
            )
        )
        module.experts = TTNNExperts.from_torch(torch_moe.experts)
        module.shared_experts = TTNNGlm4MoeMLP.from_torch(torch_moe.shared_experts)

        return module

    @run_on_devices(DeviceArch.T3K)
    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Forward pass: all-gather → gate → experts (handles dispatch/combine) → reduce-scatter → add shared.

        Args:
            x: Input tensor with shape (batch_size_per_device, 1, seq_len, hidden_size)

        Returns:
            Output tensor with same shape as input
        """
        self.num_devices = self.device.get_num_devices()
        self.num_dispatch_devices = self.device.shape[0]
        self.num_experts_per_device = even_int_div(self.config.n_routed_experts, self.num_devices)
        # Store original input for shared experts
        residual = x
        # 2. MoE gate routing
        router_logits = self.gate(x)
        # 1. All-gather to revert tensor parallelism
        x = ttnn.experimental.all_gather_async(
            x,
            dim=-1,
            multi_device_global_semaphore=self.device_state.ccl_manager.get_and_cycle_ag_semaphore_handles(1),
            barrier_semaphore=self.device_state.ccl_manager.get_and_cycle_barrier_semaphore_handle(1),
            num_links=1,
            topology=ttnn.Topology.Linear,
        )

        # Route tokens to experts
        topk_experts_indices, topk_experts_weights = self.route_tokens_to_experts(router_logits)
        x = ttnn.unsqueeze(x, 1)  # Add experts dimension for compatibility with experts module
        # 3. Experts handle dispatch → compute → combine → weight internally

        routed_output = self.experts(x, topk_experts_indices, topk_experts_weights)

        # 4. Reduce-scatter final output
        routed_output = ttnn.experimental.reduce_scatter_minimal_async(
            routed_output.to_ttnn,
            persistent_output_buffers=None,
            dim=3,
            multi_device_global_semaphore=self.device_state.ccl_manager.get_and_cycle_rs_semaphore_handles(1),
            barrier_semaphore=self.device_state.ccl_manager.get_and_cycle_barrier_semaphore_handle(1),
            num_links=1,
            cluster_axis=1,
            topology=ttnn.Topology.Ring,
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )

        # 5. Add shared experts output
        shared_output = self.shared_experts(residual)
        output = ttnn.add(routed_output, shared_output.to_ttnn)
        output = ttnn.squeeze(output, 1)  # Remove experts dimension
        return output


from models.experimental.tt_symbiote.modules.linear import TTNNLinear


def _stack_deepseek_v2_experts(experts_module_list, config):
    """
    Build a single object with gate_up_proj and down_proj from a ModuleList of
    DeepseekV2MLP-style experts (each with gate_proj, up_proj, down_proj).
    Compatible with TTNNExperts.from_torch() so V2 MoE can reuse the V3 expert stack.
    """
    experts = list(experts_module_list)
    gate_up_list = []
    down_list = []
    for expert in experts:
        if expert is None:
            continue
        gate_w = expert.gate_proj.weight
        up_w = expert.up_proj.weight
        down_w = expert.down_proj.weight
        gate_up_list.append(torch.cat([gate_w, up_w], dim=0))
        down_list.append(down_w.T)
    gate_up_proj = torch.stack(gate_up_list, dim=0)
    down_proj = torch.stack(down_list, dim=0)
    out = type("DeepseekV2ExpertsStack", (), {})()
    out.config = config
    out.gate_up_proj = gate_up_proj
    out.down_proj = down_proj
    return out


class TTNNDeepseekV2MoE(TTNNModule):
    """
    TTNN symbiote for DeepSeek V2 MoE (e.g. DeepSeek-OCR).
    Uses TTNNDeepseekOCRMoEGate (supports greedy, group_limited_greedy, noaux_tc),
    reuses TTNNExperts for moe_infer (same as V3), and TTNNGlm4MoeMLP for shared expert.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_experts_per_tok = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts

    @classmethod
    def from_torch(cls, torch_moe):
        """
        Create TTNNDeepseekV2MoE from PyTorch DeepseekV2MoE (or any MoE with
        .gate (MoEGate), .experts (ModuleList of MLPs with gate_proj/up_proj/down_proj),
        optional .shared_experts, and .config).
        """
        module = cls(torch_moe.config)
        module._fallback_torch_layer = torch_moe

        module.gate = TTNNDeepseekOCRMoEGate.from_torch(torch_moe.gate)

        stacked_experts = _stack_deepseek_v2_experts(torch_moe.experts, torch_moe.config)
        module.experts = TTNNExperts.from_torch(stacked_experts)

        if getattr(torch_moe, "shared_experts", None) is not None:
            module.shared_experts = TTNNGlm4MoeMLP.from_torch(torch_moe.shared_experts)
        else:
            module.shared_experts = None

        return module

    def preprocess_weights_impl(self):
        if hasattr(self.gate, "init_parameters"):
            self.gate.init_parameters()
        self.gate.preprocess_weights()
        self.experts.preprocess_weights()
        if self.shared_experts is not None:
            self.shared_experts.preprocess_weights()

    def _forward_ttnn(self, hidden_states):
        if hasattr(hidden_states, "to_ttnn"):
            hidden_states = hidden_states.to_ttnn
        orig_shape = list(hidden_states.shape)
        if len(orig_shape) == 3:
            batch, seq, hidden = orig_shape
            hidden_states_4d = ttnn.reshape(hidden_states, (batch, 1, seq, hidden))
        else:
            hidden_states_4d = hidden_states
            batch, _, seq, hidden = hidden_states_4d.shape
            orig_shape = [batch, seq, hidden]
        topk_idx, topk_weight, _ = self.gate(hidden_states)

        torch.save(topk_idx.to_torch, "models/experimental/tt_symbiote/tests/input_test_moe/dump/topk_idx_ttnn.pt")
        torch.save(
            topk_weight.to_torch, "models/experimental/tt_symbiote/tests/input_test_moe/dump/topk_weight_ttnn.pt"
        )
        topk_idx = topk_idx[:, :, :6]
        topk_weight = topk_weight[:, :, :6]
        print("topk_idx.shape : ", topk_idx.shape)
        print("topk_weight.shape : ", topk_weight.shape)
        # hidden_states_4d = ttnn.experimental.reduce_scatter_minimal_async(
        #     hidden_states_4d,
        #     persistent_output_buffers=None,
        #     dim=3,
        #     multi_device_global_semaphore=self.device_state.ccl_manager.get_and_cycle_rs_semaphore_handles(1),
        #     barrier_semaphore=self.device_state.ccl_manager.get_and_cycle_barrier_semaphore_handle(1),
        #     num_links=1,
        #     cluster_axis=1,
        #     topology=ttnn.Topology.Ring,
        #     chunks_per_sync=10,
        #     num_workers_per_link=2,
        #     num_buffers_per_channel=2,
        # )

        # hidden_states_4d = ttnn.experimental.all_gather_async(
        #     hidden_states_4d,
        #     dim=3,
        #     cluster_axis=1,
        #     topology=ttnn.Topology.Ring,
        #     multi_device_global_semaphore=self.device_state.ccl_manager.get_and_cycle_ag_semaphore_handles(1),
        #     num_links=1
        # )
        # composer = ttnn.concat_mesh_to_tensor_composer(self.device, dim=-1)
        # hidden_states_4d = ttnn.aggregate_tensor(hidden_states_4d, composer)
        routed_output = self.experts(hidden_states_4d, topk_idx, topk_weight)
        print("routed_output before slice : ", routed_output.shape)
        routed_output = routed_output[:, :, :, :1280]
        print("routed_output : ", routed_output.shape)

        import numpy as np

        np.savetxt(
            "models/experimental/tt_symbiote/tests/deepseek_ocr_vision_model/dump_file/tensor_ttnn.txt",
            routed_output.to_torch.to(torch.float32).cpu().numpy().reshape(-1, routed_output.to_torch.shape[-1]),
            fmt="%d",
        )
        from tests.ttnn.utils_for_testing import check_with_pcc

        routed_output_og = torch.load("models/experimental/tt_symbiote/tests/input_test_moe/dump/routed_output.pt")
        passed, msg = check_with_pcc(routed_output.to_torch.squeeze(0), routed_output_og)
        print("********** PCC: ", msg, passed)
        if hasattr(routed_output, "to_ttnn"):
            routed_output = routed_output.to_ttnn
        if self.shared_experts is not None:
            shared_out = self.shared_experts(hidden_states_4d)
            if hasattr(shared_out, "to_ttnn"):
                shared_out = shared_out.to_ttnn
            routed_output = ttnn.add(routed_output, shared_out)
        return ttnn.reshape(routed_output, orig_shape)

    def forward(self, hidden_states):
        self._used_fallback = False
        device = getattr(self, "device", None)
        if device is None:
            self._used_fallback = True
            inp = _to_torch_for_fallback(hidden_states)
            with torch.no_grad():
                return self._fallback_torch_layer(inp)
        try:
            return self._forward_ttnn(hidden_states)
        except Exception:
            self._used_fallback = True
            inp = _to_torch_for_fallback(hidden_states)
            with torch.no_grad():
                return self._fallback_torch_layer(inp)


def _to_torch_for_fallback(tensor):
    """Convert symbiote/ttnn input to torch for fallback; ttnn.to_torch copies from device if needed."""
    if isinstance(tensor, torch.Tensor):
        from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

        if not isinstance(tensor, TorchTTNNTensor):
            return tensor
        if getattr(tensor, "ttnn_tensor", None) is not None:
            return ttnn.to_torch(tensor.ttnn_tensor)
        return getattr(tensor, "to_torch", tensor.elem if getattr(tensor, "elem", None) is not None else tensor)
    if hasattr(tensor, "ttnn_tensor") and tensor.ttnn_tensor is not None:
        return ttnn.to_torch(tensor.ttnn_tensor)
    try:
        if getattr(ttnn, "is_tensor_storage_on_device", None) and ttnn.is_tensor_storage_on_device(tensor):
            return ttnn.to_torch(tensor)
    except Exception:
        pass
    if hasattr(tensor, "to_torch"):
        out = tensor.to_torch
        if callable(out):
            return out()
        return out
    return tensor


class TTNNDeepseekOCRMoEGate(TTNNModule):
    """
    MoEGate module for DeepSeek OCR.
    """

    def __init__(self, config, use_bitonic_sort=True):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux
        self.topk_method = config.topk_method
        self.n_group = config.n_group
        self.topk_group = config.topk_group

        # topk selection algorithm
        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size

        # TTNN tensors

    @classmethod
    def from_torch(cls, torch_gate):
        ttnn_gate = cls(torch_gate.config)
        ttnn_gate._fallback_torch_layer = torch_gate
        return ttnn_gate

    def init_parameters(self):
        """
        Load weights from PyTorch to Host memory (Tile Layout).
        """
        # 1. Gate Weight: Transpose for TTNN Matmul [Hidden, Experts]
        # weight_t = self._fallback_torch_layer.weight.T.contiguous()
        # self.weight = ttnn.from_torch(
        #     weight_t,
        #     dtype=ttnn.bfloat16,
        #     layout=ttnn.TILE_LAYOUT
        # )
        self.linear = TTNNLinear.from_parameters(weight=self._fallback_torch_layer.weight, bias=None)
        self.linear.preprocess_weights()
        # Buffers for group_limited_greedy and noaux_tc
        self.scatter_input = None
        self.scatter_src = None
        self.e_score_correction_bias = None
        if self.topk_method in ("group_limited_greedy", "noaux_tc"):
            self.scatter_input = ttnn.from_torch(torch.zeros((1, 1, 1, self.n_group), dtype=torch.bfloat16))
            self.scatter_src = ttnn.from_torch(torch.ones((1, 1, 1, self.topk_group), dtype=torch.bfloat16))
        if self.topk_method == "noaux_tc" and hasattr(self._fallback_torch_layer, "e_score_correction_bias"):
            bias = self._fallback_torch_layer.e_score_correction_bias
            bias = bias.reshape(1, 1, 1, -1).to(torch.bfloat16)
            self.e_score_correction_bias = ttnn.from_torch(bias, dtype=ttnn.bfloat16)

    def move_weights_to_device_impl(self):
        """
        Move weights from Host to Device.
        """
        # Move Weight
        if self.linear is not None:
            self.linear.to_device(self.device)  # Tell submodule which device to use
            self.linear.move_weights_to_device()
        if self.scatter_input is not None:
            self.scatter_input = ttnn.to_device(self.scatter_input, self.device)
            self.scatter_input = ttnn.to_layout(
                self.scatter_input, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
        if self.scatter_src is not None:
            self.scatter_src = ttnn.to_device(self.scatter_src, self.device)
            self.scatter_src = ttnn.to_layout(self.scatter_src, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if self.e_score_correction_bias is not None:
            self.e_score_correction_bias = ttnn.to_device(self.e_score_correction_bias, self.device)
            self.e_score_correction_bias = ttnn.to_layout(
                self.e_score_correction_bias, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

    def _forward_group_limited_greedy(self, scores: ttnn.Tensor):
        """Group-limited greedy: max per group -> top-k groups -> top-k experts within those groups."""
        if scores.layout != ttnn.TILE_LAYOUT:
            scores = ttnn.to_layout(scores, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # Ensure 4D (batch, 1, seq, experts) for shape[i] and downstream ops
        if len(scores.shape) == 3:
            scores = ttnn.unsqueeze(scores, 1)
        T = scores.shape[2]
        n_experts = scores.shape[3]
        experts_per_group = n_experts // self.n_group
        grouped = ttnn.reshape(scores, ttnn.Shape((1, T, self.n_group, experts_per_group)))
        group_scores = ttnn.max(grouped, dim=3, keepdim=True)  # (1, T, n_group, 1) - keep 4D for downstream
        ttnn.deallocate(grouped)
        _, topk_group_idx = ttnn.topk(group_scores, k=self.topk_group, dim=2, sorted=False)  # (1, T, topk_group, 1)
        ttnn.deallocate(group_scores)
        topk_group_idx = ttnn.typecast(topk_group_idx, ttnn.uint32)  # scatter expects integer index dtype
        # Scatter requires index shape to match source (1, 1, T, topk_group). Topk returns (1, T, topk_group) or (1, T, topk_group, 1).
        if len(topk_group_idx.shape) == 4 and topk_group_idx.shape[-1] == 1:
            topk_group_idx = ttnn.squeeze(topk_group_idx, -1)
        if len(topk_group_idx.shape) == 3:
            topk_group_idx = ttnn.unsqueeze(topk_group_idx, 1)  # (1, 1, T, topk_group)
        input_mask = ttnn.repeat(self.scatter_input, ttnn.Shape((1, 1, T, 1)))
        src_tensor = ttnn.repeat(self.scatter_src, ttnn.Shape((1, 1, T, 1)))
        active_groups_mask = ttnn.scatter(input=input_mask, index=topk_group_idx, src=src_tensor, dim=3)
        ttnn.deallocate(input_mask)
        ttnn.deallocate(src_tensor)
        ttnn.deallocate(topk_group_idx)
        active_groups_mask = ttnn.reshape(active_groups_mask, ttnn.Shape((1, T, self.n_group, 1)))
        active_experts_mask = ttnn.repeat(active_groups_mask, ttnn.Shape((1, 1, 1, experts_per_group)))
        ttnn.deallocate(active_groups_mask)
        active_experts_mask = ttnn.reshape(active_experts_mask, ttnn.Shape((1, 1, T, n_experts)))
        masked_scores = ttnn.mul(scores, active_experts_mask)
        ttnn.deallocate(active_experts_mask)
        topk_weight, topk_idx = ttnn.topk(masked_scores, k=self.top_k, dim=3, sorted=False)
        ttnn.deallocate(masked_scores)
        return topk_idx, topk_weight

    def _forward_noaux_tc(self, scores: ttnn.Tensor):
        """No-aux TC: bias-adjusted scores for group choice, top-2 sum per group -> top-k groups -> top-k experts; weights from original scores."""
        if scores.layout != ttnn.TILE_LAYOUT:
            scores = ttnn.to_layout(scores, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # Ensure 4D (batch, 1, seq, experts) for shape[i] and downstream ops
        if len(scores.shape) == 3:
            scores = ttnn.unsqueeze(scores, 1)
        T = scores.shape[2]
        n_experts = scores.shape[3]
        experts_per_group = n_experts // self.n_group
        bias = ttnn.repeat(self.e_score_correction_bias, ttnn.Shape((1, 1, T, 1)))
        bias = ttnn.to_layout(bias, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        scores_for_choice = ttnn.add(scores, bias)
        ttnn.deallocate(bias)
        grouped = ttnn.reshape(scores_for_choice, ttnn.Shape((1, T, self.n_group, experts_per_group)))
        top2_scores, _ = ttnn.topk(grouped, k=2, dim=3)
        ttnn.deallocate(grouped)
        group_scores = ttnn.sum(top2_scores, dim=3, keepdim=True)  # (1, T, n_group, 1) - keep 4D
        ttnn.deallocate(top2_scores)
        _, topk_group_idx = ttnn.topk(group_scores, k=self.topk_group, dim=2, sorted=False)  # (1, T, topk_group, 1)
        ttnn.deallocate(group_scores)
        topk_group_idx = ttnn.typecast(topk_group_idx, ttnn.uint32)  # scatter expects integer index dtype
        # Scatter requires index shape (1, 1, T, topk_group) to match source. Topk may return (1, T, topk_group) or (1, T, topk_group, 1).
        if len(topk_group_idx.shape) == 4 and topk_group_idx.shape[-1] == 1:
            topk_group_idx = ttnn.squeeze(topk_group_idx, -1)
        if len(topk_group_idx.shape) == 3:
            topk_group_idx = ttnn.unsqueeze(topk_group_idx, 1)
        input_mask = ttnn.repeat(self.scatter_input, ttnn.Shape((1, 1, T, 1)))
        src_tensor = ttnn.repeat(self.scatter_src, ttnn.Shape((1, 1, T, 1)))
        active_groups_mask = ttnn.scatter(input=input_mask, index=topk_group_idx, src=src_tensor, dim=3)
        ttnn.deallocate(input_mask)
        ttnn.deallocate(src_tensor)
        ttnn.deallocate(topk_group_idx)
        active_groups_mask = ttnn.reshape(active_groups_mask, ttnn.Shape((1, T, self.n_group, 1)))
        active_experts_mask = ttnn.repeat(active_groups_mask, ttnn.Shape((1, 1, 1, experts_per_group)))
        ttnn.deallocate(active_groups_mask)
        active_experts_mask = ttnn.reshape(active_experts_mask, ttnn.Shape((1, 1, T, n_experts)))
        masked_scores = ttnn.mul(scores_for_choice, active_experts_mask)
        ttnn.deallocate(active_experts_mask)
        _, topk_idx = ttnn.topk(masked_scores, k=self.top_k, dim=3, sorted=False)
        ttnn.deallocate(masked_scores)
        ttnn.deallocate(scores_for_choice)
        topk_weight = ttnn.gather(scores, dim=3, index=topk_idx)
        return topk_idx, topk_weight

    def forward(self, hidden_states: ttnn.Tensor):
        """
        Forward logic matching DeepSeek V3/OCR.
        Input: [Batch, 1, Seq, Hidden] or (Batch, Seq, Hidden)
        Output: (topk_weight, topk_idx) same as HF gate.
        """
        logits = self.linear(hidden_states).to_ttnn

        if self.scoring_func == "softmax":
            scores = ttnn.softmax(logits, dim=-1)
        elif self.scoring_func == "sigmoid":
            scores = ttnn.sigmoid(logits)
        else:
            raise NotImplementedError(f"insupportable scoring function for MoE gating: {self.scoring_func}")

        if self.topk_method == "greedy":
            topk_weight, topk_idx = ttnn.topk(scores, k=self.top_k, dim=-1, sorted=False)
        elif self.topk_method == "group_limited_greedy":
            topk_idx, topk_weight = self._forward_group_limited_greedy(scores)
        elif self.topk_method == "noaux_tc":
            topk_idx, topk_weight = self._forward_noaux_tc(scores)
        else:
            raise NotImplementedError(f"topk_method {self.topk_method!r} not supported")

        ### norm gate to sum 1 (match HF lines 502-507)
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = ttnn.sum(topk_weight, dim=-1, keepdim=True)
            denominator = ttnn.add(denominator, 1e-20)
            topk_weight = ttnn.div(topk_weight, denominator)
        topk_weight = ttnn.mul(topk_weight, self.routed_scaling_factor)

        ### expert-level auxiliary loss: only when training and alpha > 0 (HF 508-536)
        if getattr(self, "training", False) and self.alpha > 0.0:
            # TODO: implement aux_loss with ttnn (scatter_add / one_hot / mean)
            aux_loss = None
        else:
            aux_loss = None

        return topk_idx, topk_weight, aux_loss
