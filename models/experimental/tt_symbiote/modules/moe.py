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
from models.experimental.tt_symbiote.core.module import TTNNModule, deallocate_weights_after, run_on_devices, DeviceArch
from models.experimental.tt_symbiote.modules.linear import (
    TTNNLinear,
    TTNNLinearSilu,
    TTNNLinearLLamaIColShardedWRowSharded,
    TTNNLinearIColShardedWRowSharded,
)
from models.experimental.tt_symbiote.core.run_config import disable_trace
import math


# Helper to robustly convert various tensor types to a torch.Tensor
def _to_torch_any(tensor):
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    if isinstance(tensor, TorchTTNNTensor):
        return tensor.to_torch
    if isinstance(tensor, torch.Tensor):
        return tensor
    # Assume it's a ttnn.Tensor
    return TorchTTNNTensor(tensor).to_torch


def _to_ttnn_raw(tensor):
    """Return raw ttnn.Tensor from TorchTTNNTensor or ttnn.Tensor for use in ttnn ops."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    if tensor is None:
        raise ValueError("Expected a tensor; got None.")
    if isinstance(tensor, TorchTTNNTensor):
        if not hasattr(tensor, "to_ttnn"):
            raise AttributeError("TorchTTNNTensor has no to_ttnn property.")
        return tensor.to_ttnn
    if hasattr(tensor, "shape") and hasattr(tensor, "layout"):
        return tensor
    raise TypeError(f"Expected TorchTTNNTensor or ttnn.Tensor; got {type(tensor).__name__}.")


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
    out_subblock_w: int = None,
    per_core_M: int = 1,
):
    grid = device.compute_with_storage_grid_size()
    core_x = int(getattr(grid, "x"))
    core_y = int(getattr(grid, "y"))
    n_tiles = (int(out_features) + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE
    num_cores = max(1, core_x * core_y)
    per_core_N = max(1, int(math.ceil(n_tiles / num_cores)))
    out_block_w = per_core_N
    if out_subblock_w is None:
        out_subblock_w = min(per_core_N, 4)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(core_x, core_y),
        in0_block_w=int(in0_block_w),
        out_subblock_h=int(out_subblock_h),
        out_subblock_w=int(out_subblock_w),
        out_block_h=1,
        out_block_w=int(out_block_w),
        per_core_M=int(per_core_M),
        per_core_N=int(per_core_N),
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )


def _make_fitted_sparse_matmul_program_config(
    device,
    out_features: int,
    in0_block_w: int,
    per_core_M: int = 1,
):
    """sparse_matmul config that fits the grid to the number of output tiles.

    Unlike ``_make_sparse_matmul_program_config`` this function finds a
    rectangular core grid where every core has work, which is required by
    the sparse_matmul kernel.
    """
    grid = device.compute_with_storage_grid_size()
    max_x = int(getattr(grid, "x"))
    max_y = int(getattr(grid, "y"))
    n_tiles = (int(out_features) + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE

    best = None
    for pcn in range(1, n_tiles + 1):
        n_cores = math.ceil(n_tiles / pcn)
        if n_cores > max_x * max_y:
            continue
        if n_cores * pcn != n_tiles:
            continue
        for gy in range(1, min(n_cores, max_y) + 1):
            if n_cores % gy == 0:
                gx = n_cores // gy
                if gx <= max_x:
                    best = (gx, gy, pcn)
                    break
        if best is not None:
            break

    if best is None:
        core_x, core_y = max_x, max_y
        pcn = max(1, math.ceil(n_tiles / (core_x * core_y)))
    else:
        core_x, core_y, pcn = best

    out_subblock_w = min(pcn, 4)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(core_x, core_y),
        in0_block_w=int(in0_block_w),
        out_subblock_h=1,
        out_subblock_w=int(out_subblock_w),
        out_block_h=1,
        out_block_w=int(pcn),
        per_core_M=int(per_core_M),
        per_core_N=int(pcn),
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
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=True)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )
        scores_for_choice = router_logits_for_choice.masked_fill(~score_mask.bool(), 0.0)
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=True)[1]
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
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=True)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )
        scores_for_choice = router_logits_for_choice.masked_fill(~score_mask.bool(), 0.0)
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=True)[1]
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


class Qwen3RouteTokenToExperts(nn.Module):
    """Softmax-based routing for Qwen3-Coder-Next / Qwen3-Omni (vs sigmoid+bias for GLM/DeepSeek)."""

    def __init__(self, top_k, norm_topk_prob, routed_scaling_factor, n_routed_experts):
        super().__init__()
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.n_routed_experts = n_routed_experts
        self.use_softmax = True
        self.n_group = 1
        self.topk_group = 1
        self.register_buffer("e_score_correction_bias", torch.zeros(n_routed_experts, dtype=torch.float32))

    def forward(self, router_logits):
        probs = F.softmax(router_logits.to(torch.float32), dim=-1).to(router_logits.dtype)
        _, topk_indices = torch.topk(probs, k=self.top_k, dim=-1, sorted=False)
        topk_weights = probs.gather(1, topk_indices)
        if self.norm_topk_prob:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights


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
        # Keep experts on torch to avoid mesh_composer issues during grouped_mm_experts_forward
        # and to sidestep large TTNN expert weight allocations.
        pass

    def move_weights_to_device_impl(self):
        # No-op for torch-only experts.
        pass

    def forward(self, x, topk_experts_indices, topk_experts_weights):
        # HF grouped_mm_experts_forward must run on plain torch tensors. Wrapping these in
        # TorchTTNNTensor triggers mesh_composer paths that can fail for 2D tensors.
        x = _to_torch_any(x)
        topk_experts_indices = _to_torch_any(topk_experts_indices).to(torch.int64)
        topk_experts_weights = _to_torch_any(topk_experts_weights)
        return self.torch_layer(x, topk_experts_indices, topk_experts_weights)


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
        x = ttnn.mul(x_gate.to_ttnn, x_up.to_ttnn)
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

        # Add correction bias
        bias = ttnn.repeat(self.e_score_correction_bias, ttnn.Shape((1, 1, T, 1)))
        bias = ttnn.to_layout(bias, ttnn.TILE_LAYOUT)

        scores_with_bias = ttnn.add(scores, bias)
        ttnn.deallocate(bias)

        # Reshape into groups: (1,T,n_group,experts_per_group)
        grouped = ttnn.reshape(scores_with_bias, ttnn.Shape((1, T, self.torch_layer.n_group, experts_per_group)))

        # Top-2 experts per group
        top2_scores, _ = ttnn.topk(grouped, k=2, dim=3)
        ttnn.deallocate(grouped)

        # group_scores: (1, T, n_group)
        group_scores = ttnn.sum(top2_scores, dim=3)
        ttnn.deallocate(top2_scores)

        # Top-k groups
        _, topk_group_idx = ttnn.topk(group_scores, k=self.torch_layer.topk_group, dim=2)
        ttnn.deallocate(group_scores)

        # Build group mask via scatter
        input_mask = ttnn.repeat(self.scatter_input, ttnn.Shape((1, 1, T, 1)))

        src_tensor = ttnn.repeat(self.scatter_src, ttnn.Shape((1, 1, T, 1)))
        topk_group_idx = ttnn.unsqueeze(topk_group_idx, dim=1)
        active_groups_mask = ttnn.scatter(input=input_mask, index=topk_group_idx, src=src_tensor, dim=3)
        ttnn.deallocate(input_mask)
        ttnn.deallocate(src_tensor)
        ttnn.deallocate(topk_group_idx)

        # reshape: (1, T, n_group,1)
        active_groups_mask = ttnn.reshape(active_groups_mask, ttnn.Shape((1, T, self.torch_layer.n_group, 1)))

        # Expand group mask → expert mask
        active_experts_mask = ttnn.repeat(active_groups_mask, ttnn.Shape((1, 1, 1, experts_per_group)))
        ttnn.deallocate(active_groups_mask)

        # reshape back: (1, 1, T, n_experts)
        active_experts_mask = ttnn.reshape(active_experts_mask, ttnn.Shape((1, 1, T, n_experts)))

        # Zero out inactive experts
        masked_scores = ttnn.mul(scores_with_bias, active_experts_mask)
        ttnn.deallocate(active_experts_mask)

        # Top-k experts from active experts
        _, topk_expert_idx = ttnn.topk(masked_scores, k=self.torch_layer.top_k, dim=3)
        ttnn.deallocate(masked_scores)

        # Gather original sigmoid scores (NO bias)
        topk_weights = ttnn.gather(scores, dim=3, index=topk_expert_idx)
        ttnn.deallocate(scores)

        # Normalize weights
        denom = ttnn.sum(topk_weights, dim=3, keepdim=True) + 1e-20
        topk_weights = ttnn.div(topk_weights, denom)
        ttnn.deallocate(denom)

        # Apply scaling factor
        scale = ttnn.repeat(self.expert_scale, ttnn.Shape((1, 1, T, 1)))
        scale = ttnn.to_layout(scale, ttnn.TILE_LAYOUT)

        topk_weights = ttnn.mul(topk_weights, scale)
        ttnn.deallocate(scale)
        T = topk_weights.shape[2]

        topk_expert_idx = ttnn.reshape(topk_expert_idx, ttnn.Shape((T, self.torch_layer.top_k)))

        topk_weights = ttnn.reshape(topk_weights, ttnn.Shape((T, self.torch_layer.top_k)))

        # Canonicalize ordering: sort per-token by weight for deterministic output.
        topk_idx_t = _to_torch_any(topk_expert_idx).to(torch.int64)
        topk_w_t = _to_torch_any(topk_weights).to(torch.float32)
        # Sort weights descending and permute indices
        sorted_w, sorted_pos = torch.sort(topk_w_t, dim=1, descending=True)
        sorted_idx = torch.gather(topk_idx_t, 1, sorted_pos)
        # Convert back to ttnn formats
        topk_expert_idx = ttnn.from_torch(sorted_idx.to(torch.int32))
        topk_weights = ttnn.from_torch(sorted_w.to(torch.bfloat16))
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
        # Use TTNN router
        topk_indices, topk_weights = self.route_tokens_to_experts(router_logits)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = self.experts(hidden_states, topk_indices.to(dtype=torch.int64), topk_weights).view(*orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states.to_ttnn


class TTNNMoERouterDecode(TTNNModule):
    """TTNN-accelerated MoE router (decode mode)."""

    @classmethod
    def from_torch(cls, torch_module: Glm4MoeRouteTokenToExperts):
        instance = cls()
        instance._fallback_torch_layer = torch_module
        return instance

    def preprocess_weights_impl(self):
        r = self._fallback_torch_layer

        # Cache the torch tensors.
        self._bias_torch = r.e_score_correction_bias.reshape(1, 1, 1, -1).to(torch.bfloat16)
        self._scatter_input_torch = torch.zeros(1, 1, 1, r.n_group, dtype=torch.bfloat16)
        self._scatter_src_torch = torch.ones(1, 1, 1, r.topk_group, dtype=torch.bfloat16)
        self._scale_torch = torch.full((1, 1, 1, r.top_k), r.routed_scaling_factor, dtype=torch.bfloat16)

    def move_weights_to_device_impl(self):
        self._use_softmax = getattr(self._fallback_torch_layer, "use_softmax", False)
        self._bias_dev = ttnn.to_device(
            ttnn.from_torch(self._bias_torch, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT),
            self.device,
        )
        self._scatter_input_dev = ttnn.to_device(
            ttnn.from_torch(self._scatter_input_torch, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT),
            self.device,
        )
        self._scatter_src_dev = ttnn.to_device(
            ttnn.from_torch(self._scatter_src_torch, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT),
            self.device,
        )
        self._scale_dev = ttnn.to_device(
            ttnn.from_torch(self._scale_torch, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT),
            self.device,
        )

    def forward(self, logits: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        r = self._fallback_torch_layer

        if logits.layout != ttnn.TILE_LAYOUT:
            logits = ttnn.to_layout(logits, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        logits = ttnn.reshape(logits, ttnn.Shape((1, 1, logits.shape[0], logits.shape[1])))
        if logits.dtype != ttnn.float32:
            logits_f32 = ttnn.typecast(logits, ttnn.float32)
            ttnn.deallocate(logits)
        else:
            logits_f32 = logits

        # Qwen3 / softmax path: no bias, no group selection (keep f32 for weights to match HF)
        if getattr(self, "_use_softmax", False):
            probs_f32 = ttnn.softmax(logits_f32, dim=-1)
            ttnn.deallocate(logits_f32)
            T = probs_f32.shape[2]
            # Top-k indices from bf16 for speed; weights from f32 for precision
            probs_bf16 = ttnn.typecast(probs_f32, ttnn.bfloat16)
            _, topk_expert_idx = ttnn.topk(probs_bf16, k=r.top_k, dim=3, largest=True, sorted=False)
            ttnn.deallocate(probs_bf16)
            topk_weights = ttnn.gather(probs_f32, dim=3, index=topk_expert_idx)
            ttnn.deallocate(probs_f32)
            # Normalize weights (float32)
            denom = ttnn.sum(topk_weights, dim=3, keepdim=True) + 1e-20
            topk_weights = ttnn.div(topk_weights, denom)
            ttnn.deallocate(denom)
            # Scale (float32)
            scale_rep = ttnn.repeat(self._scale_dev, ttnn.Shape((1, 1, T, 1)))
            scale_bf16 = ttnn.to_layout(scale_rep, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            scale_f32 = ttnn.typecast(scale_bf16, ttnn.float32)
            ttnn.deallocate(scale_bf16)
            topk_weights = ttnn.mul(topk_weights, scale_f32)
            ttnn.deallocate(scale_f32)
            topk_weights = ttnn.typecast(topk_weights, ttnn.bfloat16)
            topk_expert_idx = ttnn.reshape(topk_expert_idx, ttnn.Shape((T, r.top_k)))
            topk_weights = ttnn.reshape(topk_weights, ttnn.Shape((T, r.top_k)))
            return topk_expert_idx, topk_weights

        scores_f32 = ttnn.sigmoid(logits_f32)

        T = scores_f32.shape[2]
        n_experts = scores_f32.shape[3]
        n_group = r.n_group
        experts_per_group = n_experts // n_group

        bias_rm = self._bias_dev
        bias_rep_rm = ttnn.repeat(bias_rm, ttnn.Shape((1, 1, T, 1)))
        bias = ttnn.to_layout(bias_rep_rm, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # Convert bias to float32 for stable addition
        if bias.dtype != ttnn.float32:
            bias_f32 = ttnn.typecast(bias, ttnn.float32)
            ttnn.deallocate(bias)
        else:
            bias_f32 = bias

        scores_with_bias_f32 = ttnn.add(scores_f32, bias_f32)
        ttnn.deallocate(bias_f32)

        top_k = r.top_k

        if n_group <= r.topk_group:
            # Pass 1: rough BF16 topk(k+1) to find coarse threshold
            scores_bf16_p1 = ttnn.typecast(scores_with_bias_f32, ttnn.bfloat16)
            rough_vals, _ = ttnn.topk(scores_bf16_p1, k=top_k + 1, dim=3, largest=True, sorted=True)
            ttnn.deallocate(scores_bf16_p1)
            # (k+1)-th value gives coarse threshold.
            rough_thr_bf16 = ttnn.slice(rough_vals, [0, 0, 0, top_k], [1, 1, T, top_k + 1])
            ttnn.deallocate(rough_vals)
            rough_thr_f32 = ttnn.typecast(rough_thr_bf16, ttnn.float32)
            ttnn.deallocate(rough_thr_bf16)
            # Center scores around the decision boundary (float32 precision preserved)
            scores_c1 = ttnn.sub(scores_with_bias_f32, rough_thr_f32)
            ttnn.deallocate(rough_thr_f32)
            ttnn.deallocate(scores_with_bias_f32)

            # Pass 2: refined BF16 topk(k+1) on centered scores
            scores_bf16_p2 = ttnn.typecast(scores_c1, ttnn.bfloat16)
            refined_vals, _ = ttnn.topk(scores_bf16_p2, k=top_k + 1, dim=3, largest=True, sorted=True)
            ttnn.deallocate(scores_bf16_p2)
            # Second threshold is now near 0 → BF16 step ≈ 0.0001 (very precise)
            refined_thr_bf16 = ttnn.slice(refined_vals, [0, 0, 0, top_k], [1, 1, T, top_k + 1])
            ttnn.deallocate(refined_vals)
            refined_thr_f32 = ttnn.typecast(refined_thr_bf16, ttnn.float32)
            ttnn.deallocate(refined_thr_bf16)
            scores_c2 = ttnn.sub(scores_c1, refined_thr_f32)
            ttnn.deallocate(scores_c1)
            ttnn.deallocate(refined_thr_f32)

            # Final pass: exact topk(k) on doubly-centered scores
            scores_bf16_final = ttnn.typecast(scores_c2, ttnn.bfloat16)
            ttnn.deallocate(scores_c2)
            _, topk_expert_idx = ttnn.topk(scores_bf16_final, k=top_k, dim=3, largest=True, sorted=True)
            ttnn.deallocate(scores_bf16_final)
        else:
            # Group-based selection: apply same 3-pass centering after masking
            scores_bf16 = ttnn.typecast(scores_with_bias_f32, ttnn.bfloat16)
            ttnn.deallocate(scores_with_bias_f32)

            # group scores
            grouped = ttnn.reshape(scores_bf16, ttnn.Shape((1, T, n_group, experts_per_group)))
            top2_scores, _ = ttnn.topk(grouped, k=2, dim=3)
            ttnn.deallocate(grouped)
            group_scores = ttnn.sum(top2_scores, dim=3)
            ttnn.deallocate(top2_scores)

            # top-k groups
            _, topk_group_idx = ttnn.topk(group_scores, k=r.topk_group, dim=2)
            ttnn.deallocate(group_scores)

            # group mask via scatter
            input_mask_rm = ttnn.repeat(self._scatter_input_dev, ttnn.Shape((1, 1, T, 1)))
            src_rm = ttnn.repeat(self._scatter_src_dev, ttnn.Shape((1, 1, T, 1)))
            idx_rm = ttnn.to_layout(topk_group_idx, ttnn.ROW_MAJOR_LAYOUT)
            ttnn.deallocate(topk_group_idx)
            idx_4d = ttnn.unsqueeze(idx_rm, dim=1)
            ttnn.deallocate(idx_rm)
            active_groups_rm = ttnn.scatter(input=input_mask_rm, index=idx_4d, src=src_rm, dim=3)
            ttnn.deallocate(input_mask_rm)
            ttnn.deallocate(src_rm)
            ttnn.deallocate(idx_4d)

            # expert mask
            active_groups_rm = ttnn.reshape(active_groups_rm, ttnn.Shape((1, T, n_group, 1)))
            expert_mask_rm = ttnn.repeat(active_groups_rm, ttnn.Shape((1, 1, 1, experts_per_group)))
            ttnn.deallocate(active_groups_rm)
            expert_mask_rm = ttnn.reshape(expert_mask_rm, ttnn.Shape((1, 1, T, n_experts)))
            expert_mask = ttnn.to_layout(expert_mask_rm, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(expert_mask_rm)

            # top-k active experts
            masked_scores = ttnn.mul(scores_bf16, expert_mask)
            ttnn.deallocate(scores_bf16)
            ttnn.deallocate(expert_mask)
            _, topk_expert_idx = ttnn.topk(masked_scores, k=top_k, dim=3)
            ttnn.deallocate(masked_scores)

        # gather raw sigmoid scores (no bias) for weights
        topk_weights = ttnn.gather(scores_f32, dim=3, index=topk_expert_idx)
        ttnn.deallocate(scores_f32)

        # normalise
        denom = ttnn.sum(topk_weights, dim=3, keepdim=True)
        topk_weights = ttnn.div(topk_weights, denom)
        ttnn.deallocate(denom)

        # apply routing scale
        scale_rep_rm = ttnn.repeat(self._scale_dev, ttnn.Shape((1, 1, T, 1)))
        scale_bf16 = ttnn.to_layout(scale_rep_rm, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if scale_bf16.dtype != ttnn.float32:
            scale_f32 = ttnn.typecast(scale_bf16, ttnn.float32)
            ttnn.deallocate(scale_bf16)
        else:
            scale_f32 = scale_bf16
        topk_weights = ttnn.mul(topk_weights, scale_f32)
        ttnn.deallocate(scale_f32)

        # Reshape outputs to (T, top_k).
        topk_expert_idx = ttnn.reshape(topk_expert_idx, ttnn.Shape((T, r.top_k)))
        topk_weights = ttnn.reshape(topk_weights, ttnn.Shape((T, r.top_k)))
        return topk_expert_idx, topk_weights


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

        self.tt_w1_proj = None
        self.tt_w3_proj = None
        self.tt_w2_proj = None
        self.expert_mapping_tensors = None
        self.remap_topk_mask = None

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

        module.torch_w1_proj = torch_experts.gate_up_proj[:, : torch_experts.config.moe_intermediate_size, :].permute(
            [0, 2, 1]
        )
        module.torch_w3_proj = torch_experts.gate_up_proj[:, torch_experts.config.moe_intermediate_size :, :].permute(
            [0, 2, 1]
        )
        module.torch_w2_proj = torch_experts.down_proj.permute([0, 2, 1])

        return module

    def preprocess_weights_impl(self):
        """Preprocess expert weights: convert to bfloat16 and TILE_LAYOUT."""
        self.tt_w1_proj = ttnn.from_torch(
            self.torch_w1_proj.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(self.device, dim=0),
        )
        self.tt_w3_proj = ttnn.from_torch(
            self.torch_w3_proj.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(self.device, dim=0),
        )
        self.tt_w2_proj = ttnn.from_torch(
            self.torch_w2_proj.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(self.device, dim=0),
        )
        del self.torch_w1_proj
        del self.torch_w3_proj
        del self.torch_w2_proj

    def move_weights_to_device_impl(self):
        """Move preprocessed weights to device and create mapping tensors."""

        self.num_experts_per_device = self._get_num_experts_per_device(self.config, self.device)
        self.num_devices = self.device.get_num_devices()
        self.num_dispatch_devices = self.device.shape[1]

        self.tt_w1_proj = ttnn.to_device(self.tt_w1_proj, self.device)
        self.tt_w3_proj = ttnn.to_device(self.tt_w3_proj, self.device)
        self.tt_w2_proj = ttnn.to_device(self.tt_w2_proj, self.device)

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

        hidden_tiles = self.hidden_size // ttnn.TILE_SIZE
        intermediate_tiles = self.intermediate_size // ttnn.TILE_SIZE
        self._gate_up_program_config = _make_sparse_matmul_program_config(
            device=self.device,
            out_features=int(self.intermediate_size),
            in0_block_w=min(4, hidden_tiles),
            per_core_M=1,
        )
        self._down_program_config = _make_sparse_matmul_program_config(
            device=self.device,
            out_features=int(self.hidden_size),
            in0_block_w=min(4, intermediate_tiles),
            per_core_M=1,
        )
        self._expert_compute_cfg = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
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

        if topk_experts_indices.dtype != ttnn.uint16:
            if topk_experts_indices.layout != ttnn.TILE_LAYOUT:
                topk_experts_indices = ttnn.to_layout(
                    topk_experts_indices,
                    ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            topk_experts_indices = ttnn.typecast(topk_experts_indices, ttnn.uint16)
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
        x = ttnn.typecast(x, ttnn.bfloat16)
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

        # 4. Generate sparsity tensor
        remap_topk_mask_expanded = ttnn.repeat(self.remap_topk_mask, ttnn.Shape((1, batch_size_per_device, 1, 1)))
        _, sparsity_t = ttnn.moe_expert_token_remap(
            remap_topk_mask_expanded,
            self.expert_mapping_tensors,
            all_to_all_dispatch_metadata,
            reduction_size=SPARSITY_BLOCK_SIZE,
        )

        num_sparse_blocks = num_tokens // SPARSITY_BLOCK_SIZE
        x_sparse = ttnn.reshape(post_dispatch, shape=(1, num_sparse_blocks, SPARSITY_BLOCK_SIZE, self.hidden_size))

        w1_out = ttnn.sparse_matmul(
            x_sparse,
            self.tt_w1_proj,
            sparsity=sparsity_t,
            output_tile=ttnn.Tile([SPARSITY_BLOCK_SIZE, ttnn.TILE_SIZE]),
            program_config=self._gate_up_program_config,
            compute_kernel_config=self._expert_compute_cfg,
            is_input_a_sparse=False,
            is_input_b_sparse=True,
        )
        w3_out = ttnn.sparse_matmul(
            x_sparse,
            self.tt_w3_proj,
            sparsity=sparsity_t,
            output_tile=ttnn.Tile([SPARSITY_BLOCK_SIZE, ttnn.TILE_SIZE]),
            program_config=self._gate_up_program_config,
            compute_kernel_config=self._expert_compute_cfg,
            is_input_a_sparse=False,
            is_input_b_sparse=True,
        )

        w1_activated = ttnn.silu(w1_out)
        ttnn.deallocate(w1_out)
        intermediate = ttnn.mul(w1_activated, w3_out)
        ttnn.deallocate(w1_activated)
        ttnn.deallocate(w3_out)

        intermediate = ttnn.squeeze(intermediate, 0)
        intermediate = ttnn.squeeze(intermediate, 1)

        expert_output = ttnn.sparse_matmul(
            intermediate,
            self.tt_w2_proj,
            sparsity=sparsity_t,
            output_tile=ttnn.Tile([SPARSITY_BLOCK_SIZE, ttnn.TILE_SIZE]),
            program_config=self._down_program_config,
            compute_kernel_config=self._expert_compute_cfg,
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
        combined_output = ttnn.to_layout(combined_output, ttnn.TILE_LAYOUT)

        # 9. Apply expert weights
        topk_experts_weights_rm = ttnn.to_layout(topk_experts_weights, ttnn.ROW_MAJOR_LAYOUT)
        topk_experts_weights_rm = ttnn.unsqueeze(topk_experts_weights_rm, 0)
        topk_experts_weights_rm = ttnn.unsqueeze(topk_experts_weights_rm, 0)
        topk_experts_weights_rm = ttnn.repeat(topk_experts_weights_rm, repeat_dims=(self.hidden_size, 1, 1, 1))
        topk_experts_weights_rm = ttnn.permute(topk_experts_weights_rm, (3, 1, 2, 0))
        topk_experts_weights_tile = ttnn.to_layout(topk_experts_weights_rm, ttnn.TILE_LAYOUT)
        ttnn.deallocate(topk_experts_weights_rm)

        weighted_output = ttnn.mul(
            combined_output,
            topk_experts_weights_tile,
        )

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

        # Gate weight replicated in bfloat16 for end-to-end BF16 routing.
        module._gate_weight_torch = torch_moe.gate.weight.to(torch.bfloat16)

        return module

    def preprocess_weights_impl(self):
        self._gate_weight_tt = ttnn.from_torch(
            self._gate_weight_torch.T.contiguous(),  # (hidden_size, n_experts) bfloat16
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )

    def move_weights_to_device_impl(self):
        # No mesh_mapper → replicate to all devices
        self._gate_weight_tt = ttnn.to_device(self._gate_weight_tt, self.device)

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

        # 1. All-gather to revert tensor parallelism
        x = ttnn.experimental.all_gather_async(
            x,
            dim=-1,
            multi_device_global_semaphore=self.device_state.ccl_manager.get_and_cycle_ag_semaphore_handles(1),
            barrier_semaphore=self.device_state.ccl_manager.get_and_cycle_barrier_semaphore_handle(1),
            num_links=1,
            topology=ttnn.Topology.Linear,
        )

        # 2. MoE gate routing
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if x.dtype != ttnn.float32:
            x_f32 = ttnn.typecast(x, ttnn.float32)
        else:
            x_f32 = x
        router_logits_f32 = ttnn.linear(
            x_f32,
            self._gate_weight_tt,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            ),
        )
        if x_f32 is not x:
            ttnn.deallocate(x_f32)
        # Convert back to bfloat16 for router (ttnn.sigmoid / ttnn.topk require bf16)
        router_logits = ttnn.typecast(router_logits_f32, ttnn.bfloat16)
        ttnn.deallocate(router_logits_f32)

        T = router_logits.shape[-2]
        router_logits = ttnn.reshape(router_logits, ttnn.Shape((T, self.n_routed_experts)))

        # Call router forward
        self.route_tokens_to_experts.preprocess_weights()
        self.route_tokens_to_experts.move_weights_to_device()
        topk_experts_indices, topk_experts_weights = self.route_tokens_to_experts.forward(router_logits)

        x = ttnn.unsqueeze(x, 1)  # Add experts dimension for compatibility with experts module

        # 3. Experts handle dispatch → compute → combine → weight
        routed_output = self.experts(x, topk_experts_indices, topk_experts_weights)

        # 4. Reduce-scatter final output.
        n_rs = self.device.shape[1]  # devices along cluster_axis=1
        routed_out = routed_output.to_ttnn
        if n_rs > 1:
            routed_out = ttnn.mul(routed_out, 1.0 / float(n_rs))
        routed_output = ttnn.experimental.reduce_scatter_minimal_async(
            routed_out,
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


class TTNNBailingMoE(TTNNMoE):
    """TTNN MoE for BailingMoeV2 architecture (Ling-mini-2.0 model)."""

    @classmethod
    def from_torch(cls, torch_moe):
        """
        Create TTNNBailingMoE from PyTorch BailingMoeV2SparseMoeBlock module.

        Args:
            torch_moe: PyTorch BailingMoeV2SparseMoeBlock module

        Returns:
            TTNNBailingMoE module with consolidated expert weights
        """
        # 1. Adapt config to match Glm4MoeConfig structure
        adapted_config = cls._adapt_config(torch_moe.config)

        # 2. Consolidate ModuleList experts into 3D tensors
        consolidated_experts = cls._consolidate_experts(torch_moe.experts, adapted_config)

        # 3. Adapt gate to match expected structure
        adapted_gate = cls._adapt_gate(torch_moe.gate)

        # 4. Create module instance
        module = cls(adapted_config)
        module._fallback_torch_layer = torch_moe

        # 5. Initialize submodules using parent's pattern
        module.gate = TTNNGlm4MoeTopkRouter.from_parameters(adapted_gate.weight, adapted_gate.e_score_correction_bias)

        # Create routing module
        module.route_tokens_to_experts = TTNNMoERouterDecode.from_torch(
            Glm4MoeRouteTokenToExperts(
                adapted_gate.e_score_correction_bias,
                adapted_config.n_routed_experts,
                adapted_config.n_group,
                adapted_config.topk_group,
                adapted_config.num_experts_per_tok,
                True,  # norm_topk_prob (BailingMoeV2 uses normalized probabilities)
                adapted_config.routed_scaling_factor,
            )
        )

        # Initialize experts with consolidated weights
        module.experts = TTNNExperts.from_torch(consolidated_experts)

        # Initialize shared experts
        module.shared_experts = TTNNGlm4MoeMLP.from_torch(torch_moe.shared_experts)
        module._gate_weight_torch = adapted_gate.weight.float()  # Store replicated gate weight for preprocessing
        return module

    @staticmethod
    def _adapt_config(bailing_config):
        """
        Adapt BailingMoeV2Config to match Glm4MoeConfig structure.

        Args:
            bailing_config: BailingMoeV2Config instance

        Returns:
            Adapted config with Glm4-compatible attributes
        """

        # Create a simple namespace object to hold adapted attributes
        class AdaptedConfig:
            pass

        config = AdaptedConfig()

        # Map BailingMoeV2 attributes to Glm4MoeConfig naming
        config.hidden_size = bailing_config.hidden_size
        config.moe_intermediate_size = bailing_config.moe_intermediate_size
        config.num_experts_per_tok = bailing_config.num_experts_per_tok

        # Key difference: num_experts → n_routed_experts
        config.n_routed_experts = bailing_config.num_experts
        config.n_shared_experts = bailing_config.num_shared_experts
        config.n_group = bailing_config.n_group
        config.topk_group = bailing_config.topk_group
        config.routed_scaling_factor = bailing_config.routed_scaling_factor

        # Additional attributes needed by TTNNMoE
        config.hidden_act = bailing_config.hidden_act

        return config

    @staticmethod
    def _consolidate_experts(experts_list, config):
        """
        Consolidate ModuleList of experts into 3D tensors matching Glm4MoeNaiveMoe structure.

        Args:
            experts_list: nn.ModuleList of BailingMoeV2MLP modules
            config: BailingMoeV2Config

        Returns:
            Object with gate_up_proj and down_proj as 3D tensors
        """
        num_experts = len(experts_list)
        hidden_dim = config.hidden_size
        intermediate_dim = config.moe_intermediate_size

        # Allocate 3D tensors for consolidated weights
        # gate_up_proj: [num_experts, 2*intermediate_dim, hidden_dim]
        # down_proj: [num_experts, hidden_dim, intermediate_dim]
        gate_up_proj = torch.empty(
            num_experts, 2 * intermediate_dim, hidden_dim, dtype=experts_list[0].gate_proj.weight.dtype
        )
        down_proj = torch.empty(num_experts, hidden_dim, intermediate_dim, dtype=experts_list[0].down_proj.weight.dtype)

        # Stack weights from ModuleList
        for i, expert in enumerate(experts_list):
            # Concatenate gate_proj and up_proj along intermediate dimension
            gate_up_proj[i, :intermediate_dim, :] = expert.gate_proj.weight.data
            gate_up_proj[i, intermediate_dim:, :] = expert.up_proj.weight.data
            down_proj[i, :, :] = expert.down_proj.weight.data

        # Create object with expected attributes for TTNNExperts.from_torch()
        class ConsolidatedExperts:
            pass

        consolidated = ConsolidatedExperts()
        consolidated.gate_up_proj = gate_up_proj
        consolidated.down_proj = down_proj
        consolidated.config = config

        return consolidated

    @staticmethod
    def _adapt_gate(bailing_gate):
        """
        Adapt BailingMoeV2Gate to match expected structure with e_score_correction_bias.

        Args:
            bailing_gate: BailingMoeV2Gate module

        Returns:
            Object with weight and e_score_correction_bias attributes
        """

        class AdaptedGate:
            pass

        adapted = AdaptedGate()
        adapted.weight = bailing_gate.weight

        # Key difference: expert_bias → e_score_correction_bias
        if hasattr(bailing_gate, "expert_bias"):
            adapted.e_score_correction_bias = bailing_gate.expert_bias
        else:
            # Fallback if expert_bias doesn't exist
            adapted.e_score_correction_bias = torch.zeros(bailing_gate.weight.shape[0])

        return adapted


class TTNNQwen3MoE(TTNNMoE):
    """
    TTNN MoE for Qwen3-Coder-Next / Qwen3-Omni: softmax routing + gated shared expert.

    Expects torch_moe with .gate (weight + optional bias), .experts (gate_up_proj, down_proj, config),
    .shared_expert (MLP), .shared_expert_gate (Linear). Use from_torch for Coder-Next block;
    use TTNNQwen3TalkerMoE.from_torch for Qwen3-Omni Talker block (adapts ModuleList experts).
    """

    @classmethod
    def from_torch(cls, torch_moe):
        adapted_config = cls._adapt_config(torch_moe.gate, torch_moe.experts)
        module = cls(adapted_config)
        module._fallback_torch_layer = torch_moe

        zero_bias = torch.zeros(
            torch_moe.gate.weight.shape[0],
            device=torch_moe.gate.weight.device,
            dtype=torch_moe.gate.weight.dtype,
        )
        module.gate = TTNNGlm4MoeTopkRouter.from_parameters(torch_moe.gate.weight, zero_bias)
        # Replicated gate weight for routing on all-gathered (full) hidden: (hidden_size, num_experts)
        module._gate_weight_torch = torch_moe.gate.weight.detach().T.contiguous().to(torch.bfloat16)
        module.route_tokens_to_experts = TTNNMoERouterDecode.from_torch(
            Qwen3RouteTokenToExperts(
                top_k=adapted_config.num_experts_per_tok,
                norm_topk_prob=adapted_config.norm_topk_prob,
                routed_scaling_factor=adapted_config.routed_scaling_factor,
                n_routed_experts=adapted_config.n_routed_experts,
            )
        )
        experts_wrapper = cls._wrap_experts(torch_moe.experts, adapted_config)
        module.experts = TTNNExperts.from_torch(experts_wrapper)
        module.shared_experts = TTNNGlm4MoeMLP.from_torch(torch_moe.shared_expert)
        module.shared_expert_gate = TTNNLinear.from_torch(torch_moe.shared_expert_gate)
        return module

    def preprocess_weights_impl(self):
        # Keep gate weight on host until move_weights (need device for ReplicateTensorToMesh)
        pass

    def move_weights_to_device_impl(self):
        # Replicate gate to all devices so routing uses full (hidden_size, num_experts)
        self._gate_weight_tt = ttnn.from_torch(
            self._gate_weight_torch,
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        self._gate_weight_torch = None

    @run_on_devices(DeviceArch.T3K)
    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        self.num_devices = self.device.get_num_devices()
        self.num_dispatch_devices = self.device.shape[0]
        self.num_experts_per_device = even_int_div(self.config.n_routed_experts, self.num_devices)
        residual = x

        # 1. All-gather to revert tensor parallelism
        x = ttnn.experimental.all_gather_async(
            x,
            dim=-1,
            multi_device_global_semaphore=self.device_state.ccl_manager.get_and_cycle_ag_semaphore_handles(1),
            barrier_semaphore=self.device_state.ccl_manager.get_and_cycle_barrier_semaphore_handle(1),
            num_links=1,
            topology=ttnn.Topology.Linear,
        )

        # 2. Gate routing on full (all-gathered) hidden with replicated weight
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        x_shape = list(x.shape)
        T = 1
        for d in x_shape[:-1]:
            T *= d
        H = x_shape[-1]
        x_2d = ttnn.reshape(x, ttnn.Shape((T, H)))
        router_logits = ttnn.linear(
            x_2d,
            self._gate_weight_tt,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        topk_experts_indices, topk_experts_weights = self.route_tokens_to_experts(router_logits)

        # 3. Expert dispatch → compute → combine → weight
        # Ensure x is 4D (batch, 1, seq, hidden) for TTNNExperts
        x_full = x
        if len(x.shape) == 3:
            x = ttnn.reshape(x, ttnn.Shape((x.shape[0], 1, x.shape[1], x.shape[2])))
        routed_output = self.experts(x, topk_experts_indices, topk_experts_weights)

        # 4. Reduce-scatter (scale by 1/n_rs so sum then scatter matches single-device magnitude)
        routed_out = routed_output.to_ttnn if hasattr(routed_output, "to_ttnn") else routed_output
        n_rs = self.device.shape[1]
        if n_rs > 1:
            routed_out = ttnn.mul(routed_out, 1.0 / float(n_rs))
        routed_output = ttnn.experimental.reduce_scatter_minimal_async(
            routed_out,
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

        # 5. Gated shared expert: sigmoid(gate(x_full)) * shared_expert(residual)
        shared_output = self.shared_experts(residual)
        gate_raw = self.shared_expert_gate(x_full)
        gate_raw = gate_raw.to_ttnn if hasattr(gate_raw, "to_ttnn") else gate_raw
        gate_val = ttnn.sigmoid(gate_raw)
        shared_raw = shared_output.to_ttnn if hasattr(shared_output, "to_ttnn") else shared_output
        gated_shared = ttnn.mul(gate_val, shared_raw)

        output = ttnn.add(routed_output, gated_shared)
        output = ttnn.squeeze(output, 1)
        return output

    @staticmethod
    def _adapt_config(gate, experts):
        class AdaptedConfig:
            pass

        config = AdaptedConfig()
        config.hidden_size = getattr(gate, "hidden_dim", gate.weight.shape[1])
        config.moe_intermediate_size = (
            getattr(experts, "intermediate_dim", None)
            or getattr(experts, "config", type("C", (), {"moe_intermediate_size": None})()).moe_intermediate_size
        )
        config.num_experts_per_tok = getattr(gate, "top_k", None)
        config.n_routed_experts = getattr(gate, "num_experts", gate.weight.shape[0])
        config.n_group = 1
        config.topk_group = 1
        config.routed_scaling_factor = 1.0
        config.norm_topk_prob = getattr(gate, "norm_topk_prob", False)
        config.hidden_act = getattr(experts, "act_fn", None) or "silu"
        if config.num_experts_per_tok is None:
            config.num_experts_per_tok = 8  # fallback
        return config

    @staticmethod
    def _wrap_experts(qwen3_experts, config):
        class ExpertsWrapper:
            pass

        w = ExpertsWrapper()
        w.config = config
        w.gate_up_proj = qwen3_experts.gate_up_proj
        w.down_proj = qwen3_experts.down_proj
        return w


def _consolidate_talker_experts_from_module_list(experts_module_list, config):
    """Build gate_up_proj and down_proj from HF talker experts.

    Older HF: ``experts`` is a ``ModuleList`` of per-expert ``Qwen3OmniMoeTalkerTextMLP``.
    Newer HF: ``experts`` is ``Qwen3OmniMoeTalkerTextExperts`` with stacked
    ``gate_up_proj`` / ``down_proj`` parameters (no ``len()`` / no per-expert modules).
    """
    if hasattr(experts_module_list, "gate_up_proj") and hasattr(experts_module_list, "down_proj"):
        consolidated = type("ConsolidatedExperts", (), {})()
        gu = experts_module_list.gate_up_proj
        dp = experts_module_list.down_proj
        consolidated.gate_up_proj = gu.data if isinstance(gu, torch.nn.Parameter) else gu
        consolidated.down_proj = dp.data if isinstance(dp, torch.nn.Parameter) else dp
        consolidated.config = config
        return consolidated

    num_experts = len(experts_module_list)
    interm = config.moe_intermediate_size
    hidden = config.hidden_size
    gate_up_proj = torch.empty(num_experts, 2 * interm, hidden, dtype=experts_module_list[0].gate_proj.weight.dtype)
    down_proj = torch.empty(num_experts, hidden, interm, dtype=experts_module_list[0].down_proj.weight.dtype)
    for i in range(num_experts):
        gate_up_proj[i] = torch.cat(
            [experts_module_list[i].gate_proj.weight, experts_module_list[i].up_proj.weight], dim=0
        )
        down_proj[i] = experts_module_list[i].down_proj.weight
    consolidated = type("ConsolidatedExperts", (), {})()
    consolidated.gate_up_proj = gate_up_proj
    consolidated.down_proj = down_proj
    consolidated.config = config
    return consolidated


class Qwen3OmniMoeTalkerTextExpertsTTNN(TTNNExperts):
    """
    TTNN experts for Qwen3-Omni talker MoE using sparse_matmul +
    all-to-all dispatch/combine (inherits TTNNExperts infrastructure).
    """

    @classmethod
    def from_torch(cls, torch_experts, config):
        # Accept either consolidated (gate_up_proj, down_proj) or HF ModuleList of MLPs.
        if hasattr(torch_experts, "gate_up_proj") and hasattr(torch_experts, "down_proj"):
            consolidated = torch_experts
        else:
            consolidated = _consolidate_talker_experts_from_module_list(torch_experts, config)
        module = cls(config)
        module._fallback_torch_layer = consolidated
        intermediate = config.moe_intermediate_size
        module.torch_w1_proj = consolidated.gate_up_proj.data[:, :intermediate, :].permute(0, 2, 1).contiguous()
        module.torch_w3_proj = consolidated.gate_up_proj.data[:, intermediate:, :].permute(0, 2, 1).contiguous()
        module.torch_w2_proj = consolidated.down_proj.data.permute(0, 2, 1).contiguous()
        return module

    def move_weights_to_device_impl(self):
        """Override to use fitted grid configs for Qwen3 dimensions."""
        self.num_experts_per_device = self._get_num_experts_per_device(self.config, self.device)
        self.num_devices = self.device.get_num_devices()
        self.num_dispatch_devices = self.device.shape[1]

        self.tt_w1_proj = ttnn.to_device(self.tt_w1_proj, self.device)
        self.tt_w3_proj = ttnn.to_device(self.tt_w3_proj, self.device)
        self.tt_w2_proj = ttnn.to_device(self.tt_w2_proj, self.device)

        # expert_mapping_tensors and remap_topk_mask are kept for the module lifetime;
        # they are used on every forward and are not explicitly deallocated.
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

        self.remap_topk_mask = ttnn.from_torch(
            torch.ones((1, self.num_dispatch_devices, 1, self.num_experts), dtype=torch.bfloat16),
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        hidden_tiles = self.hidden_size // ttnn.TILE_SIZE
        intermediate_tiles = self.intermediate_size // ttnn.TILE_SIZE

        self._gate_up_program_config = _make_fitted_sparse_matmul_program_config(
            device=self.device,
            out_features=int(self.intermediate_size),
            in0_block_w=min(4, hidden_tiles),
            per_core_M=1,
        )
        self._down_program_config = _make_fitted_sparse_matmul_program_config(
            device=self.device,
            out_features=int(self.hidden_size),
            in0_block_w=min(4, intermediate_tiles),
            per_core_M=1,
        )
        self._expert_compute_cfg = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )


class Qwen3OmniMoeTalkerTextMLPTTNN(TTNNModule):
    """
    TTNN SwiGLU MLP for Qwen3 shared expert: silu(gate(x)) * up(x) -> down.

    Uses TTNNLinearSilu for gate_proj and TTNNLinearIColShardedWRowSharded
    for up_proj / down_proj so the entire MLP runs on device.
    """

    @classmethod
    def from_torch(cls, torch_mlp, config=None):
        tt_module = cls()
        tt_module._fallback_torch_layer = torch_mlp
        tt_module.config = config
        tt_module.gate_proj = TTNNLinearSilu.from_torch(
            torch_mlp.gate_proj,
            linear_class=TTNNLinearIColShardedWRowSharded,
        )
        tt_module.up_proj = TTNNLinearIColShardedWRowSharded.from_torch(torch_mlp.up_proj)
        tt_module.down_proj = TTNNLinearIColShardedWRowSharded.from_torch(torch_mlp.down_proj)
        return tt_module

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        if x is None:
            raise ValueError("Qwen3OmniMoeTalkerTextMLPTTNN.forward: input x is None.")
        if not hasattr(x, "shape") or len(x.shape) < 2:
            raise ValueError(
                f"Qwen3OmniMoeTalkerTextMLPTTNN.forward: input must be at least 2D; got shape {getattr(x, 'shape', None)}."
            )
        if self.config is not None and hasattr(self.config, "hidden_size"):
            # Validate last dim when config is available (sharded: last dim may be hidden_size // n_devices)
            last_dim = int(x.shape[-1])
            if last_dim <= 0:
                raise ValueError(
                    f"Qwen3OmniMoeTalkerTextMLPTTNN.forward: input last dim must be positive; got {last_dim}."
                )
        x_gate = self.gate_proj(x)
        x_up = self.up_proj(x)
        a = _to_ttnn_raw(x_gate)
        b = _to_ttnn_raw(x_up)
        x = ttnn.mul(a, b)
        x = self.down_proj(x)
        return x


class TTNNQwen3TalkerMoE(TTNNQwen3MoE):
    """
    TTNN MoE for Qwen3-Omni talker text sparse MoE.

    Reuses TTNNQwen3MoE (softmax routing + gated shared expert). from_torch adapts
    Qwen3OmniMoeTalkerTextSparseMoeBlock: consolidates ModuleList experts into
    gate_up_proj/down_proj and builds the interface TTNNQwen3MoE.from_torch expects.
    """

    @classmethod
    def from_torch(cls, talker_block):
        """Create from a PyTorch Qwen3OmniMoeTalkerTextSparseMoeBlock."""
        qwen_config = getattr(talker_block.shared_expert, "config", None)

        class _Cfg:
            pass

        cfg = _Cfg()
        if qwen_config is not None:
            cfg.hidden_size = qwen_config.hidden_size
            cfg.moe_intermediate_size = qwen_config.moe_intermediate_size
            cfg.num_experts_per_tok = qwen_config.num_experts_per_tok
            cfg.n_routed_experts = qwen_config.num_experts
            cfg.norm_topk_prob = getattr(qwen_config, "norm_topk_prob", False)
            cfg.hidden_act = getattr(qwen_config, "hidden_act", "silu")
        else:
            cfg.hidden_size = talker_block.gate.weight.shape[1]
            ex = talker_block.experts
            if hasattr(ex, "gate_up_proj") and hasattr(ex, "intermediate_dim"):
                cfg.moe_intermediate_size = ex.intermediate_dim
                cfg.n_routed_experts = getattr(ex, "num_experts", ex.gate_up_proj.shape[0])
            elif hasattr(ex, "gate_up_proj"):
                cfg.moe_intermediate_size = ex.gate_up_proj.shape[1] // 2
                cfg.n_routed_experts = ex.gate_up_proj.shape[0]
            else:
                cfg.moe_intermediate_size = getattr(ex[0], "intermediate_size", None) or (
                    ex[0].gate_proj.weight.shape[0] if len(ex) else 0
                )
                cfg.n_routed_experts = talker_block.gate.weight.shape[0]
            cfg.num_experts_per_tok = 8
            cfg.norm_topk_prob = False
            cfg.hidden_act = "silu"

        # Consolidated experts (gate_up_proj, down_proj, config) for TTNNQwen3MoE._wrap_experts
        consolidated = _consolidate_talker_experts_from_module_list(talker_block.experts, cfg)
        consolidated.intermediate_dim = cfg.moe_intermediate_size

        # Gate adapter: TTNNQwen3MoE._adapt_config expects .weight, .hidden_dim, .num_experts, .top_k, .norm_topk_prob
        class _GateAdapter:
            pass

        gate_adapter = _GateAdapter()
        gate_adapter.weight = talker_block.gate.weight
        gate_adapter.hidden_dim = cfg.hidden_size
        gate_adapter.num_experts = cfg.n_routed_experts
        gate_adapter.top_k = cfg.num_experts_per_tok
        gate_adapter.norm_topk_prob = cfg.norm_topk_prob

        # Fake torch_moe matching TTNNQwen3MoE.from_torch(torch_moe) interface
        class _TalkerMoEAdapter:
            pass

        adapter = _TalkerMoEAdapter()
        adapter.gate = gate_adapter
        adapter.experts = consolidated
        adapter.shared_expert = talker_block.shared_expert
        adapter.shared_expert_gate = talker_block.shared_expert_gate

        module = super().from_torch(adapter)
        module._fallback_torch_layer = talker_block
        # Use Talker experts with fitted sparse matmul configs for this architecture
        module.experts = Qwen3OmniMoeTalkerTextExpertsTTNN.from_torch(consolidated, cfg)
        return module


def _thinker_experts_adapter(thinker_mlp):
    """Adapt HF thinker experts for TTNNExperts (needs config + gate_up/down tensors)."""
    hf_experts = thinker_mlp.experts
    cfg = getattr(hf_experts, "config", None)
    if cfg is None:
        cfg = type("ThinkerExpertsConfig", (), {})()
    cfg.hidden_size = getattr(cfg, "hidden_size", hf_experts.gate_up_proj.shape[2])
    cfg.moe_intermediate_size = getattr(cfg, "moe_intermediate_size", hf_experts.gate_up_proj.shape[1] // 2)
    cfg.n_routed_experts = getattr(cfg, "n_routed_experts", hf_experts.gate_up_proj.shape[0])
    cfg.num_experts_per_tok = getattr(cfg, "num_experts_per_tok", None) or getattr(thinker_mlp.gate, "top_k", 8)

    adapter = type("ThinkerExpertsAdapter", (), {})()
    adapter.gate_up_proj = hf_experts.gate_up_proj
    adapter.down_proj = hf_experts.down_proj
    adapter.config = cfg
    return adapter


class TTNNQwen3OmniMoeThinkerTextSparseMoeBlock(TTNNModule):
    """
    TTNN MoE for Qwen3-Omni thinker text sparse MoE block.

    Wraps thinker.model.layers[i].mlp: gate (routing) stays on torch; experts run via
    TT implementation Glm4MoeNaiveMoe (TTNNGlm4MoeNaiveMoe). Compatible with
    Qwen3OmniMoeThinkerTextSparseMoeBlock from HuggingFace.
    """

    @classmethod
    def from_torch(cls, thinker_mlp):
        """Create from a PyTorch thinker mlp (Qwen3OmniMoeThinkerTextSparseMoeBlock)."""
        module = cls()
        module._fallback_torch_layer = thinker_mlp
        module.gate = thinker_mlp.gate
        experts_for_tt = _thinker_experts_adapter(thinker_mlp)
        module.experts = TTNNExperts.from_torch(experts_for_tt)
        return module

    def preprocess_weights_impl(self):
        self.experts.preprocess_weights()

    def move_weights_to_device_impl(self):
        self.experts.move_weights_to_device()

    @property
    def _is_distributed(self):
        return (
            self.device_state is not None
            and hasattr(self.device_state, "ccl_manager")
            and self.device_state.ccl_manager is not None
        )

    def _maybe_all_gather(self, tensor):
        if not self._is_distributed:
            return tensor
        return ttnn.experimental.all_gather_async(
            tensor,
            dim=-1,
            multi_device_global_semaphore=self.device_state.ccl_manager.get_and_cycle_ag_semaphore_handles(1),
            barrier_semaphore=self.device_state.ccl_manager.get_and_cycle_barrier_semaphore_handle(1),
            num_links=1,
            topology=ttnn.Topology.Linear,
        )

    @run_on_devices(DeviceArch.T3K)
    def forward(self, hidden_states):
        """Run gate on torch, experts on TT; return torch tensor for downstream layers."""
        hidden_states_torch = _to_torch_any(hidden_states)
        x_flat = hidden_states_torch.reshape(-1, hidden_states_torch.shape[-1])
        with torch.no_grad():
            _, routing_weights, selected_experts = self.gate(x_flat)
        hidden_states_tt = _to_ttnn_raw(hidden_states)
        hidden_states_tt = self._maybe_all_gather(hidden_states_tt)
        if len(hidden_states_tt.shape) == 3:
            b, s, h = (int(hidden_states_tt.shape[0]), int(hidden_states_tt.shape[1]), int(hidden_states_tt.shape[2]))
            hidden_states_tt = ttnn.reshape(hidden_states_tt, ttnn.Shape((b, 1, s, h)))
        else:
            b, s, h = (
                int(hidden_states_tt.shape[0]),
                int(hidden_states_tt.shape[2]),
                int(hidden_states_tt.shape[3]),
            )

        mesh_mapper = ttnn.ReplicateTensorToMesh(self.device) if self.device.get_num_devices() > 1 else None
        topk_idx_tt = ttnn.from_torch(
            selected_experts.to(torch.int64),
            device=self.device,
            mesh_mapper=mesh_mapper,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        topk_w_tt = ttnn.from_torch(
            routing_weights.to(torch.bfloat16),
            device=self.device,
            mesh_mapper=mesh_mapper,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        # Call forward directly to avoid wrapping outputs into TorchTTNNTensor.
        expert_out = self.experts.forward(hidden_states_tt, topk_idx_tt, topk_w_tt)
        # Be defensive: forward may still return TorchTTNNTensor depending on internal ops.
        try:
            from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

            if isinstance(expert_out, TorchTTNNTensor):
                expert_out = expert_out.to_ttnn
        except Exception:
            pass
        expert_out = _to_ttnn_raw(expert_out)
        return ttnn.reshape(expert_out, ttnn.Shape((b, s, h)))


class TTNNQwen3OmniThinkerNaiveMoE(TTNNModule):
    """
    TTNN MoE for Qwen3-Omni thinker with per-expert computation on TT device.

    Gate (softmax routing) runs on torch.  Each expert's gate/up/down projections
    are stored as raw ttnn weight tensors and executed via ``ttnn.linear`` /
    ``ttnn.silu`` / ``ttnn.mul`` on the accelerator.  Uses a naive per-expert
    loop so input/output shapes are preserved (no all-gather / reduce-scatter).
    """

    @classmethod
    def from_torch(cls, thinker_mlp):
        module = cls()
        module._fallback_torch_layer = thinker_mlp
        module.gate = thinker_mlp.gate
        hf_experts = thinker_mlp.experts
        E, two_I, H = hf_experts.gate_up_proj.shape
        I = two_I // 2
        module.num_experts = E
        module.intermediate_size = I
        module._gate_w_torch = [hf_experts.gate_up_proj.data[i, :I, :] for i in range(E)]
        module._up_w_torch = [hf_experts.gate_up_proj.data[i, I:, :] for i in range(E)]
        module._down_w_torch = [hf_experts.down_proj.data[i] for i in range(E)]
        return module

    def preprocess_weights_impl(self):
        self._gate_tt_host = [
            preprocess_linear_weight(w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT) for w in self._gate_w_torch
        ]
        self._up_tt_host = [
            preprocess_linear_weight(w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT) for w in self._up_w_torch
        ]
        self._down_tt_host = [
            preprocess_linear_weight(w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT) for w in self._down_w_torch
        ]
        del self._gate_w_torch
        del self._up_w_torch
        del self._down_w_torch

    def move_weights_to_device_impl(self):
        self._gate_tt = [ttnn.to_device(w, self.device) for w in self._gate_tt_host]
        self._up_tt = [ttnn.to_device(w, self.device) for w in self._up_tt_host]
        self._down_tt = [ttnn.to_device(w, self.device) for w in self._down_tt_host]

    def deallocate_weights_impl(self):
        for w in self._gate_tt:
            ttnn.deallocate(w)
        for w in self._up_tt:
            ttnn.deallocate(w)
        for w in self._down_tt:
            ttnn.deallocate(w)
        self._weights_on_device = False

    @deallocate_weights_after
    @run_on_devices(DeviceArch.T3K)
    def forward(self, hidden_states):
        hidden_states_torch = _to_torch_any(hidden_states)
        orig_shape = hidden_states_torch.shape
        x_flat = hidden_states_torch.reshape(-1, hidden_states_torch.shape[-1])

        with torch.no_grad():
            _, routing_weights, selected_experts = self.gate(x_flat)

        num_tokens = x_flat.shape[0]
        final_hidden_states = torch.zeros_like(x_flat)
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        expert_hit = (expert_mask.sum(dim=(-1, -2)) > 0).nonzero()

        is_mesh = self.device.get_num_devices() > 1
        mesh_mapper = ttnn.ReplicateTensorToMesh(self.device) if is_mesh else None
        mesh_composer = ttnn.ConcatMeshToTensor(self.device, dim=0) if is_mesh else None

        for eidx in expert_hit:
            eidx = eidx[0].item()
            if eidx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[eidx])
            current_state = x_flat[token_idx]
            n_tok = current_state.shape[0]

            x_tt = ttnn.from_torch(
                current_state.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                mesh_mapper=mesh_mapper,
            )

            gate_tt = ttnn.linear(x_tt, self._gate_tt[eidx], memory_config=ttnn.DRAM_MEMORY_CONFIG)
            gate_tt = ttnn.silu(gate_tt)
            up_tt = ttnn.linear(x_tt, self._up_tt[eidx], memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(x_tt)

            hidden_tt = ttnn.mul(gate_tt, up_tt)
            ttnn.deallocate(gate_tt)
            ttnn.deallocate(up_tt)

            out_tt = ttnn.linear(hidden_tt, self._down_tt[eidx], memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(hidden_tt)

            out_torch = ttnn.to_torch(out_tt, mesh_composer=mesh_composer).to(torch.float32)
            ttnn.deallocate(out_tt)
            out_torch = out_torch.view(-1, out_torch.shape[-1])[:n_tok]

            current_hidden_states = out_torch * routing_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states.view(*orig_shape)
