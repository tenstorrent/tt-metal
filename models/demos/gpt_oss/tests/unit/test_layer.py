from typing import Callable, Optional

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import ttnn
from models.utility_functions import comp_pcc

from ...reference.configuration_gpt_oss import GptOssConfig
from ...reference.hf_utils import get_state_dict
from ...reference.modeling_gpt_oss import GptOssRotaryEmbedding
from ...tt.ccl import CCLManager
from ...tt.layer import DecoderLayer
from ...tt.model_config import ModelArgs
from ...tt.rope import ApplyRotaryPosEmb
from ...utils.general_utils import get_decode_mask

# ModelArgs will be instantiated inside test functions to avoid import-time loading


class ReferenceRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


class ReferenceMLP(nn.Module):
    """Reference MLP implementation combining TopK router and Experts"""

    def __init__(self, config):
        super().__init__()
        self.router = ReferenceTopKRouter(config)
        self.experts = ReferenceExperts(config)

    def forward(self, hidden_states):
        router_scores, router_indices = self.router(hidden_states)
        routed_out = self.experts(hidden_states, router_indices=router_indices, routing_weights=router_scores)
        return routed_out, router_scores


class ReferenceTopKRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.randn(self.num_experts, self.hidden_dim))
        self.bias = nn.Parameter(torch.randn(self.num_experts))

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = torch.nn.functional.linear(hidden_states, self.weight, self.bias)
        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)
        router_top_value = torch.nn.functional.softmax(router_top_value, dim=1, dtype=router_top_value.dtype)
        router_scores = torch.zeros_like(router_logits).scatter_(1, router_indices, router_top_value)
        return router_scores, router_indices


class ReferenceExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.intermediate_size = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.hidden_size = config.hidden_size
        self.expert_dim = self.intermediate_size
        self.gate_up_proj = nn.Parameter(torch.randn(self.num_experts, self.hidden_size, 2 * self.expert_dim))
        self.gate_up_proj_bias = nn.Parameter(torch.randn(self.num_experts, 2 * self.expert_dim))
        self.down_proj = nn.Parameter(torch.randn((self.num_experts, self.expert_dim, self.hidden_size)))
        self.down_proj_bias = nn.Parameter(torch.randn(self.num_experts, self.hidden_size))
        self.alpha = 1.702
        self.limit = 7.0

    def forward(self, hidden_states: torch.Tensor, router_indices=None, routing_weights=None) -> torch.Tensor:
        """Reference experts implementation - inference mode only"""
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)  # (num_tokens, hidden_size)
        num_experts = routing_weights.shape[1]

        hidden_states = hidden_states.repeat(num_experts, 1)
        hidden_states = hidden_states.view(num_experts, -1, self.hidden_size)
        gate_up = torch.bmm(hidden_states, self.gate_up_proj) + self.gate_up_proj_bias[..., None, :]
        gate, up = gate_up[..., ::2], gate_up[..., 1::2]
        gate = gate.clamp(min=None, max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        glu = gate * torch.sigmoid(gate * self.alpha)
        next_states = torch.bmm(((up + 1) * glu), self.down_proj)
        next_states = next_states + self.down_proj_bias[..., None, :]
        next_states = next_states.view(num_experts, batch_size, -1, self.hidden_size)
        next_states = next_states * routing_weights.transpose(0, 1).view(num_experts, batch_size, -1)[..., None]
        next_states = next_states.sum(dim=0)
        return next_states


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    first_half, second_half = torch.chunk(x, 2, dim=-1)
    first_ = first_half * cos - second_half * sin
    second_ = second_half * cos + first_half * sin
    return torch.cat((first_, second_), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = _apply_rotary_emb(q, cos, sin)
    k_embed = _apply_rotary_emb(k, cos, sin)
    return q_embed, k_embed


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    sinks = module.sinks.reshape(1, -1, 1, 1).expand(query.shape[0], -1, query.shape[-2], -1)
    combined_logits = torch.cat([attn_weights, sinks], dim=-1)

    # This was not in the original implementation and slightly affect results; it prevents overflow in BF16/FP16
    # when training with bsz>1 we clamp max values.

    combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
    probs = F.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)
    scores = probs[..., :-1]  # we drop the sink here
    attn_weights = nn.functional.dropout(scores, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class ReferenceAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = getattr(config, "attention_dropout", 0.0)
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=getattr(config, "attention_bias", False),
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=getattr(config, "attention_bias", False),
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=getattr(config, "attention_bias", False),
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=getattr(config, "attention_bias", False),
        )

        # Initialize sinks parameter
        self.sinks = nn.Parameter(torch.zeros(config.num_attention_heads))

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        attention_interface: Callable = eager_attention_forward
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class ReferenceDecoderLayer(nn.Module):
    """Reference decoder layer implementation that matches the TT implementation"""

    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = ReferenceAttention(config, layer_idx)
        self.mlp = ReferenceMLP(config)
        self.input_layernorm = ReferenceRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = ReferenceRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states,
        attention_mask,
        position_embeddings,
        **kwargs,
    ):
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected (MLP) part
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, router_scores = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


@pytest.mark.parametrize(
    "num_experts, experts_per_token, intermediate_size, hidden_size",
    [
        # (32, 4, 2880, 2880),  # 20B config
        (128, 4, 2880, 2880),  # 120B config
    ],
    ids=[
        # "gpt20B",
        "gpt120B",
    ],
)
@pytest.mark.parametrize("batch_size", (1,))
@pytest.mark.parametrize("seq_len", [1, 32, 64, 128, 512, 1024], ids=["s1_", "s32", "s64", "s128", "s512", "s1024"])
@pytest.mark.parametrize("layer_idx", [0])
@pytest.mark.parametrize(
    "use_real_weights",
    [
        True,
    ],
    ids=[
        "real",
    ],
)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True)
def test_decoder_layer(
    mesh_device,
    num_experts,
    experts_per_token,
    intermediate_size,
    hidden_size,
    seq_len,
    batch_size,
    layer_idx,
    use_real_weights,
    reset_seeds,
):
    mesh_device = mesh_device.create_submesh(ttnn.MeshShape((1, 8)))
    print("MESH DEVICE!", mesh_device)
    print("MESH SHAPE!", mesh_device.shape)

    # Get paths from ModelArgs to avoid code duplication
    model_args = ModelArgs(mesh_device=None, dummy_weights=True)  # dummy_weights=True to avoid loading actual weights
    gpt_dir = model_args.model_path
    local_weights_path = gpt_dir
    dtype = ttnn.bfloat8_b  # Always use bfp8

    # Create configuration
    config = GptOssConfig(
        num_local_experts=num_experts,
        intermediate_size=intermediate_size,
        hidden_size=hidden_size,
        num_experts_per_tok=experts_per_token,
    )

    sliding_window = 0
    if layer_idx % 2 == 0:
        sliding_window = config.sliding_window

    cur_seq_len = seq_len
    position_ids = torch.arange(seq_len).unsqueeze(0)

    # Create input tensors
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    mask = torch.triu(torch.full((1, 1, cur_seq_len, cur_seq_len), -float("inf")), diagonal=1)
    if sliding_window > 0:
        mask += torch.tril(torch.full((1, 1, cur_seq_len, cur_seq_len), -float("inf")), diagonal=-sliding_window)

    RopeEmbeddings = GptOssRotaryEmbedding(config)
    cos, sin = RopeEmbeddings(hidden_states, position_ids)
    position_embeddings = (cos, sin)

    # Create input tensors
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)

    tt_mask_in = mask
    if seq_len == 1:  # decode
        tt_mask_in = get_decode_mask(position_ids[0].item(), sliding_window)
        tt_mask_in = tt_mask_in.repeat(1, config.num_attention_heads // mesh_device.shape[1], 1, 1).transpose(1, 2)

    # Convert to TTNN tensors
    tt_hidden_states = ttnn.from_torch(hidden_states, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_mask = ttnn.from_torch(tt_mask_in, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_cos = ttnn.from_torch(cos.unsqueeze(-2), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_sin = ttnn.from_torch(sin.unsqueeze(-2), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_position_idx = ttnn.from_torch(position_ids, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32)

    apply_rope = ApplyRotaryPosEmb(config)
    rope_stuff = (apply_rope, tt_cos, tt_sin)

    # Create models
    reference_model = ReferenceDecoderLayer(config)

    if use_real_weights:
        # Load real weights for the layer
        layer_state_dict = get_state_dict(local_weights_path, "model.layers.0.", dtype=torch.float32)
        # Load weights into reference model
        reference_model.load_state_dict(layer_state_dict, strict=False)

    # Get state dict for TT model
    reference_state_dict = reference_model.state_dict()

    # Create TT layer state dict
    # Initialize TT model with dummy ccl_manager
    ccl_manager = CCLManager(mesh_device)  # Not needed for this test
    tt_model = DecoderLayer(
        mesh_device,
        config,
        reference_state_dict,
        layer_idx,
        ccl_manager,
        dtype=dtype,
        tensor_cache_path=model_args.weight_cache_path(dtype),
    )

    # Run forward passes
    reference_output = reference_model(hidden_states, mask, position_embeddings)

    # For TT model, we need to pass the required arguments even though they're not used
    tt_output = tt_model(
        hidden_states=tt_hidden_states,
        attention_mask=tt_mask,
        position_embeddings=rope_stuff,
        position_idx=tt_position_idx,
    )

    tt_output_tensors = ttnn.get_device_tensors(tt_output)
    # Convert TTNN output to torch
    for i in range(len(tt_output_tensors)):
        tt_output = ttnn.to_torch(tt_output_tensors[i])

        # Compare outputs
        pcc_threshold = 0.95  # TODO: Investigate
        passing, output = comp_pcc(reference_output, tt_output, pcc=pcc_threshold)
        mse = torch.nn.functional.mse_loss(reference_output, tt_output)

        # Calculate relative error metrics
        ref_variance = torch.var(reference_output)
        ref_mean_abs = torch.mean(torch.abs(reference_output))
        ref_std = torch.std(reference_output)

        relative_mse_to_variance = mse / ref_variance if ref_variance > 0 else float("inf")
        relative_mse_to_scale = mse / (ref_mean_abs**2) if ref_mean_abs > 0 else float("inf")
        snr_db = 10 * torch.log10(ref_variance / mse) if mse > 0 else float("inf")

        print(f"Decoder layer output: {output}")
        print(f"MSE: {mse:.6e}")
        print(f"Reference variance: {ref_variance:.6e}, std: {ref_std:.6e}, mean_abs: {ref_mean_abs:.6e}")
        print(f"Relative MSE to variance: {relative_mse_to_variance:.6e} ({relative_mse_to_variance*100:.4f}%)")
        print(f"Relative MSE to scale²: {relative_mse_to_scale:.6e} ({relative_mse_to_scale*100:.4f}%)")
        print(f"Signal-to-Noise Ratio: {snr_db:.2f} dB")
        print(f"Reference output range: [{torch.min(reference_output):.6e}, {torch.max(reference_output):.6e}]")
        print(f"TT output range: [{torch.min(tt_output):.6e}, {torch.max(tt_output):.6e}]")

        assert passing, "Decoder layer output mismatch"
