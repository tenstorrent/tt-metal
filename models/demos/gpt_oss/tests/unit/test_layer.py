import os

import pytest
import torch
import torch.nn as nn

import ttnn
from models.utility_functions import comp_pcc

from ...reference.configuration_gpt_oss import GptOssConfig
from ...reference.hf_utils import get_state_dict
from ...tt.ccl import CCLManager
from ...tt.layer import DecoderLayer

local_weights_path = os.environ.get("GPT_OSS_WEIGHTS_PATH", "/proj_sw/user_dev/gpt-oss/gpt-oss-20b-BF16")


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


class ReferenceDecoderLayer(nn.Module):
    """Reference decoder layer implementation that matches the TT implementation"""

    def __init__(self, config):
        super().__init__()
        self.input_layernorm = ReferenceRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = ReferenceRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = ReferenceMLP(config)

    def forward(self, hidden_states):
        # Skip attention (not implemented yet)
        # Fully Connected (MLP) part only
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, router_scores = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


@pytest.mark.parametrize(
    "num_experts, experts_per_token, intermediate_size, hidden_size",
    [
        (32, 4, 2880, 2880),  # 20B config
        (128, 4, 2880, 2880),  # 120B config
    ],
    ids=["gpt20B", "gpt120B"],
)
@pytest.mark.parametrize("batch_size", (1,))
@pytest.mark.parametrize("seq_len", [1, 32, 64, 128, 512, 1024], ids=["s1_", "s32", "s64", "s128", "s512", "s1024"])
@pytest.mark.parametrize("use_real_weights", [True, False], ids=["real", "random"])
@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_decoder_layer(
    mesh_device,
    num_experts,
    experts_per_token,
    intermediate_size,
    hidden_size,
    seq_len,
    batch_size,
    use_real_weights,
    reset_seeds,
):
    # Create configuration
    config = GptOssConfig(
        num_local_experts=num_experts,
        intermediate_size=intermediate_size,
        hidden_size=hidden_size,
        num_experts_per_tok=experts_per_token,
        rms_norm_eps=1e-6,
    )

    # Create input tensors
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)

    # Convert to TTNN tensors
    tt_hidden_states = ttnn.from_torch(hidden_states, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    # Create models
    reference_model = ReferenceDecoderLayer(config)

    if use_real_weights:
        # Load real weights for the layer
        layer_state_dict = get_state_dict(local_weights_path, "model.layers.0.", dtype=torch.float32)
        # Load weights into reference model
        reference_model.load_state_dict(layer_state_dict)

    # Get state dict for TT model
    reference_state_dict = reference_model.state_dict()

    # Create TT layer state dict
    # Initialize TT model with dummy ccl_manager
    ccl_manager = CCLManager(mesh_device)  # Not needed for this test
    tt_model = DecoderLayer(mesh_device, config, reference_state_dict, ccl_manager)

    # Run forward passes
    reference_output = reference_model(hidden_states)

    # For TT model, we need to pass the required arguments even though they're not used
    tt_output = tt_model(
        hidden_states=tt_hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
    )

    tt_output_tensors = ttnn.get_device_tensors(tt_output)
    # Convert TTNN output to torch
    for i in range(len(tt_output_tensors)):
        tt_output = ttnn.to_torch(tt_output_tensors[i])

        # Compare outputs
        passing, output = comp_pcc(reference_output, tt_output, pcc=0.99)
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
