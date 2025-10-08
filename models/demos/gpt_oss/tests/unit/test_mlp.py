import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import ttnn
from models.utility_functions import comp_pcc

from ...reference.configuration_gpt_oss import GptOssConfig
from ...reference.hf_utils import get_state_dict
from ...tt.ccl import CCLManager
from ...tt.mlp import MLP
from ...tt.model_config import ModelArgs

# ModelArgs will be instantiated inside test functions to avoid import-time loading


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
        router_logits = F.linear(hidden_states, self.weight, self.bias)  # (seq_len, num_experts)
        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)  # (seq_len, top_k)
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


@pytest.mark.parametrize(
    "num_experts, experts_per_token, intermediate_size, hidden_size",
    [
        # (32, 4, 2880, 2880),  # 20B config
        (128, 4, 2880, 2880),  # 120B config
    ],
    ids=[
        "gpt120B",
    ],
)
@pytest.mark.parametrize("batch_size", (1,))
@pytest.mark.parametrize("seq_len", [1, 32, 64, 128, 512, 1024], ids=["s1_", "s32", "s64", "s128", "s512", "s1024"])
@pytest.mark.parametrize(
    "use_real_weights",
    [
        True,
    ],
    ids=[
        "real",
    ],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
    indirect=True,
)
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
    ],
    ids=[
        "bf16",
    ],
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        (4, 8),
    ],
    indirect=True,
)
def test_mlp(
    mesh_device,
    num_experts,
    experts_per_token,
    intermediate_size,
    hidden_size,
    seq_len,
    batch_size,
    use_real_weights,
    dtype,
    reset_seeds,
):
    mesh_device = mesh_device.create_submesh(ttnn.MeshShape((1, 8)))
    print(mesh_device.shape)

    # Get paths from ModelArgs to avoid code duplication
    model_args = ModelArgs(mesh_device=None, dummy_weights=True)  # dummy_weights=True to avoid loading actual weights
    gpt_dir = model_args.model_path
    local_weights_path = gpt_dir

    # Create configuration
    config = GptOssConfig(
        num_local_experts=num_experts,
        intermediate_size=intermediate_size,
        hidden_size=hidden_size,
        num_experts_per_tok=experts_per_token,
    )

    # Create input tensors
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)

    # Convert to TTNN tensors
    tt_hidden_states = ttnn.from_torch(hidden_states, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    # Create models
    reference_model = ReferenceMLP(config)

    if use_real_weights:
        state_dict = get_state_dict(local_weights_path, "model.layers.0.mlp.", dtype=torch.float32)
        reference_model.load_state_dict(state_dict, strict=True)

    state_dict = reference_model.state_dict()

    ccl_manager = CCLManager(mesh_device)
    tt_model = MLP(
        mesh_device, config, state_dict, ccl_manager, dtype=dtype, tensor_cache_path=model_args.weight_cache_path(dtype)
    )

    # Run tt forward pass first to get the routing scores to use in reference execution
    tt_output, tt_router_scores = tt_model(tt_hidden_states)
    tt_output_tensors = ttnn.get_device_tensors(tt_output)
    tt_router_scores_tensors = ttnn.get_device_tensors(tt_router_scores)

    for i in range(len(tt_output_tensors)):
        tt_output = ttnn.to_torch(tt_output_tensors[i])
        tt_router_scores = ttnn.to_torch(tt_router_scores_tensors[i])

        # Run reference
        reference_output = reference_model.experts(hidden_states, routing_weights=tt_router_scores)

        # Compare MLP outputs
        passing, output = comp_pcc(reference_output, tt_output, pcc=0.99)
        mse = torch.nn.functional.mse_loss(reference_output, tt_output)

        # Calculate relative error metrics for MLP output
        ref_variance = torch.var(reference_output)
        ref_mean_abs = torch.mean(torch.abs(reference_output))
        ref_std = torch.std(reference_output)

        relative_mse_to_variance = mse / ref_variance if ref_variance > 0 else float("inf")
        relative_mse_to_scale = mse / (ref_mean_abs**2) if ref_mean_abs > 0 else float("inf")
        snr_db = 10 * torch.log10(ref_variance / mse) if mse > 0 else float("inf")

        print(f"MLP output: {output}")
        print(f"MSE: {mse:.6e}")
        print(f"Reference variance: {ref_variance:.6e}, std: {ref_std:.6e}, mean_abs: {ref_mean_abs:.6e}")
        print(f"Relative MSE to variance: {relative_mse_to_variance:.6e} ({relative_mse_to_variance*100:.4f}%)")
        print(f"Relative MSE to scale²: {relative_mse_to_scale:.6e} ({relative_mse_to_scale*100:.4f}%)")
        print(f"Signal-to-Noise Ratio: {snr_db:.2f} dB")
        print(f"Reference output range: [{torch.min(reference_output):.6e}, {torch.max(reference_output):.6e}]")
        print(f"TT output range: [{torch.min(tt_output):.6e}, {torch.max(tt_output):.6e}]")

        assert passing, "MLP output mismatch"

    # # Compare router scores
    # passing_router, output_router = comp_pcc(reference_router_scores, tt_router_scores, pcc=.99)
    # mse_router = torch.nn.functional.mse_loss(reference_router_scores, tt_router_scores)
    # print(f"Router scores: {output_router}")
    # print(f"Router MSE: {mse_router:.6e}")

    # assert passing_router, "Router scores mismatch"


# @pytest.mark.parametrize(
#     "num_experts, experts_per_token, intermediate_size, hidden_size",
#     [
#         (8, 2, 1024, 512),     # small config for testing
#         (32, 4, 2880, 2880),   # 20B config
#     ],
#     ids=["small", "gpt20B"],
# )
# @pytest.mark.parametrize("batch_size", (1,))
# @pytest.mark.parametrize("seq_len", [1, 32, 128], ids=["s1_", "s32", "s128"])
# def test_mlp_routing_consistency(device, num_experts, experts_per_token, intermediate_size, hidden_size, seq_len, batch_size, reset_seeds):
#     """Test that MLP routing is consistent - router scores should sum correctly"""
#     # Create configuration
#     config = GptOssConfig(
#         num_local_experts=num_experts,
#         intermediate_size=intermediate_size,
#         hidden_size=hidden_size,
#         num_experts_per_tok=experts_per_token,
#     )

#     # Create input tensors
#     hidden_states = torch.randn(batch_size, seq_len, hidden_size)

#     # Convert to TTNN tensors
#     tt_hidden_states = ttnn.from_torch(hidden_states, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

#     # Create models
#     reference_model = ReferenceMLP(config)
#     state_dict = reference_model.state_dict()

#     # Create combined state dict for MLP
#     mlp_state_dict = {
#         "router.weight": state_dict["router.weight"],
#         "router.bias": state_dict["router.bias"],
#         "experts.gate_up_proj": state_dict["experts.gate_up_proj"],
#         "experts.gate_up_proj_bias": state_dict["experts.gate_up_proj_bias"],
#         "experts.down_proj": state_dict["experts.down_proj"],
#         "experts.down_proj_bias": state_dict["experts.down_proj_bias"]
#     }

#     tt_model = MLP(config, mlp_state_dict, device)

#     # Run forward passes
#     reference_output, reference_router_scores = reference_model(hidden_states)
#     tt_output, tt_router_scores = tt_model(tt_hidden_states)

#     # Convert TTNN outputs to torch
#     tt_router_scores = ttnn.to_torch(tt_router_scores)

#     # Check routing consistency
#     ref_scores_sum = torch.sum(reference_router_scores, dim=-1)
#     tt_scores_sum = torch.sum(tt_router_scores, dim=-1)

#     # Router scores should sum to 1 (or close to 1 due to numerical precision)
#     print(f"Reference router scores sum range: [{torch.min(ref_scores_sum):.6f}, {torch.max(ref_scores_sum):.6f}]")
#     print(f"TT router scores sum range: [{torch.min(tt_scores_sum):.6f}, {torch.max(tt_scores_sum):.6f}]")

#     # Check that sums are close to 1
#     assert torch.allclose(ref_scores_sum, torch.ones_like(ref_scores_sum), atol=1e-6), "Reference router scores don't sum to 1"
#     assert torch.allclose(tt_scores_sum, torch.ones_like(tt_scores_sum), atol=1e-4), "TT router scores don't sum to 1"

#     # Check that only top-k experts are active per token
#     ref_nonzero = torch.count_nonzero(reference_router_scores, dim=-1)
#     tt_nonzero = torch.count_nonzero(tt_router_scores, dim=-1)

#     print(f"Reference active experts per token: {ref_nonzero.float().mean():.2f}")
#     print(f"TT active experts per token: {tt_nonzero.float().mean():.2f}")
#     print(f"Expected active experts per token: {experts_per_token}")

#     # Should have exactly experts_per_token active experts per token
#     assert torch.all(ref_nonzero == experts_per_token), f"Reference should have {experts_per_token} active experts per token"
#     assert torch.all(tt_nonzero == experts_per_token), f"TT should have {experts_per_token} active experts per token"
