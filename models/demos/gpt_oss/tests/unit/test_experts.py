import itertools

import pytest
import torch
import torch.nn as nn
from loguru import logger

import ttnn
from models.utility_functions import comp_pcc

from ...reference.configuration_gpt_oss import GptOssConfig
from ...tt.ccl import CCLManager
from ...tt.experts import Experts, SparseExperts
from ...tt.model_config import ModelArgs


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

    def forward(self, hidden_states: torch.Tensor, routing_weights) -> torch.Tensor:
        """
        When training it is more efficient to just loop over the experts and compute the output for each expert
        as otherwise the memory would explode.

        For inference we can sacrifice some memory and compute the output for all experts at once. By repeating the inputs.

        Args:
            hidden_states (torch.Tensor): (batch_size, seq_len, hidden_size)
            selected_experts (torch.Tensor): (batch_size * token_num, top_k)
            routing_weights (torch.Tensor): (batch_size * token_num, num_experts)
        Returns:
            torch.Tensor
        """
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
        (32, 4, 2880, 2880),  # 20B config
        #    (128, 4, 2880, 2880),  # 120B config
    ],
    ids=[
        "gpt20B",
    ],
)
@pytest.mark.parametrize("batch_size", (1,))
@pytest.mark.parametrize("seq_len", [1, 32, 64, 128, 512, 1024], ids=["s1_", "s32", "s64", "s128", "s512", "s1024"])
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
def test_experts(
    mesh_device, num_experts, experts_per_token, intermediate_size, hidden_size, seq_len, batch_size, reset_seeds
):
    print("MESH DEVICE!", mesh_device)
    mesh_device = mesh_device.create_submesh(ttnn.MeshShape((1, 2)))
    print("MESH SHAPE!", mesh_device.shape)

    # Get paths from ModelArgs to avoid code duplication
    model_args = ModelArgs(mesh_device=None, dummy_weights=True)  # dummy_weights=True to avoid loading actual weights

    # Create configuration
    config = GptOssConfig(
        num_local_experts=num_experts,
        intermediate_size=intermediate_size,
        hidden_size=hidden_size,
        num_experts_per_tok=experts_per_token,
    )

    # Create input tensors
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    routing_weights = torch.randn(batch_size * seq_len, num_experts)
    # Normalize routing weights to simulate realistic router scores
    routing_weights = torch.nn.functional.softmax(routing_weights, dim=-1)

    # Convert to TTNN tensors
    tt_hidden_states = ttnn.from_torch(hidden_states, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_routing_weights = ttnn.from_torch(
        routing_weights, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )

    # Create models
    reference_model = ReferenceExperts(config)
    state_dict = reference_model.state_dict()
    ccl_manager = CCLManager(mesh_device)
    tt_model = Experts(mesh_device, config, state_dict, ccl_manager, tensor_cache_path=None)

    # Run forward passes
    reference_output = reference_model(hidden_states, routing_weights)
    tt_output = tt_model(tt_hidden_states, tt_routing_weights)

    tt_output_tensors = ttnn.get_device_tensors(tt_output)
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

        print(f"experts_output: {output}")
        print(f"MSE: {mse:.6e}")
        print(f"Reference variance: {ref_variance:.6e}, std: {ref_std:.6e}, mean_abs: {ref_mean_abs:.6e}")
        print(f"Relative MSE to variance: {relative_mse_to_variance:.6e} ({relative_mse_to_variance*100:.4f}%)")
        print(f"Relative MSE to scale²: {relative_mse_to_scale:.6e} ({relative_mse_to_scale*100:.4f}%)")
        print(f"Signal-to-Noise Ratio: {snr_db:.2f} dB")
        print(f"Reference output range: [{torch.min(reference_output):.6e}, {torch.max(reference_output):.6e}]")
        print(f"TT output range: [{torch.min(tt_output):.6e}, {torch.max(tt_output):.6e}]")

        assert passing, "experts output mismatch"


# @pytest.mark.parametrize(
#     "num_experts, intermediate_size, hidden_size",
#     [
#         (8, 2048, 512),      # smaller config for detailed testing
#         (32, 11520, 2880),   # 20B config
#     ],
#     ids=["small", "gpt20B"],
# )
# @pytest.mark.parametrize("seq_len", [1, 32, 128], ids=["s1_", "s32", "s128"])
# def test_experts_with_sparse_routing(device, num_experts, intermediate_size, hidden_size, seq_len, reset_seeds):
#     """Test experts with sparse routing weights (only a few experts active per token)"""
#     # Create configuration
#     config = GptOssConfig(
#         num_local_experts=num_experts,
#         intermediate_size=intermediate_size,
#         hidden_size=hidden_size,
#     )

#     # Create input tensors
#     hidden_states = torch.randn(seq_len, hidden_size)

#     # Create sparse routing weights (simulate topk routing)
#     routing_weights = torch.zeros(seq_len, num_experts)
#     experts_per_token = min(4, num_experts)  # Use top-4 or fewer if num_experts < 4

#     for i in range(seq_len):
#         # Randomly select which experts are active for this token
#         active_experts = torch.randperm(num_experts)[:experts_per_token]
#         weights = torch.rand(experts_per_token)
#         weights = weights / weights.sum()  # Normalize
#         routing_weights[i, active_experts] = weights

#     # Convert to TTNN tensors
#     tt_hidden_states = ttnn.from_torch(hidden_states, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
#     tt_routing_weights = ttnn.from_torch(routing_weights, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

#     # Create models
#     reference_model = ReferenceExperts(config)
#     state_dict = reference_model.state_dict()
#     tt_model = Experts(config, state_dict, device)

#     # Run forward passes
#     reference_output = reference_model(hidden_states, routing_weights)
#     tt_output = tt_model(tt_hidden_states, tt_routing_weights)

#     # Convert TTNN output to torch
#     tt_output = ttnn.to_torch(tt_output)

#     # Compare outputs
#     passing, output = comp_allclose_and_pcc(reference_output, tt_output, atol=1e-2, rtol=1e-1)
#     print(f"sparse_experts_output: {output}")
#     assert passing, "sparse experts output mismatch"


@pytest.mark.parametrize(
    "num_experts, experts_per_token, intermediate_size, hidden_size",
    [
        # (32, 4, 2880, 2880),  # 20B config
        (128, 4, 2880, 2880),  # 120B config
    ],
    ids=["gpt20B"],
)
@pytest.mark.parametrize("batch_size", (1,))
@pytest.mark.parametrize("seq_len", [1], ids=["s1_"])
# @pytest.mark.parametrize("seq_len", [1, 32, 64, 128, 512, 1024], ids=["s1_", "s32", "s64", "s128", "s512", "s1024"])
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
def test_sparse_experts(
    mesh_device, num_experts, experts_per_token, intermediate_size, hidden_size, seq_len, batch_size, reset_seeds
):
    mesh_device = mesh_device.create_submesh(ttnn.MeshShape((1, 8)))
    print("MESH DEVICE!", mesh_device)
    print("MESH SHAPE!", mesh_device.shape)

    # Get paths from ModelArgs to avoid code duplication
    model_args = ModelArgs(mesh_device=None, dummy_weights=True)  # dummy_weights=True to avoid loading actual weights
    dtype = ttnn.bfloat8_b  # Always use bfp8

    """Test experts with sparse routing weights (only a few experts active per token)"""
    # Create configuration
    config = GptOssConfig(
        num_local_experts=num_experts,
        intermediate_size=intermediate_size,
        hidden_size=hidden_size,
        num_experts_per_tok=experts_per_token,
    )

    # Create input tensors
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)

    # Create sparse routing weights (simulate topk routing)
    routing_weights = torch.zeros(batch_size * seq_len, num_experts)

    for b, s in itertools.product(range(batch_size), range(seq_len)):
        # Randomly select which experts are active for this token
        active_experts = torch.randperm(num_experts)[:experts_per_token]
        weights = torch.rand(experts_per_token)
        weights = weights / weights.sum()  # Normalize
        routing_weights[b * seq_len + s, active_experts] = weights

    # Convert to TTNN tensors
    tt_hidden_states = ttnn.from_torch(hidden_states, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_routing_weights = ttnn.from_torch(
        routing_weights, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )

    # Create models
    reference_model = ReferenceExperts(config)
    state_dict = reference_model.state_dict()
    ccl_manager = CCLManager(mesh_device)
    tt_model = SparseExperts(
        mesh_device, config, state_dict, ccl_manager, tensor_cache_path=model_args.weight_cache_path(dtype)
    )

    # # Run forward passes
    reference_output = reference_model(hidden_states, routing_weights)
    tt_output = tt_model(tt_hidden_states, tt_routing_weights)

    tt_output_tensors = ttnn.get_device_tensors(tt_output)

    logger.info(f"tt_output_tensors: {tt_output_tensors}")
    logger.info(f"tt_output_tensors.size: {len(tt_output_tensors)}")
    logger.info(f"tt_output_tensors.shape: {tt_output_tensors[0].shape}")

    # Log if all the tensors in tt_output_tensors are the same shape
    logger.info(f"all_same_shape: {len(set(tuple(tensor.shape) for tensor in tt_output_tensors)) == 1}")

    for i in range(len(tt_output_tensors)):
        tt_output = ttnn.to_torch(tt_output_tensors[i])

        # Compare outputs
        passing, output = comp_pcc(reference_output, tt_output, pcc=0.92)
        mse = torch.nn.functional.mse_loss(reference_output, tt_output)

        # Calculate relative error metrics
        ref_variance = torch.var(reference_output)
        ref_mean_abs = torch.mean(torch.abs(reference_output))
        ref_std = torch.std(reference_output)

        relative_mse_to_variance = mse / ref_variance if ref_variance > 0 else float("inf")
        relative_mse_to_scale = mse / (ref_mean_abs**2) if ref_mean_abs > 0 else float("inf")
        snr_db = 10 * torch.log10(ref_variance / mse) if mse > 0 else float("inf")

        print(f"experts_output: {output}")
        print(f"MSE: {mse:.6e}")
        print(f"Reference variance: {ref_variance:.6e}, std: {ref_std:.6e}, mean_abs: {ref_mean_abs:.6e}")
        print(f"Relative MSE to variance: {relative_mse_to_variance:.6e} ({relative_mse_to_variance*100:.4f}%)")
        print(f"Relative MSE to scale²: {relative_mse_to_scale:.6e} ({relative_mse_to_scale*100:.4f}%)")
        print(f"Signal-to-Noise Ratio: {snr_db:.2f} dB")
        print(f"Reference output range: [{torch.min(reference_output):.6e}, {torch.max(reference_output):.6e}]")
        print(f"TT output range: [{torch.min(tt_output):.6e}, {torch.max(tt_output):.6e}]")

        assert passing, "experts output mismatch"
