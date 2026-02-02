# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Tests MoE modules with TTNN acceleration."""

import pytest
import torch

from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor
from models.experimental.tt_symbiote.modules.moe import (
    Glm4MoeConfig,
    Glm4MoeMLP,
    Glm4MoeTopkRouter,
    Glm4MoeNaiveMoe,
    Glm4MoeMoE,
    TTNNGlm4MoeMoE,
)
from models.experimental.tt_symbiote.utils.device_management import set_device, DeviceInit
from models.experimental.tt_symbiote.core.utils import compare_fn_outputs
from models.experimental.tt_symbiote.core.run_config import DistributedConfig, DistributedTensorConfig
from dataclasses import dataclass
import ttnn
from models.demos.deepseek_v3.tt.ccl import CCL


@dataclass
class MoEDistributedConfig(DistributedConfig):
    """Distributed configuration for MoE modules."""

    def __post_init__(self):
        if self.tensor_config is None:
            self.tensor_config = DistributedTensorConfig(
                mesh_mapper=ttnn.shard_tensor_to_mesh_mapper(self.mesh_device, dim=-1)
            )
        self.ccl_manager = CCL(self.mesh_device)


class MoEDeviceInit(DeviceInit):
    @staticmethod
    def init_state_impl(device):
        return MoEDistributedConfig(mesh_device=device)


@pytest.fixture
def default_moe_config():
    """Default MoE configuration for testing."""
    return Glm4MoeConfig(
        hidden_size=256,
        intermediate_size=512,
        moe_intermediate_size=128,
        num_local_experts=16,
        num_experts_per_tok=4,
        n_shared_experts=1,
        n_routed_experts=4,
        routed_scaling_factor=1.0,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_glm4_moe_mlp(device, default_moe_config):
    """Test Glm4MoeMLP with TTNN acceleration."""
    model = Glm4MoeMLP(default_moe_config).to(dtype=torch.bfloat16)
    model.eval()
    torch.set_grad_enabled(False)

    batch_size, seq_len = 2, 4
    inputs = TorchTTNNTensor(torch.randn((batch_size, seq_len, default_moe_config.hidden_size), dtype=torch.bfloat16))

    outputs_torch = model(inputs)

    # TODO: Create TTNNGlm4MoeMLP when implementing TTNN version
    # ttnn_model = TTNNGlm4MoeMLP.from_torch(model)
    # set_device(ttnn_model, device)
    # outputs_ttnn = ttnn_model(inputs)
    # compare_fn_outputs(outputs_torch, outputs_ttnn, "Glm4MoeMLP")

    # For now, just verify the torch output shape
    assert outputs_torch.shape == (batch_size, seq_len, default_moe_config.hidden_size)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_glm4_moe_topk_router(device, default_moe_config):
    """Test Glm4MoeTopkRouter with TTNN acceleration."""
    model = Glm4MoeTopkRouter(default_moe_config).to(dtype=torch.bfloat16)
    model.eval()
    torch.set_grad_enabled(False)

    batch_size, seq_len = 2, 4
    inputs = TorchTTNNTensor(torch.randn((batch_size, seq_len, default_moe_config.hidden_size), dtype=torch.bfloat16))

    outputs_torch = model(inputs)

    # TODO: Create TTNNGlm4MoeTopkRouter when implementing TTNN version
    # ttnn_model = TTNNGlm4MoeTopkRouter.from_torch(model)
    # set_device(ttnn_model, device)
    # outputs_ttnn = ttnn_model(inputs)
    # compare_fn_outputs(outputs_torch, outputs_ttnn, "Glm4MoeTopkRouter")

    # Verify router logits shape
    expected_shape = (batch_size * seq_len, default_moe_config.n_routed_experts)
    assert outputs_torch.shape == expected_shape


@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_glm4_moe_naive_moe(device, default_moe_config):
    """Test Glm4MoeNaiveMoe with TTNN acceleration."""
    model = Glm4MoeNaiveMoe(default_moe_config).to(dtype=torch.bfloat16)
    model.eval()
    torch.set_grad_enabled(False)

    batch_size, seq_len = 2, 4
    hidden_states = TorchTTNNTensor(
        torch.randn((batch_size * seq_len, default_moe_config.hidden_size), dtype=torch.bfloat16)
    )

    # Create mock top_k indices and weights
    top_k_index = torch.randint(
        0, default_moe_config.num_local_experts, (batch_size * seq_len, default_moe_config.num_experts_per_tok)
    )
    top_k_weights = torch.randn((batch_size * seq_len, default_moe_config.num_experts_per_tok), dtype=torch.bfloat16)
    top_k_weights = torch.softmax(top_k_weights, dim=-1)

    outputs_torch = model(hidden_states, top_k_index, top_k_weights)

    # TODO: Create TTNNGlm4MoeNaiveMoe when implementing TTNN version
    # ttnn_model = TTNNGlm4MoeNaiveMoe.from_torch(model)
    # set_device(ttnn_model, device)
    # outputs_ttnn = ttnn_model(hidden_states, top_k_index, top_k_weights)
    # compare_fn_outputs(outputs_torch, outputs_ttnn, "Glm4MoeNaiveMoe")

    # Verify output shape matches input
    assert outputs_torch.shape == hidden_states.shape


@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_glm4_moe_full(mesh_device, default_moe_config):
    """Test full Glm4MoeMoE module with TTNN acceleration."""
    model = Glm4MoeMoE(default_moe_config).to(dtype=torch.bfloat16)
    model.eval()
    torch.set_grad_enabled(False)

    batch_size, seq_len = 2, 4
    inputs = TorchTTNNTensor(torch.randn((batch_size, seq_len, default_moe_config.hidden_size), dtype=torch.bfloat16))

    outputs_torch = model(inputs)
    ttnn_model = TTNNGlm4MoeMoE.from_torch(model)
    set_device(ttnn_model, mesh_device, MoEDeviceInit)
    outputs_ttnn = ttnn_model(inputs)
    compare_fn_outputs(outputs_torch, outputs_ttnn, "Glm4MoeMoE")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
@pytest.mark.parametrize("num_experts,top_k", [(4, 1), (8, 2), (16, 4)])
def test_glm4_moe_scaling(device, num_experts, top_k):
    """Test MoE with different expert counts and top-k values."""
    config = Glm4MoeConfig(
        hidden_size=128,
        intermediate_size=256,
        moe_intermediate_size=64,
        num_local_experts=num_experts,
        num_experts_per_tok=top_k,
        n_routed_experts=num_experts,
        n_shared_experts=1,
    )

    model = Glm4MoeMoE(config).to(dtype=torch.bfloat16)
    model.eval()
    torch.set_grad_enabled(False)

    batch_size, seq_len = 2, 3
    inputs = TorchTTNNTensor(torch.randn((batch_size, seq_len, config.hidden_size), dtype=torch.bfloat16))

    outputs_torch = model(inputs)
    assert outputs_torch.shape == (batch_size, seq_len, config.hidden_size)
