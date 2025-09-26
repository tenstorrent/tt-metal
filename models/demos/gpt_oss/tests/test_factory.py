"""
Ultra-minimal test factory - eliminates all test duplication
"""

from typing import Dict

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import ttnn

from ..moe import MeshConfig
from ..reference.configuration_gpt_oss import GptOssConfig
from ..reference.hf_utils import get_state_dict
from ..tt.ccl import CCLManager
from ..tt.model_config import ModelArgs


class TestFactory:
    """One factory to rule them all - zero duplication testing"""

    # Common test configurations
    MESH_SHAPES = {"4x8": (4, 8), "1x8": (1, 8), "4x4": (4, 4), "2x4": (2, 4)}

    BATCH_SEQ_CONFIGS = [
        (1, 1),  # Single token
        (1, 32),  # Small sequence
        (1, 128),  # Medium sequence
    ]

    @staticmethod
    def setup_test(mesh_device, use_real_weights=False, dtype=ttnn.bfloat8_b):
        """Universal test setup - replaces all the duplicated setup code"""

        # Use mesh_device as-is (already created by conftest.py fixture)
        mesh_shape = mesh_device.shape

        # Setup ModelArgs (no import-time loading)
        model_args = ModelArgs(mesh_device=None, dummy_weights=True)

        # Setup mesh config using actual mesh shape
        mesh_config = MeshConfig(mesh_shape, tp=mesh_shape[1])

        # Setup CCL
        ccl_manager = CCLManager(mesh_device)

        # Get config and state dict
        if use_real_weights:
            config = GptOssConfig.from_pretrained(model_args.model_path, trust_remote_code=True)
            state_dict = get_state_dict(model_args.model_path, "", dtype=torch.bfloat16)
        else:
            # Use dummy config for testing
            config = GptOssConfig(
                vocab_size=201088,
                hidden_size=2048,
                intermediate_size=5632,
                num_hidden_layers=32,
                num_attention_heads=32,
                num_key_value_heads=32,
                num_local_experts=128,
                num_experts_per_tok=2,
                rms_norm_eps=1e-6,
                rope_theta=10000.0,
                sliding_window=4096,
                max_position_embeddings=131072,  # Add missing attribute
                layer_types=["full_attention"] * 32,
            )
            config.head_dim = config.hidden_size // config.num_attention_heads
            # Generate dummy state dict
            state_dict = TestFactory._generate_dummy_state_dict(config)

        return {
            "mesh_device": mesh_device,
            "model_args": model_args,
            "mesh_config": mesh_config,
            "ccl_manager": ccl_manager,
            "config": config,
            "state_dict": state_dict,
            "dtype": dtype,
            "gpt_dir": model_args.model_path,
            "tensor_cache_path": model_args.weight_cache_path(dtype),
        }

    @staticmethod
    def _generate_dummy_state_dict(config) -> Dict[str, torch.Tensor]:
        """Generate dummy state dict - scales with any model size"""

        # Extract dimensions from config (works for any model size)
        num_experts = getattr(config, "num_local_experts", 128)
        hidden_size = getattr(config, "hidden_size", 2048)
        intermediate_size = getattr(config, "intermediate_size", 5632)
        num_attention_heads = getattr(config, "num_attention_heads", 32)
        num_kv_heads = getattr(config, "num_key_value_heads", 32)
        head_dim = getattr(config, "head_dim", 64)
        vocab_size = getattr(config, "vocab_size", 201088)

        return {
            # Expert weights - scales with model size
            "gate_up_proj": torch.randn(num_experts, hidden_size, 2 * intermediate_size),
            "gate_up_proj_bias": torch.randn(num_experts, 2 * intermediate_size),
            "down_proj": torch.randn(num_experts, intermediate_size, hidden_size),
            "down_proj_bias": torch.randn(num_experts, hidden_size),
            # Router weights - scales with experts and hidden size
            "router": {
                "weight": torch.randn(num_experts, hidden_size),
                "bias": torch.randn(num_experts),
            },
            # Attention weights - scales with hidden dimensions
            "q_proj": {
                "weight": torch.randn(hidden_size, hidden_size),
                "bias": torch.randn(hidden_size),
            },
            "k_proj": {
                "weight": torch.randn(hidden_size, num_kv_heads * head_dim),
                "bias": torch.randn(num_kv_heads * head_dim),
            },
            "v_proj": {
                "weight": torch.randn(hidden_size, num_kv_heads * head_dim),
                "bias": torch.randn(num_kv_heads * head_dim),
            },
            "o_proj": {
                "weight": torch.randn(hidden_size, hidden_size),
                "bias": torch.randn(hidden_size),
            },
            # Norm weights - scales with hidden size
            "weight": torch.ones(hidden_size),
            # KV cache and attention - scales with num heads
            "sinks": torch.randn(num_attention_heads),
            # Additional embeddings for full model tests
            "embed_tokens": {"weight": torch.randn(vocab_size, hidden_size)},
            "lm_head": {"weight": torch.randn(vocab_size, hidden_size)},
        }


# Universal Reference Classes (no more duplication across files!)
class ReferenceExperts(nn.Module):
    """Unified reference experts - replaces duplicated classes"""

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

    def forward(self, hidden_states, routing_weights):
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        num_experts = routing_weights.shape[1]

        hidden_states = hidden_states.repeat(num_experts, 1).view(num_experts, -1, self.hidden_size)
        gate_up = torch.bmm(hidden_states, self.gate_up_proj) + self.gate_up_proj_bias[..., None, :]
        gate, up = gate_up[..., ::2], gate_up[..., 1::2]
        gate = gate.clamp(max=self.limit)
        up = up.clamp(-self.limit, self.limit)
        glu = gate * torch.sigmoid(gate * self.alpha)
        next_states = torch.bmm(((up + 1) * glu), self.down_proj) + self.down_proj_bias[..., None, :]

        next_states = next_states.view(num_experts, batch_size, -1, self.hidden_size)
        routing_weights = routing_weights.permute(1, 0)[..., None, None]
        final_hidden_states = (next_states * routing_weights).sum(dim=0)
        return final_hidden_states.view(batch_size, -1, self.hidden_size)


class ReferenceTopKRouter(nn.Module):
    """Unified reference router - replaces duplicated classes"""

    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.randn(self.num_experts, self.hidden_dim))
        self.bias = nn.Parameter(torch.randn(self.num_experts))

    def forward(self, hidden_states):
        batch_size, seq_len = hidden_states.shape[:2]
        hidden_states = hidden_states.view(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight, self.bias)
        routing_weights, selected_experts = torch.topk(router_logits, self.top_k, dim=-1)
        routing_weights = F.softmax(routing_weights.float(), dim=-1).to(hidden_states.dtype)

        # Convert to dense format
        final_routing_weights = torch.zeros_like(router_logits).scatter(-1, selected_experts, routing_weights)
        return final_routing_weights.view(batch_size, seq_len, -1), selected_experts


class ReferenceRMSNorm(nn.Module):
    """Unified reference norm - replaces duplicated classes"""

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


# Universal Test Parametrization Decorators (no more copy-paste!)
def parametrize_mesh(shapes=["1x8"]):
    """Universal mesh parametrization"""
    mesh_params = [pytest.param(TestFactory.MESH_SHAPES[s], id=f"{s}_grid") for s in shapes]
    return pytest.mark.parametrize("mesh_device", mesh_params, indirect=True)


def parametrize_batch_seq(configs=None):
    """Universal batch/seq parametrization"""
    configs = configs or [(1, 1), (1, 32)]
    return pytest.mark.parametrize("batch_size, seq_len", configs)


def parametrize_weights(use_real=False):
    """Universal weight parametrization"""
    return pytest.mark.parametrize("use_real_weights", [use_real], ids=["real" if use_real else "random"])


def parametrize_fabric():
    """Universal fabric parametrization"""
    return pytest.mark.parametrize(
        "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True
    )


# Test helper functions
def compare_tensors(tt_tensor, torch_tensor, mesh_device, pcc_threshold=0.99):
    """Universal tensor comparison"""
    from models.utility_functions import comp_pcc

    tt_torch = ttnn.to_torch(tt_tensor, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device))
    return comp_pcc(torch_tensor, tt_torch, pcc_threshold)
