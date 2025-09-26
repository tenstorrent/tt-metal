"""
Minimal but comprehensive core component tests - works for any GPT model size
"""

import pytest
import torch

import ttnn

from ...tt.attention import Attention
from ...tt.experts import Experts
from ...tt.mlp import MLP
from ...tt.rms_norm import RMSNorm
from ...tt.topk import TopKRouter
from ..test_factory import (
    ReferenceExperts,
    ReferenceRMSNorm,
    ReferenceTopKRouter,
    TestFactory,
    compare_tensors,
    parametrize_batch_seq,
    parametrize_mesh,
)


# Core MoE Tests - Essential for any model size
@parametrize_mesh(["1x8", "4x4", "4x8"])  # Test multiple mesh configs
@parametrize_batch_seq([(1, 1), (1, 32)])  # Single token + sequence
def test_experts_core(mesh_device, batch_size, seq_len, reset_seeds):
    """Test core expert functionality - works for GPT-20B, GPT-120B, any size"""

    setup = TestFactory.setup_test(mesh_device)
    config = setup["config"]

    # Reference vs TT implementation
    torch_model = ReferenceExperts(config)
    tt_model = Experts(
        setup["mesh_device"],
        config,
        setup["state_dict"],
        setup["ccl_manager"],
        dtype=setup["dtype"],
        tensor_cache_path=setup["tensor_cache_path"],
        mesh_config=setup["mesh_config"],
    )

    # Test data - scales with any hidden_size
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    routing_weights = torch.randn(batch_size, seq_len, config.num_local_experts)

    # Forward pass
    torch_out = torch_model(hidden_states, routing_weights)

    tt_hidden_states = ttnn.from_torch(
        hidden_states, device=setup["mesh_device"], layout=ttnn.TILE_LAYOUT, dtype=setup["dtype"]
    )
    tt_routing_weights = ttnn.from_torch(
        routing_weights, device=setup["mesh_device"], layout=ttnn.TILE_LAYOUT, dtype=setup["dtype"]
    )
    tt_out = tt_model(tt_hidden_states, tt_routing_weights)

    # Compare - works for any model size
    assert compare_tensors(tt_out, torch_out, setup["mesh_device"], pcc_threshold=0.98)


@parametrize_mesh(["1x8", "4x4"])
@parametrize_batch_seq([(1, 1), (1, 32)])
def test_topk_router(mesh_device, batch_size, seq_len, reset_seeds):
    """Test TopK routing - essential for MoE"""

    setup = TestFactory.setup_test(mesh_device)
    config = setup["config"]

    # Router state dict
    router_state = {
        "weight": torch.randn(config.num_local_experts, config.hidden_size),
        "bias": torch.randn(config.num_local_experts),
    }

    torch_model = ReferenceTopKRouter(config)
    tt_model = TopKRouter(setup["mesh_device"], config, router_state, tensor_cache_path=setup["tensor_cache_path"])

    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    torch_scores, torch_indices = torch_model(hidden_states)

    tt_hidden_states = ttnn.from_torch(
        hidden_states, device=setup["mesh_device"], layout=ttnn.TILE_LAYOUT, dtype=setup["dtype"]
    )
    tt_scores, tt_indices, _ = tt_model(tt_hidden_states)

    # Verify shapes match - works for any num_experts/hidden_size
    assert compare_tensors(tt_scores, torch_scores, setup["mesh_device"], pcc_threshold=0.95)


@parametrize_mesh(["1x8", "4x4"])
@parametrize_batch_seq([(1, 1), (1, 128)])
def test_attention_core(mesh_device, batch_size, seq_len, reset_seeds):
    """Test attention mechanism - core for any transformer"""

    setup = TestFactory.setup_test(mesh_device)
    config = setup["config"]

    # Attention state dict
    attn_state = {
        "q_proj": {
            "weight": torch.randn(config.hidden_size, config.hidden_size),
            "bias": torch.randn(config.hidden_size),
        },
        "k_proj": {
            "weight": torch.randn(config.hidden_size, config.num_key_value_heads * config.head_dim),
            "bias": torch.randn(config.num_key_value_heads * config.head_dim),
        },
        "v_proj": {
            "weight": torch.randn(config.hidden_size, config.num_key_value_heads * config.head_dim),
            "bias": torch.randn(config.num_key_value_heads * config.head_dim),
        },
        "o_proj": {
            "weight": torch.randn(config.hidden_size, config.hidden_size),
            "bias": torch.randn(config.hidden_size),
        },
        "sinks": torch.randn(config.num_attention_heads),
    }

    tt_model = Attention(
        setup["mesh_device"],
        config,
        attn_state,
        layer_idx=0,
        ccl_manager=setup["ccl_manager"],
        tensor_cache_path=setup["tensor_cache_path"],
        mesh_config=setup["mesh_config"],
    )

    # Test forward pass - scales with any model size
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    tt_hidden_states = ttnn.from_torch(
        hidden_states, device=setup["mesh_device"], layout=ttnn.TILE_LAYOUT, dtype=setup["dtype"]
    )

    tt_out = tt_model(tt_hidden_states)

    # Verify output shape matches input - works for any hidden_size
    assert tt_out.shape == tt_hidden_states.shape


@parametrize_mesh(["1x8", "4x4"])
@parametrize_batch_seq([(1, 32)])
def test_rms_norm(mesh_device, batch_size, seq_len, reset_seeds):
    """Test RMS normalization - used throughout model"""

    setup = TestFactory.setup_test(mesh_device)
    config = setup["config"]

    norm_state = {"weight": torch.ones(config.hidden_size)}

    torch_model = ReferenceRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    tt_model = RMSNorm(
        setup["mesh_device"],
        config,
        norm_state,
        tensor_cache_path=setup["tensor_cache_path"],
        mesh_config=setup["mesh_config"],
    )

    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    torch_out = torch_model(hidden_states)

    tt_hidden_states = ttnn.from_torch(
        hidden_states, device=setup["mesh_device"], layout=ttnn.TILE_LAYOUT, dtype=setup["dtype"]
    )
    tt_out = tt_model(tt_hidden_states)

    assert compare_tensors(tt_out, torch_out, setup["mesh_device"], pcc_threshold=0.99)


@parametrize_mesh(["1x8", "4x4"])
def test_full_mlp_pipeline(mesh_device, reset_seeds):
    """Test complete MLP (router + experts) - essential MoE functionality"""

    setup = TestFactory.setup_test(mesh_device)
    config = setup["config"]

    # Complete MLP state dict
    mlp_state = {
        "router": {
            "weight": torch.randn(config.num_local_experts, config.hidden_size),
            "bias": torch.randn(config.num_local_experts),
        },
        "experts": setup["state_dict"],  # Use experts from factory
    }

    tt_mlp = MLP(
        setup["mesh_device"],
        config,
        mlp_state,
        setup["ccl_manager"],
        dtype=setup["dtype"],
        tensor_cache_path=setup["tensor_cache_path"],
        mesh_config=setup["mesh_config"],
    )

    # Test with various sequence lengths
    for seq_len in [1, 32]:
        hidden_states = torch.randn(1, seq_len, config.hidden_size)
        tt_hidden_states = ttnn.from_torch(
            hidden_states, device=setup["mesh_device"], layout=ttnn.TILE_LAYOUT, dtype=setup["dtype"]
        )

        mlp_out, routing_scores = tt_mlp(tt_hidden_states)

        # Verify output shapes - works for any model size
        assert mlp_out.shape == tt_hidden_states.shape
        assert routing_scores.shape[:-1] == tt_hidden_states.shape[:-1]  # Same batch/seq, different last dim


# Mesh flexibility test - ensure works with any configuration
@pytest.mark.parametrize(
    "mesh_config",
    [
        ("4x8", 8),  # Current setup
        ("4x4", 4),  # Square mesh
        ("1x8", 8),  # Linear mesh
        ("2x4", 4),  # Smaller mesh
    ],
)
def test_mesh_flexibility(mesh_device, mesh_config, reset_seeds):
    """Test that all components work with any mesh configuration"""

    mesh_shape_name, expected_tp = mesh_config
    setup = TestFactory.setup_test(mesh_device)

    # Verify mesh config is correct
    assert setup["mesh_config"].tp == expected_tp
    assert setup["mesh_config"].mesh_shape == mesh_device.shape

    # Test that expert construction works
    tt_experts = Experts(
        setup["mesh_device"],
        setup["config"],
        setup["state_dict"],
        setup["ccl_manager"],
        mesh_config=setup["mesh_config"],
    )

    # Verify tensor parallel size calculation
    expected_shard_size = setup["config"].intermediate_size // expected_tp
    assert tt_experts.intermediate_size_per_device == expected_shard_size

    # Test forward pass works
    hidden_states = ttnn.from_torch(
        torch.randn(1, 32, setup["config"].hidden_size),
        device=setup["mesh_device"],
        layout=ttnn.TILE_LAYOUT,
        dtype=setup["dtype"],
    )
    routing_weights = ttnn.from_torch(
        torch.randn(1, 32, setup["config"].num_local_experts),
        device=setup["mesh_device"],
        layout=ttnn.TILE_LAYOUT,
        dtype=setup["dtype"],
    )

    output = tt_experts(hidden_states, routing_weights)
    assert output.shape == hidden_states.shape


# Model size flexibility test - works for GPT-20B, GPT-120B, any size
@pytest.mark.parametrize(
    "model_scale",
    [
        {"hidden_size": 2048, "num_experts": 64, "intermediate_size": 5632},  # GPT-20B scale
        {"hidden_size": 4096, "num_experts": 128, "intermediate_size": 11264},  # GPT-120B scale
        {"hidden_size": 1024, "num_experts": 32, "intermediate_size": 2816},  # Smaller test
    ],
)
def test_model_scale_flexibility(mesh_device, model_scale, reset_seeds):
    """Test components work with different model scales"""

    setup = TestFactory.setup_test(mesh_device)

    # Override config with different scales
    config = setup["config"]
    for key, value in model_scale.items():
        setattr(config, key, value)

    # Generate appropriately sized state dict
    state_dict = {
        "gate_up_proj": torch.randn(config.num_local_experts, config.hidden_size, 2 * config.intermediate_size),
        "gate_up_proj_bias": torch.randn(config.num_local_experts, 2 * config.intermediate_size),
        "down_proj": torch.randn(config.num_local_experts, config.intermediate_size, config.hidden_size),
        "down_proj_bias": torch.randn(config.num_local_experts, config.hidden_size),
    }

    # Test expert construction with different sizes
    tt_experts = Experts(
        setup["mesh_device"], config, state_dict, setup["ccl_manager"], mesh_config=setup["mesh_config"]
    )

    # Test with appropriately sized inputs
    hidden_states = ttnn.from_torch(
        torch.randn(1, 16, config.hidden_size),
        device=setup["mesh_device"],
        layout=ttnn.TILE_LAYOUT,
        dtype=setup["dtype"],
    )
    routing_weights = ttnn.from_torch(
        torch.randn(1, 16, config.num_local_experts),
        device=setup["mesh_device"],
        layout=ttnn.TILE_LAYOUT,
        dtype=setup["dtype"],
    )

    output = tt_experts(hidden_states, routing_weights)
    assert output.shape == hidden_states.shape
