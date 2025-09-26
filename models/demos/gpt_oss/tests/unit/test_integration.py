"""
Minimal integration tests - full model pipeline for any GPT model size
"""

import pytest
import torch

import ttnn

from ...tt.layer import DecoderLayer
from ...tt.model import Model
from ..test_factory import TestFactory, parametrize_batch_seq, parametrize_mesh


@parametrize_mesh(["1x8", "4x4"])
@parametrize_batch_seq([(1, 1), (1, 32)])
def test_decoder_layer_integration(mesh_device, batch_size, seq_len, reset_seeds):
    """Test complete decoder layer - combines attention + MLP + norms"""

    setup = TestFactory.setup_test(mesh_device)
    config = setup["config"]

    # Complete layer state dict
    layer_state = {
        "input_layernorm": {"weight": torch.ones(config.hidden_size)},
        "post_attention_layernorm": {"weight": torch.ones(config.hidden_size)},
        "self_attn": {
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
        },
        "mlp": {
            "router": {
                "weight": torch.randn(config.num_local_experts, config.hidden_size),
                "bias": torch.randn(config.num_local_experts),
            },
            "experts": setup["state_dict"],
        },
    }

    # Create decoder layer
    decoder_layer = DecoderLayer(
        setup["mesh_device"],
        config,
        layer_state,
        layer_idx=0,
        ccl_manager=setup["ccl_manager"],
        dtype=setup["dtype"],
        tensor_cache_path=setup["tensor_cache_path"],
        mesh_config=setup["mesh_config"],
    )

    # Test forward pass - works for any model size
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    tt_hidden_states = ttnn.from_torch(
        hidden_states, device=setup["mesh_device"], layout=ttnn.TILE_LAYOUT, dtype=setup["dtype"]
    )

    # Forward through complete layer
    layer_out = decoder_layer(tt_hidden_states)

    # Verify output shape matches input - essential for stacking layers
    assert layer_out.shape == tt_hidden_states.shape

    # Verify output is reasonable (not NaN/Inf)
    layer_out_torch = ttnn.to_torch(layer_out, mesh_composer=ttnn.ConcatMesh2dToTensor(setup["mesh_device"]))
    assert torch.isfinite(layer_out_torch).all()


@parametrize_mesh(["1x8"])  # Full model is expensive, test minimal config
def test_model_construction(mesh_device, reset_seeds):
    """Test that full model constructs correctly - essential for deployment"""

    setup = TestFactory.setup_test(mesh_device)
    config = setup["config"]

    # Minimal model state dict (just enough for construction)
    model_state = {
        "model": {
            "embed_tokens": {"weight": torch.randn(config.vocab_size, config.hidden_size)},
            "norm": {"weight": torch.ones(config.hidden_size)},
            "layers": {},
        },
        "lm_head": {"weight": torch.randn(config.vocab_size, config.hidden_size)},
    }

    # Add minimal layer states
    for layer_idx in range(min(2, config.num_hidden_layers)):  # Test just 2 layers max
        model_state["model"]["layers"][f"{layer_idx}"] = {
            "input_layernorm": {"weight": torch.ones(config.hidden_size)},
            "post_attention_layernorm": {"weight": torch.ones(config.hidden_size)},
            "self_attn": {
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
            },
            "mlp": {
                "router": {
                    "weight": torch.randn(config.num_local_experts, config.hidden_size),
                    "bias": torch.randn(config.num_local_experts),
                },
                "experts": setup["state_dict"],
            },
        }

    # Temporarily reduce layers for test
    original_layers = config.num_hidden_layers
    config.num_hidden_layers = min(2, original_layers)

    try:
        # Test model construction
        model = Model(
            setup["mesh_device"],
            config,
            model_state,
            setup["ccl_manager"],
            dtype=setup["dtype"],
            tensor_cache_path=setup["tensor_cache_path"],
            mesh_config=setup["mesh_config"],
        )

        assert model is not None
        assert len(model.layers) == config.num_hidden_layers
        assert model.mesh_config.tp == setup["mesh_config"].tp

    finally:
        # Restore original layer count
        config.num_hidden_layers = original_layers


@parametrize_mesh(["1x8", "4x4"])
def test_mesh_communication_patterns(mesh_device, reset_seeds):
    """Test CCL communication works across different mesh shapes"""

    setup = TestFactory.setup_test(mesh_device)

    # Test allreduce communication pattern
    test_tensor = torch.randn(1, 32, setup["config"].hidden_size)
    tt_tensor = ttnn.from_torch(test_tensor, device=setup["mesh_device"], layout=ttnn.TILE_LAYOUT, dtype=setup["dtype"])

    # Test mesh config allreduce
    if setup["mesh_config"].tp > 1:
        # Add batch dimension for allreduce
        tt_tensor_4d = ttnn.unsqueeze(tt_tensor, 0)
        result = setup["mesh_config"].allreduce(tt_tensor_4d, setup["ccl_manager"], pad_size=192)
        result = ttnn.squeeze(result, 0)

        # Verify shapes are preserved
        assert result.shape == tt_tensor.shape

        # Verify result is reasonable
        result_torch = ttnn.to_torch(result, mesh_composer=ttnn.ConcatMesh2dToTensor(setup["mesh_device"]))
        assert torch.isfinite(result_torch).all()
    else:
        # For TP=1, allreduce should be identity
        result = setup["mesh_config"].allreduce(tt_tensor, setup["ccl_manager"])
        assert torch.allclose(ttnn.to_torch(result), ttnn.to_torch(tt_tensor))


# Model compatibility test - ensure works across different GPT variants
@pytest.mark.parametrize(
    "model_variant",
    [
        "gpt_20b_style",  # Smaller model
        "gpt_120b_style",  # Larger model
    ],
)
def test_model_variant_compatibility(mesh_device, model_variant, reset_seeds):
    """Test compatibility across different model variants/sizes"""

    setup = TestFactory.setup_test(mesh_device)
    config = setup["config"]

    # Adjust config for different variants (without hardcoding exact values)
    if model_variant == "gpt_20b_style":
        # Smaller configuration
        scale_factor = 0.5
    else:  # gpt_120b_style
        # Larger configuration
        scale_factor = 2.0

    # Scale key dimensions proportionally
    original_hidden = config.hidden_size
    config.hidden_size = int(config.hidden_size * scale_factor)
    config.intermediate_size = int(config.intermediate_size * scale_factor)

    # Generate appropriately scaled state dict
    scaled_state = {
        "gate_up_proj": torch.randn(config.num_local_experts, config.hidden_size, 2 * config.intermediate_size),
        "gate_up_proj_bias": torch.randn(config.num_local_experts, 2 * config.intermediate_size),
        "down_proj": torch.randn(config.num_local_experts, config.intermediate_size, config.hidden_size),
        "down_proj_bias": torch.randn(config.num_local_experts, config.hidden_size),
    }

    # Test expert construction with scaled config
    from ...tt.experts import Experts

    tt_experts = Experts(
        setup["mesh_device"],
        config,
        scaled_state,
        setup["ccl_manager"],
        dtype=setup["dtype"],
        tensor_cache_path=setup["tensor_cache_path"],
        mesh_config=setup["mesh_config"],
    )

    # Test with scaled inputs
    hidden_states = torch.randn(1, 16, config.hidden_size)
    routing_weights = torch.randn(1, 16, config.num_local_experts)

    tt_hidden = ttnn.from_torch(
        hidden_states, device=setup["mesh_device"], layout=ttnn.TILE_LAYOUT, dtype=setup["dtype"]
    )
    tt_routing = ttnn.from_torch(
        routing_weights, device=setup["mesh_device"], layout=ttnn.TILE_LAYOUT, dtype=setup["dtype"]
    )

    # Forward pass should work regardless of scale
    output = tt_experts(tt_hidden, tt_routing)
    assert output.shape == tt_hidden.shape

    # Verify mesh config scales appropriately
    expected_shard_size = config.intermediate_size // setup["mesh_config"].tp
    assert tt_experts.intermediate_size_per_device == expected_shard_size

    # Restore original config
    config.hidden_size = original_hidden


# Performance regression test - ensure changes don't break basic functionality
def test_basic_functionality_preserved(mesh_device, reset_seeds):
    """Smoke test to ensure all refactoring preserves basic functionality"""

    setup = TestFactory.setup_test(mesh_device)

    # Test all core components can be constructed
    from ...tt.experts import Experts
    from ...tt.mlp import MLP
    from ...tt.rms_norm import RMSNorm
    from ...tt.topk import TopKRouter

    config = setup["config"]

    # Basic construction test
    experts = Experts(
        setup["mesh_device"], config, setup["state_dict"], setup["ccl_manager"], mesh_config=setup["mesh_config"]
    )

    router_state = {
        "weight": torch.randn(config.num_local_experts, config.hidden_size),
        "bias": torch.randn(config.num_local_experts),
    }
    router = TopKRouter(setup["mesh_device"], config, router_state)

    norm_state = {"weight": torch.ones(config.hidden_size)}
    norm = RMSNorm(setup["mesh_device"], config, norm_state, mesh_config=setup["mesh_config"])

    mlp_state = {"router": router_state, "experts": setup["state_dict"]}
    mlp = MLP(setup["mesh_device"], config, mlp_state, setup["ccl_manager"], mesh_config=setup["mesh_config"])

    # All components should construct successfully
    assert all(comp is not None for comp in [experts, router, norm, mlp])

    # Test basic forward passes work
    hidden_states = ttnn.from_torch(
        torch.randn(1, 8, config.hidden_size),
        device=setup["mesh_device"],
        layout=ttnn.TILE_LAYOUT,
        dtype=setup["dtype"],
    )

    # All should process without error
    norm_out = norm(hidden_states)
    router_scores, _, _ = router(hidden_states)
    expert_out = experts(hidden_states, router_scores)
    mlp_out, _ = mlp(hidden_states)

    # Verify all outputs have correct shapes
    assert norm_out.shape == hidden_states.shape
    assert expert_out.shape == hidden_states.shape
    assert mlp_out.shape == hidden_states.shape
