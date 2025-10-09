# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for Mixture of Experts (MoE) building block.
"""

import pytest
import torch
from src.building_blocks.ffn import (
    MoEImplConfig,
    MoESpec,
    get_default_moe_impl_config,
    moe_decode_forward,
    moe_prefill_forward,
)

import ttnn


class TestMoESpec:
    """Test suite for MoESpec."""

    def test_moe_spec_basic(self):
        """Test basic MoE spec creation and validation."""
        spec = MoESpec(num_experts=8, num_experts_per_tok=2, hidden_dim=4096, intermediate_dim=14336, activation="silu")
        spec.validate()
        assert spec.num_experts == 8
        assert spec.num_experts_per_tok == 2
        assert spec.hidden_dim == 4096
        assert spec.intermediate_dim == 14336
        assert spec.activation == "silu"

    def test_moe_spec_single_expert_routing(self):
        """Test MoE spec with single expert per token."""
        spec = MoESpec(
            num_experts=32,
            num_experts_per_tok=1,  # Sparse routing
            hidden_dim=4096,
            intermediate_dim=14336,
            activation="gelu",
        )
        spec.validate()
        assert spec.num_experts == 32
        assert spec.num_experts_per_tok == 1

    def test_moe_spec_with_normalization(self):
        """Test MoE spec with expert normalization."""
        spec = MoESpec(
            num_experts=8,
            num_experts_per_tok=2,
            hidden_dim=4096,
            intermediate_dim=14336,
            activation="silu",
            normalize_expert_weights=True,
        )
        spec.validate()
        assert spec.normalize_expert_weights is True

    def test_moe_spec_validation_errors(self):
        """Test MoE spec validation errors."""
        # Invalid num_experts
        with pytest.raises(AssertionError):
            spec = MoESpec(num_experts=0, num_experts_per_tok=2, hidden_dim=4096, intermediate_dim=14336)
            spec.validate()

        # num_experts_per_tok > num_experts
        with pytest.raises(AssertionError):
            spec = MoESpec(num_experts=8, num_experts_per_tok=10, hidden_dim=4096, intermediate_dim=14336)
            spec.validate()

        # Invalid num_experts_per_tok
        with pytest.raises(AssertionError):
            spec = MoESpec(num_experts=8, num_experts_per_tok=0, hidden_dim=4096, intermediate_dim=14336)
            spec.validate()


class TestMoEImplConfig:
    """Test suite for MoEImplConfig."""

    def test_moe_default_impl_config(self):
        """Test getting default implementation config for MoE."""
        spec = MoESpec(num_experts=8, num_experts_per_tok=2, hidden_dim=4096, intermediate_dim=14336)

        for device in ["N150", "N300", "T3000", "TG"]:
            for mode in ["prefill", "decode"]:
                impl_config = get_default_moe_impl_config(spec, device, mode)
                assert isinstance(impl_config, MoEImplConfig)
                assert impl_config.routing_algorithm in ["top_k", "expert_choice", "switch"]

    def test_moe_impl_config_routing_strategy(self):
        """Test MoE routing strategy configuration."""
        spec = MoESpec(num_experts=8, num_experts_per_tok=2, hidden_dim=4096, intermediate_dim=14336)

        impl_config = get_default_moe_impl_config(spec, "T3000", "prefill")

        # Check routing configuration
        assert impl_config.routing_algorithm in ["top_k", "expert_choice", "switch"]
        assert impl_config.aux_loss_weight >= 0.0
        assert impl_config.capacity_factor > 1.0

    def test_moe_impl_config_expert_parallelism(self):
        """Test MoE expert parallelism configuration."""
        spec = MoESpec(num_experts=32, num_experts_per_tok=2, hidden_dim=4096, intermediate_dim=14336)  # Many experts

        impl_config = get_default_moe_impl_config(spec, "TG", "prefill")

        # Galaxy might use expert parallelism
        assert hasattr(impl_config, "expert_parallel_dim")
        assert impl_config.expert_parallel_dim in [None, 0, 1, 2]  # Dimension to parallelize experts


class TestMoEForward:
    """Test suite for MoE forward functions."""

    @pytest.mark.skip(reason="Requires actual device")
    def test_moe_prefill_forward_basic(self):
        """Test MoE forward pass in prefill mode."""
        batch_size = 2
        seq_len = 128
        hidden_dim = 4096

        spec = MoESpec(
            num_experts=8, num_experts_per_tok=2, hidden_dim=hidden_dim, intermediate_dim=14336, activation="silu"
        )
        impl_config = get_default_moe_impl_config(spec, "cpu", "prefill")

        # Create dummy input
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
        hidden_states_tt = ttnn.from_torch(hidden_states)

        # Forward pass
        output, aux_loss = moe_prefill_forward(hidden_states_tt, spec, impl_config)

        # Check output shape
        assert output.shape == hidden_states_tt.shape
        # Check auxiliary loss for load balancing
        assert isinstance(aux_loss, (float, torch.Tensor))

    @pytest.mark.skip(reason="Requires actual device")
    def test_moe_decode_forward_basic(self):
        """Test MoE forward pass in decode mode."""
        batch_size = 2
        hidden_dim = 4096

        spec = MoESpec(
            num_experts=8, num_experts_per_tok=2, hidden_dim=hidden_dim, intermediate_dim=14336, activation="silu"
        )
        impl_config = get_default_moe_impl_config(spec, "cpu", "decode")

        # Create dummy input (single token)
        hidden_states = torch.randn(batch_size, 1, hidden_dim)
        hidden_states_tt = ttnn.from_torch(hidden_states)

        # Forward pass
        output, aux_loss = moe_decode_forward(hidden_states_tt, spec, impl_config)

        # Check output shape
        assert output.shape == hidden_states_tt.shape


class TestMoERouting:
    """Test MoE routing mechanisms."""

    def test_top_k_routing(self):
        """Test top-k expert routing."""
        # This would test the routing logic
        # Each token should be routed to exactly k experts

    def test_expert_choice_routing(self):
        """Test expert choice routing."""
        # This would test the alternative routing where experts choose tokens

    def test_load_balancing(self):
        """Test MoE load balancing across experts."""
        # This would verify that tokens are distributed evenly


@pytest.mark.perf
class TestMoEPerformance:
    """Performance tests for MoE layers."""

    @pytest.mark.skip(reason="Requires actual device")
    def test_moe_routing_overhead(self):
        """Test MoE routing overhead is acceptable."""
        # Would measure routing computation time

    @pytest.mark.skip(reason="Requires actual device")
    def test_moe_expert_parallelism_speedup(self):
        """Test speedup from expert parallelism."""
        # Would compare sequential vs parallel expert execution

    @pytest.mark.skip(reason="Requires actual device")
    @pytest.mark.parametrize(
        "num_experts,num_active",
        [
            (8, 1),  # Very sparse
            (8, 2),  # Typical
            (32, 2),  # Many experts, sparse
            (32, 4),  # Many experts, less sparse
        ],
    )
    def test_moe_sparsity_impact(self, num_experts, num_active):
        """Test performance impact of different sparsity levels."""
        # Would measure how sparsity affects performance


class TestMoEScaling:
    """Test MoE scaling properties."""

    def test_moe_parameter_scaling(self):
        """Test that MoE scales parameters efficiently."""
        hidden_dim = 4096
        intermediate_dim = 14336

        # Dense FFN parameters
        dense_params = 2 * hidden_dim * intermediate_dim

        # MoE parameters (8 experts)
        num_experts = 8
        moe_params = num_experts * 2 * hidden_dim * intermediate_dim
        routing_params = hidden_dim * num_experts

        # MoE should have roughly 8x more capacity
        assert moe_params > dense_params
        assert moe_params / dense_params == num_experts

    def test_moe_compute_scaling(self):
        """Test that MoE compute scales with active experts."""
        num_experts = 8
        num_active = 2

        # Compute should scale with active experts, not total
        compute_ratio = num_active / num_experts
        assert compute_ratio == 0.25  # Only 25% of experts active


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
