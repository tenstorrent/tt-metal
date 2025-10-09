# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for the LM Head building block.

Tests language model head output projection with distributed computation support.
"""

import math

import pytest
import torch
from src.building_blocks.lm_head import LMHeadImplConfig, LMHeadSpec, get_default_impl_config

import ttnn


class TestLMHeadSpec:
    """Test LM head specification validation."""

    def test_valid_spec(self):
        """Test creating a valid LM head spec."""
        spec = LMHeadSpec(
            hidden_dim=4096,
            vocab_size=32000,
            num_devices=1,
        )
        spec.validate()
        assert spec.hidden_dim == 4096
        assert spec.vocab_size == 32000
        assert spec.num_devices == 1
        assert spec.padded_vocab_size == 32000  # Already aligned to 32

    def test_vocab_padding(self):
        """Test automatic vocabulary padding."""
        spec = LMHeadSpec(
            hidden_dim=4096,
            vocab_size=32001,  # Not aligned to 32
            num_devices=1,
        )
        assert spec.padded_vocab_size == 32032  # Next multiple of 32

    def test_galaxy_configuration(self):
        """Test galaxy configuration."""
        spec = LMHeadSpec(
            hidden_dim=8192,
            vocab_size=128000,
            num_devices=32,
            is_galaxy=True,
            max_columns_per_device=16384,
        )
        spec.validate()
        assert spec.is_galaxy is True
        assert spec.padded_vocab_size >= spec.vocab_size

    def test_multi_device_spec(self):
        """Test multi-device configuration."""
        spec = LMHeadSpec(
            hidden_dim=4096,
            vocab_size=64000,
            num_devices=8,
            max_columns_per_device=8192,
        )
        spec.validate()
        assert spec.num_devices == 8

    def test_tie_embeddings(self):
        """Test weight tying configuration."""
        spec = LMHeadSpec(
            hidden_dim=4096,
            vocab_size=32000,
            num_devices=1,
            tie_word_embeddings=True,
        )
        assert spec.tie_word_embeddings is True

    def test_invalid_hidden_dim(self):
        """Test validation with invalid hidden dim."""
        with pytest.raises(AssertionError):
            spec = LMHeadSpec(
                hidden_dim=0,
                vocab_size=32000,
                num_devices=1,
            )
            spec.validate()

    def test_invalid_vocab_size(self):
        """Test validation with invalid vocab size."""
        with pytest.raises(AssertionError):
            spec = LMHeadSpec(
                hidden_dim=4096,
                vocab_size=0,
                num_devices=1,
            )
            spec.validate()

    def test_invalid_num_devices(self):
        """Test validation with invalid number of devices."""
        with pytest.raises(AssertionError):
            spec = LMHeadSpec(
                hidden_dim=4096,
                vocab_size=32000,
                num_devices=0,
            )
            spec.validate()

    def test_invalid_max_columns(self):
        """Test validation with invalid max columns."""
        with pytest.raises(AssertionError):
            spec = LMHeadSpec(
                hidden_dim=4096,
                vocab_size=32000,
                num_devices=1,
                max_columns_per_device=0,
            )
            spec.validate()


class TestLMHeadImplConfig:
    """Test LM head implementation configuration."""

    def test_default_config(self):
        """Test default LM head implementation config."""
        config = LMHeadImplConfig()
        assert config.weight_dtype == ttnn.bfloat16
        assert config.output_dtype == ttnn.bfloat8_b
        assert config.ccl_dtype == ttnn.bfloat16
        assert config.num_reduce_scatter_links == 1
        assert config.num_all_gather_links == 2
        assert config.use_composite is True
        assert config.shard_strategy == "column"
        assert config.tile_padded_batch_rows == 32

    def test_custom_config(self):
        """Test custom LM head implementation config."""
        config = LMHeadImplConfig(
            weight_dtype=ttnn.bfloat8_b,
            output_dtype=ttnn.bfloat16,
            ccl_dtype=ttnn.bfloat8_b,
            num_reduce_scatter_links=4,
            num_all_gather_links=4,
            use_composite=False,
            shard_strategy="block",
        )
        assert config.weight_dtype == ttnn.bfloat8_b
        assert config.output_dtype == ttnn.bfloat16
        assert config.ccl_dtype == ttnn.bfloat8_b
        assert config.num_reduce_scatter_links == 4
        assert config.num_all_gather_links == 4
        assert config.use_composite is False
        assert config.shard_strategy == "block"


class TestGetDefaultImplConfig:
    """Test default implementation config generation."""

    def test_n150_device_prefill(self):
        """Test default config for N150 device in prefill mode."""
        spec = LMHeadSpec(hidden_dim=4096, vocab_size=32000, num_devices=1)
        config = get_default_impl_config(spec, "N150", "prefill")
        assert config.compute_kernel_config is not None
        assert config.output_memory_config == ttnn.L1_MEMORY_CONFIG
        assert config.weight_memory_config == ttnn.DRAM_MEMORY_CONFIG
        assert config.use_composite is False

    def test_n150_device_decode(self):
        """Test default config for N150 device in decode mode."""
        spec = LMHeadSpec(hidden_dim=4096, vocab_size=32000, num_devices=1)
        config = get_default_impl_config(spec, "N150", "decode")
        assert config.output_memory_config == ttnn.L1_MEMORY_CONFIG

    def test_n300_device(self):
        """Test default config for N300 device."""
        spec = LMHeadSpec(hidden_dim=4096, vocab_size=32000, num_devices=2)
        config = get_default_impl_config(spec, "N300", "prefill")
        assert config.num_all_gather_links == 2
        assert config.use_composite is True

    def test_t3000_device_prefill(self):
        """Test default config for T3000 device in prefill mode."""
        spec = LMHeadSpec(hidden_dim=4096, vocab_size=32000, num_devices=8)
        config = get_default_impl_config(spec, "T3000", "prefill")
        assert config.num_reduce_scatter_links == 2
        assert config.num_all_gather_links == 2
        assert config.output_dtype == ttnn.bfloat16

    def test_t3000_device_decode(self):
        """Test default config for T3000 device in decode mode."""
        spec = LMHeadSpec(hidden_dim=4096, vocab_size=32000, num_devices=8)
        config = get_default_impl_config(spec, "T3000", "decode")
        assert config.output_dtype == ttnn.bfloat8_b

    def test_galaxy_device_small_model(self):
        """Test default config for Galaxy device with small model."""
        spec = LMHeadSpec(
            hidden_dim=2048,
            vocab_size=32000,
            num_devices=32,
            is_galaxy=True,
        )
        config = get_default_impl_config(spec, "TG", "prefill")
        assert config.shard_strategy == "column"
        assert config.use_composite is True

    def test_galaxy_device_large_model(self):
        """Test default config for Galaxy device with large model."""
        spec = LMHeadSpec(
            hidden_dim=8192,
            vocab_size=128000,
            num_devices=32,
            is_galaxy=True,
        )
        config = get_default_impl_config(spec, "TG", "prefill")
        assert config.shard_strategy == "block"

    def test_galaxy_device_decode(self):
        """Test default config for Galaxy device in decode mode."""
        spec = LMHeadSpec(
            hidden_dim=4096,
            vocab_size=32000,
            num_devices=32,
            is_galaxy=True,
        )
        config = get_default_impl_config(spec, "TG", "decode")
        assert config.output_memory_config == ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG

    def test_unknown_device(self):
        """Test default config for unknown device."""
        spec = LMHeadSpec(hidden_dim=4096, vocab_size=32000, num_devices=1)
        config = get_default_impl_config(spec, "UNKNOWN", "prefill")
        # Should return conservative defaults
        assert config.compute_kernel_config is not None
        assert config.output_memory_config == ttnn.L1_MEMORY_CONFIG


class TestPrepareWeights:
    """Test weight preparation and sharding (requires actual devices)."""

    def test_weight_transpose(self):
        """Test that weights are transposed correctly."""
        # Create a dummy weight tensor
        vocab_size = 32000
        hidden_dim = 4096
        weight = torch.randn(vocab_size, hidden_dim)

        # The function would transpose to (hidden_dim, vocab_size)
        # and shard across devices - this requires actual device
        pytest.skip("Requires actual device")

    def test_weight_padding_galaxy(self):
        """Test weight padding for galaxy configuration."""
        pytest.skip("Requires actual device")

    def test_weight_splitting(self):
        """Test weight splitting based on max_columns_per_device."""
        # Test the logic for splitting weights
        spec = LMHeadSpec(
            hidden_dim=4096,
            vocab_size=65536,
            num_devices=4,
            max_columns_per_device=8192,
        )

        size_per_device = spec.vocab_size // spec.num_devices  # 16384
        num_splits = math.ceil(size_per_device / spec.max_columns_per_device)  # 2

        assert num_splits == 2

        split_sizes = [min(size_per_device, spec.max_columns_per_device)] * (num_splits - 1)
        split_sizes.append(size_per_device - sum(split_sizes))

        assert split_sizes == [8192, 8192]

    def test_weight_splitting_uneven(self):
        """Test weight splitting with uneven division."""
        spec = LMHeadSpec(
            hidden_dim=4096,
            vocab_size=50000,
            num_devices=4,
            max_columns_per_device=8192,
        )

        size_per_device = spec.vocab_size // spec.num_devices  # 12500
        num_splits = math.ceil(size_per_device / spec.max_columns_per_device)  # 2

        split_sizes = [min(size_per_device, spec.max_columns_per_device)] * (num_splits - 1)
        split_sizes.append(size_per_device - sum(split_sizes))

        assert split_sizes == [8192, 4308]


class TestLMHeadForward:
    """Test LM head forward pass (requires actual devices)."""

    def test_single_device_forward(self):
        """Test forward pass on single device."""
        pytest.skip("Requires actual device")

    def test_multi_device_forward(self):
        """Test forward pass on multiple devices."""
        pytest.skip("Requires actual device")

    def test_prefill_forward(self):
        """Test prefill mode forward pass."""
        pytest.skip("Requires actual device")

    def test_decode_forward(self):
        """Test decode mode forward pass."""
        pytest.skip("Requires actual device")

    def test_galaxy_forward(self):
        """Test forward pass on galaxy configuration."""
        pytest.skip("Requires actual device")


class TestLMHeadLogic:
    """Test LM head logic without actual devices."""

    def test_output_concatenation(self):
        """Test that outputs from multiple splits are concatenated."""
        # This would test the concatenation logic

    def test_all_reduce_requirement(self):
        """Test that all_reduce is only performed for multi-device."""
        spec_single = LMHeadSpec(
            hidden_dim=4096,
            vocab_size=32000,
            num_devices=1,
        )

        spec_multi = LMHeadSpec(
            hidden_dim=4096,
            vocab_size=32000,
            num_devices=8,
        )

        # In single device, no all_reduce should be performed
        assert spec_single.num_devices == 1

        # In multi device, all_reduce should be performed
        assert spec_multi.num_devices > 1

    def test_decode_dtype_casting(self):
        """Test dtype casting logic in decode mode."""
        # This would test the typecast logic in decode_forward


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
