# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tests for Gated DeltaNet linear attention."""

import pytest
import torch

from models.demos.qwen3_coder_next.tt.deltanet_attention import GatedDeltaNetAttention
from models.demos.qwen3_coder_next.tt.model_config import Qwen3CoderNextConfig


class TestGatedDeltaNetAttention:
    @pytest.fixture
    def config(self):
        return Qwen3CoderNextConfig()

    @pytest.fixture
    def module(self, config):
        return GatedDeltaNetAttention(config, layer_idx=0)

    def test_output_shape(self, config, module):
        """Output shape matches input shape."""
        x = torch.randn(2, 16, config.hidden_size)
        output, state, _, _ = module(x)
        assert output.shape == x.shape

    def test_recurrent_state_shape(self, config, module):
        """Recurrent state has correct shape."""
        x = torch.randn(2, 16, config.hidden_size)
        _, state, _, _ = module(x)
        # State: (batch, num_key_heads, key_dim, effective_value_dim)
        value_heads_per_key_head = config.linear_num_value_heads // config.linear_num_key_heads
        effective_value_dim = config.head_dim * value_heads_per_key_head
        assert state.shape == (2, config.linear_num_key_heads, config.linear_key_head_dim, effective_value_dim)

    def test_recurrent_state_passthrough(self, config, module):
        """State from one call can be passed to the next."""
        x1 = torch.randn(2, 8, config.hidden_size)
        x2 = torch.randn(2, 4, config.hidden_size)

        _, state1, _, _ = module(x1)
        output2, state2, _, _ = module(x2, recurrent_state=state1)

        assert output2.shape == (2, 4, config.hidden_size)
        assert state2.shape == state1.shape

    def test_deterministic(self, config, module):
        """Same input produces same output."""
        x = torch.randn(2, 16, config.hidden_size)
        out1, _, _, _ = module(x)
        out2, _, _, _ = module(x)
        torch.testing.assert_close(out1, out2)

    def test_single_token_decode(self, config, module):
        """Works with seq_len=1 (decode mode)."""
        x = torch.randn(2, 1, config.hidden_size)
        output, state, _, _ = module(x)
        assert output.shape == (2, 1, config.hidden_size)
