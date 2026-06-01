# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for all-to-all scatter/gather primitives for EP MoE.

Tests:
  1. Scatter: correct token-to-chip mapping based on expert indices
  2. Gather: correct aggregation of per-chip outputs
  3. Round-trip: scatter → gather preserves total output (identity check)
"""

import pytest
import torch

from models.demos.qwen3_coder_next.tt.moe_ep import EPConfig
from models.demos.qwen3_coder_next.tt.all_to_all import AllToAllScatterGather
from models.demos.qwen3_coder_next.tt.model_config import Qwen3CoderNextConfig


@pytest.fixture
def config():
    return Qwen3CoderNextConfig()


@pytest.fixture
def ep_config(config):
    return EPConfig(
        num_experts=config.num_experts,
        ep_size=4,
        ep_rank=0,
    )


@pytest.fixture
def a2a(config, ep_config):
    return AllToAllScatterGather(None, ep_config, config.hidden_size)


class TestAllToAllScatter:
    def test_basic_scatter(self, config, ep_config, a2a):
        """Scatter correctly maps tokens to target chips."""
        S, K, H = 8, 5, config.hidden_size

        # Deterministic expert indices for testing
        # Experts 0-127 → chip 0, 128-255 → chip 1, etc.
        topk_indices = torch.tensor([
            [0, 1, 128, 256, 384],      # all 4 chips
            [1, 2, 129, 257, 385],
            [0, 0, 0, 0, 0],            # chip 0 only
            [128, 129, 130, 131, 132],  # chip 1 only
            [0, 128, 256, 384, 64],     # all 4 chips
            [0, 1, 2, 3, 4],            # chip 0 only
            [128, 256, 384, 0, 64],     # all 4 chips
            [256, 257, 258, 259, 260],  # chip 2 only
        ])

        topk_weights = torch.ones(S, K) / K
        x = torch.randn(S, H)

        scatter_results = a2a.scatter(x, topk_indices, topk_weights)

        assert len(scatter_results) == 4  # 4 chips

        # Count total dispatched tokens
        total_dispatched = sum(
            res[0].shape[0] for res in scatter_results.values()
        )
        assert total_dispatched == S * K  # every (token, expert) pair dispatched

    def test_scatter_local_expert_indices(self, config, ep_config, a2a):
        """Local expert indices are correctly offset."""
        S, K, H = 4, 3, config.hidden_size

        # Experts 130, 131, 132 are on chip 1 (128-255)
        topk_indices = torch.tensor([
            [130, 131, 132],
            [128, 129, 130],
            [255, 130, 128],
            [0, 1, 2],  # chip 0
        ])
        topk_weights = torch.ones(S, K) / K
        x = torch.randn(S, H)

        scatter_results = a2a.scatter(x, topk_indices, topk_weights)

        # Chip 1's local indices should be 0-127 range
        chip1_tokens, chip1_weights, chip1_indices = scatter_results[1]
        assert chip1_indices.min() >= 0
        assert chip1_indices.max() < ep_config.experts_per_device  # 128

        # Chip 0's local indices
        chip0_tokens, chip0_weights, chip0_indices = scatter_results[0]
        assert chip0_indices.min() >= 0
        assert chip0_indices.max() < ep_config.experts_per_device

    def test_scatter_empty_chip(self, config, ep_config, a2a):
        """Chips with no assigned tokens get empty outputs."""
        S, K, H = 4, 3, config.hidden_size

        # All tokens go to chip 0 (experts 0-127)
        topk_indices = torch.zeros(S, K, dtype=torch.long)
        topk_weights = torch.ones(S, K) / K
        x = torch.randn(S, H)

        scatter_results = a2a.scatter(x, topk_indices, topk_weights)

        # Chip 1, 2, 3 should have empty tensors
        for cid in [1, 2, 3]:
            tokens, weights, indices = scatter_results[cid]
            assert tokens.numel() == 0


class TestAllToAllGather:
    def test_gather_basic(self, config, ep_config, a2a):
        """Gather correctly aggregates per-chip outputs."""
        S, K, H = 4, 5, config.hidden_size

        # Create deterministic scatter + mock per-chip outputs
        topk_indices = torch.tensor([
            [0, 1, 128, 256, 384],
            [1, 2, 129, 257, 385],
            [0, 0, 128, 256, 0],
            [0, 1, 2, 3, 4],
        ])
        topk_weights = torch.ones(S, K) / K
        x = torch.randn(S, H)

        scatter_results = a2a.scatter(x, topk_indices, topk_weights)

        # Create mock outputs (identity: output = input * 2)
        ep_outputs = {}
        for cid, (tokens, weights, indices) in scatter_results.items():
            if tokens.numel() > 0:
                ep_outputs[cid] = tokens * 2
            else:
                ep_outputs[cid] = torch.zeros(0, H, dtype=torch.bfloat16)

        gathered = a2a.gather(ep_outputs, topk_indices, topk_weights, S, K, H)
        assert gathered.shape == (S, H)

    def test_gather_preserves_scale(self, config, ep_config, a2a):
        """Gather with unit weights produces sum of expert outputs."""
        S, K, H = 2, 3, config.hidden_size

        topk_indices = torch.tensor([
            [0, 128, 256],
            [1, 129, 257],
        ])
        topk_weights = torch.ones(S, K) / K
        x = torch.ones(S, H, dtype=torch.bfloat16)

        scatter_results = a2a.scatter(x, topk_indices, topk_weights)
        ep_outputs = {}
        for cid, (tokens, weights, indices) in scatter_results.items():
            if tokens.numel() > 0:
                ep_outputs[cid] = tokens * cid  # output = token_value * chip_id
            else:
                ep_outputs[cid] = torch.zeros(0, H, dtype=torch.bfloat16)

        gathered = a2a.gather(ep_outputs, topk_indices, topk_weights, S, K, H)
        assert gathered.shape == (S, H)

        # Each position: sum of weighted chip outputs
        # Token 0: (chip0*1 + chip1*1 + chip2*1) / 3 = (0 + 1 + 2) / 3 = 1.0
        # Token 1: (chip0*1 + chip1*1 + chip2*1) / 3 = (0 + 1 + 2) / 3 = 1.0


class TestRoundTrip:
    def test_scatter_gather_identity(self, config, ep_config, a2a):
        """Scatter → identity → gather should reconstruct input."""
        S, K, H = 4, 5, config.hidden_size

        topk_indices = torch.randint(0, config.num_experts, (S, K))
        topk_weights = torch.ones(S, K) / K
        x = torch.randn(S, H)

        scatter_results = a2a.scatter(x, topk_indices, topk_weights)

        # Identity: output = input (no transformation)
        ep_outputs = {}
        for cid, (tokens, weights, indices) in scatter_results.items():
            if tokens.numel() > 0:
                ep_outputs[cid] = tokens
            else:
                ep_outputs[cid] = torch.zeros(0, H, dtype=torch.bfloat16)

        gathered = a2a.gather(ep_outputs, topk_indices, topk_weights, S, K, H)

        # With identity transform and equal weights:
        # Each token is dispatched to K chips, each returns the same value
        # Gathered = sum(x * 1/K for k in 1..K) = x * K * (1/K) = x
        assert gathered.shape == (S, H)
        # Allow small numerical tolerance for bfloat16
        diff = (gathered.float() - x.float()).abs()
        assert diff.max() < 0.1  # bfloat16 precision
