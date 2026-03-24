# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Host-side PCC test comparing ds_ref_moe (reference/model.py) vs tt_ref_moe (reference/tt/moe/moe.py).

This test validates that both PyTorch MoE implementations produce matching results:
- Random weights (scaled down from 671B config)
- ISL = 1024
- 256 routed experts, top-8 routing
- ds_ref_moe.Gate used for both (tt_ref_moe takes external gate outputs)

No TTNN, no device code - pure PyTorch comparison.
"""

import sys
from unittest.mock import MagicMock

# Mock the kernel module before importing reference.model
# (the kernel functions are only used for fp8 quantization, not needed for bf16 testing)
sys.modules["kernel"] = MagicMock()

import pytest
import torch
import torch.nn as nn
from loguru import logger

# Import reference modules from model.py
from models.demos.deepseek_v3_d_p.reference.model import MLP, Expert, Gate, Linear, ModelArgs

# Set Linear dtype to float32 for testing (default is bfloat16)
Linear.dtype = torch.float32

from models.demos.deepseek_v3_d_p.reference.tt.moe.moe import TorchMoe
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import ExpertMapping, compute_constants, get_gate_outputs


def compute_pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute Pearson Correlation Coefficient."""
    a_flat = a.float().flatten()
    b_flat = b.float().flatten()
    return torch.corrcoef(torch.stack([a_flat, b_flat]))[0, 1].item()


class DSRefMoENoGate(nn.Module):
    """
    MoE using reference/model.py Expert and MLP classes, but accepting external gate outputs.

    This implements the same computation as reference/model.py:MoE.forward(),
    but takes pre-computed weights and indices instead of computing them via gate.
    """

    def __init__(
        self,
        dim: int,
        n_routed_experts: int,
        moe_inter_dim: int,
        n_shared_experts: int,
    ):
        super().__init__()
        self.dim = dim
        self.n_routed_experts = n_routed_experts
        # Use Expert class from reference/model.py
        self.experts = nn.ModuleList([Expert(dim, moe_inter_dim) for _ in range(n_routed_experts)])
        # Use MLP class from reference/model.py for shared expert (scaled hidden dim)
        self.shared_experts = MLP(dim, n_shared_experts * moe_inter_dim)

    def forward(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward without gate - weights/indices provided externally.

        This matches the computation in reference/model.py:MoE.forward() exactly,
        just with externally provided gate outputs.

        Args:
            x: Input tensor (batch, seq_len, dim)
            weights: Gate weights (seq_len, topk)
            indices: Expert indices (seq_len, topk)

        Returns:
            Output tensor (batch, seq_len, dim)
        """
        shape = x.size()
        x = x.view(-1, self.dim)
        y = torch.zeros_like(x)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        for i in range(self.n_routed_experts):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            y[idx] += expert(x[idx]) * weights[idx, top, None]
        z = self.shared_experts(x)
        return (y + z).view(shape)


def create_shared_weights(
    emb_dim: int,
    hidden_dim: int,
    n_routed_experts: int,
    n_shared_experts: int,
    seed: int,
) -> tuple[list[dict], dict]:
    """
    Create random weights compatible with both implementations.

    Weight shapes (HF format: out_features x in_features):
    - gate_proj (w1): (hidden_dim, emb_dim)
    - up_proj (w3):   (hidden_dim, emb_dim)
    - down_proj (w2): (emb_dim, hidden_dim)

    Returns:
        routed_weights: List of dicts with gate_proj, up_proj, down_proj per expert
        shared_weights: Dict with gate_proj, up_proj, down_proj for shared expert
    """
    torch.manual_seed(seed)

    routed_weights = []
    for _ in range(n_routed_experts):
        weights = {
            "gate_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * 0.02,
            "up_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * 0.02,
            "down_proj": torch.randn(emb_dim, hidden_dim, dtype=torch.float32) * 0.02,
        }
        routed_weights.append(weights)

    # Shared expert has scaled hidden dim
    shared_hidden_dim = n_shared_experts * hidden_dim
    shared_weights = {
        "gate_proj": torch.randn(shared_hidden_dim, emb_dim, dtype=torch.float32) * 0.02,
        "up_proj": torch.randn(shared_hidden_dim, emb_dim, dtype=torch.float32) * 0.02,
        "down_proj": torch.randn(emb_dim, shared_hidden_dim, dtype=torch.float32) * 0.02,
    }

    return routed_weights, shared_weights


def initialize_ds_ref_moe(
    emb_dim: int,
    hidden_dim: int,
    n_routed_experts: int,
    n_shared_experts: int,
    routed_weights: list[dict],
    shared_weights: dict,
) -> DSRefMoENoGate:
    """
    Initialize ds_ref_moe with provided weights.

    Mapping from weight dict to reference/model.py Expert/MLP:
    - gate_proj -> w1.weight
    - up_proj   -> w3.weight
    - down_proj -> w2.weight
    """
    ds_moe = DSRefMoENoGate(
        dim=emb_dim,
        n_routed_experts=n_routed_experts,
        moe_inter_dim=hidden_dim,
        n_shared_experts=n_shared_experts,
    )

    # Load routed expert weights
    for i, weights in enumerate(routed_weights):
        with torch.no_grad():
            ds_moe.experts[i].w1.weight.copy_(weights["gate_proj"])
            ds_moe.experts[i].w3.weight.copy_(weights["up_proj"])
            ds_moe.experts[i].w2.weight.copy_(weights["down_proj"])

    # Load shared expert weights
    with torch.no_grad():
        ds_moe.shared_experts.w1.weight.copy_(shared_weights["gate_proj"])
        ds_moe.shared_experts.w3.weight.copy_(shared_weights["up_proj"])
        ds_moe.shared_experts.w2.weight.copy_(shared_weights["down_proj"])

    return ds_moe


def create_gate(emb_dim: int, n_routed_experts: int, num_experts_per_tok: int) -> Gate:
    """
    Create Gate from reference/model.py with test configuration.

    Uses ModelArgs to configure the gate properly.
    """
    args = ModelArgs(
        dim=emb_dim,
        n_routed_experts=n_routed_experts,
        n_activated_experts=num_experts_per_tok,
        n_expert_groups=1,
        n_limited_groups=1,
        score_func="softmax",
        route_scale=1.0,
    )
    gate = Gate(args)
    # Initialize gate weights
    with torch.no_grad():
        torch.nn.init.normal_(gate.weight, std=0.02)
    return gate


@pytest.mark.parametrize("seed", [42])
def test_moe_reference_pcc(seed: int):
    """
    Compare tt_ref_moe vs ds_ref_moe (without gate).

    Uses scaled-down dimensions from DeepSeek-V3 671B:
    - emb_dim: 7168 / 32 = 224
    - hidden_dim: 2048 / 32 = 64
    - ISL: 1024
    - 256 routed experts, top-8 routing
    """
    # Test configuration (scaled down from 671B)
    seq_len = 1024
    emb_dim = 224  # 7168 / 32
    hidden_dim = 64  # 2048 / 32
    n_routed_experts = 256
    num_experts_per_tok = 8
    n_shared_experts = 1
    batch_size = 1
    dispatch_group_size = 1  # Single "chip" for host-side test
    capacity_factor = 2.0

    logger.info(f"Test config: seed={seed}, seq_len={seq_len}, emb_dim={emb_dim}, hidden_dim={hidden_dim}")
    logger.info(f"  n_routed_experts={n_routed_experts}, num_experts_per_tok={num_experts_per_tok}")

    # Create shared weights for both implementations
    routed_weights, shared_weights = create_shared_weights(
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        n_routed_experts=n_routed_experts,
        n_shared_experts=n_shared_experts,
        seed=seed,
    )

    # 1. Create ds_ref_moe using reference/model.py Expert and MLP
    logger.info("Creating ds_ref_moe (using reference/model.py Expert, MLP)...")
    ds_moe = initialize_ds_ref_moe(
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        n_routed_experts=n_routed_experts,
        n_shared_experts=n_shared_experts,
        routed_weights=routed_weights,
        shared_weights=shared_weights,
    )

    # 2. Create gate using reference/model.py Gate
    logger.info("Creating gate (using reference/model.py Gate)...")
    gate = create_gate(
        emb_dim=emb_dim,
        n_routed_experts=n_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
    )

    # Compute derived constants for tt_ref_moe
    experts_per_chip, metadata_len, max_dispatched_tokens_per_expert = compute_constants(
        seq_len_per_chip=seq_len,
        num_routed_experts=n_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        num_devices=dispatch_group_size,
        dispatch_group_size=dispatch_group_size,
        capacity_factor=capacity_factor,
    )

    # Create expert dispatch table
    expert_dispatch_table = ExpertMapping.create_dispatch_table(
        num_routed_experts=n_routed_experts,
        dispatch_group_size=dispatch_group_size,
        num_dispatch_groups=1,
    )

    # 3. Create tt_ref_moe with same weights
    logger.info("Creating tt_ref_moe...")
    tt_moe = TorchMoe(
        dispatch_group_size=dispatch_group_size,
        experts_per_chip=experts_per_chip,
        num_routed_experts=n_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        metadata_len=metadata_len,
        max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
        seq_len_per_chip=seq_len,
        emb_dim=emb_dim,
        expert_dispatch_table=expert_dispatch_table,
        routed_expert_weights=routed_weights,
        shared_expert_weights=shared_weights,
    )

    # 4. Generate test input
    torch.manual_seed(seed + 1000)  # Different seed for input
    x = torch.randn(batch_size, seq_len, emb_dim, dtype=torch.float32)
    logger.info(f"Input shape: {x.shape}")

    # 5. Get gate outputs using reference/model.py Gate
    with torch.no_grad():
        x_flat = x.view(-1, emb_dim)
        weights, indices = gate(x_flat)
    logger.info(f"Gate outputs: weights={weights.shape}, indices={indices.shape}")

    # 6. Run ds_ref_moe
    logger.info("Running ds_ref_moe...")
    with torch.no_grad():
        ds_output = ds_moe(x, weights, indices)
    logger.info(f"ds_ref_moe output shape: {ds_output.shape}")

    # 7. Prepare inputs for tt_ref_moe
    # tt_ref_moe expects shape (dispatch_group_size, seq_len, emb_dim)
    x_tt = x.squeeze(0).unsqueeze(0)  # (1, seq_len, emb_dim)

    # Reshape weights and indices for tt_ref_moe
    weights_tt = weights.view(dispatch_group_size, seq_len, num_experts_per_tok)
    indices_tt = indices.view(dispatch_group_size, seq_len, num_experts_per_tok).to(torch.int32)

    # Compute expert_offsets and expert_token_counts
    expert_offsets, expert_token_counts, _ = get_gate_outputs(
        indices_tt,
        dispatch_group_size=dispatch_group_size,
        num_routed_experts=n_routed_experts,
        experts_per_chip=experts_per_chip,
        seq_len_per_chip=seq_len,
        num_experts_per_tok=num_experts_per_tok,
    )

    # 8. Run tt_ref_moe
    logger.info("Running tt_ref_moe...")
    with torch.no_grad():
        tt_output, _ = tt_moe(
            x_tt,
            weights_tt,
            indices_tt,
            expert_offsets,
            expert_token_counts,
            return_intermediates=False,
        )
    logger.info(f"tt_ref_moe output shape: {tt_output.shape}")

    # 9. Reshape tt_output to match ds_output
    tt_output_reshaped = tt_output.view(batch_size, seq_len, emb_dim)

    # 10. Compare with PCC
    pcc = compute_pcc(ds_output, tt_output_reshaped)
    logger.info(f"PCC: {pcc:.6f}")

    # Log some statistics
    logger.info(f"ds_output: min={ds_output.min():.4f}, max={ds_output.max():.4f}, mean={ds_output.mean():.4f}")
    logger.info(
        f"tt_output: min={tt_output_reshaped.min():.4f}, max={tt_output_reshaped.max():.4f}, mean={tt_output_reshaped.mean():.4f}"
    )

    # Check for NaN/Inf
    assert not torch.isnan(ds_output).any(), "ds_output contains NaN"
    assert not torch.isnan(tt_output_reshaped).any(), "tt_output contains NaN"
    assert not torch.isinf(ds_output).any(), "ds_output contains Inf"
    assert not torch.isinf(tt_output_reshaped).any(), "tt_output contains Inf"

    # Assert PCC threshold
    assert pcc >= 0.99, f"PCC {pcc:.6f} below threshold 0.99"

    logger.info("=" * 60)
    logger.info("TEST PASSED!")
    logger.info("=" * 60)
