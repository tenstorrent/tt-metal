# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

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

# Mock the kernel module before importing reference model
# (the kernel functions are only used for fp8 quantization, not needed for bf16 testing)
sys.modules["models.demos.deepseek_v3.reference.deepseek.kernel"] = MagicMock()

import pytest
import torch
import torch.nn as nn
from loguru import logger

import ttnn

# Import reference modules from model.py
from models.demos.deepseek_v3.reference.deepseek.model import MLP, Expert, Gate, Linear, ModelArgs

# Set Linear dtype to float32 for testing (default is bfloat16)
Linear.dtype = torch.float32

from models.demos.deepseek_v3_d_p.reference.kimi_k3.modeling_kimi_moe import KimiSparseMoeBlock
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
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


def test_moe_reference_pcc():
    """
    Compare tt_ref_moe vs ds_ref_moe (without gate).

    Uses scaled-down dimensions from DeepSeek-V3 671B:
    - emb_dim: 7168 / 32 = 224
    - hidden_dim: 2048 / 32 = 64
    - ISL: 1024
    - 256 routed experts, top-8 routing (topk=8 is gate constraint)
    """
    torch.manual_seed(42)

    # Test configuration (scaled down from 671B)
    seq_len = 1024
    emb_dim = 224  # 7168 / 32
    hidden_dim = 64  # 2048 / 32
    n_routed_experts = 256
    num_experts_per_tok = 8
    n_shared_experts = 1
    batch_size = 1
    dispatch_group_size = 1  # Single "chip" for host-side test
    # ceil(N/2) of the most conservative integer N such that dgs*seq*N >= theoretical
    # worst-case dispatch buffer. Real traffic never approaches the worst case.
    dispatch_buffer_capacity_factor = 8

    logger.debug(f"Test config: seq_len={seq_len}, emb_dim={emb_dim}, hidden_dim={hidden_dim}")
    logger.debug(f"  n_routed_experts={n_routed_experts}, num_experts_per_tok={num_experts_per_tok}")

    # Create shared weights for both implementations
    routed_weights, shared_weights = create_shared_weights(
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        n_routed_experts=n_routed_experts,
        n_shared_experts=n_shared_experts,
    )

    # 1. Create ds_ref_moe using reference/model.py Expert and MLP
    logger.debug("Creating ds_ref_moe (using reference/model.py Expert, MLP)...")
    ds_moe = initialize_ds_ref_moe(
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        n_routed_experts=n_routed_experts,
        n_shared_experts=n_shared_experts,
        routed_weights=routed_weights,
        shared_weights=shared_weights,
    )

    # 2. Create gate using reference/model.py Gate
    logger.debug("Creating gate (using reference/model.py Gate)...")
    gate = create_gate(
        emb_dim=emb_dim,
        n_routed_experts=n_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
    )

    # Compute derived constants for tt_ref_moe
    (
        experts_per_chip,
        metadata_len,
        max_dispatch_buffer_token_size,
        max_dispatched_tokens_per_expert,
    ) = compute_constants(
        seq_len,
        n_routed_experts,
        num_experts_per_tok,
        dispatch_group_size,
        dispatch_group_size,
        dispatch_buffer_capacity_factor,
    )

    # Create expert dispatch table
    expert_dispatch_table = ExpertMapping.create_dispatch_table(
        num_routed_experts=n_routed_experts,
        dispatch_group_size=dispatch_group_size,
        num_dispatch_groups=1,
    )

    # 3. Create tt_ref_moe with same weights
    logger.debug("Creating tt_ref_moe...")
    tt_moe = TorchMoe(
        dispatch_group_size=dispatch_group_size,
        experts_per_chip=experts_per_chip,
        num_routed_experts=n_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        metadata_len=metadata_len,
        max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
        max_dispatch_buffer_token_size=max_dispatch_buffer_token_size,
        seq_len_per_chip=seq_len,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        expert_dispatch_table=expert_dispatch_table,
        routed_expert_weights=routed_weights,
        shared_expert_weights=shared_weights,
    )

    # 4. Generate test input
    x = torch.randn(batch_size, seq_len, emb_dim, dtype=torch.float32)
    logger.debug(f"Input shape: {x.shape}")

    # 5. Get gate outputs using reference/model.py Gate
    with torch.no_grad():
        x_flat = x.view(-1, emb_dim)
        weights, indices = gate(x_flat)
    logger.debug(f"Gate outputs: weights={weights.shape}, indices={indices.shape}")

    # 6. Run ds_ref_moe
    logger.debug("Running ds_ref_moe...")
    with torch.no_grad():
        ds_output = ds_moe(x, weights, indices)
    logger.debug(f"ds_ref_moe output shape: {ds_output.shape}")

    # 7. Prepare inputs for tt_ref_moe
    # tt_ref_moe expects shape (dispatch_group_size, seq_len, emb_dim)
    x_tt = x.squeeze(0).unsqueeze(0)  # (1, seq_len, emb_dim)

    # Reshape weights and indices for tt_ref_moe
    weights_tt = weights.view(dispatch_group_size, seq_len, num_experts_per_tok)
    indices_tt = indices.view(dispatch_group_size, seq_len, num_experts_per_tok).to(torch.int32)

    # Compute expert_offsets, expert_token_counts, and expert_region_offsets
    expert_offsets, expert_token_counts, expert_region_offsets, _ = get_gate_outputs(
        indices_tt,
        dispatch_group_size=dispatch_group_size,
        num_routed_experts=n_routed_experts,
        experts_per_chip=experts_per_chip,
        seq_len_per_chip=seq_len,
        num_experts_per_tok=num_experts_per_tok,
        expert_dispatch_table=expert_dispatch_table,
    )

    # 8. Run tt_ref_moe
    logger.debug("Running tt_ref_moe...")
    with torch.no_grad():
        tt_output, _ = tt_moe(
            x_tt,
            weights_tt,
            indices_tt,
            expert_offsets,
            expert_token_counts,
            expert_region_offsets,
            return_intermediates=False,
        )
    logger.debug(f"tt_ref_moe output shape: {tt_output.shape}")

    # 9. Reshape tt_output to match ds_output
    tt_output_reshaped = tt_output.view(batch_size, seq_len, emb_dim)

    # 10. Compare with PCC
    pcc = compute_pcc(ds_output, tt_output_reshaped)
    logger.debug(f"PCC: {pcc:.6f}")

    # Log some statistics
    logger.debug(f"ds_output: min={ds_output.min():.4f}, max={ds_output.max():.4f}, mean={ds_output.mean():.4f}")
    logger.debug(
        f"tt_output: min={tt_output_reshaped.min():.4f}, max={tt_output_reshaped.max():.4f}, mean={tt_output_reshaped.mean():.4f}"
    )

    # Check for NaN/Inf
    assert not torch.isnan(ds_output).any(), "ds_output contains NaN"
    assert not torch.isnan(tt_output_reshaped).any(), "tt_output contains NaN"
    assert not torch.isinf(ds_output).any(), "ds_output contains Inf"
    assert not torch.isinf(tt_output_reshaped).any(), "tt_output contains Inf"

    # Assert PCC threshold
    assert pcc >= 0.99, f"PCC {pcc:.6f} below threshold 0.99"

    logger.debug("=" * 60)
    logger.debug("TEST PASSED!")
    logger.debug("=" * 60)


# ---------------------------------------------------------------------------
# Kimi-K3 LatentMoE
# ---------------------------------------------------------------------------


def _k3_test_config(emb_dim, latent_dim, moe_inter, n_routed, topk, n_shared, activation, use_norm=True):
    """A scaled-down but structurally exact Kimi-K3 text config.

    Ratios that matter are preserved: ``routed_expert_hidden_size == hidden_size / 2`` (K3:
    3584/7168) and one shared expert MLP at ``moe_intermediate_size * num_shared_experts``.
    """
    from models.demos.deepseek_v3_d_p.reference.kimi_k3.configuration_kimi_k3 import KimiLinearConfig

    return KimiLinearConfig(
        hidden_size=emb_dim,
        routed_expert_hidden_size=latent_dim,
        moe_intermediate_size=moe_inter,
        intermediate_size=moe_inter * 4,  # dense layer-0 FFN; unused by the MoE block
        num_experts=n_routed,
        num_experts_per_token=topk,
        num_shared_experts=n_shared,
        num_expert_group=1,
        topk_group=1,
        moe_renormalize=True,
        moe_router_activation_func="sigmoid",
        routed_scaling_factor=1.0,
        latent_moe_use_norm=use_norm,
        hidden_act=activation,
        activation_situ_beta=KimiK3Config.ACTIVATION_SITU_BETA,
        activation_situ_linear_beta=KimiK3Config.ACTIVATION_SITU_LINEAR_BETA,
        rms_norm_eps=KimiK3Config.RMS_NORM_EPS,
    )


def _tt_moe_from_kimi_block(blk, cfg, *, seq_len, dispatch_group_size, capacity_factor, activation):
    """Build a ``TorchMoe`` that mirrors ``blk`` tensor for tensor.

    The K3 -> TT weight-name remap lives here and nowhere else:
      * routed experts ``w1 -> gate_proj``, ``w3 -> up_proj``, ``w2 -> down_proj``
      * shared expert keeps ``gate_proj`` / ``up_proj`` / ``down_proj``
      * the latent trio maps straight across
    """
    routed_weights = [
        {
            "gate_proj": e.w1.weight.detach().clone(),
            "up_proj": e.w3.weight.detach().clone(),
            "down_proj": e.w2.weight.detach().clone(),
        }
        for e in blk.experts
    ]
    shared_weights = {
        "gate_proj": blk.shared_experts.gate_proj.weight.detach().clone(),
        "up_proj": blk.shared_experts.up_proj.weight.detach().clone(),
        "down_proj": blk.shared_experts.down_proj.weight.detach().clone(),
    }
    latent_weights = {
        "down_proj": blk.routed_expert_down_proj.weight.detach().clone(),
        "up_proj": blk.routed_expert_up_proj.weight.detach().clone(),
        # Present only when the latent norm is enabled; upstream does not construct it otherwise.
        "norm": blk.routed_expert_norm.weight.detach().clone() if blk.latent_moe_use_norm else None,
    }

    (
        experts_per_chip,
        metadata_len,
        max_dispatch_buffer_token_size,
        max_dispatched_tokens_per_expert,
    ) = compute_constants(
        seq_len,
        cfg.num_experts,
        cfg.num_experts_per_token,
        dispatch_group_size,
        dispatch_group_size,
        capacity_factor,
    )
    expert_dispatch_table = ExpertMapping.create_dispatch_table(
        num_routed_experts=cfg.num_experts,
        dispatch_group_size=dispatch_group_size,
        num_dispatch_groups=1,
    )

    tt_moe = TorchMoe(
        dispatch_group_size=dispatch_group_size,
        experts_per_chip=experts_per_chip,
        num_routed_experts=cfg.num_experts,
        num_experts_per_tok=cfg.num_experts_per_token,
        metadata_len=metadata_len,
        max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
        max_dispatch_buffer_token_size=max_dispatch_buffer_token_size,
        seq_len_per_chip=seq_len,
        emb_dim=cfg.hidden_size,
        hidden_dim=cfg.moe_intermediate_size,
        expert_dispatch_table=expert_dispatch_table,
        routed_expert_weights=routed_weights,
        shared_expert_weights=shared_weights,
        # --- the K3 deltas ---
        routed_emb_dim=cfg.routed_expert_hidden_size,
        shared_hidden_dim=cfg.moe_intermediate_size * cfg.num_shared_experts,
        latent_weights=latent_weights,
        latent_use_norm=cfg.latent_moe_use_norm,
        rms_norm_eps=cfg.rms_norm_eps,
        activation=activation,
        situ_beta=cfg.activation_situ_beta,
        situ_linear_beta=cfg.activation_situ_linear_beta,
    )
    return tt_moe, experts_per_chip, expert_dispatch_table


@pytest.mark.parametrize(
    "activation, latent_use_norm",
    [
        # The checkpoint's real combination.
        ("situ", True),
        # What the device still runs outside the routed experts (shared expert, dense FFN).
        ("silu", True),
        # Upstream's own default, even though K3's checkpoint sets it true.
        ("situ", False),
    ],
    ids=["situ", "silu", "situ-no-latent-norm"],
)
def test_kimi_k3_latent_moe_reference_pcc(activation, latent_use_norm):
    """Compare ``TorchMoe``'s LatentMoE path against upstream ``KimiSparseMoeBlock``.

    Host only -- no TTNN, no device. This is the gate that pins the K3 MoE *math* before any device
    work: down-projection into the latent space, experts at the reduced width, top-k weighted sum in
    latent space, latent RMSNorm, up-projection, plus the shared expert on the *pre*-projection
    input.

    The gate is deliberately factored out: ``KimiSparseMoeBlock`` computes routing internally, so we
    call its ``gate`` once and hand the same ``(indices, weights)`` to ``TorchMoe``. That isolates
    the dataflow under test from any gate-implementation difference -- gate parity is covered by the
    device-side gate tests.

    Both activations are exercised: ``situ`` is what the checkpoint does and what the routed experts
    now run on device (#51351), and ``silu`` is what the shared expert and the dense FFN still run,
    having no SiTU kernel at their widths. Testing both means each half of that split is validated
    against upstream rather than assumed.
    """
    torch.manual_seed(42)

    # Structurally exact, scaled down ~28x on the hidden dim. Latent is exactly half of emb, as in
    # K3 (3584 / 7168), and both are tile-aligned.
    seq_len = 1024
    emb_dim = 256
    latent_dim = 128
    moe_inter = 64
    n_routed_experts = 64
    num_experts_per_tok = 8
    n_shared_experts = 2
    dispatch_group_size = 1

    # Size the dispatch buffer from the actual worst case rather than a magic number.
    # get_gate_outputs pads each expert's token count up to a TILE_SIZE boundary, so the buffer must
    # hold the dispatched tokens plus up to TILE_SIZE-1 padding per expert; underestimating it
    # overflows inside TorchDispatchModule with a bare IndexError.
    worst_case_tokens = seq_len * num_experts_per_tok + n_routed_experts * (ttnn.TILE_SIZE - 1)
    capacity_factor = -(-worst_case_tokens // (dispatch_group_size * seq_len))  # ceil-div

    cfg = _k3_test_config(
        emb_dim,
        latent_dim,
        moe_inter,
        n_routed_experts,
        num_experts_per_tok,
        n_shared_experts,
        activation,
        use_norm=latent_use_norm,
    )
    blk = KimiSparseMoeBlock(cfg).eval()
    with torch.no_grad():
        torch.nn.init.normal_(blk.gate.e_score_correction_bias, std=0.02)

    logger.debug(
        f"K3 LatentMoE: act={activation} latent_norm={latent_use_norm} emb={emb_dim} latent={latent_dim} "
        f"moe_inter={moe_inter} shared_inter={moe_inter * n_shared_experts} experts={n_routed_experts} "
        f"topk={num_experts_per_tok}"
    )
    # Guard the structure itself: if upstream ever stops taking the latent path for this config the
    # test would silently degrade into a plain-MoE comparison.
    assert blk.use_latent_moe, "upstream did not take the latent path; routed_expert_hidden_size ignored?"
    assert blk.latent_moe_use_norm == latent_use_norm, "latent RMSNorm presence does not match the config"
    assert hasattr(blk, "routed_expert_norm") == latent_use_norm, "routed_expert_norm construction mismatch"
    assert blk.experts[0].w1.weight.shape == (moe_inter, latent_dim), "routed experts are not at the latent width"
    assert blk.shared_experts.gate_proj.weight.shape == (moe_inter * n_shared_experts, emb_dim)

    tt_moe, experts_per_chip, expert_dispatch_table = _tt_moe_from_kimi_block(
        blk,
        cfg,
        seq_len=seq_len,
        dispatch_group_size=dispatch_group_size,
        capacity_factor=capacity_factor,
        activation=activation,
    )

    x = torch.randn(1, seq_len, emb_dim, dtype=torch.float32)

    # Reference: full upstream block, gate included.
    with torch.no_grad():
        ref_output = blk(x)

    # Same routing decisions for TorchMoe, taken from the same gate.
    with torch.no_grad():
        indices, weights = blk.gate(x)
    weights_tt = weights.view(dispatch_group_size, seq_len, num_experts_per_tok).float()
    indices_tt = indices.view(dispatch_group_size, seq_len, num_experts_per_tok).to(torch.int32)

    expert_offsets, expert_token_counts, expert_region_offsets, _ = get_gate_outputs(
        indices_tt,
        dispatch_group_size=dispatch_group_size,
        num_routed_experts=n_routed_experts,
        experts_per_chip=experts_per_chip,
        seq_len_per_chip=seq_len,
        num_experts_per_tok=num_experts_per_tok,
        expert_dispatch_table=expert_dispatch_table,
    )

    with torch.no_grad():
        tt_output, inter = tt_moe(
            x.squeeze(0).unsqueeze(0),
            weights_tt,
            indices_tt,
            expert_offsets,
            expert_token_counts,
            expert_region_offsets,
            return_intermediates=True,
        )

    # The latent tensors must actually be at the reduced width, otherwise the test would pass while
    # silently running the whole thing at emb_dim.
    assert inter.latent_input.shape[-1] == latent_dim, inter.latent_input.shape
    assert inter.dispatched_buffer.shape[-1] == latent_dim, inter.dispatched_buffer.shape
    assert inter.latent_routed_output.shape[-1] == latent_dim, inter.latent_routed_output.shape
    assert inter.routed_output.shape[-1] == emb_dim, inter.routed_output.shape

    tt_output_reshaped = tt_output.view(1, seq_len, emb_dim)
    pcc = compute_pcc(ref_output, tt_output_reshaped)
    logger.debug(f"PCC vs upstream KimiSparseMoeBlock: {pcc:.6f}")
    logger.debug(f"ref: min={ref_output.min():.4f} max={ref_output.max():.4f} mean={ref_output.mean():.4f}")
    logger.debug(
        f"tt : min={tt_output_reshaped.min():.4f} max={tt_output_reshaped.max():.4f} "
        f"mean={tt_output_reshaped.mean():.4f}"
    )

    assert torch.isfinite(ref_output).all(), "reference output is not finite"
    assert torch.isfinite(tt_output_reshaped).all(), "TorchMoe output is not finite"
    # Both sides are fp32 torch computing the same graph in a different order, so this should be
    # near-exact; 0.999 leaves room only for reduction-order noise.
    assert pcc >= 0.999, f"PCC {pcc:.6f} below threshold 0.999"
