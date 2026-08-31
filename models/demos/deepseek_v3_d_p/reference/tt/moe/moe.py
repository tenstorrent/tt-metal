# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Torch reference implementation of MoE (Mixture of Experts) module.

This module orchestrates the full MoE pipeline:
1. Dispatch: Route tokens to expert buffers
2. Routed Experts: Process tokens in expert-specific buffers
3. Shared Expert: Process original input
4. Combine: Reconstruct outputs to original token positions
5. Split Connection: Apply gate weights and sum expert contributions
6. Final: Add routed output + shared output

Kimi-K3 adds a "LatentMoE" variant: the routed half of that pipeline (steps 1, 2, 4, 5) runs in a
reduced ``routed_emb_dim`` latent space, entered by a shared down-projection before dispatch and left
by a latent RMSNorm plus shared up-projection after the reduce. The gate and the shared expert still
see the full ``emb_dim`` input. Enabled by passing ``routed_emb_dim``; absent, behaviour is unchanged.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from models.demos.deepseek_v3_d_p.reference.tt.moe.combine import TorchCombineModule
from models.demos.deepseek_v3_d_p.reference.tt.moe.dispatch import TorchDispatchModule
from models.demos.deepseek_v3_d_p.reference.tt.moe.expert import ACTIVATION_SILU, TorchExpert
from models.demos.deepseek_v3_d_p.reference.tt.moe.moe_intermediates import MoEIntermediates
from models.demos.deepseek_v3_d_p.reference.tt.moe.reduce import TorchReduceModule
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import ExpertMapping, get_gate_outputs
from models.tt_transformers.tt.load_checkpoints import load_hf_state_dict_filtered


class TorchLatentMoeProjections(nn.Module):
    """The three tensors that wrap Kimi-K3's routed experts in a shared latent space.

    ``down_proj`` [routed_emb_dim, emb_dim] enters the latent space before dispatch; ``norm`` and
    ``up_proj`` [emb_dim, routed_emb_dim] exits it after the top-k weighted sum. HF weight
    convention throughout: ``(out_features, in_features)``.

    The RMSNorm math is a deliberate transcription of the vendored ``KimiRMSNorm``
    (fp32 accumulate, ``weight * normalised``) so the two references cannot disagree on eps handling.
    """

    def __init__(
        self,
        emb_dim: int,
        routed_emb_dim: int,
        torch_weights: dict = None,
        use_norm: bool = True,
        rms_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.routed_emb_dim = routed_emb_dim
        self.use_norm = use_norm
        self.rms_norm_eps = rms_norm_eps

        if torch_weights is not None:
            self.down_proj = nn.Parameter(torch_weights["down_proj"].float())
            self.up_proj = nn.Parameter(torch_weights["up_proj"].float())
            norm_weight = torch_weights.get("norm")
        else:
            self.down_proj = nn.Parameter(torch.randn(routed_emb_dim, emb_dim) * 0.02)
            self.up_proj = nn.Parameter(torch.randn(emb_dim, routed_emb_dim) * 0.02)
            norm_weight = None

        if use_norm:
            self.norm_weight = nn.Parameter(torch.ones(routed_emb_dim) if norm_weight is None else norm_weight.float())
        else:
            self.norm_weight = None

    def to_latent(self, x: torch.Tensor) -> torch.Tensor:
        """emb_dim -> routed_emb_dim, applied before dispatch.

        Casts to the weight dtype rather than assuming fp32: callers hand this module whatever the
        MoE input happens to be (``initialize_test_inputs`` produces bf16, the PCC tests fp32), and
        the surrounding reference already casts at each compute site the same way.
        """
        return F.linear(x.to(self.down_proj.dtype), self.down_proj)

    def from_latent(self, y: torch.Tensor) -> torch.Tensor:
        """routed_emb_dim -> emb_dim, applied after the top-k weighted sum (norm first)."""
        y = y.to(self.up_proj.dtype)
        if self.use_norm:
            dtype = y.dtype
            t = y.float()
            t = t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + self.rms_norm_eps)
            y = self.norm_weight * t.to(dtype)
        return F.linear(y, self.up_proj)


def load_moe_weights_from_hf(
    model_id: str,
    layer_idx: int,
    num_routed_experts: int,
) -> tuple[list[dict], dict]:
    """
    Load MoE weights from HuggingFace checkpoint.

    Args:
        model_id: HuggingFace model ID (e.g., "deepseek-ai/DeepSeek-V3")
        layer_idx: Layer index to load weights from
        num_routed_experts: Number of routed experts

    Returns:
        routed_expert_weights: List of dicts with gate_proj, up_proj, down_proj per expert
        shared_expert_weights: Dict with gate_proj, up_proj, down_proj for shared expert
    """
    # Build key prefixes for this layer's MoE
    prefixes = [f"model.layers.{layer_idx}.mlp."]
    state_dict = load_hf_state_dict_filtered(model_id, prefixes)

    # Extract routed expert weights
    routed_expert_weights = []
    for expert_idx in range(num_routed_experts):
        expert_weights = {
            "gate_proj": state_dict[f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight"],
            "up_proj": state_dict[f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight"],
            "down_proj": state_dict[f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight"],
        }
        routed_expert_weights.append(expert_weights)

    # Extract shared expert weights
    shared_expert_weights = {
        "gate_proj": state_dict[f"model.layers.{layer_idx}.mlp.shared_experts.gate_proj.weight"],
        "up_proj": state_dict[f"model.layers.{layer_idx}.mlp.shared_experts.up_proj.weight"],
        "down_proj": state_dict[f"model.layers.{layer_idx}.mlp.shared_experts.down_proj.weight"],
    }

    return routed_expert_weights, shared_expert_weights


class TorchMoe(nn.Module):
    """
    Minimal MoE module connecting dispatch -> experts -> combine -> split connection.

    This module orchestrates the full MoE pipeline:
    1. Dispatch: Route tokens to expert buffers
    2. Routed Experts: Process tokens in expert-specific buffers
    3. Shared Expert: Process original input
    4. Combine: Reconstruct outputs to original token positions
    5. Split Connection: Apply gate weights and sum expert contributions
    6. Final: Add routed output + shared output
    """

    def __init__(
        self,
        dispatch_group_size: int,
        experts_per_chip: int,
        num_routed_experts: int,
        num_experts_per_tok: int,
        metadata_len: int,
        max_dispatched_tokens_per_expert: int,
        max_dispatch_buffer_token_size: int,
        seq_len_per_chip: int,
        emb_dim: int,
        hidden_dim: int,
        expert_dispatch_table: torch.Tensor,
        model_id: str = None,
        layer_idx: int = None,
        num_dispatch_groups: int = 1,
        routed_expert_weights: list = None,
        shared_expert_weights: dict = None,
        gate_weights: dict = None,
        n_expert_groups: int = None,
        n_limited_groups: int = None,
        route_scale: float = None,
        routed_emb_dim: int = None,
        shared_hidden_dim: int = None,
        latent_weights: dict = None,
        latent_use_norm: bool = True,
        rms_norm_eps: float = 1e-5,
        activation: str = ACTIVATION_SILU,
        situ_beta: float = 1.0,
        situ_linear_beta: float | None = None,
        shared_activation: str | None = None,
    ):
        """
        Initialize MinimalMoE with configuration parameters.

        All sub-modules are created internally.

        Args:
            dispatch_group_size: Number of chips in dispatch group
            experts_per_chip: Number of experts per chip
            num_routed_experts: Total number of routed experts
            num_experts_per_tok: Number of experts each token routes to
            metadata_len: Length of metadata per token
            max_dispatched_tokens_per_expert: Per-expert theoretical upper bound on the
                number of tokens any single expert may receive (full sequence length).
            max_dispatch_buffer_token_size: Total token capacity of the flat dispatch
                buffer per chip (shared across all local experts).
            seq_len_per_chip: Sequence length per chip
            emb_dim: Embedding dimension (input/output dimension)
            hidden_dim: FFN intermediate dimension
            expert_dispatch_table: Expert to chip mapping table
            model_id: Optional HuggingFace model ID to load real weights from
            layer_idx: Optional layer index for weight loading (required if model_id is set)
            num_dispatch_groups: Number of dispatch groups (default: 1)
            routed_expert_weights: Optional list of dicts with gate_proj, up_proj, down_proj per expert
            shared_expert_weights: Optional dict with gate_proj, up_proj, down_proj for shared expert
            gate_weights: Optional dict with "weight" and "e_score_correction_bias" keys for gate
            routed_emb_dim: LatentMoE routed-side width. Defaults to emb_dim (no latent space).
                When set (Kimi-K3: 3584), dispatch / experts / combine / reduce all run at this
                width and the latent projections wrap them.
            shared_hidden_dim: Shared expert's FFN intermediate. Defaults to hidden_dim. K3 needs
                this separate because its shared expert is one MLP at moe_intermediate_size *
                num_shared_experts (6144), not at moe_intermediate_size (3072).
            latent_weights: Optional dict with down_proj / up_proj / norm for the latent projections.
            latent_use_norm / rms_norm_eps: latent RMSNorm control (K3: True / 1e-5).
            activation / situ_beta / situ_linear_beta: GLU activation for the ROUTED experts, and
                for the shared expert unless shared_activation overrides it. Defaults to "silu".
                Kimi-K3's routed experts run "situ" on device (RoutedExpertActivation.SituGlu).
            shared_activation: GLU activation for the SHARED expert; defaults to activation. K3 needs
                the split because its checkpoint uses SiTU for both but the device has a SiTU kernel
                for the routed expert only, so the shared side must stay on SiLU to compare.
        """
        super().__init__()

        # Build gate internally from gate_weights
        if gate_weights is not None:
            from types import SimpleNamespace

            from models.demos.deepseek_v3.reference.modeling_deepseek import MoEGate as ReferenceMoEGate

            assert route_scale is not None, "TorchMoe requires route_scale"
            assert n_expert_groups is not None, "TorchMoe requires n_expert_groups"
            assert n_limited_groups is not None, "TorchMoe requires n_limited_groups"

            ref_config = SimpleNamespace(
                num_experts_per_tok=num_experts_per_tok,
                n_routed_experts=num_routed_experts,
                routed_scaling_factor=route_scale,
                scoring_func="sigmoid",
                topk_method="noaux_tc",
                n_group=n_expert_groups,
                topk_group=n_limited_groups,
                norm_topk_prob=True,
                hidden_size=emb_dim,
            )
            self.gate = ReferenceMoEGate(ref_config, use_bitonic_sort=False)
            self.gate.weight.data = gate_weights["weight"]
            self.gate.e_score_correction_bias.data = gate_weights["e_score_correction_bias"]
        else:
            self.gate = None
        self.dispatch_group_size = dispatch_group_size
        self.experts_per_chip = experts_per_chip
        self.num_routed_experts = num_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.num_dispatch_groups = num_dispatch_groups
        self.seq_len_per_chip = seq_len_per_chip
        self.emb_dim = emb_dim
        self.expert_dispatch_table = expert_dispatch_table

        # Defaulting to emb_dim constructs no latent projections at all.
        self.routed_emb_dim = emb_dim if routed_emb_dim is None else routed_emb_dim
        self.shared_hidden_dim = hidden_dim if shared_hidden_dim is None else shared_hidden_dim
        self.use_latent_moe = self.routed_emb_dim != emb_dim
        self.latent_projections = (
            TorchLatentMoeProjections(
                emb_dim=emb_dim,
                routed_emb_dim=self.routed_emb_dim,
                torch_weights=latent_weights,
                use_norm=latent_use_norm,
                rms_norm_eps=rms_norm_eps,
            )
            if self.use_latent_moe
            else None
        )

        # Dispatch moves latent rows, halving the fabric bytes per dispatched token.
        self.dispatch_module = TorchDispatchModule(
            dispatch_group_size=dispatch_group_size,
            experts_per_chip=experts_per_chip,
            num_routed_experts=num_routed_experts,
            num_experts_per_tok=num_experts_per_tok,
            metadata_len=metadata_len,
            max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
            max_dispatch_buffer_token_size=max_dispatch_buffer_token_size,
            seq_len_per_chip=seq_len_per_chip,
            emb_dim=self.routed_emb_dim,
            num_dispatch_groups=num_dispatch_groups,
            expert_dispatch_table=expert_dispatch_table,
        )

        # Create combine module
        self.combine_module = TorchCombineModule(
            dispatch_group_size=dispatch_group_size,
            experts_per_chip=experts_per_chip,
            num_experts_per_tok=num_experts_per_tok,
            seq_len_per_chip=seq_len_per_chip,
            num_dispatch_groups=num_dispatch_groups,
        )

        # Determine weights source
        if routed_expert_weights is not None and shared_expert_weights is not None:
            routed_weights, shared_weights = routed_expert_weights, shared_expert_weights
        elif model_id is not None and layer_idx is not None:
            logger.debug(f"Loading MoE weights from {model_id}, layer {layer_idx}")
            routed_weights, shared_weights = load_moe_weights_from_hf(model_id, layer_idx, num_routed_experts)
        else:
            routed_weights, shared_weights = None, None

        # The shared expert stays at emb_dim on the pre-projection input, with its own intermediate.
        use_identity = routed_weights is None
        situ = dict(situ_beta=situ_beta, situ_linear_beta=situ_linear_beta)
        act = dict(activation=activation, **situ)
        shared_act = dict(activation=activation if shared_activation is None else shared_activation, **situ)
        self.routed_experts = nn.ModuleList(
            [
                TorchExpert(
                    self.routed_emb_dim,
                    hidden_dim,
                    torch_weights=routed_weights[i] if routed_weights else None,
                    use_identity=use_identity,
                    **act,
                )
                for i in range(num_routed_experts)
            ]
        )
        self.shared_expert = TorchExpert(
            emb_dim,
            self.shared_hidden_dim,
            torch_weights=shared_weights,
            use_identity=use_identity,
            **shared_act,
        )

        # Create reduce module (sums over topk dimension)
        # topk_dim=2 because combined_output shape is (dispatch_group_size, seq_len, topk, emb_dim)
        self.reduce_module = TorchReduceModule(topk_dim=2)

    def forward(
        self,
        x: torch.Tensor,
        weights: torch.Tensor = None,
        indices: torch.Tensor = None,
        expert_offsets: torch.Tensor = None,
        expert_token_counts: torch.Tensor = None,
        expert_region_offsets: torch.Tensor = None,
        return_intermediates: bool = False,
    ) -> tuple[torch.Tensor, Optional[MoEIntermediates]]:
        """
        Forward pass through the full MoE pipeline.

        Args:
            x: Input tensor (dispatch_group_size, seq_len_per_chip, emb_dim)
            weights: Gate weights (dispatch_group_size, seq_len_per_chip, num_experts_per_tok).
                    Optional if gate is set — will be computed internally.
            indices: Expert indices (dispatch_group_size, seq_len_per_chip, num_experts_per_tok).
                    Optional if gate is set — will be computed internally.
            expert_offsets: Base offset for each expert from each chip.
                    Optional if gate is set — will be computed internally.
            expert_token_counts: Token counts per expert per chip.
                    Optional if gate is set — will be computed internally.
            return_intermediates: If True, return intermediate values for debugging

        Returns:
            final_output: MoE output (dispatch_group_size, seq_len_per_chip, emb_dim)
            intermediates: Optional MoEIntermediates if return_intermediates=True
        """
        gate_scores = None
        gate_indices = None
        gate_logits = None

        # Gate path: compute weights/indices internally
        if self.gate is not None and weights is None:
            x_flat = x.view(-1, self.emb_dim)
            # doing it manually because we dont want to change reference module at the moemnt; this is without activation function;
            gate_logits = x_flat @ self.gate.weight.T  # (total_tokens, n_routed_experts)
            with torch.no_grad():
                # ReferenceMoEGate returns (topk_idx, topk_weight) — indices first, weights second
                indices, weights = self.gate(x_flat.unsqueeze(0))
            # Reshape to (dispatch_group_size, seq_len_per_chip, num_experts_per_tok)
            weights = weights.view(self.dispatch_group_size, self.seq_len_per_chip, self.num_experts_per_tok)
            indices = indices.view(self.dispatch_group_size, self.seq_len_per_chip, self.num_experts_per_tok).to(
                torch.int32
            )
            gate_scores = weights
            gate_indices = indices

            # Compute expert_offsets, expert_token_counts, and expert_region_offsets from indices
            expert_offsets, expert_token_counts, expert_region_offsets, _ = get_gate_outputs(
                indices,
                self.dispatch_group_size,
                self.num_routed_experts,
                self.experts_per_chip,
                self.seq_len_per_chip,
                self.num_experts_per_tok,
                expert_dispatch_table=self.expert_dispatch_table,
            )
        else:
            assert weights is not None and indices is not None
            assert expert_offsets is not None and expert_token_counts is not None
            assert expert_region_offsets is not None

        # Step 1: Run shared expert on original input.
        # Before the down-projection: the shared expert reads the full-width pre-projection hidden.
        with torch.no_grad():
            shared_output = self.shared_expert(x.float())

        # Step 1b: LatentMoE -- project into the latent space. Everything from here to the reduce runs at
        # routed_emb_dim.
        routed_input = self.latent_projections.to_latent(x) if self.use_latent_moe else x

        # Step 2: Dispatch tokens to expert buffers
        dispatched_buffer, metadata = self.dispatch_module(routed_input, weights, indices, expert_offsets)

        # Step 3: Run routed experts on dispatch buffer slices.
        # dispatched_buffer is 4D: (num_dispatch_groups, dispatch_group_size,
        # max_dispatch_buffer_token_size, routed_emb_dim -- == emb_dim without a latent space). Each expert's
        # token region lives at expert_region_offsets[group, chip, global_expert] within
        # the flat token dim (TILE_SIZE-aligned), matching the real dispatch kernel layout.
        expert_outputs = torch.zeros_like(dispatched_buffer)
        for group in range(self.num_dispatch_groups):
            for chip in range(self.dispatch_group_size):
                for local_expert in range(self.experts_per_chip):
                    # Map (group, chip, local_expert) to global_expert using column-major ordering
                    global_expert = ExpertMapping.get_global_expert_idx(
                        group,
                        chip,
                        local_expert,
                        self.experts_per_chip,
                        self.dispatch_group_size,
                        self.num_dispatch_groups,
                        is_col_major=True,
                    )
                    token_count = expert_token_counts[group, 0, global_expert].item()

                    if token_count > 0:
                        start = int(expert_region_offsets[group, chip, global_expert].item())
                        expert_input = dispatched_buffer[group, chip, start : start + token_count, :]
                        with torch.no_grad():
                            expert_output = self.routed_experts[global_expert](expert_input.float())
                        expert_outputs[group, chip, start : start + token_count, :] = expert_output

        # Step 4: Combine routed expert outputs
        # TorchDispatchModule now outputs linearized mesh coords directly in metadata field 0,
        # so no transformation is needed before calling combine.
        combined_output = self.combine_module(expert_outputs, metadata, expert_token_counts, expert_region_offsets)

        # Step 5: Apply gate weights and sum over topk
        # combined_output: (dispatch_group_size, seq_len, topk, routed_emb_dim)
        # routed_output: (dispatch_group_size, seq_len, routed_emb_dim)
        routed_output = self.reduce_module(combined_output, weights=weights)

        # Step 5b: LatentMoE -- project back out of the latent space: RMSNorm then up-projection back to emb_dim.
        # The norm sits after the weighted top-k sum, so it sees the summed latent.
        latent_routed_output = None
        if self.use_latent_moe:
            latent_routed_output = routed_output
            routed_output = self.latent_projections.from_latent(routed_output)

        # Step 6: Final output = routed + shared
        final_output = routed_output + shared_output

        # Build intermediates if requested
        intermediates = None
        if return_intermediates:
            intermediates = MoEIntermediates(
                gate_scores=gate_scores,
                gate_indices=gate_indices,
                gate_logits=gate_logits,
                dispatched_buffer=dispatched_buffer,
                metadata=metadata,
                expert_outputs=expert_outputs,
                shared_output=shared_output,
                combined_output=combined_output,
                routed_output=routed_output,
                latent_routed_output=latent_routed_output,
                latent_input=routed_input if self.use_latent_moe else None,
                expert_token_counts=expert_token_counts,
                expert_region_offsets=expert_region_offsets,
            )

        return final_output, intermediates
