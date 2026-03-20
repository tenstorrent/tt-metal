"""
Torch reference implementation of MoE (Mixture of Experts) module.

This module orchestrates the full MoE pipeline:
1. Dispatch: Route tokens to expert buffers
2. Routed Experts: Process tokens in expert-specific buffers
3. Shared Expert: Process original input
4. Combine: Reconstruct outputs to original token positions
5. Split Connection: Apply gate weights and sum expert contributions
6. Final: Add routed output + shared output
"""

from typing import Optional

import torch
import torch.nn as nn
from loguru import logger

from models.demos.deepseek_v3_d_p.reference.tt.moe.combine import TorchCombineModule
from models.demos.deepseek_v3_d_p.reference.tt.moe.dispatch import TorchDispatchModule
from models.demos.deepseek_v3_d_p.reference.tt.moe.expert import TorchExpert
from models.demos.deepseek_v3_d_p.reference.tt.moe.moe_intermediates import MoEIntermediates
from models.demos.deepseek_v3_d_p.reference.tt.moe.reduce import TorchReduceModule
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import ExpertMapping
from models.tt_transformers.tt.load_checkpoints import load_hf_state_dict_filtered


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
        seq_len_per_chip: int,
        hidden_dim: int,
        expert_dispatch_table: torch.Tensor,
        model_id: str = None,
        layer_idx: int = None,
        num_dispatch_groups: int = 1,
        routed_expert_weights: list = None,
        shared_expert_weights: dict = None,
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
            max_dispatched_tokens_per_expert: Max tokens per expert buffer
            seq_len_per_chip: Sequence length per chip
            hidden_dim: Hidden dimension
            expert_dispatch_table: Expert to chip mapping table
            model_id: Optional HuggingFace model ID to load real weights from
            layer_idx: Optional layer index for weight loading (required if model_id is set)
            num_dispatch_groups: Number of dispatch groups (default: 1)
            routed_expert_weights: Optional list of dicts with gate_proj, up_proj, down_proj per expert
            shared_expert_weights: Optional dict with gate_proj, up_proj, down_proj for shared expert
        """
        super().__init__()
        self.dispatch_group_size = dispatch_group_size
        self.experts_per_chip = experts_per_chip
        self.num_routed_experts = num_routed_experts
        self.num_dispatch_groups = num_dispatch_groups

        # Create dispatch module
        self.dispatch_module = TorchDispatchModule(
            dispatch_group_size=dispatch_group_size,
            experts_per_chip=experts_per_chip,
            num_routed_experts=num_routed_experts,
            num_experts_per_tok=num_experts_per_tok,
            metadata_len=metadata_len,
            max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
            seq_len_per_chip=seq_len_per_chip,
            hidden_dim=hidden_dim,
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

        # Create experts
        use_identity = routed_weights is None
        self.routed_experts = nn.ModuleList(
            [
                TorchExpert(
                    hidden_dim,
                    hidden_dim,
                    torch_weights=routed_weights[i] if routed_weights else None,
                    use_identity=use_identity,
                )
                for i in range(num_routed_experts)
            ]
        )
        self.shared_expert = TorchExpert(
            hidden_dim, hidden_dim, torch_weights=shared_weights, use_identity=use_identity
        )

        # Create reduce module (sums over topk dimension)
        # topk_dim=1 because combined_output shape is (dispatch_group_size, seq_len, topk, hidden)
        self.reduce_module = TorchReduceModule(topk_dim=1)

    def forward(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        expert_offsets: torch.Tensor,
        expert_token_counts: torch.Tensor,
        return_intermediates: bool = False,
    ) -> tuple[torch.Tensor, Optional[MoEIntermediates]]:
        """
        Forward pass through the full MoE pipeline.

        Args:
            x: Input tensor (dispatch_group_size, seq_len_per_chip, hidden_dim)
            weights: Gate weights (dispatch_group_size, seq_len_per_chip, num_experts_per_tok)
            indices: Expert indices (dispatch_group_size, seq_len_per_chip, num_experts_per_tok)
            expert_offsets: Base offset for each expert from each chip
            expert_token_counts: Token counts per expert per chip
            return_intermediates: If True, return intermediate values for debugging

        Returns:
            final_output: MoE output (dispatch_group_size, seq_len_per_chip, hidden_dim)
            intermediates: Optional MoEIntermediates if return_intermediates=True
        """
        # Step 1: Run shared expert on original input
        with torch.no_grad():
            shared_output = self.shared_expert(x.float())

        # Step 2: Dispatch tokens to expert buffers
        dispatched_buffer, metadata = self.dispatch_module(x, weights, indices, expert_offsets)

        # Step 3: Run routed experts on dispatch buffer slices
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
                    token_count = expert_token_counts[group, chip, local_expert].item()

                    if token_count > 0:
                        expert_input = dispatched_buffer[group, chip, local_expert, :token_count, :]
                        with torch.no_grad():
                            expert_output = self.routed_experts[global_expert](expert_input.float())
                        expert_outputs[group, chip, local_expert, :token_count, :] = expert_output

        # Step 4: Combine routed expert outputs
        # TorchDispatchModule now outputs linearized mesh coords directly in metadata field 0,
        # so no transformation is needed before calling combine.
        combined_output = self.combine_module(expert_outputs, metadata, expert_token_counts)

        # Step 5: Apply gate weights and sum over topk
        # combined_output: (dispatch_group_size, seq_len, topk, hidden_dim)
        # routed_output: (dispatch_group_size, seq_len, hidden_dim)
        routed_output = self.reduce_module(combined_output, weights=weights)

        # Step 6: Final output = routed + shared
        final_output = routed_output + shared_output

        # Build intermediates if requested
        intermediates = None
        if return_intermediates:
            intermediates = MoEIntermediates(
                dispatched_buffer=dispatched_buffer,
                metadata=metadata,
                expert_outputs=expert_outputs,
                shared_output=shared_output,
                combined_output=combined_output,
                routed_output=routed_output,
            )

        return final_output, intermediates
