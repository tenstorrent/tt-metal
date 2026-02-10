# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""MoEGate router implementation for DeepSeek-V3 style routing."""

import torch

import ttnn

try:
    from ...utils.debug_logger import log_op, log_tensor_props
except ImportError:
    try:
        from models.tt_moe.utils.debug_logger import log_op, log_tensor_props
    except ImportError:
        from utils.debug_logger import log_op, log_tensor_props

try:
    from .base_router import BaseRouter
except ImportError:
    from components.routers.base_router import BaseRouter


class MoEGateRouter(BaseRouter):
    """
    MoEGate router with score correction bias and expert scaling.

    This router implements the DeepSeek-V3 style routing with:
    - Score correction bias for expert selection
    - Expert scaling factor for output weighting
    - Group-based top-k selection
    """

    def __init__(self, config: dict, mesh_device: ttnn.MeshDevice):
        """
        Initialize MoEGate router.

        Args:
            config: Router configuration containing:
                - num_experts: Total number of experts
                - num_experts_per_tok: Number of experts to select per token
                - n_routed_experts: Number of routed experts
                - n_group: Number of expert groups
                - topk_group: Top-k groups to select
                - score_correction_bias: Whether to apply score correction
                - routed_scaling_factor: Scaling factor for expert outputs
                - memory_config: Memory configuration string
            mesh_device: TTNN mesh device for tensor placement
        """
        self.config = config
        self.mesh_device = mesh_device

        self.num_experts = config["num_experts"]
        self.num_experts_per_tok = config["num_experts_per_tok"]
        self.n_routed_experts = config.get("n_routed_experts", self.num_experts)
        self.n_group = config.get("n_group", 1)
        self.topk_group = config.get("topk_group", 1)
        self.score_correction_bias_enabled = config.get("score_correction_bias", False)
        self.routed_scaling_factor = config.get("routed_scaling_factor", 1.0)

        # Memory configuration
        memory_config_str = config.get("memory_config", "L1_MEMORY_CONFIG")
        self.memory_config = getattr(ttnn, memory_config_str)

        # Compute kernel configuration
        compute_config_str = config.get("compute_kernel_config", "HIFI2")
        if compute_config_str:
            self.compute_config = self._get_compute_config(compute_config_str)
        else:
            self.compute_config = None

        # Weights will be loaded later
        self.gate_proj = None
        self.score_correction_bias = None
        self.expert_scale = None

        # Group mask tensors for scatter operation
        self.scatter_input_mask = None
        self.scatter_src_tensor = None

    def _get_compute_config(self, config_str: str):
        """Get compute kernel configuration based on string identifier."""
        if config_str == "HIFI2":
            return ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            )
        elif config_str == "HIFI4":
            return ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            )
        else:
            return None

    def load_weights(self, state_dict: dict, weight_path: str = None):
        """
        Load router weights from state dict.

        Args:
            state_dict: Dictionary containing:
                - weight: Gate projection weight [n_routed_experts, hidden_size]
                - e_score_correction_bias: Score correction bias [n_routed_experts]
            weight_path: Optional path for cached weights
        """
        # Gate projection weight
        if "weight" in state_dict:
            self.gate_proj = ttnn.from_torch(
                state_dict["weight"].T.unsqueeze(0).unsqueeze(0),
                device=self.mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        # Score correction bias (if enabled)
        if self.score_correction_bias_enabled and "e_score_correction_bias" in state_dict:
            self.score_correction_bias = ttnn.from_torch(
                state_dict["e_score_correction_bias"].unsqueeze(0).unsqueeze(0).unsqueeze(0),
                device=self.mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.float32,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        # Expert scaling factor
        self.expert_scale = ttnn.from_torch(
            torch.tensor([self.routed_scaling_factor]).repeat(1, self.num_experts_per_tok).unsqueeze(0).unsqueeze(0),
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Initialize scatter masks for group selection
        self.scatter_input_mask = ttnn.from_torch(
            torch.full((1, 1, 1, self.n_group), -float("inf")),
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        self.scatter_src_tensor = ttnn.from_torch(
            torch.ones((1, 1, 1, self.topk_group)),
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

    def forward(self, x: ttnn.Tensor, mode: str = "decode"):
        """
        Forward pass through MoEGate router.

        This implements the DeepSeek MoEGate logic:
        1. Linear projection to get expert logits
        2. Sigmoid activation to get scores
        3. Add score correction bias
        4. Group-based expert selection (if groups > 1)
        5. Top-k expert selection
        6. Score normalization and scaling

        Args:
            x: Input tensor [batch, seq_len, hidden_dim] or flattened
            mode: "decode" or "prefill" mode

        Returns:
            Tuple of (weights, indices) for expert selection
        """
        log_tensor_props("MoEGate input", x)

        if self.gate_proj is None:
            raise ValueError("Weights not loaded. Call load_weights first.")

        # Gate projection
        log_op(
            "ttnn.linear (gate_proj)",
            inputs=x,
            config={"memory_config": str(self.memory_config), "compute_kernel_config": str(self.compute_config)},
            output=None,
        )
        logits = ttnn.linear(
            x,
            self.gate_proj,
            memory_config=self.memory_config,
            compute_kernel_config=self.compute_config,
        )
        log_tensor_props("  output", logits)

        # Sigmoid activation to get scores
        log_op("ttnn.sigmoid", inputs=logits, output=None)
        scores = ttnn.sigmoid(logits)
        log_tensor_props("  output", scores)
        ttnn.deallocate(logits)

        # Add score correction bias if enabled
        if self.score_correction_bias_enabled and self.score_correction_bias is not None:
            # Expand bias to match scores shape
            scores_correction_bias = ttnn.repeat(self.score_correction_bias, ttnn.Shape((1, 1, scores.shape[2], 1)))
            scores_correction_bias = ttnn.to_layout(scores_correction_bias, ttnn.TILE_LAYOUT)
            scores_with_bias = ttnn.add(
                scores,
                scores_correction_bias,
                memory_config=self.memory_config,
                dtype=ttnn.bfloat16,
            )
            ttnn.deallocate(scores_correction_bias)
        else:
            scores_with_bias = scores

        # Group-based selection if n_group > 1
        if self.n_group > 1:
            # Reshape scores to expert groups
            expert_scores_grouped = ttnn.reshape(
                scores_with_bias, (1, -1, self.n_group, self.n_routed_experts // self.n_group)
            )

            # Get top-2 scores within expert groups
            TOPK_MIN_WIDTH = 64  # Minimum width for topk op
            if expert_scores_grouped.shape[3] < TOPK_MIN_WIDTH:
                expert_scores_grouped = ttnn.pad(
                    expert_scores_grouped,
                    [(0, 0), (0, 0), (0, 0), (0, TOPK_MIN_WIDTH - expert_scores_grouped.shape[3])],
                    value=-float("inf"),
                )

            topk_scores_within_groups, _ = ttnn.topk(expert_scores_grouped, k=2, dim=-1)
            ttnn.deallocate(expert_scores_grouped)

            # Sum top-2 scores within groups to get group scores
            expert_group_scores = ttnn.sum(topk_scores_within_groups, dim=3)
            ttnn.deallocate(topk_scores_within_groups)
            expert_group_scores = ttnn.unsqueeze(expert_group_scores, dim=0)

            # Get top-k expert groups
            if expert_group_scores.shape[3] < TOPK_MIN_WIDTH:
                expert_group_scores = ttnn.pad(
                    expert_group_scores,
                    [(0, 0), (0, 0), (0, 0), (0, TOPK_MIN_WIDTH - expert_group_scores.shape[3])],
                    value=-float("inf"),
                )

            _, topk_group_indices = ttnn.topk(expert_group_scores, k=self.topk_group, dim=-1)
            ttnn.deallocate(expert_group_scores)

            # Create mask for active groups
            input_mask = ttnn.repeat(self.scatter_input_mask, ttnn.Shape((1, 1, scores.shape[2], 1)))
            src_tensor = ttnn.repeat(self.scatter_src_tensor, ttnn.Shape((1, 1, scores.shape[2], 1)))

            # Scatter to create active groups mask
            active_groups_mask = ttnn.scatter(
                input=input_mask,
                index=topk_group_indices,
                src=src_tensor,
                dim=3,
            )
            ttnn.deallocate(topk_group_indices)
            ttnn.deallocate(input_mask)
            ttnn.deallocate(src_tensor)

            # Reshape and expand mask to all experts
            active_groups_mask = ttnn.reshape(active_groups_mask, (1, -1, self.n_group, 1))
            num_experts_per_group = self.n_routed_experts // self.n_group
            active_experts_mask = ttnn.repeat(active_groups_mask, ttnn.Shape((1, 1, 1, num_experts_per_group)))
            ttnn.deallocate(active_groups_mask)
            active_experts_mask = ttnn.reshape(active_experts_mask, (1, 1, -1, self.n_routed_experts))

            # Apply mask to scores
            active_experts_scores = ttnn.mul(
                scores_with_bias,
                active_experts_mask,
                memory_config=self.memory_config,
            )
            ttnn.deallocate(active_experts_mask)
        else:
            # No grouping - use all scores
            active_experts_scores = scores_with_bias

        # Get top-k experts
        topk_experts_scores_with_bias, topk_experts_indices = ttnn.topk(
            active_experts_scores,
            k=self.num_experts_per_tok,
            dim=-1,
        )
        ttnn.deallocate(active_experts_scores)
        ttnn.deallocate(topk_experts_scores_with_bias)

        # Gather original scores without bias for normalization
        topk_experts_scores = ttnn.gather(scores, dim=3, index=topk_experts_indices)
        ttnn.deallocate(scores)
        if scores_with_bias is not scores:
            ttnn.deallocate(scores_with_bias)

        # Normalize scores
        topk_expert_scores_sum = ttnn.sum(topk_experts_scores, dim=3, keepdim=True)
        # Add small epsilon to avoid division by zero
        topk_expert_scores_sum = ttnn.add(topk_expert_scores_sum, 1e-20)
        topk_experts_scores_normalized = ttnn.div(topk_experts_scores, topk_expert_scores_sum)
        ttnn.deallocate(topk_expert_scores_sum)
        ttnn.deallocate(topk_experts_scores)

        # Apply expert scaling
        expert_scale = ttnn.repeat(self.expert_scale, ttnn.Shape((1, 1, topk_experts_scores_normalized.shape[2], 1)))
        expert_scale = ttnn.to_layout(expert_scale, ttnn.TILE_LAYOUT)
        topk_experts_scores_normalized = ttnn.mul(
            topk_experts_scores_normalized,
            expert_scale,
            memory_config=self.memory_config,
            dtype=ttnn.bfloat16,
        )
        ttnn.deallocate(expert_scale)

        # Return weights and indices (note the order: weights first, then indices)
        return topk_experts_scores_normalized, topk_experts_indices
