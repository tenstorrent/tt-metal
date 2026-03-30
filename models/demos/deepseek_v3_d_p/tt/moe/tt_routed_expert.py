# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
TTNN implementation of Routed Expert module for processing dispatched tokens.

This module processes tokens that have been dispatched to local experts.
Unlike TtSharedExpert, this module:
- Does NOT use CCL (no all-gather, no reduce-scatter)
- Processes tokens that are already dispatched to each device
- Each device holds weights for `experts_per_chip` local experts
"""

import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import ExpertMapping

COMPUTE_KERNEL_CONFIG_LOFI = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)


class TtRoutedExpert(LightweightModule):
    """
    TTNN implementation of Routed Expert module.

    Processes dispatched tokens through local experts. Each device holds
    `experts_per_chip` experts and processes the tokens dispatched to them.

    Architecture (per expert):
        gate_out = x @ gate_proj
        up_out = x @ up_proj
        activated = silu(gate_out) * up_out
        output = activated @ down_proj

    Weight Layout:
        - Each expert has gate_proj, up_proj, down_proj
        - Weights are NOT sharded across devices (each device has full local expert weights)
        - gate_proj, up_proj: (emb_dim, hidden_dim)
        - down_proj: (hidden_dim, emb_dim)
    """

    def __init__(
        self,
        mesh_device,
        experts_per_chip: int,
        emb_dim: int = 7 * 1024,
        hidden_dim: int = 2 * 1024,
        max_tokens: int = 1600,
        torch_weights: list[dict] = None,
        activations_dtype=ttnn.bfloat8_b,
        weights_dtype=ttnn.bfloat4_b,
        compute_kernel_config: ttnn.WormholeComputeKernelConfig = COMPUTE_KERNEL_CONFIG_LOFI,
    ):
        """
        Initialize TtRoutedExpert module.

        Args:
            mesh_device: TTNN mesh device
            experts_per_chip: Number of local experts per chip
            emb_dim: Embedding dimension (default: 7168)
            hidden_dim: Hidden/intermediate dimension (default: 2048)
            max_tokens: Maximum tokens per expert (default: 1600, used for program config)
            torch_weights: Optional list of dicts with keys 'gate_proj', 'up_proj', 'down_proj'
                          containing torch tensors. Length must be num_devices * experts_per_chip
                          (total routed experts), with weights ordered by global expert index.
                          Note: torch weights are in HuggingFace format (out_features, in_features)
                          so they need to be transposed for TTNN matmul.
            activations_dtype: Data type for activations (default: bfloat8_b)
            weights_dtype: Data type for weights (default: bfloat4_b)
            compute_kernel_config: Compute kernel configuration
        """
        super().__init__()
        self.mesh_device = mesh_device
        self.experts_per_chip = experts_per_chip
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.max_tokens = max_tokens
        self.num_devices = mesh_device.get_num_devices()
        self.activations_dtype = activations_dtype
        self.weights_dtype = weights_dtype
        self.compute_kernel_config = compute_kernel_config

        # Build program configs for matmuls using optimal parameters from sweeps
        logger.warning(
            f"RoutedExpert: No optimal program config for given dimensions {max_tokens=}{emb_dim=}{hidden_dim=}, using defaults."
        )
        self.gate_program_config = None
        self.up_program_config = None
        self.down_program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
            in0_block_w=8,
            out_subblock_h=7,
            out_subblock_w=1,
            out_block_h=7,
            out_block_w=14,
            per_core_M=7,
            per_core_N=28,
            transpose_mcast=False,
            fuse_batch=False,
            untilize_out=True,
        )

        total_experts = self.num_devices * experts_per_chip
        logger.debug(f"Initializing TtRoutedExpert with experts_per_chip={experts_per_chip}")
        logger.debug(f"emb_dim={emb_dim}, hidden_dim={hidden_dim}")
        logger.debug(f"Mesh shape: {mesh_device.shape}, num_devices={self.num_devices}, total_experts={total_experts}")

        # Store weights for each local expert
        # Each expert has (gate_proj, up_proj, down_proj)
        self.gate_projs = []
        self.up_projs = []
        self.down_projs = []

        self.gate_projs_pc = None
        self.up_projs_pc = None
        self.down_projs_pc = None

        if torch_weights is not None:
            assert len(torch_weights) == total_experts, (
                f"Expected {total_experts} expert weights (num_devices={self.num_devices} * "
                f"experts_per_chip={experts_per_chip}), got {len(torch_weights)}"
            )
            logger.debug(f"Creating weights from provided torch tensors ({total_experts} experts)")
            # Create per-device weights: for each local expert index, stack weights from all devices
            # then shard across devices so each device gets its own expert's weights
            mesh_rows, mesh_cols = self.mesh_device.shape
            for local_expert_idx in range(experts_per_chip):
                gate_weights, up_weights, down_weights = ExpertMapping.gather_weights_for_mesh_distribution(
                    torch_weights,
                    local_expert_idx,
                    mesh_rows,
                    mesh_cols,
                    experts_per_chip,
                )

                self.gate_projs.append(
                    self._create_weight_from_torch_per_device(gate_weights, name=f"expert_{local_expert_idx}_gate")
                )
                self.up_projs.append(
                    self._create_weight_from_torch_per_device(up_weights, name=f"expert_{local_expert_idx}_up")
                )
                self.down_projs.append(
                    self._create_weight_from_torch_per_device(down_weights, name=f"expert_{local_expert_idx}_down")
                )
        else:
            logger.debug("Creating random weights (replicated across devices)")
            for i in range(experts_per_chip):
                self.gate_projs.append(self._create_random_weight((emb_dim, hidden_dim), name=f"expert_{i}_gate"))
                self.up_projs.append(self._create_random_weight((emb_dim, hidden_dim), name=f"expert_{i}_up"))
                self.down_projs.append(self._create_random_weight((hidden_dim, emb_dim), name=f"expert_{i}_down"))

    def _create_weight_from_torch_per_device(
        self, torch_weights_per_device: list[torch.Tensor], name: str
    ) -> ttnn.Tensor:
        """
        Convert list of torch weights to ttnn tensor with each device getting its own weight.

        Args:
            torch_weights_per_device: List of PyTorch weight tensors, one per device.
                                      Each in HuggingFace format (out_features, in_features).
            name: Weight name for logging

        Returns:
            TTNN tensor sharded so each device has its unique weight
        """
        # Stack weights and transpose for TTNN matmul
        # Then reshape to match mesh topology: (mesh_rows, mesh_cols, in_features, out_features)
        stacked = torch.stack([w.T.contiguous() for w in torch_weights_per_device], dim=0)
        mesh_rows, mesh_cols = self.mesh_device.shape
        in_features, out_features = stacked.shape[1], stacked.shape[2]
        stacked = stacked.reshape(mesh_rows, mesh_cols, in_features, out_features)

        logger.debug(
            f"Creating per-device weight {name}: "
            f"per-device HF shape {torch_weights_per_device[0].shape} -> "
            f"stacked shape {stacked.shape}"
        )

        mesh_mapper = ExpertMapping.get_weights_mesh_mapper(self.mesh_device)

        tt_weight = ttnn.from_torch(
            stacked,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            dtype=self.weights_dtype,
        )

        # Remove the mesh dimensions: (1, 1, in_features, out_features) -> (in_features, out_features)
        tt_weight = ttnn.squeeze(tt_weight, dim=0)
        tt_weight = ttnn.squeeze(tt_weight, dim=0)

        return tt_weight

    def _create_weight_from_torch(self, torch_weight: torch.Tensor, name: str) -> ttnn.Tensor:
        """
        Convert torch weight to ttnn tensor replicated on all devices.

        NOTE: This method is kept for backwards compatibility but is not used
        when torch_weights are provided (see _create_weight_from_torch_per_device).

        Args:
            torch_weight: PyTorch weight tensor in HuggingFace format (out_features, in_features)
            name: Weight name for logging

        Returns:
            TTNN tensor replicated on all devices
        """
        # HuggingFace format is (out_features, in_features)
        # TTNN matmul expects (in_features, out_features) for x @ weight
        torch_weight_t = torch_weight.T.contiguous()
        logger.debug(f"Creating weight {name}: HF shape {torch_weight.shape} -> TTNN shape {torch_weight_t.shape}")

        # Replicate on all devices (no sharding for routed expert weights)
        mesh_mapper = ttnn.ReplicateTensorToMesh(self.mesh_device)

        tt_weight = ttnn.from_torch(
            torch_weight_t,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            dtype=self.weights_dtype,
        )

        return tt_weight

    def _create_random_weight(self, shape: tuple, name: str) -> ttnn.Tensor:
        """
        Allocate uninitialized weight tensor on device DRAM (fast, no host transfer).

        Args:
            shape: Weight shape (in_features, out_features) for TTNN matmul
            name: Weight name for logging

        Returns:
            Uninitialized TTNN tensor on device DRAM
        """
        logger.debug(f"Allocating uninitialized weight {name} with shape {shape} on device DRAM")

        tt_weight = ttnn.allocate_tensor_on_device(
            ttnn.Shape(shape),
            self.weights_dtype,
            ttnn.TILE_LAYOUT,
            self.mesh_device,
            ttnn.DRAM_MEMORY_CONFIG,
        )

        return tt_weight

    def _expert_ffn(
        self,
        x: ttnn.Tensor,
        gate_proj: ttnn.Tensor,
        up_proj: ttnn.Tensor,
        down_proj: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """
        Single expert FFN computation.

        Args:
            x: Input tensor (batch, tokens, emb_dim)
            gate_proj: Gate projection weight (emb_dim, hidden_dim)
            up_proj: Up projection weight (emb_dim, hidden_dim)
            down_proj: Down projection weight (hidden_dim, emb_dim)

        Returns:
            Output tensor (batch, tokens, emb_dim)
        """
        # gate_out = x @ gate_proj
        gate_out = ttnn.matmul(
            x, gate_proj, program_config=self.gate_program_config, compute_kernel_config=self.compute_kernel_config
        )

        # up_out = x @ up_proj
        up_out = ttnn.matmul(
            x, up_proj, program_config=self.up_program_config, compute_kernel_config=self.compute_kernel_config
        )

        # activated = silu(gate_out) * up_out - SiLU fused with multiply
        activated = ttnn.mul(gate_out, up_out, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])

        # output = activated @ down_proj
        output = ttnn.matmul(
            activated,
            down_proj,
            program_config=self.down_program_config,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
        )

        return output

    def forward(
        self,
        dispatched_buffer: ttnn.Tensor,
        expert_token_counts: ttnn.Tensor = None,  # Unused for now, can reduce compute
    ) -> ttnn.Tensor:
        """
        Process dispatched tokens through local experts.

        Args:
            dispatched_buffer: Dispatched tokens
                shape: (experts_per_chip, max_tokens, emb_dim)
            expert_token_counts: Optional token counts per expert per chip
                If provided, only processes tokens up to the count (currently unused,
                all tokens are processed for simplicity)

        Returns:
            expert_outputs: Expert output tensor, same shape as dispatched_buffer
        """
        logger.debug(f"Forward pass: dispatched_buffer shape={dispatched_buffer.shape}")

        # Convert input to activations dtype if needed
        if dispatched_buffer.dtype != self.activations_dtype:
            logger.warning(f"{dispatched_buffer.dtype=} typecasting to {self.activations_dtype}")
            dispatched_buffer = ttnn.typecast(dispatched_buffer, self.activations_dtype)

        # Process each local expert
        # dispatched_buffer: (experts_per_chip, max_tokens, emb_dim)
        # We process expert by expert and reassemble

        expert_outputs_list = []
        for local_expert in range(self.experts_per_chip):
            signpost(f"Expert {local_expert+1}/{self.experts_per_chip}")

            # Extract tokens for this expert
            # Shape: (max_tokens, emb_dim)
            tokens = dispatched_buffer[local_expert, :, :]
            logger.debug(f"Expert {local_expert}: input shape {tokens.shape}")

            # Run FFN
            output = self._expert_ffn(
                tokens,
                self.gate_projs[local_expert],
                self.up_projs[local_expert],
                self.down_projs[local_expert],
            )
            logger.debug(f"Expert {local_expert}: output shape {output.shape}")

            # Add expert dimension back
            # Shape: (1, max_tokens, emb_dim)
            output = ttnn.unsqueeze(output, dim=0)
            expert_outputs_list.append(output)

        # Concatenate along expert dimension
        # Shape: (experts_per_chip, max_tokens, emb_dim)
        expert_outputs = ttnn.concat(expert_outputs_list, dim=0)
        logger.debug(f"Final expert_outputs shape: {expert_outputs.shape}")

        return expert_outputs
