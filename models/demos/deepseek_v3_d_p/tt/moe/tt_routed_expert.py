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
            torch_weights: Optional list of dicts with keys 'gate_proj', 'up_proj', 'down_proj'
                          containing torch tensors. Length must be experts_per_chip.
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
        self.num_devices = mesh_device.get_num_devices()
        self.activations_dtype = activations_dtype
        self.weights_dtype = weights_dtype
        self.compute_kernel_config = compute_kernel_config

        logger.info(f"Initializing TtRoutedExpert with experts_per_chip={experts_per_chip}")
        logger.info(f"emb_dim={emb_dim}, hidden_dim={hidden_dim}")
        logger.info(f"Mesh shape: {mesh_device.shape}, num_devices={self.num_devices}")

        # Store weights for each local expert
        # Each expert has (gate_proj, up_proj, down_proj)
        self.gate_projs = []
        self.up_projs = []
        self.down_projs = []

        if torch_weights is not None:
            assert (
                len(torch_weights) == experts_per_chip
            ), f"Expected {experts_per_chip} expert weights, got {len(torch_weights)}"
            logger.info("Creating weights from provided torch tensors")
            for i, weights in enumerate(torch_weights):
                self.gate_projs.append(self._create_weight_from_torch(weights["gate_proj"], name=f"expert_{i}_gate"))
                self.up_projs.append(self._create_weight_from_torch(weights["up_proj"], name=f"expert_{i}_up"))
                self.down_projs.append(self._create_weight_from_torch(weights["down_proj"], name=f"expert_{i}_down"))
        else:
            logger.info("Creating random weights")
            for i in range(experts_per_chip):
                self.gate_projs.append(self._create_random_weight((emb_dim, hidden_dim), name=f"expert_{i}_gate"))
                self.up_projs.append(self._create_random_weight((emb_dim, hidden_dim), name=f"expert_{i}_up"))
                self.down_projs.append(self._create_random_weight((hidden_dim, emb_dim), name=f"expert_{i}_down"))

    def _create_weight_from_torch(self, torch_weight: torch.Tensor, name: str) -> ttnn.Tensor:
        """
        Convert torch weight to ttnn tensor replicated on all devices.

        Args:
            torch_weight: PyTorch weight tensor in HuggingFace format (out_features, in_features)
            name: Weight name for logging

        Returns:
            TTNN tensor replicated on all devices
        """
        # HuggingFace format is (out_features, in_features)
        # TTNN matmul expects (in_features, out_features) for x @ weight
        torch_weight_t = torch_weight.T.contiguous()
        logger.info(f"Creating weight {name}: HF shape {torch_weight.shape} -> TTNN shape {torch_weight_t.shape}")

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
        Create random weight tensor replicated on all devices.

        Args:
            shape: Weight shape (in_features, out_features) for TTNN matmul
            name: Weight name for logging

        Returns:
            Random TTNN tensor replicated on all devices
        """
        logger.info(f"Creating random weight {name} with shape {shape}")

        torch_weight = torch.randn(*shape, dtype=torch.float32)

        mesh_mapper = ttnn.ReplicateTensorToMesh(self.mesh_device)

        tt_weight = ttnn.from_torch(
            torch_weight,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            dtype=self.weights_dtype,
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
        gate_out = ttnn.matmul(x, gate_proj, compute_kernel_config=self.compute_kernel_config)

        # up_out = x @ up_proj
        up_out = ttnn.matmul(x, up_proj, compute_kernel_config=self.compute_kernel_config)

        # activated = silu(gate_out) * up_out (fused)
        activated = ttnn.mul(gate_out, up_out, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])

        # output = activated @ down_proj
        output = ttnn.matmul(activated, down_proj, compute_kernel_config=self.compute_kernel_config)

        return output

    def forward(
        self,
        dispatched_buffer: ttnn.Tensor,
        expert_token_counts: ttnn.Tensor = None,
    ) -> ttnn.Tensor:
        """
        Process dispatched tokens through local experts.

        Args:
            dispatched_buffer: Dispatched tokens
                shape: (1, dispatch_group_size, experts_per_chip, max_tokens, emb_dim)
            expert_token_counts: Optional token counts per expert per chip
                shape: (1, dispatch_group_size, experts_per_chip)
                If provided, only processes tokens up to the count (currently unused,
                all tokens are processed for simplicity)

        Returns:
            expert_outputs: Expert output tensor, same shape as dispatched_buffer
        """
        logger.info(f"Forward pass: dispatched_buffer shape={dispatched_buffer.shape}")

        # Convert input to activations dtype if needed
        if dispatched_buffer.dtype != self.activations_dtype:
            logger.warning(f"{dispatched_buffer.dtype=} typecasting to {self.activations_dtype}")
            dispatched_buffer = ttnn.typecast(dispatched_buffer, self.activations_dtype)

        # Process each local expert
        # dispatched_buffer: (1, dispatch_group_size, experts_per_chip, max_tokens, emb_dim)
        # We process expert by expert and reassemble

        expert_outputs_list = []
        for local_expert in range(self.experts_per_chip):
            signpost(f"Expert {local_expert}/{self.experts_per_chip}")

            # Extract tokens for this expert
            # Shape: (1, dispatch_group_size, max_tokens, emb_dim)
            tokens = dispatched_buffer[:, :, local_expert, :, :]
            logger.info(f"Expert {local_expert}: input shape {tokens.shape}")

            # Run FFN
            output = self._expert_ffn(
                tokens,
                self.gate_projs[local_expert],
                self.up_projs[local_expert],
                self.down_projs[local_expert],
            )
            logger.info(f"Expert {local_expert}: output shape {output.shape}")

            # Add expert dimension back
            # Shape: (1, dispatch_group_size, 1, max_tokens, emb_dim)
            output = ttnn.unsqueeze(output, dim=2)
            expert_outputs_list.append(output)

        # Concatenate along expert dimension
        # Shape: (1, dispatch_group_size, experts_per_chip, max_tokens, emb_dim)
        expert_outputs = ttnn.concat(expert_outputs_list, dim=2)
        logger.info(f"Final expert_outputs shape: {expert_outputs.shape}")

        return expert_outputs
