# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
TTNN implementation of the LM Head (language model output projection) for DeepSeek V3.

Projects hidden states to vocabulary logits:
    Input:  [dispatch_group_size, seq_len, emb_dim]
    Output: [dispatch_group_size, seq_len, vocab_size]

Weight Sharding (across mesh columns):
    - weight: Sharded on output dimension (-1) across mesh columns
      Shape: [emb_dim, vocab_size]
      mesh_mapper dims=(None, -1)
"""

import torch
from loguru import logger

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config

COMPUTE_KERNEL_CONFIG_HIFI2 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)


class TtLMHead(LightweightModule):
    """
    TTNN implementation of the LM Head for DeepSeek V3.

    Architecture:
        Input: x [dispatch_group_size, seq_len, emb_dim]
        1. output = x @ weight → [dispatch_group_size, seq_len, vocab_size]
        2. All-gather output across mesh columns → [dispatch_group_size, seq_len, vocab_size]
    """

    def __init__(
        self,
        mesh_device,
        emb_dim: int = DeepSeekV3Config.EMB_SIZE,
        vocab_size: int = DeepSeekV3Config.VOCAB_SIZE,
        torch_weights: dict = None,
        num_links: int = 1,
        topology: ttnn.Topology = ttnn.Topology.Ring,
        activations_dtype=ttnn.bfloat8_b,
        weights_dtype=ttnn.bfloat4_b,
        compute_kernel_config: ttnn.WormholeComputeKernelConfig = COMPUTE_KERNEL_CONFIG_HIFI2,
    ):
        """
        Initialize TtLMHead module.

        Args:
            mesh_device: TTNN mesh device
            emb_dim: Embedding dimension (default: 7168)
            vocab_size: Vocabulary size (default: 129280)
            torch_weights: Optional dict with key 'weight' containing torch tensor [vocab_size, emb_dim]
            num_links: Number of ethernet links to use for CCL (default: 1)
            topology: CCL topology - Linear or Ring (default: Ring)
            activations_dtype: Data type for activations (default: bfloat8_b)
            weights_dtype: Data type for weights (default: bfloat4_b)
            compute_kernel_config: Compute kernel configuration
        """
        super().__init__()
        self.mesh_device = mesh_device
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.num_devices = mesh_device.get_num_devices()
        self.num_links = num_links
        self.topology = topology
        self.activations_dtype = activations_dtype
        self.weights_dtype = weights_dtype
        self.compute_kernel_config = compute_kernel_config

        logger.debug(f"Initializing TtLMHead with emb_dim={emb_dim}, vocab_size={vocab_size}")
        logger.debug(f"Mesh shape: {mesh_device.shape}, num_devices={self.num_devices}")
        logger.debug(f"CCL config: num_links={num_links}, topology={topology}")

        if torch_weights is not None:
            logger.debug("Creating weight from provided torch tensor")
            self.weight = self._create_sharded_weight_from_torch(
                torch_weights["lm_head.weight"], dims=(None, -1), name="lm_head_weight", dtype=self.weights_dtype
            )
        else:
            logger.debug("Creating random sharded weight")
            self.weight = self._create_random_sharded_weight(
                shape=(emb_dim, vocab_size), dims=(None, -1), name="lm_head_weight", dtype=self.weights_dtype
            )

    def _to_sharded_ttnn(self, torch_weight: torch.Tensor, dims: tuple, name: str, dtype: ttnn.DataType) -> ttnn.Tensor:
        """
        Convert torch weight to sharded ttnn tensor.

        Args:
            torch_weight: PyTorch weight tensor in TTNN format [in_features, out_features]
            dims: Sharding dimensions for mesh_mapper (e.g., (None, -1))
            name: Weight name for logging
            dtype: Data type for the weight tensor

        Returns:
            Sharded ttnn tensor
        """
        logger.debug(f"Creating sharded weight {name} with dims={dims}, shape={torch_weight.shape}")

        mesh_mapper = ttnn.ShardTensor2dMesh(
            self.mesh_device,
            mesh_shape=self.mesh_device.shape,
            dims=dims,
        )

        tt_weight = ttnn.from_torch(
            torch_weight,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            dtype=dtype,
        )

        logger.debug(f"Created {name}: {tt_weight.shape}")
        return tt_weight

    def _create_sharded_weight_from_torch(
        self, torch_weight: torch.Tensor, dims: tuple, name: str, dtype: ttnn.DataType
    ) -> ttnn.Tensor:
        """
        Convert HuggingFace torch weight to sharded ttnn tensor.

        HF/PyTorch nn.Linear weights are [out_features, in_features], but TTNN matmul(x, W)
        expects [in_features, out_features], so we transpose weights before sharding.
        """
        torch_weight = torch_weight.T.contiguous()
        return self._to_sharded_ttnn(torch_weight, dims, name, dtype)

    def _create_random_sharded_weight(self, shape: tuple, dims: tuple, name: str, dtype: ttnn.DataType) -> ttnn.Tensor:
        """
        Create random sharded weight in TTNN format [in_features, out_features].
        """
        torch_weight = torch.randn(*shape, dtype=torch.float32)
        return self._to_sharded_ttnn(torch_weight, dims, name, dtype)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Forward pass: project hidden states to vocabulary logits.

        Args:
            x: Input tensor [dispatch_group_size, seq_len, emb_dim]

        Returns:
            Logits tensor [dispatch_group_size, seq_len, vocab_size]
        """
        logger.debug(f"[TtLMHead.forward] INPUT SHAPES:")
        logger.debug(f"  x.shape={x.shape}")

        # ========================================
        # Step 0: Extract last 32 logits
        # ========================================
        # Actually, we only care about the last logit. However, due to the matmul constraint
        # to work on tiles, we need to extract the last tile.
        x = ttnn.narrow(x, dim=1, start=-32, length=32)
        logger.debug(f"[TtLMHead.forward] After narrow: x.shape={x.shape}")

        # ========================================
        # Step 1: All-gather x to get full emb_dim (replicated across TP axis)
        # ========================================
        # Input x is sharded: (dispatch_group_size/axis0, seq_len_per_chip, emb_dim/axis1)
        # Both shared_expert and dispatch need full emb_dim, so all-gather first
        # Only needed if there are multiple devices in TP axis (axis 1)
        if self.mesh_device.shape[1] > 1:
            x_full = ttnn.all_gather(
                x,
                dim=-1,  # Gather along emb_dim
                cluster_axis=1,  # Gather across axis 1 (TP axis)
                num_links=self.num_links,
                topology=self.topology,
            )
        else:
            x_full = x  # No TP sharding, x already has full emb_dim
        logger.debug(f"[TtLMHead.forward] x_full (after all_gather) shape: {x_full.shape}")

        # ========================================
        # Step 2: Local matmul with sharded weight
        # ========================================
        output = ttnn.matmul(x_full, self.weight, compute_kernel_config=self.compute_kernel_config)
        logger.debug(f"[TtLMHead.forward] output (after matmul) shape: {output.shape}")

        return output
