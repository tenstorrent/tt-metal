"""
TTNN implementation of Shared Expert module with multi-chip sharding and CCL.

This module demonstrates:
- Multi-chip tensor parallelism with proper weight sharding
- Collective communication operations (all-gather, reduce-scatter)
- SiLU activation fusion
"""

import torch
import ttnn
from models.common.lightweightmodule import LightweightModule
from loguru import logger


class TtSharedExpert(LightweightModule):
    """
    TTNN implementation of Shared Expert MLP with multi-chip sharding.

    Architecture with multi-chip CCL:
        Input: x [batch, seq_len, emb_dim / num_devices]
        1. All-gather x across mesh columns → x_full [batch, seq_len, emb_dim]
        2. gate_out = x_full @ gate_proj → [batch, seq_len, hidden_dim / num_devices]
        3. up_out = x_full @ up_proj → [batch, seq_len, hidden_dim / num_devices]
        4. activated = silu(gate_out) * up_out → [batch, seq_len, hidden_dim / num_devices]
        5. output_full = activated @ down_proj → [batch, seq_len, emb_dim]
        6. Reduce-scatter output across mesh columns → [batch, seq_len, emb_dim / num_devices]

    Weight Sharding (across mesh columns):
        - gate_proj, up_proj: Shard on output dimension (-1) across mesh columns
          Shape: [emb_dim, hidden_dim / num_devices]
          mesh_mapper dims=(None, -1)
        - down_proj: Shard on input dimension (-2) across mesh columns
          Shape: [hidden_dim / num_devices, emb_dim]
          mesh_mapper dims=(None, -2)
    """

    def __init__(
        self,
        mesh_device,
        emb_dim: int = 7 * 1024,
        hidden_dim: int = 2 * 1024,
        torch_weights: dict = None,
        num_links: int = 1,
        topology: ttnn.Topology = ttnn.Topology.Linear,
    ):
        """
        Initialize TtSharedExpert module.

        Args:
            mesh_device: TTNN mesh device
            emb_dim: Embedding dimension (default: 7168)
            hidden_dim: Hidden dimension (default: 2048)
            torch_weights: Optional dict with keys 'gate_proj', 'up_proj', 'down_proj' containing torch tensors
            num_links: Number of ethernet links to use for CCL (default: 1)
            topology: CCL topology - Linear or Ring (default: Linear)
        """
        super().__init__()
        self.mesh_device = mesh_device
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_devices = mesh_device.get_num_devices()
        self.num_links = num_links
        self.topology = topology

        logger.info(f"Initializing TtSharedExpert with emb_dim={emb_dim}, hidden_dim={hidden_dim}")
        logger.info(f"Mesh shape: {mesh_device.shape}, num_devices={self.num_devices}")
        logger.info(f"CCL config: num_links={num_links}, topology={topology}")

        # Create sharded weights
        if torch_weights is not None:
            logger.info("Creating weights from provided torch tensors")
            self.gate_proj = self._create_sharded_weight_from_torch(
                torch_weights["gate_proj"], dims=(None, -1), name="gate_proj"
            )
            self.up_proj = self._create_sharded_weight_from_torch(
                torch_weights["up_proj"], dims=(None, -1), name="up_proj"
            )
            self.down_proj = self._create_sharded_weight_from_torch(
                torch_weights["down_proj"], dims=(None, -2), name="down_proj"
            )
        else:
            logger.info("Creating random sharded weights")
            self.gate_proj = self._create_random_sharded_weight(
                shape=(emb_dim, hidden_dim), dims=(None, -1), name="gate_proj"
            )
            self.up_proj = self._create_random_sharded_weight(
                shape=(emb_dim, hidden_dim), dims=(None, -1), name="up_proj"
            )
            self.down_proj = self._create_random_sharded_weight(
                shape=(hidden_dim, emb_dim), dims=(None, -2), name="down_proj"
            )

    def _create_sharded_weight_from_torch(self, torch_weight: torch.Tensor, dims: tuple, name: str) -> ttnn.Tensor:
        """
        Convert torch weight to sharded ttnn tensor.

        Args:
            torch_weight: PyTorch weight tensor [in_features, out_features]
            dims: Sharding dimensions for mesh_mapper (e.g., (None, -1) or (-2, None))
            name: Weight name for logging

        Returns:
            Sharded ttnn tensor
        """
        logger.info(f"Creating sharded weight {name} with dims={dims}, shape={torch_weight.shape}")

        # Create mesh mapper for sharding
        mesh_mapper = ttnn.ShardTensor2dMesh(
            self.mesh_device,
            mesh_shape=self.mesh_device.shape,
            dims=dims,
        )

        # Convert to ttnn tensor with sharding
        tt_weight = ttnn.from_torch(
            torch_weight,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
        )

        logger.info(f"Created {name}: {tt_weight.shape}")
        return tt_weight

    def _create_random_sharded_weight(self, shape: tuple, dims: tuple, name: str) -> ttnn.Tensor:
        """
        Create random sharded weight.

        Args:
            shape: Weight shape [in_features, out_features]
            dims: Sharding dimensions for mesh_mapper
            name: Weight name for logging

        Returns:
            Random sharded ttnn tensor
        """
        logger.info(f"Creating random sharded weight {name} with dims={dims}, shape={shape}")

        # Create random torch tensor
        torch_weight = torch.randn(*shape, dtype=torch.float32)

        return self._create_sharded_weight_from_torch(torch_weight, dims, name)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Forward pass with multi-chip sharding and CCL.

        Args:
            x: Input tensor [batch, seq_len, emb_dim / num_devices]

        Returns:
            Output tensor [batch, seq_len, emb_dim / num_devices]
        """
        logger.info(f"Forward pass: input shape={x.shape}, num_links={self.num_links}, topology={self.topology}")

        # Step 1: All-gather x across mesh columns
        # Input: [batch, seq_len, emb_dim / num_devices]
        # Output: [batch, seq_len, emb_dim]
        x_gathered = ttnn.all_gather(
            x,
            dim=-1,  # Gather along last dimension
            cluster_axis=1,  # Gather along mesh columns
            num_links=self.num_links,
            topology=self.topology,
        )
        logger.info(f"After all_gather: {x_gathered.shape}")

        # Step 2: Gate projection
        # x_gathered: [batch, seq_len, emb_dim]
        # gate_proj: [emb_dim, hidden_dim / num_devices]
        # Output: [batch, seq_len, hidden_dim / num_devices]
        assert (
            x_gathered.shape[-1] == self.gate_proj.shape[-2]
        ), f"Matmul shape mismatch: x_gathered[-1]={x_gathered.shape[-1]} != gate_proj[-2]={self.gate_proj.shape[-2]}"
        gate_out = ttnn.matmul(x_gathered, self.gate_proj)
        logger.info(f"After gate_proj matmul: {gate_out.shape}")

        # Step 3: Up projection
        # x_gathered: [batch, seq_len, emb_dim]
        # up_proj: [emb_dim, hidden_dim / num_devices]
        # Output: [batch, seq_len, hidden_dim / num_devices]
        assert (
            x_gathered.shape[-1] == self.up_proj.shape[-2]
        ), f"Matmul shape mismatch: x_gathered[-1]={x_gathered.shape[-1]} != up_proj[-2]={self.up_proj.shape[-2]}"
        up_out = ttnn.matmul(x_gathered, self.up_proj)
        logger.info(f"After up_proj matmul: {up_out.shape}")

        # Step 4: SiLU activation and element-wise multiplication (fused)
        # activated = silu(gate_out) * up_out
        # Output: [batch, seq_len, hidden_dim / num_devices]
        activated = ttnn.mul(
            gate_out,
            up_out,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
        )
        logger.info(f"After SiLU fusion: {activated.shape}")

        # Step 5: Down projection
        # activated: [batch, seq_len, hidden_dim / num_devices]
        # down_proj: [hidden_dim / num_devices, emb_dim]
        # Output: [batch, seq_len, emb_dim]
        assert (
            activated.shape[-1] == self.down_proj.shape[-2]
        ), f"Matmul shape mismatch: activated[-1]={activated.shape[-1]} != down_proj[-2]={self.down_proj.shape[-2]}"
        output_full = ttnn.matmul(activated, self.down_proj)
        logger.info(f"After down_proj matmul: {output_full.shape}")

        # Step 6: Reduce-scatter output across mesh columns
        # Input: [batch, seq_len, emb_dim]
        # Output: [batch, seq_len, emb_dim / num_devices]
        output = ttnn.reduce_scatter(
            output_full,
            dim=-1,  # Scatter along last dimension
            cluster_axis=1,  # Scatter along mesh columns
            num_links=self.num_links,
            topology=self.topology,
        )
        logger.info(f"After reduce_scatter: {output.shape}")

        return output
