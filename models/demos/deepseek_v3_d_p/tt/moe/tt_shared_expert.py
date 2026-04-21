# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
TTNN implementation of Shared Expert module with multi-chip sharding and CCL.

This module demonstrates:
- Multi-chip tensor parallelism with proper weight sharding
- Collective communication operations (all-gather, reduce-scatter)
- SiLU activation fusion
"""

from pathlib import Path
from typing import Optional

import torch
from loguru import logger

import ttnn
from models.common.lightweightmodule import LightweightModule

COMPUTE_KERNEL_CONFIG_HIFI2 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)


class TtSharedExpert(LightweightModule):
    """
    TTNN implementation of Shared Expert MLP with multi-chip sharding.

    Architecture with multi-chip CCL:
        Input: x [batch, seq_len, emb_dim] (replicated across mesh columns)
        1. gate_out = x @ gate_proj → [batch, seq_len, hidden_dim / num_devices]
        2. up_out = x @ up_proj → [batch, seq_len, hidden_dim / num_devices]
        3. activated = silu(gate_out) * up_out → [batch, seq_len, hidden_dim / num_devices]
        4. output_full = activated @ down_proj → [batch, seq_len, emb_dim]
        5. Reduce-scatter output across mesh columns → [batch, seq_len, emb_dim / num_devices]

    Weight Sharding (across mesh columns):
        - gate_proj, up_proj: Shard on output dimension (-1) across mesh columns
          Shape: [emb_dim, hidden_dim / num_devices]
          mesh_mapper dims=(None, -1)
        - down_proj: Shard on input dimension (-2) across mesh columns
          Shape: [hidden_dim / num_devices, emb_dim]
          mesh_mapper dims=(None, -2)
    """

    @staticmethod
    def check_cache_complete(cache_path: Path, cache_name_prefix: str) -> bool:
        """Check if all shared expert weight cache files exist."""
        from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import pattern_exists

        for proj in ["gate_proj", "up_proj", "down_proj"]:
            if not pattern_exists(f"{cache_name_prefix}.{proj}*.tensorbin", "SharedExpert"):
                logger.debug(f"TTNN cache missing: {cache_name_prefix}.{proj}")
                return False
        return True

    @staticmethod
    def _convert_and_cache_weights(
        torch_weights: dict,
        emb_dim: int,
        hidden_dim: int,
        mesh_device: ttnn.MeshDevice,
        weights_dtype: ttnn.DataType,
        cache_path: Path | None,
        cache_name_prefix: str | None,
        device: ttnn.MeshDevice | None = None,
    ):
        """
        Shared logic for converting gate/up/down projections to ttnn with caching.

        Args:
            torch_weights: Dict with 'gate_proj', 'up_proj', 'down_proj' [out_features, in_features]
            emb_dim: Embedding dimension
            hidden_dim: Hidden dimension
            mesh_device: Mesh device reference (for mesh_mapper)
            weights_dtype: Weight data type
            cache_path: Cache directory
            cache_name_prefix: Prefix for cache files
            device: None for cache-only, mesh_device for cache+load

        Returns:
            Dict of ttnn.Tensor if device is not None, else None
        """

        def _cache_name(name):
            if cache_path is None or cache_name_prefix is None:
                return None
            return str(cache_path / f"{cache_name_prefix}.{name}")

        # Prepare post-transpose tensors
        if torch_weights is not None:
            gate_w = torch_weights["gate_proj"].T.contiguous()
            up_w = torch_weights["up_proj"].T.contiguous()
            down_w = torch_weights["down_proj"].T.contiguous()
        else:
            gate_w = torch.empty(emb_dim, hidden_dim)
            up_w = torch.empty(emb_dim, hidden_dim)
            down_w = torch.empty(hidden_dim, emb_dim)

        def _to_ttnn(tensor, dims, name):
            mesh_mapper = ttnn.ShardTensor2dMesh(
                mesh_device,
                mesh_shape=mesh_device.shape,
                dims=dims,
            )
            return ttnn.as_tensor(
                tensor,
                mesh_mapper=mesh_mapper,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                dtype=weights_dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG if device else None,
                cache_file_name=_cache_name(name),
            )

        gate_tt = _to_ttnn(gate_w, (None, -1), "gate_proj")
        up_tt = _to_ttnn(up_w, (None, -1), "up_proj")
        down_tt = _to_ttnn(down_w, (None, -2), "down_proj")

        if device is None:
            # Cache built, free host tensors
            del gate_tt, up_tt, down_tt
            return None
        else:
            # Return device tensors for __init__
            return {"gate": gate_tt, "up": up_tt, "down": down_tt}

    @staticmethod
    def build_ttnn_cache(
        torch_weights: dict,
        emb_dim: int,
        hidden_dim: int,
        mesh_device: ttnn.MeshDevice,
        weights_dtype: ttnn.DataType,
        cache_path: Path,
        cache_name_prefix: str,
    ):
        """Build TTNN cache for shared expert without device copy."""
        TtSharedExpert._convert_and_cache_weights(
            torch_weights, emb_dim, hidden_dim, mesh_device, weights_dtype, cache_path, cache_name_prefix, device=None
        )

    def __init__(
        self,
        mesh_device,
        emb_dim: int = 7 * 1024,
        hidden_dim: int = 2 * 1024,
        torch_weights: dict = None,
        num_links: int = 1,
        topology: ttnn.Topology = ttnn.Topology.Linear,
        activations_dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat8_b,
        compute_kernel_config: ttnn.WormholeComputeKernelConfig = COMPUTE_KERNEL_CONFIG_HIFI2,
        weight_cache_path: Optional[Path] = None,
        cache_name_prefix: Optional[str] = None,
        num_dispatch_subgroups: int = 1,
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
            activations_dtype: Data type for activations (default: bfloat16)
            weights_dtype: Data type for weights (default: bfloat8_b)
            compute_kernel_config: Compute kernel configuration
            weight_cache_path: Optional path for caching TTNN weight tensors
            cache_name_prefix: Optional prefix for cache file names
        """
        super().__init__()
        self.mesh_device = mesh_device
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_devices = mesh_device.get_num_devices()
        self.num_links = num_links
        self.topology = topology
        self.activations_dtype = activations_dtype
        self.weights_dtype = weights_dtype
        self.compute_kernel_config = compute_kernel_config
        self.weight_cache_path = weight_cache_path
        self.cache_name_prefix = cache_name_prefix
        self.num_dispatch_subgroups = num_dispatch_subgroups

        # Shared-expert reduce-scatter runs on cluster_axis=1; subgroups partition axis 0.
        # Axes must be orthogonal, otherwise a subgroup's reduce-scatter would pull from
        # sibling subgroups. This assertion locks that invariant.
        assert num_dispatch_subgroups >= 1, f"num_dispatch_subgroups must be >= 1 (got {num_dispatch_subgroups})"

        logger.debug(f"Initializing TtSharedExpert with emb_dim={emb_dim}, hidden_dim={hidden_dim}")
        logger.debug(f"Mesh shape: {mesh_device.shape}, num_devices={self.num_devices}")
        logger.debug(f"CCL config: num_links={num_links}, topology={topology}")

        # Create sharded weights
        if torch_weights is not None:
            logger.debug("Creating weights from provided torch tensors")
            weights = self._convert_and_cache_weights(
                torch_weights,
                emb_dim,
                hidden_dim,
                mesh_device,
                self.weights_dtype,
                weight_cache_path,
                cache_name_prefix,
                device=mesh_device,
            )
            self.gate_proj = weights["gate"]
            self.up_proj = weights["up"]
            self.down_proj = weights["down"]
        elif weight_cache_path is not None:
            logger.debug("Loading weights from cache")
            weights = self._convert_and_cache_weights(
                None,
                emb_dim,
                hidden_dim,
                mesh_device,
                self.weights_dtype,
                weight_cache_path,
                cache_name_prefix,
                device=mesh_device,
            )
            self.gate_proj = weights["gate"]
            self.up_proj = weights["up"]
            self.down_proj = weights["down"]
        else:
            logger.debug("Creating random sharded weights")
            self.gate_proj = self._create_random_sharded_weight(
                shape=(emb_dim, hidden_dim), dims=(None, -1), name="gate_proj", dtype=self.weights_dtype
            )
            self.up_proj = self._create_random_sharded_weight(
                shape=(emb_dim, hidden_dim), dims=(None, -1), name="up_proj", dtype=self.weights_dtype
            )
            self.down_proj = self._create_random_sharded_weight(
                shape=(hidden_dim, emb_dim), dims=(None, -2), name="down_proj", dtype=self.weights_dtype
            )

    def _cache_name(self, name: str) -> Optional[str]:
        if self.weight_cache_path is None or self.cache_name_prefix is None:
            return None
        return str(self.weight_cache_path / f"{self.cache_name_prefix}.{name}")

    def _to_sharded_ttnn(self, torch_weight: torch.Tensor, dims: tuple, name: str, dtype: ttnn.DataType) -> ttnn.Tensor:
        """
        Convert torch weight to sharded ttnn tensor.

        Args:
            torch_weight: PyTorch weight tensor in TTNN format [in_features, out_features]
            dims: Sharding dimensions for mesh_mapper (e.g., (None, -1) or (-2, None))
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

        tt_weight = ttnn.as_tensor(
            torch_weight,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=self._cache_name(name),
        )

        logger.debug(f"Created {name}: {tt_weight.shape}")
        return tt_weight

    def _create_sharded_weight_from_torch(
        self, torch_weight: torch.Tensor, dims: tuple, name: str, dtype: ttnn.DataType
    ) -> ttnn.Tensor:
        """
        Convert HuggingFace torch weight to sharded ttnn tensor.

        HF/PyTorch nn.Linear weights are [out_features, in_features], but TTNN matmul(x, W)
        expects [in_features, out_features], so we transpose before sharding.
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
        Forward pass with multi-chip sharding and CCL.

        Args:
            x: Input tensor [batch, seq_len, emb_dim] (replicated across mesh columns)

        Returns:
            Output tensor [batch, seq_len, emb_dim / num_devices]
        """
        batch_size = x.shape[0]
        logger.debug(f"Forward pass: input shape={x.shape}, batch_size={batch_size}")

        # Verify input is replicated (full emb_dim) when multiple mesh columns
        if self.mesh_device.shape[1] > 1:
            assert x.shape[-1] == self.emb_dim, (
                f"Input must be replicated (full emb_dim={self.emb_dim}), "
                f"but got sharded input with shape[-1]={x.shape[-1]}"
            )

        # Convert input to activations dtype if needed
        if x.dtype != self.activations_dtype:
            logger.warning(f"{x.dtype=} typecasting {self.activations_dtype}")
            x = ttnn.typecast(x, self.activations_dtype)

        # Step 1: Gate projection
        # x: [batch, seq_len, emb_dim]
        # gate_proj: [emb_dim, hidden_dim / num_devices]
        # Output: [batch, seq_len, hidden_dim / num_devices]
        assert (
            x.shape[-1] == self.gate_proj.shape[-2]
        ), f"Matmul shape mismatch: x[-1]={x.shape[-1]} != gate_proj[-2]={self.gate_proj.shape[-2]}"
        gate_out = ttnn.matmul(x, self.gate_proj, compute_kernel_config=self.compute_kernel_config)
        logger.debug(f"After gate_proj matmul: {gate_out.shape}")

        # Step 2: Up projection
        # x: [batch, seq_len, emb_dim]
        # up_proj: [emb_dim, hidden_dim / num_devices]
        # Output: [batch, seq_len, hidden_dim / num_devices]
        assert (
            x.shape[-1] == self.up_proj.shape[-2]
        ), f"Matmul shape mismatch: x[-1]={x.shape[-1]} != up_proj[-2]={self.up_proj.shape[-2]}"
        up_out = ttnn.matmul(x, self.up_proj, compute_kernel_config=self.compute_kernel_config)
        logger.debug(f"After up_proj matmul: {up_out.shape}")

        # Step 3: SiLU activation and element-wise multiplication (fused)
        # activated = silu(gate_out) * up_out
        # Output: [batch, seq_len, hidden_dim / num_devices]
        activated = ttnn.mul(
            gate_out,
            up_out,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
        )
        logger.debug(f"After SiLU fusion: {activated.shape}")

        # Step 4: Down projection
        # activated: [batch, seq_len, hidden_dim / num_devices]
        # down_proj: [hidden_dim / num_devices, emb_dim]
        # Output: [batch, seq_len, emb_dim]
        assert (
            activated.shape[-1] == self.down_proj.shape[-2]
        ), f"Matmul shape mismatch: activated[-1]={activated.shape[-1]} != down_proj[-2]={self.down_proj.shape[-2]}"
        output_full = ttnn.matmul(activated, self.down_proj, compute_kernel_config=self.compute_kernel_config)
        logger.debug(f"After down_proj matmul: {output_full.shape}")

        # Step 5: Reduce-scatter output across mesh columns
        if self.mesh_device.shape[1] > 1:
            output = ttnn.reduce_scatter(
                output_full,
                dim=-1,  # Scatter along last dimension
                cluster_axis=1,  # Scatter along mesh columns
                num_links=self.num_links,
                topology=self.topology,
            )
        else:
            output = output_full  # No need to reduce-scatter if only one device in mesh column - there is no TP
        logger.debug(f"After reduce_scatter: {output.shape}")

        return output
