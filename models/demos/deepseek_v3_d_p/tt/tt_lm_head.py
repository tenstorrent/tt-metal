# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
TTNN implementation of the LM Head (language model output projection) for DeepSeek V3.

Projects hidden states to vocabulary logits:
    Input:  [dispatch_group_size, seq_len, emb_dim]
    Output: [dispatch_group_size, TILE_SIZE, vocab_size]

Weight Sharding (across mesh columns):
    - weight: Sharded on output dimension (-1) across mesh columns
      Shape: [emb_dim, vocab_size]
      mesh_mapper dims=(None, -1)
"""

from pathlib import Path
from typing import Optional

import torch
from loguru import logger

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.tt.mla.utils import global_to_local_token_id

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
        1. x_narrow = narrow x to last TILE_SIZE tokens → [dispatch_group_size, TILE_SIZE, emb_dim]
        2. output = x @ weight → [dispatch_group_size, TILE_SIZE, vocab_size]
        3. All-gather output across mesh columns → [dispatch_group_size, TILE_SIZE, vocab_size]
    """

    @staticmethod
    def check_cache_complete(cache_path: Path) -> bool:
        """Check if LM head weight cache files exist."""
        from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import pattern_exists

        if not pattern_exists("lm_head_weight*.tensorbin", "LMHead"):
            logger.debug("TTNN cache missing: lm_head_weight")
            return False
        return True

    @staticmethod
    def _convert_and_cache_weight(
        torch_weight: torch.Tensor | None,
        emb_dim: int,
        vocab_size: int,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType,
        cache_path: Path | None,
        device: ttnn.MeshDevice | None = None,
    ) -> ttnn.Tensor | None:
        """
        Shared logic for converting LM head weight to ttnn with caching.

        Args:
            torch_weight: Weight tensor [vocab_size, emb_dim] in HF format.
                          If None, creates empty tensor for cache-loading.
            emb_dim: Embedding dimension
            vocab_size: Vocabulary size
            mesh_device: Mesh device (for mesh_mapper)
            dtype: Data type
            cache_path: Cache directory path
            device: None for cache-only, mesh_device for cache+load

        Returns:
            ttnn.Tensor if device is not None, else None
        """
        if torch_weight is not None:
            assert torch_weight.shape == (
                vocab_size,
                emb_dim,
            ), f"Weight shape mismatch: got {torch_weight.shape}, expected ({vocab_size}, {emb_dim})"
            # Transpose HF [vocab_size, emb_dim] to TTNN [emb_dim, vocab_size]
            torch_weight = torch_weight.T.contiguous()
        else:
            # Empty tensor for cache loading
            torch_weight = torch.empty(emb_dim, vocab_size)

        mesh_mapper = ttnn.ShardTensor2dMesh(
            mesh_device,
            mesh_shape=mesh_device.shape,
            dims=(None, -1),  # Shard vocab_size across TP axis (columns)
        )

        cache_file_name = str(cache_path / "lm_head_weight") if cache_path else None

        tt_weight = ttnn.as_tensor(
            torch_weight,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG if device else None,
            cache_file_name=cache_file_name,
        )

        if device is not None:
            ttnn.synchronize_device(device)

        if device is None:
            del tt_weight
            return None
        else:
            return tt_weight

    @staticmethod
    def build_ttnn_cache(
        torch_weight: torch.Tensor,
        vocab_size: int,
        emb_dim: int,
        mesh_device: ttnn.MeshDevice,
        cache_path: Path,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ):
        """Build TTNN cache for LM head weight without device copy."""
        TtLMHead._convert_and_cache_weight(
            torch_weight, emb_dim, vocab_size, mesh_device, dtype, cache_path, device=None
        )

    def __init__(
        self,
        mesh_device,
        emb_dim: int = DeepSeekV3Config.EMB_SIZE,
        vocab_size: int = DeepSeekV3Config.VOCAB_SIZE,
        torch_weight: torch.Tensor = None,
        num_links: int = 1,
        topology: ttnn.Topology = ttnn.Topology.Ring,
        activations_dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat16,
        compute_kernel_config: ttnn.WormholeComputeKernelConfig = COMPUTE_KERNEL_CONFIG_HIFI2,
        is_balanced: bool = False,
        weight_cache_path: Optional[Path] = None,
    ):
        """
        Initialize TtLMHead module.

        Args:
            mesh_device: TTNN mesh device
            emb_dim: Embedding dimension (default: 7168)
            vocab_size: Vocabulary size (default: 129280)
            torch_weight: Optional weight tensor [vocab_size, emb_dim].
                          If None, loads from cache or creates random weight.
            num_links: Number of ethernet links to use for CCL (default: 1)
            topology: CCL topology - Linear or Ring (default: Ring)
            activations_dtype: Data type for activations (default: bfloat16)
            weights_dtype: Data type for weights (default: bfloat16)
            compute_kernel_config: Compute kernel configuration
            is_balanced: If True, uses zigzag token mapping. If False (default),
                         uses sequential mapping. Should match TtMLA's is_balanced.
            weight_cache_path: Optional path to weight cache directory
        """
        super().__init__()
        self.mesh_device = mesh_device
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.num_devices = mesh_device.get_num_devices()
        self.sp_factor = mesh_device.shape[0]
        self.num_links = num_links
        self.topology = topology
        self.activations_dtype = activations_dtype
        self.weights_dtype = weights_dtype
        self.compute_kernel_config = compute_kernel_config
        self.is_balanced = is_balanced
        self.weight_cache_path = weight_cache_path

        logger.debug(f"Initializing TtLMHead with emb_dim={emb_dim}, vocab_size={vocab_size}")
        logger.debug(f"Mesh shape: {mesh_device.shape}, num_devices={self.num_devices}")
        logger.debug(f"CCL config: num_links={num_links}, topology={topology}")
        logger.debug(f"is_balanced={is_balanced}, weight_cache_path={weight_cache_path}")

        if torch_weight is not None:
            logger.debug("Creating weight from provided torch tensor")
            self.weight = self._create_weight_from_torch(torch_weight)
        elif weight_cache_path is not None:
            logger.debug("Loading weight from cache")
            self.weight = self._convert_and_cache_weight(
                None,
                self.emb_dim,
                self.vocab_size,
                self.mesh_device,
                self.weights_dtype,
                self.weight_cache_path,
                device=self.mesh_device,
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

    def _create_weight_from_torch(self, torch_weight: torch.Tensor) -> ttnn.Tensor:
        """
        Convert torch LM head weight to TP-sharded ttnn tensor.

        Uses shared static method for conversion with caching.

        Args:
            torch_weight: [vocab_size, emb_dim] in HF format

        Returns:
            Sharded ttnn tensor: each TP device gets [emb_dim, vocab_size / tp_factor]
        """
        tt_weight = self._convert_and_cache_weight(
            torch_weight,
            self.emb_dim,
            self.vocab_size,
            self.mesh_device,
            self.weights_dtype,
            self.weight_cache_path,
            device=self.mesh_device,
        )
        logger.debug(f"Created sharded LM head weight: {tt_weight.shape}")
        return tt_weight

    def forward(self, x: ttnn.Tensor, global_token_id: int) -> tuple[ttnn.Tensor, tuple[int, int]]:
        """
        Forward pass: project hidden states to vocabulary logits.

        Args:
            x: Input tensor [dispatch_group_size, seq_len, emb_dim]
            global_token_id: The global token position whose logits we need.

        Returns:
            tuple[ttnn.Tensor, tuple[int, int]]:
                - Logits tensor [dispatch_group_size, TILE_SIZE, vocab_size/tp]
                - (device_id, token_offset): which SP device holds the target token
                  and the index within the tile
        """
        logger.debug(f"[TtLMHead.forward] INPUT SHAPES:")
        logger.debug(f"  x.shape={x.shape}")

        # ========================================
        # Step 0: Extract the tile containing the target token
        # ========================================
        # Use negative indexing: seq_len is at dim -2, emb_dim at dim -1
        seq_len_per_device = x.shape[-2]
        seq_len = seq_len_per_device * self.sp_factor

        # Convert global token ID to local token ID on this device.
        device_id, local_token_id = global_to_local_token_id(
            global_token_id, self.sp_factor, seq_len, is_balanced=self.is_balanced
        )

        # We only need logits for a single token, but matmul operates on tiles.
        # Find the tile-aligned start position that contains local_token_id.
        tile_start = (local_token_id // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
        token_offset = local_token_id % ttnn.TILE_SIZE
        x = ttnn.narrow(x, dim=-2, start=tile_start, length=ttnn.TILE_SIZE)
        logger.debug(f"[TtLMHead.forward] After narrow ({local_token_id=} {tile_start=}): {x.shape=} {token_offset=}")

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

        # output is logits [1,1, TILE_SIZE, vocab_size/tp]
        # fractured over SP; TP-sharded on vocab dim
        # the last logit/token is on device_id at token_offset within the tile
        return output, (device_id, token_offset)

    def logit_to_host(self, tt_logit: ttnn.Tensor) -> torch.Tensor:
        ttnn.synchronize_device(self.mesh_device)  # ensure all computation is done before copying to host
        tt_logit_host = ttnn.to_torch(
            tt_logit,
            mesh_composer=ttnn.create_mesh_composer(
                self.mesh_device,
                config=ttnn.MeshComposerConfig(
                    dims=(0, -1),  # SP fractured on dim 0, TP concat on vocab dim
                    mesh_shape_override=ttnn.MeshShape(self.mesh_device.shape[0], self.mesh_device.shape[1]),
                ),
            ),
        ).to(torch.bfloat16)
        return tt_logit_host

    def select_first_token(self, logit_host: torch.Tensor, device_id: int, token_offset: int) -> torch.Tensor:
        return logit_host[device_id, 0, token_offset, :].unsqueeze(0).unsqueeze(0)
