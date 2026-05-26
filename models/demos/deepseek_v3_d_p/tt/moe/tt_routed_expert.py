# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
TTNN implementation of Routed Expert module for processing dispatched tokens.

This module processes tokens that have been dispatched to local experts.
Unlike TtSharedExpert, this module:
- Does NOT use CCL (no all-gather, no reduce-scatter)
- Processes tokens that are already dispatched to each device
- Each device holds weights for `experts_per_chip` local experts
"""

from pathlib import Path
from typing import Optional

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
    @staticmethod
    def check_cache_complete(cache_path: Path, cache_name_prefix: str, experts_per_chip: int) -> bool:
        """Check if all routed expert weight cache files exist."""
        from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import pattern_exists

        for local_expert_idx in range(experts_per_chip):
            for proj in ["gate", "up", "down"]:
                pattern = f"{cache_name_prefix}.local_{local_expert_idx}_{proj}*.tensorbin"
                if not pattern_exists(pattern, "RoutedExpert"):
                    logger.debug(f"TTNN cache missing: {cache_name_prefix}.local_{local_expert_idx}_{proj}")
                    return False
        return True

    @staticmethod
    def _convert_and_cache_expert_weights(
        torch_weights: list[dict] | None,
        experts_per_chip: int,
        mesh_device: ttnn.MeshDevice,
        weights_dtype: ttnn.DataType,
        cache_path: Path | None,
        cache_name_prefix: str | None,
        device: ttnn.MeshDevice | None = None,
        *,
        emb_dim: int | None = None,
        hidden_dim: int | None = None,
    ):
        """
        Shared logic for converting expert weights to ttnn with caching.

        Args:
            torch_weights: List of expert weight dicts, or None for cache-only loading.
                When None, emb_dim and hidden_dim must be provided.
            experts_per_chip: Number of experts per chip (8 for 8x4 mesh)
            mesh_device: Mesh device reference
            weights_dtype: Weight data type
            cache_path: Cache directory
            cache_name_prefix: Prefix for cache files
            device: None for cache-only, mesh_device for cache+load
            emb_dim: Required when torch_weights is None
            hidden_dim: Required when torch_weights is None

        Returns:
            (gate_projs, up_projs, down_projs) if device is not None, else None
        """
        from tqdm import tqdm

        def _cache_name(name):
            if cache_path is None or cache_name_prefix is None:
                return None
            return str(cache_path / f"{cache_name_prefix}.{name}")

        mesh_rows, mesh_cols = mesh_device.shape
        gate_tensors, up_tensors, down_tensors = [], [], []

        mode = "build-cache" if device is None else ("load-cache" if torch_weights is None else "convert")
        for local_expert_idx in tqdm(range(experts_per_chip), desc=f"Expert weights ({mode})"):
            if torch_weights is not None:
                gate_weights, up_weights, down_weights = ExpertMapping.gather_weights_for_mesh_distribution(
                    torch_weights, local_expert_idx, mesh_rows, mesh_cols, experts_per_chip
                )

                stacked_gate = torch.stack([w.T.contiguous() for w in gate_weights], dim=0)
                in_f, out_f = stacked_gate.shape[1], stacked_gate.shape[2]
                stacked_gate = stacked_gate.reshape(mesh_rows, mesh_cols, in_f, out_f)

                stacked_up = torch.stack([w.T.contiguous() for w in up_weights], dim=0).reshape(
                    mesh_rows, mesh_cols, in_f, out_f
                )

                stacked_down = torch.stack([w.T.contiguous() for w in down_weights], dim=0)
                in_f_down, out_f_down = stacked_down.shape[1], stacked_down.shape[2]
                stacked_down = stacked_down.reshape(mesh_rows, mesh_cols, in_f_down, out_f_down)
            else:
                assert emb_dim is not None and hidden_dim is not None
                stacked_gate = torch.empty(mesh_rows, mesh_cols, emb_dim, hidden_dim)
                stacked_up = torch.empty(mesh_rows, mesh_cols, emb_dim, hidden_dim)
                stacked_down = torch.empty(mesh_rows, mesh_cols, hidden_dim, emb_dim)

            mem = ttnn.DRAM_MEMORY_CONFIG if device else None
            mapper = ExpertMapping.get_weights_mesh_mapper(mesh_device)

            gate_tt = ttnn.as_tensor(
                stacked_gate,
                mesh_mapper=mapper,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                dtype=weights_dtype,
                memory_config=mem,
                cache_file_name=_cache_name(f"local_{local_expert_idx}_gate"),
            )
            up_tt = ttnn.as_tensor(
                stacked_up,
                mesh_mapper=mapper,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                dtype=weights_dtype,
                memory_config=mem,
                cache_file_name=_cache_name(f"local_{local_expert_idx}_up"),
            )
            down_tt = ttnn.as_tensor(
                stacked_down,
                mesh_mapper=mapper,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                dtype=weights_dtype,
                memory_config=mem,
                cache_file_name=_cache_name(f"local_{local_expert_idx}_down"),
            )

            if device is None:
                del gate_tt, up_tt, down_tt
            else:
                gate_tt = ttnn.squeeze(ttnn.squeeze(gate_tt, dim=0), dim=0)
                up_tt = ttnn.squeeze(ttnn.squeeze(up_tt, dim=0), dim=0)
                down_tt = ttnn.squeeze(ttnn.squeeze(down_tt, dim=0), dim=0)
                gate_tensors.append(gate_tt)
                up_tensors.append(up_tt)
                down_tensors.append(down_tt)

        return (gate_tensors, up_tensors, down_tensors) if device else None

    @staticmethod
    def build_ttnn_cache(
        torch_weights: list[dict],
        experts_per_chip: int,
        mesh_device: ttnn.MeshDevice,
        weights_dtype: ttnn.DataType,
        cache_path: Path,
        cache_name_prefix: str,
    ):
        """Build TTNN cache for routed experts without device copy."""
        TtRoutedExpert._convert_and_cache_expert_weights(
            torch_weights, experts_per_chip, mesh_device, weights_dtype, cache_path, cache_name_prefix, device=None
        )

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
        global_expert_idx_table: ttnn.Tensor,
        emb_dim: int = 7 * 1024,
        hidden_dim: int = 2 * 1024,
        max_tokens: int = 1600,
        torch_weights: list[dict] = None,
        activations_dtype=ttnn.bfloat8_b,
        weights_dtype=ttnn.bfloat4_b,
        compute_kernel_config: ttnn.WormholeComputeKernelConfig = COMPUTE_KERNEL_CONFIG_LOFI,
        weight_cache_path: Optional[Path] = None,
        cache_name_prefix: Optional[str] = None,
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
            global_expert_idx_table: TTNN tensor mapping local expert slots to global expert ids.
                          Produced by sharding ExpertMapping.create_global_expert_idx_table via
                          get_ep_mesh_mapper, so each device holds (1, 1, experts_per_chip) of
                          global ids. Required.
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
        self.weight_cache_path = weight_cache_path
        self.cache_name_prefix = cache_name_prefix
        self.global_expert_idx_table = global_expert_idx_table

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
            result = self._convert_and_cache_expert_weights(
                torch_weights,
                experts_per_chip,
                self.mesh_device,
                self.weights_dtype,
                self.weight_cache_path,
                self.cache_name_prefix,
                device=self.mesh_device,
            )
        elif weight_cache_path is not None:
            logger.debug(f"Loading weights from cache ({experts_per_chip} local experts)")
            result = self._convert_and_cache_expert_weights(
                None,
                experts_per_chip,
                self.mesh_device,
                self.weights_dtype,
                self.weight_cache_path,
                self.cache_name_prefix,
                device=self.mesh_device,
                emb_dim=emb_dim,
                hidden_dim=hidden_dim,
            )
        else:
            logger.debug(f"Creating dummy tensors for testing ({total_experts} experts)")
            torch_weights = []
            for _ in range(total_experts):
                torch_weights.append(
                    {
                        "gate_proj": torch.empty(hidden_dim, emb_dim),
                        "up_proj": torch.empty(hidden_dim, emb_dim),
                        "down_proj": torch.empty(emb_dim, hidden_dim),
                    }
                )
            result = self._convert_and_cache_expert_weights(
                torch_weights,
                experts_per_chip,
                self.mesh_device,
                self.weights_dtype,
                None,
                None,
                device=self.mesh_device,
            )

        assert result is not None, "Expected weight tensors to be returned when device is provided"
        self.gate_projs, self.up_projs, self.down_projs = result

    @staticmethod
    def shard_expert_token_counts(
        mesh_device: ttnn.MeshDevice,
        expert_token_counts: torch.Tensor,
    ) -> ttnn.Tensor:
        """
        Convert and shard the expert token counts tensor across mesh devices.

        Args:
            mesh_device: The mesh device to place the tensor on
            expert_token_counts: Total tokens per expert (sparse per group, replicated across dispatch_group_size)
                Shape: (num_dispatch_groups, dispatch_group_size, num_routed_experts) - from get_gate_outputs()

        Returns:
            TTNN tensor sharded across mesh devices.
            Per-device shape: (1, num_routed_experts)
        """
        logger.debug(f"[shard_expert_token_counts] INPUT: expert_token_counts.shape={expert_token_counts.shape}")
        mesh_mapper = ttnn.ShardTensor2dMesh(
            mesh_device,
            mesh_shape=mesh_device.shape,
            dims=(1, 0),
        )
        result = ttnn.from_torch(
            expert_token_counts,
            mesh_mapper=mesh_mapper,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            dtype=ttnn.uint32,
        )
        result = ttnn.squeeze(result, 0)
        logger.debug(f"[shard_expert_token_counts] OUTPUT: result.shape={result.shape}")
        return result

    def _cache_name(self, name: str) -> Optional[str]:
        if self.weight_cache_path is None or self.cache_name_prefix is None:
            return None
        return str(self.weight_cache_path / f"{self.cache_name_prefix}.{name}")

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
        out: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """
        Single expert FFN computation.

        Args:
            x: Input tensor. Shape is (1, tokens, emb_dim) for the Blackhole path
                (after ttnn.narrow) or (tokens, emb_dim) for the Wormhole path
                (after tensor indexing).
            gate_proj: Gate projection weight (emb_dim, hidden_dim)
            up_proj: Up projection weight (emb_dim, hidden_dim)
            down_proj: Down projection weight (hidden_dim, emb_dim)
            out: Optional pre-allocated output tensor for in-place matmul result.
                When provided, the final matmul writes directly into this buffer.
                When None, a new tensor is allocated for the output.

        Returns:
            Output tensor matching the shape of ``x``.
        """
        return ttnn.experimental.deepseek_prefill.routed_expert_ffn(
            x,
            gate_proj,
            up_proj,
            down_proj,
            compute_kernel_config=self.compute_kernel_config,
            output=out,
        )

    def forward(
        self,
        dispatched_buffer: ttnn.Tensor,
        expert_token_counts: ttnn.Tensor,
        expert_region_offsets: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """
        Blackhole forward implementation using narrow and in-place writes.

        Pre-allocates the output tensor with empty_like and uses narrow to extract
        per-expert slices and write FFN results directly into the output buffer,
        avoiding extra allocations from unsqueeze/concat.

        Args:
            dispatched_buffer: Dispatched tokens
                shape: (max_dispatch_buffer_token_size, emb_dim)
            expert_token_counts: Token counts per expert per chip
                If provided, only processes tokens up to the count (currently unused,
                all tokens are processed for simplicity)
            expert_region_offsets: Expert region start offsets per expert
                (shared across source devices in a dispatch group). Produced by
                offset_cumsum. Shape per device: (1, num_routed_experts).

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

        expert_outputs = dispatched_buffer
        for local_expert in range(self.experts_per_chip):
            signpost(f"Expert {local_expert+1}/{self.experts_per_chip}")

            # Extract tokens for this expert using the deepseek_prefill extract op,
            # which uses expert_region_offsets and expert_token_counts to slice out
            # this expert's valid rows
            tokens = ttnn.experimental.deepseek_prefill.extract(
                dispatched_buffer,
                expert_region_offsets,
                expert_token_counts,
                self.global_expert_idx_table,
                local_expert_id=local_expert,
                max_dispatched_tokens_per_expert=self.max_tokens,
            )
            logger.debug(f"Expert {local_expert}: input shape {tokens.shape}")

            # Run FFN
            output = self._expert_ffn(
                tokens,
                self.gate_projs[local_expert],
                self.up_projs[local_expert],
                self.down_projs[local_expert],
                out=None,
            )
            logger.debug(f"Expert {local_expert}: output shape {output.shape}")

            # Insert this expert's output back into the flat expert_outputs buffer at
            # the expert's region (determined by expert_region_offsets and expert_token_counts).
            expert_outputs = ttnn.experimental.deepseek_prefill.insert(
                expert_outputs,
                output,
                expert_region_offsets,
                expert_token_counts,
                self.global_expert_idx_table,
                local_expert_id=local_expert,
            )

        # Shape: (experts_per_chip, max_tokens, emb_dim)
        logger.debug(f"Final expert_outputs shape: {expert_outputs.shape}")

        return expert_outputs
