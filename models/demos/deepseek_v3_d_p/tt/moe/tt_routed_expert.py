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

from math import ceil
from pathlib import Path
from typing import Optional

import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.utility_functions import is_blackhole, is_wormhole_b0
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import ExpertMapping

COMPUTE_KERNEL_CONFIG_LOFI = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)


class TtRoutedExpert(LightweightModule):
    MAX_EXPERT_LENGTH = 2048
    MAX_EXPERT_ITERS = ceil(25_000 / MAX_EXPERT_LENGTH)

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
        self.gate_projs, self.up_projs, self.down_projs = result

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
        curr_expert_iter: int,
        x: ttnn.Tensor,
        gate_proj: ttnn.Tensor,
        up_proj: ttnn.Tensor,
        down_proj: ttnn.Tensor,
        out: Optional[ttnn.Tensor] = None,
        max_expert_iter_container: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """
        Single expert FFN computation.

        Args:
            curr_expert_iter: Index of the current chunk iteration within the expert loop.
                Used only on the Blackhole path; pass 0 on Wormhole.
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
            curr_expert_iter,
            x,
            gate_proj,
            up_proj,
            down_proj,
            compute_kernel_config=self.compute_kernel_config,
            output=out,
            max_expert_iter=max_expert_iter_container,
        )

    def _bh_forward_impl(
        self,
        dispatched_buffer: ttnn.Tensor,
        expert_token_counts: ttnn.Tensor = None,  # Unused for now, can reduce compute
        max_expert_iter_container: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """
        Blackhole forward implementation using narrow and in-place writes.

        Pre-allocates the output tensor with empty_like and uses narrow to extract
        per-expert slices and write FFN results directly into the output buffer,
        avoiding extra allocations from unsqueeze/concat.

        Early-stop behavior:
            max_expert_iter_container (device tensor, optional): threaded through
            to each routed matmul. The kernel reads it from DRAM per dispatch
            and skips iterations where curr_expert_iter > max_expert_iter.  No
            device-to-host sync - the loop dispatches every iteration and the
            on-device guard decides execute vs. skip.

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

        expert_outputs = ttnn.empty_like(dispatched_buffer)
        for local_expert in range(self.experts_per_chip):
            signpost(f"Expert {local_expert+1}/{self.experts_per_chip}")

            # Extract tokens for this expert
            # Shape: (1, max_tokens, emb_dim)
            tokens = ttnn.narrow(dispatched_buffer, dim=0, start=local_expert, length=1)
            out = ttnn.narrow(expert_outputs, dim=0, start=local_expert, length=1)
            logger.debug(f"Expert {local_expert}: input shape {tokens.shape}")

            expert_iters = ceil(tokens.shape[1] / self.MAX_EXPERT_LENGTH)
            expert_lengths = [self.MAX_EXPERT_LENGTH] * expert_iters
            expert_lengths[-1] = tokens.shape[1] - self.MAX_EXPERT_LENGTH * (expert_iters - 1)  # Handle any remainder
            start = 0
            for curr_expert_iter in range(expert_iters):
                signpost(f"FFN iteration {curr_expert_iter+1}/{expert_iters} for Expert {local_expert+1}")
                expert_tokens = ttnn.narrow(tokens, dim=1, start=start, length=expert_lengths[curr_expert_iter])
                expert_out = ttnn.narrow(out, dim=1, start=start, length=expert_lengths[curr_expert_iter])
                start += expert_lengths[curr_expert_iter]

                # Run FFN — pass curr_expert_iter so the on-device guard compares the
                # chunk iteration index against the max_expert_iter tensor.
                output = self._expert_ffn(
                    curr_expert_iter,
                    expert_tokens,
                    self.gate_projs[local_expert],
                    self.up_projs[local_expert],
                    self.down_projs[local_expert],
                    out=expert_out,
                    max_expert_iter_container=max_expert_iter_container,
                )
                logger.debug(
                    f"Expert {local_expert}: FFN iteration {curr_expert_iter+1} output shape {expert_out.shape}"
                )

            logger.debug(f"Expert {local_expert}: output shape {output.shape}")

        # Shape: (experts_per_chip, max_tokens, emb_dim)
        logger.debug(f"Final expert_outputs shape: {expert_outputs.shape}")

        return expert_outputs

    def _wh_forward_impl(
        self,
        dispatched_buffer: ttnn.Tensor,
        expert_token_counts: ttnn.Tensor = None,  # Unused for now, can reduce compute
    ) -> ttnn.Tensor:
        """
        Wormhole forward implementation using indexing, unsqueeze, and concat.

        Extracts per-expert tokens via tensor indexing, runs the FFN, then
        reassembles the output by unsqueezing and concatenating along the expert
        dimension.

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
            # curr_expert_iter is unused on the Wormhole path; pass 0.
            output = self._expert_ffn(
                0,
                tokens,
                self.gate_projs[local_expert],
                self.up_projs[local_expert],
                self.down_projs[local_expert],
                out=None,  # Let the FFN allocate output since we will concatenate later
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

    def forward(
        self,
        dispatched_buffer: ttnn.Tensor,
        expert_token_counts: ttnn.Tensor = None,  # Unused for now, can reduce compute
        max_expert_iter_container: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """
        Process dispatched tokens through local experts.

        Dispatches to the architecture-specific implementation based on the
        current device type (Wormhole or Blackhole).

        Args:
            dispatched_buffer: Dispatched tokens
                shape: (experts_per_chip, max_tokens, emb_dim)
            expert_token_counts: Optional token counts per expert per chip
                If provided, only processes tokens up to the count (currently unused,
                all tokens are processed for simplicity)
            max_expert_iter_container: Optional DRAM uint32 scalar tile.  When
                provided, each routed matmul reads it on-device and skips
                iterations where curr_expert_iter > max_expert_iter. Avoids
                any device-to-host sync. Blackhole only.

        Returns:
            expert_outputs: Expert output tensor, same shape as dispatched_buffer
        """
        if is_wormhole_b0():
            return self._wh_forward_impl(dispatched_buffer, expert_token_counts)
        elif is_blackhole():
            return self._bh_forward_impl(dispatched_buffer, expert_token_counts, max_expert_iter_container)
        else:
            raise ValueError("Unsupported device architecture")
