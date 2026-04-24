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
        tokens: ttnn.Tensor,
        out: ttnn.Tensor,
        gate_proj: ttnn.Tensor,
        up_proj: ttnn.Tensor,
        down_proj: ttnn.Tensor,
        local_expert_idx: int,
        expert_iter_length: int,
        global_expert_idx_table: Optional[ttnn.Tensor] = None,
        expert_token_counts: Optional[ttnn.Tensor] = None,
    ) -> None:
        """
        Chunked expert FFN: splits ``tokens`` into ``expert_iter_length``-sized
        chunks and dispatches each chunk through the experimental routed_expert_ffn
        op, writing results into the matching slice of ``out``.

        Each chunk passes its index as ``curr_expert_iter``. On Blackhole, when
        both ``global_expert_idx_table`` and ``expert_token_counts`` are provided,
        the op routes through the forked device op whose per-kernel guard skips
        iff ``expert_token_counts[global_expert_idx_table[local_expert_idx]]
        <= curr_expert_iter * expert_iter_length`` — no device-to-host sync. On
        Wormhole the guard parameters are ignored.

        Args:
            tokens: Per-expert input slice, shape (1, num_tokens, emb_dim).
            out: Pre-allocated per-expert output slice, same shape as ``tokens``.
            gate_proj, up_proj, down_proj: Expert projection weights.
            local_expert_idx: Index into ``global_expert_idx_table`` for this slot.
            expert_iter_length: Chunk size in tokens (guard compares against
                ``curr_expert_iter * expert_iter_length``).
            global_expert_idx_table: DRAM uint32 TILE_LAYOUT tensor shaped
                (1, 1, experts_per_chip).
            expert_token_counts: DRAM uint32 TILE_LAYOUT tensor shaped
                (1, 1, num_global_experts).
        """
        num_tokens = tokens.shape[1]
        expert_iters = ceil(num_tokens / expert_iter_length)
        expert_lengths = [expert_iter_length] * expert_iters
        expert_lengths[-1] = num_tokens - expert_iter_length * (expert_iters - 1)

        start = 0
        for curr_expert_iter in range(expert_iters):
            signpost(f"FFN iteration {curr_expert_iter+1}/{expert_iters}")
            expert_tokens = ttnn.narrow(tokens, dim=1, start=start, length=expert_lengths[curr_expert_iter])
            expert_out = ttnn.narrow(out, dim=1, start=start, length=expert_lengths[curr_expert_iter])
            start += expert_lengths[curr_expert_iter]

            ttnn.experimental.deepseek_prefill.routed_expert_ffn(
                expert_tokens,
                gate_proj,
                up_proj,
                down_proj,
                compute_kernel_config=self.compute_kernel_config,
                output=expert_out,
                global_expert_idx_table=global_expert_idx_table,
                expert_token_counts=expert_token_counts,
                local_expert_idx=local_expert_idx,
                curr_expert_iter=curr_expert_iter,
                expert_iter_length=expert_iter_length,
            )
            logger.debug(f"FFN iteration {curr_expert_iter+1} output shape {expert_out.shape}")

    def forward(
        self,
        dispatched_buffer: ttnn.Tensor,
        global_expert_idx_table: Optional[ttnn.Tensor] = None,
        expert_token_counts: Optional[ttnn.Tensor] = None,
        expert_iter_length: Optional[int] = None,
    ) -> ttnn.Tensor:
        """
        Process dispatched tokens through local experts.

        Pre-allocates the output tensor with empty_like, narrows per-expert slices
        for both input and output, and writes FFN results directly into the output
        buffer — avoiding the extra allocations that unsqueeze/concat would incur.

        Architecture dispatch happens inside the underlying experimental op, so the
        same code path runs on Wormhole and Blackhole. On Blackhole, when both
        ``global_expert_idx_table`` and ``expert_token_counts`` are supplied, each
        routed matmul reads them on-device and skips chunks where the expert's
        token count is already past the current chunk's starting offset. On
        Wormhole the guard tensors are ignored.

        Args:
            dispatched_buffer: Dispatched tokens, shape (experts_per_chip, max_tokens, emb_dim).
            global_expert_idx_table: DRAM uint32 TILE_LAYOUT tensor of shape
                (1, 1, experts_per_chip) mapping local expert slots to global
                expert ids.
            expert_token_counts: DRAM uint32 TILE_LAYOUT tensor of shape
                (1, 1, num_global_experts). Indexed by global expert id; holds
                the real token count per expert.
            expert_iter_length: Chunk size in tokens. Defaults to
                ``self.MAX_EXPERT_LENGTH`` when None.

        Returns:
            expert_outputs: Expert output tensor, same shape as ``dispatched_buffer``.
        """
        if expert_iter_length is None:
            expert_iter_length = self.MAX_EXPERT_LENGTH

        logger.debug(f"Forward pass: dispatched_buffer shape={dispatched_buffer.shape}")

        if dispatched_buffer.dtype != self.activations_dtype:
            logger.warning(f"{dispatched_buffer.dtype=} typecasting to {self.activations_dtype}")
            dispatched_buffer = ttnn.typecast(dispatched_buffer, self.activations_dtype)

        expert_outputs = ttnn.empty_like(dispatched_buffer)
        for local_expert in range(self.experts_per_chip):
            signpost(f"Expert {local_expert+1}/{self.experts_per_chip}")

            tokens = ttnn.narrow(dispatched_buffer, dim=0, start=local_expert, length=1)
            out = ttnn.narrow(expert_outputs, dim=0, start=local_expert, length=1)
            logger.debug(f"Expert {local_expert}: input shape {tokens.shape}")

            self._expert_ffn(
                tokens,
                out,
                self.gate_projs[local_expert],
                self.up_projs[local_expert],
                self.down_projs[local_expert],
                local_expert_idx=local_expert,
                expert_iter_length=expert_iter_length,
                global_expert_idx_table=global_expert_idx_table,
                expert_token_counts=expert_token_counts,
            )
            logger.debug(f"Expert {local_expert}: output shape {out.shape}")

        logger.debug(f"Final expert_outputs shape: {expert_outputs.shape}")
        return expert_outputs
