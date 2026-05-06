# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import ttnn
from models.common.utility_functions import is_blackhole

from ...layers.conv3d import ContextParallelConv3d
from ...layers.module import Module, ModuleList, Parameter
from ...layers.normalization import GroupNorm
from ...parallel.config import MochiVAEParallelConfig, vae_neighbor_pad, vae_slice_reshard
from ...parallel.manager import CCLManager
from ...utils.substate import rename_substate

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


def get_padded_size(numerator, denominator):
    return ((numerator + denominator - 1) // denominator) * denominator


class Conv1x1(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        bias: bool = True,
        swizzle_weight: Callable[[torch.Tensor], torch.Tensor] | None = None,
        mesh_device: ttnn.MeshDevice,
    ) -> None:
        """
        A 1x1 convolution implemented as a linear operation for ttnn.

        Can be instantiated from either:
        - nn.Conv3d with kernel_size=(1,1,1)
        - Conv1x1 (which is implemented as nn.Linear)

        Args:
            mesh_device: TTNN mesh device
            state_dict: Dictionary containing weights
            state_dict_prefix: Prefix for loading weights from state_dict
            in_channels: Number of input channels
            out_channels: Number of output channels
            bias: Whether to include bias
            swizzle_weight: Function to swizzle weights, useful for channel expansion
        """
        super().__init__()

        self.weight = Parameter(total_shape=[in_channels, out_channels], device=mesh_device)
        self.bias = Parameter(total_shape=[1, out_channels], device=mesh_device) if bias else None

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.swizzle_weight = swizzle_weight
        self.mesh_device = mesh_device

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        weight = state.get("weight")
        if weight is not None:
            # Load weights - supports both Conv3d(1,1,1) and Conv1x1 format
            if weight.ndim == 5:  # Conv3d weight
                # Convert from (out_channels, in_channels, 1, 1, 1) to (out_channels, in_channels)
                weight = weight.squeeze()
            weight = weight.transpose(0, 1)  # (out_channels, in_channels) -> (in_channels, out_channels)
            if self.swizzle_weight:
                weight = self.swizzle_weight(weight)
            state["weight"] = weight

        bias = state.get("bias")
        if bias is not None:
            if self.swizzle_weight:
                bias = self.swizzle_weight(bias)
            state["bias"] = bias.reshape(1, -1)

    def forward(self, x_NTHWC):
        """
        Forward pass for Conv1x1.

        Args:
            x_NTHWC: Input tensor in NTHWC layout

        Returns:
            Output tensor in NTHWC layout
        """
        # Convert to tile layout for efficient computation
        x_tile_NTHWC = ttnn.to_layout(x_NTHWC, ttnn.TILE_LAYOUT)
        ttnn.deallocate(x_NTHWC)

        # Apply linear transformation
        x_tile_NTHWO = ttnn.linear(
            x_tile_NTHWC,
            self.weight.data,
            bias=self.bias.data if self.bias is not None else None,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            core_grid=self.mesh_device.core_grid,
        )
        ttnn.deallocate(x_tile_NTHWC)

        # Convert back to row major layout
        x_NTHWO = ttnn.to_layout(x_tile_NTHWO, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(x_tile_NTHWO)

        return x_NTHWO


class ResBlock(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        causal: bool = True,
        padding_mode: str = "replicate",
        bias: bool = True,
        mesh_device: ttnn.MeshDevice,
        parallel_config: MochiVAEParallelConfig,
        ccl_manager: CCLManager,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ) -> None:
        super().__init__()

        self.core_grid_y_map = {
            # large latent
            768: {
                8: 4,  # 28 padded up to 32, divided by 8
                4: 7,  # 28/4
                2: 8,
                1: 8,
            },
            512: {
                8: 5,
                4: 7,
                2: 8,
                1: 8,
            },
            256: {
                8: 7,
                4: 8,
                2: 8,
                1: 8,
            },
        }
        self.num_out_blocks_map = {
            # Tuned entries for known H*W gathered spatial sizes.
            768: {
                40 * 50: 2,
                60 * 106: 8,
            },
            512: {
                80 * 100: 4,
                120 * 212: 10,
            },
            256: {
                160 * 200: 15,
                240 * 424: 40,
            },
            128: {
                320 * 400: 50,
                480 * 848: 140,
            },
        }
        # Max spatial elements per GroupNorm block, derived from tuned entries above.
        # Used as fallback for HW sizes not in the map (e.g. from padding differences
        # between Galaxy/T3K configs).
        self._max_hw_per_block = {768: 1000, 512: 2000, 256: 2133, 128: 2560}
        self.parallel_config = parallel_config
        self.ccl_manager = ccl_manager
        self.mesh_device = mesh_device

        grid_size_x = mesh_device.core_grid.x
        grid_size_y = (
            self.core_grid_y_map[in_channels][self.parallel_config.time_parallel.factor]
            if in_channels in (768, 512, 256)
            else mesh_device.core_grid.y
        )
        max_grid_y = 7 if is_blackhole() else mesh_device.core_grid.y
        grid_size_y = min(grid_size_y, max_grid_y)
        self.grid_size = ttnn.CoreGrid(y=grid_size_y, x=grid_size_x)

        # When spatial parallelism is used, GroupNorm runs on the full gathered
        # tensor (all frames replicated).
        h_factor = parallel_config.h_parallel.factor
        w_factor = parallel_config.w_parallel.factor
        if h_factor > 1 or w_factor > 1:
            equiv_time_factor = h_factor * w_factor
            target_y = self.core_grid_y_map.get(in_channels, {}).get(equiv_time_factor, grid_size_y)
            target_y = min(target_y, max_grid_y)
            self.spatial_norm_grid = ttnn.CoreGrid(y=target_y, x=grid_size_x)
        else:
            self.spatial_norm_grid = self.grid_size

        self.norm1 = GroupNorm(
            in_channels,
            num_groups=32,
            mesh_device=mesh_device,
            mesh_axis=None,
            core_grid=self.grid_size,
        )
        self.norm2 = GroupNorm(
            out_channels,
            num_groups=32,
            mesh_device=mesh_device,
            mesh_axis=None,
            core_grid=self.grid_size,
        )
        self.conv1 = ContextParallelConv3d(
            in_channels,
            out_channels,
            mesh_device=mesh_device,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding_mode=padding_mode,
            bias=bias,
            causal=causal,
            parallel_config=parallel_config,
            ccl_manager=ccl_manager,
            dtype=dtype,
        )
        self.conv2 = ContextParallelConv3d(
            out_channels,
            out_channels,
            mesh_device=mesh_device,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding_mode=padding_mode,
            bias=bias,
            causal=causal,
            parallel_config=parallel_config,
            ccl_manager=ccl_manager,
            dtype=dtype,
        )

        self.norm_compute_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        rename_substate(state, "norm1.norm_layer", "norm1")
        rename_substate(state, "norm2.norm_layer", "norm2")
        rename_substate(state, "conv1.conv", "conv1")
        rename_substate(state, "conv2.conv", "conv2")

    def get_tensor_shapes(self, x):
        return x.shape

    def reshape_tilize(self, x, shape):
        N, T, H, W, C = shape
        # TODO: When H * W is not a multiple of 32 (e.g. C=768: 40*50=2000),
        # tilize_with_zero_padding pads dim 2 to the next multiple of 32 (2016) with zeros.
        # Those zeros participate in GroupNorm's mean/variance calculation, corrupting the
        # normalization result.  Currently this affects both spatial and non-spatial paths
        # equally so cross-topology PCC is not impacted, but the absolute result is wrong.
        output = ttnn.tilize_with_zero_padding(
            ttnn.reshape(x, [N * T, 1, H * W, C]),
            use_multicore=True,
        )
        return output

    def untilize_reshape(self, x, shapes):
        N, T, H, W, C = shapes
        output = ttnn.reshape(ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT), [N, T, H, W, C])
        return output

    def _all_gather_hw(self, x_rm, N, T, H, W, C):
        """All-gather on H and W axes.

        Gathers H on dim=1 (outer dim) and W on dim=2.

        Args:
            x_rm: (N*T, H, W, C) in ROW_MAJOR layout (local per-device data).
        Returns:
            (N*T, H*h_factor, W*w_factor, C) in ROW_MAJOR layout.
        """
        h_factor = self.parallel_config.h_parallel.factor
        w_factor = self.parallel_config.w_parallel.factor

        x_tiled = ttnn.to_layout(x_rm, ttnn.TILE_LAYOUT)
        ttnn.deallocate(x_rm)
        if x_tiled.dtype != ttnn.bfloat16:
            x_tiled = ttnn.typecast(x_tiled, ttnn.bfloat16)

        if h_factor > 1:
            x_tiled = self.ccl_manager.all_gather(
                x_tiled,
                dim=1,
                mesh_axis=self.parallel_config.h_parallel.mesh_axis,
                use_hyperparams=False,
            )

        if w_factor > 1:
            x_tiled = self.ccl_manager.all_gather(
                x_tiled,
                dim=2,
                mesh_axis=self.parallel_config.w_parallel.mesh_axis,
                use_hyperparams=False,
            )

        x_rm = ttnn.to_layout(x_tiled, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(x_tiled)
        return x_rm

    def _partition_hw(self, x_NTHWC, N, T, H, W, C):
        """Sequentially mesh_partition on H then W axes, then neighbor pad."""
        h_factor = self.parallel_config.h_parallel.factor
        w_factor = self.parallel_config.w_parallel.factor
        # x_NTHWC is (N, T, H_full, W_full, C) in ROW_MAJOR
        if h_factor > 1:
            x_NTHWC = ttnn.mesh_partition(
                x_NTHWC,
                dim=2,
                cluster_axis=self.parallel_config.h_parallel.mesh_axis,
                memory_config=x_NTHWC.memory_config(),
            )
        if w_factor > 1:
            x_NTHWC = ttnn.mesh_partition(
                x_NTHWC,
                dim=3,
                cluster_axis=self.parallel_config.w_parallel.mesh_axis,
                memory_config=x_NTHWC.memory_config(),
            )
        x_NTHWC = ttnn.squeeze(x_NTHWC, 0)  # Get rid of N
        if h_factor > 1:
            x_NTHWC = vae_neighbor_pad(
                self.ccl_manager,
                x_NTHWC,
                cluster_axis=self.parallel_config.h_parallel.mesh_axis,
                dim=1,
                padding_left=1,
                padding_right=1,
                padding_mode="replicate",
            )
        if w_factor > 1:
            x_NTHWC = vae_neighbor_pad(
                self.ccl_manager,
                x_NTHWC,
                cluster_axis=self.parallel_config.w_parallel.mesh_axis,
                dim=2,
                padding_left=1,
                padding_right=1,
                padding_mode="replicate",
            )
        x_NTHWC = ttnn.unsqueeze(x_NTHWC, 0)
        return x_NTHWC

    @staticmethod
    def _valid_multicast(nvr, num_batches):
        """Check the uniform-multicast-group constraint from GroupNorm.

        When nvr < num_batches each virtual row handles at most one batch
        element, so multicast groups are trivially uniform.  When
        nvr >= num_batches, nvr must divide evenly by num_batches so every
        batch contributes the same number of rows to each multicast group.
        """
        return nvr < num_batches or nvr % num_batches == 0

    @staticmethod
    def _valid_norm_grid(norm, batch_size, HW):
        """Find a valid core_grid for GroupNorm given the actual tensor dimensions.

        The ttnn group_norm kernel requires:
            num_virtual_rows = (grid_x / num_virtual_cols) * grid_y
            Ht % num_virtual_rows == 0  and  num_virtual_rows <= Ht
            nvr < num_batches  OR  nvr % num_batches == 0  (uniform multicast)
        where Ht = batch_size * ceil(HW / 32).

        The grid computed at init time may be invalid when the actual HW differs
        from what was expected (e.g. after slicing padding, or on topologies
        where the per-device T differs).
        """
        Ht = batch_size * ((HW + 31) // 32)
        nvc = norm.num_virtual_cols
        # Try (grid_x, grid_y) combinations from largest to smallest.
        # Reducing grid_x is needed when no grid_y works for the max grid_x
        # (e.g. grid_x=12, nvc=4 → nvr=3*gy, but Ht not divisible by 3).
        for gx in range(norm.core_grid.x, nvc - 1, -1):
            for gy in range(norm.core_grid.y, 0, -1):
                nvr = (gx // nvc) * gy
                if nvr > 0 and nvr <= Ht and Ht % nvr == 0 and ResBlock._valid_multicast(nvr, batch_size):
                    return ttnn.CoreGrid(x=gx, y=gy)
        # Ultimate fallback: nvr=1 always satisfies the multicast constraint
        # (1 < batch_size for batch_size > 1) and divides any Ht.
        return ttnn.CoreGrid(x=nvc, y=1)

    @staticmethod
    def _min_valid_norm_batch(norm, HW):
        """Find the minimum batch_size for which a valid GroupNorm grid exists.

        Returns the smallest batch_size >= 1 such that _valid_norm_grid returns
        a grid where nvr actually divides Ht = batch_size * ceil(HW/32).
        """
        hw_tiles = (HW + 31) // 32
        nvc = norm.num_virtual_cols
        for bs in range(1, hw_tiles + 1):
            Ht = bs * hw_tiles
            for gx in range(norm.core_grid.x, nvc - 1, -1):
                for gy in range(norm.core_grid.y, 0, -1):
                    nvr = (gx // nvc) * gy
                    if nvr > 0 and nvr <= Ht and Ht % nvr == 0 and ResBlock._valid_multicast(nvr, bs):
                        return bs
        return 1

    @staticmethod
    def _safe_num_out_blocks(Ht, num_out_blocks):
        """Ensure num_out_blocks divides Ht evenly.

        The GroupNorm kernel requires Ht % num_out_blocks == 0.  When
        batch_size varies (e.g. from T-chunking), a tuned num_out_blocks
        value may no longer divide Ht.  Search upward first (more blocks
        = smaller CBs per iteration = less L1 pressure), then downward.
        """
        if num_out_blocks <= 0:
            return 1
        if Ht % num_out_blocks == 0:
            return num_out_blocks
        # Prefer more blocks to reduce per-iteration CB size and avoid
        # L1 overflow.  Ht values with large prime factors (e.g.
        # 4*199=796) have no divisors between 4 and 199; searching
        # downward would pick 4 which overflows L1.
        for candidate in range(num_out_blocks + 1, Ht + 1):
            if Ht % candidate == 0:
                return candidate
        for candidate in range(num_out_blocks - 1, 0, -1):
            if Ht % candidate == 0:
                return candidate
        return 1

    def _run_norm(self, norm, x_tiled, batch_size, HW, num_out_blocks, compute_kernel_config=None):
        """Run GroupNorm with auto-validated core_grid."""
        original_grid = norm.core_grid
        valid_grid = self._valid_norm_grid(norm, batch_size, HW)
        if valid_grid.x != original_grid.x or valid_grid.y != original_grid.y:
            norm.core_grid = valid_grid
        Ht = batch_size * ((HW + 31) // 32)
        num_out_blocks = self._safe_num_out_blocks(Ht, num_out_blocks)
        # GroupNorm kernel requires bf16 input; cast and restore if needed.
        input_dtype = x_tiled.dtype
        if input_dtype != ttnn.bfloat16:
            x_tiled = ttnn.typecast(x_tiled, ttnn.bfloat16)
        x_tiled = norm(x_tiled, num_out_blocks, compute_kernel_config=compute_kernel_config)
        if input_dtype != ttnn.bfloat16:
            x_tiled = ttnn.typecast(x_tiled, input_dtype)
        if norm.core_grid.x != original_grid.x or norm.core_grid.y != original_grid.y:
            norm.core_grid = original_grid
        return x_tiled

    def _norm_silu_spatial_chunk(
        self,
        x_4d_rm,
        norm,
        batch_size,
        H_full,
        W_full,
        C,
        logical_h,
        logical_w,
        output_dtype=None,
    ):
        """GroupNorm+silu on all-gathered spatial data for a single T-chunk.

        Args:
            x_4d_rm: (batch_size, H_full, W_full, C) in ROW_MAJOR layout
            batch_size: number of frames in this chunk
        Returns:
            (batch_size, H_full, W_full, C) in ROW_MAJOR layout
        """
        needs_h_unpad = logical_h > 0 and H_full > logical_h
        needs_w_unpad = logical_w > 0 and W_full > logical_w

        x_rm = x_4d_rm
        if needs_h_unpad or needs_w_unpad:
            norm_h = logical_h if needs_h_unpad else H_full
            norm_w = logical_w if needs_w_unpad else W_full
            x_rm = x_rm[:, :norm_h, :norm_w, :]
            ttnn.deallocate(x_4d_rm)
        else:
            norm_h, norm_w = H_full, W_full

        # TODO: When norm_h * norm_w is not a multiple of 32 (e.g. C=768: 40*50=2000),
        # tilize_with_zero_padding pads dim 2 to the next multiple of 32 (2016) with zeros.
        # Those zeros participate in GroupNorm's mean/variance calculation, corrupting the
        # normalization result.  Currently this affects both spatial and non-spatial paths
        # equally so cross-topology PCC is not impacted, but the absolute result is wrong.
        #
        # NOTE: tilize MUST run before deallocating x_rm — reshape returns a view
        # sharing the same device buffer, so deallocating x_rm first would free the
        # buffer that tilize needs to read from (use-after-free).
        x_tiled = ttnn.tilize_with_zero_padding(
            ttnn.reshape(x_rm, [batch_size, 1, norm_h * norm_w, C]),
            use_multicore=True,
        )
        ttnn.deallocate(x_rm)

        # Use padded HW from the tilized tensor shape for num_out_blocks lookup,
        # matching the non-spatial path which uses x_tiled.shape[2].  For C=768
        # logical HW=2000 pads to 2016; the map has an entry for 2000 (→2) but
        # the non-spatial fallback for 2016 gives 3.
        HW = x_tiled.shape[2]
        num_out_blocks = self.num_out_blocks_map.get(C, {}).get(
            HW, (HW + self._max_hw_per_block.get(C, 1000) - 1) // self._max_hw_per_block.get(C, 1000)
        )
        # Override GroupNorm grid to match the non-spatial (time-parallel) path.
        saved_grid = norm.core_grid
        norm.core_grid = self.spatial_norm_grid

        x_tiled = self._run_norm(
            norm,
            x_tiled,
            batch_size,
            HW,
            num_out_blocks,
            compute_kernel_config=self.norm_compute_config,
        )
        if output_dtype is not None and x_tiled.dtype != output_dtype:
            x_tiled = ttnn.typecast(x_tiled, output_dtype)
        x_tiled = ttnn.silu(x_tiled, output_tensor=x_tiled)

        # Restore original grid
        norm.core_grid = saved_grid

        x_rm = ttnn.to_layout(x_tiled, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(x_tiled)

        if needs_h_unpad or needs_w_unpad:
            x_rm = ttnn.reshape(x_rm, [batch_size, norm_h, norm_w, C])
            pad_w = W_full - norm_w
            pad_h = H_full - norm_h
            if pad_w > 0:
                last_col = x_rm[:, :, norm_w - 1 : norm_w, :]
                pad_block_w = ttnn.repeat(last_col, [1, 1, pad_w, 1])
                ttnn.deallocate(last_col)
                x_rm = ttnn.concat([x_rm, pad_block_w], dim=2)
                ttnn.deallocate(pad_block_w)
            if pad_h > 0:
                last_row = x_rm[:, norm_h - 1 : norm_h, :, :]
                pad_block_h = ttnn.repeat(last_row, [1, pad_h, 1, 1])
                ttnn.deallocate(last_row)
                x_rm = ttnn.concat([x_rm, pad_block_h], dim=1)
                ttnn.deallocate(pad_block_h)
        return x_rm

    def _gather_norm_partition(self, x_4d, norm, N, T, H, W, C, logical_h, logical_w):
        """All-gather → GroupNorm+silu → mesh_partition, with T-chunking to limit peak memory.

        GroupNorm is per-frame independent, so we can process a subset of T frames
        at a time.  This avoids allocating the full gathered tensor (which can be
        many GB at the C=128 stage with large T).

        After chunked processing, neighbor_pad is applied once on the full result.

        Args:
            x_4d: (N*T, H, W, C) in ROW_MAJOR layout (per-device data)
        Returns:
            (N, T, H+pad, W+pad, C) in ROW_MAJOR layout (per-device, after neighbor_pad)
        """
        h_factor = self.parallel_config.h_parallel.factor
        w_factor = self.parallel_config.w_parallel.factor
        H_full = H * h_factor
        W_full = W * w_factor
        NT = N * T
        output_dtype = x_4d.dtype  # all_gather forces bf16; restore this dtype after norm

        # Compute the maximum frames per chunk from a 1 GB memory budget.
        # _all_gather_hw casts to bf16 before gathering, so 2 bytes per element.
        max_frames = max(1, (1024 * 1024 * 1024) // (H_full * W_full * C * 2))

        # GroupNorm requires a minimum batch_size for certain HW values
        # (e.g. HW=2000 → 63 tile rows has no valid grid for batch_size=1).
        needs_h_unpad = logical_h > 0 and H_full > logical_h
        needs_w_unpad = logical_w > 0 and W_full > logical_w
        norm_h = logical_h if needs_h_unpad else H_full
        norm_w = logical_w if needs_w_unpad else W_full
        min_norm_batch = self._min_valid_norm_batch(norm, norm_h * norm_w)

        # Find the largest divisor of NT in [min_norm_batch, max_frames].
        # This guarantees all chunks are equal size and satisfy GroupNorm constraints.
        chunk_nt = NT  # default: no chunking
        if NT > max_frames:
            for d in range(min(max_frames, NT), max(min_norm_batch, 1) - 1, -1):
                if NT % d == 0:
                    chunk_nt = d
                    break
            else:
                # No exact divisor found; use max_frames (accept unequal last chunk).
                chunk_nt = max(max_frames, min_norm_batch)
        num_chunks = (NT + chunk_nt - 1) // chunk_nt

        if num_chunks == 1:
            x_gathered = self._all_gather_hw(x_4d, N, T, H, W, C)
            x_normed = self._norm_silu_spatial_chunk(
                x_gathered,
                norm,
                NT,
                H_full,
                W_full,
                C,
                logical_h,
                logical_w,
                output_dtype=output_dtype,
            )
            x_normed = ttnn.reshape(x_normed, [N, T, H_full, W_full, C])
            return self._partition_hw(x_normed, N, T, H, W, C)

        partitioned_chunks = []
        for chunk_idx, start in enumerate(range(0, NT, chunk_nt)):
            end = min(start + chunk_nt, NT)
            bs = end - start
            # Slice this chunk from the per-device data.
            # reallocate ensures the slice is an independent buffer — without it,
            # the slice may share x_4d's buffer, and _all_gather_hw's internal
            # deallocate would free x_4d, corrupting subsequent chunks.
            x_chunk = ttnn.reallocate(x_4d[start:end, :, :, :])

            x_chunk = self._all_gather_hw(x_chunk, 1, bs, H, W, C)

            x_chunk = self._norm_silu_spatial_chunk(
                x_chunk,
                norm,
                bs,
                H_full,
                W_full,
                C,
                logical_h,
                logical_w,
                output_dtype=output_dtype,
            )

            # Partition back to per-device spatial size (mesh_partition only, no neighbor_pad)
            x_chunk = ttnn.reshape(x_chunk, [1, bs, H_full, W_full, C])
            if h_factor > 1:
                x_chunk = ttnn.mesh_partition(
                    x_chunk,
                    dim=2,
                    cluster_axis=self.parallel_config.h_parallel.mesh_axis,
                    memory_config=x_chunk.memory_config(),
                )
            if w_factor > 1:
                x_chunk = ttnn.mesh_partition(
                    x_chunk,
                    dim=3,
                    cluster_axis=self.parallel_config.w_parallel.mesh_axis,
                    memory_config=x_chunk.memory_config(),
                )
            x_chunk = ttnn.squeeze(x_chunk, 0)  # remove N dim → (bs, H, W, C)
            partitioned_chunks.append(x_chunk)

        # Deallocate the original input now that all chunks are processed
        ttnn.deallocate(x_4d)

        # Concatenate partitioned chunks along T → (N*T, H, W, C)
        if len(partitioned_chunks) > 1:
            x_partitioned = ttnn.concat(partitioned_chunks, dim=0)
            for chunk in partitioned_chunks:
                ttnn.deallocate(chunk)
        else:
            x_partitioned = partitioned_chunks[0]

        # Neighbor pad once on the full result
        x_partitioned = ttnn.reshape(x_partitioned, [T, H, W, C])
        if h_factor > 1:
            x_partitioned = vae_neighbor_pad(
                self.ccl_manager,
                x_partitioned,
                cluster_axis=self.parallel_config.h_parallel.mesh_axis,
                dim=1,
                padding_left=1,
                padding_right=1,
                padding_mode="replicate",
            )
        if w_factor > 1:
            x_partitioned = vae_neighbor_pad(
                self.ccl_manager,
                x_partitioned,
                cluster_axis=self.parallel_config.w_parallel.mesh_axis,
                dim=2,
                padding_left=1,
                padding_right=1,
                padding_mode="replicate",
            )
        x_partitioned = ttnn.unsqueeze(x_partitioned, 0)  # Restore N dim → (1, T, H+pad, W+pad, C)

        return x_partitioned

    def forward(self, x_NTHWC: ttnn.Tensor, logical_h: int = 0, logical_w: int = 0) -> ttnn.Tensor:
        shapes = self.get_tensor_shapes(x_NTHWC)
        N, T, H, W, C = shapes
        h_factor = self.parallel_config.h_parallel.factor
        w_factor = self.parallel_config.w_parallel.factor
        is_spatial_parallel = h_factor > 1 or w_factor > 1

        if is_spatial_parallel:
            # Save residual from local data
            residual_tiled_NTHWC = ttnn.tilize_with_zero_padding(
                ttnn.reshape(x_NTHWC, [N * T, 1, H * W, C]),
                use_multicore=True,
            )
            # Reshape for separate H/W all-gathers
            x_4d = ttnn.reshape(x_NTHWC, [N * T, H, W, C])
            # Do NOT deallocate x_NTHWC here — reshape returns a view sharing
            # the same buffer.  Deallocating the parent frees the buffer that
            # x_4d (and its later slices) still needs to read from.
            residual_tiled_NTHWC = ttnn.reallocate(residual_tiled_NTHWC)
            # T-chunked gather → norm → silu → partition (+ neighbor_pad)
            x_NTHWC = self._gather_norm_partition(x_4d, self.norm1, N, T, H, W, C, logical_h, logical_w)
        else:
            x_tiled_NTHWC = self.reshape_tilize(x_NTHWC, shapes)
            ttnn.deallocate(x_NTHWC)
            residual_tiled_NTHWC = x_tiled_NTHWC
            H_full = H
            W_full = W
            gathered_shapes = (N, T, H_full, W_full, C)
            HW = x_tiled_NTHWC.shape[2]
            num_out_blocks = self.num_out_blocks_map.get(C, {}).get(
                HW, (HW + self._max_hw_per_block.get(C, 1000) - 1) // self._max_hw_per_block.get(C, 1000)
            )
            x_norm_tiled_NTHWC = self._run_norm(
                self.norm1,
                x_tiled_NTHWC,
                N * T,
                HW,
                num_out_blocks,
                compute_kernel_config=self.norm_compute_config,
            )
            x_norm_tiled_NTHWC = ttnn.silu(x_norm_tiled_NTHWC, output_tensor=x_norm_tiled_NTHWC)
            x_NTHWC = self.untilize_reshape(x_norm_tiled_NTHWC, gathered_shapes)
            ttnn.deallocate(x_norm_tiled_NTHWC)

        x_conv1_NTHWC = self.conv1(x_NTHWC)
        ttnn.deallocate(x_NTHWC)

        if is_spatial_parallel:
            # T-chunked gather → norm → silu → partition for conv1 output
            x_conv1_4d = ttnn.reshape(x_conv1_NTHWC, [N * T, H, W, C])
            # Do NOT deallocate x_conv1_NTHWC — see conv1 path comment above.
            x_NTHWC = self._gather_norm_partition(x_conv1_4d, self.norm2, N, T, H, W, C, logical_h, logical_w)
        else:
            x_conv1_tiled_NTHWC = self.reshape_tilize(x_conv1_NTHWC, shapes)
            ttnn.deallocate(x_conv1_NTHWC)
            HW = x_conv1_tiled_NTHWC.shape[2]
            num_out_blocks = self.num_out_blocks_map.get(C, {}).get(
                HW, (HW + self._max_hw_per_block.get(C, 1000) - 1) // self._max_hw_per_block.get(C, 1000)
            )
            x_tiled_NTHWC = self._run_norm(
                self.norm2,
                x_conv1_tiled_NTHWC,
                N * T,
                HW,
                num_out_blocks,
                compute_kernel_config=self.norm_compute_config,
            )
            ttnn.deallocate(x_conv1_tiled_NTHWC)
            x_tiled_NTHWC = ttnn.silu(x_tiled_NTHWC, output_tensor=x_tiled_NTHWC)
            x_NTHWC = self.untilize_reshape(x_tiled_NTHWC, gathered_shapes)
            ttnn.deallocate(x_tiled_NTHWC)

        x_conv2_NTHWC = self.conv2(x_NTHWC)
        ttnn.deallocate(x_NTHWC)
        x_conv2_tiled_NTHWC = self.reshape_tilize(x_conv2_NTHWC, shapes)
        ttnn.deallocate(x_conv2_NTHWC)

        x_tiled_NTHWC = ttnn.add(x_conv2_tiled_NTHWC, residual_tiled_NTHWC)
        x_NTHWC = self.untilize_reshape(x_tiled_NTHWC, shapes)
        ttnn.deallocate(x_conv2_tiled_NTHWC)
        ttnn.deallocate(residual_tiled_NTHWC)
        ttnn.deallocate(x_tiled_NTHWC)
        return x_NTHWC


class CausalUpsampleBlock(Module):
    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        in_channels: int,
        out_channels: int,
        num_res_blocks: int = 0,
        *,
        parallel_config: MochiVAEParallelConfig,
        ccl_manager: CCLManager,
        temporal_expansion: int = 2,
        spatial_expansion: int = 2,
        temporal_offset: int = 0,
        has_attention: bool = False,
        causal: bool = True,
        prune_bottleneck: bool = False,
        padding_mode: str = "replicate",
        bias: bool = True,
    ) -> None:
        super().__init__()

        assert causal
        assert not prune_bottleneck
        assert not has_attention
        self.mesh_device = mesh_device
        self.resnets = ModuleList(
            ResBlock(
                in_channels,
                in_channels,
                mesh_device=mesh_device,
                causal=causal,
                padding_mode=padding_mode,
                bias=bias,
                parallel_config=parallel_config,
                ccl_manager=ccl_manager,
            )
            for _ in range(num_res_blocks)
        )

        self.temporal_expansion = temporal_expansion
        self.spatial_expansion = spatial_expansion
        self.temporal_offset = temporal_offset
        self.out_channels = out_channels
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        self.reshard_time_map = {
            80 * 100: 84,
            160 * 200: 168,
            320 * 400: 168,
            120 * 212: 84,
            240 * 424: 168,
            480 * 848: 168,
        }

        # Swizzle conv1x1 weights
        def swizzle_weight(w):
            # X (C texp sexp sexp) -> X (texp sexp sexp C)
            w = w.reshape(-1, out_channels, temporal_expansion, spatial_expansion, spatial_expansion)
            w = w.permute(0, 2, 3, 4, 1)
            w = w.reshape(-1, temporal_expansion * spatial_expansion * spatial_expansion * out_channels)
            return w.squeeze()

        self.proj = Conv1x1(
            mesh_device=mesh_device,
            in_channels=in_channels,
            out_channels=out_channels * temporal_expansion * (spatial_expansion**2),
            bias=bias,
            swizzle_weight=swizzle_weight,
        )

    def depth_to_spacetime(self, x_NTHWC):
        """Depth-to-spacetime: rearrange channels into temporal and spatial expansion.

        Decomposes the 8D permute into three steps using at most 6D tensors,
        avoiding potential issues with high-dimensional permute on multi-device.

        Channel layout (after weight swizzle): (texp, sexp_h, sexp_w, C_out)
        Output: (B, T*texp, H*sexp, W*sexp, C_out)
        """
        texp, sexp = self.temporal_expansion, self.spatial_expansion
        B, T, H, W, C = x_NTHWC.shape
        C_out = self.out_channels
        sexp2_C = sexp * sexp * C_out  # sexp_h * sexp_w * C_out

        # Step 1: Temporal expansion
        # (B, T, H, W, texp, sexp_h*sexp_w*C_out)
        x = ttnn.reshape(x_NTHWC, [B, T, H, W, texp, sexp2_C])
        # Move texp before H,W: (B, T, texp, H, W, sexp_h*sexp_w*C_out)
        x = ttnn.permute(x, [0, 1, 4, 2, 3, 5])
        # Merge T*texp
        x = ttnn.reshape(x, [B, T * texp, H, W, sexp2_C])

        # Step 2: H spatial expansion
        # (B, T*texp, H, W, sexp_h, sexp_w*C_out)
        x = ttnn.reshape(x, [B, T * texp, H, W, sexp, sexp * C_out])
        # Move sexp_h before W: (B, T*texp, H, sexp_h, W, sexp_w*C_out)
        x = ttnn.permute(x, [0, 1, 2, 4, 3, 5])
        # Merge H*sexp_h
        x = ttnn.reshape(x, [B, T * texp, H * sexp, W, sexp * C_out])

        # Step 3: W spatial expansion — just reshape (no permute needed)
        # In the current layout, the last dim is sexp_w * C_out.
        # Splitting this as (W*sexp_w, C_out) is a valid reshape because the
        # flat memory order is identical: w*(sexp*C_out) + s_w*C_out + c.
        x = ttnn.reshape(x, [B, T * texp, H * sexp, W * sexp, C_out])

        if texp > 1 and self.temporal_offset > 0 and self.parallel_config.time_parallel.factor == 1:
            # Only slice when all temporal data is on each device.
            # When time-parallel, reshard_output handles the offset.
            x = ttnn.slice(x, [0, self.temporal_offset, 0, 0, 0], [B, T * texp, H * sexp, W * sexp, C_out])
        return x

    def reshard_output(self, x_NTHWC):
        N, T, H, W, C = x_NTHWC.shape
        num_devices = self.parallel_config.time_parallel.factor
        if not (num_devices > 1 and self.temporal_expansion > 1):
            return x_NTHWC
        expected_T = self.reshard_time_map[
            H * W * self.parallel_config.h_parallel.factor * self.parallel_config.w_parallel.factor
        ]
        input_is_padded = T * num_devices != expected_T
        if self.temporal_offset > 0 or input_is_padded:
            padded_T = ((expected_T + num_devices - 1) // num_devices) * num_devices
            x_NTHWC = ttnn.squeeze(x_NTHWC, 0)
            x_NTHWC = vae_slice_reshard(
                self.ccl_manager,
                x_NTHWC,
                cluster_axis=self.parallel_config.time_parallel.mesh_axis,
                dim=0,
                output_shape=padded_T,
                output_offset=self.temporal_offset,
            )
            x_NTHWC = ttnn.unsqueeze(x_NTHWC, 0)
        return x_NTHWC

    def forward(self, x_NTHWC: ttnn.Tensor, logical_h: int = 0, logical_w: int = 0) -> ttnn.Tensor:
        N, T, H, W, C = x_NTHWC.shape
        for block in self.resnets:
            x_NTHWC = block(x_NTHWC, logical_h, logical_w)
        x_NTHWO = self.proj(x_NTHWC)
        x_NTHWC = self.depth_to_spacetime(x_NTHWO)
        ttnn.deallocate(x_NTHWO)
        x_NTHWC = self.reshard_output(x_NTHWC)
        return x_NTHWC


class MochiVAEDecoder(Module):
    def __init__(
        self,
        out_channels: int,
        *,
        mesh_device: ttnn.MeshDevice,
        parallel_config: MochiVAEParallelConfig,
        ccl_manager: CCLManager,
        base_channels: int = 128,
        channel_multipliers: Sequence[int] = (1, 2, 4, 6),
        temporal_expansions: Sequence[int] = (1, 2, 3),
        spatial_expansions: Sequence[int] = (2, 2, 2),
        num_res_blocks: Sequence[int] = (3, 3, 4, 6, 3),
        latent_dim: int = 12,
        has_attention: Sequence[bool] = (False, False, False, False, False),
        nonlinearity: str = "silu",
        output_nonlinearity: str = "silu",
        causal: bool = True,
        latents_mean=None,
        latents_std=None,
        scaling_factor=1.0,
    ) -> None:
        """
        TTNN implementation of the VAE Decoder.
        """
        super().__init__()

        self.input_channels = latent_dim
        self.output_channels = out_channels
        self.base_channels = base_channels
        self.channel_multipliers = channel_multipliers
        self.num_res_blocks = num_res_blocks
        self.temporal_expansions = temporal_expansions
        self.spatial_expansions = spatial_expansions
        self.output_nonlinearity = output_nonlinearity
        self.mesh_device = mesh_device
        self.config = lambda: None
        self.config.latents_mean = latents_mean
        self.config.latents_std = latents_std
        self.config.scaling_factor = scaling_factor
        assert nonlinearity == "silu"
        assert causal
        assert not any(has_attention), "Attention is not supported in the decoder"

        # Calculate channels for each level
        ch = [mult * base_channels for mult in channel_multipliers]
        self.num_up_blocks = len(ch) - 1
        self.parallel_config = parallel_config
        self.ccl_manager = ccl_manager
        assert len(num_res_blocks) == self.num_up_blocks + 2

        assert len(temporal_expansions) == len(spatial_expansions) == self.num_up_blocks
        assert len(num_res_blocks) == len(has_attention) == self.num_up_blocks + 2

        # Create the initial projection from latent space
        self.input_proj = Conv1x1(latent_dim, ch[-1], mesh_device=mesh_device)

        # First set of residual blocks
        self.first_blocks = ModuleList(
            ResBlock(
                ch[-1],
                ch[-1],
                mesh_device=mesh_device,
                causal=causal,
                padding_mode="replicate",
                parallel_config=parallel_config,
                ccl_manager=ccl_manager,
            )
            for _ in range(num_res_blocks[-1])
        )

        # Create upsampling blocks
        self.up_blocks = ModuleList(
            CausalUpsampleBlock(
                mesh_device=mesh_device,
                in_channels=ch[-i - 1],
                out_channels=ch[-i - 2],
                num_res_blocks=num_res_blocks[-i - 2],
                temporal_expansion=temporal_expansions[-i - 1],
                spatial_expansion=spatial_expansions[-i - 1],
                causal=causal,
                padding_mode="replicate",
                parallel_config=parallel_config,
                ccl_manager=ccl_manager,
            )
            for i in range(len(ch) - 1)
        )

        # Last set of residual blocks
        self.last_blocks = ModuleList(
            ResBlock(
                ch[0],
                ch[0],
                mesh_device=mesh_device,
                causal=causal,
                padding_mode="replicate",
                parallel_config=parallel_config,
                ccl_manager=ccl_manager,
            )
            for _ in range(num_res_blocks[0])
        )

        # Final output projection
        self.output_proj = Conv1x1(ch[0], out_channels, mesh_device=mesh_device)

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        rename_substate(state, "conv_in", "input_proj")
        rename_substate(state, "proj_out", "output_proj")

        rename_substate(state, "block_in.resnets", "first_blocks")
        rename_substate(state, "block_out.resnets", "last_blocks")

    def dealloc(self):
        self.input_proj.dealloc()
        for block in self.first_blocks:
            block.dealloc()
        for block in self.up_blocks:
            block.dealloc()
        for block in self.last_blocks:
            block.dealloc()
        self.output_proj.dealloc()

    def _get_shard_dims(self):
        """Get ShardTensor2dMesh/ConcatMesh2dToTensor dims based on parallel config.

        For axes with no parallelism (factor=1), uses dummy dims.
        These are safe because those axes have mesh_shape=1 (no actual sharding/concat).
        Dims must be unique (required by the framework).
        """
        dims = [0, 1]
        if self.parallel_config.h_parallel.factor > 1:
            dims[self.parallel_config.h_parallel.mesh_axis] = 2
        if self.parallel_config.w_parallel.factor > 1:
            dims[self.parallel_config.w_parallel.mesh_axis] = 3
        if self.parallel_config.time_parallel.factor > 1:
            dims[self.parallel_config.time_parallel.mesh_axis] = 1
        return dims

    def prepare_input(self, x_NCTHW):
        N, C, T, H, W = x_NCTHW.shape
        x_NTHWC = x_NCTHW.permute(0, 2, 3, 4, 1)  # [N, T, H, W, C]

        if self.parallel_config.time_parallel.factor > 1 and T % self.parallel_config.time_parallel.factor:
            padded_T = get_padded_size(T, self.parallel_config.time_parallel.factor)
            x_NTHWC = torch.nn.functional.pad(x_NTHWC, pad=(0, 0, 0, 0, 0, 0, 0, padded_T - T))
        if self.parallel_config.w_parallel.factor > 1 and W % self.parallel_config.w_parallel.factor:
            padded_W = get_padded_size(W, self.parallel_config.w_parallel.factor)
            x_NTHWC = torch.nn.functional.pad(x_NTHWC, pad=(0, 0, 0, padded_W - W))
        if self.parallel_config.h_parallel.factor > 1 and H % self.parallel_config.h_parallel.factor:
            padded_H = get_padded_size(H, self.parallel_config.h_parallel.factor)
            x_NTHWC = torch.nn.functional.pad(x_NTHWC, pad=(0, 0, 0, 0, 0, padded_H - H))

        dims = self._get_shard_dims()

        tt_x_NTHWC = ttnn.from_torch(
            x_NTHWC,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, mesh_shape=tuple(self.mesh_device.shape), dims=dims),
        )

        return tt_x_NTHWC

    def postprocess_output(self, tt_x_NTHWC, input_shape):
        N, C, T, H, W = input_shape

        dims = self._get_shard_dims()

        # Convert TT output to torch tensor (includes padding columns)
        x_NTHWC_torch = ttnn.to_torch(
            tt_x_NTHWC,
            mesh_composer=ttnn.ConcatMesh2dToTensor(
                self.mesh_device, mesh_shape=tuple(self.mesh_device.shape), dims=dims
            ),
        )
        ttnn.deallocate(tt_x_NTHWC)

        # Compute valid output spatial dimensions from the input (latent)
        # dimensions and the decoder's spatial expansion factors.
        sexp_product = 1
        for s in self.spatial_expansions:
            sexp_product *= s
        H_out = H * sexp_product
        W_out = W * sexp_product

        # Slice off padding introduced for spatial parallelism.  The padding
        # is concentrated at the global boundary (last device), so slicing
        # must happen AFTER gathering, not per-device.
        x_NTHWC_torch = x_NTHWC_torch[:, :, :H_out, :W_out, :]

        x_NCTHW_torch = x_NTHWC_torch.permute(0, 4, 1, 2, 3)  # [N, C, T, H, W]
        return x_NCTHW_torch

    def forward(self, x_NTHWC: ttnn.Tensor, logical_h: int = 0, logical_w: int = 0) -> ttnn.Tensor:
        """
        Forward pass for the decoder.

        Args:
            x_NTHWC: Input tensor in NTHWC layout
            logical_h: Unpadded height (0 = no padding removal needed)
            logical_w: Unpadded width (0 = no padding removal needed)

        Returns:
            Output tensor in NTHWC layout
        """
        # Initial projection
        x_NTHWC = self.input_proj(x_NTHWC)

        # First set of residual blocks
        for block in self.first_blocks:
            x_res_NTHWC = block(x_NTHWC, logical_h, logical_w)
            ttnn.deallocate(x_NTHWC)
            x_NTHWC = x_res_NTHWC

        # Upsampling blocks
        for block in self.up_blocks:
            x_res_NTHWC = block(x_NTHWC, logical_h, logical_w)
            ttnn.deallocate(x_NTHWC)
            x_NTHWC = x_res_NTHWC
            # Update logical dims after spatial expansion
            if logical_h > 0:
                logical_h *= block.spatial_expansion
            if logical_w > 0:
                logical_w *= block.spatial_expansion

        # Last set of residual blocks
        for block in self.last_blocks:
            x_res_NTHWC = block(x_NTHWC, logical_h, logical_w)
            ttnn.deallocate(x_NTHWC)
            x_NTHWC = x_res_NTHWC

        # Apply output nonlinearity if needed
        if self.output_nonlinearity == "silu":
            x_tile_NTHWC = ttnn.to_layout(x_NTHWC, ttnn.TILE_LAYOUT)
            ttnn.deallocate(x_NTHWC)
            x_tile_NTHWC = ttnn.silu(x_tile_NTHWC, output_tensor=x_tile_NTHWC)  # in-place
            x_NTHWC = ttnn.to_layout(x_tile_NTHWC, ttnn.ROW_MAJOR_LAYOUT)
            ttnn.deallocate(x_tile_NTHWC)
        else:
            assert not self.output_nonlinearity  # StyleGAN3 omits the to-RGB nonlinearity.

        # Final projection
        x_NTHWC = self.output_proj(x_NTHWC)

        return x_NTHWC

    def decode(self, x_NCTHW, return_dict):
        input_shape = x_NCTHW.shape
        N, C, T, H, W = input_shape
        tt_x_NTHWC = self.prepare_input(x_NCTHW)

        logical_h = H if self.parallel_config.h_parallel.factor > 1 else 0
        logical_w = W if self.parallel_config.w_parallel.factor > 1 else 0
        tt_x_NTHWC = self(tt_x_NTHWC, logical_h, logical_w)

        x_NCTHW_torch = self.postprocess_output(tt_x_NTHWC, input_shape)

        return [x_NCTHW_torch]
