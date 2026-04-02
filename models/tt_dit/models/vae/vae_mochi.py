# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole


def _debug_stats(label, tt_tensor):
    """Print shape and statistics of a ttnn tensor for debugging."""
    t = ttnn.to_torch(ttnn.get_device_tensors(tt_tensor)[0]).float()
    logger.info(
        f"[DEBUG] {label}: shape={list(tt_tensor.shape)} layout={tt_tensor.layout} "
        f"mean={t.mean().item():.6f} std={t.std().item():.6f} "
        f"min={t.min().item():.6f} max={t.max().item():.6f}"
    )


def _get_shard_dims(parallel_config):
    """Compute ConcatMesh2dToTensor dims from parallel config (same logic as test helper)."""
    dims = [0, 1]
    if parallel_config.h_parallel.factor > 1:
        dims[parallel_config.h_parallel.mesh_axis] = 2
    if parallel_config.w_parallel.factor > 1:
        dims[parallel_config.w_parallel.mesh_axis] = 3
    if parallel_config.time_parallel.factor > 1:
        dims[parallel_config.time_parallel.mesh_axis] = 1
    return dims


from ...layers.conv3d import ContextParallelConv3d
from ...layers.module import Module, ModuleList, Parameter
from ...layers.normalization import GroupNorm
from ...parallel.config import MochiVAEParallelConfig, vae_all_gather, vae_neighbor_pad, vae_slice_reshard
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
        output = ttnn.tilize_with_zero_padding(
            ttnn.reshape(x, [N * T, 1, H * W, C]),
            use_multicore=True,
        )
        return output

    def pre_all_gather_reshape_norm_1(self, x, shape):
        N, T, H, W, C = shape
        dim_2_1 = ttnn.reshape(x, [N * T, 1, H * W, C])
        residual = ttnn.tilize_with_zero_padding(
            dim_2_1,
            use_multicore=True,
        )
        assert not (C % 32)
        if not (H * W * (C // 32) % 32):
            output = ttnn.reshape(residual, [N * T, 1, H * W * (C // 32), 32])
            gather_dim = 2
        else:
            output = ttnn.reshape(residual, [1, 1, N * T, H * W * C])
            gather_dim = 3
        ttnn.deallocate(dim_2_1)
        return gather_dim, residual, output

    def pre_all_gather_reshape_norm_2(self, x, shape):
        N, T, H, W, C = shape
        assert not (C % 32)
        if not (H * W * (C // 32) % 32):
            x = ttnn.reshape(x, (N * T, 1, H * W * (C // 32), 32))
            gather_dim = 2
        else:
            x = ttnn.reshape(x, (1, 1, N * T, H * W * C))
            gather_dim = 3
        return gather_dim, x

    def sharded_reshape_untilize_tilize(self, x, input_shapes, output_shapes, deallocate_input):
        N, T, H, W, C = input_shapes
        dim0, dim1, dim2, dim3 = output_shapes
        if C == 768:
            x = ttnn.reshape(x, (1, 1, N * T * H, W * C))
        output = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        if deallocate_input:
            ttnn.deallocate(x)
        output = ttnn.reshape(output, [dim0, dim1, dim2, dim3])
        output = ttnn.tilize_with_zero_padding(
            output,
            use_multicore=True,
        )
        return output

    def untilize_reshape(self, x, shapes):
        N, T, H, W, C = shapes
        output = ttnn.reshape(ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT), [N, T, H, W, C])
        return output

    def _dump_tensor(self, name, tensor, dump_dir, valid_NTHWC, is_gathered=False):
        """Gather tensor to full unpadded shape and save to dump_dir.

        Args:
            name: filename stem (e.g. "00_input")
            tensor: ttnn tensor (may be in any layout)
            dump_dir: directory to save into (None = no-op)
            valid_NTHWC: (N, T, H, W, C) unpadded dimensions
            is_gathered: True if tensor is replicated across devices
                         (post all-gather).  We partition it back to local
                         chunks before reading — device-0 reads of replicated
                         tensors are unreliable.
        """
        if dump_dir is None:
            return
        import os

        N, T, H, W, C = valid_NTHWC
        rm = ttnn.to_layout(tensor, ttnn.ROW_MAJOR_LAYOUT) if tensor.layout != ttnn.ROW_MAJOR_LAYOUT else tensor
        if is_gathered:
            # Partition replicated tensor back to local chunks so we can read
            # reliably via ConcatMesh2dToTensor (device-0 reads are unreliable
            # for replicated tensors).
            tmp = rm
            h_factor = self.parallel_config.h_parallel.factor
            w_factor = self.parallel_config.w_parallel.factor
            if h_factor > 1:
                tmp = ttnn.mesh_partition(
                    tmp,
                    dim=2,
                    cluster_axis=self.parallel_config.h_parallel.mesh_axis,
                    memory_config=tmp.memory_config(),
                )
            if w_factor > 1:
                tmp = ttnn.mesh_partition(
                    tmp,
                    dim=3,
                    cluster_axis=self.parallel_config.w_parallel.mesh_axis,
                    memory_config=tmp.memory_config(),
                )
            shard_dims = _get_shard_dims(self.parallel_config)
            t = ttnn.to_torch(
                tmp,
                mesh_composer=ttnn.ConcatMesh2dToTensor(
                    self.mesh_device, mesh_shape=tuple(self.mesh_device.shape), dims=shard_dims
                ),
            ).float()
            if tmp is not rm:
                ttnn.deallocate(tmp)
        else:
            shard_dims = _get_shard_dims(self.parallel_config)
            t = ttnn.to_torch(
                rm,
                mesh_composer=ttnn.ConcatMesh2dToTensor(
                    self.mesh_device, mesh_shape=tuple(self.mesh_device.shape), dims=shard_dims
                ),
            ).float()
        if rm is not tensor:
            ttnn.deallocate(rm)
        # Reshape to 5D (N, T, H_maybe_padded, W_maybe_padded, C) if needed
        if t.dim() == 4:
            t = t.reshape(N, T, t.shape[1], t.shape[2], C)
        # Slice to valid unpadded dims
        t = t[:N, :T, :H, :W, :C]
        torch.save(t, os.path.join(dump_dir, f"{name}.pt"))
        logger.info(f"[DUMP] {name}: shape={list(t.shape)} mean={t.mean():.6f} std={t.std():.6f}")

    def _all_gather_hw(self, x_rm, N, T, H, W, C):
        """All-gather on H and W axes using tile-padded spatial dimensions.

        All-gather requires each device's extent on the gather dim to be a
        multiple of 32 (TILE_SIZE).  Per-device H or W may not satisfy this
        (e.g. H=30, W=27).  Gathering on H or W directly would interleave
        tile-padding zeros between devices' valid data.

        Strategy: reshape so H (or W) is in the tile-row dim (dim[-2]) and use
        tilize_with_zero_padding to pad it to the next tile boundary.  After
        all-gather, reshape to (batch, factor, H_pad, ...) and slice out the
        padding rows to recover (batch, H*factor, ...).

        Args:
            x_rm: (N*T, H, W, C) in ROW_MAJOR layout (local per-device data).
        Returns:
            (N*T, H*h_factor, W*w_factor, C) in ROW_MAJOR layout.
        """
        h_factor = self.parallel_config.h_parallel.factor
        w_factor = self.parallel_config.w_parallel.factor
        H_full = H * h_factor
        W_full = W * w_factor

        if h_factor > 1:
            # Phase 1: H-gather.
            # Shape as (N*T, 1, H, W*C) so H is the tile-row dim.
            # tilize_with_zero_padding pads H → H_pad = ceil_32(H).
            wc = W * C
            assert wc % 32 == 0, f"W*C={wc} must be a multiple of 32; W={W}, C={C}"
            x_flat = ttnn.reshape(x_rm, [N * T, 1, H, wc])
            ttnn.deallocate(x_rm)
            x_tiled = ttnn.tilize_with_zero_padding(x_flat, use_multicore=True)
            ttnn.deallocate(x_flat)
            H_pad = x_tiled.shape[2]  # ceil_32(H), may equal H
            x_tiled = vae_all_gather(
                self.ccl_manager,
                x_tiled,
                cluster_axis=self.parallel_config.h_parallel.mesh_axis,
                dim=2,
                reshape=False,
            )
            x_rm = ttnn.to_layout(x_tiled, ttnn.ROW_MAJOR_LAYOUT)
            ttnn.deallocate(x_tiled)
            if H_pad != H:
                # Strip tile-padding rows from each device's chunk
                x_rm = ttnn.reshape(x_rm, [N * T, h_factor, H_pad, W, C])
                x_rm = ttnn.slice(x_rm, [0, 0, 0, 0, 0], [N * T, h_factor, H, W, C])
                x_rm = ttnn.reshape(x_rm, [N * T, H_full, W, C])
            else:
                x_rm = ttnn.reshape(x_rm, [N * T, H_full, W, C])

        if w_factor > 1:
            # Phase 2: W-gather.
            # Shape as (N*T*H_full, 1, W, C) so W is the tile-row dim.
            # tilize_with_zero_padding pads W → W_pad = ceil_32(W).
            nt_h = N * T * H_full
            assert C % 32 == 0, f"C={C} must be a multiple of 32"
            x_flat = ttnn.reshape(x_rm, [nt_h, 1, W, C])
            ttnn.deallocate(x_rm)
            x_tiled = ttnn.tilize_with_zero_padding(x_flat, use_multicore=True)
            ttnn.deallocate(x_flat)
            W_pad = x_tiled.shape[2]  # ceil_32(W), may equal W
            x_tiled = vae_all_gather(
                self.ccl_manager,
                x_tiled,
                cluster_axis=self.parallel_config.w_parallel.mesh_axis,
                dim=2,
                reshape=False,
            )
            x_rm = ttnn.to_layout(x_tiled, ttnn.ROW_MAJOR_LAYOUT)
            ttnn.deallocate(x_tiled)
            if W_pad != W:
                # Strip tile-padding from each device's chunk
                x_rm = ttnn.reshape(x_rm, [nt_h, w_factor, W_pad, C])
                x_rm = ttnn.slice(x_rm, [0, 0, 0, 0], [nt_h, w_factor, W, C])
                x_rm = ttnn.reshape(x_rm, [N * T, H_full, W_full, C])
            else:
                x_rm = ttnn.reshape(x_rm, [N * T, H_full, W_full, C])

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
    def _valid_norm_grid(norm, batch_size, HW):
        """Find a valid core_grid for GroupNorm given the actual tensor dimensions.

        The ttnn group_norm kernel requires:
            num_virtual_rows = (grid_x / num_virtual_cols) * grid_y
            Ht % num_virtual_rows == 0  and  num_virtual_rows <= Ht
        where Ht = batch_size * ceil(HW / 32).

        The grid computed at init time may be invalid when the actual HW differs
        from what was expected (e.g. after slicing padding, or on topologies
        where the per-device T differs).
        """
        Ht = batch_size * ((HW + 31) // 32)
        grid_x = norm.core_grid.x
        nvc = norm.num_virtual_cols
        for gy in range(norm.core_grid.y, 0, -1):
            nvr = (grid_x // nvc) * gy
            if nvr <= Ht and Ht % nvr == 0:
                return ttnn.CoreGrid(x=grid_x, y=gy)
        return ttnn.CoreGrid(x=grid_x, y=1)

    def _run_norm(self, norm, x_tiled, batch_size, HW, num_out_blocks, dump_dir=None, dump_tag=None):
        """Run GroupNorm with auto-validated core_grid."""
        original_grid = norm.core_grid
        valid_grid = self._valid_norm_grid(norm, batch_size, HW)
        if valid_grid.y != original_grid.y:
            norm.core_grid = valid_grid
        Ht = batch_size * ((HW + 31) // 32)
        logger.info(
            f"[NORM] batch_size={batch_size} HW={HW} Ht={Ht} num_out_blocks={num_out_blocks} "
            f"grid=({norm.core_grid.x},{norm.core_grid.y}) nvc={norm.num_virtual_cols}"
        )
        if dump_dir is not None and dump_tag is not None:
            import os

            t = ttnn.to_torch(ttnn.get_device_tensors(x_tiled)[0]).float()
            torch.save(t, os.path.join(dump_dir, f"{dump_tag}_pre_norm.pt"))
            logger.info(f"[DUMP] {dump_tag}_pre_norm: shape={list(t.shape)} mean={t.mean():.6f} std={t.std():.6f}")
        x_tiled = norm(x_tiled, num_out_blocks)
        if norm.core_grid is not original_grid:
            norm.core_grid = original_grid
        return x_tiled

    def _norm_silu_spatial(self, x_4d_rm, norm, N, T, H_full, W_full, C, logical_h, logical_w, dump_dir=None):
        """GroupNorm+silu on all-gathered spatial data, removing padding before norm.

        When spatial parallel factors don't evenly divide the spatial dimensions,
        the tensor is padded with zeros. These zeros contaminate GroupNorm statistics
        (which normalizes across spatial dimensions). This method slices to logical
        (unpadded) dims before GroupNorm, then re-pads with replicated edge values
        after silu.

        Args:
            x_4d_rm: (N*T, H_full, W_full, C) in ROW_MAJOR layout after all-gather
            norm: GroupNorm module
            logical_h, logical_w: unpadded spatial dims (0 = no unpadding needed)
        Returns:
            (N, T, H_full, W_full, C) in ROW_MAJOR layout
        """
        needs_h_unpad = logical_h > 0 and H_full > logical_h
        needs_w_unpad = logical_w > 0 and W_full > logical_w
        norm_name = "norm1" if norm is self.norm1 else "norm2"

        _debug_stats(f"{norm_name}: input (all-gathered)", x_4d_rm)

        x_rm = x_4d_rm
        if needs_h_unpad or needs_w_unpad:
            norm_h = logical_h if needs_h_unpad else H_full
            norm_w = logical_w if needs_w_unpad else W_full
            x_rm = x_rm[:, :norm_h, :norm_w, :]
            ttnn.deallocate(x_4d_rm)
            _debug_stats(f"{norm_name}: after slice to ({norm_h},{norm_w})", x_rm)
        else:
            norm_h, norm_w = H_full, W_full

        x_tiled = ttnn.reshape(x_rm, [N * T, 1, norm_h * norm_w, C])
        ttnn.deallocate(x_rm)
        x_tiled = ttnn.tilize_with_zero_padding(x_tiled, use_multicore=True)

        _debug_stats(f"{norm_name}: pre-norm (reshaped+tilized)", x_tiled)

        HW = norm_h * norm_w
        num_out_blocks = self.num_out_blocks_map.get(C, {}).get(
            HW, (HW + self._max_hw_per_block.get(C, 1000) - 1) // self._max_hw_per_block.get(C, 1000)
        )
        dump_tag = ("01" if norm is self.norm1 else "03") if dump_dir else None
        x_tiled = self._run_norm(norm, x_tiled, N * T, HW, num_out_blocks, dump_dir=dump_dir, dump_tag=dump_tag)

        _debug_stats(f"{norm_name}: post-norm", x_tiled)

        x_tiled = ttnn.silu(x_tiled, output_tensor=x_tiled)

        _debug_stats(f"{norm_name}: post-silu", x_tiled)

        x_rm = ttnn.to_layout(x_tiled, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(x_tiled)

        # Dump post-silu, pre-repad: freshly created by norm kernel, safe to
        # read from device 0 (not an all-gather buffer).
        if dump_dir is not None:
            import os

            tag = "01a" if norm is self.norm1 else "03a"
            t = ttnn.to_torch(ttnn.get_device_tensors(x_rm)[0]).float()
            t = t.reshape(N * T, norm_h, norm_w, C)[:, : logical_h or norm_h, : logical_w or norm_w, :]
            torch.save(t, os.path.join(dump_dir, f"{tag}_norm_silu_pre_repad.pt"))
            logger.info(
                f"[DUMP] {tag}_norm_silu_pre_repad: shape={list(t.shape)} mean={t.mean():.6f} std={t.std():.6f}"
            )

        if needs_h_unpad or needs_w_unpad:
            x_rm = ttnn.reshape(x_rm, [N * T, norm_h, norm_w, C])
            pad_w = W_full - norm_w
            pad_h = H_full - norm_h
            # Replicate-pad edges to restore full spatial dims.  The conv3d
            # kernel window overlaps into padding positions at the boundary,
            # so they must hold the replicated edge value (matching the
            # reference model's replicate padding) rather than zeros.
            if pad_w > 0:
                last_col = x_rm[:, :, norm_w - 1 : norm_w, :]
                x_rm = ttnn.concat([x_rm] + [last_col] * pad_w, dim=2)
            if pad_h > 0:
                last_row = x_rm[:, norm_h - 1 : norm_h, :, :]
                x_rm = ttnn.concat([x_rm] + [last_row] * pad_h, dim=1)
            _debug_stats(f"{norm_name}: after re-pad to ({H_full},{W_full})", x_rm)
        return ttnn.reshape(x_rm, [N, T, H_full, W_full, C])

    def forward(
        self, x_NTHWC: ttnn.Tensor, logical_h: int = 0, logical_w: int = 0, dump_dir: str | None = None
    ) -> ttnn.Tensor:
        shapes = self.get_tensor_shapes(x_NTHWC)
        N, T, H, W, C = shapes
        h_factor = self.parallel_config.h_parallel.factor
        w_factor = self.parallel_config.w_parallel.factor
        is_spatial_parallel = h_factor > 1 or w_factor > 1

        # Valid (unpadded) spatial dims for dump slicing
        valid_h = logical_h if logical_h > 0 else H * h_factor
        valid_w = logical_w if logical_w > 0 else W * w_factor
        valid_T = T * self.parallel_config.time_parallel.factor
        valid = (N, valid_T, valid_h, valid_w, C)

        self._dump_tensor("00_input", x_NTHWC, dump_dir, valid, is_gathered=False)

        if is_spatial_parallel:
            # Save residual from local data
            residual_tiled_NTHWC = ttnn.tilize_with_zero_padding(
                ttnn.reshape(x_NTHWC, [N * T, 1, H * W, C]),
                use_multicore=True,
            )
            # Reshape for separate H/W all-gathers (pass ROW_MAJOR;
            # _all_gather_hw flattens into tile-aligned dims for each gather)
            x_4d = ttnn.reshape(x_NTHWC, [N * T, H, W, C])
            ttnn.deallocate(x_NTHWC)
            x_4d = ttnn.reallocate(x_4d)  # copy to safe buffer before residual reallocate
            residual_tiled_NTHWC = ttnn.reallocate(residual_tiled_NTHWC)
            x_4d = self._all_gather_hw(x_4d, N, T, H, W, C)
            H_full = H * h_factor
            W_full = W * w_factor
            # GroupNorm + silu with padding removal to prevent zero contamination
            x_NTHWC = self._norm_silu_spatial(
                x_4d, self.norm1, N, T, H_full, W_full, C, logical_h, logical_w, dump_dir=dump_dir
            )
            self._dump_tensor("01_post_norm1_silu", x_NTHWC, dump_dir, valid, is_gathered=True)
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
                self.norm1, x_tiled_NTHWC, N * T, HW, num_out_blocks, dump_dir=dump_dir, dump_tag="01"
            )
            x_norm_tiled_NTHWC = ttnn.silu(x_norm_tiled_NTHWC, output_tensor=x_norm_tiled_NTHWC)
            x_NTHWC = self.untilize_reshape(x_norm_tiled_NTHWC, gathered_shapes)
            ttnn.deallocate(x_norm_tiled_NTHWC)
            # Dump from device 0 (safe: fresh norm output, not all-gather buffer)
            if dump_dir is not None:
                import os

                t = ttnn.to_torch(ttnn.get_device_tensors(x_NTHWC)[0]).float()
                torch.save(t, os.path.join(dump_dir, "01a_norm_silu_pre_repad.pt"))
                logger.info(
                    f"[DUMP] 01a_norm_silu_pre_repad (non-spatial): shape={list(t.shape)} mean={t.mean():.6f} std={t.std():.6f}"
                )
            self._dump_tensor("01_post_norm1_silu", x_NTHWC, dump_dir, valid, is_gathered=False)

        if is_spatial_parallel:
            x_NTHWC = self._partition_hw(x_NTHWC, N, T, H, W, C)

        x_conv1_NTHWC = self.conv1(x_NTHWC)
        ttnn.deallocate(x_NTHWC)

        self._dump_tensor("02_post_conv1", x_conv1_NTHWC, dump_dir, valid, is_gathered=False)

        if is_spatial_parallel:
            # All-gather conv1 output then GroupNorm + silu with padding removal
            x_conv1_4d = ttnn.reshape(x_conv1_NTHWC, [N * T, H, W, C])
            ttnn.deallocate(x_conv1_NTHWC)
            x_conv1_4d = ttnn.reallocate(x_conv1_4d)
            x_conv1_4d = self._all_gather_hw(x_conv1_4d, N, T, H, W, C)
            x_NTHWC = self._norm_silu_spatial(
                x_conv1_4d, self.norm2, N, T, H_full, W_full, C, logical_h, logical_w, dump_dir=dump_dir
            )
            self._dump_tensor("03_post_norm2_silu", x_NTHWC, dump_dir, valid, is_gathered=True)
        else:
            x_conv1_tiled_NTHWC = self.reshape_tilize(x_conv1_NTHWC, shapes)
            ttnn.deallocate(x_conv1_NTHWC)
            HW = x_conv1_tiled_NTHWC.shape[2]
            num_out_blocks = self.num_out_blocks_map.get(C, {}).get(
                HW, (HW + self._max_hw_per_block.get(C, 1000) - 1) // self._max_hw_per_block.get(C, 1000)
            )
            x_tiled_NTHWC = self._run_norm(
                self.norm2, x_conv1_tiled_NTHWC, N * T, HW, num_out_blocks, dump_dir=dump_dir, dump_tag="03"
            )
            ttnn.deallocate(x_conv1_tiled_NTHWC)
            x_tiled_NTHWC = ttnn.silu(x_tiled_NTHWC, output_tensor=x_tiled_NTHWC)
            x_NTHWC = self.untilize_reshape(x_tiled_NTHWC, gathered_shapes)
            ttnn.deallocate(x_tiled_NTHWC)
            if dump_dir is not None:
                import os

                t = ttnn.to_torch(ttnn.get_device_tensors(x_NTHWC)[0]).float()
                torch.save(t, os.path.join(dump_dir, "03a_norm_silu_pre_repad.pt"))
                logger.info(
                    f"[DUMP] 03a_norm_silu_pre_repad (non-spatial): shape={list(t.shape)} mean={t.mean():.6f} std={t.std():.6f}"
                )
            self._dump_tensor("03_post_norm2_silu", x_NTHWC, dump_dir, valid, is_gathered=False)

        if is_spatial_parallel:
            x_NTHWC = self._partition_hw(x_NTHWC, N, T, H, W, C)

        x_conv2_NTHWC = self.conv2(x_NTHWC)
        ttnn.deallocate(x_NTHWC)
        x_conv2_tiled_NTHWC = self.reshape_tilize(x_conv2_NTHWC, shapes)
        ttnn.deallocate(x_conv2_NTHWC)

        x_tiled_NTHWC = ttnn.add(x_conv2_tiled_NTHWC, residual_tiled_NTHWC)

        x_NTHWC = self.untilize_reshape(x_tiled_NTHWC, shapes)
        ttnn.deallocate(x_conv2_tiled_NTHWC)
        ttnn.deallocate(residual_tiled_NTHWC)
        ttnn.deallocate(x_tiled_NTHWC)

        self._dump_tensor("04_output", x_NTHWC, dump_dir, valid, is_gathered=False)

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
        for i, block in enumerate(self.resnets):
            x_NTHWC = block(x_NTHWC, logical_h, logical_w)
            _debug_stats(f"upsample: post-resblock-{i}", x_NTHWC)
        x_NTHWO = self.proj(x_NTHWC)
        _debug_stats("upsample: post-proj", x_NTHWO)
        x_NTHWC = self.depth_to_spacetime(x_NTHWO)
        ttnn.deallocate(x_NTHWO)
        _debug_stats("upsample: post-d2s (pre-reallocate)", x_NTHWC)
        # DEBUG: Force contiguous allocation after d2s to test if
        # permute→reshape left data non-contiguous
        x_NTHWC = ttnn.reallocate(x_NTHWC)
        _debug_stats("upsample: post-d2s (post-reallocate)", x_NTHWC)
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

        # Convert TT output to torch tensor
        x_NTHWC_torch = ttnn.to_torch(
            tt_x_NTHWC,
            mesh_composer=ttnn.ConcatMesh2dToTensor(
                self.mesh_device, mesh_shape=tuple(self.mesh_device.shape), dims=dims
            ),
        )
        ttnn.deallocate(tt_x_NTHWC)

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
