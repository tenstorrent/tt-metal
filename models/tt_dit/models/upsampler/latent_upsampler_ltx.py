# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""On-device port of ``ltx_core.model.upsampler.LatentUpsampler`` (spatial 2x only)."""

from __future__ import annotations

import json
import os

import torch
from loguru import logger
from safetensors import safe_open
from safetensors.torch import load_file

import ttnn

from ...layers.module import Module, ModuleList
from ...layers.normalization import GroupNorm3D
from ...parallel.config import DiTParallelConfig, VaeHWParallelConfig
from ...parallel.manager import CCLManager
from ...utils import cache as cache_module
from ...utils.conv3d import ConvDims, conv3d_blocking_hash, conv_pad_height, conv_pad_width
from ...utils.tensor import fast_device_to_host, typed_tensor_2dshard
from ..vae.vae_ltx import LTXCausalConv3d


def _all_gather_hw(x: ttnn.Tensor, pc: VaeHWParallelConfig, ccl: CCLManager) -> ttnn.Tensor:
    if pc.height_parallel.factor > 1:
        x = ccl.all_gather(x, dim=2, mesh_axis=pc.height_parallel.mesh_axis, use_hyperparams=False)
    if pc.width_parallel.factor > 1:
        x = ccl.all_gather(x, dim=3, mesh_axis=pc.width_parallel.mesh_axis, use_hyperparams=False)
    return x


def _mesh_partition_hw(x: ttnn.Tensor, pc: VaeHWParallelConfig) -> ttnn.Tensor:
    if pc.height_parallel.factor > 1:
        x = ttnn.mesh_partition(x, dim=2, cluster_axis=pc.height_parallel.mesh_axis)
    if pc.width_parallel.factor > 1:
        x = ttnn.mesh_partition(x, dim=3, cluster_axis=pc.width_parallel.mesh_axis)
    return x


def _gn_hw_sharded(
    gn: GroupNorm3D,
    x: ttnn.Tensor,
    pc: VaeHWParallelConfig,
    ccl: CCLManager,
    logical_h: int,
    logical_w: int,
) -> ttnn.Tensor:
    """All-gather H/W, run GroupNorm on the full extent (excluding mesh-factor pad
    rows/cols), then mesh_partition back. ``logical_h/logical_w<=0`` means full extent."""
    if x.layout != ttnn.ROW_MAJOR_LAYOUT:
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    x = _all_gather_hw(x, pc, ccl)
    B, T, padded_h, padded_w, C = x.shape
    lh = logical_h if logical_h > 0 else padded_h
    lw = logical_w if logical_w > 0 else padded_w
    cropped = lh < padded_h or lw < padded_w
    if cropped:
        # GroupNorm is configured for the logical extent, so it must run on the
        # cropped tensor (slice needs ROW_MAJOR — x already is from above).
        x = x[:, :, :lh, :lw, :]
    x = gn(x)
    if not cropped:
        return _mesh_partition_hw(x, pc)
    # Re-zero-pad the mesh-factor rows/cols so mesh_partition divides evenly.
    # ttnn.pad only pads the trailing dims of a ROW_MAJOR tensor, so reshape to 4D
    # to pad H and W. Partition stays in ROW_MAJOR: a sub-tile-wide W shard (e.g.
    # 32→8) can't be sliced from a tilized cropped tensor, so tilizing first then
    # partitioning fails. Restore TILE last so the return matches the non-cropped path.
    x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    x = ttnn.reshape(x, (B * T, lh, lw, C))
    if lw < padded_w:
        x = ttnn.pad(x, [(0, 0), (0, 0), (0, padded_w - lw), (0, 0)], value=0.0)
    if lh < padded_h:
        x = ttnn.pad(x, [(0, 0), (0, padded_h - lh), (0, 0), (0, 0)], value=0.0)
    x = ttnn.reshape(x, (B, T, padded_h, padded_w, C))
    x = _mesh_partition_hw(x, pc)
    return ttnn.to_layout(x, ttnn.TILE_LAYOUT)


def _depth_to_space_bthwc_2x(x: ttnn.Tensor) -> ttnn.Tensor:
    """``PixelShuffleND(dims=2)`` on a ``(B, T, H, W, 4C)`` BTHWC tensor.

    Channel order is (1,2,2,C); the feeding conv reorders via depth_to_space_stride=(1,2,2).
    """
    B, T, H, W, total_c = x.shape
    C = total_c // 4
    x = ttnn.reshape(x, (B, T, H, W, 1, 2, 2, C))
    x = ttnn.permute(x, (0, 1, 4, 2, 5, 3, 6, 7))
    return ttnn.reshape(x, (B, T, H * 2, W * 2, C))


class LTXUpsamplerResBlock(Module):
    """Mirrors ``ltx_core.model.upsampler.res_block.ResBlock`` (dims=3, zero temporal pad)."""

    def __init__(
        self,
        channels: int,
        mid_channels: int | None = None,
        *,
        gn_input_nhw: int,
        gn_num_batches: int = 1,
        mesh_device: ttnn.MeshDevice,
        parallel_config: VaeHWParallelConfig,
        ccl_manager: CCLManager,
        dtype: ttnn.DataType = ttnn.bfloat16,
        temporal_padding_mode: str = "zeros",
        conv_dims: ConvDims | None = None,
    ) -> None:
        super().__init__()
        if mid_channels is None:
            mid_channels = channels
        self.parallel_config = parallel_config
        self.ccl_manager = ccl_manager
        self.mesh_device = mesh_device

        conv_kwargs = dict(
            kernel_size=3,
            stride=1,
            mesh_device=mesh_device,
            parallel_config=parallel_config,
            ccl_manager=ccl_manager,
            dtype=dtype,
            temporal_padding_mode=temporal_padding_mode,
            conv_dims=conv_dims,
        )
        gn_kwargs = dict(
            num_groups=32,
            input_nhw=gn_input_nhw,
            num_batches=gn_num_batches,
            mesh_device=mesh_device,
            dtype=dtype,
        )
        self.conv1 = LTXCausalConv3d(channels, mid_channels, **conv_kwargs)
        self.norm1 = GroupNorm3D(num_channels=mid_channels, **gn_kwargs)
        self.conv2 = LTXCausalConv3d(mid_channels, channels, **conv_kwargs)
        self.norm2 = GroupNorm3D(num_channels=channels, **gn_kwargs)

    def forward(self, x: ttnn.Tensor, logical_h: int = 0, logical_w: int = 0) -> ttnn.Tensor:
        pc, ccl = self.parallel_config, self.ccl_manager
        residual = x
        x = self.conv1(x, causal=False, logical_h=logical_h, logical_w=logical_w)
        x = _gn_hw_sharded(self.norm1, x, pc, ccl, logical_h, logical_w)
        x = ttnn.to_layout(ttnn.silu(x), ttnn.ROW_MAJOR_LAYOUT)
        x = self.conv2(x, causal=False, logical_h=logical_h, logical_w=logical_w)
        x = _gn_hw_sharded(self.norm2, x, pc, ccl, logical_h, logical_w)
        if residual.layout != x.layout:
            residual = ttnn.to_layout(residual, x.layout)
        return ttnn.silu(ttnn.add(x, residual))


def compute_upsampler_dims(
    *,
    input_hw: tuple[int, int],
    num_frames: int,
    h_factor: int = 1,
    w_factor: int = 1,
) -> tuple[ConvDims, ConvDims, ConvDims]:
    """``(pre_dims, ups_dims, post_dims)`` for the upsampler's three conv shape
    classes. T = cur_T + (kT-1); H/W are per-device shards of mesh-factor-padded
    spatial."""
    H_in, W_in = input_hw
    padded_h = ((H_in + h_factor - 1) // h_factor) * h_factor
    padded_w = ((W_in + w_factor - 1) // w_factor) * w_factor
    H_dev = padded_h // h_factor
    W_dev = padded_w // w_factor
    pre = ConvDims(T=num_frames + 2, H=H_dev, W=W_dev)
    ups = ConvDims(T=num_frames, H=H_dev, W=W_dev)
    post = ConvDims(T=num_frames + 2, H=H_dev * 2, W=W_dev * 2)
    return pre, ups, post


class LTXLatentUpsampler(Module):
    """Mirrors ``ltx_core.model.upsampler.model.LatentUpsampler`` (spatial 2x, dims=3)."""

    def __init__(
        self,
        *,
        input_hw: tuple[int, int],
        in_channels: int = 128,
        mid_channels: int = 512,
        num_blocks_per_stage: int = 4,
        spatial_upsample: bool = True,
        temporal_upsample: bool = False,
        spatial_scale: float = 2.0,
        rational_resampler: bool = False,
        mesh_device: ttnn.MeshDevice,
        parallel_config: VaeHWParallelConfig,
        ccl_manager: CCLManager,
        dtype: ttnn.DataType = ttnn.bfloat16,
        num_frames: int | None = None,
    ) -> None:
        super().__init__()
        if not spatial_upsample or temporal_upsample or rational_resampler or float(spatial_scale) != 2.0:
            raise NotImplementedError(
                "LTXLatentUpsampler only supports spatial_upsample=True, temporal_upsample=False, "
                "rational_resampler=False, spatial_scale=2.0."
            )

        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.num_blocks_per_stage = num_blocks_per_stage
        self.mesh_device = mesh_device
        self.parallel_config = parallel_config
        self.ccl_manager = ccl_manager

        H_in, W_in = input_hw
        H_out, W_out = H_in * 2, W_in * 2

        # DRAM GroupNorm grid is pinned from T*H*W at construction.
        assert num_frames is not None, "LTXLatentUpsampler requires a concrete num_frames"
        pre_gn_nhw = num_frames * H_in * W_in
        post_gn_nhw = num_frames * H_out * W_out

        pre_dims, ups_dims, post_dims = compute_upsampler_dims(
            input_hw=input_hw,
            num_frames=num_frames,
            h_factor=parallel_config.height_parallel.factor,
            w_factor=parallel_config.width_parallel.factor,
        )

        block_kwargs = dict(
            mesh_device=mesh_device,
            parallel_config=parallel_config,
            ccl_manager=ccl_manager,
            dtype=dtype,
            temporal_padding_mode="zeros",
        )

        self.initial_conv = LTXCausalConv3d(
            in_channels, mid_channels, kernel_size=3, stride=1, conv_dims=pre_dims, **block_kwargs
        )
        self.initial_norm = GroupNorm3D(
            num_channels=mid_channels, num_groups=32, input_nhw=pre_gn_nhw, mesh_device=mesh_device, dtype=dtype
        )

        self.res_blocks = ModuleList(
            [
                LTXUpsamplerResBlock(mid_channels, gn_input_nhw=pre_gn_nhw, conv_dims=pre_dims, **block_kwargs)
                for _ in range(num_blocks_per_stage)
            ]
        )

        self.upsampler = LTXCausalConv3d(
            mid_channels,
            4 * mid_channels,
            kernel_size=(1, 3, 3),
            stride=1,
            conv_dims=ups_dims,
            depth_to_space_stride=(1, 2, 2),
            **block_kwargs,
        )

        self.post_upsample_res_blocks = ModuleList(
            [
                LTXUpsamplerResBlock(mid_channels, gn_input_nhw=post_gn_nhw, conv_dims=post_dims, **block_kwargs)
                for _ in range(num_blocks_per_stage)
            ]
        )

        self.final_conv = LTXCausalConv3d(
            mid_channels, in_channels, kernel_size=3, stride=1, conv_dims=post_dims, **block_kwargs
        )

        # Set by ``from_checkpoint`` for ``reload_weights`` (the disk-cache path).
        self._checkpoint_path: str | None = None
        self._dit_parallel_config: DiTParallelConfig | None = None

    @classmethod
    def from_checkpoint(
        cls,
        upsampler_path: str,
        *,
        input_hw: tuple[int, int],
        latent_frames: int,
        mesh_device: ttnn.MeshDevice,
        parallel_config: VaeHWParallelConfig,
        ccl_manager: CCLManager,
        dit_parallel_config: DiTParallelConfig,
    ) -> "LTXLatentUpsampler":
        """Build a latent upsampler from a checkpoint's JSON metadata config.

        Stage-1 shape math (``input_hw`` / ``latent_frames``) is computed by the caller and passed
        in — the model never reads pipeline state. ``dit_parallel_config`` is retained for the cache
        key used by ``reload_weights``.
        """
        with safe_open(upsampler_path, framework="pt") as f:
            cfg = json.loads(f.metadata()["config"])
        ups = cls(
            input_hw=input_hw,
            in_channels=cfg["in_channels"],
            mid_channels=cfg["mid_channels"],
            num_blocks_per_stage=cfg["num_blocks_per_stage"],
            spatial_upsample=cfg["spatial_upsample"],
            temporal_upsample=cfg["temporal_upsample"],
            spatial_scale=cfg["spatial_scale"],
            rational_resampler=cfg["rational_resampler"],
            mesh_device=mesh_device,
            parallel_config=parallel_config,
            ccl_manager=ccl_manager,
            num_frames=latent_frames,
        )
        ups._checkpoint_path = upsampler_path
        ups._dit_parallel_config = dit_parallel_config
        return ups

    def reload_weights(self) -> None:
        """Push upsampler weights onto the mesh via the disk cache. Blocking-hash subfolder
        invalidates the cache when conv3d ``C_in_block`` changes. Idempotent (no-op if loaded)."""
        if self.is_loaded():
            return
        assert self._checkpoint_path is not None, "reload_weights requires construction via from_checkpoint"

        def _state_provider() -> dict[str, torch.Tensor]:
            logger.info(f"Upsampler cache miss — loading safetensors: {self._checkpoint_path}")
            return load_file(self._checkpoint_path)

        blocking_key = conv3d_blocking_hash(self)
        subfolder = f"upsampler_{blocking_key}" if blocking_key else "upsampler"
        cache_module.load_model(
            self,
            model_name=os.path.basename(self._checkpoint_path).removesuffix(".safetensors"),
            subfolder=subfolder,
            parallel_config=self._dit_parallel_config,
            mesh_shape=tuple(self.mesh_device.shape),
            mesh_device=self.mesh_device,
            get_torch_state_dict=_state_provider,
        )
        logger.info("Loaded TTNN latent upsampler")

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        """Reference upsampler is ``Sequential[Conv2d, PixelShuffleND]``; flatten to Conv3d."""
        if "upsampler.0.weight" in state:
            state["upsampler.weight"] = state.pop("upsampler.0.weight").unsqueeze(2)
        if "upsampler.0.bias" in state:
            state["upsampler.bias"] = state.pop("upsampler.0.bias")
        for k in [k for k in state if k.startswith("upsampler.") and k not in ("upsampler.weight", "upsampler.bias")]:
            del state[k]

    def _encode_input(self, latent_BCFHW: torch.Tensor) -> tuple[ttnn.Tensor, int, int]:
        """BCFHW → mesh-sharded BTHWC, H/W zero-padded up to mesh factors. Returns pre-pad
        ``(logical_h, logical_w)`` for conv masks and norm crops."""
        latent = latent_BCFHW.permute(0, 2, 3, 4, 1).contiguous()
        latent, logical_h = conv_pad_height(latent, self.parallel_config.height_parallel.factor)
        latent, logical_w = conv_pad_width(latent, self.parallel_config.width_parallel.factor)
        x = typed_tensor_2dshard(
            latent,
            self.mesh_device,
            shard_mapping={
                self.parallel_config.height_parallel.mesh_axis: 2,
                self.parallel_config.width_parallel.mesh_axis: 3,
            },
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
        )
        return x, logical_h, logical_w

    def _decode_output(self, x_BTHWC: ttnn.Tensor, logical_h: int, logical_w: int) -> torch.Tensor:
        concat_dims = [None, None]
        concat_dims[self.parallel_config.height_parallel.mesh_axis] = 2
        concat_dims[self.parallel_config.width_parallel.mesh_axis] = 3
        result = fast_device_to_host(x_BTHWC, self.mesh_device, concat_dims, ccl_manager=self.ccl_manager)
        result = result[:, :, :logical_h, :logical_w, :]
        return result.permute(0, 4, 1, 2, 3).contiguous()

    def forward(self, latent_BCFHW: torch.Tensor) -> torch.Tensor:
        x, logical_h, logical_w = self._encode_input(latent_BCFHW)
        pc, ccl = self.parallel_config, self.ccl_manager

        x = self.initial_conv(x, causal=False, logical_h=logical_h, logical_w=logical_w)
        x = _gn_hw_sharded(self.initial_norm, x, pc, ccl, logical_h, logical_w)
        x = ttnn.to_layout(ttnn.silu(x), ttnn.ROW_MAJOR_LAYOUT)

        for block in self.res_blocks:
            x = ttnn.to_layout(block(x, logical_h, logical_w), ttnn.ROW_MAJOR_LAYOUT)

        x = self.upsampler(x, causal=False, logical_h=logical_h, logical_w=logical_w)
        x = _depth_to_space_bthwc_2x(x)
        logical_h, logical_w = logical_h * 2, logical_w * 2

        for block in self.post_upsample_res_blocks:
            x = ttnn.to_layout(block(x, logical_h, logical_w), ttnn.ROW_MAJOR_LAYOUT)

        x = self.final_conv(x, causal=False, logical_h=logical_h, logical_w=logical_w)

        return self._decode_output(x, logical_h, logical_w)
