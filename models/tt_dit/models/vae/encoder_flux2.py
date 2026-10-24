# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""FLUX.2 KL autoencoder ENCODER on device (the FLUX.2 Klein reference-image path): the mirror of
``vae_flux2.py::Flux2VaeDecoder``, built from the same ``vae.py`` blocks, plus the strided downsample the decoder
has no use for.

Reference (BFL ``autoencoder.py`` ``Encoder``, diffusers ``AutoencoderKLFlux2.encoder``): ``conv_in`` 3 -> 128 (3x3,
pad 1); four levels of two ``ResnetBlock`` s with channels (128, 256, 512, 512) and a ``Downsample`` after the first
three (``conv(F.pad(x, (0, 1, 0, 1)))``, 3x3 stride 2 pad 0: output ``floor(H / 2) x floor(W / 2)``, i.e.
``floor((H + 1 - 3) / 2) + 1``; an odd extent loses its last row: 35 -> 17, measured on torch and the device); the
mid block (resnet, attention, resnet); ``quant_conv(conv_out(swish(norm_out(h))))`` with ``conv_out`` 512 -> 64 and
the 1x1 ``quant_conv`` 64 -> 64. ``AutoEncoder.encode`` keeps the first 32 output channels (the mean half; diffusers
``sample_mode="argmax"``, ``pipeline_flux2_klein.py:468``), patchifies 2x2 to 128 token channels (``c * 4 + py * 2 +
px``, the exact inverse of ``Flux2VaeDecoder.preprocess_and_unpatchify``) and normalizes with the
``BatchNorm2d(128, eps=1e-4, affine=False)`` running statistics (``pipeline_flux2_klein.py:470-477``).

Layout: activations are ``[N, H, W, C]`` as in the decoder; channels are tensor-parallel over ``tp_axis`` (every
GroupNorm then runs the plain ``layers/normalization.py::GroupNorm`` path); the spatial axes are NOT sharded in this
version (``h_factor`` / ``w_factor`` must be 1): a stride-2 convolution under an H split needs an even per-device row
count at every level, which only holds for ``H % 64 == 0``. The image enters as ``[1, H, W, 3]`` bf16 ROW_MAJOR
replicated. The BatchNorm normalize runs in fp32 on device (the conv output typecast to fp32, ``(z - mean) * inv_std``
with ``inv_std = 1 / sqrt(var + eps)`` computed on the host in fp32 at load time), then the tokens are cast to bf16.
Precision elsewhere is the decoder's: bf16 weights and activations, HiFi4 fp32-accumulated convolutions and
GroupNorms on Blackhole, the mid-block attention as ``VaeAttention`` runs it.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import torch

import ttnn

from ...layers.conv2d import Conv2d
from ...layers.linear import Linear
from ...layers.module import Module, ModuleList, Parameter
from ...parallel.config import Flux2VaeParallelConfig
from ...parallel.manager import CCLManager
from ...utils.substate import pop_substate, rename_substate
from ...utils.tracing import traced_function
from .vae import VaeContext, VaeConv2d, VaeMidBlock, VaeNormDesc, VaeNormDescGroup, VaeResnetBlock, _norm

if TYPE_CHECKING:
    from collections.abc import Sequence


class VaeDownsample2d(Module):
    """``Downsample`` (reference ``autoencoder.py:34-40``): ``conv(F.pad(x, (0, 1, 0, 1)))`` with a 3x3 stride-2 pad-0
    convolution == ``ttnn.conv2d`` with ``stride=2`` and the asymmetric padding ``[top 0, bottom 1, left 0, right 1]``
    (``ttnn/operations/conv2d.py``: a 4-tuple padding is ``[pad_top, pad_bottom, pad_left, pad_right]``). Output
    ``floor(H / 2) x floor(W / 2)``. Channels stay TP over ``ctx.tp_axis`` (input-channel sharded, reduce-scattered
    output, as every ``in == out`` conv of the shipped resnets). No spatial sharding (module docstring).

    DRAM slicing: ``Conv2d`` keys its slice table by ``(h, w, cin, cout)`` without the stride, so this conv would take
    the stride-1 resnet's count for the same input shape, and ttnn allows at most ``ceil(output_width / 32)`` slices
    for a TILE output (``op_slicing.cpp::compute_max_num_slices``; the stride-2 output is half as wide, so the shared
    count can exceed it: ``TT_FATAL ... Number of slices (8) exceeds the maximum allowed (4)`` at the level-2
    downsample of 1024x1024). ``forward`` therefore gives the conv an INSTANCE slice table (shadowing the class-level
    ``Conv2d.slice_params``) with that key capped at the bound: ``slice_count(table_count, output_width)``. Fewer
    slices only change the DRAM slicing of the same convolution (each slice is a full-precision conv over a column
    band; fp32 accumulation per output element is unchanged)."""

    def __init__(self, *, channels: int, ctx: VaeContext) -> None:
        super().__init__()
        if ctx.h_factor > 1 or ctx.w_factor > 1:
            raise NotImplementedError("spatial sharding of the strided VAE downsample is not implemented")
        self.conv = Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=2,
            padding=(0, 1, 0, 1),
            mesh_device=ctx.device,
            in_mesh_axis=ctx.tp_axis,
            out_mesh_axis=None,
            ccl_manager=ctx.ccl_manager,
        )
        self._slice_overrides: dict[tuple, int] = {}

    @staticmethod
    def output_extent(extent: int) -> int:
        """``floor((extent + 1 - 3) / 2) + 1`` = ``floor(extent / 2)`` (pad 0 + 1, kernel 3, stride 2)."""
        return (int(extent) + 1 - 3) // 2 + 1

    @staticmethod
    def slice_count(table_count: int, output_width: int) -> int:
        """The stride-1 table count of the shared key capped at ttnn's ``ceil(output_width / 32)``."""
        return min(int(table_count), max(1, -(-int(output_width) // 32)))

    def _cap_slices(self, h: int, w: int) -> None:
        conv = self.conv
        mesh = tuple(conv.mesh_device.shape)
        key = (h, w, conv.in_channels, conv.out_channels)
        if key in self._slice_overrides:
            return
        table = Conv2d.slice_params.get(mesh, Conv2d.slice_default)
        if key not in table:
            self._slice_overrides[key] = 0  # no entry: the conv runs unsliced, as Conv2d would
            return
        count = self.slice_count(table[key], self.output_extent(w))
        self._slice_overrides[key] = count
        instance_table = dict(conv.__dict__.get("slice_params", {}).get(mesh, {}))
        instance_table[key] = count
        conv.slice_params = {**conv.__dict__.get("slice_params", {}), mesh: instance_table}

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        _, h, w, _ = x.shape
        self._cap_slices(int(h), int(w))
        return self.conv.forward(x, use_persistent_buffer=False)


class VaeDownBlock(Module):
    """diffusers ``DownEncoderBlock2D``: ``num_layers`` resnets then an optional downsampler."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        downsample: bool,
        norm: VaeNormDesc,
        ctx: VaeContext,
    ) -> None:
        super().__init__()
        self.resnets = ModuleList(
            VaeResnetBlock(
                in_channels=in_channels if i == 0 else out_channels, out_channels=out_channels, norm=norm, ctx=ctx
            )
            for i in range(num_layers)
        )
        self.downsampler = VaeDownsample2d(channels=out_channels, ctx=ctx) if downsample else None

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        rename_substate(state, "downsamplers.0", "downsampler")

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        for block in self.resnets:
            x = block.forward(x)
        if self.downsampler is not None:
            x = self.downsampler.forward(x)
        return x


class Flux2VaeEncoder(Module):
    """Module docstring. ``encode`` is the whole of ``AutoEncoder.encode``: image ``[1, H, W, 3]`` -> tokens
    ``[1, (H/16) * (W/16), 128]`` bf16 in ``latent_ids`` (row-major) order."""

    _PATCH_SIZE = 2

    def __init__(
        self,
        *,
        in_channels: int = 3,
        block_out_channels: Sequence[int] = (128, 256, 512, 512),
        layers_per_block: int = 2,
        z_channels: int = 32,
        parallel_config: Flux2VaeParallelConfig,
        device: ttnn.MeshDevice,
        ccl_manager: CCLManager | None,
        use_conv3d: bool = False,
    ) -> None:
        super().__init__()

        ctx = VaeContext(
            tp_axis=parallel_config.tp_parallel.mesh_axis if parallel_config.tp_parallel is not None else None,
            device=device,
            ccl_manager=ccl_manager,
            h_mesh_axis=parallel_config.h_parallel.mesh_axis if parallel_config.h_parallel is not None else None,
            h_factor=parallel_config.h_parallel.factor if parallel_config.h_parallel is not None else 1,
            w_mesh_axis=parallel_config.w_parallel.mesh_axis if parallel_config.w_parallel is not None else None,
            w_factor=parallel_config.w_parallel.factor if parallel_config.w_parallel is not None else 1,
            use_conv3d=use_conv3d,
        )
        if ctx.tp_axis is not None and ctx.ccl_manager is None:
            msg = "ccl_manager must be provided if tensor parallelism is used"
            raise ValueError(msg)
        if ctx.h_factor > 1 or ctx.w_factor > 1:
            raise NotImplementedError("Flux2VaeEncoder shards channels only; spatial sharding is not implemented")

        channel_counts = [block_out_channels[0], *block_out_channels]
        eps = 1e-6
        norm = VaeNormDescGroup(num_groups=32, eps=eps)

        self.conv_in = VaeConv2d(in_channels, channel_counts[0], kernel_size=3, padding=1, ctx=ctx)
        self.down_blocks = ModuleList(
            VaeDownBlock(
                in_channels=ch_in,
                out_channels=ch_out,
                num_layers=layers_per_block,
                downsample=i != len(channel_counts) - 2,
                norm=norm,
                ctx=ctx,
            )
            for i, (ch_in, ch_out) in enumerate(itertools.pairwise(channel_counts))
        )
        self.mid_block = VaeMidBlock(num_channels=channel_counts[-1], norm=norm, ctx=ctx)
        self.conv_norm_out = _norm(norm, num_channels=channel_counts[-1], ctx=ctx, activation_fn="silu")
        # 64 output channels: 16 per device at TP4 is not a tile, so the last conv runs replicated on the gathered
        # 512-channel input (the decoder's conv_out is the mirror).
        self.conv_out = VaeConv2d(
            channel_counts[-1], 2 * z_channels, kernel_size=3, padding=1, tensor_parallel=False, ctx=ctx
        )
        self.quant_conv = Linear(2 * z_channels, 2 * z_channels, mesh_device=ctx.device)

        bn_size = self._PATCH_SIZE**2 * z_channels
        self.bn_running_mean = Parameter(total_shape=[bn_size], device=ctx.device, dtype=ttnn.float32)
        self.bn_inv_std = Parameter(total_shape=[bn_size], device=ctx.device, dtype=ttnn.float32)
        self.bn_eps = 1e-4

        self._z_channels = z_channels
        self._ctx = ctx

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        # the decoder half and the training counter are not ours
        pop_substate(state, "decoder")
        pop_substate(state, "post_quant_conv")
        state.pop("bn.num_batches_tracked", None)

        if "quant_conv.weight" in state:
            state["quant_conv.weight"] = state["quant_conv.weight"].squeeze(2, 3)

        if "bn.running_mean" in state:
            state["bn_running_mean"] = state.pop("bn.running_mean").float()
        if "bn.running_var" in state:
            # BatchNorm2d eval: (z - mean) / sqrt(var + eps), fp32 (autoencoder.py:155, 165)
            state["bn_inv_std"] = 1.0 / torch.sqrt(state.pop("bn.running_var").float() + self.bn_eps)

        rename_substate(state, "encoder", "")

    @traced_function(device=lambda self: self._ctx.device, clone_prep_inputs=False)
    def forward(self, x: ttnn.Tensor, /) -> ttnn.Tensor:
        """``[1, H, W, 3]`` bf16 replicated -> ``[1, H/8, W/8, 2 * z_channels]`` bf16 replicated (mean | logvar halves,
        before the patchify)."""
        x = self.conv_in.forward(x)
        for block in self.down_blocks:
            x = block.forward(x)
        x = self.mid_block.forward(x)
        x = self.conv_norm_out.forward(x)
        if self._ctx.ccl_manager is not None and self._ctx.tp_axis is not None:
            x = self._ctx.ccl_manager.all_gather(x, dim=-1, mesh_axis=self._ctx.tp_axis, use_hyperparams=True)
        x = self.conv_out.forward(x)
        return self.quant_conv.forward(x)

    def normalize_and_patchify(self, out: ttnn.Tensor, *, height: int, width: int) -> ttnn.Tensor:
        """``AutoEncoder.encode`` after the encoder (autoencoder.py:159-166): the first ``z_channels`` channels, the 2x2
        patchify to ``[1, (H/16) * (W/16), 4 * z_channels]`` with token channel ``c * 4 + py * 2 + px``, the BatchNorm
        normalize in fp32, the bf16 cast. ``height`` / ``width`` are the IMAGE size."""
        p = self._PATCH_SIZE
        scale = 8  # the encoder's spatial compression before the patchify
        h8, w8 = height // scale, width // scale
        assert tuple(out.shape) == (1, h8, w8, 2 * self._z_channels), tuple(out.shape)
        assert h8 % p == 0 and w8 % p == 0, f"{height}x{width}: the latent {h8}x{w8} is not a multiple of {p}"
        mean = ttnn.slice(out, [0, 0, 0, 0], [1, h8, w8, self._z_channels])
        z = ttnn.to_layout(mean, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(mean)
        z = z.reshape([1, h8 // p, p, w8 // p, p, self._z_channels])
        z = ttnn.permute(z, [0, 1, 3, 5, 2, 4])  # [1, h, w, C, py, px]
        z = z.reshape([1, (h8 // p) * (w8 // p), self._z_channels * p * p])
        z = ttnn.to_layout(z, ttnn.TILE_LAYOUT)
        z32 = ttnn.typecast(z, ttnn.float32)
        ttnn.deallocate(z)
        centred = ttnn.subtract(z32, self.bn_running_mean.data)
        ttnn.deallocate(z32)
        normalized = ttnn.multiply(centred, self.bn_inv_std.data)
        ttnn.deallocate(centred)
        tokens = ttnn.typecast(normalized, ttnn.bfloat16)
        ttnn.deallocate(normalized)
        return tokens

    def encode(self, image: ttnn.Tensor, *, height: int, width: int, traced: bool = False) -> ttnn.Tensor:
        """``[1, H, W, 3]`` bf16 replicated -> ``[1, (H/16) * (W/16), 128]`` bf16 replicated tokens."""
        out = self.forward(image, traced=traced)
        tokens = self.normalize_and_patchify(out, height=height, width=width)
        if not traced:
            ttnn.deallocate(out)
        return tokens


__all__ = ["Flux2VaeEncoder", "VaeDownBlock", "VaeDownsample2d"]
