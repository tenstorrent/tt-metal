# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""LTX-2 audio vocoder (Stage B): BigVGAN-v2 with AMP1 blocks.

fp32 mandatory: bf16 accumulation degrades spectral metrics through the 108-conv
chain, so every Conv1d/Snake/anti-alias filter runs at ``dtype=ttnn.float32`` (HiFi4
+ ``fp32_dest_acc_en`` + ``packer_l1_acc``). Works on ``(B, C, T)`` torch, converted
to ``(B, T, C)`` ROW_MAJOR at the device boundary for ``Conv1dViaConv3d``.
"""

from __future__ import annotations

from typing import List, Sequence

import torch

import ttnn

from ...layers.audio_ops import (
    ConvTranspose1dViaConv3d,
    Snake,
    SnakeBeta,
    _AlignedOutConv1d,
    _all_gather_t,
    _partition_t,
    _set_tpad_tail,
    partition_channel,
)
from ...layers.audio_resample import Activation1d
from ...layers.module import Module, ModuleList
from ...parallel.config import ParallelFactor
from ...parallel.manager import CCLManager


class DilatedConv1d(_AlignedOutConv1d):
    """Dilated 1D conv: a symmetric ("same") zeros-pad ``Conv1dViaConv3d`` with
    ``dilation`` passed through to conv3d. For the AMP block's ``(k-1)*d`` (always
    even) the base's ``eff_k // 2`` halo equals the symmetric ``same_pad``."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        dilation: int = 1,
        bias: bool = True,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.float32,
        parallel_config: ParallelFactor | None = None,
        ccl_manager: CCLManager | None = None,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            dilation=dilation,
            padding_mode="zeros",
            bias=bias,
            mesh_device=mesh_device,
            dtype=dtype,
            parallel_config=parallel_config,
            ccl_manager=ccl_manager,
        )


class AMPBlock1(Module):
    """Three parallel residual branches with anti-aliased SnakeBeta activations."""

    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int = 3,
        dilation: Sequence[int] = (1, 3, 5),
        activation: str = "snakebeta",
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.float32,
        parallel_config: ParallelFactor | None = None,
        ccl_manager: CCLManager | None = None,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.num_branches = len(dilation)
        self.mesh_device = mesh_device

        act_cls = SnakeBeta if activation == "snakebeta" else Snake

        self.convs1 = ModuleList(
            [
                DilatedConv1d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    dilation=dilation[i],
                    bias=True,
                    mesh_device=mesh_device,
                    dtype=dtype,
                    parallel_config=parallel_config,
                    ccl_manager=ccl_manager,
                )
                for i in range(self.num_branches)
            ]
        )
        self.convs2 = ModuleList(
            [
                DilatedConv1d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    dilation=1,
                    bias=True,
                    mesh_device=mesh_device,
                    dtype=dtype,
                    parallel_config=parallel_config,
                    ccl_manager=ccl_manager,
                )
                for i in range(self.num_branches)
            ]
        )
        # alpha_logscale=True: the checkpoint stores log α / log β and
        # Snake/SnakeBeta collapses it at load time.
        self.acts1 = ModuleList(
            [
                Activation1d(
                    channels=channels,
                    activation=act_cls(
                        channels,
                        alpha_logscale=True,
                        mesh_device=mesh_device,
                        dtype=dtype,
                        parallel_config=parallel_config,
                    ),
                    mesh_device=mesh_device,
                    dtype=dtype,
                    parallel_config=parallel_config,
                    ccl_manager=ccl_manager,
                )
                for _ in range(self.num_branches)
            ]
        )
        self.acts2 = ModuleList(
            [
                Activation1d(
                    channels=channels,
                    activation=act_cls(
                        channels,
                        alpha_logscale=True,
                        mesh_device=mesh_device,
                        dtype=dtype,
                        parallel_config=parallel_config,
                    ),
                    mesh_device=mesh_device,
                    dtype=dtype,
                    parallel_config=parallel_config,
                    ccl_manager=ccl_manager,
                )
                for _ in range(self.num_branches)
            ]
        )

    def forward(self, x_BTC: ttnn.Tensor, set_tail=None) -> ttnn.Tensor:
        # set_tail(xd, mode) materializes the tile-align pad image to each op's boundary when
        # T-sharded (acts replicate the real boundary, convs zero it); identity when unsharded.
        st = set_tail if set_tail is not None else (lambda xd, mode: xd)

        def _apply(op, x, mode):
            xs = st(x, mode)
            y = op(xs)
            if xs is not x:
                ttnn.deallocate(xs)
            return y

        for i in range(self.num_branches):
            xt = _apply(self.acts1[i], x_BTC, "replicate")
            nxt = _apply(self.convs1[i], xt, "zeros")
            ttnn.deallocate(xt)
            xt = _apply(self.acts2[i], nxt, "replicate")
            ttnn.deallocate(nxt)
            nxt = _apply(self.convs2[i], xt, "zeros")
            ttnn.deallocate(xt)
            x_new = ttnn.add(x_BTC, nxt)
            ttnn.deallocate(nxt)
            if i > 0:
                ttnn.deallocate(x_BTC)
            x_BTC = x_new
        return x_BTC


class Vocoder(Module):
    """BigVGAN-v2 AMP1 vocoder for LTX-2 audio decode (Stage B).

    Maps mel ``(B, 2, T_frames, mel_bins)`` to a waveform
    ``(B, 2, T_frames * prod(upsample_rates))``. fp32 everywhere (see module
    docstring).
    """

    def __init__(
        self,
        *,
        resblock_kernel_sizes: List[int] | None = None,
        upsample_rates: List[int] | None = None,
        upsample_kernel_sizes: List[int] | None = None,
        resblock_dilation_sizes: List[List[int]] | None = None,
        upsample_initial_channel: int = 1536,
        resblock: str = "AMP1",
        activation: str = "snakebeta",
        use_tanh_at_final: bool = False,
        apply_final_activation: bool = True,
        use_bias_at_final: bool = False,
        in_channels: int = 128,
        out_channels: int = 2,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.float32,
        parallel_config: ParallelFactor | None = None,
        ccl_manager: CCLManager | None = None,
    ) -> None:
        super().__init__()

        if resblock_kernel_sizes is None:
            resblock_kernel_sizes = [3, 7, 11]
        if upsample_rates is None:
            upsample_rates = [5, 2, 2, 2, 2, 2]
        if upsample_kernel_sizes is None:
            upsample_kernel_sizes = [11, 4, 4, 4, 4, 4]
        if resblock_dilation_sizes is None:
            resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]

        if resblock != "AMP1":
            raise NotImplementedError(f"only AMP1 is supported, got {resblock!r}")

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.use_tanh_at_final = use_tanh_at_final
        self.apply_final_activation = apply_final_activation
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample_rates = list(upsample_rates)
        self.mesh_device = mesh_device
        self.dtype = dtype
        self.parallel_config = parallel_config
        self.ccl_manager = ccl_manager
        self._tpad_mask_cache: dict = {}

        self.conv_pre = _AlignedOutConv1d(
            in_channels=in_channels,
            out_channels=upsample_initial_channel,
            kernel_size=7,
            stride=1,
            padding_mode="zeros",
            bias=True,
            mesh_device=mesh_device,
            dtype=dtype,
            parallel_config=parallel_config,
            ccl_manager=ccl_manager,
        )

        self.ups = ModuleList(
            [
                ConvTranspose1dViaConv3d(
                    in_channels=upsample_initial_channel // (2**i),
                    out_channels=upsample_initial_channel // (2 ** (i + 1)),
                    kernel_size=upsample_kernel_sizes[i],
                    stride=upsample_rates[i],
                    bias=True,
                    mesh_device=mesh_device,
                    dtype=dtype,
                    parallel_config=parallel_config,
                    ccl_manager=ccl_manager,
                )
                for i in range(self.num_upsamples)
            ]
        )

        # 3 x num_upsamples AMP blocks, row-major over (stage, branch).
        self.resblocks = ModuleList()
        for i in range(self.num_upsamples):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for ks, ds in zip(resblock_kernel_sizes, resblock_dilation_sizes, strict=True):
                self.resblocks.append(
                    AMPBlock1(
                        channels=ch,
                        kernel_size=ks,
                        dilation=ds,
                        activation=activation,
                        mesh_device=mesh_device,
                        dtype=dtype,
                        parallel_config=parallel_config,
                        ccl_manager=ccl_manager,
                    )
                )

        final_channels = upsample_initial_channel // (2**self.num_upsamples)

        self.act_post = Activation1d(
            channels=final_channels,
            activation=SnakeBeta(
                final_channels,
                alpha_logscale=True,
                mesh_device=mesh_device,
                dtype=dtype,
                parallel_config=parallel_config,
            ),
            mesh_device=mesh_device,
            dtype=dtype,
            parallel_config=parallel_config,
            ccl_manager=ccl_manager,
        )

        self.conv_post = _AlignedOutConv1d(
            in_channels=final_channels,
            out_channels=out_channels,
            kernel_size=7,
            stride=1,
            padding_mode="zeros",
            bias=use_bias_at_final,
            mesh_device=mesh_device,
            dtype=dtype,
            parallel_config=parallel_config,
            ccl_manager=ccl_manager,
            # Final waveform has out_channels=2 — too small to channel-shard; keep
            # the output full so the trailing all-gather is a no-op.
            channel_shard_output=False,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        pass

    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """``mel_spec``: ``(B, 2, T_frames, mel_bins)`` stereo or
        ``(B, T_frames, mel_bins)`` mono → ``(B, out_channels, T_frames * prod(rates))``.
        """
        x_t = mel_spec.transpose(2, 3) if mel_spec.dim() == 4 else mel_spec.transpose(1, 2).unsqueeze(1)
        if x_t.dim() == 4:
            assert x_t.shape[1] == 2, f"stereo input must have 2 channels, got {x_t.shape[1]}"
            B, S, F, T = x_t.shape
            x_t = x_t.reshape(B, S * F, T)
        B, C, T = x_t.shape
        assert C == self.in_channels, f"expected {self.in_channels} input channels, got {C}"

        x_BTC_torch = x_t.transpose(1, 2).float().contiguous()

        sharded = self.parallel_config is not None and self.parallel_config.factor > 1
        # Pad T to a multiple of (TILE_HEIGHT * factor) so mesh_partition produces
        # tile-aligned per-chip shards. The extras propagate at upsampled length
        # and get cropped from the final waveform.
        t_pad = 0
        if sharded:
            factor = self.parallel_config.factor
            tile_h = 32
            align = tile_h * factor
            rem = x_BTC_torch.shape[1] % align
            if rem != 0:
                t_pad = align - rem
                x_BTC_torch = torch.nn.functional.pad(x_BTC_torch, (0, 0, 0, t_pad))

        x_dev = ttnn.from_torch(x_BTC_torch, device=self.mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=self.dtype)

        if sharded:
            x_dev = ttnn.to_layout(x_dev, ttnn.TILE_LAYOUT)
            x_dev = _partition_t(x_dev, self.parallel_config)
            x_dev = ttnn.to_layout(x_dev, ttnn.ROW_MAJOR_LAYOUT)

        # Channel-TP: split C across the channel axis up front so conv_pre's gather
        # reconstructs full C_in. (Gathering a channel-replicated tensor would
        # duplicate it.) conv_post leaves its output full, so no trailing gather.
        x_dev = partition_channel(x_dev, self.parallel_config, dim=2)

        def _set_tail(xd, cumrate, mode):
            # The tile-align pad image (t_pad*cumrate rows at the global tail) propagates as
            # signal. Each non-causal op's kept-boundary output must see what unsharded sees:
            # the gather-to-full upsamplers and zeros-pad convs want zeros, the replicate-pad
            # activations want the real last row. Materialize the pad image to the op's own
            # boundary right before it. No-op when unsharded (t_pad == 0).
            if t_pad == 0:
                return xd
            return _set_tpad_tail(
                xd,
                t_pad * cumrate,
                mode=mode,
                mesh_device=self.mesh_device,
                parallel_config=self.parallel_config,
                cache=self._tpad_mask_cache,
            )

        cumrate = 1
        x_dev = self.conv_pre(x_dev)

        for i in range(self.num_upsamples):
            x_dev = _set_tail(x_dev, cumrate, "zeros")  # ups gathers T to full and zero-pads internally
            x_dev = self.ups[i](x_dev)
            cumrate *= self.upsample_rates[i]
            stage_set_tail = (lambda c: (lambda xd, mode: _set_tail(xd, c, mode)))(cumrate)
            start = i * self.num_kernels
            # Mean over the num_kernels parallel AMP branches. Each block sets its own op
            # boundaries (acts replicate, convs zeros) via stage_set_tail.
            block_outputs = []
            for idx in range(start, start + self.num_kernels):
                block_outputs.append(self.resblocks[idx](x_dev, set_tail=stage_set_tail))
            ttnn.deallocate(x_dev)
            acc = block_outputs[0]
            for k in range(1, self.num_kernels):
                new_acc = ttnn.add(acc, block_outputs[k])
                ttnn.deallocate(acc)
                ttnn.deallocate(block_outputs[k])
                acc = new_acc
            x_dev = ttnn.multiply(acc, 1.0 / self.num_kernels)
            ttnn.deallocate(acc)

        x_dev = _set_tail(x_dev, cumrate, "replicate")  # act_post is a replicate-pad activation
        x_dev = self.act_post(x_dev)
        x_dev = _set_tail(x_dev, cumrate, "zeros")  # conv_post is zeros-pad
        x_dev = self.conv_post(x_dev)

        if self.apply_final_activation:
            if self.use_tanh_at_final:
                x_dev = ttnn.tanh(x_dev)
            else:
                x_dev = ttnn.clamp(x_dev, -1.0, 1.0)

        if sharded:
            x_dev = ttnn.to_layout(x_dev, ttnn.TILE_LAYOUT)
            x_dev = _all_gather_t(self.ccl_manager, x_dev, self.parallel_config)
            x_dev = ttnn.to_layout(x_dev, ttnn.ROW_MAJOR_LAYOUT)

        x_host = ttnn.to_torch(ttnn.get_device_tensors(x_dev)[0])
        # Trim padded out channels in case the conv left them.
        x_host = x_host[..., : self.out_channels]
        # Crop the upsampled image of the input T-padding.
        if t_pad > 0:
            prod_rates = 1
            for r in self.upsample_rates:
                prod_rates *= r
            x_host = x_host[:, : x_host.shape[1] - t_pad * prod_rates, :]
        x_host = x_host.transpose(-1, -2).contiguous()
        return x_host
