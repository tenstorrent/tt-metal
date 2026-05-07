# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TTNN [`SeamlessM4Tv2CodeHifiGan`] — embeddings, duration predictor, HiFi-GAN (inference).

Follows Hugging Face layout from ``transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2``.
Uses ``ttnn.conv1d`` for Conv1d (same pattern as speech encoder) and ``ttnn.conv_transpose2d``
with ``H=1`` for ConvTranspose1d (same idea as ``models/demos/vision/segmentation/vanilla_unet``).

Duration expansion uses PyTorch ``repeat_interleave`` on CPU for exact parity with HF when
batch size is 1 (see PCC test).
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
import ttnn

from models.experimental.seamless_m4t_v2_large.reference.torch_code_hifigan import (
    compute_dur_output_lengths,
    compute_hifigan_output_lengths,
)


def _core_grid(device: ttnn.Device) -> ttnn.CoreGrid:
    grid = device.compute_with_storage_grid_size()
    return ttnn.CoreGrid(y=grid.y, x=grid.x)


class TTSeamlessM4Tv2CodeHifiGan:
    """Inference forward for HF ``SeamlessM4Tv2CodeHifiGan``."""

    def __init__(self, device: ttnn.Device, parameters: Any, config: Any):
        self.device = device
        self.p = parameters
        self.cfg = config
        self.leaky_slope = float(config.leaky_relu_slope)
        self.num_kernels = len(config.resblock_kernel_sizes)

        self._compute = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def _linear(self, x: ttnn.Tensor, weight: ttnn.Tensor, bias: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.linear(
            x,
            weight,
            bias=bias,
            core_grid=_core_grid(self.device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self._compute,
        )

    def _layer_norm(self, x: ttnn.Tensor, *, weight: ttnn.Tensor, bias: ttnn.Tensor, eps: float) -> ttnn.Tensor:
        return ttnn.layer_norm(
            x,
            weight=weight,
            bias=bias,
            epsilon=eps,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _conv1d(
        self,
        x_nlc: ttnn.Tensor,
        *,
        weight: ttnn.Tensor,
        bias: Optional[ttnn.Tensor],
        batch: int,
        input_length: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        groups: int,
        dilation: int = 1,
    ) -> Tuple[ttnn.Tensor, int]:
        conv_config = ttnn.Conv1dConfig(
            weights_dtype=ttnn.bfloat16,
            shard_layout=None,
            deallocate_activation=False,
        )
        out, out_len = ttnn.conv1d(
            input_tensor=x_nlc,
            weight_tensor=weight,
            in_channels=in_channels,
            out_channels=out_channels,
            device=self.device,
            bias_tensor=bias,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            batch_size=batch,
            input_length=input_length,
            conv_config=conv_config,
            compute_config=self._compute,
            groups=groups,
            dilation=dilation,
            dtype=ttnn.bfloat16,
            return_output_dim=True,
        )
        out_len = int(out_len)
        if ttnn.is_sharded(out):
            out = ttnn.sharded_to_interleaved(out, ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.reshape(out, (batch, out_len, out_channels))
        if ttnn.is_sharded(out):
            out = ttnn.sharded_to_interleaved(out, ttnn.DRAM_MEMORY_CONFIG)
        return out, out_len

    def _conv_transpose1d_nlc(
        self,
        x_nlc: ttnn.Tensor,
        *,
        layer: Any,
        batch: int,
        input_length: int,
        in_channels: int,
        out_channels: int,
    ) -> Tuple[ttnn.Tensor, int]:
        """Maps HF ``ConvTranspose1d`` to ``ttnn.conv_transpose2d`` with singleton width."""
        k = int(layer["kernel_size"])
        s = int(layer["stride"])
        p = int(layer["padding"])
        weight = layer["weight"]
        bias = layer["bias"]

        x_nhwc = ttnn.reshape(x_nlc, (batch, input_length, 1, in_channels))
        conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat16,
            shard_layout=None,
            deallocate_activation=False,
            output_layout=ttnn.TILE_LAYOUT,
        )
        out_4d, out_hw = ttnn.conv_transpose2d(
            input_tensor=x_nhwc,
            weight_tensor=weight,
            in_channels=in_channels,
            out_channels=out_channels,
            device=self.device,
            bias_tensor=bias,
            kernel_size=(k, 1),
            stride=(s, 1),
            padding=(p, 0),
            output_padding=(0, 0),
            dilation=(1, 1),
            batch_size=batch,
            input_height=input_length,
            input_width=1,
            conv_config=conv_config,
            compute_config=self._compute,
            groups=1,
            return_output_dim=True,
            return_weights_and_bias=False,
            dtype=ttnn.bfloat16,
        )
        out_h, out_w = int(out_hw[0]), int(out_hw[1])
        assert out_w == 1
        out_nlc = ttnn.reshape(out_4d, (batch, out_h, out_channels))
        if ttnn.is_sharded(out_nlc):
            out_nlc = ttnn.sharded_to_interleaved(out_nlc, ttnn.DRAM_MEMORY_CONFIG)
        return out_nlc, out_h

    def _dur_predictor(self, x_nlc: ttnn.Tensor, *, batch: int, seq: int, dp: Any) -> torch.Tensor:
        """Returns ``log_dur_pred`` as CPU ``torch.Tensor`` ``[B, T]`` (float32)."""
        c1, c2 = dp.conv1, dp.conv2
        h, tlen = self._conv1d(
            x_nlc,
            weight=c1.weight,
            bias=c1.bias,
            batch=batch,
            input_length=seq,
            in_channels=c1.in_channels,
            out_channels=c1.out_channels,
            kernel_size=c1.kernel_size,
            stride=1,
            padding=c1.padding,
            groups=1,
        )
        h = ttnn.relu(h, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        h = self._layer_norm(h, weight=dp.ln1.weight, bias=dp.ln1.bias, eps=dp.ln1.eps)
        h, tlen = self._conv1d(
            h,
            weight=c2.weight,
            bias=c2.bias,
            batch=batch,
            input_length=tlen,
            in_channels=c2.in_channels,
            out_channels=c2.out_channels,
            kernel_size=c2.kernel_size,
            stride=1,
            padding=c2.padding,
            groups=1,
        )
        h = ttnn.relu(h, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        h = self._layer_norm(h, weight=dp.ln2.weight, bias=dp.ln2.bias, eps=dp.ln2.eps)
        log_dur = self._linear(h, dp.proj.weight, dp.proj.bias)
        ttnn.deallocate(h)
        log_cpu = ttnn.to_torch(ttnn.from_device(log_dur)).to(torch.float32).squeeze(-1)
        ttnn.deallocate(log_dur)
        return log_cpu

    def _resblock(self, x_nlc: ttnn.Tensor, rb: Any, *, batch: int, tlen: int, channels: int) -> ttnn.Tensor:
        """One HF ``HifiGanResidualBlock``; ``x`` is ``[B,T,C]``."""
        for c1p, c2p in zip(rb.convs1, rb.convs2):
            residual = x_nlc
            x_nlc = ttnn.leaky_relu(x_nlc, negative_slope=self.leaky_slope, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            x_nlc, tlen = self._conv1d(
                x_nlc,
                weight=c1p["weight"],
                bias=c1p["bias"],
                batch=batch,
                input_length=tlen,
                in_channels=channels,
                out_channels=channels,
                kernel_size=int(c1p["kernel_size"]),
                stride=1,
                padding=int(c1p["padding"]),
                groups=1,
                dilation=int(c1p["dilation"]),
            )
            x_nlc = ttnn.leaky_relu(x_nlc, negative_slope=self.leaky_slope, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            x_nlc, tlen = self._conv1d(
                x_nlc,
                weight=c2p["weight"],
                bias=c2p["bias"],
                batch=batch,
                input_length=tlen,
                in_channels=channels,
                out_channels=channels,
                kernel_size=int(c2p["kernel_size"]),
                stride=1,
                padding=int(c2p["padding"]),
                groups=1,
                dilation=int(c2p["dilation"]),
            )
            x_nlc = ttnn.add(x_nlc, residual, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return x_nlc

    def _hifi_gan(self, x_nlc: ttnn.Tensor, hg: Any, *, batch: int, tlen: int) -> ttnn.Tensor:
        cp = hg.conv_pre
        h, tlen = self._conv1d(
            x_nlc,
            weight=cp.weight,
            bias=cp.bias,
            batch=batch,
            input_length=tlen,
            in_channels=int(cp.in_channels),
            out_channels=int(cp.out_channels),
            kernel_size=int(cp.kernel_size),
            stride=1,
            padding=int(cp.padding),
            groups=1,
        )
        ttnn.deallocate(x_nlc)

        for i, up_layer in enumerate(hg.upsampler):
            h = ttnn.leaky_relu(h, negative_slope=self.leaky_slope, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            h, tlen = self._conv_transpose1d_nlc(
                h,
                layer=up_layer,
                batch=batch,
                input_length=tlen,
                in_channels=int(up_layer["in_channels"]),
                out_channels=int(up_layer["out_channels"]),
            )
            channels = self.cfg.upsample_initial_channel // (2 ** (i + 1))
            acc = None
            for j in range(self.num_kernels):
                rb = hg.resblocks[i * self.num_kernels + j]
                br = self._resblock(h, rb, batch=batch, tlen=tlen, channels=channels)
                if acc is None:
                    acc = br
                else:
                    acc = ttnn.add(acc, br, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                    ttnn.deallocate(br)
            scale = 1.0 / float(self.num_kernels)
            acc_scaled = ttnn.multiply(acc, scale, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(acc)
            ttnn.deallocate(h)
            h = acc_scaled

        h = ttnn.leaky_relu(h, negative_slope=self.leaky_slope, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        cpost = hg.conv_post
        h, tlen = self._conv1d(
            h,
            weight=cpost.weight,
            bias=cpost.bias,
            batch=batch,
            input_length=tlen,
            in_channels=int(cpost.in_channels),
            out_channels=int(cpost.out_channels),
            kernel_size=int(cpost.kernel_size),
            stride=1,
            padding=int(cpost.padding),
            groups=1,
        )
        h = ttnn.tanh(h, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return h

    def forward(
        self,
        input_ids: ttnn.Tensor,
        speaker_id: ttnn.Tensor,
        lang_id: ttnn.Tensor,
        *,
        input_ids_torch: torch.Tensor,
    ) -> Tuple[ttnn.Tensor, torch.Tensor]:
        """
        Args:
            input_ids: ``uint32`` ``[B, T]`` on device (unit tokens).
            speaker_id: ``uint32`` ``[B, 1]`` on device.
            lang_id: ``uint32`` ``[B, 1]`` on device.
            input_ids_torch: host ``LongTensor`` for length math (padding id); must match ``input_ids``.

        Returns:
            ``(waveform, lengths)`` — waveform ``[B, T_audio, 1]`` as ``[B, T_audio]`` after squeeze on host;
            ``lengths`` CPU ``LongTensor`` ``[B]``.
        """
        batch = int(input_ids.shape[0])
        seq = int(input_ids.shape[1])
        if batch != 1:
            raise NotImplementedError("TT vocoder currently supports batch size 1 (matches HF PCC path).")

        ue = self.p.unit_embedding.weight
        use = ttnn.embedding(input_ids, weight=ue, layout=ttnn.TILE_LAYOUT)

        sp = self.p.speaker_embedding.weight
        la = self.p.language_embedding.weight
        sp_e = ttnn.embedding(ttnn.squeeze(speaker_id, 1), weight=sp, layout=ttnn.TILE_LAYOUT)
        lang_e = ttnn.embedding(ttnn.squeeze(lang_id, 1), weight=la, layout=ttnn.TILE_LAYOUT)

        use_host_bf16 = ttnn.to_torch(ttnn.from_device(use)).to(torch.bfloat16)

        dp = self.p.dur_predictor
        log_dur_cpu = self._dur_predictor(use, batch=batch, seq=seq, dp=dp)
        ttnn.deallocate(use)

        dur_out = torch.clamp(torch.round(torch.expm1(log_dur_cpu)).long(), min=1)

        # HF: hidden_states [B, E, T]; repeat_interleave along time (dim=-1 after B,E,T layout).
        u_bet = use_host_bf16.permute(0, 2, 1).contiguous()
        expanded = torch.repeat_interleave(u_bet[0], dur_out[0].reshape(-1), dim=-1).unsqueeze(0)
        t_audio = expanded.shape[-1]

        lang_bf16 = ttnn.to_torch(ttnn.from_device(lang_e)).to(torch.bfloat16)
        spk_bf16 = ttnn.to_torch(ttnn.from_device(sp_e)).to(torch.bfloat16)
        ttnn.deallocate(lang_e)
        ttnn.deallocate(sp_e)

        lang_bt = lang_bf16.unsqueeze(-1).expand(batch, lang_bf16.shape[-1], t_audio)
        spk_bt = spk_bf16.unsqueeze(-1).expand(batch, spk_bf16.shape[-1], t_audio)
        merged_bct = torch.cat([lang_bt, expanded, spk_bt], dim=1)

        merged_nlc = ttnn.from_torch(
            merged_bct.permute(0, 2, 1).contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        wav = self._hifi_gan(merged_nlc, self.p.hifi_gan, batch=batch, tlen=t_audio)

        unit_lens = compute_dur_output_lengths(
            input_ids_torch,
            dur_out,
            pad_token_id=int(self.cfg.t2u_pad_token_id),
        )
        lengths = compute_hifigan_output_lengths(unit_lens, self.cfg)
        return wav, lengths
