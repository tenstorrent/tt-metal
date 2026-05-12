# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TTNN [`SeamlessM4Tv2CodeHifiGan`] — embeddings, duration predictor, HiFi-GAN (inference).

Follows Hugging Face layout from ``transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2``.
Uses ``ttnn.conv1d`` for Conv1d (same pattern as speech encoder) and ``ttnn.conv_transpose2d``
with ``H=1`` for ConvTranspose1d (same idea as ``models/demos/vision/segmentation/vanilla_unet``).

All tensor math runs on device (no PyTorch fallback). HiFi-GAN must see the **same**
temporal length as Hugging Face: every ``conv1d`` / ``conv_transpose`` output length depends on
``input_length``. Padding or fixing ``t_audio`` to a larger constant changes those lengths end-to-end,
so the waveform no longer matches HF even after cropping — that is why a **single scalar**
``t_audio = sum(durations)`` is read once for shape control (``.item()`` on the last cumsum); it is
not a compute fallback.

  - Duration math (``expm1`` / ``round`` / ``minimum duration 1``) uses TTNN ops.
  - HF's ``repeat_interleave`` is ``embeddings @ H`` with ``ttnn.cumsum`` + comparisons.
  - Lang/speaker broadcast uses ``ttnn.repeat`` to that same ``t_audio``.
  - Output lengths are an on-device ``ttnn.Tensor``; ``lengths`` stays on device for callers.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
import ttnn


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

    def _dur_predictor_dev(self, x_nlc: ttnn.Tensor, *, batch: int, seq: int, dp: Any) -> ttnn.Tensor:
        """Returns log-duration prediction as a device tensor of shape ``[B, T_units]`` (bf16)."""
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
        log_dur = self._linear(h, dp.proj.weight, dp.proj.bias)  # [B, T_units, 1]
        ttnn.deallocate(h)
        log_dur_2d = ttnn.squeeze(log_dur, -1)  # [B, T_units]
        ttnn.deallocate(log_dur)
        return log_dur_2d

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

        # Match HF: line 2489 of ``modeling_seamless_m4t_v2.py`` calls
        # ``nn.functional.leaky_relu(hidden_states)`` with the default 0.01 slope, NOT
        # ``self.leaky_relu_slope``. The other leaky_relus in the upsample loop and inside the
        # residual blocks do use ``cfg.leaky_relu_slope``.
        h = ttnn.leaky_relu(h, negative_slope=0.01, memory_config=ttnn.DRAM_MEMORY_CONFIG)
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
        input_ids_torch: Optional[torch.Tensor] = None,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """
        Args:
            input_ids: ``uint32`` ``[B, T]`` on device (unit tokens).
            speaker_id: ``uint32`` ``[B, 1]`` on device.
            lang_id: ``uint32`` ``[B, 1]`` on device.
            input_ids_torch: kept for API compatibility; unused.

        Returns:
            ``(waveform, lengths)`` — both device tensors. ``waveform`` is bfloat16
            ``[B, T_wav_max, 1]``; ``lengths`` is int32 ``[B]`` and gives the valid
            audio sample count per row. For ``B > 1`` rows are right-padded with zeros to
            ``T_wav_max = max_b T_wav_b``; consumers must crop using ``lengths``.

        Notes:
            * ``B == 1`` runs a single fused on-device program (the fast path).
            * ``B > 1`` mirrors the Hugging Face implementation, which also processes rows
              sequentially (see ``modeling_seamless_m4t_v2.py`` lines 2611-2623). Each row
              goes through exactly the same ``_forward_one`` device program, so per-row PCC
              is identical to the ``B == 1`` baseline. Padding and final concat happen on
              device — no PyTorch compute is introduced.
        """
        del input_ids_torch  # API-compat stub.
        batch = int(input_ids.shape[0])
        if batch == 1:
            return self._forward_one(input_ids, speaker_id, lang_id)
        return self._forward_batched(input_ids, speaker_id, lang_id, batch=batch)

    # ------------------------------------------------------------------------------- B == 1

    def _forward_one(
        self,
        input_ids: ttnn.Tensor,
        speaker_id: ttnn.Tensor,
        lang_id: ttnn.Tensor,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """Single-sample forward (``B == 1``); fully on device modulo one shape int."""
        batch = int(input_ids.shape[0])
        seq = int(input_ids.shape[1])
        assert batch == 1, "_forward_one expects B == 1; use forward() for B > 1."

        # ---------- embeddings (all on device) ----------
        ue = self.p.unit_embedding.weight
        use = ttnn.embedding(input_ids, weight=ue, layout=ttnn.TILE_LAYOUT)  # [B, T_units, E_unit]

        sp = self.p.speaker_embedding.weight
        la = self.p.language_embedding.weight
        sp_e = ttnn.embedding(ttnn.squeeze(speaker_id, 1), weight=sp, layout=ttnn.TILE_LAYOUT)
        lang_e = ttnn.embedding(ttnn.squeeze(lang_id, 1), weight=la, layout=ttnn.TILE_LAYOUT)

        # ---------- duration prediction (device) ----------
        dp = self.p.dur_predictor
        log_dur = self._dur_predictor_dev(use, batch=batch, seq=seq, dp=dp)  # [B, T_units] bf16

        # HF: dur_out = clamp(round(expm1(log_dur)), min=1).long()
        e = ttnn.expm1(log_dur, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(log_dur)
        r = ttnn.round(e, decimals=0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(e)
        dur_bf = ttnn.maximum(r, 1.0, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [B, T_units] bf16
        ttnn.deallocate(r)

        # ---------- exclusive cumulative duration (device) ----------
        # Use float32 internally so integer comparisons are exact for any plausible t_audio.
        dur_f32 = ttnn.typecast(dur_bf, ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        cumsum_inc = ttnn.cumsum(dur_f32, dim=-1, dtype=ttnn.float32)  # [B, T_units] inclusive
        cumsum_prev = ttnn.subtract(cumsum_inc, dur_f32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(dur_f32)

        # HiFi-GAN ``input_length`` must match HF's repeat-interleave length (sum of durations).
        # One scalar read for Python ``range`` / tensor shapes only (B=1).
        t_audio = int(ttnn.to_torch(ttnn.from_device(cumsum_inc)).reshape(batch, seq)[0, -1].item())
        if t_audio < 1:
            raise RuntimeError(f"Computed t_audio={t_audio}; expected positive duration sum.")

        # ---------- expansion via embeddings @ H (device) ----------
        # H[b, i, j] = 1 iff cumsum_prev[b, i] <= j < cumsum_inc[b, i].
        frame_idx = ttnn.arange(
            start=0,
            end=t_audio,
            step=1,
            dtype=ttnn.float32,
            device=self.device,
        )
        # Reshape for broadcasting: [B, T_units, 1] vs [1, 1, t_audio].
        c_b = ttnn.reshape(cumsum_inc, (batch, seq, 1))
        cp_b = ttnn.reshape(cumsum_prev, (batch, seq, 1))
        f_b = ttnn.reshape(frame_idx, (1, 1, t_audio))
        ttnn.deallocate(frame_idx)

        lower = ttnn.ge(f_b, cp_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        upper = ttnn.lt(f_b, c_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(c_b)
        ttnn.deallocate(cp_b)
        ttnn.deallocate(f_b)
        H_mask = ttnn.logical_and(lower, upper, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(lower)
        ttnn.deallocate(upper)
        H = ttnn.typecast(H_mask, ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(H_mask)

        # use: [B, T_units, E_unit] -> [B, E_unit, T_units]; expand to [B, E_unit, t_audio].
        use_BEC = ttnn.permute(use, (0, 2, 1))
        ttnn.deallocate(use)
        expanded_BCT = ttnn.matmul(
            use_BEC,
            H,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self._compute,
        )
        ttnn.deallocate(use_BEC)
        ttnn.deallocate(H)

        # ---------- broadcast lang/spk to ``t_audio`` (device) ----------
        lang_dim = int(lang_e.shape[-1])
        spk_dim = int(sp_e.shape[-1])
        lang_BC1 = ttnn.reshape(lang_e, (batch, lang_dim, 1))
        spk_BC1 = ttnn.reshape(sp_e, (batch, spk_dim, 1))
        ttnn.deallocate(lang_e)
        ttnn.deallocate(sp_e)
        lang_BCT = ttnn.repeat(lang_BC1, [1, 1, t_audio])
        spk_BCT = ttnn.repeat(spk_BC1, [1, 1, t_audio])
        ttnn.deallocate(lang_BC1)
        ttnn.deallocate(spk_BC1)

        merged_BCT = ttnn.concat([lang_BCT, expanded_BCT, spk_BCT], dim=1)
        ttnn.deallocate(lang_BCT)
        ttnn.deallocate(expanded_BCT)
        ttnn.deallocate(spk_BCT)
        merged_NLC = ttnn.permute(merged_BCT, (0, 2, 1))
        ttnn.deallocate(merged_BCT)

        # ---------- HiFi-GAN (device) ----------
        wav = self._hifi_gan(merged_NLC, self.p.hifi_gan, batch=batch, tlen=t_audio)

        # ---------- length math (device int32) ----------
        lengths = self._compute_output_lengths_dev(input_ids, cumsum_inc, batch=batch, seq=seq)
        ttnn.deallocate(cumsum_inc)
        ttnn.deallocate(cumsum_prev)
        ttnn.deallocate(dur_bf)
        return wav, lengths

    # ------------------------------------------------------------------------------- B > 1

    def _forward_batched(
        self,
        input_ids: ttnn.Tensor,
        speaker_id: ttnn.Tensor,
        lang_id: ttnn.Tensor,
        *,
        batch: int,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """
        Multi-sample forward for any ``B > 1``. Mirrors HF's per-sample loop in
        ``modeling_seamless_m4t_v2.py`` (lines 2611-2623). Each row runs through the
        identical ``_forward_one`` device program, so per-row PCC is unchanged. Padding
        each row's waveform up to the batch maximum and the final concat are done on
        device via ``ttnn.pad`` and ``ttnn.concat``; no PyTorch compute is introduced.
        """
        seq = int(input_ids.shape[1])

        wavs: list[ttnn.Tensor] = []  # one [1, T_wav_b, 1] device tensor per row
        wav_lens: list[int] = []  # T_wav_b (Python ints from tensor metadata)
        valid_lens: list[ttnn.Tensor] = []  # one [1] int32 device tensor per row

        for b in range(batch):
            ids_b = ttnn.slice(input_ids, [b, 0], [b + 1, seq])
            spk_b = ttnn.slice(speaker_id, [b, 0], [b + 1, 1])
            lang_b = ttnn.slice(lang_id, [b, 0], [b + 1, 1])

            wav_b, len_b = self._forward_one(ids_b, spk_b, lang_b)
            ttnn.deallocate(ids_b)
            ttnn.deallocate(spk_b)
            ttnn.deallocate(lang_b)

            wavs.append(wav_b)
            wav_lens.append(int(wav_b.shape[1]))
            valid_lens.append(len_b)

        t_wav_max = max(wav_lens)

        # Pad each row to ``[1, t_wav_max, 1]`` on device. ``ttnn.pad`` requires ROW_MAJOR
        # for arbitrary (non-tile-aligned) trailing-dim sizes, so we briefly drop layout.
        padded: list[ttnn.Tensor] = []
        for wav_b, t_b in zip(wavs, wav_lens):
            if wav_b.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
                wav_rm = ttnn.to_layout(wav_b, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(wav_b)
            else:
                wav_rm = wav_b
            if t_b == t_wav_max:
                padded.append(wav_rm)
            else:
                wav_pad = ttnn.pad(
                    wav_rm,
                    padding=[(0, 0), (0, t_wav_max - t_b), (0, 0)],
                    value=0.0,
                )
                ttnn.deallocate(wav_rm)
                padded.append(wav_pad)

        waveform = ttnn.concat(padded, dim=0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for w in padded:
            ttnn.deallocate(w)

        lengths = ttnn.concat(valid_lens, dim=0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for l in valid_lens:
            ttnn.deallocate(l)

        return waveform, lengths

    # ------------------------------------------------------------ on-device length math

    def _compute_output_lengths_dev(
        self,
        input_ids: ttnn.Tensor,
        cumsum_inc: ttnn.Tensor,
        *,
        batch: int,
        seq: int,
    ) -> ttnn.Tensor:
        """
        Port of HF's ``_get_dur_output_lengths`` + ``_get_output_hifigan_lengths``, fully on device.

        Returns an int32 device tensor of shape ``[B]`` — the valid output prefix length
        per batch row. Nothing is moved to host.
        """
        pad_id = int(self.cfg.t2u_pad_token_id)

        # not_pad: [B, T_units] -> sum -> [B]
        not_pad = ttnn.ne(input_ids, pad_id, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        not_pad_f32 = ttnn.typecast(not_pad, ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(not_pad)
        unit_lengths = ttnn.sum(not_pad_f32, dim=1)  # [B]
        ttnn.deallocate(not_pad_f32)
        # HF clamps to [0, T_units - 1] before gather.
        unit_lengths = ttnn.minimum(unit_lengths, float(seq - 1), memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Gather column ``unit_lengths[b]`` from row ``b`` of ``cumsum_inc``.
        # one_hot[b, i] = 1 iff i == unit_lengths[b]   then  out[b] = sum_i cumsum[b,i] * one_hot[b,i]
        col_idx = ttnn.arange(
            start=0,
            end=seq,
            step=1,
            dtype=ttnn.float32,
            device=self.device,
        )  # [T_units]
        ul_2d = ttnn.reshape(unit_lengths, (batch, 1))
        ci_2d = ttnn.reshape(col_idx, (1, seq))
        ttnn.deallocate(col_idx)
        oh_bool = ttnn.eq(ci_2d, ul_2d, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(ci_2d)
        ttnn.deallocate(ul_2d)
        ttnn.deallocate(unit_lengths)
        oh = ttnn.typecast(oh_bool, ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(oh_bool)
        gathered = ttnn.multiply(cumsum_inc, oh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(oh)
        unit_lens = ttnn.sum(gathered, dim=1)  # [B]
        ttnn.deallocate(gathered)

        # _get_output_hifigan_lengths: integer pipeline of conv / transpose-conv length formulas.
        x = unit_lens

        def _conv_out_length(
            x_t: ttnn.Tensor, kernel_size: int, stride: int, pad: int, dilation: int = 1
        ) -> ttnn.Tensor:
            # floor((x + 2*pad - dilation*(kernel-1) - 1) / stride) + 1
            offset = 2 * pad - dilation * (kernel_size - 1) - 1
            num = ttnn.add(x_t, float(offset), memory_config=ttnn.DRAM_MEMORY_CONFIG)
            div = ttnn.div(num, float(stride), memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(num)
            f = ttnn.floor(div, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(div)
            out = ttnn.add(f, 1.0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(f)
            return out

        def _transpose_conv_out_length(
            x_t: ttnn.Tensor, kernel_size: int, stride: int, pad: int, dilation: int = 1
        ) -> ttnn.Tensor:
            # (x - 1) * stride - 2*pad + dilation*(kernel-1) + 1
            const = -2 * pad + dilation * (kernel_size - 1) + 1 - stride  # since (x-1)*s = x*s - s
            scaled = ttnn.multiply(x_t, float(stride), memory_config=ttnn.DRAM_MEMORY_CONFIG)
            out = ttnn.add(scaled, float(const), memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(scaled)
            return out

        x = _conv_out_length(x, 7, 1, 3)

        for upsample_rate, kernel_size in zip(self.cfg.upsample_rates, self.cfg.upsample_kernel_sizes):
            x = _transpose_conv_out_length(x, kernel_size, upsample_rate, (kernel_size - upsample_rate) // 2)

        for _ in range(len(self.cfg.upsample_rates)):
            for kernel_size, dilation in zip(self.cfg.resblock_kernel_sizes, self.cfg.resblock_dilation_sizes):
                for dil in dilation:
                    x = _conv_out_length(x, kernel_size, 1, (kernel_size - 1) * dil // 2, dilation=dil)
                for _dil in dilation:
                    x = _conv_out_length(x, kernel_size, 1, (kernel_size - 1) // 2, dilation=1)

        x = _conv_out_length(x, 7, 1, 3)

        # Cast to int32 and reshape to [B]; remains on device.
        x_int = ttnn.typecast(x, ttnn.int32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(x)
        x_int = ttnn.reshape(x_int, (batch,))
        return x_int
