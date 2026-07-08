# SPDX-FileCopyrightText: (c) 2026 Tenstorrent
# SPDX-License-Identifier: Apache-2.0
"""TTNN port of the Higgs Audio v2 DAC decoder (audio codes -> waveform).

Mirrors transformers `DacDecoder`: an initial Conv1d, five DacDecoderBlocks
(Snake -> ConvTranspose1d upsample -> 3 dilated DacResidualUnits), then a final
Snake + Conv1d + tanh(Identity). Total upsample 960 (24kHz @ 25 code/s).

1D convs run on `ttnn.conv1d`; the transposed (upsampling) convs map to
`ttnn.conv_transpose2d` with height=1. Snake is elementwise:
    snake(x) = x + (1/(alpha+1e-9)) * sin(alpha*x)^2   (alpha per-channel).

Tensors flow channel-last [1, L, C] (the conv ops' NHWC-style layout).
"""
from __future__ import annotations

import torch
import ttnn


def _compute_cfg(device):
    return ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=False
    )


def tt_conv1d(device, x, weight, bias, in_ch, out_ch, L, k, stride=1, padding=0, dilation=1):
    """x: ttnn [1, L, in_ch] ROW_MAJOR (channel-last). Returns ([1, L_out, out_ch], w, b)."""
    # act_block_h_override caps the activation circular-buffer size so the codec
    # convs fit in L1 alongside a co-resident LLM (demo runs both on one chip).
    cfg = ttnn.Conv1dConfig(weights_dtype=ttnn.bfloat16, act_block_h_override=32)
    out, [w, b] = ttnn.conv1d(
        input_tensor=x,
        weight_tensor=weight,
        bias_tensor=bias,
        device=device,
        in_channels=in_ch,
        out_channels=out_ch,
        batch_size=1,
        input_length=L,
        kernel_size=k,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=1,
        dtype=ttnn.bfloat16,
        conv_config=cfg,
        compute_config=_compute_cfg(device),
        return_weights_and_bias=True,
        return_output_dim=False,
    )
    return out, w, b


def tt_conv_transpose1d(device, x, weight, bias, in_ch, out_ch, L, k, stride, padding, output_padding=0):
    """ConvTranspose1d via conv_transpose2d with height=1.

    x: ttnn [1, L, in_ch] ROW_MAJOR. weight torch [in, out, k] -> [in, out, 1, k].
    """
    cfg = ttnn.Conv2dConfig(weights_dtype=ttnn.bfloat16)  # auto shard layout
    out, [w, b] = ttnn.conv_transpose2d(
        input_tensor=x,
        weight_tensor=weight,
        bias_tensor=bias,
        device=device,
        in_channels=in_ch,
        out_channels=out_ch,
        kernel_size=(1, k),
        stride=(1, stride),
        padding=(0, padding),
        output_padding=(0, output_padding),
        dilation=(1, 1),
        groups=1,
        batch_size=1,
        input_height=1,
        input_width=L,
        conv_config=cfg,
        compute_config=_compute_cfg(device),
        mirror_kernel=True,  # PyTorch ConvTranspose flips the kernel; match it
        return_weights_and_bias=True,
        return_output_dim=False,
    )
    return out, w, b


def prep_conv1d_weight(w_torch, b_torch=None):
    """torch Conv1d weight [out,in,k] / bias [out] -> ttnn ROW_MAJOR tensors (host)."""
    w = ttnn.from_torch(w_torch.float(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    b = None
    if b_torch is not None:
        b = ttnn.from_torch(b_torch.float().reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    return w, b


def prep_conv_t1d_weight(w_torch, b_torch=None):
    """torch ConvTranspose1d weight [in,out,k] -> [in,out,1,k] ttnn ROW_MAJOR."""
    w = ttnn.from_torch(w_torch.float().unsqueeze(2), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    b = None
    if b_torch is not None:
        b = ttnn.from_torch(b_torch.float().reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    return w, b


def tt_snake(device, x_rm, alpha_torch, L, C):
    """Snake activation on channel-last [1, L, C]. alpha_torch: [1, C, 1] (per-channel).

    Returns ROW_MAJOR [1, L, C].
    """
    x = ttnn.to_layout(x_rm, ttnn.TILE_LAYOUT)
    x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
    alpha = ttnn.from_torch(
        alpha_torch.reshape(1, 1, C).float(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    inv = ttnn.reciprocal(ttnn.add(alpha, 1e-9))
    ax = ttnn.mul(x, alpha)  # broadcast [1,1,C] over L
    s = ttnn.sin(ax)
    s2 = ttnn.mul(s, s)
    out = ttnn.add(x, ttnn.mul(s2, inv))
    out = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)
    return out


def _to_rm(x):
    """Normalize a conv output (sharded/tile) to interleaved ROW_MAJOR [1, L, C]."""
    try:
        x = ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)
    except Exception:
        pass
    return ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)


class TtDacDecoder:
    """TTNN port of transformers DacDecoder. Reads weights/params live from the
    HF `acoustic_decoder` module (host); runs the conv stack on `device`.

    forward: quantized_acoustic torch [1, C0, T] -> waveform torch [1, 1, T*960].
    """

    def __init__(self, device, hf_decoder):
        self.device = device
        self.dec = hf_decoder
        self._wc = {}  # weight cache keyed by module id

    def _conv1d(self, conv, x, L):
        key = id(conv)
        if key not in self._wc:
            self._wc[key] = prep_conv1d_weight(conv.weight.data, conv.bias.data)
        w, b = self._wc[key]
        k, s, p, d = conv.kernel_size[0], conv.stride[0], conv.padding[0], conv.dilation[0]
        out, w, b = tt_conv1d(self.device, x, w, b, conv.in_channels, conv.out_channels, L, k, s, p, d)
        self._wc[key] = (w, b)  # cache device-prepped weights
        Lout = (L + 2 * p - d * (k - 1) - 1) // s + 1
        return _to_rm(out), Lout

    def _convt(self, ct, x, L):
        key = id(ct)
        if key not in self._wc:
            self._wc[key] = prep_conv_t1d_weight(ct.weight.data, ct.bias.data)
        w, b = self._wc[key]
        k, s, p, op = ct.kernel_size[0], ct.stride[0], ct.padding[0], ct.output_padding[0]
        out, w, b = tt_conv_transpose1d(self.device, x, w, b, ct.in_channels, ct.out_channels, L, k, s, p, op)
        self._wc[key] = (w, b)
        Lout = (L - 1) * s - 2 * p + (k - 1) + op + 1
        return _to_rm(out), Lout

    def _snake(self, sn, x, L):
        C = sn.alpha.shape[1]
        return tt_snake(self.device, x, sn.alpha.data, L, C)

    def _res_unit(self, ru, x, L):
        h = self._snake(ru.snake1, x, L)
        h, L1 = self._conv1d(ru.conv1, h, L)
        h = self._snake(ru.snake2, h, L1)
        h, L2 = self._conv1d(ru.conv2, h, L1)
        # residual (same padding keeps L2 == L); add channel-last in tile layout
        xt = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        ht = ttnn.to_layout(h, ttnn.TILE_LAYOUT)
        out = ttnn.add(xt, ht)
        return ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT), L2

    def _block(self, blk, x, L):
        x = self._snake(blk.snake1, x, L)
        x, L = self._convt(blk.conv_t1, x, L)
        x, L = self._res_unit(blk.res_unit1, x, L)
        x, L = self._res_unit(blk.res_unit2, x, L)
        x, L = self._res_unit(blk.res_unit3, x, L)
        return x, L

    def _run_conv(self, x, T):
        """x: ttnn channel-last [1, T, C0] ROW_MAJOR -> torch waveform [1, 1, T*960]."""
        L = T
        x, L = self._conv1d(self.dec.conv1, x, L)
        for blk in self.dec.block:
            x, L = self._block(blk, x, L)
        x = self._snake(self.dec.snake1, x, L)
        x, L = self._conv1d(self.dec.conv2, x, L)  # -> 1 channel
        # tanh is Identity in this checkpoint; skip.
        wave = ttnn.to_torch(ttnn.from_device(x)).reshape(1, -1, 1)[:, :L, :]
        return wave.transpose(1, 2)  # [1, 1, L]

    def forward_ttnn(self, x, T):
        """Run the conv stack on an already-on-device channel-last [1, T, C0] tensor
        (used by the fully-on-device path where RVQ+fc2 produced x on device)."""
        return self._run_conv(x, T)

    def forward(self, quantized_acoustic):
        """quantized_acoustic: torch [1, C0, T] -> torch waveform [1, 1, T*960]."""
        B, C0, T = quantized_acoustic.shape
        assert B == 1
        x = ttnn.from_torch(
            quantized_acoustic.transpose(1, 2).contiguous().float(),  # [1, T, C0]
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
        )
        return self._run_conv(x, T)


class TtRvqDequant:
    """On-device RVQ dequant + fc2: audio_codes [1, K, T] (0..1023) -> ttnn channel-last
    [1, T, 256] (the DacDecoder's input, so it feeds the conv stack with no host round-trip).

    Mirrors HiggsAudioV2TokenizerResidualVectorQuantization.decode + fc2: per codebook,
    codebook lookup (`ttnn.embedding`) -> project_out (`ttnn.linear`, 64->1024); sum the 8
    codebooks; fc2 (`ttnn.linear`, 1024->256). Weights read live from the HF quantizer/fc2
    (host) and prepped to device once. PCC 0.99998 vs host fp32.
    """

    def __init__(self, device, quantizer, fc2):
        self.device = device
        self.K = len(quantizer.quantizers)
        rep = ttnn.ReplicateTensorToMesh(device)

        def _tile(w):
            return ttnn.from_torch(
                w.contiguous().float(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=rep
            )

        # embedding weight tables must be ROW_MAJOR bf16 [codebook_size, codebook_dim]
        self.embed = [
            ttnn.from_torch(
                quantizer.quantizers[k].codebook.embed.contiguous().float(),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                mesh_mapper=rep,
            )
            for k in range(self.K)
        ]
        # torch Linear y = x @ W.T; ttnn.linear(x, Wtt) = x @ Wtt, so Wtt = W.T
        self.po_w = [_tile(quantizer.quantizers[k].project_out.weight.t()) for k in range(self.K)]
        self.po_b = [_tile(quantizer.quantizers[k].project_out.bias.view(1, 1, -1)) for k in range(self.K)]
        self.fc2_w = _tile(fc2.weight.t())
        self.fc2_b = _tile(fc2.bias.view(1, 1, -1))

    def __call__(self, audio_codes):
        """audio_codes: torch [1, K, T] -> ttnn channel-last [1, T, 256] ROW_MAJOR, and T."""
        rep = ttnn.ReplicateTensorToMesh(self.device)
        T = audio_codes.shape[-1]
        acc = None
        for k in range(self.K):
            ck = ttnn.from_torch(
                audio_codes[:, k, :].to(torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                mesh_mapper=rep,
            )
            ek = ttnn.to_layout(ttnn.embedding(ck, self.embed[k]), ttnn.TILE_LAYOUT)  # [1, T, 64]
            pk = ttnn.linear(ek, self.po_w[k], bias=self.po_b[k])  # [1, T, 1024]
            acc = pk if acc is None else ttnn.add(acc, pk)
        qa = ttnn.linear(acc, self.fc2_w, bias=self.fc2_b)  # [1, T, 256] channel-last
        return ttnn.to_layout(qa, ttnn.ROW_MAJOR_LAYOUT), T


def tt_decode(device, hf_model, audio_codes, ttdec=None, rvq=None):
    """Full codec decode, FULLY ON DEVICE: audio_codes [1, K, T] (0..1023) -> waveform
    [1, 1, T*960].

    RVQ dequant + fc2 (`TtRvqDequant`) and the DacDecoder conv stack (`TtDacDecoder`) both
    run on TTNN; the RVQ output feeds the conv stack channel-last with no host round-trip.
    Nothing stays on host. Mirrors HiggsAudioV2TokenizerModel.decode. PCC ~0.999 vs HF.
    """
    ttdec = ttdec or TtDacDecoder(device, hf_model.acoustic_decoder)
    rvq = rvq or TtRvqDequant(device, hf_model.quantizer, hf_model.fc2)
    x, T = rvq(audio_codes)  # on-device [1, T, 256]
    return ttdec.forward_ttnn(x, T)
