# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""TTNN port of Kokoro `TextEncoder` (embedding + Conv1d blocks + BiLSTM)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

import ttnn
from models.experimental.kokoro.tt.ttnn_kokoro_conv import Conv1dParams, conv1d_nlc, weight_norm_weight
from models.experimental.kokoro.tt.ttnn_kokoro_lstm import bilstm_nlc, preprocess_pytorch_lstm_1layer


@dataclass(frozen=True)
class TextEncoderParams:
    embedding_weight: ttnn.Tensor
    convs: list[Conv1dParams]
    ln_weight: list[ttnn.Tensor]
    ln_bias: list[ttnn.Tensor]
    lstm_fwd: object
    lstm_rev: object
    ln_eps: float


def preprocess_text_encoder(
    text_encoder: nn.Module,
    device: ttnn.Device,
    *,
    weights_dtype=ttnn.bfloat16,
) -> TextEncoderParams:
    emb_w = ttnn.from_torch(
        text_encoder.embedding.weight.detach().cpu(),
        dtype=weights_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    conv_params: list[Conv1dParams] = []
    ln_w: list[ttnn.Tensor] = []
    ln_b: list[ttnn.Tensor] = []

    for block in text_encoder.cnn:
        conv = block[0]
        ln = block[1]

        # weight_norm conv stores weight_g / weight_v
        w = weight_norm_weight(conv.weight_v.detach().cpu(), conv.weight_g.detach().cpu())
        b = conv.bias.detach().cpu() if conv.bias is not None else None

        # TTNN conv1d expects *host* weights in ROW_MAJOR layout; it will preprocess/move internally.
        w_tt = ttnn.from_torch(w, dtype=weights_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
        b_tt = (
            ttnn.from_torch(b.reshape(1, 1, 1, -1), dtype=weights_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
            if b is not None
            else None
        )
        conv_params.append(
            Conv1dParams(
                weight=w_tt,
                bias=b_tt,
                in_channels=conv.in_channels,
                out_channels=conv.out_channels,
                kernel_size=conv.kernel_size[0],
                stride=conv.stride[0],
                padding=conv.padding[0],
                groups=conv.groups,
            )
        )

        ln_w.append(
            ttnn.from_torch(
                ln.gamma.detach().cpu(),
                dtype=weights_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        )
        ln_b.append(
            ttnn.from_torch(
                ln.beta.detach().cpu(),
                dtype=weights_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        )

    fwd, rev = preprocess_pytorch_lstm_1layer(text_encoder.lstm, device, weights_dtype=weights_dtype)
    assert rev is not None, "TextEncoder uses bidirectional LSTM"

    return TextEncoderParams(
        embedding_weight=emb_w,
        convs=conv_params,
        ln_weight=ln_w,
        ln_bias=ln_b,
        lstm_fwd=fwd,
        lstm_rev=rev,
        ln_eps=1e-5,
    )


class TtKokoroTextEncoder:
    def __init__(self, device: ttnn.Device, params: TextEncoderParams):
        self.device = device
        self.params = params
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4
        )

    def __call__(
        self,
        input_ids: torch.LongTensor,
        input_lengths: torch.LongTensor,
        text_mask: torch.Tensor,
    ) -> ttnn.Tensor:
        """
        Args:
            input_ids: [B, T] torch on CPU
            input_lengths: [B] torch on CPU
            text_mask: [B, T] torch bool, True for masked positions

        Returns:
            t_en: TTNN tensor [B, C, T] like reference TextEncoder output.
        """
        B, T = input_ids.shape
        dev = self.device

        tt_ids = ttnn.from_torch(
            input_ids,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=dev,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        x = ttnn.embedding(tt_ids, self.params.embedding_weight, layout=ttnn.TILE_LAYOUT)  # [B,T,C]
        ttnn.deallocate(tt_ids)

        # Apply mask: masked positions -> 0
        # Build mask as float and multiply.
        m = ttnn.from_torch(
            (~text_mask).to(torch.float32).unsqueeze(-1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev
        )
        x = ttnn.multiply(x, m)
        ttnn.deallocate(m)

        # Conv blocks operate on NLC; conv1d expects NLC. Reference conv is on [B,C,T] so transpose first.
        # We'll transpose NLC -> NCL in torch space? Instead do layout permute using ttnn.permute and reshape.
        # Convert [B,T,C] -> [B,C,T] by permute then back to [B,T,C] for conv1d wrapper.
        # Here conv1d wrapper expects NLC, but conv weights were trained in NCL; TTNN conv1d uses NLC with weight [out,in,k]
        # so we can keep NLC.
        for i, convp in enumerate(self.params.convs):
            x = conv1d_nlc(x_nlc=x, params=convp, device=dev, compute_config=self.compute_kernel_config)
            # conv1d may return sharded tensors; normalize to interleaved for following elementwise/norm ops
            if ttnn.is_tensor_storage_on_device(x):
                try:
                    x = ttnn.sharded_to_interleaved(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                except Exception:
                    pass
            # LayerNorm over channels: use ttnn.layer_norm on last dim (C) in NLC
            x = ttnn.layer_norm(
                x,
                weight=self.params.ln_weight[i],
                bias=self.params.ln_bias[i],
                epsilon=self.params.ln_eps,
                compute_kernel_config=self.compute_kernel_config,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            x = ttnn.leaky_relu(x, negative_slope=0.2, memory_config=ttnn.DRAM_MEMORY_CONFIG)

            # re-apply mask
            m = ttnn.from_torch(
                (~text_mask).to(torch.float32).unsqueeze(-1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev
            )
            x = ttnn.multiply(x, m)
            ttnn.deallocate(m)

        # BiLSTM over NLC
        x = bilstm_nlc(
            x_nlc=x,
            fwd=self.params.lstm_fwd,
            rev=self.params.lstm_rev,
            compute_kernel_config=self.compute_kernel_config,
        )

        # Output wants [B, C, T]
        x = ttnn.permute(x, (0, 2, 1))  # [B, 2H, T]
        return x
