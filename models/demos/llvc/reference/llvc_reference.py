# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Self-contained PyTorch reference for the LLVC voice-conversion generator.

This is a faithful port of the KoeAI LLVC generator (``Net``) defined at
https://github.com/KoeAI/LLVC/blob/main/model.py and its cached convolution
prenet from ``cached_convnet.py``. Training-only components (discriminators,
losses) are intentionally omitted, and the ``speechbrain`` positional encoding
is re-implemented inline so the reference has no third-party model deps.

The reference is used for two purposes:

1. Loading the official LLVC checkpoint (``state_dict`` keys match KoeAI's).
2. Producing golden outputs for per-module and end-to-end PCC checks against the
   TTNN implementation in ``models/demos/llvc/tt``.

LLVC is a *waveform-to-waveform* any-to-one converter: it does not use a
separate neural vocoder. The final ``ConvTranspose1d`` + ``tanh`` directly
synthesises the converted 16 kHz waveform, so "vocoder integration" for this
model reduces to this output-synthesis stage.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def mod_pad(x: Tensor, chunk_size: int, pad: tuple[int, int]) -> tuple[Tensor, int]:
    """Right-pad ``x`` so its length is a multiple of ``chunk_size``, then apply ``pad``."""
    mod = 0
    if (x.shape[-1] % chunk_size) != 0:
        mod = chunk_size - (x.shape[-1] % chunk_size)
    x = F.pad(x, (0, mod))
    x = F.pad(x, pad)
    return x, mod


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding matching ``speechbrain``'s implementation.

    Returns only the positional term (added by the caller), shaped ``[1, T, C]``.
    """

    def __init__(self, model_dim: int, max_len: int = 200):
        super().__init__()
        pe = torch.zeros(max_len, model_dim, requires_grad=False)
        positions = torch.arange(0, max_len).unsqueeze(1).float()
        denominator = torch.exp(torch.arange(0, model_dim, 2).float() * -(math.log(10000.0) / model_dim))
        pe[:, 0::2] = torch.sin(positions * denominator)
        pe[:, 1::2] = torch.cos(positions * denominator)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        return self.pe[:, : x.size(1)].clone().detach()


class LayerNormPermuted(nn.LayerNorm):
    """LayerNorm over the channel dim for ``[B, C, T]`` tensors."""

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 1)  # [B, T, C]
        x = super().forward(x)
        x = x.permute(0, 2, 1)  # [B, C, T]
        return x


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, dilation=dilation),
            LayerNormPermuted(in_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            LayerNormPermuted(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class DilatedCausalConvEncoder(nn.Module):
    """Dilated causal convolution encoder with per-layer streaming context buffers."""

    def __init__(self, channels, num_layers, kernel_size=3):
        super().__init__()
        self.channels = channels
        self.num_layers = num_layers
        self.kernel_size = kernel_size

        self.buf_lengths = [(kernel_size - 1) * 2**i for i in range(num_layers)]
        self.buf_indices = [0]
        for i in range(num_layers - 1):
            self.buf_indices.append(self.buf_indices[-1] + self.buf_lengths[i])

        _dcc_layers = OrderedDict()
        for i in range(num_layers):
            _dcc_layers.update(
                {
                    "dcc_%d"
                    % i: DepthwiseSeparableConv(channels, channels, kernel_size=3, stride=1, padding=0, dilation=2**i)
                }
            )
        self.dcc_layers = nn.Sequential(_dcc_layers)

    def init_ctx_buf(self, batch_size, device):
        return torch.zeros(
            (batch_size, self.channels, (self.kernel_size - 1) * (2**self.num_layers - 1)),
            device=device,
        )

    def forward(self, x: Tensor, ctx_buf: Tensor) -> tuple[Tensor, Tensor]:
        for i in range(self.num_layers):
            buf_start_idx = self.buf_indices[i]
            buf_end_idx = self.buf_indices[i] + self.buf_lengths[i]
            dcc_in = torch.cat((ctx_buf[..., buf_start_idx:buf_end_idx], x), dim=-1)
            ctx_buf[..., buf_start_idx:buf_end_idx] = dcc_in[..., -self.buf_lengths[i] :]
            x = x + self.dcc_layers[i](dcc_in)
        return x, ctx_buf


class CausalTransformerDecoderLayer(nn.TransformerDecoderLayer):
    def forward(self, tgt: Tensor, memory: Optional[Tensor] = None, chunk_size: int = 1):
        tgt_last_tok = tgt[:, -chunk_size:, :]

        tmp_tgt, sa_map = self.self_attn(tgt_last_tok, tgt, tgt, attn_mask=None, key_padding_mask=None)
        tgt_last_tok = tgt_last_tok + self.dropout1(tmp_tgt)
        tgt_last_tok = self.norm1(tgt_last_tok)

        ca_map = None
        if memory is not None:
            tmp_tgt, ca_map = self.multihead_attn(tgt_last_tok, memory, memory, attn_mask=None, key_padding_mask=None)
            tgt_last_tok = tgt_last_tok + self.dropout2(tmp_tgt)
            tgt_last_tok = self.norm2(tgt_last_tok)

        tmp_tgt = self.linear2(self.dropout(self.activation(self.linear1(tgt_last_tok))))
        tgt_last_tok = tgt_last_tok + self.dropout3(tmp_tgt)
        tgt_last_tok = self.norm3(tgt_last_tok)
        return tgt_last_tok, sa_map, ca_map


class CausalTransformerDecoder(nn.Module):
    """Causal transformer decoder over fixed ``ctx_len`` past chunks."""

    def __init__(self, model_dim, ctx_len, chunk_size, num_layers, nhead, use_pos_enc, ff_dim, dropout):
        super().__init__()
        self.num_layers = num_layers
        self.model_dim = model_dim
        self.ctx_len = ctx_len
        self.chunk_size = chunk_size
        self.nhead = nhead
        self.use_pos_enc = use_pos_enc
        self.unfold = nn.Unfold(kernel_size=(ctx_len + chunk_size, 1), stride=chunk_size)
        self.pos_enc = SinusoidalPositionalEncoding(model_dim, max_len=200)
        self.tf_dec_layers = nn.ModuleList(
            [
                CausalTransformerDecoderLayer(
                    d_model=model_dim, nhead=nhead, dim_feedforward=ff_dim, batch_first=True, dropout=dropout
                )
                for _ in range(num_layers)
            ]
        )

    def init_ctx_buf(self, batch_size, device):
        return torch.zeros((batch_size, self.num_layers + 1, self.ctx_len, self.model_dim), device=device)

    def _causal_unfold(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        x = x.permute(0, 2, 1)
        x = self.unfold(x.unsqueeze(-1))
        x = x.permute(0, 2, 1)
        x = x.reshape(B, -1, C, self.ctx_len + self.chunk_size)
        x = x.reshape(-1, C, self.ctx_len + self.chunk_size)
        x = x.permute(0, 2, 1)
        return x

    def forward(self, tgt: Tensor, mem: Tensor, ctx_buf: Tensor, probe: bool = False):
        mem, _ = mod_pad(mem, self.chunk_size, (0, 0))
        tgt, mod = mod_pad(tgt, self.chunk_size, (0, 0))

        B, C, T = tgt.shape
        tgt = tgt.permute(0, 2, 1)
        mem = mem.permute(0, 2, 1)

        mem = torch.cat((ctx_buf[:, 0, :, :], mem), dim=1)
        ctx_buf[:, 0, :, :] = mem[:, -self.ctx_len :, :]
        mem_ctx = self._causal_unfold(mem)
        if self.use_pos_enc:
            mem_ctx = mem_ctx + self.pos_enc(mem_ctx)

        K = 1000
        for i, tf_dec_layer in enumerate(self.tf_dec_layers):
            tgt = torch.cat((ctx_buf[:, i + 1, :, :], tgt), dim=1)
            ctx_buf[:, i + 1, :, :] = tgt[:, -self.ctx_len :, :]

            tgt_ctx = self._causal_unfold(tgt)
            if self.use_pos_enc and i == 0:
                tgt_ctx = tgt_ctx + self.pos_enc(tgt_ctx)
            tgt = torch.zeros_like(tgt_ctx)[:, -self.chunk_size :, :]
            for j in range(int(math.ceil(tgt.shape[0] / K))):
                tgt[j * K : (j + 1) * K], _sa, _ca = tf_dec_layer(
                    tgt_ctx[j * K : (j + 1) * K], mem_ctx[j * K : (j + 1) * K], self.chunk_size
                )
            tgt = tgt.reshape(B, T, C)

        tgt = tgt.permute(0, 2, 1)
        if mod != 0:
            tgt = tgt[..., :-mod]
        return tgt, ctx_buf


class MaskNet(nn.Module):
    def __init__(
        self,
        enc_dim,
        num_enc_layers,
        dec_dim,
        dec_buf_len,
        dec_chunk_size,
        num_dec_layers,
        use_pos_enc,
        skip_connection,
        proj,
        decoder_dropout,
    ):
        super().__init__()
        self.skip_connection = skip_connection
        self.proj = proj

        self.encoder = DilatedCausalConvEncoder(channels=enc_dim, num_layers=num_enc_layers)

        self.proj_e2d_e = nn.Sequential(
            nn.Conv1d(enc_dim, dec_dim, kernel_size=1, stride=1, padding=0, groups=dec_dim), nn.ReLU()
        )
        self.proj_e2d_l = nn.Sequential(
            nn.Conv1d(enc_dim, dec_dim, kernel_size=1, stride=1, padding=0, groups=dec_dim), nn.ReLU()
        )
        self.proj_d2e = nn.Sequential(
            nn.Conv1d(dec_dim, enc_dim, kernel_size=1, stride=1, padding=0, groups=dec_dim), nn.ReLU()
        )

        self.decoder = CausalTransformerDecoder(
            model_dim=dec_dim,
            ctx_len=dec_buf_len,
            chunk_size=dec_chunk_size,
            num_layers=num_dec_layers,
            nhead=8,
            use_pos_enc=use_pos_enc,
            ff_dim=2 * dec_dim,
            dropout=decoder_dropout,
        )

    def forward(self, x, l, enc_buf, dec_buf):
        e, enc_buf = self.encoder(x, enc_buf)
        l = l.unsqueeze(2) * e

        if self.proj:
            e = self.proj_e2d_e(e)
            m = self.proj_e2d_l(l)
            m, dec_buf = self.decoder(m, e, dec_buf)
        else:
            m, dec_buf = self.decoder(l, e, dec_buf)

        if self.proj:
            m = self.proj_d2e(m)

        if self.skip_connection:
            m = l + m
        return m, enc_buf, dec_buf


class ResidualBlock(nn.Module):
    """Gated (tanh * sigmoid) residual conv block used by the cached prenet."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout, use_2d):
        super().__init__()
        self.use_2d = use_2d
        self.filter = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.gate = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.dropout = nn.Dropout1d(dropout)
        self.output_crop = dilation * (kernel_size - 1)

    def forward(self, x: Tensor) -> Tensor:
        filtered = torch.tanh(self.filter(x))
        gated = torch.sigmoid(self.gate(x))
        residual = filtered * gated
        x = F.pad(x, (0, 0, 0, residual.shape[1] - x.shape[1]))
        output = x[..., self.output_crop :] + residual
        output = self.dropout(output)
        return output


class CachedConvNet(nn.Module):
    """Streaming causal conv stack with per-layer context buffers."""

    def __init__(
        self,
        num_channels,
        kernel_sizes,
        dilations,
        dropout,
        combine_residuals,
        use_residual_blocks,
        out_channels,
        use_2d,
    ):
        super().__init__()
        assert len(kernel_sizes) == len(dilations)
        assert len(kernel_sizes) == len(out_channels)
        self.num_layers = len(kernel_sizes)
        self.ctx_height = max(out_channels)
        self.down_convs = nn.ModuleList()
        self.num_channels = num_channels
        self.kernel_sizes = kernel_sizes
        self.combine_residuals = combine_residuals
        self.use_2d = use_2d

        self.buf_lengths = [(k - 1) * d for k, d in zip(kernel_sizes, dilations)]
        self.buf_indices = [0]
        for i in range(len(kernel_sizes) - 1):
            self.buf_indices.append(self.buf_indices[-1] + self.buf_lengths[i])

        block = ResidualBlock if use_residual_blocks else None
        assert block is not None, "This reference only supports use_residual_blocks=True (LLVC default)."

        for i in range(self.num_layers):
            in_channel = num_channels if i == 0 else out_channels[i - 1]
            self.down_convs.append(
                block(
                    in_channels=in_channel,
                    out_channels=out_channels[i],
                    kernel_size=kernel_sizes[i],
                    dilation=dilations[i],
                    dropout=dropout,
                    use_2d=use_2d,
                )
            )

    def init_ctx_buf(self, batch_size, device):
        return torch.zeros((batch_size, self.ctx_height, sum(self.buf_lengths))).to(device)

    def forward(self, x: Tensor, ctx: Tensor) -> tuple[Tensor, Tensor]:
        for i in range(self.num_layers):
            buf_start_idx = self.buf_indices[i]
            buf_end_idx = self.buf_indices[i] + self.buf_lengths[i]
            conv_in = torch.cat((ctx[..., : x.shape[-2], buf_start_idx:buf_end_idx], x), dim=-1)
            ctx[..., : x.shape[1], buf_start_idx:buf_end_idx] = conv_in[..., -self.buf_lengths[i] :]

            if self.combine_residuals == "add":
                x = x + self.down_convs[i](conv_in)
            elif self.combine_residuals == "multiply":
                x = x * self.down_convs[i](conv_in)
            else:
                x = self.down_convs[i](conv_in)
        return x, ctx


class Net(nn.Module):
    """LLVC generator. Weight-compatible with the official KoeAI checkpoint."""

    def __init__(
        self,
        label_len,
        L=8,
        enc_dim=512,
        num_enc_layers=10,
        dec_dim=256,
        dec_buf_len=100,
        num_dec_layers=2,
        dec_chunk_size=72,
        out_buf_len=2,
        use_pos_enc=True,
        skip_connection=True,
        proj=True,
        lookahead=True,
        decoder_dropout=0.0,
        convnet_config=None,
    ):
        super().__init__()
        self.L = L
        self.dec_chunk_size = dec_chunk_size
        self.out_buf_len = out_buf_len
        self.enc_dim = enc_dim
        self.lookahead = lookahead

        self.convnet_config = convnet_config
        if convnet_config is not None and convnet_config["convnet_prenet"]:
            self.convnet_pre = CachedConvNet(
                1,
                convnet_config["kernel_sizes"],
                convnet_config["dilations"],
                convnet_config["dropout"],
                convnet_config["combine_residuals"],
                convnet_config["use_residual_blocks"],
                convnet_config["out_channels"],
                use_2d=False,
            )

        kernel_size = 3 * L if lookahead else L
        self.in_conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=enc_dim, kernel_size=kernel_size, stride=L, padding=0, bias=False),
            nn.ReLU(),
        )

        label_len = 1
        self.label_embedding = nn.Sequential(
            nn.Linear(label_len, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, enc_dim),
            nn.LayerNorm(enc_dim),
            nn.ReLU(),
        )

        self.mask_gen = MaskNet(
            enc_dim=enc_dim,
            num_enc_layers=num_enc_layers,
            dec_dim=dec_dim,
            dec_buf_len=dec_buf_len,
            dec_chunk_size=dec_chunk_size,
            num_dec_layers=num_dec_layers,
            use_pos_enc=use_pos_enc,
            skip_connection=skip_connection,
            proj=proj,
            decoder_dropout=decoder_dropout,
        )

        self.out_conv = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=enc_dim,
                out_channels=1,
                kernel_size=(out_buf_len + 1) * L,
                stride=L,
                padding=out_buf_len * L,
                bias=False,
            ),
            nn.Tanh(),
        )

    def init_buffers(self, batch_size, device):
        enc_buf = self.mask_gen.encoder.init_ctx_buf(batch_size, device)
        dec_buf = self.mask_gen.decoder.init_ctx_buf(batch_size, device)
        out_buf = torch.zeros(batch_size, self.enc_dim, self.out_buf_len, device=device)
        return enc_buf, dec_buf, out_buf

    def forward(
        self,
        x,
        init_enc_buf=None,
        init_dec_buf=None,
        init_out_buf=None,
        convnet_pre_ctx=None,
        pad=True,
    ):
        label = torch.zeros(x.shape[0], 1, device=x.device)
        mod = 0
        if pad:
            pad_size = (self.L, self.L) if self.lookahead else (0, 0)
            x, mod = mod_pad(x, chunk_size=self.L, pad=pad_size)

        if hasattr(self, "convnet_pre"):
            if convnet_pre_ctx is None:
                convnet_pre_ctx = self.convnet_pre.init_ctx_buf(x.shape[0], x.device)
            convnet_out, convnet_pre_ctx = self.convnet_pre(x, convnet_pre_ctx)
            if self.convnet_config["skip_connection"] == "add":
                x = x + convnet_out
            elif self.convnet_config["skip_connection"] == "multiply":
                x = x * convnet_out
            else:
                x = convnet_out

        if init_enc_buf is None or init_dec_buf is None or init_out_buf is None:
            assert init_enc_buf is None and init_dec_buf is None and init_out_buf is None
            enc_buf, dec_buf, out_buf = self.init_buffers(x.shape[0], x.device)
        else:
            enc_buf, dec_buf, out_buf = init_enc_buf, init_dec_buf, init_out_buf

        x = self.in_conv(x)
        l = self.label_embedding(label)
        m, enc_buf, dec_buf = self.mask_gen(x, l, enc_buf, dec_buf)

        x = x * m
        x = torch.cat((out_buf, x), dim=-1)
        out_buf = x[..., -self.out_buf_len :]
        x = self.out_conv(x)

        if mod != 0:
            x = x[:, :, :-mod]

        if init_enc_buf is None:
            return x
        return x, enc_buf, dec_buf, out_buf, convnet_pre_ctx


DEFAULT_MODEL_PARAMS = dict(
    label_len=1,
    L=16,
    enc_dim=512,
    num_enc_layers=8,
    dec_dim=256,
    num_dec_layers=1,
    dec_buf_len=13,
    dec_chunk_size=13,
    out_buf_len=4,
    use_pos_enc=True,
    decoder_dropout=0.1,
    convnet_config=dict(
        convnet_prenet=True,
        out_channels=[1] * 12,
        kernel_sizes=[3] * 12,
        dilations=[1] * 12,
        dropout=0.5,
        combine_residuals=None,
        skip_connection="add",
        use_residual_blocks=True,
    ),
)


def build_reference_model(model_params: dict | None = None) -> Net:
    """Instantiate the reference generator with the official LLVC config by default."""
    params = dict(DEFAULT_MODEL_PARAMS if model_params is None else model_params)
    model = Net(**params)
    model.eval()
    return model
