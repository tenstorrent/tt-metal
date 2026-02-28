# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Reference PyTorch implementation of LLVC (Low-Latency Low-Resource Voice Conversion).
Based on KoeAI/LLVC (MIT License): https://github.com/KoeAI/LLVC

This module contains the reference model used for comparing accuracy with the TTNN implementation.
"""

import math
from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def mod_pad(x, chunk_size, pad):
    """Mod pad the input to perform integer number of inferences."""
    mod = 0
    if (x.shape[-1] % chunk_size) != 0:
        mod = chunk_size - (x.shape[-1] % chunk_size)

    x = F.pad(x, (0, mod))
    x = F.pad(x, pad)
    return x, mod


class LayerNormPermuted(nn.LayerNorm):
    """LayerNorm that operates on channels-first format [B, C, T]."""

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [B, T, C]
        x = super().forward(x)
        x = x.permute(0, 2, 1)  # [B, C, T]
        return x


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolutions."""

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

    def forward(self, x):
        return self.layers(x)


class DilatedCausalConvEncoder(nn.Module):
    """
    A dilated causal convolution based encoder for encoding
    time domain audio input into latent space.
    """

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
            dcc_layer = DepthwiseSeparableConv(channels, channels, kernel_size=3, stride=1, padding=0, dilation=2**i)
            _dcc_layers.update({"dcc_%d" % i: dcc_layer})
        self.dcc_layers = nn.Sequential(_dcc_layers)

    def init_ctx_buf(self, batch_size, device):
        return torch.zeros(
            (batch_size, self.channels, (self.kernel_size - 1) * (2**self.num_layers - 1)),
            device=device,
        )

    def forward(self, x, ctx_buf):
        x.shape[-1]
        for i in range(self.num_layers):
            buf_start_idx = self.buf_indices[i]
            buf_end_idx = self.buf_indices[i] + self.buf_lengths[i]

            dcc_in = torch.cat((ctx_buf[..., buf_start_idx:buf_end_idx], x), dim=-1)
            ctx_buf[..., buf_start_idx:buf_end_idx] = dcc_in[..., -self.buf_lengths[i] :]
            x = x + self.dcc_layers[i](dcc_in)

        return x, ctx_buf


class CausalTransformerDecoderLayer(torch.nn.TransformerDecoderLayer):
    """
    Adapted from causal-transformer-decoder.
    """

    def forward(self, tgt: Tensor, memory: Optional[Tensor] = None, chunk_size: int = 1) -> Tensor:
        tgt_last_tok = tgt[:, -chunk_size:, :]

        # self attention
        tmp_tgt, sa_map = self.self_attn(tgt_last_tok, tgt, tgt, attn_mask=None, key_padding_mask=None)
        tgt_last_tok = tgt_last_tok + self.dropout1(tmp_tgt)
        tgt_last_tok = self.norm1(tgt_last_tok)

        # encoder-decoder attention
        if memory is not None:
            tmp_tgt, ca_map = self.multihead_attn(tgt_last_tok, memory, memory, attn_mask=None, key_padding_mask=None)
            tgt_last_tok = tgt_last_tok + self.dropout2(tmp_tgt)
            tgt_last_tok = self.norm2(tgt_last_tok)
        else:
            ca_map = None

        # feed-forward network
        tmp_tgt = self.linear2(self.dropout(self.activation(self.linear1(tgt_last_tok))))
        tgt_last_tok = tgt_last_tok + self.dropout3(tmp_tgt)
        tgt_last_tok = self.norm3(tgt_last_tok)
        return tgt_last_tok, sa_map, ca_map


class CausalTransformerDecoder(nn.Module):
    """
    A causal transformer decoder which decodes input vectors using
    precisely `ctx_len` past vectors in the sequence.
    """

    def __init__(self, model_dim, ctx_len, chunk_size, num_layers, nhead, use_pos_enc, ff_dim, dropout):
        super().__init__()
        self.num_layers = num_layers
        self.model_dim = model_dim
        self.ctx_len = ctx_len
        self.chunk_size = chunk_size
        self.nhead = nhead
        self.use_pos_enc = use_pos_enc
        self.unfold = nn.Unfold(kernel_size=(ctx_len + chunk_size, 1), stride=chunk_size)
        self.pos_enc = _PositionalEncoding(model_dim, max_len=200)
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

    def _causal_unfold(self, x):
        B, T, C = x.shape
        x = x.permute(0, 2, 1)
        x = self.unfold(x.unsqueeze(-1))
        x = x.permute(0, 2, 1)
        x = x.reshape(B, -1, C, self.ctx_len + self.chunk_size)
        x = x.reshape(-1, C, self.ctx_len + self.chunk_size)
        x = x.permute(0, 2, 1)
        return x

    def forward(self, tgt, mem, ctx_buf, probe=False):
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
                tgt[j * K : (j + 1) * K], _sa_map, _ca_map = tf_dec_layer(
                    tgt_ctx[j * K : (j + 1) * K], mem_ctx[j * K : (j + 1) * K], self.chunk_size
                )
            tgt = tgt.reshape(B, T, C)

        tgt = tgt.permute(0, 2, 1)
        if mod != 0:
            tgt = tgt[..., :-mod]

        return tgt, ctx_buf


class MaskNet(nn.Module):
    """Mask generation network combining encoder and decoder."""

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

    def forward(self, x, label_emb, enc_buf, dec_buf):
        e, enc_buf = self.encoder(x, enc_buf)
        label_emb = label_emb.unsqueeze(2) * e

        if self.proj:
            e = self.proj_e2d_e(e)
            m = self.proj_e2d_l(label_emb)
            m, dec_buf = self.decoder(m, e, dec_buf)
        else:
            m, dec_buf = self.decoder(label_emb, e, dec_buf)

        if self.proj:
            m = self.proj_d2e(m)

        if self.skip_connection:
            m = label_emb + m

        return m, enc_buf, dec_buf


class ResidualBlock(nn.Module):
    """Residual block for CachedConvNet."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout, use_2d):
        super().__init__()
        self.use_2d = use_2d
        if use_2d:
            self.filter = nn.Conv2d(in_channels, out_channels, kernel_size, dilation=dilation)
            self.gate = nn.Conv2d(in_channels, out_channels, kernel_size, dilation=dilation)
            self.dropout = nn.Dropout2d(dropout)
        else:
            self.filter = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
            self.gate = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
            self.dropout = nn.Dropout1d(dropout)
        self.output_crop = dilation * (kernel_size - 1)

    def forward(self, x):
        filtered = torch.tanh(self.filter(x))
        gated = torch.sigmoid(self.gate(x))
        residual = filtered * gated
        if self.use_2d:
            x = F.pad(x, (0, 0, 0, 0, 0, residual.shape[1] - x.shape[1]))
            output = x[..., self.output_crop :, self.output_crop :] + residual
        else:
            x = F.pad(x, (0, 0, 0, residual.shape[1] - x.shape[1]))
            output = x[..., self.output_crop :] + residual
        output = self.dropout(output)
        return output


class CausalConvBlock(nn.Module):
    """Simple causal convolution block."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout, use_2d):
        super().__init__()
        if use_2d:
            conv_layer = nn.Conv2d
            batchnorm_layer = nn.BatchNorm2d
            dropout_layer = nn.Dropout2d
        else:
            conv_layer = nn.Conv1d
            batchnorm_layer = nn.BatchNorm1d
            dropout_layer = nn.Dropout1d
        self.conv = nn.Sequential(
            conv_layer(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, dilation=dilation),
            batchnorm_layer(num_features=out_channels),
            dropout_layer(dropout),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class CachedConvNet(nn.Module):
    """Cached convolutional network for streaming pre-processing."""

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
        use_pool=False,
        pool_kernel=2,
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
        self.use_pool = use_pool

        self.buf_lengths = [(k - 1) * d for k, d in zip(kernel_sizes, dilations)]
        self.buf_indices = [0]
        for i in range(len(kernel_sizes) - 1):
            self.buf_indices.append(self.buf_indices[-1] + self.buf_lengths[i])

        if use_residual_blocks:
            block = ResidualBlock
        else:
            block = CausalConvBlock

        if self.use_pool:
            self.pool = nn.AvgPool1d(kernel_size=pool_kernel)

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

    def init_ctx_buf(self, batch_size, device, height=None):
        if height is not None:
            up_ctx = torch.zeros((batch_size, self.ctx_height, height, sum(self.buf_lengths))).to(device)
        else:
            up_ctx = torch.zeros((batch_size, self.ctx_height, sum(self.buf_lengths))).to(device)
        return up_ctx

    def forward(self, x, ctx):
        if self.use_pool:
            x = self.pool(x)

        for i in range(self.num_layers):
            buf_start_idx = self.buf_indices[i]
            buf_end_idx = self.buf_indices[i] + self.buf_lengths[i]

            if self.use_2d:
                conv_in = torch.cat((ctx[..., : x.shape[1], : x.shape[-2], buf_start_idx:buf_end_idx], x), dim=-1)
            else:
                conv_in = torch.cat((ctx[..., : x.shape[1], buf_start_idx:buf_end_idx], x), dim=-1)

            if self.use_2d:
                ctx[..., : x.shape[1], : x.shape[-2], buf_start_idx:buf_end_idx] = conv_in[..., -self.buf_lengths[i] :]
            else:
                ctx[..., : x.shape[1], buf_start_idx:buf_end_idx] = conv_in[..., -self.buf_lengths[i] :]

            if self.use_2d:
                conv_in = F.pad(conv_in, (0, 0, self.buf_lengths[i] // 2, self.buf_lengths[i] // 2))

            if self.combine_residuals == "add":
                x = x + self.down_convs[i](conv_in)
            elif self.combine_residuals == "multiply":
                x = x * self.down_convs[i](conv_in)
            else:
                x = self.down_convs[i](conv_in)

        if self.use_pool:
            x = F.interpolate(x, scale_factor=self.pool.kernel_size[0])

        return x, ctx


class Net(nn.Module):
    """
    LLVC Network - main model combining all components for voice conversion.
    """

    def __init__(
        self,
        label_len,
        L=16,
        enc_dim=512,
        num_enc_layers=8,
        dec_dim=256,
        dec_buf_len=13,
        num_dec_layers=1,
        dec_chunk_size=13,
        out_buf_len=4,
        use_pos_enc=True,
        skip_connection=True,
        proj=True,
        lookahead=True,
        decoder_dropout=0.1,
        convnet_config=None,
    ):
        super().__init__()
        self.L = L
        self.dec_chunk_size = dec_chunk_size
        self.out_buf_len = out_buf_len
        self.enc_dim = enc_dim
        self.lookahead = lookahead

        self.convnet_config = convnet_config
        if convnet_config is not None and convnet_config.get("convnet_prenet", False):
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

    def forward(self, x, init_enc_buf=None, init_dec_buf=None, init_out_buf=None, convnet_pre_ctx=None, pad=True):
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
        label_emb = self.label_embedding(label)
        m, enc_buf, dec_buf = self.mask_gen(x, label_emb, enc_buf, dec_buf)

        x = x * m
        x = torch.cat((out_buf, x), dim=-1)
        out_buf = x[..., -self.out_buf_len :]
        x = self.out_conv(x)

        if mod != 0:
            x = x[:, :, :-mod]

        if init_enc_buf is None:
            return x
        else:
            return x, enc_buf, dec_buf, out_buf, convnet_pre_ctx


class _PositionalEncoding(nn.Module):
    """Simple sinusoidal positional encoding (no external dependency)."""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1), :]


def get_default_config():
    """Returns the default LLVC model configuration."""
    return {
        "label_len": 1,
        "L": 16,
        "enc_dim": 512,
        "num_enc_layers": 8,
        "dec_dim": 256,
        "num_dec_layers": 1,
        "dec_buf_len": 13,
        "dec_chunk_size": 13,
        "out_buf_len": 4,
        "use_pos_enc": True,
        "decoder_dropout": 0.1,
        "convnet_config": {
            "convnet_prenet": True,
            "out_channels": [1] * 12,
            "kernel_sizes": [3] * 12,
            "dilations": [1] * 12,
            "dropout": 0.5,
            "combine_residuals": None,
            "skip_connection": "add",
            "use_residual_blocks": True,
        },
    }
