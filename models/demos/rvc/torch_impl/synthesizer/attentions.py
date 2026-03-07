# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import math

import torch
from torch import nn
from torch.nn import functional as F

from models.demos.rvc.torch_impl.synthesizer.modules import LayerNorm
from models.demos.rvc.torch_impl.utils import linear_channel_first


class Encoder(nn.Module):
    def __init__(
        self,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size=1,
        window_size=10,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_layers = int(n_layers)
        self.kernel_size = kernel_size
        self.window_size = window_size

        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()
        for _ in range(self.n_layers):
            self.attn_layers.append(
                MultiHeadAttention(
                    hidden_channels,
                    hidden_channels,
                    n_heads,
                    window_size=window_size,
                )
            )
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                )
            )
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    def forward(self, x):
        zippep = zip(self.attn_layers, self.norm_layers_1, self.ffn_layers, self.norm_layers_2, strict=True)
        for attn_layers, norm_layers_1, ffn_layers, norm_layers_2 in zippep:
            y = attn_layers(x, x)
            x = norm_layers_1(x + y)

            y = ffn_layers(x)
            x = norm_layers_2(x + y)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels,
        out_channels,
        n_heads,
        window_size=None,
    ):
        super().__init__()
        assert channels % n_heads == 0

        self.n_heads = n_heads
        self.window_size = window_size

        self.k_channels = channels // n_heads
        self.linear_q = nn.Linear(channels, channels)
        self.linear_k = nn.Linear(channels, channels)
        self.linear_v = nn.Linear(channels, channels)
        self.linear_o = nn.Linear(channels, out_channels)

        if window_size is not None:
            n_heads_rel = 1
            rel_stddev = self.k_channels**-0.5
            self.emb_rel_k = nn.Parameter(torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev)
            self.emb_rel_v = nn.Parameter(torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev)

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        q = linear_channel_first(x, self.linear_q)
        k = linear_channel_first(c, self.linear_k)
        v = linear_channel_first(c, self.linear_v)

        x = self.attention(q, k, v)

        x = linear_channel_first(x, self.linear_o)
        return x

    def attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ):
        # reshape [b, d, t] -> [b, n_h, t, d_k]
        b, d, t_s = key.size()
        t_t = query.size(2)
        query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
        key = key.view(b, self.n_heads, self.k_channels, t_s)
        value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)

        scores = torch.matmul(query / math.sqrt(self.k_channels), key)
        if self.window_size is not None:
            assert t_s == t_t, "Relative attention is only available for self-attention."
            key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, t_s)
            rel_logits = self._matmul_with_relative_keys(query / math.sqrt(self.k_channels), key_relative_embeddings)
            scores_local = self._relative_position_to_absolute_position(rel_logits)
            scores = scores + scores_local
        p_attn = F.softmax(scores, dim=-1)  # [b, n_h, t_t, t_s]
        output = torch.matmul(p_attn, value)
        if self.window_size is not None:
            relative_weights = self._absolute_position_to_relative_position(p_attn)
            value_relative_embeddings = self._get_relative_embeddings(self.emb_rel_v, t_s)
            output = output + self._matmul_with_relative_values(relative_weights, value_relative_embeddings)
        output = output.transpose(2, 3).contiguous().view(b, d, t_t)  # [b, n_h, t_t, d_k] -> [b, d, t_t]
        return output

    def _matmul_with_relative_values(self, x, y):
        """
        x: [b, h, l, m]
        y: [h or 1, m, d]
        ret: [b, h, l, d]
        """
        ret = torch.matmul(x, y.unsqueeze(0))
        return ret

    def _matmul_with_relative_keys(self, x, y):
        """
        x: [b, h, l, d]
        y: [h or 1, m, d]
        ret: [b, h, l, m]
        """
        ret = torch.matmul(x, y.unsqueeze(0).transpose(-2, -1))
        return ret

    def _get_relative_embeddings(self, relative_embeddings, length: int):
        # Pad first before slice to avoid using cond ops.
        pad_length: int = max(length - (self.window_size + 1), 0)
        slice_start_position = max((self.window_size + 1) - length, 0)
        slice_end_position = slice_start_position + 2 * length - 1
        if pad_length > 0:
            padded_relative_embeddings = F.pad(
                relative_embeddings,
                [0, 0, pad_length, pad_length, 0, 0],
            )
        else:
            padded_relative_embeddings = relative_embeddings
        used_relative_embeddings = padded_relative_embeddings[:, slice_start_position:slice_end_position]
        return used_relative_embeddings

    def _relative_position_to_absolute_position(self, x):
        """
        x: [b, h, l, 2*l-1]
        ret: [b, h, l, l]
        """
        _, _, length, _ = x.size()
        device = x.device
        idx_row = torch.arange(length, device=device).view(length, 1)
        idx_col = torch.arange(length, device=device).view(1, length)
        rel_idx = idx_col - idx_row + (length - 1)  # [l, l], in [0, 2*l-2]
        rel_idx = rel_idx.view(1, 1, length, length).expand(x.size(0), x.size(1), length, length)
        x_final = x.gather(dim=3, index=rel_idx)
        return x_final

    def _absolute_position_to_relative_position(self, x):
        """
        x: [b, h, l, l]
        ret: [b, h, l, 2*l-1]
        """
        batch, heads, length, _ = x.size()
        device = x.device
        idx_row = torch.arange(length, device=device).view(length, 1)
        idx_col = torch.arange(length, device=device).view(1, length)
        rel_idx = idx_col - idx_row + (length - 1)  # [l, l], in [0, 2*l-2]
        rel_idx = rel_idx.view(1, 1, length, length).expand(batch, heads, length, length)
        out = x.new_zeros(batch, heads, length, 2 * length - 1)
        out.scatter_(dim=3, index=rel_idx, src=x)
        return out


class FFN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        filter_channels,
        kernel_size,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size

        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding="same")
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size, padding="same")

    def forward(self, x: torch.Tensor):
        x = self.conv_1(x)
        x = torch.relu(x)

        x = self.conv_2(x)
        return x
