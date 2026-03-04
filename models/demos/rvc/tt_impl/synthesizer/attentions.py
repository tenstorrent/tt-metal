# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import torch

import ttnn
from models.demos.rvc.tt_impl.conv1d import Conv1d


class MultiHeadAttention:
    def __init__(
        self,
        *,
        device: ttnn.MeshDevice,
        channels: int,
        out_channels: int,
        n_heads: int,
        window_size: int | None = None,
        conv_config: ttnn.Conv1dConfig | None = None,
        compute_config: ttnn.DeviceComputeKernelConfig | None = None,
    ) -> None:
        if channels % n_heads != 0:
            raise ValueError("channels must be divisible by n_heads")

        self.device = device
        self.n_heads = n_heads
        self.window_size = window_size
        self.k_channels = channels // n_heads

        self.conv_q = Conv1d(
            device=device,
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
        )
        self.conv_k = Conv1d(
            device=device,
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
        )
        self.conv_v = Conv1d(
            device=device,
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
        )
        self.conv_o = Conv1d(
            device=device,
            in_channels=channels,
            out_channels=out_channels,
            kernel_size=1,
        )
        self.emb_rel_k: ttnn.Tensor | None = None
        self.emb_rel_v: ttnn.Tensor | None = None

    def load_parameters(self, parameters: dict[str, torch.Tensor], prefix: str = "") -> None:
        self.conv_q.load_parameters(parameters=parameters, key="conv_q", prefix=prefix)
        self.conv_k.load_parameters(parameters=parameters, key="conv_k", prefix=prefix)
        self.conv_v.load_parameters(parameters=parameters, key="conv_v", prefix=prefix)
        self.conv_o.load_parameters(parameters=parameters, key="conv_o", prefix=prefix)
        if self.window_size is not None:
            key_emb_rel_k = f"{prefix}emb_rel_k" if prefix else "emb_rel_k"
            key_emb_rel_v = f"{prefix}emb_rel_v" if prefix else "emb_rel_v"
            if key_emb_rel_k not in parameters:
                raise KeyError(f"Missing required parameter: {key_emb_rel_k}")
            if key_emb_rel_v not in parameters:
                raise KeyError(f"Missing required parameter: {key_emb_rel_v}")

            emb_rel_k = parameters[key_emb_rel_k].detach().reshape(1, 1, -1, self.k_channels)
            emb_rel_v = parameters[key_emb_rel_v].detach().reshape(1, 1, -1, self.k_channels)
            self.emb_rel_k = ttnn.from_torch(
                emb_rel_k,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
            )
            self.emb_rel_v = ttnn.from_torch(
                emb_rel_v,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
            )

    def __call__(self, x: ttnn.Tensor, c: ttnn.Tensor) -> ttnn.Tensor:
        q = self._project_qkv(self.conv_q(x), transpose_k=False)
        k = self._project_qkv(self.conv_k(c), transpose_k=True)
        v = self._project_qkv(self.conv_v(c), transpose_k=False)
        q_scaled = q * (1.0 / math.sqrt(self.k_channels))
        scores = ttnn.matmul(q_scaled, k)
        _, _, target_length, source_length = scores.shape
        if self.window_size is not None:
            if source_length != target_length:
                raise ValueError("Relative attention is only available for self-attention.")
            if self.emb_rel_k is None or self.emb_rel_v is None:
                raise ValueError("Relative embeddings are not loaded.")

            key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, source_length)
            rel_logits = self._matmul_with_relative_keys(q_scaled, key_relative_embeddings)
            scores_local = self._relative_position_to_absolute_position(rel_logits)
            scores = scores + scores_local

        p_attn = ttnn.softmax(scores, dim=-1)
        output = ttnn.matmul(p_attn, v)
        if self.window_size is not None:
            assert self.emb_rel_v is not None
            relative_weights = self._absolute_position_to_relative_position(p_attn)
            value_relative_embeddings = self._get_relative_embeddings(self.emb_rel_v, source_length)
            output = output + self._matmul_with_relative_values(relative_weights, value_relative_embeddings)

        output = ttnn.permute(output, (0, 2, 1, 3))
        output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)
        batch_size, length, _, _ = output.shape
        output = ttnn.reshape(output, (batch_size, length, self.n_heads * self.k_channels))

        out_tt = self.conv_o(output)
        batch_size, _, length, channels = out_tt.shape
        out_tt = ttnn.reshape(out_tt, (batch_size, length, channels))
        return out_tt

    def _project_qkv(self, x: ttnn.Tensor, *, transpose_k: bool) -> ttnn.Tensor:
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        batch_size, _, length, channels = x.shape
        x = ttnn.reshape(x, (batch_size, length, self.n_heads, self.k_channels))
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        if transpose_k:
            x_p = ttnn.permute(x, (0, 2, 3, 1))
            return x_p
        x_p = ttnn.permute(x, (0, 2, 1, 3))
        return x_p

    def _matmul_with_relative_values(self, x: ttnn.Tensor, y: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.matmul(x, y)

    def _matmul_with_relative_keys(self, x: ttnn.Tensor, y: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.matmul(x, y, transpose_b=True)

    def _get_relative_embeddings(self, relative_embeddings: ttnn.Tensor, length: int) -> ttnn.Tensor:
        if self.window_size is None:
            raise ValueError("window_size must be set for relative attention.")
        pad_length: int = max(length - (self.window_size + 1), 0)
        slice_start_position = max((self.window_size + 1) - length, 0)
        slice_end_position = slice_start_position + 2 * length - 1

        embeddings = ttnn.to_layout(relative_embeddings, ttnn.ROW_MAJOR_LAYOUT)
        if pad_length > 0:
            embeddings = ttnn.pad(
                embeddings,
                padding=((0, 0), (0, 0), (pad_length, pad_length), (0, 0)),
                value=0.0,
            )
        embeddings = ttnn.slice(
            embeddings,
            (0, 0, slice_start_position, 0),
            (1, 1, slice_end_position, self.k_channels),
        )
        return ttnn.to_layout(embeddings, ttnn.TILE_LAYOUT)

    def _relative_position_to_absolute_position(self, x: ttnn.Tensor) -> ttnn.Tensor:
        batch, heads, length, _ = x.shape
        idx_row = torch.arange(length, dtype=torch.int64).view(length, 1)
        idx_col = torch.arange(length, dtype=torch.int64).view(1, length)
        rel_idx = idx_col - idx_row + (length - 1)
        rel_idx = rel_idx.view(1, 1, length, length).expand(batch, heads, length, length).contiguous()
        rel_idx_tt = ttnn.from_torch(
            rel_idx,
            dtype=ttnn.uint32,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        x_final = ttnn.gather(x, dim=3, index=rel_idx_tt)
        return x_final

    def _absolute_position_to_relative_position(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        batch, heads, length, _ = x.shape
        idx_row = torch.arange(length, dtype=torch.int64).view(length, 1)
        idx_col = torch.arange(length, dtype=torch.int64).view(1, length)
        rel_idx = idx_col - idx_row + (length - 1)
        rel_idx = rel_idx.view(1, 1, length, length).expand(batch, heads, length, length).contiguous()
        rel_idx_tt = ttnn.from_torch(
            rel_idx,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
        )
        out = ttnn.zeros(
            (batch, heads, length, 2 * length - 1),
            dtype=x.dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
        )
        out = ttnn.scatter(out, dim=3, index=rel_idx_tt, src=x)
        return ttnn.to_layout(out, ttnn.TILE_LAYOUT)


class FFN:
    def __init__(
        self,
        *,
        device: ttnn.MeshDevice,
        in_channels: int,
        out_channels: int,
        filter_channels: int,
        kernel_size: int,
        conv_config: ttnn.Conv1dConfig | None = None,
        compute_config: ttnn.DeviceComputeKernelConfig | None = None,
    ) -> None:
        self.conv_1 = Conv1d(
            device=device,
            in_channels=in_channels,
            out_channels=filter_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.conv_2 = Conv1d(
            device=device,
            in_channels=filter_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

    def load_parameters(self, parameters: dict[str, torch.Tensor], prefix: str = "") -> None:
        self.conv_1.load_parameters(parameters=parameters, key="conv_1", prefix=prefix)
        self.conv_2.load_parameters(parameters=parameters, key="conv_2", prefix=prefix)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x0 = x
        x1 = self.conv_1(x0)
        batch, _, length, channel = x1.shape
        x1 = ttnn.reshape(x1, (batch, length, channel))
        x1 = ttnn.to_layout(x1, ttnn.TILE_LAYOUT)
        x2 = ttnn.relu(x1)
        x2 = ttnn.to_layout(x2, ttnn.ROW_MAJOR_LAYOUT)
        x3 = self.conv_2(x2)
        batch, _, length, channel = x3.shape
        x3 = ttnn.reshape(x3, (batch, length, channel))
        return x3
