# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import torch

import ttnn
from models.demos.rvc.tt_impl.conv1d import Conv1d
from models.demos.rvc.tt_impl.linear import Linear


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

        self.linear_q = Linear(
            device=device,
            in_features=channels,
            out_features=channels,
        )
        self.linear_k = Linear(
            device=device,
            in_features=channels,
            out_features=channels,
        )
        self.linear_v = Linear(
            device=device,
            in_features=channels,
            out_features=channels,
        )
        self.linear_o = Linear(
            device=device,
            in_features=channels,
            out_features=out_channels,
        )
        self.emb_rel_k: ttnn.Tensor | None = None
        self.emb_rel_v: ttnn.Tensor | None = None

    def load_parameters(self, parameters: dict[str, torch.Tensor], prefix: str = "") -> None:
        q_key = "linear_q" if (f"{prefix}linear_q.weight" if prefix else "linear_q.weight") in parameters else "conv_q"
        k_key = "linear_k" if (f"{prefix}linear_k.weight" if prefix else "linear_k.weight") in parameters else "conv_k"
        v_key = "linear_v" if (f"{prefix}linear_v.weight" if prefix else "linear_v.weight") in parameters else "conv_v"
        o_key = "linear_o" if (f"{prefix}linear_o.weight" if prefix else "linear_o.weight") in parameters else "conv_o"
        self.linear_q.load_parameters(parameters=parameters, key=q_key, prefix=prefix)
        self.linear_k.load_parameters(parameters=parameters, key=k_key, prefix=prefix)
        self.linear_v.load_parameters(parameters=parameters, key=v_key, prefix=prefix)
        self.linear_o.load_parameters(parameters=parameters, key=o_key, prefix=prefix)
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
        q = self._project_qkv(self.linear_q(x), transpose_k=False)
        k = self._project_qkv(self.linear_k(c), transpose_k=True)
        v = self._project_qkv(self.linear_v(c), transpose_k=False)
        q_scaled = q * (1.0 / math.sqrt(self.k_channels))
        scores = ttnn.matmul(q_scaled, k)
        _, _, target_length, source_length = scores.shape
        if self.window_size is not None:
            if source_length != target_length:
                raise ValueError("Relative attention is only available for self-attention.")
            if self.emb_rel_k is None or self.emb_rel_v is None:
                raise ValueError("Relative embeddings are not loaded.")

            key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, source_length)
            rel_logits = ttnn.matmul(q_scaled, key_relative_embeddings, transpose_b=True)
            scores_local = self._relative_position_to_absolute_position(rel_logits)
            scores = scores + scores_local

        p_attn = ttnn.softmax(scores, dim=-1)
        output = ttnn.matmul(p_attn, v)
        if self.window_size is not None:
            assert self.emb_rel_v is not None
            relative_weights = self._absolute_position_to_relative_position(p_attn)
            value_relative_embeddings = self._get_relative_embeddings(self.emb_rel_v, source_length)
            output = output + ttnn.matmul(relative_weights, value_relative_embeddings)

        output = ttnn.transformer.concatenate_heads(output)

        out_tt = self.linear_o(output)
        return out_tt

    def _project_qkv(self, x: ttnn.Tensor, *, transpose_k: bool) -> ttnn.Tensor:
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        batch_size, length, channels = x.shape
        x = ttnn.reshape(x, (batch_size, length, self.n_heads, self.k_channels))
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        if transpose_k:
            x_p = ttnn.permute(x, (0, 2, 3, 1))
            return x_p
        x_p = ttnn.permute(x, (0, 2, 1, 3))
        return x_p

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
        idx_row = ttnn.unsqueeze(ttnn.arange(start=0, end=length, dtype=ttnn.int32, device=self.device), dim=1)
        idx_col = ttnn.unsqueeze(ttnn.arange(start=0, end=length, dtype=ttnn.int32, device=self.device), dim=0)
        rel_idx = idx_col - idx_row + (length - 1)
        rel_idx = ttnn.expand(ttnn.reshape(rel_idx, shape=(1, 1, length, length)), (batch, heads, length, length))
        rel_idx = ttnn.typecast(ttnn.to_layout(rel_idx, ttnn.TILE_LAYOUT), ttnn.uint32)
        x_final = ttnn.gather(x, dim=3, index=rel_idx)
        return x_final

    def _absolute_position_to_relative_position(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        batch, heads, length, _ = x.shape
        idx_row = ttnn.unsqueeze(ttnn.arange(start=0, end=length, dtype=ttnn.int32, device=self.device), dim=1)
        idx_col = ttnn.unsqueeze(ttnn.arange(start=0, end=length, dtype=ttnn.int32, device=self.device), dim=0)
        rel_idx = idx_col - idx_row + (length - 1)
        rel_idx = ttnn.expand(ttnn.reshape(rel_idx, shape=(1, 1, length, length)), (batch, heads, length, length))
        out = ttnn.zeros(
            (batch, heads, length, 2 * length - 1),
            dtype=x.dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
        )
        out = ttnn.scatter(out, dim=3, index=rel_idx, src=x)
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
            padding="same",
            activation="relu",
        )
        self.conv_2 = Conv1d(
            device=device,
            in_channels=filter_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding="same",
        )

    def load_parameters(self, parameters: dict[str, torch.Tensor], prefix: str = "") -> None:
        self.conv_1.load_parameters(parameters=parameters, key="conv_1", prefix=prefix)
        self.conv_2.load_parameters(parameters=parameters, key="conv_2", prefix=prefix)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x0 = x
        x1 = self.conv_1(x0)
        x2 = self.conv_2(x1)
        return x2
