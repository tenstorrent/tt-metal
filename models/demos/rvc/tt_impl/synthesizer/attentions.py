# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import torch

import ttnn
from models.demos.rvc.tt_impl.conv1d import Conv1d
from models.demos.rvc.tt_impl.linear import Linear


def ttnn_gather_fallback(x: ttnn.Tensor, dim: int, index: ttnn.Tensor, device) -> ttnn.Tensor:
    # Fallback implementation of gather using to_host, torch.gather, and from_torch.
    # needed since ttnn.gather is 4-8x slower than this fallback version
    x_torch = ttnn.to_torch(x)
    index_torch = ttnn.to_torch(index).to(torch.int64)
    gathered_torch = torch.gather(x_torch, dim=dim, index=index_torch)
    gathered = ttnn.from_torch(gathered_torch, dtype=x.dtype, layout=x.layout, device=device)
    return gathered


class MultiHeadAttention:
    def __init__(
        self,
        device: ttnn.MeshDevice,
        in_features: int,
        out_features: int,
        num_heads: int,
        window_size: int | None = None,
    ) -> None:
        if in_features % num_heads != 0:
            raise ValueError("in_features must be divisible by num_heads")

        self.device = device
        self.num_heads = num_heads
        self.window_size = window_size
        self.features_per_head = in_features // num_heads

        self.linear_q = Linear(
            device=device,
            in_features=in_features,
            out_features=in_features,
        )
        self.linear_k = Linear(
            device=device,
            in_features=in_features,
            out_features=in_features,
        )
        self.linear_v = Linear(
            device=device,
            in_features=in_features,
            out_features=in_features,
        )
        self.linear_o = Linear(
            device=device,
            in_features=in_features,
            out_features=out_features,
        )
        self.emb_rel_k: ttnn.Tensor | None = None
        self.emb_rel_v: ttnn.Tensor | None = None
        self.relative_position_cache: dict[int : ttnn.Tensor] = {}
        self.index_cache: dict[int : ttnn.Tensor] = {}

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], module_prefix: str = "") -> None:
        q_key = (
            "linear_q"
            if (f"{module_prefix}linear_q.weight" if module_prefix else "linear_q.weight") in state_dict
            else "conv_q"
        )
        k_key = (
            "linear_k"
            if (f"{module_prefix}linear_k.weight" if module_prefix else "linear_k.weight") in state_dict
            else "conv_k"
        )
        v_key = (
            "linear_v"
            if (f"{module_prefix}linear_v.weight" if module_prefix else "linear_v.weight") in state_dict
            else "conv_v"
        )
        o_key = (
            "linear_o"
            if (f"{module_prefix}linear_o.weight" if module_prefix else "linear_o.weight") in state_dict
            else "conv_o"
        )
        self.linear_q.load_state_dict(state_dict=state_dict, key=q_key, module_prefix=module_prefix)
        self.linear_k.load_state_dict(state_dict=state_dict, key=k_key, module_prefix=module_prefix)
        self.linear_v.load_state_dict(state_dict=state_dict, key=v_key, module_prefix=module_prefix)
        self.linear_o.load_state_dict(state_dict=state_dict, key=o_key, module_prefix=module_prefix)
        if self.window_size is not None:
            key_emb_rel_k = f"{module_prefix}emb_rel_k" if module_prefix else "emb_rel_k"
            key_emb_rel_v = f"{module_prefix}emb_rel_v" if module_prefix else "emb_rel_v"
            if key_emb_rel_k not in state_dict:
                raise KeyError(f"Missing required parameter: {key_emb_rel_k}")
            if key_emb_rel_v not in state_dict:
                raise KeyError(f"Missing required parameter: {key_emb_rel_v}")

            emb_rel_k = state_dict[key_emb_rel_k].reshape(1, 1, -1, self.features_per_head)
            emb_rel_v = state_dict[key_emb_rel_v].reshape(1, 1, -1, self.features_per_head)
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
        q = self._reshape_to_heads(self.linear_q(x))
        k = self._reshape_to_heads(self.linear_k(c))
        v = self._reshape_to_heads(self.linear_v(c))
        q_scaled = ttnn.mul(q, 1.0 / math.sqrt(self.features_per_head), output_tensor=q)
        scores = ttnn.matmul(q_scaled, k, transpose_b=True)
        _, _, target_length, source_length = scores.shape
        if self.window_size is not None:
            if source_length != target_length:
                raise ValueError("Relative attention is only available for self-attention.")
            if self.emb_rel_k is None or self.emb_rel_v is None:
                raise ValueError("Relative embeddings are not loaded.")

            key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, source_length)
            rel_logits = ttnn.matmul(q_scaled, key_relative_embeddings, transpose_b=True)
            scores_local = self._relative_to_absolute_position(rel_logits)
            scores = ttnn.add(scores, scores_local, output_tensor=scores)

        p_attn = ttnn.softmax_in_place(scores, dim=-1)
        output = ttnn.matmul(p_attn, v)
        if self.window_size is not None:
            assert self.emb_rel_v is not None
            relative_weights = self._absolute_to_relative_position(p_attn)
            value_relative_embeddings = self._get_relative_embeddings(self.emb_rel_v, source_length)
            output = ttnn.add(output, ttnn.matmul(relative_weights, value_relative_embeddings), output_tensor=output)
        output = ttnn.transformer.concatenate_heads(output)

        out_tt = self.linear_o(output)
        return out_tt

    def _reshape_to_heads(self, x: ttnn.Tensor) -> ttnn.Tensor:
        batch_size, length, channels = x.shape
        x = ttnn.reshape(x, (batch_size, length, self.num_heads, self.features_per_head))
        x_p = ttnn.permute(x, (0, 2, 1, 3))
        return x_p

    def _get_relative_embeddings(self, relative_embeddings: ttnn.Tensor, length: int) -> ttnn.Tensor:
        if self.window_size is None:
            raise ValueError("window_size must be set for relative attention.")
        pad_length: int = max(length - (self.window_size + 1), 0)
        slice_start_position = max((self.window_size + 1) - length, 0)
        slice_end_position = slice_start_position + 2 * length - 1

        embeddings = relative_embeddings
        if pad_length > 0:
            embeddings = ttnn.pad(
                embeddings,
                padding=((0, 0), (0, 0), (pad_length, pad_length), (0, 0)),
                value=0.0,
            )
        embeddings = ttnn.slice(
            embeddings,
            (0, 0, slice_start_position, 0),
            (1, 1, slice_end_position, self.features_per_head),
        )
        return ttnn.to_layout(embeddings, ttnn.TILE_LAYOUT)

    def _get_rel_idx_tensor(self, length: int) -> ttnn.Tensor:
        if length in self.index_cache:
            return self.index_cache[length]
        idx_row = ttnn.unsqueeze(ttnn.arange(start=0, end=length, dtype=ttnn.uint32, device=self.device), dim=1)
        idx_col = ttnn.unsqueeze(
            ttnn.arange(start=length - 1, end=2 * length - 1, dtype=ttnn.uint32, device=self.device), dim=0
        )
        rel_idx = idx_col - idx_row
        rel_idx = ttnn.expand(ttnn.reshape(rel_idx, shape=(1, 1, length, length)), (1, 1, length, length))
        self.index_cache[length] = rel_idx
        return rel_idx

    def _relative_to_absolute_position(self, x: ttnn.Tensor) -> ttnn.Tensor:
        batch, heads, length, _ = x.shape
        rel_idx = self._get_rel_idx_tensor(length)
        out = ttnn_gather_fallback(x, dim=3, index=rel_idx, device=self.device)
        return out

    def _absolute_to_relative_position(self, x: ttnn.Tensor) -> ttnn.Tensor:
        batch, heads, length, _ = x.shape
        rel_idx = self._get_rel_idx_tensor(length)
        if length in self.relative_position_cache:
            out = self.relative_position_cache[length]
        else:
            out = ttnn.zeros(
                (batch, heads, length, 2 * length - 1),
                dtype=x.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )
            self.relative_position_cache[length] = out
        out = ttnn.scatter(out, dim=3, index=rel_idx, src=x)
        return out


class FFN:
    def __init__(
        self,
        device: ttnn.MeshDevice,
        in_channels: int,
        out_channels: int,
        filter_channels: int,
        kernel_size: int,
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

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], module_prefix: str = "") -> None:
        self.conv_1.load_state_dict(state_dict=state_dict, key="conv_1", module_prefix=module_prefix)
        self.conv_2.load_state_dict(state_dict=state_dict, key="conv_2", module_prefix=module_prefix)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = self.conv_1(x)
        out = self.conv_2(x)
        return out
