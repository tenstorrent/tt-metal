# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import math
from models.experimental.uniad.tt.matmul_helpers import linear_flatten_batch


class TtMultiheadAttention:
    def __init__(
        self,
        params,
        device,
        embed_dims=256,
        num_heads=8,
        init_cfg=None,
        batch_first=False,
    ):
        params = params.attn
        super().__init__()
        self.params = params
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.device = device
        self.batch_first = batch_first
        self.attn_in_proj__weight = params.in_proj_weight
        self.attn_in_proj__weight = ttnn.to_layout(self.attn_in_proj__weight, layout=ttnn.TILE_LAYOUT)
        self.attn_in_proj__bias = params.in_proj_bias
        self.attn_in_proj__bias = ttnn.to_layout(self.attn_in_proj__bias, layout=ttnn.TILE_LAYOUT)
        self.attn_in_proj__weight_permute = self.attn_in_proj__weight
        self.attn_in_proj__bias_squeeze = ttnn.squeeze(self.attn_in_proj__bias, 0)
        # The q/k/v weight slice+transpose and bias slice are pure functions of
        # these static weights — precompute once instead of redoing 3 slices +
        # 3 permutes (6 extra device programs) on every forward.
        _w = self.attn_in_proj__weight_permute
        _b = self.attn_in_proj__bias_squeeze
        _e = self.embed_dims
        self._q_weight = ttnn.permute(_w[:_e, :], (1, 0))
        self._k_weight = ttnn.permute(_w[_e : 2 * _e, :], (1, 0))
        self._v_weight = ttnn.permute(_w[2 * _e :, :], (1, 0))
        self._q_bias = _b[:_e]
        self._k_bias = _b[_e : 2 * _e]
        self._v_bias = _b[2 * _e :]
        self.attn_out_proj_weight = params.out_proj.weight
        self.attn_out_proj_weight = ttnn.to_layout(self.attn_out_proj_weight, layout=ttnn.TILE_LAYOUT)
        self.attn_out_proj_bias = params.out_proj.bias
        self.attn_out_proj_bias = ttnn.to_layout(self.attn_out_proj_bias, layout=ttnn.TILE_LAYOUT)

    def __call__(
        self,
        query,
        key=None,
        value=None,
        identity=None,
        query_pos=None,
        key_pos=None,
        attn_mask=None,
        key_padding_mask=None,
        batch_first=False,
        **kwargs,
    ):
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                if query_pos.shape == key.shape:
                    key_pos = query_pos

        if query_pos is not None:
            if query.get_layout() != ttnn.TILE_LAYOUT:
                query = ttnn.to_layout(query, ttnn.TILE_LAYOUT)
            if query_pos.get_layout() != ttnn.TILE_LAYOUT:
                query_pos = ttnn.to_layout(query_pos, ttnn.TILE_LAYOUT)
            query = query + query_pos
        if key_pos is not None:
            if key.get_layout() != ttnn.TILE_LAYOUT:
                key = ttnn.to_layout(key, ttnn.TILE_LAYOUT)
            if key_pos.get_layout() != ttnn.TILE_LAYOUT:
                key_pos = ttnn.to_layout(key_pos, ttnn.TILE_LAYOUT)
            key = key + key_pos

        if batch_first:
            query = ttnn.permute(query, (1, 0))
            key = ttnn.permute(key, (1, 0))
            value = ttnn.permute(value, (1, 0))

        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape

        q_weight = self._q_weight
        k_weight = self._k_weight
        v_weight = self._v_weight

        q_bias = self._q_bias
        k_bias = self._k_bias
        v_bias = self._v_bias

        q_batch_size, q_sequence_size, q_hidden_size = query.shape
        q_head_size = q_hidden_size // self.num_heads

        k_batch_size, k_sequence_size, k_hidden_size = key.shape
        k_head_size = k_hidden_size // self.num_heads

        v_batch_size, v_sequence_size, v_hidden_size = value.shape
        v_head_size = v_hidden_size // self.num_heads

        # Flatten leading dims into M (e.g. (901, 1, 256) -> (1, 1, 901, 256))
        # so the matmul heuristic sees the full row count instead of bsz=1.
        query = linear_flatten_batch(query, q_weight, bias=q_bias)

        key = linear_flatten_batch(key, k_weight, bias=k_bias)

        if value.get_layout() != ttnn.TILE_LAYOUT:
            value = ttnn.to_layout(value, ttnn.TILE_LAYOUT)
        value = linear_flatten_batch(value, v_weight, bias=v_bias)

        query = ttnn.reshape(query, (tgt_len, bsz * self.num_heads, q_head_size))
        query = ttnn.permute(query, (1, 0, 2))

        key = ttnn.reshape(key, (k_batch_size, bsz * self.num_heads, q_head_size))
        key = ttnn.permute(key, (1, 0, 2))

        value = ttnn.reshape(value, (v_batch_size, bsz * self.num_heads, q_head_size))
        value = ttnn.permute(value, (1, 0, 2))

        src_len = key.shape[1]

        B, Nt, E = query.shape
        q_scaled = query * math.sqrt(1.0 / float(E))
        key_transposed = ttnn.permute(key, (0, 2, 1))

        if attn_mask is not None:
            attn_output_weights = ttnn.matmul(q_scaled, key_transposed)
            attn_output_weights = attn_output_weights + attn_mask
        else:
            attn_output_weights = ttnn.matmul(q_scaled, key_transposed)

        attn_output_weights = ttnn.softmax(attn_output_weights, dim=-1)

        attn_output = ttnn.matmul(attn_output_weights, value)

        attn_output = ttnn.permute(attn_output, (1, 0, 2))
        attn_output = ttnn.reshape(attn_output, (tgt_len * bsz, embed_dim))

        attn_output = ttnn.linear(attn_output, self.attn_out_proj_weight, bias=self.attn_out_proj_bias)
        attn_output = ttnn.reshape(attn_output, (tgt_len, bsz, attn_output.shape[1]))
        attn_output_weights = ttnn.reshape(attn_output_weights, (bsz, self.num_heads, tgt_len, src_len))
        attn_output_weights = ttnn.to_layout(attn_output_weights, ttnn.ROW_MAJOR_LAYOUT)
        attn_output_weights = ttnn.mean(attn_output_weights, dim=1)
        identity = ttnn.to_layout(identity, ttnn.TILE_LAYOUT)

        return attn_output + identity
