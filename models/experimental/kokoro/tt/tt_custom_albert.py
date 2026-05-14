# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""TTNN port of Kokoro's :class:`~models.experimental.kokoro.reference.modules.CustomAlbert`.

``CustomAlbert`` is a thin wrapper around ``transformers.AlbertModel`` that returns
``last_hidden_state``. The forward path:

    embeddings (word + pos + token_type, LN)
    -> embedding_hidden_mapping_in (E -> H)
    -> for i in range(num_hidden_layers):
           group_idx, inner_idx = (i // layers_per_group) % num_hidden_groups, i % inner_group_num
           layer = albert_layer_groups[group_idx].albert_layers[inner_idx]
           x = layer.attention(x, mask)            # qkv, sdpa, dense, residual + LN
           x = layer.full_layer_layer_norm(        # FFN + residual + LN
                  ffn_output(activation(ffn(x))) + x )
    -> return x   # [B, T, hidden_size]

PyTorch is used only at preprocessing time and to upload host inputs (ids / mask).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

import ttnn


# --- params -----------------------------------------------------------------


@dataclass(frozen=True)
class TTAlbertLayerParams:
    """One ALBERT layer (attention + FFN). Linear weights stored for ``transpose_b=True``."""

    q_w: ttnn.Tensor
    q_b: ttnn.Tensor
    k_w: ttnn.Tensor
    k_b: ttnn.Tensor
    v_w: ttnn.Tensor
    v_b: ttnn.Tensor
    dense_w: ttnn.Tensor
    dense_b: ttnn.Tensor
    attn_ln_w: ttnn.Tensor
    attn_ln_b: ttnn.Tensor
    ffn_w: ttnn.Tensor
    ffn_b: ttnn.Tensor
    ffn_output_w: ttnn.Tensor
    ffn_output_b: ttnn.Tensor
    full_ln_w: ttnn.Tensor
    full_ln_b: ttnn.Tensor


@dataclass(frozen=True)
class TTCustomAlbertParams:
    """Device-resident weights and config for :class:`TTCustomAlbert`."""

    word_emb: ttnn.Tensor
    pos_emb: ttnn.Tensor
    token_type_emb: ttnn.Tensor
    emb_ln_w: ttnn.Tensor
    emb_ln_b: ttnn.Tensor
    emb_map_w: ttnn.Tensor
    emb_map_b: ttnn.Tensor
    layer_groups: tuple[tuple[TTAlbertLayerParams, ...], ...]
    num_hidden_layers: int
    num_hidden_groups: int
    inner_group_num: int
    num_attention_heads: int
    hidden_size: int
    embedding_size: int
    layer_norm_eps: float


# --- preprocess --------------------------------------------------------------


def _t(t: torch.Tensor, *, device, dtype, layout=ttnn.TILE_LAYOUT) -> ttnn.Tensor:
    return ttnn.from_torch(
        t.detach().cpu(),
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _upload_linear(linear: nn.Linear, device, dtype) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    w = _t(linear.weight, device=device, dtype=dtype)
    b = _t(linear.bias.reshape(1, 1, 1, -1), device=device, dtype=dtype)
    return w, b


def _upload_layernorm(ln: nn.LayerNorm, device, dtype) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    w = _t(ln.weight, device=device, dtype=dtype)
    b = _t(ln.bias, device=device, dtype=dtype)
    return w, b


def preprocess_tt_custom_albert(
    albert_model: nn.Module,
    device: ttnn.Device,
    *,
    weights_dtype=ttnn.bfloat16,
) -> TTCustomAlbertParams:
    """Upload a ``transformers.AlbertModel`` (or ``CustomAlbert``) to device."""
    cfg = albert_model.config
    emb = albert_model.embeddings

    word_emb = _t(emb.word_embeddings.weight, device=device, dtype=weights_dtype)
    pos_emb = _t(emb.position_embeddings.weight, device=device, dtype=weights_dtype)
    token_type_emb = _t(emb.token_type_embeddings.weight, device=device, dtype=weights_dtype)
    emb_ln_w, emb_ln_b = _upload_layernorm(emb.LayerNorm, device, weights_dtype)

    emb_map_w, emb_map_b = _upload_linear(albert_model.encoder.embedding_hidden_mapping_in, device, weights_dtype)

    layer_groups: list[tuple[TTAlbertLayerParams, ...]] = []
    for group in albert_model.encoder.albert_layer_groups:
        inner: list[TTAlbertLayerParams] = []
        for layer in group.albert_layers:
            attn = layer.attention
            q_w, q_b = _upload_linear(attn.query, device, weights_dtype)
            k_w, k_b = _upload_linear(attn.key, device, weights_dtype)
            v_w, v_b = _upload_linear(attn.value, device, weights_dtype)
            dense_w, dense_b = _upload_linear(attn.dense, device, weights_dtype)
            attn_ln_w, attn_ln_b = _upload_layernorm(attn.LayerNorm, device, weights_dtype)
            ffn_w, ffn_b = _upload_linear(layer.ffn, device, weights_dtype)
            ffn_out_w, ffn_out_b = _upload_linear(layer.ffn_output, device, weights_dtype)
            full_ln_w, full_ln_b = _upload_layernorm(layer.full_layer_layer_norm, device, weights_dtype)
            inner.append(
                TTAlbertLayerParams(
                    q_w=q_w,
                    q_b=q_b,
                    k_w=k_w,
                    k_b=k_b,
                    v_w=v_w,
                    v_b=v_b,
                    dense_w=dense_w,
                    dense_b=dense_b,
                    attn_ln_w=attn_ln_w,
                    attn_ln_b=attn_ln_b,
                    ffn_w=ffn_w,
                    ffn_b=ffn_b,
                    ffn_output_w=ffn_out_w,
                    ffn_output_b=ffn_out_b,
                    full_ln_w=full_ln_w,
                    full_ln_b=full_ln_b,
                )
            )
        layer_groups.append(tuple(inner))

    return TTCustomAlbertParams(
        word_emb=word_emb,
        pos_emb=pos_emb,
        token_type_emb=token_type_emb,
        emb_ln_w=emb_ln_w,
        emb_ln_b=emb_ln_b,
        emb_map_w=emb_map_w,
        emb_map_b=emb_map_b,
        layer_groups=tuple(layer_groups),
        num_hidden_layers=int(cfg.num_hidden_layers),
        num_hidden_groups=int(cfg.num_hidden_groups),
        inner_group_num=int(cfg.inner_group_num),
        num_attention_heads=int(cfg.num_attention_heads),
        hidden_size=int(cfg.hidden_size),
        embedding_size=int(cfg.embedding_size),
        layer_norm_eps=float(cfg.layer_norm_eps),
    )


# --- helpers ----------------------------------------------------------------


def _split_heads(
    x: ttnn.Tensor,
    *,
    B: int,
    T: int,
    num_heads: int,
    head_size: int,
) -> ttnn.Tensor:
    """Reshape linear output into ``[B, num_heads, T, head_size]`` (TILE).

    Accepts any rank where the total element count matches ``B * T * num_heads * head_size``;
    ``ttnn.linear`` may emit leading singleton dims when given a rank-3 input.
    """
    x_rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    x_rm = ttnn.reshape(x_rm, [B, T, num_heads, head_size])
    x_tile = ttnn.to_layout(x_rm, ttnn.TILE_LAYOUT)
    ttnn.deallocate(x_rm)
    out = ttnn.permute(x_tile, (0, 2, 1, 3))
    ttnn.deallocate(x_tile)
    return out


def _merge_heads(
    x_bhTd: ttnn.Tensor,
    *,
    B: int,
    T: int,
    hidden_size: int,
) -> ttnn.Tensor:
    """``[B, num_heads, T, head_size]`` (TILE) -> ``[B, T, H]`` (TILE)."""
    x = ttnn.permute(x_bhTd, (0, 2, 1, 3))
    x_rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    ttnn.deallocate(x)
    x_rm = ttnn.reshape(x_rm, [B, T, hidden_size])
    out = ttnn.to_layout(x_rm, ttnn.TILE_LAYOUT)
    ttnn.deallocate(x_rm)
    return out


def _build_extended_mask(
    attention_mask: torch.Tensor,
    *,
    device: ttnn.Device,
    dtype=ttnn.bfloat16,
) -> ttnn.Tensor:
    """``[B, T]`` (1=keep, 0=pad) -> additive mask ``[B, 1, 1, T]`` (TILE) with large neg on pad."""
    m = attention_mask.to(torch.float32)
    extended = (1.0 - m).unsqueeze(1).unsqueeze(2) * -1.0e4
    return ttnn.from_torch(
        extended,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


# --- modules ----------------------------------------------------------------


class TTAlbertLayer:
    """One ALBERT layer: SDPA-style attention -> FFN, each with residual + LayerNorm."""

    __slots__ = ("params", "num_heads", "head_size", "hidden_size", "layer_norm_eps", "compute_kernel_config")

    def __init__(
        self,
        params: TTAlbertLayerParams,
        *,
        num_heads: int,
        hidden_size: int,
        layer_norm_eps: float,
        compute_kernel_config,
    ) -> None:
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_attention_heads"
        self.params = params
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.hidden_size = hidden_size
        self.layer_norm_eps = layer_norm_eps
        self.compute_kernel_config = compute_kernel_config

    def _attention(
        self,
        x: ttnn.Tensor,
        attention_mask: ttnn.Tensor,
        *,
        memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
    ) -> ttnn.Tensor:
        p = self.params
        x_shape = list(x.shape)
        B, T = int(x_shape[-3]), int(x_shape[-2])

        q_btH = ttnn.linear(
            x,
            p.q_w,
            bias=p.q_b,
            transpose_b=True,
            memory_config=memory_config,
            compute_kernel_config=self.compute_kernel_config,
        )
        k_btH = ttnn.linear(
            x,
            p.k_w,
            bias=p.k_b,
            transpose_b=True,
            memory_config=memory_config,
            compute_kernel_config=self.compute_kernel_config,
        )
        v_btH = ttnn.linear(
            x,
            p.v_w,
            bias=p.v_b,
            transpose_b=True,
            memory_config=memory_config,
            compute_kernel_config=self.compute_kernel_config,
        )

        q = _split_heads(q_btH, B=B, T=T, num_heads=self.num_heads, head_size=self.head_size)
        ttnn.deallocate(q_btH)
        k = _split_heads(k_btH, B=B, T=T, num_heads=self.num_heads, head_size=self.head_size)
        ttnn.deallocate(k_btH)
        v = _split_heads(v_btH, B=B, T=T, num_heads=self.num_heads, head_size=self.head_size)
        ttnn.deallocate(v_btH)

        scale = 1.0 / (self.head_size**0.5)
        scores = ttnn.matmul(
            q,
            k,
            transpose_b=True,
            memory_config=memory_config,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        scores = ttnn.multiply(scores, scale, memory_config=memory_config)
        scores = ttnn.add(scores, attention_mask, memory_config=memory_config)

        probs = ttnn.softmax(scores, dim=-1, memory_config=memory_config)
        ttnn.deallocate(scores)

        ctx_bhTd = ttnn.matmul(
            probs,
            v,
            memory_config=memory_config,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(probs)
        ttnn.deallocate(v)

        ctx_btH = _merge_heads(ctx_bhTd, B=B, T=T, hidden_size=self.hidden_size)
        ttnn.deallocate(ctx_bhTd)

        projected = ttnn.linear(
            ctx_btH,
            p.dense_w,
            bias=p.dense_b,
            transpose_b=True,
            memory_config=memory_config,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(ctx_btH)

        residual = ttnn.add(x, projected, memory_config=memory_config)
        ttnn.deallocate(projected)

        out = ttnn.layer_norm(
            residual,
            weight=p.attn_ln_w,
            bias=p.attn_ln_b,
            epsilon=self.layer_norm_eps,
            memory_config=memory_config,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(residual)
        return out

    def _ffn(
        self,
        x: ttnn.Tensor,
        *,
        memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
    ) -> ttnn.Tensor:
        p = self.params
        h = ttnn.linear(
            x,
            p.ffn_w,
            bias=p.ffn_b,
            transpose_b=True,
            memory_config=memory_config,
            compute_kernel_config=self.compute_kernel_config,
        )
        # Reference uses gelu_new (tanh approximation): ttnn.gelu(fast_and_approximate_mode=True).
        h_act = ttnn.gelu(h, fast_and_approximate_mode=True, memory_config=memory_config)
        ttnn.deallocate(h)
        h2 = ttnn.linear(
            h_act,
            p.ffn_output_w,
            bias=p.ffn_output_b,
            transpose_b=True,
            memory_config=memory_config,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(h_act)
        residual = ttnn.add(h2, x, memory_config=memory_config)
        ttnn.deallocate(h2)
        out = ttnn.layer_norm(
            residual,
            weight=p.full_ln_w,
            bias=p.full_ln_b,
            epsilon=self.layer_norm_eps,
            memory_config=memory_config,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(residual)
        return out

    def forward(self, x: ttnn.Tensor, attention_mask: ttnn.Tensor) -> ttnn.Tensor:
        a = self._attention(x, attention_mask)
        o = self._ffn(a)
        ttnn.deallocate(a)
        return o

    __call__ = forward


class TTCustomAlbert:
    """ALBERT encoder returning ``last_hidden_state`` ``[B, T, hidden_size]``."""

    def __init__(self, device: ttnn.Device, params: TTCustomAlbertParams) -> None:
        self.device = device
        self.params = params
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi3,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
        )
        self._layers: tuple[tuple[TTAlbertLayer, ...], ...] = tuple(
            tuple(
                TTAlbertLayer(
                    lp,
                    num_heads=params.num_attention_heads,
                    hidden_size=params.hidden_size,
                    layer_norm_eps=params.layer_norm_eps,
                    compute_kernel_config=self.compute_kernel_config,
                )
                for lp in group
            )
            for group in params.layer_groups
        )

    def _embed(
        self,
        input_ids: torch.LongTensor,
        token_type_ids: torch.LongTensor,
    ) -> ttnn.Tensor:
        """word + position + token-type embeddings followed by LayerNorm."""
        B, T = input_ids.shape

        ids_tt = ttnn.from_torch(
            input_ids.to(torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        word_e = ttnn.embedding(ids_tt, self.params.word_emb, layout=ttnn.TILE_LAYOUT)
        ttnn.deallocate(ids_tt)

        tids_tt = ttnn.from_torch(
            token_type_ids.to(torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        token_type_e = ttnn.embedding(tids_tt, self.params.token_type_emb, layout=ttnn.TILE_LAYOUT)
        ttnn.deallocate(tids_tt)

        position_ids = torch.arange(T, dtype=torch.int32).unsqueeze(0).expand(B, T).contiguous()
        pids_tt = ttnn.from_torch(
            position_ids,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        pos_e = ttnn.embedding(pids_tt, self.params.pos_emb, layout=ttnn.TILE_LAYOUT)
        ttnn.deallocate(pids_tt)

        emb = ttnn.add(word_e, token_type_e, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(word_e)
        ttnn.deallocate(token_type_e)
        emb_sum = ttnn.add(emb, pos_e, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(emb)
        ttnn.deallocate(pos_e)

        out = ttnn.layer_norm(
            emb_sum,
            weight=self.params.emb_ln_w,
            bias=self.params.emb_ln_b,
            epsilon=self.params.layer_norm_eps,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(emb_sum)
        return out

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.LongTensor | None = None,
    ) -> ttnn.Tensor:
        """
        Args:
            input_ids: ``[B, T]`` token indices (CPU long).
            attention_mask: ``[B, T]`` with ``1`` = keep, ``0`` = pad. Defaults to all ones.
            token_type_ids: ``[B, T]``. Defaults to zeros (single-segment).

        Returns:
            ``[B, T, hidden_size]`` on device, TILE layout.
        """
        B, T = input_ids.shape

        if attention_mask is None:
            attention_mask = torch.ones((B, T), dtype=torch.int32)
        if token_type_ids is None:
            token_type_ids = torch.zeros((B, T), dtype=torch.long)

        emb = self._embed(input_ids, token_type_ids)

        hidden = ttnn.linear(
            emb,
            self.params.emb_map_w,
            bias=self.params.emb_map_b,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(emb)

        ext_mask = _build_extended_mask(attention_mask, device=self.device)

        num_layers = self.params.num_hidden_layers
        num_groups = self.params.num_hidden_groups
        inner = self.params.inner_group_num
        layers_per_group = num_layers // num_groups

        for i in range(num_layers):
            group_idx = i // layers_per_group
            inner_idx = (i - group_idx * layers_per_group) % inner
            layer = self._layers[group_idx][inner_idx]
            new_hidden = layer(hidden, ext_mask)
            ttnn.deallocate(hidden)
            hidden = new_hidden

        ttnn.deallocate(ext_mask)

        # ``ttnn.linear`` / ``ttnn.layer_norm`` may leave the tensor at rank 4 with a leading singleton.
        while len(hidden.shape) > 3:
            hidden = ttnn.squeeze(hidden, 0)
        return hidden

    __call__ = forward
