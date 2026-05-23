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
           x = layer.attention(x, mask)            # fused QKV, manual attn, dense, residual + LN
           x = layer.full_layer_layer_norm(        # FFN + residual + LN
                  ffn_output(activation(ffn(x))) + x )
    -> return x   # [B, T, hidden_size]

PyTorch is used only at preprocessing time and to upload host inputs (ids / mask).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

import ttnn


# --- params -----------------------------------------------------------------


@dataclass(frozen=True)
class TTAlbertLayerParams:
    """One ALBERT layer (attention + FFN). Fused QKV weights for ``transpose_b=True`` linear."""

    qkv_w: ttnn.Tensor
    qkv_b: ttnn.Tensor
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


def _largest_divisor(n: int, max_divisor: int = 8) -> int:
    if n <= 0:
        return 1
    for i in range(max_divisor, 0, -1):
        if n % i == 0:
            return i
    return 1


def _upload_fused_qkv(attn, device, dtype, *, head_size: int) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """Fuse ``query``/``key``/``value`` into one linear for a single device matmul.

    Bake ``1/sqrt(head_size)`` into Q weights so attention scores need no extra scale op.
    """
    q_scale = 1.0 / math.sqrt(head_size)
    qkv_weight = torch.cat(
        [attn.query.weight * q_scale, attn.key.weight, attn.value.weight],
        dim=0,
    )
    qkv_bias = torch.cat(
        [attn.query.bias * q_scale, attn.key.bias, attn.value.bias],
        dim=0,
    )
    w = _t(qkv_weight, device=device, dtype=dtype)
    b = _t(qkv_bias.reshape(1, 1, 1, -1), device=device, dtype=dtype)
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

    head_size = int(cfg.hidden_size) // int(cfg.num_attention_heads)
    layer_groups: list[tuple[TTAlbertLayerParams, ...]] = []
    for group in albert_model.encoder.albert_layer_groups:
        inner: list[TTAlbertLayerParams] = []
        for layer in group.albert_layers:
            attn = layer.attention
            qkv_w, qkv_b = _upload_fused_qkv(attn, device, weights_dtype, head_size=head_size)
            dense_w, dense_b = _upload_linear(attn.dense, device, weights_dtype)
            attn_ln_w, attn_ln_b = _upload_layernorm(attn.LayerNorm, device, weights_dtype)
            ffn_w, ffn_b = _upload_linear(layer.ffn, device, weights_dtype)
            ffn_out_w, ffn_out_b = _upload_linear(layer.ffn_output, device, weights_dtype)
            full_ln_w, full_ln_b = _upload_layernorm(layer.full_layer_layer_norm, device, weights_dtype)
            inner.append(
                TTAlbertLayerParams(
                    qkv_w=qkv_w,
                    qkv_b=qkv_b,
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


def _fix_b1th_after_linear(x: ttnn.Tensor, *, B: int, T: int, width: int) -> ttnn.Tensor:
    """``ttnn.linear`` may return ``[1, B, T, C]`` instead of ``[B, 1, T, C]``."""
    sh = list(x.shape)
    if len(sh) == 4 and int(sh[0]) == 1 and int(sh[1]) == B:
        return ttnn.permute(x, (1, 0, 2, 3))
    if len(sh) == 4 and int(sh[0]) == B and int(sh[1]) == 1:
        return ttnn.reshape(x, [B, 1, T, width])
    raise ValueError(f"unexpected linear output shape {sh} for B={B} T={T} width={width}")


def _to_b1th(x: ttnn.Tensor, *, B: int, T: int, width: int) -> ttnn.Tensor:
    """Normalize activations to ``[B, 1, T, width]`` for ``nlp_create_qkv_heads``."""
    while len(x.shape) > 4:
        x = ttnn.squeeze(x, 0)
    sh = list(x.shape)
    if len(sh) == 3:
        return ttnn.reshape(x, [B, 1, T, width])
    if len(sh) == 4 and int(sh[0]) == 1 and int(sh[1]) == B:
        return ttnn.permute(x, (1, 0, 2, 3))
    if len(sh) == 4 and int(sh[0]) == B and int(sh[1]) == 1:
        return ttnn.reshape(x, [B, 1, T, width])
    raise ValueError(f"unexpected shape {sh} for B={B} T={T} width={width}")


def _from_b1th(x: ttnn.Tensor, *, B: int, T: int, hidden_size: int) -> ttnn.Tensor:
    """``[B, 1, T, H]`` or ``[1, B, T, H]`` -> ``[B, T, H]`` TILE."""
    while len(x.shape) > 4:
        x = ttnn.squeeze(x, 0)
    sh = list(x.shape)
    if len(sh) == 3:
        return ttnn.reshape(x, [B, T, hidden_size])
    if len(sh) == 4 and int(sh[0]) == 1 and int(sh[1]) == B:
        x = ttnn.permute(x, (1, 0, 2, 3))
    elif not (len(sh) == 4 and int(sh[0]) == B and int(sh[1]) == 1):
        raise ValueError(f"unexpected shape {sh} for B={B} T={T} H={hidden_size}")
    return ttnn.reshape(x, [B, T, hidden_size])


def _mcast_1d_dram_program_config(
    device: ttnn.Device,
    *,
    seq_len: int,
    k: int,
    n: int,
    fused_activation=None,
    fp32_dest_acc_en: bool = True,
    prefer_subblock_h: int | None = None,
    prefer_subblock_w: int | None = None,
) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig:
    """DRAM multicast-1D config aligned with ``get_mcast_1d_config`` (``fuse_batch=False``)."""
    tile = ttnn.TILE_SIZE
    grid_size = device.compute_with_storage_grid_size()
    grid_x, grid_y = grid_size.x, grid_size.y
    num_cores = grid_x * grid_y

    per_core_m = max(1, math.ceil(seq_len / tile))
    per_core_n = max(1, math.ceil(math.ceil(n / num_cores) / tile))
    in0_block_w = 2 if (k // tile) % 2 == 0 else 1

    max_subblock = 4 if fp32_dest_acc_en else 8
    out_subblock_w = max([i for i in range(1, max_subblock + 1) if per_core_n % i == 0], default=1)
    out_subblock_h = max(
        [i for i in range(1, max_subblock + 1) if per_core_m % i == 0 and i * out_subblock_w <= max_subblock],
        default=1,
    )
    if prefer_subblock_w is not None and per_core_n % prefer_subblock_w == 0:
        out_subblock_w = prefer_subblock_w
    if (
        prefer_subblock_h is not None
        and per_core_m % prefer_subblock_h == 0
        and prefer_subblock_h * out_subblock_w <= max_subblock
    ):
        out_subblock_h = prefer_subblock_h

    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(grid_x, grid_y),
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        fuse_batch=False,
        fused_activation=fused_activation,
        mcast_in0=True,
    )


def _batched_attn_matmul_program_config(
    device: ttnn.Device,
    *,
    seq_len: int,
    k_dim: int,
    n_dim: int,
    fp32_dest_acc_en: bool = True,
) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig:
    """Batched head matmul (QK / PV): ``fuse_batch=False``, ``mcast_in0=False``."""
    tile = ttnn.TILE_SIZE
    grid_size = device.compute_with_storage_grid_size()
    grid_x, grid_y = grid_size.x, grid_size.y
    num_cores = grid_x * grid_y

    m_tiles = max(1, math.ceil(seq_len / tile))
    k_tiles = max(1, math.ceil(k_dim / tile))
    n_tiles = max(1, math.ceil(n_dim / tile))

    per_core_m = max(1, -(-m_tiles // num_cores))
    while m_tiles % per_core_m != 0:
        per_core_m += 1

    max_subblock = 4 if fp32_dest_acc_en else 8
    out_subblock_w = min(n_tiles, max_subblock)
    while out_subblock_w > 1 and n_tiles % out_subblock_w != 0:
        out_subblock_w -= 1
    out_subblock_h = min(per_core_m, max_subblock // max(out_subblock_w, 1))
    while out_subblock_h > 1 and per_core_m % out_subblock_h != 0:
        out_subblock_h -= 1

    in0_block_w = min(4, k_tiles)
    while in0_block_w > 1 and k_tiles % in0_block_w != 0:
        in0_block_w -= 1

    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(grid_x, grid_y),
        in0_block_w=in0_block_w,
        out_subblock_h=max(1, out_subblock_h),
        out_subblock_w=max(1, out_subblock_w),
        per_core_M=per_core_m,
        per_core_N=n_tiles,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=False,
    )


def _dram_layernorm_program_config() -> ttnn.LayerNormDefaultProgramConfig:
    return ttnn.LayerNormDefaultProgramConfig()


# gelu_new (PyTorch tanh approx) for fused FFN activation
_FFN_FUSED_GELU = ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU, 1.0)


@dataclass(frozen=True)
class TTAlbertMatmulProgramConfigs:
    """Per-(B, T) DRAM matmul program configs for Albert linears and attention."""

    qkv: ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig
    dense: ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig
    ffn_in: ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig
    ffn_out: ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig
    emb_map: ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig
    attn_qk: ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig
    attn_pv: ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig


def _build_matmul_program_configs(
    device: ttnn.Device,
    *,
    T: int,
    hidden_size: int,
    head_size: int,
    embedding_size: int,
    intermediate_size: int,
    fp32_dest_acc_en: bool = True,
) -> TTAlbertMatmulProgramConfigs:
    subblock_h = 2 if T >= ttnn.TILE_SIZE else 1
    return TTAlbertMatmulProgramConfigs(
        qkv=_mcast_1d_dram_program_config(
            device,
            seq_len=T,
            k=hidden_size,
            n=3 * hidden_size,
            fp32_dest_acc_en=fp32_dest_acc_en,
            prefer_subblock_h=subblock_h,
        ),
        dense=_mcast_1d_dram_program_config(
            device,
            seq_len=T,
            k=hidden_size,
            n=hidden_size,
            fp32_dest_acc_en=fp32_dest_acc_en,
        ),
        ffn_in=_mcast_1d_dram_program_config(
            device,
            seq_len=T,
            k=hidden_size,
            n=intermediate_size,
            fused_activation=_FFN_FUSED_GELU,
            fp32_dest_acc_en=fp32_dest_acc_en,
        ),
        ffn_out=_mcast_1d_dram_program_config(
            device,
            seq_len=T,
            k=intermediate_size,
            n=hidden_size,
            fp32_dest_acc_en=fp32_dest_acc_en,
        ),
        emb_map=_mcast_1d_dram_program_config(
            device,
            seq_len=T,
            k=embedding_size,
            n=hidden_size,
            fp32_dest_acc_en=fp32_dest_acc_en,
        ),
        attn_qk=_batched_attn_matmul_program_config(
            device,
            seq_len=T,
            k_dim=head_size,
            n_dim=T,
            fp32_dest_acc_en=fp32_dest_acc_en,
        ),
        attn_pv=_batched_attn_matmul_program_config(
            device,
            seq_len=T,
            k_dim=T,
            n_dim=head_size,
            fp32_dest_acc_en=fp32_dest_acc_en,
        ),
    )


def _build_extended_mask(
    attention_mask: torch.Tensor | None,
    *,
    B: int,
    T: int,
    device: ttnn.Device,
    dtype=ttnn.bfloat16,
) -> ttnn.Tensor:
    """``[B, T]`` (1=keep, 0=pad) -> additive mask ``[B, 1, 1, T]`` (TILE) with large neg on pad.

    When ``attention_mask`` is ``None`` (no padding), returns an all-zeros mask directly
    on device without any torch ops.
    """
    if attention_mask is None:
        return ttnn.zeros(
            [B, 1, 1, T], dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
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
    """One ALBERT layer: fused QKV + manual attention -> FFN, each with residual + LayerNorm."""

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
        matmul_pcs: TTAlbertMatmulProgramConfigs,
        ln_program_config: ttnn.LayerNormDefaultProgramConfig,
        memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
    ) -> ttnn.Tensor:
        p = self.params
        x_shape = list(x.shape)
        B, T = int(x_shape[-3]), int(x_shape[-2])

        x_b1th = _to_b1th(x, B=B, T=T, width=self.hidden_size)
        xqkv = ttnn.linear(
            x_b1th,
            p.qkv_w,
            bias=p.qkv_b,
            transpose_b=True,
            program_config=matmul_pcs.qkv,
            memory_config=memory_config,
            compute_kernel_config=self.compute_kernel_config,
        )
        xqkv = _fix_b1th_after_linear(xqkv, B=B, T=T, width=3 * self.hidden_size)

        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            xqkv,
            num_heads=self.num_heads,
            num_kv_heads=self.num_heads,
            transpose_k_heads=False,
            memory_config=memory_config,
        )
        ttnn.deallocate(xqkv)

        scores = ttnn.matmul(
            q,
            k,
            transpose_b=True,
            program_config=matmul_pcs.attn_qk,
            memory_config=memory_config,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        scores = ttnn.add(scores, attention_mask, memory_config=memory_config)

        probs = ttnn.softmax(scores, dim=-1, memory_config=memory_config)
        ttnn.deallocate(scores)

        ctx_heads = ttnn.matmul(
            probs,
            v,
            program_config=matmul_pcs.attn_pv,
            memory_config=memory_config,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(probs)
        ttnn.deallocate(v)

        ctx_b1th = ttnn.experimental.nlp_concat_heads(ctx_heads, memory_config=memory_config)
        ttnn.deallocate(ctx_heads)

        projected = ttnn.linear(
            ctx_b1th,
            p.dense_w,
            bias=p.dense_b,
            transpose_b=True,
            program_config=matmul_pcs.dense,
            memory_config=memory_config,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(ctx_b1th)
        projected_btH = _from_b1th(projected, B=B, T=T, hidden_size=self.hidden_size)
        ttnn.deallocate(projected)

        residual = ttnn.add(x, projected_btH, memory_config=memory_config)
        ttnn.deallocate(projected_btH)

        out = ttnn.layer_norm(
            residual,
            weight=p.attn_ln_w,
            bias=p.attn_ln_b,
            epsilon=self.layer_norm_eps,
            program_config=ln_program_config,
            memory_config=memory_config,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(residual)
        return out

    def _ffn(
        self,
        x: ttnn.Tensor,
        *,
        matmul_pcs: TTAlbertMatmulProgramConfigs,
        ln_program_config: ttnn.LayerNormDefaultProgramConfig,
        memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
    ) -> ttnn.Tensor:
        p = self.params
        h_act = ttnn.linear(
            x,
            p.ffn_w,
            bias=p.ffn_b,
            transpose_b=True,
            program_config=matmul_pcs.ffn_in,
            memory_config=memory_config,
            compute_kernel_config=self.compute_kernel_config,
        )
        h2 = ttnn.linear(
            h_act,
            p.ffn_output_w,
            bias=p.ffn_output_b,
            transpose_b=True,
            program_config=matmul_pcs.ffn_out,
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
            program_config=ln_program_config,
            memory_config=memory_config,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(residual)
        return out

    def forward(
        self,
        x: ttnn.Tensor,
        attention_mask: ttnn.Tensor,
        *,
        matmul_pcs: TTAlbertMatmulProgramConfigs,
        ln_program_config: ttnn.LayerNormDefaultProgramConfig,
    ) -> ttnn.Tensor:
        a = self._attention(x, attention_mask, matmul_pcs=matmul_pcs, ln_program_config=ln_program_config)
        o = self._ffn(a, matmul_pcs=matmul_pcs, ln_program_config=ln_program_config)
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
        self._intermediate_size = int(params.layer_groups[0][0].ffn_w.shape[-1])
        self._head_size = params.hidden_size // params.num_attention_heads
        self._matmul_pc_cache: dict[tuple[int, int], TTAlbertMatmulProgramConfigs] = {}
        self._ln_program_config = _dram_layernorm_program_config()
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

    def _matmul_program_configs(self, B: int, T: int) -> TTAlbertMatmulProgramConfigs:
        key = (B, T)
        if key not in self._matmul_pc_cache:
            self._matmul_pc_cache[key] = _build_matmul_program_configs(
                self.device,
                T=T,
                hidden_size=self.params.hidden_size,
                head_size=self._head_size,
                embedding_size=self.params.embedding_size,
                intermediate_size=self._intermediate_size,
                fp32_dest_acc_en=True,
            )
        return self._matmul_pc_cache[key]

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

        position_ids = torch.from_numpy(np.tile(np.arange(T, dtype=np.int32), (B, 1)))
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
            program_config=self._ln_program_config,
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

        if token_type_ids is None:
            token_type_ids = torch.from_numpy(np.zeros((B, T), dtype=np.int32))

        emb = self._embed(input_ids, token_type_ids)

        matmul_pcs = self._matmul_program_configs(B, T)

        hidden = ttnn.linear(
            emb,
            self.params.emb_map_w,
            bias=self.params.emb_map_b,
            transpose_b=True,
            program_config=matmul_pcs.emb_map,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(emb)

        # attention_mask=None means full-length (no padding): ext_mask is all-zeros (no-op add).
        ext_mask = _build_extended_mask(attention_mask, B=B, T=T, device=self.device)

        num_layers = self.params.num_hidden_layers
        num_groups = self.params.num_hidden_groups
        inner = self.params.inner_group_num
        layers_per_group = num_layers // num_groups

        for i in range(num_layers):
            group_idx = i // layers_per_group
            inner_idx = (i - group_idx * layers_per_group) % inner
            layer = self._layers[group_idx][inner_idx]
            new_hidden = layer(
                hidden,
                ext_mask,
                matmul_pcs=matmul_pcs,
                ln_program_config=self._ln_program_config,
            )
            ttnn.deallocate(hidden)
            hidden = new_hidden

        ttnn.deallocate(ext_mask)

        while len(hidden.shape) > 3:
            hidden = ttnn.squeeze(hidden, 0)
        return hidden

    __call__ = forward
