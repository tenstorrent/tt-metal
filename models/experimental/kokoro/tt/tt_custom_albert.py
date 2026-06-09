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
from typing import Union

import torch
import torch.nn as nn

import ttnn

from models.experimental.kokoro.tt.tt_matmul_memory import maybe_reshard_to_caller

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


# Sweep winners at T=64 (Kokoro PL-BERT): (grid_x, grid_y), in0_block_w.
# See models/experimental/kokoro/tests/perf/test_custom_albert_matmul_sweep.py.
_T64_LINEAR_TUNING: dict[str, tuple[tuple[int, int], int]] = {
    "emb_map": ((8, 3), 4),
    "qkv": ((9, 8), 4),
    "dense": ((4, 6), 8),
    "ffn_in": ((8, 4), 8),
    "ffn_out": ((8, 3), 8),
}


def _grids_for_cores(cores: int, gx_max: int, gy_max: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for gx in range(1, gx_max + 1):
        if cores % gx:
            continue
        gy = cores // gx
        if gy <= gy_max:
            out.append((gx, gy))
    return out


def _pick_1d_in0_grid(
    device: ttnn.Device,
    *,
    n: int,
    preferred: tuple[int, int] | None,
) -> tuple[int, int]:
    """Divisor-legal (gx, gy) for 1D_in0 (cores must divide N-tiles)."""
    tile = ttnn.TILE_SIZE
    nt = n // tile
    grid = device.compute_with_storage_grid_size()
    gx_max, gy_max = int(grid.x), int(grid.y)
    max_cores = gx_max * gy_max

    if preferred is not None:
        px, py = preferred
        if px <= gx_max and py <= gy_max and px * py <= max_cores and nt % (px * py) == 0:
            return preferred

    cands: set[tuple[int, int]] = set()
    for c in range(min(nt, max_cores), 0, -1):
        if nt % c:
            continue
        cands.update(_grids_for_cores(c, gx_max, gy_max))
    if not cands:
        return 1, 1
    return max(cands, key=lambda g: (g[0] * g[1], g[0]))


def _ws_out_mem_config() -> ttnn.MemoryConfig:
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1)


def _ws_output_eligible(
    B: int,
    T: int,
    *,
    device: ttnn.Device,
    out_features: int,
    kind: str | None = None,
) -> bool:
    """Width-sharded matmul output: Kokoro (B=1, T=64) when N-tiles fit the sweep core grid."""
    if B != 1 or T != 64:
        return False
    tile = ttnn.TILE_SIZE
    n_tiles = int(out_features) // tile
    if int(out_features) % tile != 0:
        return False

    dev_grid = device.compute_with_storage_grid_size()
    gx_max, gy_max = int(dev_grid.x), int(dev_grid.y)

    if kind is not None:
        tuning = _T64_LINEAR_TUNING.get(kind)
        if tuning is not None:
            px, py = tuning[0]
            grid_cores = px * py
            # Width-sharding needs N-tiles spread evenly across the tuned grid;
            # per_core_N may be >1 (e.g. ffn_in: 64 tiles over 32 cores), so the
            # grid must divide n_tiles, not merely be >= it.
            if px <= gx_max and py <= gy_max and n_tiles % grid_cores == 0:
                return True
            return False

    return n_tiles <= gx_max * gy_max


def _linear_tuned(
    x: ttnn.Tensor,
    weight: ttnn.Tensor,
    *,
    bias: ttnn.Tensor,
    program_config: ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig,
    compute_kernel_config,
    memory_config: ttnn.MemoryConfig = ttnn.L1_MEMORY_CONFIG,
    use_ws_output: bool,
) -> ttnn.Tensor:
    """Sweep-tuned linear: L1 width-sharded output when eligible, else L1 interleaved."""
    if use_ws_output:
        out = ttnn.linear(
            x,
            weight,
            bias=bias,
            transpose_b=True,
            program_config=program_config,
            memory_config=_ws_out_mem_config(),
            compute_kernel_config=compute_kernel_config,
        )
        return maybe_reshard_to_caller(out, memory_config)
    out = ttnn.linear(
        x,
        weight,
        bias=bias,
        transpose_b=True,
        program_config=program_config,
        memory_config=memory_config,
        compute_kernel_config=compute_kernel_config,
    )
    return out


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
    embedding_dtype=ttnn.bfloat16,
) -> TTCustomAlbertParams:
    """Upload a ``transformers.AlbertModel`` (or ``CustomAlbert``) to device.

    ``embedding_dtype`` must stay ``bfloat16``: ``ttnn.embedding`` requires BF16 weights on device.

    Embedding tables are stored ROW_MAJOR: ``ttnn.embedding`` row-gathers in row-major,
    so a TILE-layout table would be untilized on every forward (UntilizeWithUnpadding).
    """
    cfg = albert_model.config
    emb = albert_model.embeddings

    rm = ttnn.ROW_MAJOR_LAYOUT
    word_emb = _t(emb.word_embeddings.weight, device=device, dtype=embedding_dtype, layout=rm)
    pos_emb = _t(emb.position_embeddings.weight, device=device, dtype=embedding_dtype, layout=rm)
    token_type_emb = _t(emb.token_type_embeddings.weight, device=device, dtype=embedding_dtype, layout=rm)
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


def _is_l1_interleaved(mc: ttnn.MemoryConfig) -> bool:
    return mc.buffer_type == ttnn.BufferType.L1 and mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED


def _ensure_dram_interleaved(x: ttnn.Tensor) -> ttnn.Tensor:
    """Hand off final BERT output to downstream ops (KModel) that expect DRAM activations."""
    mc = x.memory_config()
    if mc.buffer_type == ttnn.BufferType.DRAM and mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED:
        return x
    out = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
    if out is not x:
        ttnn.deallocate(x)
    return out


def _fix_b1th_after_linear(
    x: ttnn.Tensor,
    *,
    B: int,
    T: int,
    width: int,
    memory_config: ttnn.MemoryConfig = ttnn.L1_MEMORY_CONFIG,
) -> ttnn.Tensor:
    """``ttnn.linear`` may return ``[1, B, T, C]`` instead of ``[B, 1, T, C]``."""
    sh = list(x.shape)
    if len(sh) == 4 and int(sh[0]) == 1 and int(sh[1]) == B:
        return ttnn.permute(x, (1, 0, 2, 3), memory_config=memory_config)
    if len(sh) == 4 and int(sh[0]) == B and int(sh[1]) == 1:
        return ttnn.reshape(x, [B, 1, T, width], memory_config=memory_config)
    raise ValueError(f"unexpected linear output shape {sh} for B={B} T={T} width={width}")


def _to_b1th(
    x: ttnn.Tensor,
    *,
    B: int,
    T: int,
    width: int,
    memory_config: ttnn.MemoryConfig = ttnn.L1_MEMORY_CONFIG,
) -> ttnn.Tensor:
    """Normalize activations to ``[B, 1, T, width]`` for ``nlp_create_qkv_heads``."""
    while len(x.shape) > 4:
        x = ttnn.squeeze(x, 0)
    sh = list(x.shape)
    if len(sh) == 3:
        return ttnn.reshape(x, [B, 1, T, width], memory_config=memory_config)
    if len(sh) == 4 and int(sh[0]) == 1 and int(sh[1]) == B:
        return ttnn.permute(x, (1, 0, 2, 3), memory_config=memory_config)
    if len(sh) == 4 and int(sh[0]) == B and int(sh[1]) == 1:
        return ttnn.reshape(x, [B, 1, T, width], memory_config=memory_config)
    raise ValueError(f"unexpected shape {sh} for B={B} T={T} width={width}")


def _from_b1th(
    x: ttnn.Tensor,
    *,
    B: int,
    T: int,
    hidden_size: int,
    memory_config: ttnn.MemoryConfig = ttnn.L1_MEMORY_CONFIG,
) -> ttnn.Tensor:
    """``[B, 1, T, H]`` or ``[1, B, T, H]`` -> ``[B, T, H]`` TILE."""
    while len(x.shape) > 4:
        x = ttnn.squeeze(x, 0)
    sh = list(x.shape)
    if len(sh) == 3:
        return ttnn.reshape(x, [B, T, hidden_size], memory_config=memory_config)
    if len(sh) == 4 and int(sh[0]) == 1 and int(sh[1]) == B:
        x = ttnn.permute(x, (1, 0, 2, 3), memory_config=memory_config)
    elif not (len(sh) == 4 and int(sh[0]) == B and int(sh[1]) == 1):
        raise ValueError(f"unexpected shape {sh} for B={B} T={T} H={hidden_size}")
    return ttnn.reshape(x, [B, T, hidden_size], memory_config=memory_config)


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
    grid_xy: tuple[int, int] | None = None,
    in0_block_w: int | None = None,
    width_sharded_out: bool = False,
) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig:
    """DRAM multicast-1D config aligned with ``get_mcast_1d_config`` (``fuse_batch=False``)."""
    tile = ttnn.TILE_SIZE
    if grid_xy is not None:
        grid_x, grid_y = grid_xy
    else:
        grid_size = device.compute_with_storage_grid_size()
        grid_x, grid_y = grid_size.x, grid_size.y
    num_cores = grid_x * grid_y

    per_core_m = max(1, math.ceil(seq_len / tile))
    per_core_n = max(1, math.ceil(math.ceil(n / num_cores) / tile))
    if in0_block_w is None:
        in0_block_w = 2 if (k // tile) % 2 == 0 else 1

    max_subblock = 4 if fp32_dest_acc_en else 8
    if width_sharded_out:
        out_subblock_h = 1
        out_subblock_w = max([i for i in range(1, max_subblock + 1) if per_core_n % i == 0], default=1)
    else:
        out_subblock_w = max([i for i in range(1, max_subblock + 1) if per_core_n % i == 0], default=1)
        out_subblock_h = max(
            [i for i in range(1, max_subblock + 1) if per_core_m % i == 0 and i * out_subblock_w <= max_subblock],
            default=1,
        )
    if prefer_subblock_w is not None and per_core_n % prefer_subblock_w == 0:
        out_subblock_w = prefer_subblock_w
    if (
        not width_sharded_out
        and prefer_subblock_h is not None
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


def _tuned_1d_linear_program_config(
    device: ttnn.Device,
    *,
    batch: int,
    seq_len: int,
    k: int,
    n: int,
    kind: str,
    fused_activation=None,
    fp32_dest_acc_en: bool = True,
    width_sharded_out: bool = True,
) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig:
    """Build 1D_in0 program config; apply sweep grids only for Kokoro (B=1, T=64)."""
    use_t64 = batch == 1 and seq_len == 64
    ws_out = width_sharded_out and _ws_output_eligible(batch, seq_len, device=device, out_features=n, kind=kind)
    prefer_subblock_h = 2 if kind == "qkv" and seq_len >= ttnn.TILE_SIZE else None
    if kind == "qkv" and seq_len < ttnn.TILE_SIZE:
        prefer_subblock_h = 1

    # Resolve the sweep-tuned (grid, in0_block_w) independent of whether the output
    # is width-sharded: an interleaved-output op (e.g. ffn_out) should still run on
    # its tuned grid rather than dropping to the full-grid fallback.
    tuning = _T64_LINEAR_TUNING.get(kind) if use_t64 else None
    grid_xy = tuning[0] if tuning else None
    ibw = tuning[1] if tuning else None
    if grid_xy is not None:
        px, py = grid_xy
        dev_grid = device.compute_with_storage_grid_size()
        max_cores = int(dev_grid.x) * int(dev_grid.y)
        if px > int(dev_grid.x) or py > int(dev_grid.y) or px * py > max_cores:
            grid_xy = None
            ibw = None

    return _mcast_1d_dram_program_config(
        device,
        seq_len=seq_len,
        k=k,
        n=n,
        fused_activation=fused_activation,
        fp32_dest_acc_en=fp32_dest_acc_en,
        prefer_subblock_h=prefer_subblock_h,
        grid_xy=grid_xy,
        in0_block_w=ibw,
        width_sharded_out=ws_out,
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


@dataclass(frozen=True)
class TTAlbertWidthShardedLnConfig:
    """L1 WIDTH-sharded LayerNorm: full ``M=B×T`` rows, hidden dim split across cores."""

    input_mem_config: ttnn.MemoryConfig
    program_config: ttnn.LayerNormShardedMultiCoreProgramConfig


TTAlbertLnConfig = Union[TTAlbertWidthShardedLnConfig, ttnn.LayerNormDefaultProgramConfig]


def _default_ln_program_config() -> ttnn.LayerNormDefaultProgramConfig:
    return ttnn.LayerNormDefaultProgramConfig()


def _ln_uses_width_sharding(B: int, T: int, normalized_size: int) -> bool:
    """WIDTH-sharded LN needs tile-aligned ``B×T`` and hidden/embedding dim."""
    tile = ttnn.TILE_SIZE
    return (B * T) % tile == 0 and normalized_size % tile == 0


def _ln_width_num_cores(normalized_size: int, *, max_cores: int) -> int:
    kt = normalized_size // ttnn.TILE_SIZE
    for n in range(min(kt, max_cores), 0, -1):
        if kt % n == 0:
            return n
    return 1


def _ln_core_grid(num_cores: int, device: ttnn.Device) -> ttnn.CoreGrid:
    grid = device.compute_with_storage_grid_size()
    max_x, max_y = int(grid.x), int(grid.y)
    for y in range(max_y, 0, -1):
        if num_cores % y == 0:
            x = num_cores // y
            if x <= max_x:
                return ttnn.CoreGrid(x=x, y=y)
    raise ValueError(f"cannot place {num_cores} LN cores on {max_x}x{max_y} grid")


def _build_width_sharded_ln_config(
    device: ttnn.Device,
    *,
    batch_rows: int,
    normalized_size: int,
) -> TTAlbertWidthShardedLnConfig:
    tile = ttnn.TILE_SIZE
    if batch_rows % tile != 0:
        raise ValueError(f"batch_rows {batch_rows} must be divisible by TILE_SIZE {tile}")
    if normalized_size % tile != 0:
        raise ValueError(f"normalized_size {normalized_size} must be divisible by TILE_SIZE {tile}")

    grid = device.compute_with_storage_grid_size()
    num_cores = _ln_width_num_cores(normalized_size, max_cores=int(grid.x) * int(grid.y))
    core_grid = _ln_core_grid(num_cores, device)
    shard_width = normalized_size // num_cores
    input_mem_config = ttnn.create_sharded_memory_config(
        (batch_rows, shard_width),
        core_grid,
        ttnn.ShardStrategy.WIDTH,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    block_h = batch_rows // tile
    block_w = shard_width // tile
    subblock_w = _largest_divisor(block_w, max_divisor=4)
    device_grid = device.compute_with_storage_grid_size()
    program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=(int(device_grid.x), int(device_grid.y)),
        subblock_w=subblock_w,
        block_h=block_h,
        block_w=block_w,
        inplace=False,
    )
    return TTAlbertWidthShardedLnConfig(
        input_mem_config=input_mem_config,
        program_config=program_config,
    )


def _width_sharded_layer_norm(
    x: ttnn.Tensor,
    *,
    weight: ttnn.Tensor,
    bias: ttnn.Tensor,
    epsilon: float,
    ln_cfg: TTAlbertWidthShardedLnConfig,
    compute_kernel_config,
    output_mem_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    """Interleaved activations -> WIDTH-sharded LN -> interleaved output."""
    mc = x.memory_config()
    if mc.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED and mc.buffer_type == ttnn.BufferType.L1:
        sharded = x
    else:
        sharded = ttnn.to_memory_config(x, ln_cfg.input_mem_config)
        if sharded is not x:
            ttnn.deallocate(x)

    normed = ttnn.layer_norm(
        sharded,
        weight=weight,
        bias=bias,
        epsilon=epsilon,
        program_config=ln_cfg.program_config,
        memory_config=ln_cfg.input_mem_config,
        compute_kernel_config=compute_kernel_config,
    )
    if normed is not sharded:
        ttnn.deallocate(sharded)

    return ttnn.sharded_to_interleaved(normed, memory_config=output_mem_config)


def _apply_layer_norm(
    x: ttnn.Tensor,
    *,
    weight: ttnn.Tensor,
    bias: ttnn.Tensor,
    epsilon: float,
    ln_cfg: TTAlbertLnConfig,
    compute_kernel_config,
    output_mem_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    if isinstance(ln_cfg, TTAlbertWidthShardedLnConfig):
        return _width_sharded_layer_norm(
            x,
            weight=weight,
            bias=bias,
            epsilon=epsilon,
            ln_cfg=ln_cfg,
            compute_kernel_config=compute_kernel_config,
            output_mem_config=output_mem_config,
        )
    out = ttnn.layer_norm(
        x,
        weight=weight,
        bias=bias,
        epsilon=epsilon,
        program_config=ln_cfg,
        memory_config=output_mem_config,
        compute_kernel_config=compute_kernel_config,
    )
    if out is not x:
        ttnn.deallocate(x)
    return out


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
    B: int,
    T: int,
    hidden_size: int,
    head_size: int,
    embedding_size: int,
    intermediate_size: int,
    fp32_dest_acc_en: bool = True,
) -> TTAlbertMatmulProgramConfigs:
    return TTAlbertMatmulProgramConfigs(
        qkv=_tuned_1d_linear_program_config(
            device,
            batch=B,
            seq_len=T,
            k=hidden_size,
            n=3 * hidden_size,
            kind="qkv",
            fp32_dest_acc_en=fp32_dest_acc_en,
            width_sharded_out=True,
        ),
        dense=_tuned_1d_linear_program_config(
            device,
            batch=B,
            seq_len=T,
            k=hidden_size,
            n=hidden_size,
            kind="dense",
            fp32_dest_acc_en=fp32_dest_acc_en,
            width_sharded_out=True,
        ),
        ffn_in=_tuned_1d_linear_program_config(
            device,
            batch=B,
            seq_len=T,
            k=hidden_size,
            n=intermediate_size,
            kind="ffn_in",
            fused_activation=_FFN_FUSED_GELU,
            fp32_dest_acc_en=fp32_dest_acc_en,
            width_sharded_out=True,
        ),
        ffn_out=_tuned_1d_linear_program_config(
            device,
            batch=B,
            seq_len=T,
            k=intermediate_size,
            n=hidden_size,
            kind="ffn_out",
            fp32_dest_acc_en=fp32_dest_acc_en,
            width_sharded_out=False,
        ),
        emb_map=_tuned_1d_linear_program_config(
            device,
            batch=B,
            seq_len=T,
            k=embedding_size,
            n=hidden_size,
            kind="emb_map",
            fp32_dest_acc_en=fp32_dest_acc_en,
            width_sharded_out=True,
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
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )


# --- modules ----------------------------------------------------------------


class TTAlbertLayer:
    """One ALBERT layer: fused QKV + manual attention -> FFN, each with residual + LayerNorm."""

    __slots__ = (
        "params",
        "device",
        "num_heads",
        "head_size",
        "hidden_size",
        "layer_norm_eps",
        "compute_kernel_config",
    )

    def __init__(
        self,
        params: TTAlbertLayerParams,
        *,
        num_heads: int,
        hidden_size: int,
        layer_norm_eps: float,
        compute_kernel_config,
        device: ttnn.Device,
    ) -> None:
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_attention_heads"
        self.params = params
        self.device = device
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
        ln_cfg: TTAlbertLnConfig,
        memory_config: ttnn.MemoryConfig = ttnn.L1_MEMORY_CONFIG,
    ) -> ttnn.Tensor:
        p = self.params
        x_shape = list(x.shape)
        B, T = int(x_shape[-3]), int(x_shape[-2])

        if not _is_l1_interleaved(x.memory_config()):
            x_l1 = ttnn.to_memory_config(x, memory_config)
            if x_l1 is not x:
                ttnn.deallocate(x)
            x = x_l1

        x_residual = ttnn.clone(x, memory_config=memory_config)
        x_b1th = _to_b1th(x, B=B, T=T, width=self.hidden_size, memory_config=memory_config)
        xqkv = _linear_tuned(
            x_b1th,
            p.qkv_w,
            bias=p.qkv_b,
            program_config=matmul_pcs.qkv,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=memory_config,
            use_ws_output=_ws_output_eligible(B, T, device=self.device, out_features=3 * self.hidden_size, kind="qkv"),
        )
        xqkv = _fix_b1th_after_linear(xqkv, B=B, T=T, width=3 * self.hidden_size, memory_config=memory_config)

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

        projected = _linear_tuned(
            ctx_b1th,
            p.dense_w,
            bias=p.dense_b,
            program_config=matmul_pcs.dense,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=memory_config,
            use_ws_output=_ws_output_eligible(B, T, device=self.device, out_features=self.hidden_size, kind="dense"),
        )
        ttnn.deallocate(ctx_b1th)
        projected_btH = _from_b1th(projected, B=B, T=T, hidden_size=self.hidden_size, memory_config=memory_config)
        ttnn.deallocate(projected)

        residual = ttnn.add(x_residual, projected_btH, memory_config=memory_config)
        ttnn.deallocate(x_residual)
        ttnn.deallocate(projected_btH)

        out = _apply_layer_norm(
            residual,
            weight=p.attn_ln_w,
            bias=p.attn_ln_b,
            epsilon=self.layer_norm_eps,
            ln_cfg=ln_cfg,
            compute_kernel_config=self.compute_kernel_config,
            output_mem_config=memory_config,
        )
        return out

    def _ffn(
        self,
        x: ttnn.Tensor,
        *,
        matmul_pcs: TTAlbertMatmulProgramConfigs,
        ln_cfg: TTAlbertLnConfig,
        memory_config: ttnn.MemoryConfig = ttnn.L1_MEMORY_CONFIG,
    ) -> ttnn.Tensor:
        p = self.params
        x_shape = list(x.shape)
        B, T = int(x_shape[-3]), int(x_shape[-2])
        if not _is_l1_interleaved(x.memory_config()):
            x_l1 = ttnn.to_memory_config(x, memory_config)
            if x_l1 is not x:
                ttnn.deallocate(x)
            x = x_l1

        x_residual = ttnn.clone(x, memory_config=memory_config)
        h_act = _linear_tuned(
            x,
            p.ffn_w,
            bias=p.ffn_b,
            program_config=matmul_pcs.ffn_in,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=memory_config,
            use_ws_output=_ws_output_eligible(
                B, T, device=self.device, out_features=int(p.ffn_w.shape[-2]), kind="ffn_in"
            ),
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
        residual = ttnn.add(h2, x_residual, memory_config=memory_config)
        ttnn.deallocate(x_residual)
        ttnn.deallocate(h2)
        out = _apply_layer_norm(
            residual,
            weight=p.full_ln_w,
            bias=p.full_ln_b,
            epsilon=self.layer_norm_eps,
            ln_cfg=ln_cfg,
            compute_kernel_config=self.compute_kernel_config,
            output_mem_config=memory_config,
        )
        return out

    def forward(
        self,
        x: ttnn.Tensor,
        attention_mask: ttnn.Tensor,
        *,
        matmul_pcs: TTAlbertMatmulProgramConfigs,
        ln_cfg: TTAlbertLnConfig,
    ) -> ttnn.Tensor:
        a = self._attention(x, attention_mask, matmul_pcs=matmul_pcs, ln_cfg=ln_cfg)
        o = self._ffn(a, matmul_pcs=matmul_pcs, ln_cfg=ln_cfg)
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
        self._intermediate_size = int(params.layer_groups[0][0].ffn_w.shape[-2])
        self._head_size = params.hidden_size // params.num_attention_heads
        self._matmul_pc_cache: dict[tuple[int, int], TTAlbertMatmulProgramConfigs] = {}
        self._ln_cfg_cache: dict[tuple[int, int, int], TTAlbertLnConfig] = {}
        self._layers: tuple[tuple[TTAlbertLayer, ...], ...] = tuple(
            tuple(
                TTAlbertLayer(
                    lp,
                    num_heads=params.num_attention_heads,
                    hidden_size=params.hidden_size,
                    layer_norm_eps=params.layer_norm_eps,
                    compute_kernel_config=self.compute_kernel_config,
                    device=device,
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
                B=B,
                T=T,
                hidden_size=self.params.hidden_size,
                head_size=self._head_size,
                embedding_size=self.params.embedding_size,
                intermediate_size=self._intermediate_size,
                fp32_dest_acc_en=True,
            )
        return self._matmul_pc_cache[key]

    def _ln_config(self, B: int, T: int, normalized_size: int) -> TTAlbertLnConfig:
        key = (B, T, normalized_size)
        if key not in self._ln_cfg_cache:
            if _ln_uses_width_sharding(B, T, normalized_size):
                self._ln_cfg_cache[key] = _build_width_sharded_ln_config(
                    self.device,
                    batch_rows=B * T,
                    normalized_size=normalized_size,
                )
            else:
                self._ln_cfg_cache[key] = _default_ln_program_config()
        return self._ln_cfg_cache[key]

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

        position_ids = torch.arange(T, dtype=torch.int32).unsqueeze(0).expand(B, -1)
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

        ln_cfg = self._ln_config(B, T, self.params.embedding_size)
        out = _apply_layer_norm(
            emb_sum,
            weight=self.params.emb_ln_w,
            bias=self.params.emb_ln_b,
            epsilon=self.params.layer_norm_eps,
            ln_cfg=ln_cfg,
            compute_kernel_config=self.compute_kernel_config,
            output_mem_config=ttnn.L1_MEMORY_CONFIG,
        )
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
            token_type_ids = torch.zeros(B, T, dtype=torch.int32)

        emb = self._embed(input_ids, token_type_ids)

        matmul_pcs = self._matmul_program_configs(B, T)
        ln_cfg = self._ln_config(B, T, self.params.hidden_size)

        hidden = _linear_tuned(
            emb,
            self.params.emb_map_w,
            bias=self.params.emb_map_b,
            program_config=matmul_pcs.emb_map,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            use_ws_output=_ws_output_eligible(
                B, T, device=self.device, out_features=self.params.hidden_size, kind="emb_map"
            ),
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
                ln_cfg=ln_cfg,
            )
            ttnn.deallocate(hidden)
            hidden = new_hidden

        ttnn.deallocate(ext_mask)

        while len(hidden.shape) > 3:
            hidden = ttnn.squeeze(hidden, 0)
        return _ensure_dram_interleaved(hidden)

    __call__ = forward
