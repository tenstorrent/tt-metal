# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Blackhole TP4 vision attention / MLP helpers (L1 activations, swept program configs).

Used by ``TTNNDotsVision*TP4BH`` subclasses and the TP4 block unit tests on P150x4 /
P300x2 meshes.  Hardware-swept winners target the BH 11×10 grid at S=11264.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import ttnn
from ttnn.operations.transformer import SDPAProgramConfig

if TYPE_CHECKING:
    from models.experimental.tt_symbiote.modules.dots_ocr_vision import (
        TTNNDotsVisionAttention,
        TTNNDotsVisionBlock,
        TTNNDotsVisionMLP,
    )

_DTYPE_TILE_BYTES = {
    ttnn.bfloat16: 2048,
    ttnn.bfloat8_b: 1088,
    ttnn.bfloat4_b: 544,
}

_L1_PER_CORE_BYTES = 1500 * 1024
_VISION_TP4_SEQ_LEN = 11264
_VISION_TP4_HIDDEN = 1536
_VISION_TP4_MERGED_SEQ = 2816  # SEQ_LEN // spatial_merge_size^2
_VISION_TP4_MLP_SIZE = 6144  # HIDDEN * spatial_merge_size^2


def _largest_divisor_le(value: int, limit: int) -> int:
    for c in range(min(value, limit), 0, -1):
        if value % c == 0:
            return c
    return 1


def _tensor_in_l1(t: ttnn.Tensor) -> bool:
    try:
        return t.memory_config().buffer_type == ttnn.BufferType.L1
    except Exception:
        return False


def ensure_l1_tensor(t: ttnn.Tensor, *, dtype: ttnn.DataType | None = None) -> ttnn.Tensor:
    """Tile-layout + L1 interleaved. Optionally cast dtype while moving."""
    l1 = ttnn.L1_MEMORY_CONFIG
    if t.layout != ttnn.TILE_LAYOUT:
        kwargs: dict = {"memory_config": l1}
        if dtype is not None:
            kwargs["dtype"] = dtype
        t = ttnn.to_layout(t, ttnn.TILE_LAYOUT, **kwargs)
    elif dtype is not None and t.dtype != dtype:
        t = ttnn.typecast(t, dtype, memory_config=l1)
    if not _tensor_in_l1(t):
        t = ttnn.to_memory_config(t, l1)
    return t


def rot_mats_l1(rot_mats: tuple[ttnn.Tensor, ttnn.Tensor]) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """RoPE cos/sin tables in L1 so rotary_embedding reads/writes L1 activations."""
    cos, sin = rot_mats
    return ensure_l1_tensor(cos), ensure_l1_tensor(sin)


def _best_dst_subblock(ob_h: int, per_core_n: int, *, dst_budget: int = 8) -> tuple[int, int, int]:
    """Return (out_subblock_h, out_subblock_w, dst_area) maximising DST register use."""
    best_area = 0
    best_h = best_w = 1
    for h in range(min(ob_h, dst_budget), 0, -1):
        if ob_h % h != 0:
            continue
        for w in range(min(per_core_n, dst_budget // h), 0, -1):
            if per_core_n % w != 0:
                continue
            area = h * w
            if area > best_area:
                best_area = area
                best_h = h
                best_w = w
    return best_h, best_w, best_area


def bh_tp4_matmul_pc(
    device,
    m_dim: int,
    k_dim: int,
    n_dim: int,
    *,
    in0_dtype: ttnn.DataType = ttnn.bfloat8_b,
    out_dtype: ttnn.DataType = ttnn.bfloat8_b,
    l1_resident_bytes_per_core: int = 0,
):
    """L1-aware 2D-mcast program config search for BH TP4 vision matmuls.

    CB model uses ``in0_tile_bytes`` / ``out_tile_bytes`` so BFP8 in0/out paths
    get tighter budgets than the legacy BF16-only estimator.
    """
    grid = device.compute_with_storage_grid_size()
    grid_x, grid_y = int(grid.x), int(grid.y)
    tile = 32

    if m_dim % tile or k_dim % tile or n_dim % tile:
        return None

    m_tiles = m_dim // tile
    k_tiles = k_dim // tile
    n_tiles = n_dim // tile

    in0_tile_b = _DTYPE_TILE_BYTES[in0_dtype]
    out_tile_b = _DTYPE_TILE_BYTES[out_dtype]
    in0_block_w = _largest_divisor_le(k_tiles, 8)
    cb_budget_bytes = max(256 * 1024, _L1_PER_CORE_BYTES - l1_resident_bytes_per_core)

    def _per_core_n_candidates(n_max: int) -> list[int]:
        cands = {n for n in range(1, min(n_tiles, 24) + 1) if n_tiles % n == 0}
        cands.add((n_tiles + n_max - 1) // n_max)
        return sorted(cands)

    best_pc = None
    best_score = (-1, -1, -(2**31), -(2**31))

    for transpose_mcast in (True, False):
        m_grid_max = grid_x if transpose_mcast else grid_y
        n_grid_max = grid_y if transpose_mcast else grid_x

        for eff_mg in range(min(m_tiles, m_grid_max), 0, -1):
            if m_tiles % eff_mg != 0:
                continue
            per_core_m = m_tiles // eff_mg
            if per_core_m > 64:
                continue

            for per_core_n in _per_core_n_candidates(n_grid_max):
                if per_core_n > 24:
                    continue
                actual_ng = (n_tiles + per_core_n - 1) // per_core_n
                if actual_ng > n_grid_max:
                    continue

                in1_fixed = 2 * in0_block_w * per_core_n * 1088
                partial_fixed = per_core_m * per_core_n * 2048
                fixed_cb_bytes = in1_fixed + partial_fixed
                if fixed_cb_bytes >= cb_budget_bytes:
                    continue

                remaining_bytes = cb_budget_bytes - fixed_cb_bytes
                best_ob_h = 0
                best_sub = (1, 1, 0)

                for ob_h in range(per_core_m, 0, -1):
                    if per_core_m % ob_h != 0:
                        continue
                    in0_bytes = 2 * ob_h * in0_block_w * in0_tile_b
                    interm_bytes = ob_h * per_core_n * 2048
                    out_bytes = 2 * ob_h * per_core_n * out_tile_b
                    if in0_bytes + interm_bytes + out_bytes > remaining_bytes:
                        continue
                    sub = _best_dst_subblock(ob_h, per_core_n)
                    if sub[2] > best_sub[2] or (sub[2] == best_sub[2] and ob_h < best_ob_h):
                        best_ob_h = ob_h
                        best_sub = sub

                if best_ob_h == 0:
                    continue

                sub_h, sub_w, dst_area = best_sub
                total_cores = eff_mg * actual_ng
                score = (dst_area, int(not transpose_mcast), -best_ob_h, -total_cores)

                if score > best_score:
                    best_score = score
                    gx = eff_mg if transpose_mcast else actual_ng
                    gy = actual_ng if transpose_mcast else eff_mg
                    best_pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                        compute_with_storage_grid_size=(gx, gy),
                        in0_block_w=in0_block_w,
                        out_subblock_h=sub_h,
                        out_subblock_w=sub_w,
                        out_block_h=best_ob_h,
                        out_block_w=per_core_n,
                        per_core_M=per_core_m,
                        per_core_N=per_core_n,
                        transpose_mcast=transpose_mcast,
                        fused_activation=None,
                        fuse_batch=False,
                    )

    return best_pc


def _l1_shard_bytes_per_core(device, m_dim: int, n_dim: int, dtype: ttnn.DataType) -> int:
    grid = device.compute_with_storage_grid_size()
    nc = int(grid.x) * int(grid.y)
    tile = 32
    mt = m_dim // tile
    nt = n_dim // tile
    return ((mt * nt + nc - 1) // nc) * _DTYPE_TILE_BYTES[dtype]


def bh_tp4_qkv_pc(device):
    """Hardware-swept QKV matmul for 11264×1536×1152 on BH P150 11×10 (~148 μs).

    Silicon sweep 2026-06-07 (bf8 in0/out, norm output L1 resident):
    grid=(9,8) tm=False M=44 N=4 obh=22 ibw=8 sub=(2,4).
    """
    grid = device.compute_with_storage_grid_size()
    if int(grid.x) == 11 and int(grid.y) == 10:
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(9, 8),
            in0_block_w=8,
            out_subblock_h=2,
            out_subblock_w=4,
            out_block_h=22,
            out_block_w=4,
            per_core_M=44,
            per_core_N=4,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
        )
    return None


def bh_tp4_o_proj_pc(device, *, seq_len: int = _VISION_TP4_SEQ_LEN, ctx_dim: int = 384):
    """Hardware-swept o_proj for 11264×384×1536, BFP8×BFP8→BFP8 L1 (~80 μs).

    Silicon sweep 2026-06-07: grid=(8,8) tm=False M=44 N=6 obh=22 ibw=6 sub=(2,3).
    BFP8 matmul out removes the TypecastDeviceOperation before the bf8 residual add.
    """
    grid = device.compute_with_storage_grid_size()
    ctx_resident = _l1_shard_bytes_per_core(device, seq_len, ctx_dim, ttnn.bfloat8_b)
    if int(grid.x) == 11 and int(grid.y) == 10:
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=6,
            out_subblock_h=2,
            out_subblock_w=3,
            out_block_h=22,
            out_block_w=6,
            per_core_M=44,
            per_core_N=6,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
        )
    return bh_tp4_matmul_pc(
        device,
        seq_len,
        ctx_dim,
        _VISION_TP4_HIDDEN,
        in0_dtype=ttnn.bfloat8_b,
        out_dtype=ttnn.bfloat8_b,
        l1_resident_bytes_per_core=ctx_resident,
    )


def bh_tp4_mlp_gate_up_pc(device):
    """Gate / up matmul for 11264×1536×1056 on BH P150 11×10 (~134–138 μs).

    Silicon sweep 2026-06-07: grid=(11,8) tm=False M=44 N=3 obh=22 ibw=8 sub=(2,3).
    Shared by fc1 and fc3 (bf8 / bf4 out differ only in packer path).
    """
    grid = device.compute_with_storage_grid_size()
    if int(grid.x) == 11 and int(grid.y) == 10:
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(11, 8),
            in0_block_w=8,
            out_subblock_h=2,
            out_subblock_w=3,
            out_block_h=22,
            out_block_w=3,
            per_core_M=44,
            per_core_N=3,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
        )
    return None


def bh_tp4_mlp_down_pc(device, *, seq_len: int = _VISION_TP4_SEQ_LEN, itp: int = 1056):
    """Down matmul for 11264×1056×1536 on BH P150 11×10 (~166 μs).

    Silicon sweep 2026-06-07: grid=(8,8) tm=False M=44 N=6 obh=22 ibw=3 sub=(2,3).
    """
    grid = device.compute_with_storage_grid_size()
    gum_resident = _l1_shard_bytes_per_core(device, seq_len, itp, ttnn.bfloat8_b)
    if int(grid.x) == 11 and int(grid.y) == 10:
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=3,
            out_subblock_h=2,
            out_subblock_w=3,
            out_block_h=22,
            out_block_w=6,
            per_core_M=44,
            per_core_N=6,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
        )
    return bh_tp4_matmul_pc(
        device,
        seq_len,
        itp,
        _VISION_TP4_HIDDEN,
        in0_dtype=ttnn.bfloat8_b,
        out_dtype=ttnn.bfloat8_b,
        l1_resident_bytes_per_core=gum_resident,
    )


_VISION_TP4_PATCH_K = 608  # 588 patch features padded to tile multiple (32)


def bh_tp4_patch_embed_pc(device):
    """Patch-embed projection ``11264×608×1536`` on BH P150 11×10 (~131 μs).

    Silicon sweep 2026-06-07 (BFP8 L1 in0/out): auto ~2145 μs → swept ~131 μs.
    ``perf_tp1vt.txt`` baseline was ~2198 μs with HiFi4 BF16 in0 and no PC.

    Winner: grid=(11,8) tm=True M=32 N=6 obh=32 ibw=1 sub=(8,1).
    """
    grid = device.compute_with_storage_grid_size()
    if int(grid.x) == 11 and int(grid.y) == 10:
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(11, 8),
            in0_block_w=1,
            out_subblock_h=8,
            out_subblock_w=1,
            out_block_h=32,
            out_block_w=6,
            per_core_M=32,
            per_core_N=6,
            transpose_mcast=True,
            fused_activation=None,
            fuse_batch=False,
        )
    return bh_tp4_matmul_pc(
        device,
        _VISION_TP4_SEQ_LEN,
        _VISION_TP4_PATCH_K,
        _VISION_TP4_HIDDEN,
        in0_dtype=ttnn.bfloat8_b,
        out_dtype=ttnn.bfloat8_b,
        l1_resident_bytes_per_core=0,
    )


def bh_tp4_merger_fc2_pc(device):
    """Patch-merger fc2 ``2816×6144×384`` (col-sharded out) on BH P150 11×10.

    Microbench with BFP8 L1 in0/out and LoFi: auto-config ~140 μs beats
    every ``bh_tp4_matmul_pc`` candidate (~900 μs).  Return ``None`` so the
    forward path keeps auto-config.  (``perf_tp1vt.txt``'s ~1383 μs was HiFi4
    without an explicit PC in an older trace.)
    """
    _ = device
    return None


def bh_tp4_merger_fc1_pc(device):
    """Patch merger fc1 for 2816×6144×6144 on BH P150 11×10 (~566 μs).

    Silicon sweep 2026-06-07: grid=(11,8) tm=True M=8 N=24 obh=8 ibw=4 sub=(2,4).
    Auto-config was ~17.8 ms on the same shape (L1 in0, DRAM weights).
    """
    grid = device.compute_with_storage_grid_size()
    if int(grid.x) == 11 and int(grid.y) == 10:
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(11, 8),
            in0_block_w=4,
            out_subblock_h=2,
            out_subblock_w=4,
            out_block_h=8,
            out_block_w=24,
            per_core_M=8,
            per_core_N=24,
            transpose_mcast=True,
            fused_activation=None,
            fuse_batch=False,
        )
    return bh_tp4_matmul_pc(
        device,
        _VISION_TP4_MERGED_SEQ,
        _VISION_TP4_MLP_SIZE,
        _VISION_TP4_MLP_SIZE,
        in0_dtype=ttnn.bfloat8_b,
        out_dtype=ttnn.bfloat8_b,
        l1_resident_bytes_per_core=0,
    )


def bh_tp4_sdpa_pc(device) -> SDPAProgramConfig:
    """Hardware-swept SDPA for TP4 vision attn on BH P150 11×10."""
    grid = device.compute_with_storage_grid_size()
    q_chunk, k_chunk = (128, 1024) if int(grid.x) == 11 and int(grid.y) == 10 else (256, 1024)
    return SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
        q_chunk_size=q_chunk,
        k_chunk_size=k_chunk,
        exp_approx_mode=True,
    )


def bh_tp4_vision_mlp_pc(
    device,
    m_dim: int,
    k_dim: int,
    n_dim: int,
    *,
    l1_resident_bytes_per_core: int = 0,
):
    """Optimal 2D-mcast program config for BH TP4 vision MLP matmuls."""
    grid = device.compute_with_storage_grid_size()
    grid_x, grid_y = int(grid.x), int(grid.y)
    tile = 32

    if m_dim % tile or k_dim % tile or n_dim % tile:
        return None

    m_tiles = m_dim // tile
    k_tiles = k_dim // tile
    n_tiles = n_dim // tile

    in0_block_w = _largest_divisor_le(k_tiles, 8)
    cb_budget_bytes = max(256 * 1024, _L1_PER_CORE_BYTES - l1_resident_bytes_per_core)

    best_pc = None
    best_score = (-(2**31), -1, -1)

    for transpose_mcast in (True, False):
        m_grid_max = grid_x if transpose_mcast else grid_y
        n_grid_max = grid_y if transpose_mcast else grid_x

        eff_mg = m_grid_max
        while eff_mg > 1 and m_tiles % eff_mg != 0:
            eff_mg -= 1

        per_core_m = m_tiles // eff_mg
        per_core_n = (n_tiles + n_grid_max - 1) // n_grid_max
        actual_ng = (n_tiles + per_core_n - 1) // per_core_n

        if per_core_n > 24 or per_core_m > 64:
            continue

        in1_fixed = 2 * in0_block_w * per_core_n * 1088
        partial_fixed = per_core_m * per_core_n * 2048
        fixed_cb_bytes = in1_fixed + partial_fixed

        if fixed_cb_bytes >= cb_budget_bytes:
            continue

        remaining_bytes = cb_budget_bytes - fixed_cb_bytes
        divisors = sorted([h for h in range(1, per_core_m + 1) if per_core_m % h == 0], reverse=True)

        for ob_h in divisors:
            in0_bytes = 2 * ob_h * in0_block_w * 2048
            interm_bytes = ob_h * per_core_n * 2048
            out_bytes = ob_h * per_core_n * 2048
            if in0_bytes + interm_bytes + out_bytes > remaining_bytes:
                continue

            cand_area = 0
            cand_h = cand_w = 1
            dst = 8
            for h in range(min(ob_h, dst), 0, -1):
                if ob_h % h != 0:
                    continue
                for w in range(min(per_core_n, dst // h), 0, -1):
                    if per_core_n % w != 0:
                        continue
                    if h * w > cand_area:
                        cand_area = h * w
                        cand_h = h
                        cand_w = w
                    break

            outer_m_iters = per_core_m // ob_h
            total_cores = eff_mg * actual_ng
            score = (-outer_m_iters, cand_area, total_cores)

            if score > best_score:
                best_score = score
                gx = eff_mg if transpose_mcast else actual_ng
                gy = actual_ng if transpose_mcast else eff_mg
                best_pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(gx, gy),
                    in0_block_w=in0_block_w,
                    out_subblock_h=cand_h,
                    out_subblock_w=cand_w,
                    out_block_h=ob_h,
                    out_block_w=per_core_n,
                    per_core_M=per_core_m,
                    per_core_N=per_core_n,
                    transpose_mcast=transpose_mcast,
                    fused_activation=None,
                    fuse_batch=False,
                )
            break

    return best_pc


def init_attn_tp4_bh_configs(
    attn: TTNNDotsVisionAttention,
    *,
    seq_len: int = _VISION_TP4_SEQ_LEN,
    hidden: int = _VISION_TP4_HIDDEN,
) -> None:
    """Attach hardware-swept matmul / SDPA program configs to a TP4 attention module."""
    device = attn.device
    ndev = int(getattr(attn, "_tp_ndev", 1))
    heads_per_tp = attn.num_heads // ndev
    ctx_dim = heads_per_tp * attn.head_dim
    qkv_out = ctx_dim * 3

    x_resident = _l1_shard_bytes_per_core(device, seq_len, hidden, ttnn.bfloat8_b)
    qkv_resident = _l1_shard_bytes_per_core(device, seq_len, qkv_out // ndev, ttnn.bfloat8_b)

    attn._bh_tp4_qkv_pc = bh_tp4_qkv_pc(device) or bh_tp4_matmul_pc(
        device,
        seq_len,
        hidden,
        qkv_out // ndev,
        in0_dtype=ttnn.bfloat8_b,
        out_dtype=ttnn.bfloat8_b,
        l1_resident_bytes_per_core=x_resident + qkv_resident,
    )
    attn._bh_tp4_o_pc = bh_tp4_o_proj_pc(device, seq_len=seq_len, ctx_dim=ctx_dim)
    attn._bh_tp4_sdpa_pc = bh_tp4_sdpa_pc(device)
    attn._bh_tp4_seq_len = seq_len


def init_patch_embed_tp4_bh_configs(patch_embed) -> None:
    """Attach swept program config for the patch-embed projection matmul."""
    device = patch_embed.device
    patch_embed._bh_tp4_proj_pc = bh_tp4_patch_embed_pc(device)


def init_merger_tp4_bh_configs(merger) -> None:
    """Attach swept program configs for patch-merger fc1/fc2 matmuls."""
    device = merger.device
    merger._bh_tp4_merger_fc1_pc = bh_tp4_merger_fc1_pc(device)
    merger._bh_tp4_merger_fc2_pc = bh_tp4_merger_fc2_pc(device)


def init_mlp_tp4_bh_configs(
    mlp: TTNNDotsVisionMLP,
    *,
    seq_len: int = _VISION_TP4_SEQ_LEN,
    hidden: int = _VISION_TP4_HIDDEN,
) -> None:
    """Attach hardware-swept matmul program configs to a TP4 MLP module."""
    device = mlp.device
    itp = int(mlp._intermediate_size) // int(mlp._tp_ndev)
    gate_up = bh_tp4_mlp_gate_up_pc(device)
    mlp._bh_tp4_gate_pc = gate_up or bh_tp4_vision_mlp_pc(device, seq_len, hidden, itp)
    mlp._bh_tp4_up_pc = gate_up or bh_tp4_vision_mlp_pc(device, seq_len, hidden, itp)
    mlp._bh_tp4_down_pc = bh_tp4_mlp_down_pc(device, seq_len=seq_len, itp=itp) or bh_tp4_vision_mlp_pc(
        device, seq_len, itp, hidden
    )


def vision_attn_tp4_bh_forward(
    attn: TTNNDotsVisionAttention,
    hidden_states: ttnn.Tensor,
    rot_mats: tuple[ttnn.Tensor, ttnn.Tensor] | None = None,
    cu_seqlens=None,
    attention_mask: ttnn.Tensor | None = None,
    attention_logical_seq_len: int | None = None,
) -> ttnn.Tensor:
    """TP4 BH vision attention: L1 activations, Ring CCL, swept matmul / SDPA configs."""
    _ = cu_seqlens, attention_mask, attention_logical_seq_len
    l1 = ttnn.L1_MEMORY_CONFIG
    ndev = int(getattr(attn, "_tp_ndev", 1))
    heads_per_tp = attn.num_heads // ndev
    head_dim = attn.head_dim

    x = ensure_l1_tensor(hidden_states)

    qkv_pc = getattr(attn, "_bh_tp4_qkv_pc", None)
    o_pc = getattr(attn, "_bh_tp4_o_pc", None)
    sdpa_cfg = getattr(attn, "_bh_tp4_sdpa_pc", None)

    qkv = ttnn.linear(
        x,
        attn.tt_qkv_weight,
        bias=attn.tt_qkv_bias,
        dtype=ttnn.bfloat8_b,
        memory_config=l1,
        compute_kernel_config=attn.compute_kernel_config,
        program_config=qkv_pc,
    )

    q, k, v = ttnn.experimental.nlp_create_qkv_heads(
        qkv,
        num_heads=heads_per_tp,
        num_kv_heads=heads_per_tp,
        transpose_k_heads=False,
        memory_config=l1,
    )
    ttnn.deallocate(qkv)

    if rot_mats is not None and len(rot_mats) == 2:
        cos, sin = rot_mats_l1(rot_mats)
        q = ttnn.experimental.rotary_embedding(q, cos, sin, memory_config=l1)
        k = ttnn.experimental.rotary_embedding(k, cos, sin, memory_config=l1)

    v = ttnn.typecast(v, ttnn.bfloat4_b, memory_config=l1)
    ctx = ttnn.transformer.scaled_dot_product_attention(
        q,
        k,
        v,
        is_causal=False,
        attn_mask=None,
        program_config=sdpa_cfg,
        compute_kernel_config=attn.sdpa_compute_kernel_config,
        memory_config=l1,
    )
    ttnn.deallocate(q)
    ttnn.deallocate(k)
    ttnn.deallocate(v)

    ctx = ttnn.experimental.nlp_concat_heads(ctx, memory_config=l1)

    # BFP8 out (bf16/bf8 in0 × bf8 weight): no separate TypecastDeviceOperation before CCL.
    partial = ttnn.linear(
        ctx,
        attn.tt_o_proj_weight,
        bias=None,
        dtype=ttnn.bfloat8_b,
        memory_config=l1,
        compute_kernel_config=attn.compute_kernel_config,
        program_config=o_pc,
    )
    ttnn.deallocate(ctx)

    scattered = ttnn.reduce_scatter(
        partial,
        dim=3,
        num_links=1,
        cluster_axis=1,
        memory_config=l1,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(partial)

    out = ttnn.all_gather(
        scattered,
        dim=3,
        num_links=1,
        cluster_axis=1,
        memory_config=l1,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(scattered)

    if attn.tt_o_proj_bias is not None:
        out = ttnn.add(out, attn.tt_o_proj_bias, memory_config=l1)
    return out


def vision_mlp_tp4_bh_forward(mlp: TTNNDotsVisionMLP, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
    """TP4 BH vision SwiGLU MLP: N-shard fc1/fc3, K-shard fc2 + all_reduce, all L1."""
    l1 = ttnn.L1_MEMORY_CONFIG

    x = ensure_l1_tensor(hidden_states)

    gate_pc = getattr(mlp, "_bh_tp4_gate_pc", None)
    up_pc = getattr(mlp, "_bh_tp4_up_pc", None)
    down_pc = getattr(mlp, "_bh_tp4_down_pc", None)

    gate = ttnn.linear(
        x,
        mlp.tt_fc1_weight,
        bias=mlp.tt_fc1_bias,
        dtype=ttnn.bfloat8_b,
        memory_config=l1,
        compute_kernel_config=mlp.compute_kernel_config,
        program_config=gate_pc,
    )

    up = ttnn.linear(
        x,
        mlp.tt_fc3_weight,
        bias=mlp.tt_fc3_bias,
        dtype=ttnn.bfloat4_b,
        memory_config=l1,
        compute_kernel_config=mlp.compute_kernel_config,
        program_config=up_pc,
    )

    gate_up_mul = ttnn.mul(
        gate,
        up,
        input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
        fast_and_approximate_mode=True,
        dtype=ttnn.bfloat8_b,
        memory_config=l1,
    )
    ttnn.deallocate(gate)
    ttnn.deallocate(up)

    partial = ttnn.linear(
        gate_up_mul,
        mlp.tt_fc2_weight,
        bias=None,
        dtype=ttnn.bfloat8_b,
        memory_config=l1,
        compute_kernel_config=mlp.compute_kernel_config,
        program_config=down_pc,
    )
    ttnn.deallocate(gate_up_mul)

    out = ttnn.all_reduce(
        partial,
        num_links=1,
        cluster_axis=1,
        topology=ttnn.Topology.Ring,
        memory_config=l1,
    )
    ttnn.deallocate(partial)
    return out


def vision_block_tp4_bh_forward(
    block: TTNNDotsVisionBlock,
    hidden_states: ttnn.Tensor,
    rot_mats: tuple[ttnn.Tensor, ttnn.Tensor] | None = None,
    cu_seqlens=None,
    attention_mask: ttnn.Tensor | None = None,
    attention_logical_seq_len: int | None = None,
) -> ttnn.Tensor:
    """Full vision block with every local activation tensor in L1 interleaved memory."""
    l1 = ttnn.L1_MEMORY_CONFIG
    hidden_states = ensure_l1_tensor(hidden_states, dtype=ttnn.bfloat8_b)

    if rot_mats is not None:
        rot_mats = rot_mats_l1(rot_mats)

    residual = hidden_states
    normed = block.norm1(hidden_states, output_l1=True)
    attn_out = block.attn(
        normed,
        rot_mats=rot_mats,
        cu_seqlens=cu_seqlens,
        attention_mask=attention_mask,
        attention_logical_seq_len=attention_logical_seq_len,
    )
    ttnn.deallocate(normed)
    hidden_states = ttnn.add(residual, attn_out, dtype=ttnn.bfloat8_b, memory_config=l1)
    ttnn.deallocate(attn_out)
    ttnn.deallocate(residual)

    residual = hidden_states
    normed = block.norm2(hidden_states, output_l1=True)
    mlp_out = block.mlp(normed)
    ttnn.deallocate(normed)
    hidden_states = ttnn.add(residual, mlp_out, dtype=ttnn.bfloat8_b, memory_config=l1)
    ttnn.deallocate(mlp_out)
    ttnn.deallocate(residual)

    return hidden_states


def vision_patch_embed_tp4_bh_forward(
    patch_embed,
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor | None = None,
) -> ttnn.Tensor:
    """Patch embed with BFP8 L1 activations and swept projection matmul PC."""
    import torch.nn.functional as F

    l1 = ttnn.L1_MEMORY_CONFIG
    proj_pc = getattr(patch_embed, "_bh_tp4_proj_pc", None)
    mapper = ttnn.ReplicateTensorToMesh(patch_embed.device) if patch_embed.device.get_num_devices() > 1 else None
    compute_kc = patch_embed.vision_matmul_compute_kernel_config
    norm_kc = patch_embed.vision_norm_compute_kernel_config

    if pixel_values.dim() == 2:
        x = pixel_values.to(torch.bfloat16).unsqueeze(0).unsqueeze(0)
        k_padded = getattr(patch_embed, "_proj_k_padded", x.shape[-1])
        if k_padded != x.shape[-1]:
            x = F.pad(x, (0, k_padded - x.shape[-1]))
        x_tt = ttnn.from_torch(
            x,
            device=patch_embed.device,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=l1,
            mesh_mapper=mapper,
        )
        out = ttnn.linear(
            x_tt,
            patch_embed.tt_proj_weight,
            bias=patch_embed.tt_proj_bias,
            transpose_b=True,
            dtype=ttnn.bfloat8_b,
            memory_config=l1,
            program_config=proj_pc,
            compute_kernel_config=compute_kc,
        )
        ttnn.deallocate(x_tt)
        if patch_embed.tt_norm_weight is not None:
            out = ttnn.rms_norm(
                out,
                weight=patch_embed.tt_norm_weight,
                epsilon=1e-5,
                memory_config=l1,
                compute_kernel_config=norm_kc,
            )
        return ensure_l1_tensor(out, dtype=ttnn.bfloat8_b)

    B, C, H, W = pixel_values.shape
    if grid_thw is not None:
        g = grid_thw.detach().cpu() if hasattr(grid_thw, "is_cuda") and grid_thw.is_cuda else grid_thw
        if g.dim() == 1:
            g = g.unsqueeze(0)
        temporal = int(g[0, 0].item())
        height_patches = int(g[0, 1].item())
        width_patches = int(g[0, 2].item())
    else:
        temporal = 1
        height_patches = H // patch_embed.patch_size
        width_patches = W // patch_embed.patch_size

    num_patches = temporal * height_patches * width_patches
    temporal_patch_size = temporal
    x = pixel_values.view(-1, C, temporal_patch_size, patch_embed.patch_size, patch_embed.patch_size)
    x = x[:, :, 0]
    x = x.reshape(1, 1, num_patches, C * patch_embed.patch_size * patch_embed.patch_size).to(torch.bfloat16)

    k_padded = getattr(patch_embed, "_proj_k_padded", x.shape[-1])
    if k_padded != x.shape[-1]:
        x = F.pad(x, (0, k_padded - x.shape[-1]))

    x_tt = ttnn.from_torch(
        x,
        device=patch_embed.device,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=l1,
        mesh_mapper=mapper,
    )
    out = ttnn.linear(
        x_tt,
        patch_embed.tt_proj_weight,
        bias=patch_embed.tt_proj_bias,
        transpose_b=True,
        dtype=ttnn.bfloat8_b,
        memory_config=l1,
        program_config=proj_pc,
        compute_kernel_config=compute_kc,
    )
    ttnn.deallocate(x_tt)
    if patch_embed.tt_norm_weight is not None:
        out = ttnn.rms_norm(
            out,
            weight=patch_embed.tt_norm_weight,
            epsilon=1e-5,
            dtype=ttnn.bfloat8_b,
            memory_config=l1,
            compute_kernel_config=norm_kc,
        )
    return ensure_l1_tensor(out, dtype=ttnn.bfloat8_b)


def vision_patch_merger_tp4_bh_forward(merger, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
    """Patch merger with L1 interleaved activations (BH 11×10 DRAM-matmul fallback)."""
    l1 = ttnn.L1_MEMORY_CONFIG
    if not hasattr(merger, "_bh_tp4_merger_fc1_pc"):
        merger._bh_tp4_merger_fc1_pc = bh_tp4_merger_fc1_pc(merger.device)
    if not hasattr(merger, "_bh_tp4_merger_fc2_pc"):
        merger._bh_tp4_merger_fc2_pc = bh_tp4_merger_fc2_pc(merger.device)
    fc1_pc = merger._bh_tp4_merger_fc1_pc
    fc2_pc = merger._bh_tp4_merger_fc2_pc
    hidden_states = ensure_l1_tensor(hidden_states)

    if hidden_states.layout != ttnn.TILE_LAYOUT:
        hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT, memory_config=l1)

    b0, b1, r, h = (
        int(hidden_states.shape[0]),
        int(hidden_states.shape[1]),
        int(hidden_states.shape[2]),
        int(hidden_states.shape[3]),
    )
    flat = r * h
    if flat % int(merger.mlp_size) != 0:
        raise ValueError(f"PatchMerger reshape: S*H={flat} not divisible by mlp_size={merger.mlp_size}")
    new_r = flat // int(merger.mlp_size)

    if merger._use_layer_norm:
        hidden_states = ttnn.layer_norm(
            hidden_states,
            weight=merger.tt_ln_weight,
            bias=merger.tt_ln_bias,
            epsilon=1e-6,
            memory_config=l1,
        )
    elif merger.tt_ln_weight is not None:
        hidden_states = ttnn.rms_norm(
            hidden_states,
            weight=merger.tt_ln_weight,
            epsilon=1e-6,
            memory_config=l1,
        )

    hidden_states = ttnn.reshape(hidden_states, (b0, b1, new_r, int(merger.mlp_size)), memory_config=l1)
    compute_kc = getattr(merger, "compute_kernel_config", None)

    # fc1 in0 is [1,1,2816,6144] in L1 (~17 MB mesh-wide).  The swept 2D-mcast
    # program config's CBs clash with that resident buffer on BH; stage in0 to
    # DRAM for the matmul only (weights already DRAM).  Output stays L1.
    fc1_in = ttnn.to_memory_config(hidden_states, ttnn.DRAM_MEMORY_CONFIG)
    if fc1_in is not hidden_states:
        ttnn.deallocate(hidden_states)
    hidden_states = ttnn.linear(
        fc1_in,
        merger.tt_w1,
        bias=merger.tt_w1_bias,
        dtype=ttnn.bfloat8_b,
        memory_config=l1,
        program_config=fc1_pc,
        compute_kernel_config=compute_kc,
    )
    if fc1_in is not hidden_states:
        ttnn.deallocate(fc1_in)
    hidden_states = ttnn.gelu(hidden_states, memory_config=l1)
    fc2_kwargs = dict(
        dtype=ttnn.bfloat8_b,
        memory_config=l1,
        compute_kernel_config=compute_kc,
    )
    if fc2_pc is not None:
        fc2_kwargs["program_config"] = fc2_pc
    return ttnn.linear(
        hidden_states,
        merger.tt_w2,
        bias=merger.tt_w2_bias,
        **fc2_kwargs,
    )


def vision_tower_post_patch_embed_tp4_bh(
    tower,
    x: ttnn.Tensor,
    grid_thw: torch.Tensor,
) -> ttnn.Tensor:
    """Vision trunk: TP4 BH blocks + post-trunk norm + merger, all L1 activations."""
    import torch

    l1 = ttnn.L1_MEMORY_CONFIG
    if grid_thw is None:
        raise ValueError("grid_thw is required for Dots vision")
    if grid_thw.dim() == 1:
        grid_thw = grid_thw.unsqueeze(0)

    if isinstance(x, torch.Tensor):
        rep = ttnn.ReplicateTensorToMesh(tower.device) if tower.device.get_num_devices() > 1 else None
        x = x.unsqueeze(1) if x.dim() == 3 else x
        x = ttnn.from_torch(
            x.to(torch.bfloat16),
            device=tower.device,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=l1,
            mesh_mapper=rep,
        )
    if len(x.shape) == 3:
        x = ttnn.reshape(x, (1, 1, int(x.shape[1]), int(x.shape[2])), memory_config=l1)

    x = ensure_l1_tensor(x, dtype=ttnn.bfloat8_b)
    actual_seq_len = int(x.shape[2])
    rot_mats, cu_seqlens = tower.rope.build(grid_thw, actual_seq_len)
    rot_mats = rot_mats_l1(rot_mats)

    for block in tower.blocks:
        x = block.forward(
            x,
            rot_mats=rot_mats,
            cu_seqlens=cu_seqlens,
        )

    if tower.post_trunk_norm is not None:
        x = tower.post_trunk_norm(x, output_l1=True)

    if tower.patch_merger is not None:
        x = vision_patch_merger_tp4_bh_forward(tower.patch_merger, x)

    return ensure_l1_tensor(x)
