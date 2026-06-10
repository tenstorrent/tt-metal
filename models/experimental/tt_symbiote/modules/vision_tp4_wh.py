# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Wormhole N150×4 TP4 vision matmul program configs (8×8 compute grid).

Hardware-swept winners and L1-aware search for the dots.ocr vision tower on
``MESH_DEVICE=N150x4`` (Wormhole B0, 1×4 PCIe mesh).  Do not use the Blackhole
``vision_tp4_bh`` tables here — those target the 11×10 BH grid.
"""

from __future__ import annotations

import ttnn

_DTYPE_TILE_BYTES = {
    ttnn.bfloat16: 2048,
    ttnn.bfloat8_b: 1088,
    ttnn.bfloat4_b: 544,
}

# Wormhole usable L1 per Tensix (matches program.cpp clash budget on N150).
_L1_PER_CORE_BYTES = 1395 * 1024
_VISION_TP4_MERGED_SEQ = 2816  # 11264 // spatial_merge_size^2
_VISION_TP4_MLP_SIZE = 6144  # HIDDEN * spatial_merge_size^2


def _largest_divisor_le(value: int, limit: int) -> int:
    for c in range(min(value, limit), 0, -1):
        if value % c == 0:
            return c
    return 1


def _best_dst_subblock(ob_h: int, per_core_n: int, *, dst_budget: int = 8) -> tuple[int, int, int]:
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


def _l1_shard_bytes_per_core(device, m_dim: int, k_dim: int, dtype: ttnn.DataType) -> int:
    grid = device.compute_with_storage_grid_size()
    nc = int(grid.x) * int(grid.y)
    tile = 32
    mt = m_dim // tile
    kt = k_dim // tile
    return ((mt * kt + nc - 1) // nc) * _DTYPE_TILE_BYTES[dtype]


def wh_tp4_matmul_pc(
    device,
    m_dim: int,
    k_dim: int,
    n_dim: int,
    *,
    in0_dtype: ttnn.DataType = ttnn.bfloat8_b,
    out_dtype: ttnn.DataType = ttnn.bfloat8_b,
    l1_resident_bytes_per_core: int = 0,
):
    """L1-aware 2D-mcast search for Wormhole N150×4 vision matmuls."""
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


def enumerate_wh_tp4_matmul_pc_candidates(
    device,
    m_dim: int,
    k_dim: int,
    n_dim: int,
    *,
    in0_dtype: ttnn.DataType = ttnn.bfloat8_b,
    out_dtype: ttnn.DataType = ttnn.bfloat8_b,
    l1_resident_bytes_per_core: int = 0,
):
    """Yield ``(label, program_config, score)`` candidates for WH matmul sweeps.

    Same L1-aware 2D-mcast search space as :func:`wh_tp4_matmul_pc`, but emits
    every viable candidate (not just the analytical winner) so a hardware sweep
    can pick the true fastest config for the WH 8×8 grid.
    """
    grid = device.compute_with_storage_grid_size()
    grid_x, grid_y = int(grid.x), int(grid.y)
    tile = 32

    if m_dim % tile or k_dim % tile or n_dim % tile:
        return

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

    for transpose_mcast in (True, False):
        m_grid_max = grid_x if transpose_mcast else grid_y
        n_grid_max = grid_y if transpose_mcast else grid_x
        tm = "T" if transpose_mcast else "F"

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
                for ob_h in range(per_core_m, 0, -1):
                    if per_core_m % ob_h != 0:
                        continue
                    in0_bytes = 2 * ob_h * in0_block_w * in0_tile_b
                    interm_bytes = ob_h * per_core_n * 2048
                    out_bytes = 2 * ob_h * per_core_n * out_tile_b
                    if in0_bytes + interm_bytes + out_bytes > remaining_bytes:
                        continue
                    sub_h, sub_w, dst_area = _best_dst_subblock(ob_h, per_core_n)
                    gx = eff_mg if transpose_mcast else actual_ng
                    gy = actual_ng if transpose_mcast else eff_mg
                    label = (
                        f"tm{tm}_g{gx}x{gy}_M{per_core_m}_N{per_core_n}"
                        f"_obh{ob_h}_ibw{in0_block_w}_sub{sub_h}x{sub_w}"
                    )
                    pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                        compute_with_storage_grid_size=(gx, gy),
                        in0_block_w=in0_block_w,
                        out_subblock_h=sub_h,
                        out_subblock_w=sub_w,
                        out_block_h=ob_h,
                        out_block_w=per_core_n,
                        per_core_M=per_core_m,
                        per_core_N=per_core_n,
                        transpose_mcast=transpose_mcast,
                        fused_activation=None,
                        fuse_batch=False,
                    )
                    yield label, pc, (dst_area, -ob_h)


def wh_tp4_merger_fc1_pc(device):
    """Patch-merger fc1 ``2816×6144×6144`` on Wormhole N150 8×8.

    Silicon sweep 2026-06-10 (BFP8 L1 in0/out, LoFi, L1 in0 co-resident):
    grid=(8,8) tm=False M=11 N=24 obh=1 ibw=8 sub=(1,8) ~5.1 ms.
    """
    grid = device.compute_with_storage_grid_size()
    resident = _l1_shard_bytes_per_core(device, _VISION_TP4_MERGED_SEQ, _VISION_TP4_MLP_SIZE, ttnn.bfloat8_b)
    gx, gy = int(grid.x), int(grid.y)
    if gx >= 8 and gy >= 8:
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=8,
            out_subblock_h=1,
            out_subblock_w=8,
            out_block_h=1,
            out_block_w=24,
            per_core_M=11,
            per_core_N=24,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
        )
    return wh_tp4_matmul_pc(
        device,
        _VISION_TP4_MERGED_SEQ,
        _VISION_TP4_MLP_SIZE,
        _VISION_TP4_MLP_SIZE,
        in0_dtype=ttnn.bfloat8_b,
        out_dtype=ttnn.bfloat8_b,
        l1_resident_bytes_per_core=resident,
    )


def wh_tp4_merger_fc2_pc(device):
    """fc2 ``2816×6144×384`` — auto-config wins on WH (see ``bh_tp4_merger_fc2_pc`` note)."""
    _ = device
    return None


def wh_tp4_o_proj_pc(device, *, seq_len: int = 11264, ctx_dim: int = 384):
    """o_proj ``11264×384×1536``, BFP8×BFP8→BFP8 L1, on WH 8×8.

    Silicon sweep 2026-06-10 (in0 L1 BFP8 / out L1 BFP8, LoFi, in0 co-resident):
    grid=(6,8) tm=False M=44 N=8 obh=11 ibw=6 sub=(1,8) ~219 µs — 4.1× the prior
    generic-search fallback (~893 µs) and 1.9× the earlier BF16-out winner
    (obh=4 sub=(4,2), ~419 µs). Spreading N=1536 over 6 columns at per_core_N=8
    with out_block_h=11 (4 outer-M iters) beats the 8-column per_core_N=6 layout.
    """
    grid = device.compute_with_storage_grid_size()
    if int(grid.x) >= 8 and int(grid.y) >= 8 and seq_len == 11264 and ctx_dim == 384:
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(6, 8),
            in0_block_w=6,
            out_subblock_h=1,
            out_subblock_w=8,
            out_block_h=11,
            out_block_w=8,
            per_core_M=44,
            per_core_N=8,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
        )
    ctx_resident = _l1_shard_bytes_per_core(device, seq_len, ctx_dim, ttnn.bfloat8_b)
    return wh_tp4_matmul_pc(
        device,
        seq_len,
        ctx_dim,
        1536,
        in0_dtype=ttnn.bfloat8_b,
        out_dtype=ttnn.bfloat8_b,
        l1_resident_bytes_per_core=ctx_resident,
    )


def wh_tp4_mlp_down_pc(device, *, seq_len: int = 11264, itp: int = 1056):
    """Down matmul ``11264×1056×1536``, BFP8×BFP8→BFP8 L1, on WH 8×8.

    Silicon sweep 2026-06-10 (in0 L1 BFP8 / out L1 BFP8, LoFi, in0 co-resident):
    grid=(6,8) tm=False M=44 N=8 obh=11 ibw=3 sub=(1,8) ~525 µs — 3.4× the generic
    L1 search (~1771 µs). The search ties on dst_area=8 across out_block_h and
    then prefers the *smallest* obh (=1 → 44 outer-M passes → 44× weight DRAM
    re-streams); this matmul is weight-DRAM-bound, so obh=11 (4 passes) wins big.
    """
    grid = device.compute_with_storage_grid_size()
    if int(grid.x) >= 8 and int(grid.y) >= 8 and seq_len == 11264 and itp == 1056:
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(6, 8),
            in0_block_w=3,
            out_subblock_h=1,
            out_subblock_w=8,
            out_block_h=11,
            out_block_w=8,
            per_core_M=44,
            per_core_N=8,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
        )
    gum_resident = _l1_shard_bytes_per_core(device, seq_len, itp, ttnn.bfloat8_b)
    return wh_tp4_matmul_pc(
        device,
        seq_len,
        itp,
        1536,
        in0_dtype=ttnn.bfloat8_b,
        out_dtype=ttnn.bfloat8_b,
        l1_resident_bytes_per_core=gum_resident,
    )


def wh_tp4_qkv_pc(device):
    """QKV ``11264×1536×1152`` per device on WH 8×8 (2026-06-10 swept: grid=(6,8) obh=11)."""
    grid = device.compute_with_storage_grid_size()
    if int(grid.x) >= 8 and int(grid.y) >= 8:
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(6, 8),
            in0_block_w=8,
            out_subblock_h=1,
            out_subblock_w=6,
            out_block_h=11,
            out_block_w=6,
            per_core_M=44,
            per_core_N=6,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
        )
    return wh_tp4_matmul_pc(device, 11264, 1536, 1152, in0_dtype=ttnn.bfloat16, out_dtype=ttnn.bfloat8_b)


def wh_tp4_mlp_gate_up_pc(device):
    """Gate/up ``11264×1536×1056``, BF16×BFP8→BFP8(gate)/BFP4(up) L1, on WH 8×8.

    Silicon 2026-06-10: grid=(8,7) tm=True M=44 N=5 obh=11 ibw=8 sub=(1,5), ~38.9%
    TFLOPs. This matmul's in0 is **BF16** (replicated hidden) with ibw=8, so the
    activation CB grows ~2.7× per out_block_h vs the BFP8-input o_proj/down; obh=11
    is the largest that fits L1 with the BF16 input co-resident (obh=22 needs a
    720 KB in0 CB, obh=44 a 1.44 MB one). The generic ``wh_tp4_matmul_pc`` search
    ties on dst_area and picks obh=1 (44 outer-M passes) — far slower — so pin the
    swept winner here. Matches the proven ``bh_tp4_vision_mlp_pc`` result on WH.
    """
    grid = device.compute_with_storage_grid_size()
    if int(grid.x) >= 8 and int(grid.y) >= 8:
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 7),
            in0_block_w=8,
            out_subblock_h=1,
            out_subblock_w=5,
            out_block_h=11,
            out_block_w=5,
            per_core_M=44,
            per_core_N=5,
            transpose_mcast=True,
            fused_activation=None,
            fuse_batch=False,
        )
    return wh_tp4_matmul_pc(device, 11264, 1536, 1056, in0_dtype=ttnn.bfloat16, out_dtype=ttnn.bfloat8_b)
