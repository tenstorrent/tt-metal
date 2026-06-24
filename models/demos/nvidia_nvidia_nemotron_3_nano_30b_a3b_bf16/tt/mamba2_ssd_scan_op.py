# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Fused Mamba2 SSD chunked-scan op for NemotronH-30B prefill.

Replaces the Python chunk loop (94K+ iterations at ISL=256K) with a single
C++ unified kernel dispatch per M-layer.  Expected speedup: 30–100× for
large ISL.

Usage (drop-in for the chunk loop in mamba2_prefill.py):

    from .mamba2_ssd_scan_op import mamba2_ssd_scan

    y_full, h_next = mamba2_ssd_scan(
        mesh_device=mesh_device,
        x_dt_pad=x_dt_pad,    # [B, S_pad, H, D]
        B_pad=B_pad,           # [B, S_pad, N_GROUPS, N]
        C_pad=C_pad,           # [B, S_pad, N_GROUPS, N]
        x_pad=x_pad,           # [B, S_pad, H, D]
        log_decay_pad=log_decay_pad,  # [B, S_pad, H]
        h_prev=h_prev,         # [B, H, D, N] or None
        D_tt=D_tt,             # [1, H, 1, 1]
        n_chunks=n_chunks,     # S_pad // CHUNK_SIZE
        mesh_mapper=mesh_mapper,
    )

Returns:
    y_full : [B, S_pad, H, D]   — output for all chunks
    h_next : [B, H, D, N]       — SSM state after last chunk
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import UnifiedKernelDescriptor

# ---------------------------------------------------------------------------
# Constants (must match mamba2_prefill.py and mamba2_ssd_scan.hpp exactly)
# ---------------------------------------------------------------------------
_NUM_HEADS = 64
_HEAD_DIM = 64
_N_GROUPS = 8
_SSM_STATE_SIZE = 128
_CHUNK_SIZE = 64
_TILE = 32
_HEADS_PER_GROUP = 8

# Tile counts per logical tensor chunk
_XDT_TILES = (_CHUNK_SIZE // _TILE) * (_HEAD_DIM // _TILE)  # 4
_B_TILES = (_CHUNK_SIZE // _TILE) * (_SSM_STATE_SIZE // _TILE)  # 8
_GAMMA_TILES = _CHUNK_SIZE // _TILE  # 2
_H_TILES = (_HEAD_DIM // _TILE) * (_SSM_STATE_SIZE // _TILE)  # 8
_L_TILES = (_CHUNK_SIZE // _TILE) ** 2  # 4

_HPP_PATH = str(Path(__file__).parent.parent / "unified_kernels" / "mamba2_ssd_scan.hpp")

_BF16_TILE_BYTES = _TILE * _TILE * 2  # 2048 bytes per bf16 tile

# CB IDs (must match HPP)
_CB_X_DT = 0
_CB_B = 1
_CB_C = 2
_CB_X = 3
_CB_LOGL = 4
_CB_H = 5
_CB_Y = 6
_CB_QK = 7
_CB_HOUT = 8
_CB_DSKIP = 9
_CB_LOGGAMMA = 10
_CB_LOGDELTA = 11
_CB_LOGGSCALAR = 12
_CB_L_EXP = 13
_CB_YCROSS = 14
_CB_YINTRA = 15


# ---------------------------------------------------------------------------
# Config (compile-time parameters — frozen for use as dict key)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Mamba2SSDScanConfig:
    """Compile-time config for the fused SSM scan kernel."""

    grid_r: int = 8
    grid_c: int = 8
    num_heads: int = _NUM_HEADS
    head_dim: int = _HEAD_DIM
    ssm_state_size: int = _SSM_STATE_SIZE
    n_groups: int = _N_GROUPS
    chunk_size: int = _CHUNK_SIZE
    math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi2
    fp32_dest_acc_en: bool = False
    packer_l1_acc: bool = False
    math_approx_mode: bool = False


# ---------------------------------------------------------------------------
# Program hash cache (stable hash for C++ kernel compile cache hit)
# ---------------------------------------------------------------------------
_HASH_CACHE: dict[Mamba2SSDScanConfig, int] = {}


def _build_cbs_and_hash(cfg: Mamba2SSDScanConfig) -> tuple:
    """Build static descriptor components once per config; cache the hash.

    Returns (cbs, core_range_set, compile_args, compute_config, static_hash).
    ProgramDescriptor.custom_program_hash is computed once from the static
    (no-runtime-args) descriptor so every subsequent call re-uses the same
    C++ program cache entry.
    """
    core_range = ttnn.CoreRange(
        ttnn.CoreCoord(0, 0),
        ttnn.CoreCoord(cfg.grid_c - 1, cfg.grid_r - 1),
    )
    core_range_set = ttnn.CoreRangeSet([core_range])

    compute_config = ttnn.ComputeConfigDescriptor(
        math_fidelity=cfg.math_fidelity,
        math_approx_mode=cfg.math_approx_mode,
        fp32_dest_acc_en=cfg.fp32_dest_acc_en,
    )

    compile_args = [
        cfg.grid_r,  # GRID_R
        cfg.grid_c,  # GRID_C
        cfg.num_heads,  # NUM_HEADS
        cfg.n_groups,  # N_GROUPS
        cfg.chunk_size,  # CHUNK_SIZE
        cfg.head_dim,  # HEAD_DIM
        cfg.ssm_state_size,  # SSM_STATE_SIZE
    ]

    def _cb(cb_id: int, num_pages: int) -> ttnn.CBDescriptor:
        page_size = _BF16_TILE_BYTES
        fmt = ttnn.CBFormatDescriptor(
            buffer_index=cb_id,
            data_format=ttnn.bfloat16,
            page_size=page_size,
        )
        return ttnn.CBDescriptor(
            total_size=num_pages * page_size,
            core_ranges=core_range_set,
            format_descriptors=[fmt],
        )

    cbs = [
        _cb(_CB_X_DT, _XDT_TILES * 2),  # CB 0: x_dt double-buf
        _cb(_CB_B, _B_TILES * 2),  # CB 1: B double-buf
        _cb(_CB_C, _B_TILES * 2),  # CB 2: C double-buf
        _cb(_CB_X, _XDT_TILES * 2),  # CB 3: x double-buf
        _cb(_CB_LOGL, _L_TILES * 2),  # CB 4: log_L double-buf
        _cb(_CB_H, _H_TILES * 2),  # CB 5: h double-buf (RMW)
        _cb(_CB_Y, _XDT_TILES),  # CB 6: y output
        _cb(_CB_QK, _L_TILES),  # CB 7: QK scratch
        _cb(_CB_HOUT, _H_TILES),  # CB 8: h_out
        _cb(_CB_DSKIP, 1),  # CB 9: D_skip scalar
        _cb(_CB_LOGGAMMA, _GAMMA_TILES * 2),  # CB10: log_gamma double-buf
        _cb(_CB_LOGDELTA, _GAMMA_TILES * 2),  # CB11: log_delta double-buf
        _cb(_CB_LOGGSCALAR, 2),  # CB12: log_gscalar double-buf
        _cb(_CB_L_EXP, _L_TILES),  # CB13: L_EXP scratch
        _cb(_CB_YCROSS, _L_TILES),  # CB14: YCROSS scratch
        _cb(_CB_YINTRA, _XDT_TILES),  # CB15: y_intra scratch (not seen by BRISC)
    ]

    # Compute hash from static descriptor (no runtime args) — used every call
    static_unified = UnifiedKernelDescriptor(
        kernel_source=_HPP_PATH,
        core_ranges=core_range_set,
        ncrisc_compile_time_args=compile_args,
        brisc_compile_time_args=compile_args,
        trisc_compile_time_args=compile_args,
        trisc_compute_config=compute_config,
        # No runtime args for hash computation
    )
    static_desc = ttnn.ProgramDescriptor(
        kernels=static_unified.get_kernel_descriptors().kernels,
        cbs=cbs,
    )
    static_hash = ttnn.compute_program_descriptor_hash(static_desc)

    return cbs, core_range_set, compile_args, compute_config, static_hash


def _get_static_parts(cfg: Mamba2SSDScanConfig):
    if cfg not in _HASH_CACHE:
        _HASH_CACHE[cfg] = _build_cbs_and_hash(cfg)
    return _HASH_CACHE[cfg]


def _make_descriptor(
    cfg: Mamba2SSDScanConfig,
    ncrisc_rt_args: list,
    brisc_rt_args: list,
    n_chunks: int,
) -> ttnn.ProgramDescriptor:
    """Build a fresh ProgramDescriptor with runtime args set for this call."""
    cbs, core_range_set, compile_args, compute_config, static_hash = _get_static_parts(cfg)

    trisc_rt_args = [n_chunks]
    unified_kernel = UnifiedKernelDescriptor(
        kernel_source=_HPP_PATH,
        core_ranges=core_range_set,
        ncrisc_compile_time_args=compile_args,
        brisc_compile_time_args=compile_args,
        trisc_compile_time_args=compile_args,
        ncrisc_common_runtime_args=ncrisc_rt_args,
        brisc_common_runtime_args=brisc_rt_args,
        trisc_common_runtime_args=trisc_rt_args,
        trisc_compute_config=compute_config,
    )
    desc = ttnn.ProgramDescriptor(
        kernels=unified_kernel.get_kernel_descriptors().kernels,
        cbs=cbs,
    )
    desc.custom_program_hash = static_hash
    return desc


# ---------------------------------------------------------------------------
# Pre-allocated output tensors (output_prealloc KB record)
# ---------------------------------------------------------------------------
_OUTPUT_CACHE: dict = {}


def _get_outputs(
    mesh_device: ttnn.MeshDevice,
    H: int,
    n_chunks: int,
    C: int,
    D: int,
    N: int,
    mesh_mapper,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """Lazily allocate (or return cached) pre-allocated output tensors.

    y is [H, n_chunks, C, D] in the kernel's head-first layout.
    h_out is [H, D, N].
    """
    key = (id(mesh_device), H, n_chunks, C, D, N)
    if key not in _OUTPUT_CACHE:
        _OUTPUT_CACHE[key] = (
            ttnn.from_torch(
                torch.zeros(H, n_chunks, C, D, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                mesh_mapper=mesh_mapper,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
            ttnn.from_torch(
                torch.zeros(H, D, N, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                mesh_mapper=mesh_mapper,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
        )
    return _OUTPUT_CACHE[key]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def mamba2_ssd_scan(
    mesh_device: ttnn.MeshDevice,
    x_dt_pad: ttnn.Tensor,  # [B, S_pad, H, D]
    B_pad: ttnn.Tensor,  # [B, S_pad, N_GROUPS, N]
    C_pad: ttnn.Tensor,  # [B, S_pad, N_GROUPS, N]
    x_pad: ttnn.Tensor,  # [B, S_pad, H, D]
    log_decay_pad: ttnn.Tensor,  # [B, S_pad, H]
    h_prev: Optional[ttnn.Tensor],  # [B, H, D, N] or None
    D_tt: ttnn.Tensor,  # [1, H, 1, 1]
    n_chunks: int,
    cfg: Optional[Mamba2SSDScanConfig] = None,
    mesh_mapper=None,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """Execute the fused Mamba2 SSD chunked-scan kernel.

    Returns:
        y_full : [B, S_pad, H, D]  — output tokens (all chunks)
        h_next : [B, H, D, N]      — SSM state after last chunk
    """
    if cfg is None:
        cfg = Mamba2SSDScanConfig()

    B = x_dt_pad.shape[0]
    S_pad = x_dt_pad.shape[1]
    H, D, N, G, C = _NUM_HEADS, _HEAD_DIM, _SSM_STATE_SIZE, _N_GROUPS, _CHUNK_SIZE

    # ---- Pre-process: transpose to head-first layout ----
    # The kernel expects [H, n_chunks, C, D] so BRISC can stream contiguous chunks
    # per core. Python cost: 5 permutes + 5 reshapes = trivial vs 94K loop iterations.

    _xdt = ttnn.reshape(
        ttnn.permute(x_dt_pad, [0, 2, 1, 3], memory_config=ttnn.DRAM_MEMORY_CONFIG),
        [H, n_chunks, C, D],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    _x = ttnn.reshape(
        ttnn.permute(x_pad, [0, 2, 1, 3], memory_config=ttnn.DRAM_MEMORY_CONFIG),
        [H, n_chunks, C, D],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    _B = ttnn.reshape(
        ttnn.permute(B_pad, [0, 2, 1, 3], memory_config=ttnn.DRAM_MEMORY_CONFIG),
        [G, n_chunks, C, N],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    _C = ttnn.reshape(
        ttnn.permute(C_pad, [0, 2, 1, 3], memory_config=ttnn.DRAM_MEMORY_CONFIG),
        [G, n_chunks, C, N],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    # log_decay: [B, S_pad, H] → [H, n_chunks * C] (all chunks for head h contiguous)
    _logd = ttnn.reshape(
        ttnn.permute(log_decay_pad, [0, 2, 1], memory_config=ttnn.DRAM_MEMORY_CONFIG),
        [H, n_chunks * C],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    # h_in: [B, H, D, N] → [H, D, N]  (B=1; kernel indexes per head)
    if h_prev is not None:
        _h_in = ttnn.reshape(h_prev, [H, D, N], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _h_in_owned = False
    else:
        _h_in = ttnn.from_torch(
            torch.zeros(H, D, N, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=mesh_mapper,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        _h_in_owned = True
    # D_skip: [1, H, 1, 1] → [H, 1]
    _D = ttnn.reshape(D_tt, [H, 1], memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # ---- Pre-allocated outputs (reused each call) ----
    y_out, h_out = _get_outputs(mesh_device, H, n_chunks, C, D, N, mesh_mapper)

    # ---- Build descriptor with runtime args ----
    # ProgramDescriptor is rebuilt each call (cheap Python objects).
    # C++ kernel compilation cache hit guaranteed via custom_program_hash.
    ncrisc_rt_args = [
        _xdt.buffer_address(),
        _B.buffer_address(),
        _C.buffer_address(),
        _x.buffer_address(),
        _logd.buffer_address(),
        _h_in.buffer_address(),
        _D.buffer_address(),
        n_chunks,
    ]
    brisc_rt_args = [
        y_out.buffer_address(),
        h_out.buffer_address(),
        n_chunks,
    ]
    d = _make_descriptor(cfg, ncrisc_rt_args, brisc_rt_args, n_chunks)

    # ---- Dispatch ----
    io_tensors = [_xdt, _B, _C, _x, _logd, _h_in, _D, y_out, h_out]
    ttnn.generic_op(io_tensors, d)

    # ---- Post-process: transpose output back to [B, S_pad, H, D] ----
    # y_out: [H, n_chunks, C, D] → [B, S_pad, H, D]
    y_hsd = ttnn.reshape(y_out, [H, S_pad, D], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    y_bshd = ttnn.permute(
        ttnn.reshape(y_hsd, [1, H, S_pad, D], memory_config=ttnn.DRAM_MEMORY_CONFIG),
        [0, 2, 1, 3],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # h_out: [H, D, N] → [B, H, D, N]
    h_next = ttnn.reshape(h_out, [B, H, D, N], memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # ---- Deallocate temporaries ----
    for _t in [_xdt, _x, _B, _C, _logd, _D, y_hsd]:
        _t.deallocate(True)
    if _h_in_owned:
        _h_in.deallocate(True)

    return y_bshd, h_next
