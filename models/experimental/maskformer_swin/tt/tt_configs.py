# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Centralised TTNN program/memory configs for MaskFormer Swin-B.

These helpers keep all sharding/interleaving knobs in one place so they can be
tuned without editing the decoder/head implementations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
import inspect
import os

from .ttnn_compat import ttnn


@dataclass
class DecoderProgramConfigs:
    sdpa: Optional[Any] = None
    matmul_qkv: Optional[Any] = None
    matmul_out: Optional[Any] = None
    matmul_mlp: Optional[Any] = None
    core_grid: Optional[Any] = None
    sequence_memory: Optional[Any] = None
    prefer_l1: bool = False
    notes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HeadsProgramConfigs:
    matmul: Optional[Any] = None
    core_grid: Optional[Any] = None
    activation_memory: Optional[Any] = None
    notes: Dict[str, Any] = field(default_factory=dict)


def _init_with_signature(cls, overrides: Dict[str, Any]) -> Optional[Any]:
    """Best-effort instantiate a TTNN config class with partial overrides."""

    try:
        sig = inspect.signature(cls)
    except Exception:
        return None
    kwargs: Dict[str, Any] = {}
    for name, param in sig.parameters.items():
        if name in overrides:
            kwargs[name] = overrides[name]
        elif param.default is not inspect._empty:
            kwargs[name] = param.default
        elif param.kind in (param.VAR_KEYWORD, param.VAR_POSITIONAL):
            continue
        else:
            # Conservative fallback values
            kwargs[name] = overrides.get(name, 1)
    try:
        return cls(**kwargs)
    except Exception:
        return None


def _grid_tuple() -> Tuple[int, int]:
    # Wormhole N300 has an 8x8 grid; keep one core spare if using SDPA
    return (8, 7)


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _maybe_sdpa_program_config(seq_q: int, seq_k: int, grid: Tuple[int, int]) -> Optional[Any]:
    if ttnn is None or not hasattr(ttnn, "SDPAProgramConfig"):
        return None
    tile = 32

    def _round_up(value: int, multiple: int) -> int:
        return ((value + multiple - 1) // multiple) * multiple

    # SDPA requires chunk sizes to be multiples of TILE_WIDTH (32). Prefer conservative chunk sizes that
    # work well for MaskFormer query lengths (typically 100) while remaining valid for all sequence sizes.
    q_chunk = _round_up(max(64, min(128, int(seq_q))), tile)
    k_chunk = _round_up(max(128, min(256, int(seq_k))), tile)
    return _init_with_signature(
        ttnn.SDPAProgramConfig,
        {
            "compute_with_storage_grid_size": grid,
            "q_chunk_size": q_chunk,
            "k_chunk_size": k_chunk,
            "exp_approx_mode": False,
        },
    )


def _maybe_matmul_program_config(grid: Tuple[int, int]) -> Optional[Any]:
    if ttnn is None or not hasattr(ttnn, "MatmulProgramConfig"):
        return None
    return _init_with_signature(
        ttnn.MatmulProgramConfig,
        {
            "compute_with_storage_grid_size": grid,
            "in0_block_w": 1,
            "out_subblock_h": 1,
            "out_subblock_w": 1,
            "per_core_M": 1,
            "per_core_N": 1,
            "transpose_m0": False,
            "transpose_m1": False,
        },
    )


def build_decoder_program_configs(
    *,
    seq_q: int,
    seq_k: int,
    hidden_dim: int,
    num_heads: int,
    batch_size: int = 1,
) -> DecoderProgramConfigs:
    """Program + memory defaults for decoder attention/MLP."""

    if ttnn is None:
        return DecoderProgramConfigs()

    grid = _grid_tuple()
    enable_sdpa = _env_flag("MASKFORMER_TT_ENABLE_SDPA", True)
    disable_matmul_pc = _env_flag("MASKFORMER_TT_DISABLE_MATMUL_PC", False)
    disable_core_grid = _env_flag("MASKFORMER_TT_DISABLE_CORE_GRID", False)

    sdpa_pc = _maybe_sdpa_program_config(seq_q, seq_k, grid) if enable_sdpa else None
    mm_pc = None if disable_matmul_pc else _maybe_matmul_program_config(grid)
    core_grid = None
    if (not disable_core_grid) and hasattr(ttnn, "CoreGrid"):
        try:
            core_grid = ttnn.CoreGrid(y=grid[1], x=grid[0])  # type: ignore[attr-defined]
        except Exception:
            core_grid = None

    # Prefer L1 when the activation payload is small enough (per batch), unless overridden.
    l1_default = seq_q * hidden_dim < 128 * 256 and seq_k * hidden_dim < 512 * 256
    prefer_l1 = _env_flag("MASKFORMER_TT_ENABLE_L1_SEQ", l1_default)
    if _env_flag("MASKFORMER_TT_FORCE_DRAM_SEQ", False):
        prefer_l1 = False
    seq_mem_cfg = getattr(ttnn, "L1_MEMORY_CONFIG", None) if prefer_l1 else getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
    if seq_mem_cfg is None and hasattr(ttnn, "DRAM_MEMORY_CONFIG"):
        seq_mem_cfg = ttnn.DRAM_MEMORY_CONFIG

    return DecoderProgramConfigs(
        sdpa=sdpa_pc,
        matmul_qkv=mm_pc,
        matmul_out=mm_pc,
        matmul_mlp=mm_pc,
        core_grid=core_grid,
        sequence_memory=seq_mem_cfg,
        prefer_l1=prefer_l1,
        notes={
            "grid": grid,
            "batch": batch_size,
            "heads": num_heads,
            "q_chunk": seq_q,
            "k_chunk": seq_k,
            "enable_sdpa": enable_sdpa,
            "disable_matmul_pc": disable_matmul_pc,
            "disable_core_grid": disable_core_grid,
        },
    )


def build_heads_program_configs(num_queries: int, hidden_dim: int) -> HeadsProgramConfigs:
    """Program/memory defaults for MaskFormer heads."""

    if ttnn is None:
        return HeadsProgramConfigs()

    grid = _grid_tuple()
    disable_matmul_pc = _env_flag("MASKFORMER_TT_DISABLE_MATMUL_PC", False)
    disable_core_grid = _env_flag("MASKFORMER_TT_DISABLE_CORE_GRID", False)
    core_grid = None
    if (not disable_core_grid) and hasattr(ttnn, "CoreGrid"):
        try:
            core_grid = ttnn.CoreGrid(y=grid[1], x=grid[0])  # type: ignore[attr-defined]
        except Exception:
            core_grid = None
    mm_pc = None if disable_matmul_pc else _maybe_matmul_program_config(grid)
    # Use DRAM for head activations to reduce L1 pressure on larger images.
    use_l1 = _env_flag("MASKFORMER_TT_HEADS_USE_L1_ACT", False)
    act_mem = getattr(ttnn, "L1_MEMORY_CONFIG", None) if use_l1 else getattr(ttnn, "DRAM_MEMORY_CONFIG", None)

    return HeadsProgramConfigs(
        matmul=mm_pc,
        core_grid=core_grid,
        activation_memory=act_mem,
        notes={"grid": grid, "disable_matmul_pc": disable_matmul_pc, "disable_core_grid": disable_core_grid},
    )
