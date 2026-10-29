# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Single-chip prefill projections: ttnn.linear, or ttnn.experimental.minimal_matmul when
QWEN36_PREFILL_MINIMAL_MATMUL=1 (default off). Same compute kernel config in both; minimal_matmul is 3-5x faster
on the prefill shapes. At small M (<= 128 rows, <= 256 rows for N >= 4096) a 1D-mcast ttnn.linear is faster, and
it fuses SiLU cheaply up to 512 rows; minimal_matmul's fused SiLU is slow, so above that SiLU is left to the caller
(``fuses_silu`` tells it). The knob is read on every call; a test may set ``FORCE`` (True/False) to override the env."""
import os

import ttnn

FORCE = None  # None: follow QWEN36_PREFILL_MINIMAL_MATMUL; True/False: override (tests, A/B runs)

# M/K/N block sizes in tiles. Subblock (h, w), or None for the op default; (1, 4) was 5-18% faster than the
# default in a sweep on 4096-row prefill shapes.
BLOCK_M, BLOCK_K, BLOCK_N = 8, 8, 8
SUBBLOCK = (1, 4)

_configs = {}
_small_configs = {}


def enabled():
    if FORCE is not None:
        return FORCE
    return os.environ.get("QWEN36_PREFILL_MINIMAL_MATMUL", "0") == "1"


def _config(device, m_tiles, k_tiles, n_tiles):
    grid = device.compute_with_storage_grid_size()
    key = (grid.x, grid.y, m_tiles, k_tiles, n_tiles)
    cfg = _configs.get(key)
    if cfg is None:
        kw = {}
        if SUBBLOCK is not None:
            kw = dict(subblock_h=SUBBLOCK[0], subblock_w=SUBBLOCK[1])
        cfg = _configs[key] = ttnn.MinimalMatmulConfig(
            M_block_size=max(1, min(BLOCK_M, m_tiles)),
            K_block_size=max(1, min(BLOCK_K, k_tiles)),
            N_block_size=max(1, min(BLOCK_N, n_tiles)),
            compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
            **kw,
        )
    return cfg


def _small_m_path(m_tiles, n, activation):
    """None: minimal_matmul. "plain"/"silu": 1D-mcast linear at small M (silu fused when asked)."""
    if m_tiles <= 4 or (m_tiles <= 8 and n >= 4096) or (activation == "silu" and m_tiles <= 16):
        return "silu" if activation == "silu" else "plain"
    return None


def _rows_tiles(x):
    m = 1
    for d in list(x.shape)[:-1]:
        m *= d
    return (m + 31) // 32


def fuses_silu(x, w):
    """True if prefill_linear(x, w, activation="silu") applies SiLU itself; else the caller must apply it."""
    if not enabled() or x.device().get_num_devices() != 1:
        return True
    return _small_m_path(_rows_tiles(x), w.shape[-1], "silu") is not None


def _small_config(device, m_tiles, k_tiles, n_tiles, activation):
    grid = device.compute_with_storage_grid_size()
    key = (grid.x, grid.y, m_tiles, k_tiles, n_tiles, activation)
    cfg = _small_configs.get(key)
    if cfg is None:
        per_core_n = -(-n_tiles // (grid.x * grid.y))
        sub_w = max(d for d in range(1, min(4, per_core_n) + 1) if per_core_n % d == 0)
        in0_w = max(d for d in (8, 4, 2, 1) if k_tiles % d == 0)
        cfg = _small_configs[key] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(grid.x, grid.y),
            in0_block_w=in0_w,
            out_subblock_h=1,
            out_subblock_w=sub_w,
            per_core_M=m_tiles,
            per_core_N=per_core_n,
            fuse_batch=True,
            fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU) if activation == "silu" else None,
            mcast_in0=True,
        )
    return cfg


def prefill_linear(x, w, compute_kernel_config, activation=None, memory_config=None):
    """x [B, S, K] (or [1, 1, S, K]) @ w [K, N], optional fused SiLU. Single-chip prefill only (no bias)."""
    if not enabled() or x.device().get_num_devices() != 1:
        kw = {} if activation is None else {"activation": activation}
        if memory_config is not None:
            kw["memory_config"] = memory_config
        return ttnn.linear(x, w, compute_kernel_config=compute_kernel_config, **kw)
    assert activation in (None, "silu"), activation
    m_tiles = _rows_tiles(x)
    k_tiles, n_tiles = (x.shape[-1] + 31) // 32, (w.shape[-1] + 31) // 32
    path = _small_m_path(m_tiles, w.shape[-1], activation)
    if path is not None:
        pc = _small_config(x.device(), m_tiles, k_tiles, n_tiles, path if path == "silu" else None)
        return ttnn.linear(
            x,
            w,
            program_config=pc,
            compute_kernel_config=compute_kernel_config,
            memory_config=memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG,
        )
    cfg = _config(x.device(), m_tiles, k_tiles, n_tiles)
    kw = {}
    if memory_config is not None:
        kw["memory_config"] = memory_config
    return ttnn.experimental.minimal_matmul(x, w, config=cfg, compute_kernel_config=compute_kernel_config, **kw)
