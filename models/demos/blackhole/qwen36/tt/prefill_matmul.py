# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Single-chip prefill projections: ttnn.linear, or ttnn.experimental.minimal_matmul when
QWEN36_PREFILL_MINIMAL_MATMUL=1 (default off). Same compute kernel config in both; minimal_matmul is 3-5x faster
on the prefill shapes. The knob is read on every call; a test may set ``FORCE`` (True/False) to override the env."""
import os

import ttnn

FORCE = None  # None: follow QWEN36_PREFILL_MINIMAL_MATMUL; True/False: override (tests, A/B runs)

# M/K/N block sizes in tiles. Subblock (h, w), or None for the op default; (1, 4) was 5-18% faster than the
# default in a sweep on 4096-row prefill shapes.
BLOCK_M, BLOCK_K, BLOCK_N = 8, 8, 8
SUBBLOCK = (1, 4)

_configs = {}


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


def prefill_linear(x, w, compute_kernel_config, activation=None, memory_config=None):
    """x [B, S, K] (or [1, 1, S, K]) @ w [K, N], optional fused SiLU. Single-chip prefill only (no bias)."""
    if not enabled() or x.device().get_num_devices() != 1:
        kw = {} if activation is None else {"activation": activation}
        if memory_config is not None:
            kw["memory_config"] = memory_config
        return ttnn.linear(x, w, compute_kernel_config=compute_kernel_config, **kw)
    assert activation in (None, "silu"), activation
    m_tiles = 1
    for d in list(x.shape)[:-1]:
        m_tiles *= d
    m_tiles = (m_tiles + 31) // 32
    cfg = _config(x.device(), m_tiles, (x.shape[-1] + 31) // 32, (w.shape[-1] + 31) // 32)
    kw = {}
    if activation == "silu":
        kw["fused_activation"] = ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)
    if memory_config is not None:
        kw["memory_config"] = memory_config
    return ttnn.experimental.minimal_matmul(x, w, config=cfg, compute_kernel_config=compute_kernel_config, **kw)
