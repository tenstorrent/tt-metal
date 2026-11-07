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
_mcast2d_configs = {}
_acc_ccs = {}


def fp32_acc():
    """QWEN36_PREFILL_MM_FP32_ACC (default "1"): "0" runs the minimal_matmul paths with fp32_dest_acc_en=False."""
    return os.environ.get("QWEN36_PREFILL_MM_FP32_ACC", "1") != "0"


def _no_fp32_cc(cc):
    """Copy of cc with fp32 dest accumulation off and packer L1 accumulation on (cached per input config)."""
    out = _acc_ccs.get(id(cc))
    if out is None:
        out = type(cc)(
            math_fidelity=cc.math_fidelity,
            math_approx_mode=cc.math_approx_mode,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        _acc_ccs[id(cc)] = (cc, out)
        return out
    return out[1]


def _subblock(n_block, default):
    if not fp32_acc() and n_block % 8 == 0:
        return (1, 8)
    return default


def enabled():
    if FORCE is not None:
        return FORCE
    return os.environ.get("QWEN36_PREFILL_MINIMAL_MATMUL", "0") == "1"


# Measured on Blackhole p150 (11x10), LoFi + fp32 acc, bit-identical to 8/8/8. Used when fp32 acc is on.
# (K, N) -> (blocks for 4096 <= M < 8192, blocks for M >= 8192); blocks = (M_block, K_block, N_block) in tiles.
_SHAPE_BLOCKS = {
    (2048, 8224): ((8, 4, 16), (8, 4, 16)),
    (2048, 8256): ((8, 4, 16), (8, 4, 16)),
    (2048, 5120): ((8, 4, 16), (8, 16, 8)),
    (2048, 6144): ((8, 4, 16), (8, 16, 4)),
    (6144, 2048): ((16, 4, 8), (8, 8, 8)),
    (2048, 2048): ((16, 4, 8), (8, 4, 8)),
}

# Re-swept with fp32 acc off (trace-timed, LoFi bf16 x bfp8). Overrides _SHAPE_BLOCKS when fp32 acc is off:
# (K, N) -> (blocks for 512 < M < 4096, blocks for 4096 <= M < 8192, blocks for M >= 8192); None keeps the
# fp32-acc table (8/8/8 below 4096). A (blocks, subblock) entry sets the subblock; otherwise _subblock picks it.
_NO_ACC_BLOCKS = {
    (2048, 8256): (((4, 4, 24), None), ((4, 4, 24), None), ((4, 4, 32), (1, 4))),  # (1, 8) at N block 32: 2% slower
    (2048, 5120): (((8, 4, 16), None), None, None),
    # attention q|k|v; K blocks match the old fused q|k|v|gate entry
    (2048, 3072): (((8, 4, 16), (1, 4)), ((16, 4, 8), (1, 8)), ((8, 16, 8), (1, 8))),
}


def _blocks(m_tiles, k_tiles, n_tiles):
    """(M, K, N) block sizes in tiles and an explicit subblock (None: let _subblock pick)."""
    m = m_tiles * 32
    key = (k_tiles * 32, n_tiles * 32)
    if not fp32_acc() and 512 < m:
        tiers = _NO_ACC_BLOCKS.get(key)
        tier = tiers[0 if m < 4096 else 1 if m < 8192 else 2] if tiers else None
        if tier is not None:
            return (*tier[0], tier[1])
    entry = _SHAPE_BLOCKS.get(key) if m >= 4096 else None
    if entry is None:
        return BLOCK_M, BLOCK_K, BLOCK_N, None
    return (*(entry[1] if m >= 8192 else entry[0]), None)


SWIGLU_ROWS_MIN = 512  # fused SwiGLU from this many rows up (below it SiLU is fused into the 1D-mcast w1)
SWIGLU_BLOCKS = (4, 32, 8)  # (M, K, N) blocks for the fused SwiGLU matmul; K block 64 overflows L1
SWIGLU_BLOCKS_NO_ACC = (4, 16, 12)  # same, swept with fp32 acc off


def swiglu_enabled():
    return enabled() and os.environ.get("QWEN36_PREFILL_FUSED_SWIGLU", "0") == "1"


def _config(device, m_tiles, k_tiles, n_tiles):
    grid = device.compute_with_storage_grid_size()
    bm, bk, bn, explicit_sub = _blocks(m_tiles, k_tiles, n_tiles)
    key = (grid.x, grid.y, m_tiles, k_tiles, n_tiles, bm, bk, bn, fp32_acc())
    cfg = _configs.get(key)
    if cfg is None:
        kw = {}
        nb = max(1, min(bn, n_tiles))
        sub = explicit_sub or _subblock(nb, SUBBLOCK)
        if sub is not None:
            kw = dict(subblock_h=sub[0], subblock_w=sub[1])
        cfg = _configs[key] = ttnn.MinimalMatmulConfig(
            M_block_size=max(1, min(bm, m_tiles)),
            K_block_size=max(1, min(bk, k_tiles)),
            N_block_size=nb,
            compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
            **kw,
        )
    return cfg


def _small_m_path(m_tiles, n, activation):
    """None: minimal_matmul. "plain"/"silu": 1D-mcast linear at small M (silu fused when asked); at M <= 256 rows
    from N = 3072 up (the attention q|k|v matmul included)."""
    if m_tiles <= 4 or (m_tiles <= 8 and n >= 3072) or (activation == "silu" and m_tiles <= 16):
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


def _mcast2d_config(device, m_tiles, k_tiles):
    """2D-mcast config for N=2048 at 128-512 rows, or None when the grid is too small."""
    grid = device.compute_with_storage_grid_size()
    key = (grid.x, grid.y, m_tiles, k_tiles)
    if key not in _mcast2d_configs:
        gy = 4 if m_tiles == 4 else 8
        cfg = None
        if grid.x >= 11 and grid.y >= gy:
            cfg = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(11, gy),
                in0_block_w=8,
                out_subblock_h=1,
                out_subblock_w=3,
                per_core_M=m_tiles // gy,
                per_core_N=6,
                transpose_mcast=False,
                fused_activation=None,
            )
        _mcast2d_configs[key] = cfg
    return _mcast2d_configs[key]


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
    if activation is None and n_tiles == 64 and m_tiles in (4, 8, 16) and k_tiles % 8 == 0:
        # 2D mcast, bit-identical: 8-22% faster than the paths below for N=2048 at 128-512 rows.
        pc = _mcast2d_config(x.device(), m_tiles, k_tiles)
        if pc is not None:
            return ttnn.linear(
                x,
                w,
                program_config=pc,
                compute_kernel_config=compute_kernel_config,
                memory_config=memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG,
            )
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
    if not fp32_acc():
        compute_kernel_config = _no_fp32_cc(compute_kernel_config)
    return ttnn.experimental.minimal_matmul(x, w, config=cfg, compute_kernel_config=compute_kernel_config, **kw)


def prefill_swiglu(x, w_gu, compute_kernel_config, memory_config=None):
    """silu(x @ gate) * (x @ up) in one minimal_matmul; w_gu is the tile-pair-interleaved [gate|up] weight
    [K, 2N] (prepare_for_fused_swiglu); output width N. Single-chip prefill only."""
    m_tiles = _rows_tiles(x)
    k_tiles, n_tiles = (x.shape[-1] + 31) // 32, (w_gu.shape[-1] + 31) // 32
    grid = x.device().compute_with_storage_grid_size()
    bm, bk, bn = SWIGLU_BLOCKS if fp32_acc() else SWIGLU_BLOCKS_NO_ACC
    key = ("swiglu", grid.x, grid.y, m_tiles, k_tiles, n_tiles, bm, bk, bn, fp32_acc())
    cfg = _configs.get(key)
    if cfg is None:
        nb = max(1, min(bn, n_tiles))
        sub = _subblock(nb, SUBBLOCK)
        cfg = _configs[key] = ttnn.MinimalMatmulConfig(
            M_block_size=max(1, min(bm, m_tiles)),
            K_block_size=max(1, min(bk, k_tiles)),
            N_block_size=nb,
            subblock_h=sub[0],
            subblock_w=sub[1],
            compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
        )
    kw = {} if memory_config is None else {"memory_config": memory_config}
    if not fp32_acc():
        compute_kernel_config = _no_fp32_cc(compute_kernel_config)
    return ttnn.experimental.minimal_matmul(
        x, w_gu, config=cfg, compute_kernel_config=compute_kernel_config, fuse_swiglu=True, **kw
    )
