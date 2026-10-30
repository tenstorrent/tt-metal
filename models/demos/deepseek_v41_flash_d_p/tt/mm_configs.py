# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Matmul program configs (setting MATMUL_CONFIG, P.23): every dense ttnn.linear / ttnn.matmul of the model goes
through ``linear`` / ``matmul`` here. MATMUL_CONFIG=auto passes no program config (ttnn picks it: the path before
P.23); WO_A_CONFIG=auto (P.24) does the same for the batched attention wo_a call alone. tuned passes the config
``program_config`` derives from the call's own M, K, N, dtypes, fp32 DEST and the mesh's compute grid (a rule, not a
per-call constant for one box), or none (auto) where the rule has no entry. Only the program config changes: dtypes,
compute kernel config, memory configs and the op sequence are the caller's.

The rule and its numbers come from the sweep tests/perf/test_matmul_sweep.py (one chip, the model's exact calls as
captured in the model, L1 budget = what the model leaves free there); the table of results is in BREADCRUMBS P.23.
"""

from __future__ import annotations

import math

import ttnn

from .settings import settings

TILE = 32
TILE_BYTES = {ttnn.bfloat16: 2048, ttnn.float32: 4096, ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576}
# Per-core CB bytes a tuned config may use. The model's forward keeps L1 buffers of its own (fabric semaphores, the
# weighted combine's arena is freed per call); the capture (test_matmul_sweep.py::test_capture) records the lowest
# occupied L1 address at every matmul call: see BREADCRUMBS P.23 for the measured free L1.
L1_BUDGET = 1_100_000  # the model leaves 1382 KiB (2x2, P.23 capture); ~300 KiB kept for other boxes / users

# Diagnostics only (tests/perf/test_matmul_sweep.py sets it to a list): one dict per call.
RECORD: list | None = None
_cache: dict = {}


def mode() -> str:
    return settings.get("MATMUL_CONFIG")


def tuned() -> bool:
    """A config is passed (tuned: the rule; galaxy: the 8x4 table, else the rule)."""
    return mode() in ("tuned", "galaxy")


def _dt(d) -> str:
    return str(d).split(".")[-1]  # DataType.BFLOAT16 -> BFLOAT16


def table_key(shape, wshape, a_dt, w_dt, o_dt, fp32: bool, gx: int, gy: int) -> tuple:
    """The GALAXY key of one call: program_config's cache key with dtype names."""
    return (tuple(shape), tuple(wshape), _dt(a_dt), _dt(w_dt), _dt(o_dt), bool(fp32), int(gx), int(gy))


def build(spec):
    """(kind, config) of a table / sweep spec: ("2d" | "1d" | "bmm" | "minimal", {params}) or ("auto", None) -> None."""
    kind, p = spec
    if kind == "auto":
        return None
    if p is None:  # minimal_matmul with its own default blocking
        return (kind, None)
    p = dict(p)
    g = p.pop("grid")
    grid = ttnn.CoreCoord(g[0], g[1])
    if kind == "minimal":
        return ("minimal", ttnn.MinimalMatmulConfig(compute_with_storage_grid_size=grid, **p))
    if kind == "2d":
        return (
            "mm",
            ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=grid, fused_activation=None, **p
            ),
        )
    if kind == "1d":
        return (
            "mm",
            ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=grid, fused_activation=None, fuse_batch=True, **p
            ),
        )
    if kind == "bmm":
        return ("mm", ttnn.MatmulMultiCoreReuseProgramConfig(compute_with_storage_grid_size=grid, **p))
    raise ValueError(f"unknown matmul config kind {kind!r}")


def _ckc_fp32(ckc) -> bool:
    return bool(getattr(ckc, "fp32_dest_acc_en", False))


def _l1_view(device) -> dict:
    v = ttnn.get_memory_view(device, ttnn.BufferType.L1)
    base = ttnn.get_allocator_base_address(device, ttnn.BufferType.L1)
    low = None
    for blk in v.block_table:
        if str(blk.get("allocated", "")).lower() in ("yes", "true", "1"):
            a = int(blk["address"])
            low = a if low is None else min(low, a)
    # block addresses are offsets from the allocator base; L1 buffers sit at the top, CBs grow from the base
    return {
        "l1_base": int(base),
        "l1_bank_bytes": int(v.total_bytes_per_bank),
        "l1_allocated": int(v.total_bytes_allocated_per_bank),
        "l1_largest_free": int(v.largest_contiguous_bytes_free_per_bank),
        "l1_lowest_allocated": low,
        "l1_cb_free": int(v.total_bytes_per_bank) if low is None else low,
    }


def _record(op, site, x, w, dtype, ckc, memory_config, pc, transpose_b=False):
    RECORD.append(
        {
            "op": op,
            "site": site,
            "a": list(x.shape),
            "b": list(w.shape),
            "a_dtype": str(x.dtype),
            "b_dtype": str(w.dtype),
            "out_dtype": str(dtype),
            "a_mem": str(x.memory_config()),
            "b_mem": str(w.memory_config()),
            "out_mem": str(memory_config),
            "transpose_b": transpose_b,
            "fidelity": str(ckc.math_fidelity),
            "fp32_dest_acc_en": _ckc_fp32(ckc),
            "packer_l1_acc": bool(getattr(ckc, "packer_l1_acc", False)),
            "math_approx_mode": bool(getattr(ckc, "math_approx_mode", False)),
            "config": None if pc is None else f"{pc[0]}: {pc[1]}",
            **_l1_view(x.device()),
        }
    )


def _largest_div(n: int, cap: int) -> int:
    for d in range(min(n, cap), 0, -1):
        if n % d == 0:
            return d
    return 1


# MATMUL_CONFIG=galaxy (P.19, galaxy_8x4): the best config of every distinct dense matmul call of the shipped model on
# the 8x4 (SP=32, ENGRAM_TABLE=resident; the capture at 51200 and at 512000 holds the same 187 calls, 12 distinct),
# keyed by table_key: (input shape, weight shape, input / weight / output dtype, fp32 DEST, grid x, grid y) -> build()
# spec. Swept on single cards (tests/perf/test_matmul_sweep.py::test_sweep_card, P23_SHARD=i/12, OV-27), every
# candidate held to the mesh's 12x10 grid and the L1 the model leaves (1358 KiB for CBs); merged by merge_shards into
# generated/ds41_p23/galaxy_table.py. Acceptance per entry (as P.23): deterministic over 2 runs, CBs within that L1,
# output equal to auto or within one bf16 rounding (all 12: bit-identical to auto), error vs float64 not worse than
# the current (tuned) config's; a candidate replaces the current one only when > 1 % faster. Any other call (other
# chunk lengths, other boxes' shapes or grids) takes the rule. Per-site numbers: BREADCRUMBS P.19 (galaxy_8x4-full).
GALAXY: dict = {
    # attn.wo_a x40 a chunk: 0.2060 -> 0.2060 ms (current)
    ((1, 8, 160, 4096), (1, 8, 4096, 1024), "BFLOAT8_B", "BFLOAT8_B", "BFLOAT8_B", True, 12, 10): (
        "2d",
        {
            "grid": (11, 5),
            "transpose_mcast": False,
            "fuse_batch": False,
            "in0_block_w": 32,
            "out_subblock_h": 1,
            "out_subblock_w": 3,
            "out_block_h": 1,
            "out_block_w": 3,
            "per_core_M": 1,
            "per_core_N": 3,
        },
    ),
    # attn.wo_b x40 a chunk: 0.1506 -> 0.1506 ms (current)
    ((1, 1, 160, 8192), (1, 1, 8192, 5120), "BFLOAT8_B", "BFLOAT8_B", "BFLOAT16", True, 12, 10): (
        "1d",
        {
            "grid": (12, 10),
            "mcast_in0": True,
            "in0_block_w": 8,
            "out_subblock_h": 1,
            "out_subblock_w": 2,
            "out_block_h": 5,
            "out_block_w": 2,
            "per_core_M": 5,
            "per_core_N": 2,
        },
    ),
    # attn.wq_b x40 a chunk: 0.1897 -> 0.1897 ms (current)
    ((1, 1, 160, 1280), (1, 1, 1280, 32768), "BFLOAT8_B", "BFLOAT8_B", "BFLOAT16", True, 12, 10): (
        "1d",
        {
            "grid": (12, 10),
            "mcast_in0": True,
            "in0_block_w": 8,
            "out_subblock_h": 1,
            "out_subblock_w": 3,
            "out_block_h": 5,
            "out_block_w": 9,
            "per_core_M": 5,
            "per_core_N": 9,
        },
    ),
    # compressor x3 a chunk: 0.0431 -> 0.0431 ms (current)
    ((1, 1, 160, 5120), (1, 1, 5120, 1024), "BFLOAT16", "BFLOAT16", "FLOAT32", True, 12, 10): (
        "2d",
        {
            "grid": (11, 5),
            "transpose_mcast": False,
            "fuse_batch": True,
            "in0_block_w": 16,
            "out_subblock_h": 1,
            "out_subblock_w": 3,
            "out_block_h": 1,
            "out_block_w": 3,
            "per_core_M": 1,
            "per_core_N": 3,
        },
    ),
    # compressor x1 a chunk: 0.0293 -> 0.0293 ms (current)
    ((1, 1, 160, 5120), (1, 1, 5120, 512), "BFLOAT16", "BFLOAT16", "FLOAT32", True, 12, 10): (
        "2d",
        {
            "grid": (8, 5),
            "transpose_mcast": False,
            "fuse_batch": True,
            "in0_block_w": 16,
            "out_subblock_h": 1,
            "out_subblock_w": 2,
            "out_block_h": 1,
            "out_block_w": 2,
            "per_core_M": 1,
            "per_core_N": 2,
        },
    ),
    # compressor x3 a chunk: 0.0042 -> 0.0042 ms (current)
    ((1, 1, 320, 512), (1, 1, 512, 128), "BFLOAT16", "BFLOAT16", "BFLOAT16", True, 12, 10): (
        "2d",
        {
            "grid": (4, 10),
            "transpose_mcast": False,
            "fuse_batch": True,
            "in0_block_w": 16,
            "out_subblock_h": 1,
            "out_subblock_w": 1,
            "out_block_h": 1,
            "out_block_w": 1,
            "per_core_M": 1,
            "per_core_N": 1,
        },
    ),
    # compressor x1 a chunk: 0.0067 -> 0.0067 ms (current)
    ((1, 1, 640, 512), (1, 1, 512, 128), "BFLOAT16", "BFLOAT16", "BFLOAT16", True, 12, 10): (
        "2d",
        {
            "grid": (4, 10),
            "transpose_mcast": False,
            "fuse_batch": True,
            "in0_block_w": 16,
            "out_subblock_h": 2,
            "out_subblock_w": 1,
            "out_block_h": 2,
            "out_block_w": 1,
            "per_core_M": 2,
            "per_core_N": 1,
        },
    ),
    # engram_kv x2 a chunk: 0.2728 -> 0.2237 ms (minimal M4 K16 N8 sb1x4)
    ((1, 1, 640, 6144), (1, 1, 6144, 6400), "BFLOAT8_B", "BFLOAT8_B", "BFLOAT16", True, 12, 10): (
        "minimal",
        {"grid": (12, 10), "M_block_size": 4, "K_block_size": 16, "N_block_size": 8, "subblock_h": 1, "subblock_w": 4},
    ),
    # hc.fn x1 a chunk: 0.0298 -> 0.0298 ms (current)
    ((1, 1, 640, 5120), (1, 1, 5120, 32), "BFLOAT16", "FLOAT32", "FLOAT32", True, 12, 10): (
        "2d",
        {
            "grid": (1, 10),
            "transpose_mcast": False,
            "fuse_batch": True,
            "in0_block_w": 16,
            "out_subblock_h": 2,
            "out_subblock_w": 1,
            "out_block_h": 2,
            "out_block_w": 1,
            "per_core_M": 2,
            "per_core_N": 1,
        },
    ),
    # indexer.weights_proj x8 a chunk: 0.0168 -> 0.0168 ms (current)
    ((1, 1, 160, 5120), (1, 1, 5120, 32), "BFLOAT16", "BFLOAT16", "BFLOAT16", True, 12, 10): (
        "2d",
        {
            "grid": (1, 5),
            "transpose_mcast": False,
            "fuse_batch": True,
            "in0_block_w": 16,
            "out_subblock_h": 1,
            "out_subblock_w": 1,
            "out_block_h": 1,
            "out_block_w": 1,
            "per_core_M": 1,
            "per_core_N": 1,
        },
    ),
    # indexer.wq_b x8 a chunk: 0.0269 -> 0.0269 ms (current)
    ((1, 1, 160, 1280), (1, 1, 1280, 4096), "BFLOAT8_B", "BFLOAT8_B", "BFLOAT16", True, 12, 10): (
        "1d",
        {
            "grid": (12, 10),
            "mcast_in0": True,
            "in0_block_w": 8,
            "out_subblock_h": 1,
            "out_subblock_w": 2,
            "out_block_h": 5,
            "out_block_w": 2,
            "per_core_M": 5,
            "per_core_N": 2,
        },
    ),
    # q_a x40 a chunk: 0.0549 -> 0.0450 ms (2d 12x5 pm1 pn5 ob1x5 bw16 sb1x1)
    ((1, 1, 160, 5120), (1, 1, 5120, 1792), "BFLOAT8_B", "BFLOAT8_B", "BFLOAT16", True, 12, 10): (
        "2d",
        {
            "grid": (12, 5),
            "in0_block_w": 16,
            "out_subblock_h": 1,
            "out_subblock_w": 1,
            "out_block_h": 1,
            "out_block_w": 5,
            "per_core_M": 1,
            "per_core_N": 5,
            "transpose_mcast": False,
            "fuse_batch": True,
        },
    ),
}

# minimal_matmul blocking (tiles): the sweep's best within 1-2 % on every bf16-output projection of this run
MIN_BLOCK_M, MIN_BLOCK_K, MIN_BLOCK_N = 4, 4, 8
# at most this many row tiles: a wide-N bf16 projection on 1D multicast (P.9: the 160-row projections of ATTN_SP32_PROJ)
SMALL_M_TILES = 8


def rule(M: int, K: int, N: int, a_dt, w_dt, o_dt, fp32: bool, grid, batch: int = 1, l1_budget: int = L1_BUDGET):
    """(kind, config) of one call, or None (auto). kind "minimal": ttnn.experimental.minimal_matmul with a
    MinimalMatmulConfig; kind "mm": ttnn.linear / matmul with a program config. The rule:
      - a batched matmul with one weight per batch (attention wo_a): 2D multicast with fuse_batch=False (auto's own
        family: each core loops the batches, fp32 partials) at the widest in0 block, 32 tiles of K included: auto
        picks in0_block_w 1 there (K-51); bit-identical to auto, 4.1x (P.24). Never MatmulMultiCoreReuseProgramConfig:
        its K-block partials lose fp32 (rel vs float64 9.5e-3..2.8e-2 vs auto's 4.4e-3, DS41-113);
      - bf16 output, N wider than 4 tiles per grid column and at most SMALL_M_TILES row tiles (P.9, 8x4 with
        ATTN_SP32_PROJ=on: wq_b 160 x 1280 x 32768, wo_b 160 x 8192 x 5120 with full bfp8 weights; the indexer wq_b):
        1D multicast of in0 (every core all M rows, N split over the whole grid), in0 block 8 tiles of K: bit-identical
        to auto, wq_b 0.234 -> 0.199 ms, wo_b 0.279 -> 0.158 (minimal_matmul 0.275 / 0.283; probe ds41_p9_mm);
      - bf16 output, N wider than 4 tiles per grid column (q_a, wq_b, wo_b, engram wkv, indexer wq_b): minimal_matmul
        (MIN_BLOCK_*, subblock 2 x 2): 3-4.5x faster than auto, output equal to auto or within 1e-5 rel (the last
        bf16 rounding), the same error vs float64. fp32 outputs never (minimal's fp32 output is further from float64:
        router 4.33e-3 vs 4.07e-3);
      - otherwise 2D multicast (MatmulMultiCoreReuseMultiCastProgramConfig): M tiles over the grid's rows, N tiles
        over its columns, the widest in0 block (16, 8, 4, 2, 1 tiles of K) whose CBs fit ``l1_budget``, the largest
        out subblock in DEST (4 tiles with fp32 DEST, else 8): bit-identical to auto (same kernels), 1.2-3.3x
        (compressor, hc fn, router; the rule of GLM-5.3's tt/mm_configs.mm2d).
    Every count comes from the arguments (M rows a chip, the grid of this mesh): any box gets a config or auto."""
    if M % TILE or K % TILE or N % TILE:
        return None
    Mt, Kt, Nt = M // TILE, K // TILE, N // TILE
    if batch == 1 and o_dt == ttnn.bfloat16 and Nt > 4 * grid.x and Mt <= SMALL_M_TILES:
        pc = _mcast_1d(Mt, Kt, Nt, a_dt, w_dt, o_dt, fp32, grid, l1_budget)
        if pc is not None:
            return pc
    if batch == 1 and o_dt == ttnn.bfloat16 and Nt > 4 * grid.x and Mt >= MIN_BLOCK_M:
        kb = _largest_div(Kt, MIN_BLOCK_K)
        return (
            "minimal",
            ttnn.MinimalMatmulConfig(
                M_block_size=MIN_BLOCK_M,
                K_block_size=kb,
                N_block_size=MIN_BLOCK_N,
                subblock_h=2,
                subblock_w=2,
                compute_with_storage_grid_size=grid,
            ),
        )
    a, b, o = TILE_BYTES.get(a_dt), TILE_BYTES.get(w_dt), TILE_BYTES.get(o_dt)
    if None in (a, b, o):
        return None
    pm = -(-Mt // grid.y)
    gy = -(-Mt // pm)
    pn = -(-Nt // grid.x)
    gx = -(-Nt // pn)
    dst = 4 if fp32 else 8
    # the largest out subblock h x w <= DEST (h | pm, w | pn), the wider one on a tie
    sh, sw = max(
        ((h, w) for h in range(1, pm + 1) for w in range(1, pn + 1) if pm % h == 0 and pn % w == 0 and h * w <= dst),
        key=lambda t: (t[0] * t[1], t[1]),
    )
    for bw in (32, 16, 8, 4, 2, 1) if batch > 1 else (16, 8, 4, 2, 1):
        if Kt % bw:
            continue
        interm = pm * pn * 4096 if fp32 else 0  # fp32 partials between K blocks (packer_l1_acc off)
        if 2 * pm * bw * a + 2 * bw * pn * b + pm * pn * o + interm <= l1_budget:
            return (
                "mm",
                ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
                    in0_block_w=bw,
                    out_subblock_h=sh,
                    out_subblock_w=sw,
                    out_block_h=pm,
                    out_block_w=pn,
                    per_core_M=pm,
                    per_core_N=pn,
                    transpose_mcast=False,
                    fused_activation=None,
                    fuse_batch=batch == 1,
                ),
            )
    return None


def _mcast_1d(Mt, Kt, Nt, a_dt, w_dt, o_dt, fp32, grid, l1_budget):
    """1D multicast of in0 over the whole grid (per_core_M = Mt, N split over the cores), the widest in0 block of 8,
    4, 2, 1 tiles of K whose CBs fit ``l1_budget``; None when nothing fits."""
    a, b, o = TILE_BYTES.get(a_dt), TILE_BYTES.get(w_dt), TILE_BYTES.get(o_dt)
    if None in (a, b, o):
        return None
    pn = -(-Nt // (grid.x * grid.y))
    dst = 4 if fp32 else 8
    sh, sw = max(
        ((h, w) for h in range(1, Mt + 1) for w in range(1, pn + 1) if Mt % h == 0 and pn % w == 0 and h * w <= dst),
        key=lambda t: (t[1], t[0]),
    )
    for bw in (8, 4, 2, 1):
        if Kt % bw:
            continue
        interm = Mt * pn * 4096 if fp32 else 0
        if 2 * Mt * bw * a + 2 * bw * pn * b + Mt * pn * o + interm <= l1_budget:
            return (
                "mm",
                ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=grid,
                    in0_block_w=bw,
                    out_subblock_h=sh,
                    out_subblock_w=sw,
                    out_block_h=Mt,
                    out_block_w=pn,
                    per_core_M=Mt,
                    per_core_N=pn,
                    fuse_batch=True,
                    fused_activation=None,
                    mcast_in0=True,
                ),
            )
    return None


def program_config(x, w, dtype, ckc, batched: bool = False):
    """The tuned (kind, config) of one call, or None (auto); cached by (M, K, N, batch, dtypes, fp32 DEST, grid).
    A batched call (attention wo_a) under WO_A_CONFIG=auto: None (P.24's switch; the path before it).
    MATMUL_CONFIG=galaxy: the GALAXY entry of the same key when there is one, else the rule."""
    shape, wshape = tuple(x.shape), tuple(w.shape)
    if batched and math.prod(wshape[:-2]) > 1 and settings.get("WO_A_CONFIG") == "auto":
        return None
    dev = x.device()
    grid = dev.compute_with_storage_grid_size()
    fp32 = _ckc_fp32(ckc)
    galaxy = mode() == "galaxy"
    key = (shape, wshape, x.dtype, w.dtype, dtype, fp32, grid.x, grid.y, galaxy)
    if key in _cache:
        return _cache[key]
    if galaxy:
        spec = GALAXY.get(table_key(shape, wshape, x.dtype, w.dtype, dtype, fp32, grid.x, grid.y))
        if spec is not None:
            _cache[key] = pc = build(spec)
            return pc
    batch = math.prod(wshape[:-2]) if batched else 1
    if batch > 1 and math.prod(shape[:-2]) != batch:  # a broadcast over per-batch weights: auto
        _cache[key] = None
        return None
    M = math.prod(shape[:-1]) if batch == 1 else shape[-2]
    pc = rule(M, shape[-1], wshape[-1], x.dtype, w.dtype, dtype, fp32, grid, batch)
    _cache[key] = pc
    return pc


def _run(op, x, w, pc, dtype, ckc, memory_config, out=None):
    # ``out`` (HOST_ALLOC=on, P.15): a load-time output the op writes (optional_output_tensor); minimal_matmul has none
    # (the op returns a new wrapper of out's buffer: hand back ``out`` itself, the object ccl.dealloc keeps, DS41-149)
    kw = {} if out is None else {"optional_output_tensor": out}
    if pc is None:
        r = op(x, w, dtype=dtype, compute_kernel_config=ckc, memory_config=memory_config, **kw)
        return r if out is None else out
    kind, cfg = pc
    if kind == "minimal":
        kw = {} if cfg is None else {"config": cfg}
        return ttnn.experimental.minimal_matmul(
            x, w, dtype=dtype, compute_kernel_config=ckc, memory_config=memory_config, **kw
        )
    r = op(x, w, dtype=dtype, compute_kernel_config=ckc, memory_config=memory_config, program_config=cfg, **kw)
    return r if out is None else out


def linear(x, w, *, dtype, compute_kernel_config, memory_config, site: str = "", out=None):
    """ttnn.linear(x, w) with the MATMUL_CONFIG config (x [..., M, K], w [1, 1, K, N]). ``out``: written in place
    (HOST_ALLOC=on, P.15; the result is then ``out`` itself)."""
    pc = program_config(x, w, dtype, compute_kernel_config) if tuned() else None
    if RECORD is not None:
        _record("linear", site, x, w, dtype, compute_kernel_config, memory_config, pc)
    return _run(ttnn.linear, x, w, pc, dtype, compute_kernel_config, memory_config, out)


def matmul(x, w, *, dtype, compute_kernel_config, memory_config, site: str = "", out=None, rule: bool = True):
    """ttnn.matmul(x, w) with the MATMUL_CONFIG config (w may carry its own batch: [1, B, K, N]). ``out``: as linear.
    ``rule`` False: no config under MATMUL_CONFIG=tuned (a site that was not routed here before P.19: rope.pairs),
    the GALAXY entry or the rule under galaxy."""
    batched = len(w.shape) >= 3 and math.prod(tuple(w.shape)[:-2]) > 1
    on = tuned() and (rule or mode() == "galaxy")
    pc = program_config(x, w, dtype, compute_kernel_config, batched=batched) if on else None
    if RECORD is not None:
        _record("matmul", site, x, w, dtype, compute_kernel_config, memory_config, pc)
    return _run(ttnn.matmul, x, w, pc, dtype, compute_kernel_config, memory_config, out)
