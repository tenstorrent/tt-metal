# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Single-chip program-config knob lab for the dense Gemma 4 decode matmuls (31B / 12B).

One decode matmul (``qkv``, ``out_proj``, ``gate_up``, ``down``, ``lm_head``) of one model at the per-chip shard
shape of the serving mesh (tensor parallel 4 = the QB2 line, 8 = one Galaxy engine column) and one decode batch
(1 / 32): the weight is built at that shape in the dtype the model loads it with (attention / MLP bfp8 per
precision_overrides.json, lm_head bf16), the activation is the ``[1, 1, b, K]`` bf16 row tile, and every candidate
program config is timed as a traced loop (``GEMMA4_LAB_OPS_PER_TRACE`` calls per trace, ``GEMMA4_LAB_REPLAYS``
replays per batch, ``GEMMA4_LAB_BATCHES`` batches; the median of the batch means is the reported us). One
``RESULT lab ...`` line per candidate (knobs, PCC vs the baseline and vs the fp32 golden, us, DRAM MB, roofline us
at ``GEMMA4_LAB_DRAM_GBPS`` = 512 GB/s per P150, ratio) and one ``RESULT best ...`` line per shape.

Boundary layouts (what the fused ops around the matmul produce / consume) are parameters:
``GEMMA4_LAB_ACT`` = ``dram`` (DRAM interleaved, today's), ``l1`` (L1 interleaved) or ``ws[:<cores>]`` (L1
width-sharded over the first <cores> worker cores; default the DRAM-sharded decode grid); ``GEMMA4_LAB_OUT`` =
``dram`` / ``l1`` / ``ws`` (the matmul's L1 width-sharded output stays sharded); ``GEMMA4_LAB_IN1`` = ``both`` /
``il`` / ``ws`` (the DRAM-sharded kernel family needs the width-sharded weight, the 1D-multicast family the
interleaved one). The layouts apply to every candidate, the baseline included.

Candidates: the model's current config first (the DRAM-width-sharded kernel with its activation reshards for
qkv / out_proj / gate_up / down, the 1D-multicast lm_head config), the op's auto config, the resolver's 1D config
(tt/decode_matmul.py), the 2D prefill config on one row tile, the DRAM-sharded family (K block, fidelity / fp32
Dest, the multi-core pipeline workers per bank, the kernel alone, the row view below one tile) and the 1D-multicast
family (K block, fidelity / fp32 Dest, grid and per-core N / output subblock). ``KnobSpace`` enumerates the knobs
device-free, bounded by the L1 budget and the per-core DRAM stream bound; grids come from the device's
``compute_with_storage_grid_size`` (the whole grid and its divisor sub-grids, never a literal list), so the same
lab runs on a Wormhole 8x8 and a Blackhole 13x10, and the resolver table (``DecodeMMConfig``) carries grids as
fractions of the device grid or per-arch entries clamped to the largest fitting grid; the device sweep walks it by coordinate
descent (K block, then fidelity, then grid). A candidate whose output differs from the baseline's at
PCC < ``GEMMA4_LAB_PCC`` (0.9999) is rejected (``ok=0``) and never ``best``.

    TT_VISIBLE_DEVICES=<chip> pytest models/demos/gemma4/tests/unit/test_matmul_knob_lab.py \\
        -k "m31B and qkv and tp4 and bs32" -s

``GEMMA4_LAB_STAGES`` (default ``all``) limits the families (comma list of ``baseline,auto,resolver,2d,ds,1d``);
``GEMMA4_LAB_BUDGET_S`` (1000) stops a sweep that would overrun the job's wall budget.
"""

import math
import os
import time
from contextlib import contextmanager
from typing import NamedTuple, Optional

import pytest
import torch

import ttnn
from models.demos.gemma4.tt.decode_matmul import decode_1d_progcfg, decode_compute_kernel_config, decode_in0_block_w
from models.demos.gemma4.tt.dram_sharded import (
    TILE_SIZE,
    DramShardedLinear,
    _decode_core_grid,
    _estimate_decode_l1_bytes_for_in0,
    _padded_n_tiles,
    activation_memcfg,
    can_dram_shard,
    prefill_max_cols_default,
    prefill_progcfg,
)
from models.demos.gemma4.tt.model import _compute_per_device_vocab, _get_lm_head_program_config

SINGLE_CHIP = [
    pytest.param(
        1,
        {"fabric_config": ttnn.FabricConfig.DISABLED, "trace_region_size": 64_000_000, "l1_small_size": 32768},
        id="single-chip",
    )
]

# Dense model dims (config.json of google/gemma-4-31B-it / google/gemma-4-12B-it): hidden, q heads, sliding kv heads,
# global kv heads, sliding / global head dim, MLP intermediate, vocab, layer counts (sliding + global).
MODELS = {
    "31B": dict(
        hidden=5376,
        heads=32,
        kv=16,
        global_kv=4,
        head_dim=256,
        global_head_dim=512,
        inter=21504,
        vocab=262144,
        sliding_layers=50,
        global_layers=10,
    ),
    "12B": dict(
        hidden=3840,
        heads=16,
        kv=8,
        global_kv=1,
        head_dim=256,
        global_head_dim=512,
        inter=15360,
        vocab=262144,
        sliding_layers=40,
        global_layers=8,
    ),
}
OPS = ("qkv", "out_proj", "gate_up", "down", "lm_head")

REPLAYS = int(os.environ.get("GEMMA4_LAB_REPLAYS", "20"))
OPS_PER_TRACE = int(os.environ.get("GEMMA4_LAB_OPS_PER_TRACE", "10"))
BATCHES = int(os.environ.get("GEMMA4_LAB_BATCHES", "3"))
DRAM_GBPS = float(os.environ.get("GEMMA4_LAB_DRAM_GBPS", "512"))
# One worker core's sustained DRAM read stream (conservative); a config whose per-core weight read at this rate
# would take more than GEMMA4_LAB_ROOFLINE_SLACK x the DRAM roofline is not enumerated.
CORE_GBPS = float(os.environ.get("GEMMA4_LAB_CORE_GBPS", "20"))
ROOFLINE_SLACK = float(os.environ.get("GEMMA4_LAB_ROOFLINE_SLACK", "8"))
PCC_MIN = float(os.environ.get("GEMMA4_LAB_PCC", "0.9999"))
BUDGET_S = float(os.environ.get("GEMMA4_LAB_BUDGET_S", "1000"))
STAGES = os.environ.get("GEMMA4_LAB_STAGES", "all")
ACT = os.environ.get("GEMMA4_LAB_ACT", "dram")
OUT = os.environ.get("GEMMA4_LAB_OUT", "dram")
IN1 = os.environ.get("GEMMA4_LAB_IN1", "both")
# Static circular buffers of one core (BH P150 L1 minus headroom, tt/dram_sharded.py).
L1_BUDGET = 1_572_864 - 64_000
TILE_BYTES = {ttnn.bfloat16: 2048, ttnn.bfloat8_b: 1088}
BYTES_PER_ELEM = {ttnn.bfloat16: 2.0, ttnn.bfloat8_b: 1088 / 1024}
DTYPE_TAG = {ttnn.bfloat16: "bf16", ttnn.bfloat8_b: "bfp8"}
FIDELITY = {"lofi": ttnn.MathFidelity.LoFi, "hifi2": ttnn.MathFidelity.HiFi2, "hifi4": ttnn.MathFidelity.HiFi4}


def _ceil_tile(v: int) -> int:
    return ((v + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE


def op_shapes(model: str, op: str, tp: int):
    """Per-chip ``(shape_tag, K, N, dtype, calls_per_step)`` of one decode matmul at tensor parallel ``tp``: the
    fused QKV / o_proj of a sliding and of a global layer (attention/weights.py: q heads / tp, kv heads / tp or
    one replicated kv head when kv heads < tp), the fused gate_up / down of the MLP (shared_mlp.py: intermediate /
    tp tile-aligned), and the lm_head vocab shard (model.py: tile-aligned then a power of two)."""
    d = MODELS[model]
    shapes = []
    if op in ("qkv", "out_proj"):
        for tag, hd, kv_heads, layers in (
            ("sliding", d["head_dim"], d["kv"], d["sliding_layers"]),
            ("global", d["global_head_dim"], d["global_kv"], d["global_layers"]),
        ):
            q = d["heads"] // tp * hd
            kv = hd if kv_heads < tp else (kv_heads // tp) * hd
            if op == "qkv":
                shapes.append((tag, d["hidden"], q + 2 * kv, ttnn.bfloat8_b, layers))
            else:
                shapes.append((tag, q, d["hidden"], ttnn.bfloat8_b, layers))
    elif op == "gate_up":
        inter = _ceil_tile(d["inter"] // tp)
        shapes.append(("mlp", d["hidden"], 2 * inter, ttnn.bfloat8_b, d["sliding_layers"] + d["global_layers"]))
    elif op == "down":
        inter = _ceil_tile(d["inter"] // tp)
        shapes.append(("mlp", inter, d["hidden"], ttnn.bfloat8_b, d["sliding_layers"] + d["global_layers"]))
    elif op == "lm_head":
        shapes.append(("lm_head", d["hidden"], _compute_per_device_vocab(d["vocab"], tp), ttnn.bfloat16, 1))
    else:
        raise ValueError(op)
    return shapes


def _divisors(n: int):
    return [d for d in range(1, n + 1) if n % d == 0]


def _largest_divisor(n: int, cap: int) -> int:
    for d in range(min(n, cap), 0, -1):
        if n % d == 0:
            return d
    return 1


def _dest_tiles(fp32: bool) -> int:
    """Output tiles one subblock may hold: 16 Dest tiles, half-sync (the 1D / 2D factories), halved for fp32."""
    return 4 if fp32 else 8


def _l1_1d(bw: int, pcn: int, wdtype) -> int:
    """Static CB estimate of the 1D mcast_in0 kernel on one core: double-buffered in0 and in1 blocks + output."""
    return 2 * bw * 2048 + 2 * bw * pcn * TILE_BYTES[wdtype] + 2 * pcn * 2048


def candidate_grids(gx: int, gy: int, min_fraction: float = 0.25):
    """The device grid and its divisor sub-grids ``(gx / dx, gy / dy)`` holding at least ``min_fraction`` of its
    cores, largest first: the one rule gives 13x10 / 13x5 on a Blackhole P150 and 8x8 / 8x4 / 4x8 / 4x4 / 8x2 /
    2x8 on a Wormhole; no grid is ever enumerated from a literal list."""
    grids = set()
    for dx in _divisors(gx):
        for dy in _divisors(gy):
            vx, vy = gx // dx, gy // dy
            if vx * vy >= min_fraction * gx * gy:
                grids.add((vx, vy))
    return sorted(grids, key=lambda g: (-g[0] * g[1], -g[0]))


def grid_spec_of(vx: int, vy: int, gx: int, gy: int) -> str:
    """A grid as a fraction of the device grid (``full``, ``x/2``, ``y/5``, ``x/2+y/5``) when it is a divisor
    sub-grid, else the explicit ``VXxVY`` (which the resolver table carries as a per-arch entry)."""
    if (vx, vy) == (gx, gy):
        return "full"
    if gx % vx == 0 and gy % vy == 0:
        parts = [f"x/{gx // vx}"] if gx // vx > 1 else []
        if gy // vy > 1:
            parts.append(f"y/{gy // vy}")
        return "+".join(parts)
    return f"{vx}x{vy}"


def resolve_grid(spec, gx: int, gy: int, arch: str = "blackhole"):
    """``spec`` on a ``gx x gy`` device grid -> ``(x, y)``: ``full``; fractions ``x/D``, ``y/D``, ``x/D+y/D``
    (``gx // D``, at least 1); an explicit ``VXxVY`` string, a ``(x, y)`` pair or a per-arch dict
    ``{"blackhole": (13, 10), "wormhole_b0": (8, 8)}`` (the entry of ``arch``, else the first) clamped to the
    largest fitting grid: a row tuned on 13x10 resolves to 8x8 on a Wormhole, never to None."""
    if isinstance(spec, dict):
        spec = spec.get(arch) or next(iter(spec.values()))
    if isinstance(spec, (tuple, list)):
        return max(1, min(int(spec[0]), gx)), max(1, min(int(spec[1]), gy))
    spec = str(spec)
    if spec == "full":
        return gx, gy
    if "/" in spec:
        vx, vy = gx, gy
        for part in spec.split("+"):
            axis, _, d = part.partition("/")
            if axis == "x":
                vx = max(1, gx // int(d))
            elif axis == "y":
                vy = max(1, gy // int(d))
            else:
                raise ValueError(f"grid spec {spec!r}")
        return vx, vy
    vx, vy = (int(v) for v in spec.split("x"))
    return max(1, min(vx, gx)), max(1, min(vy, gy))


# ---- resolver table format (proposal: moves verbatim into tt/decode_matmul.py on the coordinator's go) ----------


class DecodeMMConfig(NamedTuple):
    """One row of the decode matmul resolver table, keyed on ``(model, op, tp, batch bucket)``."""

    family: str  # "1d": 1D multicast on the interleaved weight | "ds": the DRAM-width-sharded kernel (Blackhole-only)
    grid: object  # resolve_grid spec ("full", "x/2+y/5", "13x10", per-arch dict); "ds" = the kernel's own gcd grid
    in0_block_w: int
    per_core_N: Optional[int]  # None = ceil(N tiles / cores) on the resolved grid; a value is raised to cover N
    out_subblock_w: int  # upper bound; the resolver takes the largest divisor of per-core N that fits Dest
    fidelity: str  # "lofi" | "hifi2" | "hifi4"
    fp32_acc: bool
    workers_per_bank: int = 0  # ds multi-core pipeline workers per DRAM bank (0 = single reader)
    out: str = "dram"  # "dram" | "l1" | "ws"
    arch: str = "any"  # "blackhole": the family exists only there (can_dram_shard is Blackhole-only) -> None elsewhere


ENV_PREFIX = (
    "GEMMA4_DECODE_MM_CFG_"  # GEMMA4_DECODE_MM_CFG_QKV="1d:grid=y/2,bw=24,pc=2,osw=1,fid=hifi2,acc=1,wpb=0,out=dram"
)
BATCH_BUCKETS = (1, 32)


def batch_bucket(batch: int) -> int:
    return 1 if batch <= 1 else 32


def parse_config_spec(spec: str) -> DecodeMMConfig:
    """The ``config=`` string of a ``RESULT best`` line (or the env override) -> a table row."""
    family, _, rest = spec.partition(":")
    kv = dict(item.split("=", 1) for item in rest.split(",") if item)
    pc = kv.get("pc", "")
    return DecodeMMConfig(
        family,
        kv.get("grid", "full"),
        int(kv.get("bw", 8)),
        int(pc) if pc.isdigit() else None,
        int(kv.get("osw", 1)),
        kv.get("fid", "hifi2"),
        kv.get("acc", "1") == "1",
        int(kv.get("wpb", 0)),
        kv.get("out", "dram"),
        "blackhole" if family == "ds" else "any",
    )


def resolve_1d_knobs(cfg: DecodeMMConfig, gx: int, gy: int, n_tiles: int, arch: str = "blackhole"):
    """``(x, y, per_core_N, out_subblock_w)`` of a 1D row on a device grid: the grid per ``resolve_grid``, the
    per-core N raised to cover N tiles on that many cores, the subblock the largest divisor of the per-core N not
    above the row's bound nor Dest."""
    vx, vy = resolve_grid(cfg.grid, gx, gy, arch)
    pcn = max(1, math.ceil(n_tiles / (vx * vy)))
    if cfg.per_core_N:
        pcn = max(pcn, int(cfg.per_core_N))
    osw = _largest_divisor(pcn, max(1, min(int(cfg.out_subblock_w), _dest_tiles(cfg.fp32_acc))))
    return vx, vy, pcn, osw


def resolve_decode_mm_config(table, model: str, op: str, tp: int, batch: int, arch: str):
    """The env override when set, else the table row; None = today's path. A row of a Blackhole-only family
    resolves to None on any other arch (that kernel does not exist there; the row says so in ``arch``)."""
    env = os.environ.get(ENV_PREFIX + op.upper())
    cfg = parse_config_spec(env) if env else table.get((model, op, tp, batch_bucket(batch)))
    if cfg is not None and cfg.arch != "any" and cfg.arch != arch:
        return None
    return cfg


def current_config_rows():
    """Today's served configs in the table format (the lab's baselines): the DRAM-sharded kernel for qkv /
    out_proj / gate_up / down (LoFi, Blackhole-only) and the 1D lm_head config (K block = the largest power of two
    dividing the K tiles, subblock <= 4, LoFi, whole grid)."""
    rows = {}
    for model, d in MODELS.items():
        kt = d["hidden"] // TILE_SIZE
        bw = 32
        while bw > 1 and kt % bw:
            bw //= 2
        for tp in (4, 8):
            for b in BATCH_BUCKETS:
                for op in ("qkv", "out_proj", "gate_up", "down"):
                    rows[(model, op, tp, b)] = DecodeMMConfig(
                        "ds", "ds", 8, None, 1, "lofi", False, 0, "dram", "blackhole"
                    )
                rows[(model, "lm_head", tp, b)] = DecodeMMConfig("1d", "full", bw, None, 4, "lofi", False)
    return rows


# The proposed best rows (filled from the sweeps' RESULT best lines; empty until then, no default changes).
DECODE_MM_TABLE = {}


class KnobSpace:
    """Device-free enumeration of the program-config knobs of one ``[32, K] @ [K, N]`` decode matmul on a
    ``grid_xy`` worker grid, bounded by the L1 budget and the per-core DRAM stream bound.

    ``ds_cores`` is the DRAM-sharded kernel's core grid (``_decode_core_grid``; None when the shape has none),
    ``act_cores`` the activation's width-shard core count when the activation arrives sharded (the 1D kernel then
    needs the K block to divide the shard width; None for an interleaved activation).
    """

    DS_FIDELITIES = (("lofi", False), ("lofi", True), ("hifi2", False), ("hifi2", True))
    ONE_D_FIDELITIES = (
        ("hifi2", True),
        ("lofi", False),
        ("lofi", True),
        ("hifi2", False),
        ("hifi4", False),
        ("hifi4", True),
    )
    PIPELINE_WORKERS = (2, 3, 4)
    PIPELINE_BLOCKS = (2, 1, 4, 8)

    def __init__(self, k, n, wdtype, grid_xy, ds_cores=None, act_cores=None):
        self.k, self.n, self.wdtype = k, n, wdtype
        self.kt, self.nt = k // TILE_SIZE, n // TILE_SIZE
        self.gx, self.gy = grid_xy
        self.ds_cores = ds_cores
        self.act_cores = act_cores
        self.weight_bytes = k * n * BYTES_PER_ELEM[wdtype]
        self.dram_bytes = self.weight_bytes + TILE_SIZE * k * 2 + TILE_SIZE * n * 2
        self.roofline_us = self.dram_bytes / (DRAM_GBPS * 1e3)

    def core_bound_ok(self, cores: int) -> bool:
        return self.weight_bytes / max(1, cores) / (CORE_GBPS * 1e3) <= ROOFLINE_SLACK * self.roofline_us

    # -- DRAM-sharded family (in1 width-sharded over the DRAM banks, in0 width-sharded in L1) --
    def ds_blocks(self):
        """K blocks of the single-reader kernel: divide K, divide or span the activation shard width, fit L1."""
        if not self.ds_cores:
            return []
        shard_w = max(1, self.kt // self.ds_cores)
        nt_pad = _padded_n_tiles(self.n)
        return [
            bw
            for bw in _divisors(self.kt)
            if (shard_w % bw == 0 or bw % shard_w == 0)
            and _estimate_decode_l1_bytes_for_in0(nt_pad, bw, self.wdtype) <= L1_BUDGET
        ]

    def ds_pipeline(self):
        """(workers per bank, K block) of the multi-core pipeline: workers x 8 banks must fit the grid."""
        if not self.ds_cores:
            return []
        return [
            (w, bw)
            for w in self.PIPELINE_WORKERS
            if w * 8 <= self.gx * self.gy
            for bw in self.PIPELINE_BLOCKS
            if self.kt % bw == 0
        ]

    # -- 1D multicast family (in1 interleaved, one tile row of activation multicast in K blocks) --
    def one_d_pcn(self, cores: int) -> int:
        return max(1, math.ceil(self.nt / cores))

    def one_d_blocks(self, pcn: int):
        """K blocks on ``pcn`` tile columns per core: divide K (and the activation shard width when sharded), fit L1."""
        shard_w = max(1, self.kt // self.act_cores) if self.act_cores else None
        return [
            bw
            for bw in _divisors(self.kt)
            if _l1_1d(bw, pcn, self.wdtype) <= L1_BUDGET and (shard_w is None or shard_w % bw == 0)
        ]

    def one_d_grids(self, bw: int, fp32: bool):
        """(gx, gy, per_core_N, out_subblock_w) variants: the whole grid at the minimum / +1 / double per-core N and
        the divisor sub-grids of the device grid (``candidate_grids``), each with the widest subblock that fits
        Dest and with 1."""
        base = self.one_d_pcn(self.gx * self.gy)
        shapes = [(self.gx, self.gy, base), (self.gx, self.gy, base + 1), (self.gx, self.gy, 2 * base)]
        for vx, vy in candidate_grids(self.gx, self.gy):
            if (vx, vy) != (self.gx, self.gy):
                shapes.append((vx, vy, self.one_d_pcn(vx * vy)))
        out, seen = [], set()
        for vx, vy, pcn in shapes:
            cores = math.ceil(self.nt / pcn)
            if cores > vx * vy or _l1_1d(bw, pcn, self.wdtype) > L1_BUDGET or not self.core_bound_ok(cores):
                continue
            for osw in sorted({1, _largest_divisor(pcn, _dest_tiles(fp32))}):
                key = (vx, vy, pcn, osw)
                if key not in seen:
                    seen.add(key)
                    out.append(key)
        return out

    def enumerate(self):
        """Every knob combination the stages can visit (the cross product; the sweep itself is coordinate descent)."""
        cands = []
        for bw in self.ds_blocks():
            for fid, acc in self.DS_FIDELITIES:
                cands.append(dict(family="ds", bw=bw, fid=fid, acc=acc, wpb=0, cores=self.ds_cores))
        for w, bw in self.ds_pipeline():
            for fid, acc in self.DS_FIDELITIES:
                cands.append(dict(family="ds", bw=bw, fid=fid, acc=acc, wpb=w, cores=8 * w))
        for fid, acc in self.ONE_D_FIDELITIES:
            for bw in self.one_d_blocks(self.one_d_pcn(self.gx * self.gy)):
                for vx, vy, pcn, osw in self.one_d_grids(bw, acc):
                    cands.append(
                        dict(
                            family="1d",
                            bw=bw,
                            fid=fid,
                            acc=acc,
                            grid=(vx, vy),
                            pcn=pcn,
                            osw=osw,
                            cores=math.ceil(self.nt / pcn),
                        )
                    )
        return cands


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1).double()
    b = b.reshape(-1).double()
    if not (torch.isfinite(a).all() and torch.isfinite(b).all()) or a.numel() != b.numel():
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    den = a.norm() * b.norm()
    return float(a @ b / den) if den > 0 else float("nan")


def _note(exc: BaseException) -> str:
    text = str(exc).replace("\n", " ")
    for marker in ("TT_FATAL", "TT_THROW", "RuntimeError"):
        pos = text.find(marker)
        if pos >= 0:
            text = text[pos:]
            break
    return "_".join(text.split())[:140] or type(exc).__name__


@contextmanager
def _env(overrides):
    saved = {k: os.environ.get(k) for k in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _width_sharded_memcfg(k: int, cores: int, grid):
    """L1 width-sharded ``[32, k]`` over the first ``cores`` worker cores (row-major on the grid)."""
    rows = math.ceil(cores / grid.x)
    ranges = []
    for r in range(rows):
        last = min(grid.x, cores - r * grid.x) - 1
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, r), ttnn.CoreCoord(last, r)))
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(ttnn.CoreRangeSet(set(ranges)), (TILE_SIZE, k // cores), ttnn.ShardOrientation.ROW_MAJOR),
    )


class Cand:
    """One program-config candidate: ``call()`` runs the matmul (and whatever reshards the form needs) on the
    persistent activation and returns the output tensor; ``knobs`` is what the RESULT line prints."""

    def __init__(self, name: str, family: str, call, knobs: dict, env=None):
        self.name = name
        self.family = family
        self.call = call
        self.knobs = knobs
        self.env = env or {}


class ShapeLab:
    def __init__(self, mesh_device, model, op, shape_tag, tp, batch, k, n, wdtype, per_step):
        self.dev = mesh_device
        self.model, self.op, self.shape_tag, self.tp, self.batch = model, op, shape_tag, tp, batch
        self.k, self.n, self.wdtype, self.per_step = k, n, wdtype, per_step
        self.grid = mesh_device.compute_with_storage_grid_size()
        self.arch = mesh_device.arch()
        self.kt, self.nt = k // TILE_SIZE, n // TILE_SIZE
        self.results = []
        self.baseline = None
        self.ds_able = wdtype == ttnn.bfloat8_b and IN1 != "il" and can_dram_shard(k, n, dtype=wdtype)
        self.ds_cores = _decode_core_grid(k, n)[2] if self.ds_able else None
        self.act_kind, self.act_cores = self._act_layout()
        self.space = KnobSpace(
            k, n, wdtype, (self.grid.x, self.grid.y), ds_cores=self.ds_cores, act_cores=self.act_cores
        )
        self.dram_bytes, self.roofline_us = self.space.dram_bytes, self.space.roofline_us
        self.tag = (
            f"model={model} op={op} shape={shape_tag} tp={tp} b={batch} K={k} N={n} wdtype={DTYPE_TAG[wdtype]} "
            f"act={ACT} outcfg={OUT}"
        )

    def _act_layout(self):
        if ACT.startswith("ws"):
            cores = (
                int(ACT.split(":")[1])
                if ":" in ACT
                else (self.ds_cores or _largest_divisor(self.kt, self.grid.x * self.grid.y))
            )
            if cores > self.grid.x * self.grid.y or self.kt % cores:
                raise ValueError(
                    f"GEMMA4_LAB_ACT={ACT}: {cores} shards of {self.kt} K tiles on a {self.grid.x}x{self.grid.y} grid"
                )
            return "ws", cores
        if ACT == "l1":
            return "l1", None
        if ACT == "dram":
            return "dram", None
        raise ValueError(f"GEMMA4_LAB_ACT={ACT}")

    # ---- tensors ---------------------------------------------------------------------------------------------
    def build(self):
        torch.manual_seed(1234)
        w = (torch.randn(1, 1, self.k, self.n) * 0.02).to(torch.bfloat16)
        x = (torch.randn(1, 1, self.batch, self.k) * 2).to(torch.bfloat16)
        if self.wdtype == ttnn.bfloat8_b:
            w_q = ttnn.to_torch(ttnn.from_torch(w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)).float()
        else:
            w_q = w.float()
        self.golden = (x.float().reshape(self.batch, self.k) @ w_q.reshape(self.k, self.n)).reshape(self.batch, self.n)
        del w_q
        self.persistent = []
        self.x_dev = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.dev)
        self.persistent.append(self.x_dev)
        if self.batch == TILE_SIZE:
            self.x32 = self.x_dev
        else:
            x32 = torch.nn.functional.pad(x, (0, 0, 0, TILE_SIZE - self.batch))
            self.x32 = ttnn.from_torch(x32, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.dev)
            self.persistent.append(self.x32)
        self.lin = None
        self.w_ws = None
        if self.ds_able:
            self.lin = DramShardedLinear(w, self.dev, None, k=self.k, n=self.n, dtype=self.wdtype, cache_file_name=None)
            self.w_ws = self.lin.weight
            self.persistent.append(self.w_ws)
            self.ds_act_mc = activation_memcfg(self.k, self.n)
        self.w_il = None
        if IN1 != "ws" or not self.ds_able:
            if self.lin is not None:
                self.w_il = ttnn.to_memory_config(self.w_ws, ttnn.DRAM_MEMORY_CONFIG)
            else:
                self.w_il = ttnn.from_torch(
                    w,
                    dtype=self.wdtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.dev,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            self.persistent.append(self.w_il)
        # the activation in the boundary layout: 1D kernels take it as is; the DRAM-sharded kernel takes its own
        # shard grid (no reshard when the boundary layout already is that grid)
        if self.act_kind == "dram":
            self.x_act = self.x_dev
        elif self.act_kind == "l1":
            self.x_act = ttnn.to_memory_config(self.x_dev, ttnn.L1_MEMORY_CONFIG)
            self.persistent.append(self.x_act)
        else:
            self.x_act = ttnn.to_memory_config(self.x32, _width_sharded_memcfg(self.k, self.act_cores, self.grid))
            self.persistent.append(self.x_act)
        self.x_ds = None
        if self.ds_able:
            if self.act_kind == "ws" and self.act_cores == self.ds_cores:
                self.x_ds = self.x_act
            else:
                self.x_ds = ttnn.to_memory_config(self.x32, self.ds_act_mc)
                self.persistent.append(self.x_ds)
        ttnn.synchronize_device(self.dev)
        print(
            f"LAB {self.tag} grid={self.grid.x}x{self.grid.y} dram_mb={self.dram_bytes / 1e6:.1f} "
            f"roofline_us={self.roofline_us:.1f} ds_cores={self.ds_cores} act_cores={self.act_cores} "
            f"per_step={self.per_step} knob_space={len(self.space.enumerate())}"
        )

    def free(self):
        for t in self.persistent:
            t.deallocate(True)
        self.persistent = []

    # ---- helpers ---------------------------------------------------------------------------------------------
    def ckc(self, fidelity: str, fp32: bool):
        return ttnn.init_device_compute_kernel_config(
            self.arch,
            math_fidelity=FIDELITY[fidelity],
            math_approx_mode=False,
            fp32_dest_acc_en=fp32,
            packer_l1_acc=not fp32,
        )

    def out_mc(self, out=None):
        out = out or OUT
        if out == "ws":
            return ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG
        if out == "l1":
            return ttnn.L1_MEMORY_CONFIG
        return ttnn.DRAM_MEMORY_CONFIG

    def pc_1d(self, gx, gy, bw, pcn, osw):
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(gx, gy),
            in0_block_w=bw,
            out_subblock_h=1,
            out_subblock_w=osw,
            per_core_M=1,
            per_core_N=pcn,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )

    def knobs_1d(self, gx, gy, bw, pcn, osw, fidelity, fp32, out=None):
        return dict(
            grid=f"{gx}x{gy}",
            gspec=grid_spec_of(gx, gy, self.grid.x, self.grid.y),
            cores=min(gx * gy, math.ceil(self.nt / pcn)),
            bw=bw,
            osub=f"1x{osw}",
            pc=f"1x{pcn}",
            fid=fidelity,
            acc=int(fp32),
            in1="il",
            out=out or OUT,
            wpb=0,
            form="plain",
        )

    def cand_1d(self, name, gx, gy, bw, pcn, osw, fidelity, fp32, out=None):
        pc = self.pc_1d(gx, gy, bw, pcn, osw)
        ckc = self.ckc(fidelity, fp32)
        out_mc = self.out_mc(out)
        w, x = self.w_il, self.x_act

        def call():
            return ttnn.linear(x, w, program_config=pc, compute_kernel_config=ckc, memory_config=out_mc)

        return Cand(name, "1d", call, self.knobs_1d(gx, gy, bw, pcn, osw, fidelity, fp32, out))

    def ds_pc(self, bw, workers=1, pipeline=False):
        base = self.lin._decode_pc
        pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=bw, per_core_M=base.per_core_M, per_core_N=base.per_core_N, fused_activation=None
        )
        pc.num_workers_per_dram_bank = workers
        pc.use_multicore_pipeline = pipeline
        return pc

    def ds_knobs(self, pc, fidelity, fp32, form, out=None):
        rows, cols, cores = _decode_core_grid(self.k, self.n)
        workers = int(getattr(pc, "num_workers_per_dram_bank", 1))
        pipeline = bool(getattr(pc, "use_multicore_pipeline", False))
        return dict(
            grid=f"{cols}x{rows}",
            gspec="ds",
            cores=cores,
            bw=int(pc.in0_block_w),
            osub="1x1",
            pc=f"1x{int(pc.per_core_N)}",
            fid=fidelity or "lofi",  # an explicit program config without a compute config runs at LoFi
            acc=int(fp32),
            in1="ws",
            out=out or OUT,
            wpb=(workers if pipeline else 0),
            form=form,
        )

    def cand_ds(self, name, pc, fidelity=None, fp32=False, form="explicit", env=None, out=None):
        """DRAM-sharded candidates. ``model``: the DramShardedLinear call exactly as the model runs it on the
        boundary activation (reshard in, kernel, reshard out, Pad / Slice or the row view below one tile; DRAM
        interleaved activations only). ``explicit``: reshard the 32-row tile onto the kernel's grid unless the
        boundary layout already is that grid, the kernel, then the output to the boundary layout (``ws`` keeps
        the kernel's L1 width-sharded output). ``mm``: the kernel alone on the persistent sharded activation."""
        lin = self.lin
        ckc = self.ckc(fidelity, fp32) if fidelity else None
        out = out or OUT
        if form == "model":
            x = self.x_act

            def call():
                lin._decode_pc = pc
                return lin(x, compute_kernel_config=ckc, out_memory_config=self.out_mc(out) if out != "ws" else None)

        elif form == "explicit":
            w, act_mc, x_in, x_ready = lin.weight, self.ds_act_mc, self.x32, self.x_ds is self.x_act
            out_mc = self.out_mc(out)

            def call():
                x_sh = self.x_ds if x_ready else ttnn.to_memory_config(x_in, act_mc)
                res = ttnn.linear(
                    x_sh,
                    w,
                    program_config=pc,
                    compute_kernel_config=ckc,
                    memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                )
                if not x_ready:
                    x_sh.deallocate(True)
                if out != "ws":
                    moved = ttnn.to_memory_config(res, out_mc)
                    res.deallocate(True)
                    res = moved
                return res

        else:
            x_sh, w = self.x_ds, lin.weight
            out = "ws"

            def call():
                return ttnn.linear(
                    x_sh,
                    w,
                    program_config=pc,
                    compute_kernel_config=ckc,
                    memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                )

        return Cand(name, "ds", call, self.ds_knobs(pc, fidelity, fp32, form, out), env)

    # ---- timing ----------------------------------------------------------------------------------------------
    def _rows(self, t: torch.Tensor) -> torch.Tensor:
        return t.float().reshape(-1, int(t.shape[-1]))[: self.batch, : self.n]

    def time_candidate(self, cand: Cand):
        """Compile pass (host-side validation errors surface here, before any capture), then OPS_PER_TRACE calls
        in one trace, one warm replay, BATCHES x REPLAYS timed replays; the last call's output is the traced
        result the PCC guard uses."""
        dev = self.dev
        with _env(cand.env):
            out = cand.call()
            ttnn.synchronize_device(dev)
            eager = self._rows(ttnn.to_torch(out))
            out.deallocate(True)
            outs = []
            tid = ttnn.begin_trace_capture(dev, cq_id=0)
            try:
                outs = [cand.call() for _ in range(OPS_PER_TRACE)]
            finally:
                ttnn.end_trace_capture(dev, tid, cq_id=0)
            try:
                ttnn.synchronize_device(dev)
                ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
                means = []
                for _ in range(BATCHES):
                    ttnn.synchronize_device(dev)
                    t0 = time.perf_counter()
                    for _ in range(REPLAYS):
                        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
                    ttnn.synchronize_device(dev)
                    means.append((time.perf_counter() - t0) / (REPLAYS * OPS_PER_TRACE) * 1e6)
                traced = self._rows(ttnn.to_torch(outs[-1]))
            finally:
                ttnn.release_trace(dev, tid)
                for o in outs:
                    o.deallocate(True)
        return eager, traced, means

    def run(self, cand: Cand) -> dict:
        res = dict(cand=cand.name, family=cand.family, knobs=cand.knobs, ok=0, us=float("nan"), note="-", runs=[])
        res["pcc_base"] = res["pcc_golden"] = float("nan")
        try:
            eager, traced, means = self.time_candidate(cand)
        except Exception as exc:  # a rejected config (validation / L1) is a result, not a test failure
            res["note"] = _note(exc)
        else:
            res["us"] = sorted(means)[len(means) // 2]
            res["runs"] = means
            res["pcc_golden"] = _pcc(traced, self.golden)
            res["pcc_base"] = 1.0 if self.baseline is None else _pcc(traced, self.baseline)
            if not torch.equal(eager, traced):
                res["note"] = f"trace_vs_eager_pcc={_pcc(eager, traced):.6f}"
            elif not math.isfinite(res["pcc_base"]):
                res["note"] = "non_finite_output"
            elif res["pcc_base"] < PCC_MIN:
                res["note"] = f"pcc_below_{PCC_MIN}"
            else:
                res["ok"] = 1
            if self.baseline is None and res["ok"]:
                self.baseline = traced
                res["is_baseline"] = 1
        self.results.append(res)
        k = res["knobs"]
        runs = "/".join(f"{m:.1f}" for m in res["runs"])
        ratio = res["us"] / self.roofline_us if math.isfinite(res["us"]) else float("nan")
        print(
            f"RESULT lab {self.tag} cand={res['cand']} family={res['family']} grid={k['grid']} gridspec={k['gspec']} cores={k['cores']} "
            f"in0_block_w={k['bw']} out_subblock={k['osub']} per_core={k['pc']} fidelity={k['fid']} fp32acc={k['acc']} "
            f"in1={k['in1']} out={k['out']} workers_per_bank={k['wpb']} form={k['form']} ok={res['ok']} "
            f"pcc_base={res['pcc_base']:.6f} pcc_golden={res['pcc_golden']:.6f} us={res['us']:.2f} runs={runs} "
            f"dram_mb={self.dram_bytes / 1e6:.1f} roofline_us={self.roofline_us:.2f} ratio={ratio:.2f} "
            f"note={res['note']}"
        )
        return res

    def best(self, family=None, form=None):
        ok = [
            r
            for r in self.results
            if r["ok"] and (family is None or r["family"] == family) and (form is None or r["knobs"]["form"] == form)
        ]
        return min(ok, key=lambda r: r["us"]) if ok else None

    def report_best(self):
        base = next((r for r in self.results if r.get("is_baseline")), None)
        best = self.best()
        if base is None or best is None:
            print(f"RESULT best {self.tag} baseline=none best=none")
            return
        k = best["knobs"]
        cfg = (
            f"{best['family']}:grid={k['gspec']},bw={k['bw']},osub={k['osub']},pc={k['pc']},fid={k['fid']},"
            f"acc={k['acc']},in1={k['in1']},out={k['out']},wpb={k['wpb']},form={k['form']}"
        )
        gain = (base["us"] - best["us"]) / base["us"] * 100
        print(
            f"RESULT best {self.tag} baseline={base['cand']} baseline_us={base['us']:.2f} best={best['cand']} "
            f"us={best['us']:.2f} gain_pct={gain:.1f} roofline={self.roofline_us:.2f} "
            f"ratio={best['us'] / self.roofline_us:.2f} config={cfg} per_step={self.per_step} "
            f"step_ms_baseline={base['us'] * self.per_step / 1e3:.3f} step_ms_best={best['us'] * self.per_step / 1e3:.3f}"
        )

    # ---- candidate stages ------------------------------------------------------------------------------------
    def stage_baseline(self):
        if self.lin is not None:
            form = "model" if self.act_kind == "dram" else "explicit"
            return [self.cand_ds("baseline_ds", self.lin._decode_pc, form=form)]
        if self.op == "lm_head" and self.w_il is not None:
            pc = _get_lm_head_program_config(self.dev, TILE_SIZE, self.k, int(self.w_il.shape[-1]))
            if pc is not None:
                knobs = self.knobs_1d(
                    self.grid.x,
                    self.grid.y,
                    int(pc.in0_block_w),
                    int(pc.per_core_N),
                    int(pc.out_subblock_w),
                    "lofi",
                    False,
                )
                x, w, out_mc = self.x_act, self.w_il, self.out_mc()

                def call():
                    if OUT == "dram":
                        return ttnn.linear(x, w, program_config=pc)  # the model's call (no memory config)
                    return ttnn.linear(x, w, program_config=pc, memory_config=out_mc)

                return [Cand("baseline_lmh1d", "1d", call, knobs)]
        return [self.cand_auto("baseline_auto")]

    def cand_auto(self, name="auto"):
        x, w, out_mc = self.x_act, self.w_il, self.out_mc()
        knobs = dict(
            grid="auto",
            gspec="auto",
            cores=0,
            bw=0,
            osub="auto",
            pc="auto",
            fid="hifi2",
            acc=0,
            in1="il",
            out=OUT,
            wpb=0,
            form="plain",
        )
        return Cand(name, "auto", lambda: ttnn.linear(x, w, memory_config=out_mc), knobs)

    def stage_resolver(self):
        pc = decode_1d_progcfg(self.dev, self.k, int(self.w_il.shape[-1]))
        ckc = decode_compute_kernel_config(self.dev)
        x, w, out_mc = self.x_act, self.w_il, self.out_mc()
        fid = os.environ.get("GEMMA4_DECODE_MM_FIDELITY", "hifi2").lower()
        acc = os.environ.get("GEMMA4_DECODE_MM_FP32_ACC", "1") == "1"
        knobs = self.knobs_1d(self.grid.x, self.grid.y, int(pc.in0_block_w), int(pc.per_core_N), 1, fid, acc)
        call = lambda: ttnn.linear(
            x, w, program_config=pc, compute_kernel_config=ckc, memory_config=out_mc
        )  # noqa: E731
        return [Cand("1d_resolver", "1d", call, knobs)]

    def stage_2d(self):
        pc = prefill_progcfg(TILE_SIZE, self.k, self.n, max_cols=prefill_max_cols_default())
        gx, gy = pc.compute_with_storage_grid_size
        cands = []
        for in1, w in (("ws", self.w_ws), ("il", self.w_il)):
            if w is None:
                continue
            x, out_mc = self.x_act, self.out_mc()
            knobs = dict(
                grid=f"{gx}x{gy}",
                gspec=grid_spec_of(gx, gy, self.grid.x, self.grid.y),
                cores=min(gx, math.ceil(self.nt / int(pc.per_core_N))),
                bw=int(pc.in0_block_w),
                osub=f"{int(pc.out_subblock_h)}x{int(pc.out_subblock_w)}",
                pc=f"1x{int(pc.per_core_N)}",
                fid="lofi",
                acc=0,
                in1=in1,
                out=OUT,
                wpb=0,
                form="plain",
            )
            call = lambda x=x, w=w: ttnn.linear(x, w, program_config=pc, memory_config=out_mc)  # noqa: E731
            cands.append(Cand(f"2d_prefillcfg_{in1}", "2d", call, knobs))
        return cands

    def run_stage_ds(self, elapsed):
        base_bw = int(self.lin._decode_pc.in0_block_w)
        # S1: K block of the single-reader kernel (LoFi, as the model runs it)
        for bw in self.space.ds_blocks():
            if bw == base_bw or elapsed():
                continue
            self.run(self.cand_ds(f"ds_bw{bw}", self.ds_pc(bw)))
        b1 = self.best("ds")
        bw = b1["knobs"]["bw"] if b1 else base_bw
        # S2: fidelity / fp32 Dest at the best block
        for fid, fp32 in KnobSpace.DS_FIDELITIES:
            if (fid, fp32) == ("lofi", False) or elapsed():
                continue
            self.run(self.cand_ds(f"ds_bw{bw}_{fid}_acc{int(fp32)}", self.ds_pc(bw), fid, fp32))
        b2 = self.best("ds")
        fid, acc = (b2["knobs"]["fid"], bool(b2["knobs"]["acc"])) if b2 else ("lofi", False)
        fid_kw = None if (fid, acc) == ("lofi", False) else fid
        # S3: the multi-core pipeline (workers per bank at the fitted K block 2, then the other blocks)
        pipe = self.space.ds_pipeline()
        for w, pbw in pipe:
            if pbw != 2 or elapsed():
                continue
            self.run(self.cand_ds(f"ds_pipe{w}_bw2", self.ds_pc(2, w, True), fid_kw, acc))
        b3 = self.best("ds")
        if b3 and b3["knobs"]["wpb"]:
            w = b3["knobs"]["wpb"]
            for pw, pbw in pipe:
                if pw != w or pbw == 2 or elapsed():
                    continue
                self.run(self.cand_ds(f"ds_pipe{w}_bw{pbw}", self.ds_pc(pbw, w, True), fid_kw, acc))
        # the kernel alone and the row view below one tile, at the best DRAM-sharded config
        b4 = self.best("ds")
        if b4 is None:
            return
        kb = b4["knobs"]
        pc = self.ds_pc(kb["bw"], kb["wpb"] or 1, bool(kb["wpb"]))
        fid_b = None if (kb["fid"], kb["acc"]) == ("lofi", 0) else kb["fid"]
        if not elapsed():
            self.run(self.cand_ds("ds_best_mm", pc, fid_b, bool(kb["acc"]), form="mm"))
        if self.batch < TILE_SIZE and self.act_kind == "dram" and not elapsed():
            self.run(
                self.cand_ds(
                    "ds_best_rowview", pc, fid_b, bool(kb["acc"]), form="model", env={"GEMMA4_DRAM_SHARD_ROW_VIEW": "1"}
                )
            )

    def run_stage_1d(self, elapsed):
        if self.w_il is None:
            return
        gx, gy = self.grid.x, self.grid.y
        pcn0 = self.space.one_d_pcn(gx * gy)
        fid, fp32 = KnobSpace.ONE_D_FIDELITIES[0]
        # S1: K block over the whole grid, one tile column per core (HiFi2, fp32 Dest: the resolver's compute config)
        for bw in self.space.one_d_blocks(pcn0):
            if elapsed():
                return
            osw = _largest_divisor(pcn0, _dest_tiles(fp32))
            self.run(self.cand_1d(f"1d_bw{bw}", gx, gy, bw, pcn0, osw, fid, fp32))
        b1 = self.best("1d", form="plain")
        if b1 is None:
            return
        bw = b1["knobs"]["bw"]
        # S2: fidelity / fp32 Dest at the best block
        for fid2, fp2 in KnobSpace.ONE_D_FIDELITIES[1:]:
            if elapsed():
                return
            osw = _largest_divisor(pcn0, _dest_tiles(fp2))
            self.run(self.cand_1d(f"1d_bw{bw}_{fid2}_acc{int(fp2)}", gx, gy, bw, pcn0, osw, fid2, fp2))
        b2 = self.best("1d", form="plain")
        fid, fp32, bw = b2["knobs"]["fid"], bool(b2["knobs"]["acc"]), b2["knobs"]["bw"]
        # S3: grid / per-core N / output subblock at the best block and fidelity
        for vx, vy, pcn, osw in self.space.one_d_grids(bw, fp32):
            if (vx, vy, pcn, osw) == (gx, gy, pcn0, _largest_divisor(pcn0, _dest_tiles(fp32))) or elapsed():
                continue
            self.run(self.cand_1d(f"1d_g{vx}x{vy}_pc{pcn}_osw{osw}", vx, vy, bw, pcn, osw, fid, fp32))


def _stage_enabled(name: str) -> bool:
    return STAGES == "all" or name in [s.strip() for s in STAGES.split(",")]


@pytest.mark.parametrize("model_tag", ["31B", "12B"], ids=["m31B", "m12B"])
@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("tp", [4, 8], ids=["tp4", "tp8"])
@pytest.mark.parametrize("batch", [1, 32], ids=["bs1", "bs32"])
@pytest.mark.parametrize("mesh_device, device_params", SINGLE_CHIP, indirect=True)
def test_matmul_knob_lab(mesh_device, device_params, model_tag, op, tp, batch):
    t_start = time.perf_counter()

    def elapsed():
        over = time.perf_counter() - t_start > BUDGET_S
        if over:
            print(f"RESULT note budget_exhausted after {time.perf_counter() - t_start:.0f}s model={model_tag} op={op}")
        return over

    for shape_tag, k, n, wdtype, per_step in op_shapes(model_tag, op, tp):
        lab = ShapeLab(mesh_device, model_tag, op, shape_tag, tp, batch, k, n, wdtype, per_step)
        lab.build()
        try:
            for cand in lab.stage_baseline():
                lab.run(cand)
            assert lab.baseline is not None, f"baseline failed: {lab.results[-1]['note']}"
            if _stage_enabled("auto") and lab.w_il is not None and not any(r["family"] == "auto" for r in lab.results):
                lab.run(lab.cand_auto())
            if _stage_enabled("resolver") and lab.w_il is not None and not elapsed():
                for cand in lab.stage_resolver():
                    lab.run(cand)
            if _stage_enabled("2d") and not elapsed():
                for cand in lab.stage_2d():
                    lab.run(cand)
            if _stage_enabled("ds") and lab.lin is not None and not elapsed():
                lab.run_stage_ds(elapsed)
            if _stage_enabled("1d") and not elapsed():
                lab.run_stage_1d(elapsed)
            lab.report_best()
        finally:
            lab.free()


# ---- host-only checks of the shape table and the knob space ---------------------------------------------------


@pytest.mark.parametrize(
    "model, op, tp, expected",
    [
        ("31B", "qkv", 4, [("sliding", 5376, 4096), ("global", 5376, 5120)]),
        ("31B", "qkv", 8, [("sliding", 5376, 2048), ("global", 5376, 3072)]),
        ("31B", "out_proj", 4, [("sliding", 2048, 5376), ("global", 4096, 5376)]),
        ("31B", "gate_up", 8, [("mlp", 5376, 5376)]),
        ("31B", "down", 8, [("mlp", 2688, 5376)]),
        ("12B", "qkv", 4, [("sliding", 3840, 2048), ("global", 3840, 3072)]),
        ("12B", "out_proj", 8, [("sliding", 512, 3840), ("global", 1024, 3840)]),
        ("12B", "gate_up", 4, [("mlp", 3840, 7680)]),
        ("12B", "down", 4, [("mlp", 3840, 3840)]),
        ("31B", "lm_head", 4, [("lm_head", 5376, 65536)]),
        ("12B", "lm_head", 8, [("lm_head", 3840, 32768)]),
    ],
)
def test_op_shapes_host(model, op, tp, expected):
    """The per-chip shard shapes the lab sweeps are the served ones (G6's inventory table, attention/weights.py and
    shared_mlp.py at TP 4 / 8). Host only."""
    got = [(tag, k, n) for tag, k, n, _dtype, _per_step in op_shapes(model, op, tp)]
    assert got == expected
    dtypes = {dtype for _t, _k, _n, dtype, _p in op_shapes(model, op, tp)}
    assert dtypes == {ttnn.bfloat16 if op == "lm_head" else ttnn.bfloat8_b}


@pytest.mark.parametrize("gx, gy", [(13, 10), (12, 10), (8, 8)], ids=["bh13x10", "bh12x10", "wh8x8"])
@pytest.mark.parametrize(
    "k, n, wdtype, act_cores",
    [(5376, 4096, ttnn.bfloat8_b, None), (5376, 10752, ttnn.bfloat8_b, None), (5376, 4096, ttnn.bfloat8_b, 8)],
    ids=["wqkv31b_tp4", "gateup31b_tp4", "wqkv31b_tp4_actws8"],
)
def test_knob_space_host(k, n, wdtype, act_cores, gx, gy):
    """The device-free knob enumeration on a Blackhole and a Wormhole grid: every entry divides K, fits the L1
    estimate and the per-core stream bound, the resolver's block and the DRAM-sharded grid's blocks are in it,
    the grids are the device grid and its divisor sub-grids, and the cross product stays bounded."""
    ds_cores = _decode_core_grid(k, n)[2]
    space = KnobSpace(k, n, wdtype, (gx, gy), ds_cores=ds_cores, act_cores=act_cores)
    kt = k // TILE_SIZE
    ds_blocks = space.ds_blocks()
    assert ds_blocks and all(kt % bw == 0 for bw in ds_blocks)
    assert all(_estimate_decode_l1_bytes_for_in0(_padded_n_tiles(n), bw, wdtype) <= L1_BUDGET for bw in ds_blocks)
    pcn = space.one_d_pcn(gx * gy)
    blocks = space.one_d_blocks(pcn)
    assert all(kt % bw == 0 and _l1_1d(bw, pcn, wdtype) <= L1_BUDGET for bw in blocks)
    if act_cores:
        # a sharded activation: only blocks that divide the 21-tile shard (the resolver's 8 does not)
        assert blocks == [bw for bw in blocks if (kt // act_cores) % bw == 0] and 8 not in blocks and 21 in blocks
    else:
        assert decode_in0_block_w(k) in blocks
    grids = space.one_d_grids(blocks[0], True)
    assert (gx, gy, pcn, 1) in grids
    assert all(math.ceil(space.nt / p) <= vx * vy and osw <= 4 and p % osw == 0 for vx, vy, p, osw in grids)
    assert all(gx % vx == 0 and gy % vy == 0 for vx, vy, _p, _o in grids)  # divisor sub-grids of the device grid
    assert candidate_grids(13, 10) == [(13, 10), (13, 5)]
    assert {(8, 4), (4, 8), (4, 4)} <= set(candidate_grids(8, 8))
    cands = space.enumerate()
    assert 20 <= len(cands) <= 2000
    assert all(c["cores"] >= 1 and space.core_bound_ok(c["cores"]) or c["family"] == "ds" for c in cands)
    # the per-core stream bound prunes a one-core 1D config of a 22 MB weight
    assert not space.core_bound_ok(1)


# Sample rows exercising every grid-spec form on both grids (not a proposal; the sweeps fill DECODE_MM_TABLE).
SAMPLE_TABLE_ROWS = {
    ("31B", "qkv", 4, 32): DecodeMMConfig("1d", "y/2", 24, None, 1, "hifi2", True),
    ("12B", "gate_up", 8, 32): DecodeMMConfig(
        "1d", {"blackhole": (13, 10), "wormhole_b0": (8, 8)}, 12, 2, 2, "hifi2", True
    ),
    ("31B", "down", 4, 1): DecodeMMConfig("1d", "13x10", 8, None, 4, "lofi", False),
}


@pytest.mark.parametrize("gx, gy, arch", [(13, 10, "blackhole"), (8, 8, "wormhole_b0")], ids=["bh13x10", "wh8x8"])
def test_resolver_table_both_grids(gx, gy, arch, monkeypatch):
    """Every row of the resolver table (today's configs, the proposed rows, the sample grid-spec forms) resolves on
    a Blackhole 13x10 and a Wormhole 8x8 grid: 1D rows to a grid that fits, a per-core N that covers N, a subblock
    that divides it within Dest, a K block dividing K and CBs within L1; an explicit grid larger than the device
    clamps to the largest fitting grid (never None); a Blackhole-only family resolves to None on Wormhole and the
    row says so; the env override wins and round-trips the RESULT best config string."""
    rows = dict(current_config_rows())
    rows.update(DECODE_MM_TABLE)
    rows.update(SAMPLE_TABLE_ROWS)
    for (model, op, tp, batch), cfg in rows.items():
        monkeypatch.delenv(ENV_PREFIX + op.upper(), raising=False)
        got = resolve_decode_mm_config(rows, model, op, tp, batch, arch)
        if cfg.arch != "any" and cfg.arch != arch:
            assert got is None and cfg.family == "ds" and cfg.arch == "blackhole"
            continue
        assert got == cfg
        if cfg.family == "ds":
            continue  # its grid is the kernel's own gcd grid (_decode_core_grid), not a device-grid fraction
        for _tag, k, n, wdtype, _per in op_shapes(model, op, tp):
            nt = n // TILE_SIZE
            vx, vy, pcn, osw = resolve_1d_knobs(cfg, gx, gy, nt, arch)
            assert 1 <= vx <= gx and 1 <= vy <= gy, (model, op, tp, cfg)
            assert pcn * vx * vy >= nt and pcn % osw == 0 and osw <= _dest_tiles(cfg.fp32_acc)
            assert (k // TILE_SIZE) % cfg.in0_block_w == 0
            assert _l1_1d(cfg.in0_block_w, pcn, wdtype) <= L1_BUDGET, (model, op, tp, cfg, gx, gy, pcn)
    assert resolve_grid("13x10", gx, gy, arch) == (min(13, gx), min(10, gy))
    assert resolve_grid({"blackhole": (13, 10), "wormhole_b0": (8, 8)}, gx, gy, arch) == (gx, gy)
    assert resolve_grid("x/2+y/5", gx, gy, arch) == (gx // 2, max(1, gy // 5))
    assert grid_spec_of(gx, gy // 2, gx, gy) == "y/2" and grid_spec_of(gx, gy, gx, gy) == "full"
    monkeypatch.setenv(ENV_PREFIX + "QKV", "1d:grid=y/2,bw=24,pc=2,osw=1,fid=hifi2,acc=1,wpb=0,out=dram")
    cfg = resolve_decode_mm_config(rows, "31B", "qkv", 4, 32, arch)
    assert cfg.in0_block_w == 24 and cfg.per_core_N == 2 and resolve_grid(cfg.grid, gx, gy, arch) == (gx, gy // 2)
