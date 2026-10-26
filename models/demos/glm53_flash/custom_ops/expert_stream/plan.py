# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-side roofline model and planners for the routed-expert streaming ops (no ttnn, no device).

Three things live here (all pure Python; ``tests/test_moe_roofline_host.py`` exercises them on the 13x10 and 8x8
grids, per the generic-kernel rule):

1. **Bytes and floors** of one decode MoE layer per chip for the two layouts (``expert_bytes``, ``layer_floor_us``):
   the balanced hybrid read (column c's 72 experts x the row's 256-wide slice: 3.34 MB per active expert in bfp8)
   and the EP32 full-width read (9 experts x 26.7 MB per chip, 17.8 MB gate|up + 8.9 MB down).

2. **The weight-read parameter planner** (``plan_expert_stream`` -> ``StreamPlan``): groups (cores per DRAM bank),
   the pair-unit split (cores per expert), the reader prefetch depth, the NOC packet size, the down readers and
   the gather rotation, sized from the grid, the weight format and the expected active-expert count against the
   DRAM roofline. The cost model is calibrated on the single-chip measurements of 2026-10-20
   (``doc/mesh/expert_kernel_sweep.md`` sections 9-12); its constants carry their provenance and the two
   hypotheses (H1: the x-row gather of many cores convoys on the same L1 pages; H2: a single core streams one bank at
   ~11 GB/s) are what the device A/B in ``tests/test_moe_roofline_mesh.py`` measures. The plan is a *knob*
   (``GLM53_EXPERT_PLAN=roofline``); the default path keeps today's values (``legacy_plan``).

3. **EP32 skew** (``plan_ep_placement``, ``chip_bytes``, ``column_skew``): the per-chip expected active bytes from a
   measured expert popularity, a balanced placement (LPT over expected bytes, hot experts spread one per chip), the
   optional replication of the hottest experts, and the column-max model the step follows (every column waits for
   its slowest chip: 33-502 us at 4 users/row, 112-661 at 32, ``_logs/mesh/report_prof_ep32_ep_b{4,32}``).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

TILE = 32
BF16_TILE_BYTES = 2048
BFP8_TILE_BYTES = 1088
BFP4_TILE_BYTES = 576
TILE_BYTES = {"bfp8": BFP8_TILE_BYTES, "bfp4": BFP4_TILE_BYTES, "bf16": BF16_TILE_BYTES}
MAX_PACKET_BYTES = 16384  # Blackhole NOC max burst

# ---------------------------------------------------------------------------------------------------- roofline
DRAM_PEAK_GBS = 512.0  # Blackhole p150 GDDR6 (the sweep doc's "% of peak" base)
# the chip-wide ceiling of the EP gate|up / down streams when every core streams (the busiest chip of the 4-users/row
# EP32 profile, (6,0): gate|up 336 us and down 166 us, i.e. 160 MB / 80 MB at ~480 GB/s IF all 9 of its experts were
# active, 93 % of the 512 GB/s peak; an inference, since the profile does not record the active count). The hybrid
# block op (72 narrow slices, ppb = 1) plateaued lower: 300-330 GB/s at 69 of 72 active (sweep sections 8/9, 64 %).
STREAM_PLATEAU_GBS = 470.0
HYBRID_BLOCK_PLATEAU_GBS = 330.0
# H2: one core streaming one bank alone: 278 KB (a bfp8 gate|up pair at K = 4096) in ~23.2 us -> ~12 GB/s. Derived
# from the units A/B of 2026-10-20 (per-expert 288 us vs units 579 us at 2 active experts, T = 32: see
# ``fit_gather_convoy``, which solves stream + compute per pair = 29.1 us); the sparse-matmul sweep saw ~14 GB/s per
# core with 1 KB tile reads.
CORE_STREAM_GBS = 12.0
# per (gate, up) pair and 32-row pass: 2 x K/32 tile matmuls (HiFi2, ~32 cycles each at 1.35 GHz) + the SFPU
# clamped SwiGLU + the D @ h product + pack / reconfig: ~6 us at K = 4096
PAIR_COMPUTE_US = 6.0
# the tilize of the pass's 32 x K bf16 rows (K/32 = 128 tiles) plus the D tile build: ~6 us
TILIZE_US = 6.0
# H1 (the convoy): every gathering core reads the pass's token rows from the same L1 pages in the same order, so the
# reads serialize on one source core's NOC port at a time: ~2.6 us per concurrent gatherer (fit below). With the
# rows rotated per core (``gather_rotate``) the sources spread over min(n, rows) L1 banks: ~0.16 us per gatherer.
GATHER_US_PER_CORE_CONVOY = 2.6
GATHER_US_PER_CORE_ROTATED = 0.16
# fixed per-launch cost of a streaming program (routing scan, mailbox handshakes, the resident x load, the final
# barrier): the 33 us floor of the idle chips in the EP32 profile (7 us gate|up + 26 us down with 0 active experts)
LAUNCH_US_GATE_UP = 7.0
LAUNCH_US_DOWN = 26.0
# the down op at the EP shape (I = 2048, 8 readers per bank, 2 columns each): every core streams 139 KB per active
# expert and reads every pass's 128 KB activation block; measured 76-109 us at 1-3 active experts, ~200-300 at 9
DOWN_US_PER_EXPERT = 22.0
DOWN_US_PER_PASS = 4.0

L1_BYTES = 1_464_832  # Blackhole worker L1
L1_RESERVED_BYTES = 120_000  # firmware, mailboxes, the generic op's own CB bookkeeping (empirical margin)


def fmt(dtype) -> str:
    """ttnn dtype or name -> 'bfp8' | 'bfp4' | 'bf16'."""
    if isinstance(dtype, str):
        return dtype
    name = getattr(dtype, "name", str(dtype)).lower()
    if "bfloat4" in name or "bfp4" in name:
        return "bfp4"
    if "bfloat8" in name or "bfp8" in name:
        return "bfp8"
    return "bf16"


def expert_bytes(K: int, I: int, H: int, w13: str = "bfp8", w2: str = "bfp8") -> tuple[int, int]:
    """(gate|up bytes, down bytes) of ONE expert's slice [K, 2I] / [I, H] in the given formats."""
    gu = (K // TILE) * (2 * I // TILE) * TILE_BYTES[fmt(w13)]
    dn = (I // TILE) * (H // TILE) * TILE_BYTES[fmt(w2)]
    return gu, dn


def floor_us(nbytes: float, gbs: float = DRAM_PEAK_GBS) -> float:
    return nbytes / (gbs * 1e3)  # bytes / (GB/s * 1e3) = us


@dataclass(frozen=True)
class LayerShape:
    E: int = 288
    H: int = 4096
    I: int = 2048
    k: int = 8
    rows: int = 8
    cols: int = 4

    @property
    def D(self) -> int:
        return self.rows * self.cols


def layer_floor_us(
    shape: LayerShape, layout: str, users_per_row: int, active: int | None = None, w13: str = "bfp8", w2: str = "bfp8"
) -> dict:
    """Per-chip bytes and DRAM floor of the routed-expert read of one decode MoE layer.

    ``hybrid``: chip (r, c) streams the row slice (I / rows) of its column's E / cols experts that any of the
    rows x users gathered tokens routes to (``active`` of 72; the T256 real routing: 28 at 4 users/row, ~31 at
    32). ``ep``: the chip's E / D full-width experts that any token of its column's rows routes to (``active`` of
    9; real routing 2-5 at 4 users/row, up to 9 at 32)."""
    tokens = shape.rows * users_per_row
    if layout == "hybrid":
        E_local, I_local = shape.E // shape.cols, shape.I // shape.rows
        gu, dn = expert_bytes(shape.H, I_local, shape.H, w13, w2)
        if active is None:  # every expert any gathered token picks: 1 - (1 - 8/288)^tokens of the 72
            active = round(E_local * (1 - (1 - shape.k / shape.E) ** tokens))
        total = active * (gu + dn)
        return dict(
            layout=layout, tokens=tokens, active=active, of=E_local, gu=gu, dn=dn, bytes=total, floor_us=floor_us(total)
        )
    E_local = shape.E // shape.D
    gu, dn = expert_bytes(shape.H, shape.I, shape.H, w13, w2)
    if active is None:
        active = round(E_local * (1 - (1 - shape.k / shape.E) ** tokens))
    total = active * (gu + dn)
    return dict(
        layout=layout, tokens=tokens, active=active, of=E_local, gu=gu, dn=dn, bytes=total, floor_us=floor_us(total)
    )


# ---------------------------------------------------------------------------------------------------- stream plan
@dataclass
class StreamPlan:
    """The weight-read parameters of the dispatch ops (``ExpertDispatch`` kwargs) plus the model's expectation."""

    groups: int = 12  # gate|up cores per DRAM bank
    units: bool = False  # pair-unit LPT (``*_gate_up_dispatch_units.cpp``) instead of whole experts per core
    unit_pairs: int = 1  # (gate, up) pairs per unit in units mode: ppb / unit_pairs cores per bank per expert
    prefetch_pairs: int = 1  # pairs in flight ahead of the compute (reader slots = prefetch_pairs + 1)
    packet_tiles: int | None = None  # NOC packet in weight tiles (None = the largest divisor <= 16 KB)
    down_readers: int = 8  # down cores per bank (16 / readers output tile columns each)
    gather_rotate: bool = False  # rotate each core's row-gather start (breaks the convoy, H1)
    # expectations (us) for the planned point
    expected_gate_up_us: float = 0.0
    expected_down_us: float = 0.0
    expected_floor_us: float = 0.0
    notes: list[str] = field(default_factory=list)

    def op_kwargs(self) -> dict:
        return dict(
            groups=self.groups,
            units=self.units,
            unit_pairs=self.unit_pairs,
            prefetch_pairs=self.prefetch_pairs,
            packet_tiles=self.packet_tiles,
            down_readers=self.down_readers,
            gather_rotate=self.gather_rotate,
        )

    def as_dict(self) -> dict:
        return asdict(self)


def legacy_plan(E_local: int, I: int, banks: int = 8) -> StreamPlan:
    """Today's values: 12 groups, whole experts per core (units only by GLM53_STREAM_UNITS), one pair ahead, the
    largest packet, 8 down readers at the EP width (2 at I = 256: ``ExpertDownStream`` picks it), no rotation."""
    return StreamPlan(groups=12, units=False, unit_pairs=1, prefetch_pairs=1, packet_tiles=None, down_readers=8)


def packet_tiles_default(kt: int, tile_bytes: int) -> int:
    return max(d for d in range(1, kt + 1) if kt % d == 0 and d * tile_bytes <= MAX_PACKET_BYTES)


def gate_up_l1_bytes(kt: int, tile_bytes: int, prefetch_pairs: int, T: int, E: int, ep: bool) -> int:
    """L1 of the gate|up dispatch op's CBs (``ExpertGateUpDispatch.__call__``): x row block + x tiles (2 x kt bf16
    tiles), the weight ring ((prefetch + 1) pairs), the routing / index / weight tiles, D, h, table, out."""
    rt = max(1, T // TILE)
    e_tiles = -(-E // TILE)
    c_tiles = rt + 1 if ep else e_tiles
    r_tiles = rt if ep else rt * e_tiles
    ring = (prefetch_pairs + 1) * 2 * kt * tile_bytes
    return 2 * kt * BF16_TILE_BYTES + ring + (r_tiles + 2 * c_tiles + 2 + 2 + 2) * BF16_TILE_BYTES + 2 * 128


def max_prefetch_pairs(
    kt: int, tile_bytes: int, T: int, E: int, ep: bool, budget: int = L1_BYTES - L1_RESERVED_BYTES
) -> int:
    best = 1
    for p in range(1, 8):
        if gate_up_l1_bytes(kt, tile_bytes, p, T, E, ep) <= budget:
            best = p
    return best


def core_grid_banks(grid: tuple[int, int], banks: int = 8) -> int:
    """How many worker cores per bank the grid can place (``_BankStreamOp``: the bank's optimal core + neighbours)."""
    gx, gy = grid
    return (gx * gy) // banks


def gate_up_time_us(
    *,
    kt: int,
    ppb: int,
    tile_bytes: int,
    groups: int,
    units: bool,
    unit_pairs: int,
    n_active: int,
    passes_per_expert: float,
    gather_rotate: bool,
    banks: int = 8,
    tokens: int = 32,
) -> tuple[float, dict]:
    """The cost model of one gate|up dispatch launch (us): the slowest core's stream + compute + gather, with the
    chip's DRAM plateau and the per-core single-stream rate (H2) and the gather convoy (H1) as the two regimes."""
    pair_bytes = 2 * kt * tile_bytes
    passes = max(1, math.ceil(passes_per_expert))
    if n_active == 0:
        return LAUNCH_US_GATE_UP, dict(cores=0)
    if units:
        u = max(1, min(unit_pairs, ppb))
        chunks = ppb // u  # units per expert per bank
        n_units = n_active * chunks  # per bank
        per_core_units = math.ceil(n_units / groups)
        cores_busy = min(groups, n_units) * banks
        # each unit: u pairs streamed (re-streamed per pass when u > 1 and passes > 1), one gather + tilize per
        # pass, u pair computes per pass
        stream_reps = 1 if u == 1 else passes
        stream_bytes_core = per_core_units * u * pair_bytes * stream_reps
        gatherers = cores_busy
    else:
        per_core_units = math.ceil(n_active / groups)  # whole experts per core
        cores_busy = min(groups, n_active) * banks
        stream_reps = 1 if ppb == 1 else passes
        stream_bytes_core = per_core_units * ppb * pair_bytes * stream_reps
        gatherers = cores_busy
        u = ppb
    stream_us = stream_bytes_core / (CORE_STREAM_GBS * 1e3)  # the busiest core's stream at the single-core rate
    per_pass_gather = gather_us(gatherers, gather_rotate, tokens)
    compute_us = per_core_units * passes * (per_pass_gather + u * PAIR_COMPUTE_US)
    # the calibrated (serial) form: the 2026-10-20 points are reproduced with stream + gather + compute added per
    # core (the one-pair-ahead ring hides little at these sizes); the chip's DRAM ceiling bounds the whole launch
    serial = LAUNCH_US_GATE_UP + stream_us + compute_us
    total_bytes = n_active * ppb * pair_bytes * banks * (1 if (units and u == 1) or ppb == 1 else passes)
    ceiling = floor_us(total_bytes, STREAM_PLATEAU_GBS)
    total = max(serial, ceiling)
    detail = dict(
        cores=cores_busy,
        per_core_units=per_core_units,
        stream_us=stream_us,
        compute_us=compute_us,
        serial_us=serial,
        ceiling_us=ceiling,
        gather_us_per_pass=per_pass_gather,
    )
    return total, detail


def gather_us(gatherers: int, rotate: bool, tokens: int) -> float:
    """Gather + tilize of one 32-row pass on one core when ``gatherers`` cores do it concurrently (H1)."""
    if rotate:
        spread = min(gatherers, max(1, tokens))
        return TILIZE_US + GATHER_US_PER_CORE_ROTATED * gatherers * (gatherers / spread) ** 0.5
    return TILIZE_US + GATHER_US_PER_CORE_CONVOY * gatherers


def down_time_us(n_active: int, n_passes: int, I: int, H: int, tile_bytes: int, readers: int, banks: int = 8) -> float:
    if n_active == 0:
        return LAUNCH_US_DOWN
    seg_tiles = I // TILE
    cols = (H // TILE // banks) // readers
    block = cols * seg_tiles * tile_bytes
    cores = readers * banks
    stream_us = n_active * block / (CORE_STREAM_GBS * 1e3)
    act_us = n_passes * DOWN_US_PER_PASS
    serial = LAUNCH_US_DOWN + max(stream_us, n_active * DOWN_US_PER_EXPERT) + act_us
    return max(serial, floor_us(n_active * block * cores, STREAM_PLATEAU_GBS))


def fit_gather_convoy(
    per_expert_us: float = 288.0, units_us: float = 579.0, ppb: int = 8, n_active: int = 2, groups: int = 12
):
    """The two-point fit behind H1 / H2 (the traced 8x4 EP A/B of 2026-10-20, T = 32, 2 active experts, 1 pass):
    per-expert mode = 16 gathering cores, one core streams 8 pairs; units mode = 96 cores, 2 units on the busiest.
    Solves S + C (stream + compute per pair) and g (gather us per concurrent gatherer)."""
    n_per_expert = n_active * 8
    n_units = min(groups, n_active * ppb) * 8
    units_per_core = math.ceil(n_active * ppb / groups)
    # per_expert = ppb (S+C) + g n_per_expert ; units = units_per_core ((S+C) + g n_units)
    a = [[ppb, n_per_expert], [units_per_core, units_per_core * n_units]]
    b = [per_expert_us - LAUNCH_US_GATE_UP - TILIZE_US, units_us - LAUNCH_US_GATE_UP - units_per_core * TILIZE_US]
    det = a[0][0] * a[1][1] - a[0][1] * a[1][0]
    sc = (b[0] * a[1][1] - a[0][1] * b[1]) / det
    g = (a[0][0] * b[1] - b[0] * a[1][0]) / det
    return dict(stream_plus_compute_us=sc, gather_us_per_core=g)


def plan_expert_stream(
    *,
    grid: tuple[int, int],
    E_local: int,
    K: int,
    I: int,
    H: int,
    T: int,
    w13: str = "bfp8",
    w2: str = "bfp8",
    expected_active: float,
    tokens_per_expert: float = 4.0,
    ep: bool = True,
    banks: int = 8,
) -> StreamPlan:
    """Pick the streaming parameters for a (grid, format, expected activity) point by the cost model.

    ``grid``: the chip's compute grid (13x10 on Blackhole p150, 8x8 on the forced-8x8 test case); ``T``: the op's
    token rows (the dispatched rows, tile-padded); ``expected_active``: active experts per launch;
    ``tokens_per_expert``: rows per active expert (passes = ceil(rows / 32))."""
    kt = K // TILE
    tb13 = TILE_BYTES[fmt(w13)]
    tb2 = TILE_BYTES[fmt(w2)]
    ppb = (I // TILE) // banks
    max_groups = min(32, core_grid_banks(grid, banks))
    n_active = max(1, round(expected_active))
    passes = max(1.0, tokens_per_expert / TILE)
    pre_max = max_prefetch_pairs(kt, tb13, T, E_local, ep)
    best: StreamPlan | None = None
    best_t = float("inf")
    candidates = []
    for groups in sorted({g for g in (6, 8, 12, 15, 16, max_groups) if 1 <= g <= max_groups}):
        for units, unit_pairs in [(False, 1)] + [(True, u) for u in range(1, ppb + 1) if ppb % u == 0]:
            for rotate in (False, True):
                t, detail = gate_up_time_us(
                    kt=kt,
                    ppb=ppb,
                    tile_bytes=tb13,
                    groups=groups,
                    units=units,
                    unit_pairs=unit_pairs,
                    n_active=n_active,
                    passes_per_expert=passes,
                    gather_rotate=rotate,
                    banks=banks,
                    tokens=T,
                )
                candidates.append((t, groups, units, unit_pairs, rotate, detail))
                if t < best_t - 1e-9:
                    best_t = t
                    best = StreamPlan(groups=groups, units=units, unit_pairs=unit_pairs, gather_rotate=rotate)
    assert best is not None
    # prefetch: as deep as L1 allows when the stream is the bound (bfp4 pairs are half the size: 2-3 ahead fit)
    best.prefetch_pairs = pre_max if pre_max >= 2 else 1
    best.packet_tiles = packet_tiles_default(kt, tb13)
    # down readers: the fewest that keep <= 8 output columns per core and the expert block <= 128 tiles
    C = H // TILE // banks
    seg = I // TILE
    readers = next(r for r in (1, 2, 4, 8, 16) if C % r == 0 and C // r <= 8 and (C // r) * seg <= 128)
    best.down_readers = readers
    n_passes = n_active * math.ceil(passes)
    best.expected_gate_up_us = best_t
    best.expected_down_us = down_time_us(n_active, n_passes, I, H, tb2, readers, banks)
    gu, dn = expert_bytes(K, I, H, w13, w2)
    best.expected_floor_us = floor_us(n_active * (gu + dn))
    best.notes = [
        f"grid {grid[0]}x{grid[1]}: {max_groups} cores per bank available",
        f"{n_active} active experts x ({gu / 1e6:.1f} + {dn / 1e6:.1f}) MB -> floor {best.expected_floor_us:.0f} us at {DRAM_PEAK_GBS:.0f} GB/s",
        f"prefetch {best.prefetch_pairs} pair(s) ahead (L1 allows {pre_max}), packet {best.packet_tiles} tiles",
    ]
    return best


def plan_from_env(env: dict, E_local: int, I: int, banks: int = 8) -> StreamPlan | None:
    """The knob plumbing: ``GLM53_EXPERT_PLAN`` unset / ``legacy`` -> None (today's constructor defaults);
    ``explicit`` -> the individual knobs (``GLM53_STREAM_GROUPS``, ``GLM53_STREAM_UNITS``, ``GLM53_STREAM_UNIT_PAIRS``,
    ``GLM53_STREAM_PREFETCH``, ``GLM53_STREAM_PACKET_TILES``, ``GLM53_DISPATCH_DOWN_READERS``,
    ``GLM53_STREAM_GATHER_ROTATE``) over the legacy values; ``roofline`` -> ``plan_expert_stream`` for the point
    described by ``GLM53_EXPERT_PLAN_GRID`` (``13x10``), ``GLM53_EXPERT_PLAN_ACTIVE`` (expected active experts),
    ``GLM53_EXPERT_PLAN_T`` (dispatched rows), the formats from ``GLM53_EXPERT_W13_DTYPE`` / ``GLM53_MODEL_EXPERT_DOWN_DTYPE``,
    with the explicit knobs applied on top."""
    mode = env.get("GLM53_EXPERT_PLAN", "legacy")
    if mode == "legacy":
        return None
    if mode == "roofline":
        gx, gy = (int(v) for v in env.get("GLM53_EXPERT_PLAN_GRID", "13x10").split("x"))
        plan = plan_expert_stream(
            grid=(gx, gy),
            E_local=E_local,
            K=4096,
            I=I,
            H=4096,
            T=int(env.get("GLM53_EXPERT_PLAN_T", "32")),
            w13=env.get("GLM53_EXPERT_W13_DTYPE", "bfp8") or "bfp8",
            w2=env.get("GLM53_MODEL_EXPERT_DOWN_DTYPE", "bfp8") or "bfp8",
            expected_active=float(env.get("GLM53_EXPERT_PLAN_ACTIVE", "3")),
            tokens_per_expert=float(env.get("GLM53_EXPERT_PLAN_ROWS_PER_EXPERT", "4")),
            ep=True,
            banks=banks,
        )
    elif mode == "explicit":
        plan = legacy_plan(E_local, I, banks)
    else:
        raise ValueError(f"GLM53_EXPERT_PLAN={mode!r}: legacy | explicit | roofline")
    if "GLM53_STREAM_GROUPS" in env:
        plan.groups = int(env["GLM53_STREAM_GROUPS"])
    if "GLM53_STREAM_UNITS" in env:
        plan.units = env["GLM53_STREAM_UNITS"] == "1"
    if "GLM53_STREAM_UNIT_PAIRS" in env:
        plan.unit_pairs = int(env["GLM53_STREAM_UNIT_PAIRS"])
    if "GLM53_STREAM_PREFETCH" in env:
        plan.prefetch_pairs = int(env["GLM53_STREAM_PREFETCH"])
    if "GLM53_STREAM_PACKET_TILES" in env:
        plan.packet_tiles = int(env["GLM53_STREAM_PACKET_TILES"])
    if "GLM53_DISPATCH_DOWN_READERS" in env:
        plan.down_readers = int(env["GLM53_DISPATCH_DOWN_READERS"])
    if "GLM53_STREAM_GATHER_ROTATE" in env:
        plan.gather_rotate = env["GLM53_STREAM_GATHER_ROTATE"] == "1"
    return plan


# ---------------------------------------------------------------------------------------------------- E1 (2026-10-26)
# Decode roofline action E1: the block ops' fixed cost (selector matmul + x copy folded, the word-wise scan, the fused
# single program) and the 256-user dispatch form. Knobs (all default = today's path, read at construction in
# tt/moe.py): GLM53_STREAM_FOLD=1 (routing_fold), GLM53_STREAM_SCAN=words (scan_words), GLM53_STREAM_FUSED=1 (fused),
# GLM53_EXPERTS_32ROW=dense | dispatch | units (the 32-users/row expert form; dense = today's GLM53_DENSE_EXPERTS pass).
E1_FOLD_ENV = "GLM53_STREAM_FOLD"
E1_SCAN_ENV = "GLM53_STREAM_SCAN"
E1_FUSED_ENV = "GLM53_STREAM_FUSED"
EXPERTS_32ROW_ENV = "GLM53_EXPERTS_32ROW"
EXPERTS_32ROW_MODES = ("dense", "dispatch", "units")
NUM_L1_SEMAPHORES = 16  # tt_metal/impl/buffers/semaphore.hpp NUM_SEMAPHORES (host-side L1 semaphores per core)
FUSED_SLOT_BITS = 4  # arrival counter per (group, expert-in-group): counts to `banks` (8) in 4 bits
FUSED_SLOTS_PER_SEM = 7  # 7 x 4 = 28 bits: bit 31 (the NOC atomic's wrap bit) is never touched


def e1_stream_knobs(env: dict) -> dict:
    """``{"routing_fold", "scan_words", "fused"}`` from the env (defaults = today's program)."""
    scan = env.get(E1_SCAN_ENV, "elements")
    if scan not in ("elements", "words"):
        raise ValueError(f"{E1_SCAN_ENV}={scan!r}: elements | words")
    return dict(
        routing_fold=env.get(E1_FOLD_ENV, "0") == "1",
        scan_words=scan == "words",
        fused=env.get(E1_FUSED_ENV, "0") == "1",
    )


def experts_32row_mode(env: dict) -> str:
    mode = env.get(EXPERTS_32ROW_ENV, "dense")
    if mode not in EXPERTS_32ROW_MODES:
        raise ValueError(f"{EXPERTS_32ROW_ENV}={mode!r}: expected one of {EXPERTS_32ROW_MODES}")
    return mode


def fused_semaphores(groups: int, E_local: int) -> tuple[int, int]:
    """(semaphores per gate|up group, total) of the fused op's arrival counters: group g's k-th expert lives in the
    4-bit slot ``k % 7`` of semaphore ``g * per_group + k // 7``; a group has at most ceil(E / groups) experts."""
    slots = -(-E_local // groups)
    per_group = -(-slots // FUSED_SLOTS_PER_SEM)
    total = groups * per_group
    if total > NUM_L1_SEMAPHORES:
        raise ValueError(
            f"fused op: {groups} groups x {per_group} semaphores = {total} > {NUM_L1_SEMAPHORES} L1 semaphores "
            f"(E_local = {E_local}: pick groups in {fused_group_choices(E_local)})"
        )
    return per_group, total


def fused_group_choices(E_local: int, max_groups: int = 16) -> list[int]:
    out = []
    for g in range(1, max_groups + 1):
        slots = -(-E_local // g)
        if g * (-(-slots // FUSED_SLOTS_PER_SEM)) <= NUM_L1_SEMAPHORES:
            out.append(g)
    return out


def down_readers_for(I: int, H: int, banks: int = 8) -> int:
    """``op._down_readers``: the fewest readers per bank keeping <= 8 output columns and <= 128 tiles per block."""
    C, seg = H // TILE // banks, I // TILE
    return next(r for r in (1, 2, 4, 8, 16) if C % r == 0 and C // r <= 8 and (C // r) * seg <= 128)


def fused_core_plan(
    grid: tuple[int, int],
    E_local: int,
    I: int,
    H: int,
    banks: int = 8,
    groups: int | None = None,
    down_readers: int | None = None,
) -> dict:
    """Cores of the fused single program on a grid: the down readers per bank as the two-op form (2 at I = 256, 4 at
    1024, 8 at 2048), then the gate|up groups = the largest of today's 12 that fits beside them and keeps the
    arrival-counter semaphores within the 16 per core (13x10: 12 + 2 per bank = 112 cores; 8x8: 6 + 2 = 64)."""
    gx, gy = grid
    cores = gx * gy
    readers = down_readers_for(I, H, banks) if down_readers is None else int(down_readers)
    assert readers * banks <= cores, (readers, banks, cores)
    room = (cores - readers * banks) // banks
    choices = fused_group_choices(E_local)
    if groups is None:
        cand = [g for g in choices if g <= min(12, room)]
        assert cand, f"no fused group count fits {grid} with {readers} down readers"
        groups = max(cand)
    else:
        groups = int(groups)
        assert groups <= room, f"{groups} gate|up groups do not fit beside {readers} down readers on {grid}"
        assert groups in choices, f"{groups} groups need more than {NUM_L1_SEMAPHORES} semaphores at E = {E_local}"
    spg, n_sems = fused_semaphores(groups, E_local)
    return dict(
        groups=groups,
        down_readers=readers,
        gate_up_cores=groups * banks,
        down_cores=readers * banks,
        sems_per_group=spg,
        semaphores=n_sems,
    )


def plan_for_32row(
    grid: tuple[int, int],
    E_local: int,
    I: int,
    H: int,
    T: int = 256,
    w13: str = "bfp8",
    w2: str = "bfp8",
    units: bool = False,
    expected_active: float | None = None,
    banks: int = 8,
    roofline: bool = False,
) -> StreamPlan:
    """The dispatch-op plan of the 32-users/row point (GLM53_EXPERTS_32ROW=dispatch | units). Default: the MEASURED
    configuration (12 groups, 8 down readers = 2 output columns per core on 64 cores, no rotation: 779 us at 58
    active on the chip test, expert_kernel_sweep.md section 10). ``roofline`` (GLM53_EXPERT_PLAN=roofline): the cost
    model's choice at T gathered tokens with the real-routing activity (31 of 72 experts at 256 tokens) and
    T x 8 / 288 rows per expert (one 32-row pass) -- a hypothesis (H1 rotation, 15 groups) the device A/B ranks. The
    units form is forced when asked (its pair split only exists at ppb > 1: the (2,4) block's I = 1024; at the 8x4's
    I = 256 it is the per-expert assignment). The planner's down expectation is calibrated at the EP width (I = 2048)
    and is not used for the choice here."""
    tokens_per_expert = T * 8 / 288
    if expected_active is None:
        expected_active = round(E_local * (1 - (1 - 8 / 288) ** T) * 0.9)  # 31 of 72 at T = 256 (measured 2026-10-20)
    if roofline:
        plan = plan_expert_stream(
            grid=grid,
            E_local=E_local,
            K=H,
            I=I,
            H=H,
            T=T,
            w13=w13,
            w2=w2,
            expected_active=expected_active,
            tokens_per_expert=tokens_per_expert,
            ep=False,
            banks=banks,
        )
    else:
        plan = legacy_plan(E_local, I, banks)
        plan.groups = min(12, core_grid_banks(grid, banks) - 1)  # 12 on 13x10; 7 on 8x8 (the down op's 64 cores aside)
        plan.notes = [f"measured 32-users/row form (expert_kernel_sweep.md section 10: 779 us at 58 active) on {grid}"]
    plan.down_readers = (
        8  # the dispatch down op's split (2 output columns per core, 64 cores: the measured 779 us form)
    )
    if units:
        plan.units = True
        ppb = (I // TILE) // banks
        plan.unit_pairs = max(1, min(plan.unit_pairs, ppb))
        plan.notes.append("units form forced by GLM53_EXPERTS_32ROW=units")
    return plan


# per-phase model of one block-op launch pair (the E1 table): calibrated on the 2026-10-20 chip measurements of the
# 8x4 shape (28 active: 341 us traced, gate|up ~235 / down ~106 eager split; 43 active 346 / 155; x in DRAM +30 us)
PHASE_SCAN_ELEMENT_CYCLES = 12.0  # scan_active: tile_elem arithmetic + one volatile 16-bit load + compare per element
PHASE_SCAN_WORD_CYCLES = 2.5  # scan_active_words: one volatile 32-bit load + OR per word (8 per face row)
PHASE_CLOCK_GHZ = 1.35
PHASE_LAUNCH_US = 3.0  # program launch + kernel start + the inter-op gap (0.5-1.5 us of gap in the profiles)
PHASE_X_LOAD_US = 10.0  # 128 x 2 KB tile reads per core from the L1-interleaved copy, 96 cores convoying (H1)
PHASE_X_LOAD_DRAM_US = 40.0  # the same from DRAM (measured: 380 vs 351 us gate|up eager, 2026-10-20)
PHASE_SELECTOR_US = 8.1  # the per-column selector matmul [32, 288] x [288, 72] (hyb profile)
PHASE_X_COPY_US = 2.3  # to_memory_config DRAM -> L1
PHASE_GAP_US = 1.0  # one inter-op gap
PHASE_STREAM_GBS = 376.0  # the two-point fit's marginal rate (tt/roofline.py EXPERT_GBS)
PHASE_TAIL_US = 8.0  # the last pair's compute + act write after the stream ends (gate|up); output pack + write (down)
PHASE_DOWN_ACT_US = 1.5  # one act[e] load (8 x 2 KB + barrier), overlapped with the compute after the first
PHASE_DOWN_FIRST_US = 4.0  # the first expert's block (70 KB at the shared rate) before the down compute starts
PHASE_RESIDUAL_US = 25.0  # measured 351 (341 traced + the two host ops) minus the modelled phases at 28 active: the
# column-ramp / pipeline bubbles the two-point fit calls fixed cost; carried unchanged into every variant (conservative)


def scan_us(E_local: int, M: int, n_active: int, tiles: int, words: bool) -> float:
    """The routing scan of one kernel (both data-movement RISCs run it, concurrently; the reader's is the critical
    one: no weight read is issued before it ends)."""
    if words:
        loads = M * tiles * 16  # 8 words x 2 face rows per token row per tile
        picks_local = 8 * E_local / 288  # a token's picks that land on this chip's experts (~2 of its 8)
        nonzero_rows = M * min(2 * tiles, picks_local) if n_active else 0  # face rows that need the bit tests
        decode = nonzero_rows * 8 * 6  # 8 words x ~6 cycles of bit tests per non-zero face row
        cycles = loads * PHASE_SCAN_WORD_CYCLES + decode
    else:
        cycles = ((E_local - n_active) * M + n_active * M / 4) * PHASE_SCAN_ELEMENT_CYCLES
    return cycles / (PHASE_CLOCK_GHZ * 1e3)


def launch_phase_table(
    T_gathered: int,
    n_active: int,
    *,
    E_local: int = 72,
    K: int = 4096,
    I: int = 256,
    H: int = 4096,
    w13: str = "bfp8",
    w2: str = "bfp8",
    routing_fold: bool = False,
    scan_words: bool = False,
    fused: bool = False,
    x_l1: bool = True,
    banks: int = 8,
) -> dict:
    """Per-phase rows of one block-op launch pair at ``T_gathered`` tokens (M = 32 rows per launch, tile-padded;
    the 8x4 shape by default). Each row: (phase, bytes, floor_us, modelled_us, bound, action). The modelled column
    is the host model (nothing measured here); ``total_us`` sums it; the anchors are the 2026-10-20 chip points."""
    gu, dn = expert_bytes(K, I, H, w13, w2)
    tiles = (E_local + 31) // 32
    M = TILE
    rows = []

    def row(phase, nbytes, floor, model, bound, action):
        rows.append(dict(phase=phase, bytes=nbytes, floor_us=floor, modelled_us=model, bound=bound, action=action))

    scan_now = scan_us(E_local, M, n_active, tiles, scan_words)
    scan_today = scan_us(E_local, M, n_active, tiles, False)
    scan_w = scan_us(E_local, M, n_active, tiles, True)
    # host ops before the launch
    if routing_fold:
        row("selector matmul (host op)", 0, 0.0, 0.0, "folded", "routing_fold: the kernels read the column window")
        row("x copy DRAM -> L1 (host op)", 0, 0.0, 0.0, "folded", "the gather tilizes straight into L1 (compact)")
    else:
        row(
            "selector matmul (host op)",
            32 * 288 * 2 + 288 * 72 * 2,
            0.1,
            PHASE_SELECTOR_US + PHASE_GAP_US,
            "launch/gap",
            "E1 (a): routing_fold",
        )
        row(
            "x copy DRAM -> L1 (host op)",
            32 * K * 2,
            0.7,
            PHASE_X_COPY_US + PHASE_GAP_US,
            "launch/gap",
            "E1 (a): tilize into L1",
        )
    # gate|up launch
    row("gate|up launch", 0, 0.0, PHASE_LAUNCH_US, "launch/gap", "E1 (b): one launch (fused)")
    row("routing read (3 tiles / core)", tiles * 2048, 0.0, 1.0, "latency", "keep")
    row(
        "routing scan (NCRISC + BRISC, concurrent)",
        0,
        0.0,
        scan_now,
        "RISC loop",
        f"E1 (b): scan_words ({scan_today:.0f} -> {scan_w:.0f} us)" if not scan_words else "scan_words",
    )
    row(
        "x load (128 tiles / core, resident)",
        32 * K * 2,
        0.0,
        PHASE_X_LOAD_US if x_l1 else PHASE_X_LOAD_DRAM_US,
        "L1/NoC",
        "keep in L1",
    )
    gu_bytes = n_active * gu
    row(
        "gate|up weight stream",
        gu_bytes,
        floor_us(gu_bytes, DRAM_PEAK_GBS),
        floor_us(gu_bytes, PHASE_STREAM_GBS),
        "DRAM",
        "C12 bfp4 (bytes); the fused form overlaps the down stream",
    )
    row("gate|up compute tail + act write", 2048 * banks * n_active, 0.0, PHASE_TAIL_US, "compute/latency", "keep")
    # down launch
    row("down launch", 0, 0.0, 0.0 if fused else PHASE_LAUNCH_US, "launch/gap", "E1 (b): fused")
    row(
        "down routing read + scan",
        tiles * 2048,
        0.0,
        0.0 if fused else 1.0 + scan_now,
        "RISC loop",
        "E1 (b): overlapped in the fused program",
    )
    row(
        "act loads (first; the rest overlap)",
        n_active * (I // TILE) * 2048,
        0.0,
        0.0 if fused else PHASE_DOWN_ACT_US,
        "latency",
        "keep",
    )
    dn_bytes = n_active * dn
    dn_stream = floor_us(dn_bytes, PHASE_STREAM_GBS)
    row(
        "down weight stream",
        dn_bytes,
        floor_us(dn_bytes, DRAM_PEAK_GBS),
        dn_stream * (0.25 if fused else 1.0),
        "DRAM",
        "fused: ~3/4 hidden under the gate|up stream (same DRAM; the exposed part is the last experts')",
    )
    row("output write (8 tiles / core)", (H // TILE) * 2048, 0.0, PHASE_TAIL_US, "latency", "keep")
    row(
        "column ramp / pipeline bubbles (measured residual)",
        0,
        0.0,
        PHASE_RESIDUAL_US,
        "launch/gap",
        "the first column of every core at the shared DRAM rate (~35 us of compute idle) is not additive; the residual is what the fit calls fixed",
    )
    total = sum(r["modelled_us"] for r in rows)
    floor = floor_us(gu_bytes + dn_bytes, DRAM_PEAK_GBS)
    return dict(T=T_gathered, active=n_active, rows=rows, total_us=total, floor_us=floor, bytes=gu_bytes + dn_bytes)


def e1_expected_us(n_active: int = 28, **kw) -> dict:
    """The E1 ladder at one activity: today / fold / fold + words / fold + words + fused (modelled us per layer)."""
    out = {}
    out["today"] = launch_phase_table(32, n_active, **kw)["total_us"]
    out["fold"] = launch_phase_table(32, n_active, routing_fold=True, **kw)["total_us"]
    out["fold+words"] = launch_phase_table(32, n_active, routing_fold=True, scan_words=True, **kw)["total_us"]
    out["fold+words+fused"] = launch_phase_table(32, n_active, routing_fold=True, scan_words=True, fused=True, **kw)[
        "total_us"
    ]
    return out


# ---------------------------------------------------------------------------------------------------- EP32 skew
def active_prob(p: float, tokens: int) -> float:
    """P(an expert with per-token pick probability p is active among ``tokens`` tokens)."""
    return 1.0 - (1.0 - min(1.0, max(0.0, p))) ** tokens


def popularity_from_routing(idx: Sequence[Sequence[int]], E: int) -> list[float]:
    """Top-k id lists (one per token) -> p_e = fraction of tokens picking expert e (sum over e = k)."""
    counts = [0] * E
    for row in idx:
        for e in row:
            counts[int(e)] += 1
    n = max(1, len(idx))
    return [c / n for c in counts]


def chip_bytes(experts: Sequence[int], popularity: Sequence[float], tokens: int, gu: int, dn: int) -> float:
    """Expected active-expert bytes of a chip holding ``experts`` when its column sees ``tokens`` tokens per step."""
    return sum(active_prob(popularity[e], tokens) * (gu + dn) for e in experts)


def plan_ep_placement(
    popularity: Sequence[float],
    D: int,
    tokens: int,
    gu: int,
    dn: int,
    replicate: int = 0,
    slots_per_chip: int | None = None,
) -> list[list[int]]:
    """Expert -> chip placement balancing the EXPECTED active bytes per chip (LPT: experts by expected bytes
    descending, each to the least loaded chip with a free slot). ``replicate`` hottest experts get a second copy
    on another chip (the router then alternates their tokens between the two copies by token parity:
    ``RouterTopk`` PERM table with two columns); every chip holds ``slots_per_chip`` slots (default
    ceil((E + replicate) / D), unused slots are left empty = never streamed)."""
    E = len(popularity)
    if slots_per_chip is None:
        slots_per_chip = -(-(E + replicate) // D)
    assert slots_per_chip * D >= E + replicate, (slots_per_chip, D, E, replicate)
    weight = [active_prob(p, tokens) * (gu + dn) for p in popularity]
    order = sorted(range(E), key=lambda e: (-weight[e], e))
    items = [(e, weight[e]) for e in order]
    for e in order[:replicate]:  # a replica halves the tokens of both copies: same expected bytes, half the rows
        items.append((e, weight[e]))
    items.sort(key=lambda it: (-it[1], it[0]))
    load = [0.0] * D
    chips: list[list[int]] = [[] for _ in range(D)]
    for e, w in items:
        cand = [d for d in range(D) if len(chips[d]) < slots_per_chip and e not in chips[d]]
        d = min(cand, key=lambda d: (load[d], d))
        chips[d].append(e)
        load[d] += w
    return chips


def contiguous_placement(E: int, D: int) -> list[list[int]]:
    E_local = E // D
    return [list(range(d * E_local, (d + 1) * E_local)) for d in range(D)]


def placement_perm(chips: Sequence[Sequence[int]], E: int, slots_per_chip: int) -> list[list[int]]:
    """Global expert id -> its permuted slot id(s) ``d * slots + i`` (one per copy, the primary first). The
    kernels and the dispatch mapping work in the permuted space, where chip d's experts are the contiguous
    ``[d * slots, (d + 1) * slots)`` as today; the router applies the table when it writes the ids."""
    perm: list[list[int]] = [[] for _ in range(E)]
    for d, lst in enumerate(chips):
        assert len(lst) <= slots_per_chip, (d, len(lst), slots_per_chip)
        for i, e in enumerate(lst):
            perm[e].append(d * slots_per_chip + i)
    for e in range(E):
        assert perm[e], f"expert {e} is on no chip"
    return perm


def placement_hash(chips: Sequence[Sequence[int]]) -> str:
    return hashlib.sha1(json.dumps([list(map(int, c)) for c in chips]).encode()).hexdigest()[:8]


def load_placement(path: str | Path) -> list[list[int]]:
    data = json.loads(Path(path).read_text())
    chips = data["chips"] if isinstance(data, dict) else data
    return [[int(e) for e in c] for c in chips]


def save_placement(path: str | Path, chips: Sequence[Sequence[int]], meta: dict | None = None) -> None:
    Path(path).write_text(json.dumps({"chips": [list(map(int, c)) for c in chips], **(meta or {})}, indent=1))


def column_skew(
    chips: Sequence[Sequence[int]],
    popularity: Sequence[float],
    rows: int,
    cols: int,
    users_per_row: int,
    gu: int,
    dn: int,
    gbs: float = STREAM_PLATEAU_GBS,
) -> dict:
    """Per-column max / mean expected bytes and the step's expected expert time (the column waits for its slowest
    chip; the 4 columns run the same step so the layer waits for the slowest column)."""
    tokens = rows * users_per_row
    per_chip = [chip_bytes(chips[r * cols + c], popularity, tokens, gu, dn) for r in range(rows) for c in range(cols)]
    col_max = [max(per_chip[r * cols + c] for r in range(rows)) for c in range(cols)]
    col_mean = [sum(per_chip[r * cols + c] for r in range(rows)) / rows for c in range(cols)]
    return dict(
        tokens=tokens,
        per_chip_mb=[b / 1e6 for b in per_chip],
        col_max_mb=[b / 1e6 for b in col_max],
        col_mean_mb=[b / 1e6 for b in col_mean],
        step_max_mb=max(col_max) / 1e6,
        balanced_mb=sum(per_chip) / len(per_chip) / 1e6,
        expected_us=floor_us(max(col_max), gbs),
        balanced_us=floor_us(sum(per_chip) / len(per_chip), gbs),
    )


def simulate_skew(
    chips: Sequence[Sequence[int]],
    idx_rows: Sequence[Sequence[int]],
    rows: int,
    cols: int,
    users_per_row: int,
    gu: int,
    dn: int,
    perm: Sequence[Sequence[int]] | None = None,
) -> dict:
    """Realized (not expected) per-chip active bytes for one step whose tokens are ``idx_rows`` (the first
    rows x users_per_row lists; with replicas, token t takes copy t % n_copies): the column max over the step."""
    tokens = rows * users_per_row
    E = max(max(r) for r in idx_rows) + 1
    owner: dict[int, list[int]] = {}
    for d, lst in enumerate(chips):
        for e in lst:
            owner.setdefault(e, []).append(d)
    active = [set() for _ in chips]
    for t, row in enumerate(idx_rows[:tokens]):
        for e in row:
            copies = owner[e]
            d = copies[t % len(copies)]
            active[d].add(e)
    per_chip = [len(a) * (gu + dn) for a in active]
    col_max = [max(per_chip[r * cols + c] for r in range(rows)) for c in range(cols)]
    return dict(
        per_chip_active=[len(a) for a in active],
        per_chip_mb=[b / 1e6 for b in per_chip],
        col_max_mb=[b / 1e6 for b in col_max],
        step_max_mb=max(col_max) / 1e6,
        mean_mb=sum(per_chip) / len(per_chip) / 1e6,
        E=E,
    )
