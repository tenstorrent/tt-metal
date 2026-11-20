# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""One chip's routed experts for 8 token rows as one program, the picked experts streamed from DRAM (BFP4).

Core roles (kernels/):

    hub (1 core)        x ([8, 6144] as [8, 32] BF16 row tiles) and the routing result ([U, ids | w]) to every core
                        by multicast, then one multicast semaphore increment releases them. Holds the output buffer.
    gate|up (G / bank)  stream their bank's K range ("piece") of a picked expert's pair of gate / up tile columns,
                        custom_mm against x (LoFi, FP32 accumulation), send the two FP32 partial tiles to the unit's
                        finisher. A bank's cores take the bank's pieces of the union round-robin.
    finisher (NF)       sum a unit's S partials (FP32), silu(min(gate, 10)) * clamp(up, +-10) * w[row] -> one BF16
                        [8, 32] activation tile; a slot's P tiles are multicast together to every core; slot i goes
                        to finisher i % NF.
    down (D / bank)     own 24 / D output tile columns; per union slot (in order) stream their columns of the expert's
                        down weights and accumulate act . W into DST, which holds the columns for the whole union;
                        the tiles go to the hub's output buffer at the end.

Every reader core streams only from its own DRAM bank and every bank holds the same bytes of every expert
(layout.py), so the banks are loaded equally for any union. DRAM reads use NOC0, everything else NOC1.

For whoever composes this program (fused/layer_mlp): build it with ``mode="rect", down_per_bank=4`` (place_rect:
every role a few rectangles, launch floor 12.4 us against 53.8 us with the default placement, log mk5_board_10; the
defaults are kept for the older tests and the expert-parallel share). With it a chip has 48 gate|up and 32 down
cores, and the output buffer is complete when hand-over word SEM_OUT of the hub equals ``len(op.all_d)`` (32), not
48. x comes in already normalised. The descriptor is for one chip: a 32-chip program needs one RoutedExperts
placement per chip (the banks' endpoint rows can differ between chips).

Measured on one chip (traced, 20 replays, one synchronize, rectangular placement, share of 1/32): with the routing
given by the host 44.0 / 59.7 / 83.8 / 136.9 us for unions of 8 / 16 / 32 / 64 experts (log mk5_board_10); with
the router on the chip (router=True, router_candidates=True) and the shared expert as a two-term plain slot 98.0 /
116.6 / 148.4 / 202.4 us (log mk5_latency_3), of which the hub releases the expert cores 63 to 68 us after it
starts (scores back after 38 to 43 us, then the top 8 of 8 rows in software; log mk5_latency_4).

Interface to the cross-chip reduce. The chip's partial output is [8, 6144] BF16 in the output buffer, one core's
L1 (here the hub's; the down cores take its core and address as arguments): 192 row tiles of [8, 32], tile n = hidden
columns [32 n, 32 n + 32) as 8 rows x 16 values of columns 0..15, then 8 x 16 of columns 16..31; 512 B a tile,
98304 B in all (8 fabric packets at the mesh's 12352 B payload). Each down core writes its 4 (or 8) adjacent tiles once,
at the end, then increments hand-over word 4 (SEM_OUT) of that core's sync buffer: the output is complete when it
equals the number of down cores (48, or 24, 32). The sum over the chips of a split (tensor parallel: all 32; expert
parallel: all 32, zeros from the chips without a picked expert) is the MoE's routed output, plus the shared expert
when it runs as a plain slot.

What is known about the arithmetic (for fused/fidelity, which owns what custom_mm does to its operands). Every
expert matmul here is custom_mm, LoFi, FP32 accumulation, DefaultSplitAcc (the down matmul DenseSplitAcc with 6 or 8
columns a core), x and the activation as [8, 32] BF16 tiles (in0, SrcB), BFP4 weights (in1, SrcA).
tests/test_routed.py test_stage_accuracy reads single stages out through identity down weights (logs mk5_board_6 and
mk5_board_7, dumps of x / weights / output in generated/hy4_preview/logs/mk5_dump/stage_*.pt):
* the gate and the up sums are 3.1 to 3.3 % rms off exact arithmetic on the same BFP4 weights (gain 0.990), uniformly
  over rows and columns; silu adds nothing, the activation is 4.8 % off, the whole program 5.7 % (PCC 0.9985 against
  exact arithmetic on the BFP4 weights, gain 0.982; 0.981 to 0.982 against the exact weights, random weights);
* it does not come from x's bits: the same 3.0 % with x rounded to 4 significant bits;
* it goes away with weights of one significant bit (+-2^-7): 0.8 %, and 0.6 % with x of 4 bits as well;
* it is there with one product per MVMUL row (x non-zero in every 16th column: 3.4 %) and with a single K tile
  (3.3 %), so it is not the accumulation over K nor the 16-term sum;
* a regression of the error on the weights' mantissa bits or 3-bit codes times x explains none of it, and no K tile
  is missing or doubled.
With the checkpoint's weights the picture is different (tests/test_real.py, layers 1 to 5, the 32 shares of 1/32
run one after the other on one chip and summed, logs mk5_real_3 and mk5_real_4): against exact arithmetic on the
stored weights the sum has PCC 0.99998 (0.6 % rms) and a uniform gain of 0.9862 to 0.9867, for the routed experts
(BFP4), the shared expert (two BFP4 terms) and the whole chip program alike. Against the reference the routed
experts have PCC 0.9887 / 0.9877 / 0.9850 / 0.9825 / 0.9821, which is what the BFP4 format alone gives, and gain
0.973 to 0.976 (the format's 0.986 to 0.990 times the device's 0.9865). So the 3 % above is a property of the
random-weight probes (iid normal weights of std 0.006, one share on its own) that the checkpoint's weights do not
show: errors of 5.7 % a share, independent between the shares, would leave the sum of 32 between 5.7 % (outputs
that add like noise) and 1.0 % (outputs that add coherently) off, not 0.6 %. What in the random weights makes the
difference is not known, and a single share with checkpoint weights has not been compared on its own: test_real.py
dumps the first share's outputs (first_routed, first_shared, first_chip) next to the sums for that.
The FP32 partial sums are unpacked as Tf32 (e8m10) by the finisher, not as BF16 (jit_build/genfiles.cpp:1074).
bias_correction / TRUNCATION_CORRECTION exist for the truncations of a LoFi path but were written before these
probes and are off; nothing here has been measured with them on.

What fused/fidelity found (its GUIDE.md; 2026-11-19): LoFi cuts the row operand to 7 significant bits (BFP4 weights
pass whole), and custom_mm's merge of the split accumulators cuts the second one to BF16; the packer rounds FP32 to
BF16 (it does not cut), and the 3 % of the random-weight probes above is the reference, not the matmul:
layout.to_device_stream creates a sharded tensor, which ttnn rounds to BF16 before BFP4, so the device held other
weights than bfp4_round (host_round=True avoids it; the prepared checkpoint streams are packed on the host and were
never affected). With ``fidelity="p02"`` the expert matmuls run through hifi_mm at the multiplier's phases 0 and 2
with an FP32 merge: against exact arithmetic on the stored weights the summed shares go from 1.5 % rms and gain
0.9864 to 0.25 to 0.29 % and gain 0.9992 (layers 1 to 5: routed, shared and chip program alike), against the
reference the routed gain from 0.973-0.976 to 0.985-0.989 with the PCC and the picks unchanged, at the same traced
latency (99.4 / 115.8 / 154.9 / 205.5 us for unions of 8 / 16 / 32 / 64 against 100.2 / 116.6 / 151.2 / 207.9 in
the same session; logs mk9_moe_real2, mk9_host_real_exact_new, mk9_moe_latency2, mk9_moe_latency_custom). The default
stays custom_mm.

Buffers at the same L1 address on every core (CBs declared on the whole grid, allocated first): x, the meta page,
the activation buffer (by finisher, then its slots in order, P tiles each), the finishers' partial buffer. Cross-core
hand-overs are counters in the meta page, zeroed by the hub's multicast and incremented by NOC atomics, and a few
host-zeroed words per core for the releases (no program semaphores: they cost launch time); every wait on another
core is bounded
(kernels/moe_common.hpp HY4_SPIN_LIMIT) and reported in the status words, so a broken hand-over ends the program with
a wrong output instead of hanging the chip.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field
from pathlib import Path

import torch

import ttnn

from .layout import (
    BANKS,
    BFP4_TILE_BYTES,
    COLS_PER_BANK,
    K_TILES,
    MAX_SLOTS,
    MAX_UNITS,
    META_IN_WORDS,
    ROWS,
    TILE,
    Share,
)

KERNEL_DIR = Path(__file__).parent / "kernels"
MOCK = bool(os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH"))  # the mock Galaxy compiles and checks, runs nothing
# State on 2026-11-19: the program runs on the board, one chip at a time (tensor-parallel shares of 1/32 and 1/8,
# whole experts, with and without the router; checkpoint layers 1 to 5, logs mk5_real_3 and mk5_real_4). Its first
# three runs (log mk5_board_1) hung chip 0 until the 120 s operation timeout: the hub's NOC1 multicast gave the
# rectangle's corners in ascending order, which stalls the sender on Blackhole (kernels/moe_dm.hpp grid_mcast_addr has
# the rule and its sources). Every NOC wait of the data-movement kernels is bounded since. A variant that has not run
# on the board (another placement, new defines, more than one chip) is new kernel code there, so running on the
# device stays an explicit decision:
BOARD_GATE = (
    "fused/moe runs on the device only with HY4_MOE_BOARD=1 (see program.py: a hand-over that never arrives used to "
    "hang the chip, log mk5_board_1). Run a variant that is new on the board alone, smallest case first "
    "(tests/test_routed.py -k 'tp32 and U1'), with a short TT_METAL_OPERATION_TIMEOUT_SECONDS, and check the status "
    "words."
)
ROW_TILE = 2 * TILE * ROWS  # bytes of a [8, 32] BF16 tile
ROW_TILE_F32 = 2 * ROW_TILE
META_PAGE = 2 * META_IN_WORDS + 4 * MAX_SLOTS + 4 * MAX_UNITS + 64  # kernels/moe_common.hpp META_PAGE_BYTES
MAX_PACKET = 16384
GATE_UP_KB = 48  # K tiles per gate | up block: 96 tiles = 55296 B = 4 packets of 13824 B
IDLE = 0xFFFF  # CFG_INDEX of a core that runs a role's kernels with no work (kernels/moe_common.hpp)

CB_X, CB_META, CB_ACT, CB_PART, CB_CTL, CB_W, CB_SC, CB_OUT = 0, 1, 2, 3, 4, 5, 6, 16
# RoutedExperts(fidelity=...): "custom_mm" is the tree's LoFi matmul as the kernels were written; the others run the
# expert matmuls through hifi_mm (fused/fidelity/kernels/hifi_mm.h, GUIDE.md) with (passes, phase step) and add the
# split accumulators in FP32. "p02" (the multiplier's phases 0 and 2) takes a BFP4 weight and a BF16 row operand
# whole: it is the setting for this program, and it costs about as much as LoFi (the second pass hides under the
# unpack of the weights).
FIDELITY = {"lofi": (1, 1), "hifi2": (2, 1), "hifi3": (3, 1), "hifi4": (4, 1), "p02": (2, 2)}

# A float with uniformly distributed lower bits loses on average 2^-n * 0.7213 (= E[1 / mantissa]) of its value when it
# is cut to n significant bits. The LoFi path cuts three times toward zero: x's 8th bit in the gate | up matmul
# (2^-8), the FP32 partial sums when they are unpacked as BF16 (2^-8), and the activation when it is packed as BF16
# and its 8th bit is dropped by the down matmul (2 * 2^-8). The corrections are the reciprocals of the mean loss.
_CUT = 2.0**-8 * 0.7213
TRUNCATION_CORRECTION = (1.0 / (1.0 - 2 * _CUT), 1.0 / (1.0 - 2 * _CUT))


def f32_bits(v):
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def packets(block_bytes):
    """Fewest equal NOC packets (multiples of 64 B, at most 16 KB) a block splits into."""
    for n in range(-(-block_bytes // MAX_PACKET), block_bytes + 1):
        if block_bytes % n == 0 and (block_bytes // n) % 64 == 0:
            return n
    raise ValueError(block_bytes)


def bank_anchors(device):
    """The logical worker core next to each DRAM bank's NOC0 endpoint."""
    try:
        a = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    except TypeError:
        a = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0, ttnn.MeshCoordinate(0, 0))
    return [a[b] for b in sorted(a)] if isinstance(a, dict) else list(a)


@dataclass
class Placement:
    """Logical cores of a chip by role. readers[bank] = that bank's streaming cores, nearest its anchor first."""

    grid: tuple
    readers: dict
    spare: list
    noc: dict = field(default_factory=dict)
    # mode "rect": idle cores that only make a role's core set a rectangle (they run the role's kernels with no
    # work), and where the hub, the finishers and the router cores go so that each is one rectangle too
    idle_gate_up: list = field(default_factory=list)
    idle_down: list = field(default_factory=list)
    hub: object = None
    finisher_cells: list = field(default_factory=list)
    router_cells: list = field(default_factory=list)

    def describe(self):
        return {b: [(c.x, c.y) for c in cs] for b, cs in self.readers.items()}


def place(device, per_bank, banks=None, mode="side", split=None):
    """per_bank reader cores for each DRAM bank in ``banks`` (default all).

    NOC0 routes a DRAM read's data east along the row of the bank's endpoint, then south down the reader's column.
    mode "side": a bank takes cores on its own side of the chip (the columns from its anchor to the next DRAM
    column), rows at and below its anchor row first (the short way round for NOC0), nearest first. mode "row": the
    anchor's row first across the whole chip, then the rows below. mode "any": nearest by wrap-around NOC0 distance.
    mode "block" (measurement only): cores in grid order regardless of the banks, the first `split` of every bank
    in one block of rows and the rest in the next.
    Banks pick in turn, one core each, so no bank is left with only far cores.
    """
    grid = device.compute_with_storage_grid_size()
    anchors = bank_anchors(device)
    banks = list(range(len(anchors))) if banks is None else list(banks)
    noc = {
        (x, y): device.worker_core_from_logical_core(ttnn.CoreCoord(x, y)) for x in range(grid.x) for y in range(grid.y)
    }
    # The multicast rectangle is the NOC box of the logical grid: it must hold no Tensix core outside the grid
    # (columns 8 and 9 of Blackhole are the router-only and the DRAM column).
    xs = [noc[(x, 0)].x for x in range(grid.x)]
    ys = [noc[(0, y)].y for y in range(grid.y)]
    assert xs == [x for x in range(xs[0], xs[-1] + 1) if x not in (8, 9)], f"worker columns are not a NOC box: {xs}"
    assert ys == list(range(ys[0], ys[0] + grid.y)), f"worker rows are not a NOC box: {ys}"
    assert all(noc[(x, y)].x == xs[x] and noc[(x, y)].y == ys[y] for x in range(grid.x) for y in range(grid.y))
    anchor_cols = sorted({anchors[b].x for b in range(len(anchors))})

    def segment(b):
        ax = anchors[b].x
        later = [c for c in anchor_cols if c > ax]
        return range(ax, later[0] if later else grid.x)

    def cost(b, x, y):
        ax, ay = anchors[b].x, anchors[b].y
        dy = (y - ay) % grid.y  # rows below the anchor, wrapping
        dx = (x - ax) % grid.x
        if mode == "side":
            return (0 if x in segment(b) else 1, dy, dx)
        if mode == "row":
            return (dy, dx)
        return (dx + dy, dy)

    order = {
        b: sorted(((x, y) for x in range(grid.x) for y in range(grid.y)), key=lambda c: cost(b, *c)) for b in banks
    }
    used, readers = set(), {b: [] for b in banks}
    if mode == "block":
        # measurement only: the cores in grid order, whatever their bank's endpoint: each role is whole rows
        cells = [(x, y) for y in range(grid.y) for x in range(grid.x)]
        first = split if split is not None else per_bank // 2
        n1, n2 = first * len(banks), (per_bank - first) * len(banks)
        for i, b in enumerate(banks):
            mine = (
                cells[i * first : (i + 1) * first]
                + cells[n1 + i * (per_bank - first) : n1 + (i + 1) * (per_bank - first)]
            )
            readers[b] = [ttnn.CoreCoord(*c) for c in mine]
            used.update(mine)
    for _ in range(per_bank if mode != "block" else 0):
        for b in banks:
            c = next(c for c in order[b] if c not in used)
            used.add(c)
            readers[b].append(ttnn.CoreCoord(*c))
    spare = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x) if (x, y) not in used]
    return Placement((grid.x, grid.y), readers, spare, noc)


def place_rect(device, gate_up_cols=3, down_cols=2, rows_per_bank=2):
    """Bank-local AND rectangular: launching a program costs time per core range of every kernel group (53.8 us for
    this program with the readers nearest their banks in about 29 ranges, 9.8 us with every role as whole rows: log
    mk5_board_9), so every role is a few rectangles here.

    A side of the chip is the columns from a DRAM column's first worker column to the next DRAM column. On each
    side the gate|up cores are the first gate_up_cols columns and the down cores the next down_cols columns, over ALL
    rows: two rectangles per role. A bank owns rows_per_bank rows of its side: its endpoint's row and the next free
    rows below it (NOC0 carries a read east along the endpoint's row, then south; the last bank of a side wraps to
    the first free rows). Rows of a side that no bank owns hold idle cores: they run their role's kernels with an
    empty work list, which costs nothing but keeps the role one rectangle. The hub, the finishers and the router
    cores are columns of the first side's unused columns."""
    grid = device.compute_with_storage_grid_size()
    anchors = bank_anchors(device)
    noc = {
        (x, y): device.worker_core_from_logical_core(ttnn.CoreCoord(x, y)) for x in range(grid.x) for y in range(grid.y)
    }
    xs = [noc[(x, 0)].x for x in range(grid.x)]
    ys = [noc[(0, y)].y for y in range(grid.y)]
    assert xs == [x for x in range(xs[0], xs[-1] + 1) if x not in (8, 9)], f"worker columns are not a NOC box: {xs}"
    assert ys == list(range(ys[0], ys[0] + grid.y)), f"worker rows are not a NOC box: {ys}"
    side_cols = sorted({a.x for a in anchors})
    readers, idle_g, idle_d, free_cols = {}, [], [], {}
    for i, x0 in enumerate(side_cols):
        x1 = side_cols[i + 1] if i + 1 < len(side_cols) else grid.x
        assert x1 - x0 >= gate_up_cols + down_cols, f"side at column {x0} has {x1 - x0} columns"
        g_cols = list(range(x0, x0 + gate_up_cols))
        d_cols = list(range(x0 + gate_up_cols, x0 + gate_up_cols + down_cols))
        free_cols[x0] = list(range(x0 + gate_up_cols + down_cols, x1))
        mine = sorted((b for b in range(len(anchors)) if anchors[b].x == x0), key=lambda b: anchors[b].y)
        assert len(mine) * rows_per_bank <= grid.y
        taken = {anchors[b].y for b in mine}
        rows = {b: [anchors[b].y] for b in mine}
        for b in mine:  # the next free rows below the endpoint's row, wrapping
            y = anchors[b].y
            while len(rows[b]) < rows_per_bank:
                y = (y + 1) % grid.y
                if y not in taken:
                    taken.add(y)
                    rows[b].append(y)
        for b in mine:
            g = [ttnn.CoreCoord(x, y) for y in rows[b] for x in g_cols]
            d = [ttnn.CoreCoord(x, y) for y in rows[b] for x in d_cols]
            readers[b] = g + d
        idle_rows = [y for y in range(grid.y) if y not in taken]
        idle_g += [ttnn.CoreCoord(x, y) for y in idle_rows for x in g_cols]
        idle_d += [ttnn.CoreCoord(x, y) for y in idle_rows for x in d_cols]
    free = free_cols[side_cols[0]]
    assert len(free) >= 2, "the hub, the finishers and the router cores need two free columns on the first side"
    hub = ttnn.CoreCoord(free[0], grid.y - 1)
    finisher_cells = [ttnn.CoreCoord(free[0], y) for y in range(grid.y - 1)]
    router_cells = [ttnn.CoreCoord(free[1], y) for y in range(grid.y)]
    used = {(c.x, c.y) for cs in readers.values() for c in cs} | {(c.x, c.y) for c in idle_g + idle_d}
    spare = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x) if (x, y) not in used]
    return Placement((grid.x, grid.y), readers, spare, noc, idle_g, idle_d, hub, finisher_cells, router_cells)


def _crs(cores):
    out = ttnn.CoreRangeSet([ttnn.CoreRange(cores[0], cores[0])])
    for c in cores[1:]:
        out = out.merge(ttnn.CoreRangeSet([ttnn.CoreRange(c, c)]))
    return out


def hub_tensor(device, hub, data, dtype):
    """A row-major tensor in the hub core's L1 (one shard)."""
    shard = ttnn.ShardSpec(_crs([hub]), list(data.shape), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.from_torch(
        data,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard),
    )


class RoutedExperts:
    """The program of one chip. ``share``: what the chip holds of an expert (layout.Share); ``max_slots``: the
    largest union it is built for (buffer sizes)."""

    BIAS_ARG = 6  # bias_rt: the hub's runtime arguments 6 .. 6 + E - 1 are the bias

    def __init__(
        self,
        device,
        share: Share,
        max_slots=None,
        gate_up_per_bank=6,
        down_per_bank=6,
        finishers=6,
        banks=None,
        mode="side",
        gate_up_ring=(3, 1),
        down_ring=None,
        spin_limit=None,
        defines=(),
        router=False,
        num_experts=256,
        expert_base=0,
        expert_count=None,
        n_plain=0,
        plain_id0=None,
        plain_terms=1,
        bias_correction=(1.0, 1.0),
        router_candidates=False,
        dummy_semaphores=0,
        fidelity="custom_mm",
        stream_rt=False,
        bias_rt=False,
        hub_tensors=None,
    ):
        """router: the hub runs the router on the chip (8 router cores, one per 32 experts; set_router gives it the
        weights) instead of taking the routing result from the host. expert_base / expert_count: the chip's own
        experts among the router's num_experts (expert parallel; default all), numbered from expert_base in the
        weight streams. n_plain / plain_id0: always-on plain slots the router appends to the union (the shared
        expert: stream entry plain_id0). plain_terms = 2: a plain slot's weights are stored as two consecutive
        stream entries, two BFP4 terms whose sum the matmuls accumulate (layout.two_terms): the accuracy of BFP8 at
        LoFi, for the shared expert and the dense FFN. bias_correction = (k_in, k_act): factors the finisher applies
        to the gate / up sums and to the activation against the truncations of the LoFi path (TRUNCATION_CORRECTION;
        1.0 = off).
        fidelity: "custom_mm" (default: the kernels as they were, bit for bit) or a key of FIDELITY: the gate|up and
        down matmuls through hifi_mm at that pass schedule with the FP32 merge of the split accumulators
        (kernels/gate_up_compute_hifi.cpp, down_compute_hifi.cpp); "p02" is the one to use. Nothing else changes:
        the same cores, buffers, streams, hand-overs, finishers and output layout."""
        assert fidelity == "custom_mm" or fidelity in FIDELITY, fidelity
        self.fidelity = fidelity
        # One program for many layers (fused/model). stream_rt: every stream reader takes its stream's address as a
        # common runtime argument of the launch (the gate|up, down and router streams given to descriptor() / held
        # in w_router) instead of from its sync words. bias_rt: the hub takes the router's bias as runtime arguments
        # (bias_values, set by set_router or by the caller) and writes its bias buffer itself.
        # hub_tensors: {"x", "out", "meta", "status", "scores", "bias", "picks", "cand": tensor or an object with
        # buffer_address()} the caller owns (another instance's, or ranges of its own L1); the rest are made here.
        self.stream_rt, self.bias_rt = stream_rt, bias_rt
        self.bias_values = None
        self._given = dict(hub_tensors or {})
        self.bias_correction = bias_correction
        # the router cores cut their 32 experts to each row's 8 best; the hub merges 8 lists instead of scanning 256
        self.router_candidates = router_candidates
        self.dummy_semaphores = dummy_semaphores  # measurement only: program semaphores nobody uses
        self.device, self.share = device, share
        self.router = router
        self.num_experts = num_experts
        self.expert_base = expert_base
        self.expert_count = num_experts if expert_count is None else expert_count
        self.n_plain = n_plain
        self.plain_id0 = self.expert_count if plain_id0 is None else plain_id0
        self.router_cols = num_experts // TILE
        P, S = share.P, share.S
        if max_slots is None:  # 8 rows x 8 picks and the plain slots, as far as the buffers go
            max_slots = min(64 + n_plain, MAX_UNITS // P)
        self.max_slots = max_slots
        self.max_units = max_slots * P
        assert max_slots <= MAX_SLOTS and self.max_units <= MAX_UNITS, (max_slots, P)
        assert COLS_PER_BANK % down_per_bank == 0
        self.cols = COLS_PER_BANK // down_per_bank
        assert self.cols <= 8
        self.G, self.D, self.NF = gate_up_per_bank, down_per_bank, finishers
        self.banks = list(range(BANKS)) if banks is None else list(banks)
        n_router = self.router_cols if router else 0
        if mode == "rect":
            assert banks is None and self.G % 2 == 0 and self.D % 2 == 0, "mode rect: all banks, two rows a bank"
            self.placement = pl = place_rect(device, self.G // 2, self.D // 2)
            assert finishers <= len(pl.finisher_cells) and n_router <= len(pl.router_cells)
            self.hub = pl.hub
            self.fin = pl.finisher_cells[:finishers]
            self.r_cores = pl.router_cells[:n_router]
        else:
            self.placement = pl = place(device, self.G + self.D, self.banks, mode, split=self.G)
            assert len(pl.spare) >= 1 + finishers + n_router, f"{len(pl.spare)} spare cores: hub, finishers, router"
            self.hub = pl.spare[0]
            self.fin = pl.spare[1 : 1 + finishers]
            self.r_cores = pl.spare[1 + finishers : 1 + finishers + n_router]
        self.g_cores = {b: pl.readers[b][: self.G] for b in self.banks}
        self.d_cores = {b: pl.readers[b][self.G :] for b in self.banks}
        self.all_g = [c for b in self.banks for c in self.g_cores[b]]
        self.all_d = [c for b in self.banks for c in self.d_cores[b]]
        # the cores that run a role's kernels: the working ones and the idle ones that complete the rectangles
        self.g_kernel_cores = self.all_g + list(pl.idle_gate_up)
        self.d_kernel_cores = self.all_d + list(pl.idle_down)
        self.grid_crs = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(pl.grid[0] - 1, pl.grid[1] - 1))]
        )
        self.gate_up_ring = gate_up_ring
        # down: one block per expert (cols * P tiles); small blocks need a deep ring to keep DRAM requests in flight
        self.down_block = self.cols * P * BFP4_TILE_BYTES
        if down_ring is None:
            n = max(3, min(12, -(-65536 // self.down_block)))
            down_ring = (n, n - 2)
        self.down_ring = down_ring
        self.defines = [("ROWS", str(ROWS))] + ([("HY4_SPIN_LIMIT", f"{spin_limit}u")] if spin_limit else [])
        self.defines += list(defines)
        if self.cols > 4:
            self.defines.append(("DOWN_DENSE", "1"))
        assert plain_terms in (1, 2)
        self.plain_terms = plain_terms
        if plain_terms > 1:
            self.defines.append(("PLAIN_TERMS", str(plain_terms)))

        # the hand-over words: 16 uint32 on every core of the grid, host-zeroed, the same L1 address everywhere
        n_cores = pl.grid[0] * pl.grid[1]
        self.sync = ttnn.from_torch(
            torch.zeros(n_cores, 16, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(self.grid_crs, [1, 16], ttnn.ShardOrientation.ROW_MAJOR),
            ),
        )
        self.defines.append(("HY4_SYNC", f"{self.sync.buffer_address()}u"))
        self._core_words_key = None
        self.w_router = None

        # the hub's buffers
        def own(name, data, dtype):
            return self._given[name] if name in self._given else hub_tensor(device, self.hub, data, dtype)

        self.x = own("x", torch.zeros(K_TILES, 2 * 16 * ROWS), ttnn.bfloat16)
        self.out = own("out", torch.zeros(K_TILES, 2 * 16 * ROWS), ttnn.bfloat16)
        self.meta = own("meta", torch.zeros(1, META_IN_WORDS, dtype=torch.int32), ttnn.uint16)
        self.status = own("status", torch.zeros(1, 16 + 2 * 160, dtype=torch.int32), ttnn.uint32)
        if router:
            self.scores = own("scores", torch.zeros(self.router_cols, 32 * ROWS), ttnn.float32)
            self.bias = own("bias", torch.zeros(1, num_experts, dtype=torch.int32), ttnn.int32)
            self.picks = own("picks", torch.zeros(ROWS, 16, dtype=torch.int32), ttnn.uint16)
            self.cand = own("cand", torch.zeros(self.router_cols, ROWS * 8 * 3, dtype=torch.int32), ttnn.int32)

    # ------------------------------------------------------------------------------------------------------------
    def _noc(self, c):
        return self.placement.noc[(c.x, c.y)]

    def _vcs(self, cores_with_bank):
        """A NOC0 virtual channel per reader, distinct from earlier readers on the same row where possible."""
        vcs = []
        for i, (core, bank, r) in enumerate(cores_with_bank):
            vc = (bank + r) & 0x3
            for j in range(i):
                if cores_with_bank[j][0].y == core.y and vcs[j] == vc:
                    vc = (vc + 1) & 0x3
            vcs.append(vc)
        return vcs

    def _cb(self, index, cores, dtype, page, pages, row_tile=False):
        kwargs = dict(buffer_index=index, data_format=dtype, page_size=page)
        if row_tile:
            kwargs["tile"] = ttnn.TileDescriptor(ttnn.Tile([ROWS, TILE]))
        return ttnn.CBDescriptor(
            total_size=page * pages, core_ranges=cores, format_descriptors=[ttnn.CBFormatDescriptor(**kwargs)]
        )

    def descriptor(self, w_gate_up, w_down):
        """The ProgramDescriptor and its io tensors (the output buffer last)."""
        share, P, S, NF = self.share, self.share.P, self.share.S, self.NF
        pl = self.placement
        hub_noc = self._noc(self.hub)
        lo, hi = pl.noc[(0, 0)], pl.noc[(pl.grid[0] - 1, pl.grid[1] - 1)]
        mc = [lo.x, lo.y, hi.x, hi.y, pl.grid[0] * pl.grid[1] - 1]
        g_crs, d_crs = _crs(self.g_kernel_cores), _crs(self.d_kernel_cores)
        f_crs, h_crs = _crs(self.fin), _crs([self.hub])
        SL = -(-self.max_slots // NF)  # slots a finisher can own
        part_pages = SL * P * S * 2  # a finisher's slots x P units x S partials x 2 tiles
        ring_slots = min(8, SL)
        gr, gx = self.gate_up_ring
        dr, dx = self.down_ring
        g_block_tiles = GATE_UP_KB * 2
        g_block = g_block_tiles * BFP4_TILE_BYTES
        d_block_tiles = self.cols * P

        cbs = [  # the whole-grid buffers first: the same address on every core
            self._cb(CB_X, self.grid_crs, ttnn.bfloat16, ROW_TILE, K_TILES, row_tile=True),
            self._cb(CB_META, self.grid_crs, ttnn.uint32, META_PAGE, 1),
            self._cb(CB_ACT, self.grid_crs, ttnn.bfloat16, ROW_TILE, NF * SL * P, row_tile=True),
            self._cb(CB_PART, self.grid_crs, ttnn.float32, ROW_TILE_F32, part_pages, row_tile=True),
            self._cb(CB_CTL, self.grid_crs, ttnn.uint32, 1024, 1),  # control words; a router core's scratch
            self._cb(CB_W, g_crs, ttnn.bfloat4_b, BFP4_TILE_BYTES, gr * g_block_tiles),
            self._cb(CB_OUT, g_crs, ttnn.float32, ROW_TILE_F32, 4, row_tile=True),
            self._cb(CB_W, d_crs, ttnn.bfloat4_b, BFP4_TILE_BYTES, dr * d_block_tiles),
            self._cb(CB_OUT, d_crs, ttnn.bfloat16, ROW_TILE, self.cols, row_tile=True),
            self._cb(CB_SC, f_crs, ttnn.float32, ROW_TILE_F32, 2, row_tile=True),
            self._cb(CB_OUT, f_crs, ttnn.bfloat16, ROW_TILE, ring_slots * P, row_tile=True),
        ]
        # no program semaphores (the hand-overs are words of self.sync); dummy ones only to measure what they cost
        sems = [
            ttnn.SemaphoreDescriptor(i, ttnn.CoreType.WORKER, self.grid_crs, 0) for i in range(self.dummy_semaphores)
        ]

        status_addr, out_addr = self.status.buffer_address(), self.out.buffer_address()
        dm = ttnn.DataMovementConfigDescriptor
        reader_cfg = dm(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0)
        writer_cfg = dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1)
        compute_cfg = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, math_approx_mode=False
        )

        def kernel(name, cores, ct, rt, config, defines=(), stream=None):
            extra = {}
            if stream is not None and self.stream_rt:  # a stream reader: its stream's address with the launch
                defines = list(defines) + [("HY4_STREAM_RT", "1")]
                extra["common_runtime_args"] = [stream.buffer_address()]
            return ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / name),
                core_ranges=cores,
                compile_time_args=ct,
                runtime_args=rt,
                config=config,
                defines=self.defines + list(defines),
                **extra,
            )

        # hub
        hub_rt = ttnn.RuntimeArgs()
        io = [w_gate_up, w_down, self.x, self.meta, self.status, self.sync]
        io = [t for t in io if isinstance(t, ttnn.Tensor)]  # (hub buffers given as bare addresses are not tensors)
        if not self.router:
            hub_rt[self.hub.x][self.hub.y] = [self.x.buffer_address(), self.meta.buffer_address(), status_addr]
            kernels = [kernel("hub.cpp", h_crs, [CB_X, CB_META, K_TILES * ROW_TILE] + mc, hub_rt, writer_cfg)]
        else:
            kernels = self._router_kernels(kernel, cbs, mc, hub_noc, h_crs, reader_cfg, writer_cfg, status_addr)
            io += [
                t for t in (self.w_router, self.scores, self.bias, self.picks, self.cand) if isinstance(t, ttnn.Tensor)
            ]

        # the expert matmuls: the original kernels, or the hifi_mm ones at the chosen pass schedule
        if self.fidelity == "custom_mm":
            gate_up_compute, down_compute, fid_defines = "gate_up_compute.cpp", "down_compute.cpp", []
        else:
            passes, step = FIDELITY[self.fidelity]
            gate_up_compute, down_compute = "gate_up_compute_hifi.cpp", "down_compute_hifi.cpp"
            fid_defines = [("HY4_FID_PASSES", str(passes)), ("HY4_FID_STEP", str(step))]

        # gate | up
        # per-core constants go into the cores' sync words (self._write_core_words), not into runtime arguments
        self._write_core_words(w_gate_up, w_down)
        item_blocks = share.piece_k // GATE_UP_KB
        reader_ct = [CB_W, CB_CTL, CB_META, share.pb, share.piece_bytes, item_blocks, g_block_tiles, g_block]
        reader_ct += [packets(g_block), gr, gx, share.gate_up_stride, self.G, 1, 0]
        writer_ct = [CB_X, CB_META, CB_CTL, CB_PART, CB_OUT, K_TILES, share.pb, BANKS, S, P, NF]
        writer_ct += [hub_noc.x, hub_noc.y, status_addr, self.G]
        for c in self.fin:
            n = self._noc(c)
            writer_ct += [n.x, n.y]
        kernels += [
            kernel("stream_reader.cpp", g_crs, reader_ct, [], reader_cfg, stream=w_gate_up),
            kernel("gate_up_writer.cpp", g_crs, writer_ct, [], writer_cfg),
            kernel(
                gate_up_compute,
                g_crs,
                [CB_X, CB_W, CB_OUT, K_TILES, GATE_UP_KB, item_blocks, share.pb, S, share.piece_k, self.G],
                [],
                compute_cfg,
                fid_defines,
            ),
        ]

        # finishers
        fin_ct = [CB_META, CB_PART, CB_SC, CB_OUT, CB_ACT, S, P, NF] + mc + [hub_noc.x, hub_noc.y, status_addr]
        fin_ct += [SL, ring_slots]
        fin_cc = [CB_PART, CB_SC, CB_OUT, S, f32_bits(10.0), f32_bits(-10.0), f32_bits(-3.0e38), P, NF]
        fin_cc += [f32_bits(self.bias_correction[0]), f32_bits(self.bias_correction[1])]
        kernels += [
            kernel("finish_writer.cpp", f_crs, fin_ct, [], writer_cfg),
            kernel("finish_compute.cpp", f_crs, fin_cc, [], compute_cfg),
        ]

        # down
        reader_ct = [CB_W, CB_CTL, CB_META, 1, self.down_block, 1, d_block_tiles, self.down_block]
        reader_ct += [packets(self.down_block), dr, dx, share.down_stride, 1, 0, self.down_block]
        writer_ct = [CB_META, CB_CTL, CB_ACT, CB_OUT, P, self.cols, hub_noc.x, hub_noc.y, NF]
        writer_ct += [out_addr, status_addr, COLS_PER_BANK]
        kernels += [
            kernel("stream_reader.cpp", d_crs, reader_ct, [], reader_cfg, stream=w_down),
            kernel("down_writer.cpp", d_crs, writer_ct, [], writer_cfg),
            kernel(down_compute, d_crs, [CB_ACT, CB_W, CB_OUT, P, self.cols, NF, SL], [], compute_cfg, fid_defines),
        ]
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
        return program, io + ([self.out] if isinstance(self.out, ttnn.Tensor) else [])

    def _router_kernels(self, kernel, cbs, mc, hub_noc, h_crs, reader_cfg, writer_cfg, status_addr):
        """The hub with the router and the router cores: core c streams column c of the router weights (BF16 tiles,
        bank c, blocks of 8 K tiles = one 16 KB packet) and computes the 32 experts' scores for the 8 rows."""
        assert self.w_router is not None, "set_router first"
        r_crs = _crs(self.r_cores)
        kb, bf16_tile = 8, 2048
        cbs += [
            self._cb(CB_W, r_crs, ttnn.bfloat16, bf16_tile, 4 * kb),
            self._cb(CB_OUT, r_crs, ttnn.float32, ROW_TILE_F32, 1, row_tile=True),
        ]
        rdef = [("ROUTER_CANDIDATES", "1")] if self.router_candidates else []
        hub_args = [
            self.x.buffer_address(),
            status_addr,
            self.scores.buffer_address(),
            self.bias.buffer_address(),
            self.picks.buffer_address(),
            self.cand.buffer_address(),
        ]
        if self.bias_rt:
            hub_args += self.bias_args()
            rdef = rdef + [("HY4_BIAS_ARG", str(self.BIAS_ARG))]
        hub_rt = ttnn.RuntimeArgs()
        hub_rt[self.hub.x][self.hub.y] = hub_args
        hub_ct = [CB_X, CB_META, K_TILES * ROW_TILE] + mc
        hub_ct += [len(self.r_cores), self.num_experts, 8, round(2.827 * 65536)]
        hub_ct += [self.expert_base, self.expert_count, self.n_plain, self.plain_id0]
        item = K_TILES * bf16_tile
        reader_ct = [CB_W, CB_CTL, CB_META, 1, item, K_TILES // kb, kb, kb * bf16_tile, 1, 4, 2, 0, 1, 0, 0]
        router_ct = [CB_X, CB_CTL, CB_OUT, K_TILES, hub_noc.x, hub_noc.y]
        router_ct += [self.scores.buffer_address(), self.bias.buffer_address(), self.cand.buffer_address()]
        hifi4 = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
        )
        return [
            kernel("hub_router.cpp", h_crs, hub_ct, hub_rt, writer_cfg, defines=rdef),
            kernel(
                "stream_reader.cpp",
                r_crs,
                reader_ct,
                [],
                reader_cfg,
                defines=[("STREAM_FIXED", "1")],
                stream=self.w_router,
            ),
            kernel("router_writer.cpp", r_crs, router_ct, [], writer_cfg, [d for d in rdef if d[0] != "HY4_BIAS_ARG"]),
            kernel("router_compute.cpp", r_crs, [CB_X, CB_W, CB_OUT, K_TILES, kb], [], hifi4),
        ]

    def bias_args(self):
        """bias_rt: the bias as the hub's runtime arguments BIAS_ARG .. (Q30 words)."""
        assert self.bias_values is not None and len(self.bias_values) == self.num_experts, "bias_rt: set_router first"
        return [int(v) & 0xFFFFFFFF for v in self.bias_values]

    def set_router(self, weight, bias):
        """The router's weight [E, 6144] (stored BF16, exact for the BF16 checkpoint: column c of weight.T in DRAM
        bank c) and its selection bias [E] (as Q30 integers)."""
        E = self.num_experts
        assert tuple(weight.shape) == (E, K_TILES * TILE) and tuple(bias.shape) == (E,)
        from .layout import stream_memory_config

        self.w_router = ttnn.from_torch(
            weight.float().T.contiguous()[None, None],
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=stream_memory_config(K_TILES * TILE),
        )
        q = torch.round(bias.double() * (1 << 30)).to(torch.int32).reshape(1, E)
        self.bias_values = q.flatten().tolist()
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(q, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT), self.bias)

    def set_x(self, x):
        """Host -> the hub's x buffer: [8, 6144]."""
        from .layout import to_row_tiles

        host_x = ttnn.from_torch(to_row_tiles(x.float()), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.copy_host_to_device_tensor(host_x, self.x)

    def read_router(self):
        """What the router left for the host: scores [8, E] (FP32 sigmoid), the rows' picks [8, 8] (expert ids, best
        first) and their weights [8, 8] (BF16)."""
        from .layout import from_row_tiles

        scores = from_row_tiles(ttnn.to_torch(self.scores).float())
        picks = ttnn.to_torch(self.picks).to(torch.int32) & 0xFFFF
        ids = picks[:, :8]
        weights = picks[:, 8:].contiguous().to(torch.int16).view(torch.bfloat16).float()
        return scores, ids, weights

    def set_inputs(self, x, ids, weights, n_plain=0):
        """Host -> the hub's buffers: x [8, 6144], the union's local expert ids, weights [8, U]; the last n_plain
        slots are plain SwiGLU (no clamp)."""
        from .layout import meta_words, to_row_tiles

        host_x = ttnn.from_torch(to_row_tiles(x.float()), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        host_meta = ttnn.from_torch(meta_words(ids, weights, n_plain), dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.copy_host_to_device_tensor(host_x, self.x)
        ttnn.copy_host_to_device_tensor(host_meta, self.meta)

    def __call__(self, w_gate_up, w_down):
        """Run on the routing result and x currently in the hub's buffers; returns the output buffer. On the mock
        device the program is compiled and checked, not run."""
        program, io = self.descriptor(w_gate_up, w_down)
        if MOCK:
            ttnn.experimental.prepare_generic_op(io, program)
        else:
            if os.environ.get("HY4_MOE_BOARD") != "1":
                raise RuntimeError(BOARD_GATE)
            ttnn.generic_op(io, program)
        return self.out

    def _core_words(self, w_gate_up, w_down):
        """[cores, 16] sync words: 0..7 the hand-over words (zero), 8.. each core's constants (moe_common.hpp CFG_*):
        its DRAM bank, its index in its role, the virtual channel of its DRAM reads, its weight stream's address."""
        gx = self.placement.grid[0]
        words = torch.zeros(self.placement.grid[0] * self.placement.grid[1], 16, dtype=torch.int64)

        def put(core, bank, index, vc, addr):
            assert addr % 32 == 0 and addr >> 5 < 1 << 31, hex(addr)  # stored >> 5: it stays a positive int32
            words[core.y * gx + core.x, 8:12] = torch.tensor([bank, index, vc, addr >> 5])

        g_list = [(c, b, j) for b in self.banks for j, c in enumerate(self.g_cores[b])]
        for (c, b, j), vc in zip(g_list, self._vcs(g_list)):
            put(c, b, j, vc, w_gate_up.buffer_address())
        d_list = [(c, b, d) for b in self.banks for d, c in enumerate(self.d_cores[b])]
        for (c, b, d), vc in zip(d_list, self._vcs(d_list)):
            put(c, b, d, vc, w_down.buffer_address())
        for c in self.placement.idle_gate_up + self.placement.idle_down:
            put(c, 0, IDLE, 0, 0)
        for f, c in enumerate(self.fin):
            put(c, 0, f, 0, 0)
        for col, c in enumerate(self.r_cores):
            put(c, col % BANKS, col, col & 0x3, self.w_router.buffer_address() if self.w_router is not None else 0)
        return words

    def _write_core_words(self, w_gate_up, w_down):
        """Host -> every core's sync words, when the streams (their addresses) changed. Also clears the hand-over
        words."""
        key = (
            w_gate_up.buffer_address(),
            w_down.buffer_address(),
            self.w_router.buffer_address() if self.router and self.w_router is not None else 0,
        )
        if self.stream_rt:  # the words hold no address a kernel reads: written once
            key = ("stream_rt",)
        if key != self._core_words_key:
            words = self._core_words(w_gate_up, w_down)
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(words.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT), self.sync
            )
            self._core_words_key = key

    def reset_sync(self):
        """Clear the hand-over words on every core: after a run that reported a timeout a late hand-over may have
        left one set."""
        self._core_words_key = None  # the next descriptor() writes the cores' words again, hand-over words zero

    def prepare(self, w_gate_up, w_down):
        """Compile the program and check its buffers without running it."""
        program, io = self.descriptor(w_gate_up, w_down)
        ttnn.experimental.prepare_generic_op(io, program)

    def read_output(self):
        """The output buffer as [8, 6144] and the status words (word 0: runs; 1..4: last timeout or failed send of
        a gate|up / finisher / down / router core, 5: the hub could not send; 0 when none)."""
        from .layout import from_row_tiles

        out = from_row_tiles(ttnn.to_torch(self.out).float())
        status = ttnn.to_torch(self.status).flatten().tolist()
        return out, [int(v) & 0xFFFFFFFF for v in status]

    def read_timing(self):
        """With the HY4_TIMING define: per role, each core's (logical x, y, cycles from the release to its first
        hand-over, cycles to its end), and the hub's cycles to (x sent, scores in, release sent)."""
        words = [int(v) & 0xFFFFFFFF for v in ttnn.to_torch(self.status).flatten().tolist()]

        def core(c):
            n = self._noc(c)
            i = 16 + 2 * ((n.y - 2) * 16 + n.x)
            return (c.x, c.y, words[i], words[i + 1])

        return {
            "hub": words[8:11],
            "gate_up": {b: [core(c) for c in self.g_cores[b]] for b in self.banks},
            "down": {b: [core(c) for c in self.d_cores[b]] for b in self.banks},
            "finish": [core(c) for c in self.fin],
        }

    def bytes_read(self, n_slots, n_plain=0):
        """Expert weight bytes the chip streams for a union of n_slots, n_plain of them plain (with all banks
        active)."""
        entries = n_slots + n_plain * (self.plain_terms - 1)
        return entries * self.share.expert_bytes * len(self.banks) // BANKS
