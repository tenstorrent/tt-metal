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

Interface to the cross-chip reduce. The chip's partial output is [8, 6144] BF16 in the output buffer, one core's
L1 (here the hub's; the down cores take its core and address as arguments): 192 row tiles of [8, 32], tile n = hidden
columns [32 n, 32 n + 32) as 8 rows x 16 values of columns 0..15, then 8 x 16 of columns 16..31; 512 B a tile,
98304 B in all (8 fabric packets at the mesh's 12352 B payload). Each down core writes its 4 (or 8) adjacent tiles once,
at the end, then increments program semaphore 4 (SEM_OUT) of that core: the buffer is complete when the semaphore
equals the number of down cores (48, or 24). The sum over the chips of a split (tensor parallel: all 32; expert
parallel: all 32, zeros from the chips without a picked expert) is the MoE's routed output, plus the shared expert
when it runs as a plain slot.

Buffers at the same L1 address on every core (CBs declared on the whole grid, allocated first): x, the meta page,
the activation buffer (tile slot * P + pair), the finishers' partial buffer. Cross-core hand-overs are counters in the
meta page, zeroed by the hub's multicast and incremented by NOC atomics; every wait on another core is bounded
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
# State on 2026-11-19: the first three runs of this program on the board (tp32, a union of 1, 8 and 64) hung chip 0
# until the 120 s operation timeout: the hub's NOC1 multicast gave the rectangle's corners in ascending order, which
# stalls the sender on Blackhole (kernels/moe_dm.hpp grid_mcast_addr has the rule and its sources). The corners are
# fixed and every NOC wait of the data-movement kernels is bounded since; with that, tp32 with unions of 1 to 64 runs
# on one chip. Accuracy and rate work continues, so running it on the board stays an explicit decision:
BOARD_GATE = (
    "fused/moe is work in progress on the board (see program.py: its first runs hung chip 0 before a fix). "
    "Set HY4_MOE_BOARD=1 to run it on the device, one chip, smallest case first "
    "(tests/test_routed.py -k 'tp32 and U1'), and check the status words."
)
ROW_TILE = 2 * TILE * ROWS  # bytes of a [8, 32] BF16 tile
ROW_TILE_F32 = 2 * ROW_TILE
META_PAGE = 2 * META_IN_WORDS + 4 * MAX_SLOTS + 4 * MAX_UNITS + 64  # kernels/moe_common.hpp META_PAGE_BYTES
MAX_PACKET = 16384
GATE_UP_KB = 48  # K tiles per gate | up block: 96 tiles = 55296 B = 4 packets of 13824 B

CB_X, CB_META, CB_ACT, CB_PART, CB_CTL, CB_W, CB_SC, CB_OUT = 0, 1, 2, 3, 4, 5, 6, 16

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

    def describe(self):
        return {b: [(c.x, c.y) for c in cs] for b, cs in self.readers.items()}


def place(device, per_bank, banks=None, mode="side"):
    """per_bank reader cores for each DRAM bank in ``banks`` (default all).

    NOC0 routes a DRAM read's data east along the row of the bank's endpoint, then south down the reader's column.
    mode "side": a bank takes cores on its own side of the chip (the columns from its anchor to the next DRAM
    column), rows at and below its anchor row first (the short way round for NOC0), nearest first. mode "row": the
    anchor's row first across the whole chip, then the rows below. mode "any": nearest by wrap-around NOC0 distance.
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
    for _ in range(per_bank):
        for b in banks:
            c = next(c for c in order[b] if c not in used)
            used.add(c)
            readers[b].append(ttnn.CoreCoord(*c))
    spare = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x) if (x, y) not in used]
    return Placement((grid.x, grid.y), readers, spare, noc)


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
    ):
        """router: the hub runs the router on the chip (8 router cores, one per 32 experts; set_router gives it the
        weights) instead of taking the routing result from the host. expert_base / expert_count: the chip's own
        experts among the router's num_experts (expert parallel; default all), numbered from expert_base in the
        weight streams. n_plain / plain_id0: always-on plain slots the router appends to the union (the shared
        expert: stream entry plain_id0). plain_terms = 2: a plain slot's weights are stored as two consecutive
        stream entries, two BFP4 terms whose sum the matmuls accumulate (layout.two_terms): the accuracy of BFP8 at
        LoFi, for the shared expert and the dense FFN. bias_correction = (k_in, k_act): factors the finisher applies
        to the gate / up sums and to the activation against the truncations of the LoFi path (TRUNCATION_CORRECTION;
        1.0 = off)."""
        self.bias_correction = bias_correction
        # the router cores cut their 32 experts to each row's 8 best; the hub merges 8 lists instead of scanning 256
        self.router_candidates = router_candidates
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
        self.placement = pl = place(device, self.G + self.D, self.banks, mode)
        n_router = self.router_cols if router else 0
        assert len(pl.spare) >= 1 + finishers + n_router, f"{len(pl.spare)} spare cores: hub, finishers, router"
        self.hub = pl.spare[0]
        self.fin = pl.spare[1 : 1 + finishers]
        self.r_cores = pl.spare[1 + finishers : 1 + finishers + n_router]
        self.g_cores = {b: pl.readers[b][: self.G] for b in self.banks}
        self.d_cores = {b: pl.readers[b][self.G :] for b in self.banks}
        self.all_g = [c for b in self.banks for c in self.g_cores[b]]
        self.all_d = [c for b in self.banks for c in self.d_cores[b]]
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

        # the hub's buffers
        self.x = hub_tensor(device, self.hub, torch.zeros(K_TILES, 2 * 16 * ROWS), ttnn.bfloat16)
        self.out = hub_tensor(device, self.hub, torch.zeros(K_TILES, 2 * 16 * ROWS), ttnn.bfloat16)
        self.meta = hub_tensor(device, self.hub, torch.zeros(1, META_IN_WORDS, dtype=torch.int32), ttnn.uint16)
        self.status = hub_tensor(device, self.hub, torch.zeros(1, 16 + 2 * 160, dtype=torch.int32), ttnn.uint32)
        if router:
            self.scores = hub_tensor(device, self.hub, torch.zeros(self.router_cols, 32 * ROWS), ttnn.float32)
            self.bias = hub_tensor(device, self.hub, torch.zeros(1, num_experts, dtype=torch.int32), ttnn.int32)
            self.picks = hub_tensor(device, self.hub, torch.zeros(ROWS, 16, dtype=torch.int32), ttnn.uint16)
            self.cand = hub_tensor(
                device, self.hub, torch.zeros(self.router_cols, ROWS * 8 * 3, dtype=torch.int32), ttnn.int32
            )
            self.w_router = None

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
        g_crs, d_crs, f_crs, h_crs = _crs(self.all_g), _crs(self.all_d), _crs(self.fin), _crs([self.hub])
        part_pages = -(-self.max_slots // NF) * P * S * 2  # a finisher's slots x P units x S partials x 2 tiles
        gr, gx = self.gate_up_ring
        dr, dx = self.down_ring
        g_block_tiles = GATE_UP_KB * 2
        g_block = g_block_tiles * BFP4_TILE_BYTES
        d_block_tiles = self.cols * P

        cbs = [  # the whole-grid buffers first: the same address on every core
            self._cb(CB_X, self.grid_crs, ttnn.bfloat16, ROW_TILE, K_TILES, row_tile=True),
            self._cb(CB_META, self.grid_crs, ttnn.uint32, META_PAGE, 1),
            self._cb(CB_ACT, self.grid_crs, ttnn.bfloat16, ROW_TILE, self.max_units, row_tile=True),
            self._cb(CB_PART, self.grid_crs, ttnn.float32, ROW_TILE_F32, part_pages, row_tile=True),
            self._cb(CB_CTL, self.grid_crs, ttnn.uint32, 1024, 1),  # control words; a router core's scratch
            self._cb(CB_W, g_crs, ttnn.bfloat4_b, BFP4_TILE_BYTES, gr * g_block_tiles),
            self._cb(CB_OUT, g_crs, ttnn.float32, ROW_TILE_F32, 4, row_tile=True),
            self._cb(CB_W, d_crs, ttnn.bfloat4_b, BFP4_TILE_BYTES, dr * d_block_tiles),
            self._cb(CB_OUT, d_crs, ttnn.bfloat16, ROW_TILE, self.cols, row_tile=True),
            self._cb(CB_SC, f_crs, ttnn.float32, ROW_TILE_F32, 2, row_tile=True),
            self._cb(CB_OUT, f_crs, ttnn.bfloat16, ROW_TILE, 2 * P, row_tile=True),  # two slots of P tiles
        ]
        sems = [ttnn.SemaphoreDescriptor(i, ttnn.CoreType.WORKER, self.grid_crs, 0) for i in range(5)]

        status_addr, out_addr = self.status.buffer_address(), self.out.buffer_address()
        dm = ttnn.DataMovementConfigDescriptor
        reader_cfg = dm(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0)
        writer_cfg = dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1)
        compute_cfg = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, math_approx_mode=False
        )

        def kernel(name, cores, ct, rt, config, defines=()):
            return ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / name),
                core_ranges=cores,
                compile_time_args=ct,
                runtime_args=rt,
                config=config,
                defines=self.defines + list(defines),
            )

        # hub
        hub_rt = ttnn.RuntimeArgs()
        io = [w_gate_up, w_down, self.x, self.meta, self.status]
        if not self.router:
            hub_rt[self.hub.x][self.hub.y] = [self.x.buffer_address(), self.meta.buffer_address(), status_addr]
            kernels = [kernel("hub.cpp", h_crs, [CB_X, CB_META, K_TILES * ROW_TILE] + mc, hub_rt, writer_cfg)]
        else:
            kernels = self._router_kernels(kernel, cbs, mc, hub_noc, h_crs, reader_cfg, writer_cfg, status_addr)
            io += [self.w_router, self.scores, self.bias, self.picks, self.cand]

        # gate | up
        g_list = [(c, b, j) for b in self.banks for j, c in enumerate(self.g_cores[b])]
        g_vcs = self._vcs(g_list)
        r_rt, w_rt, c_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for (c, b, j), vc in zip(g_list, g_vcs):
            r_rt[c.x][c.y] = [w_gate_up.buffer_address(), b, vc, j, self.G, 0]
            w_rt[c.x][c.y] = [j, self.G, b]
            c_rt[c.x][c.y] = [(b % S) * share.piece_k, j, self.G]
        item_blocks = share.piece_k // GATE_UP_KB
        reader_ct = [CB_W, CB_CTL, CB_META, share.pb, share.piece_bytes, item_blocks, g_block_tiles, g_block]
        reader_ct += [packets(g_block), gr, gx, share.gate_up_stride]
        writer_ct = [CB_X, CB_META, CB_CTL, CB_PART, CB_OUT, K_TILES, share.pb, BANKS, S, P, NF]
        writer_ct += [hub_noc.x, hub_noc.y, status_addr]
        for c in self.fin:
            n = self._noc(c)
            writer_ct += [n.x, n.y]
        kernels += [
            kernel("stream_reader.cpp", g_crs, reader_ct, r_rt, reader_cfg),
            kernel("gate_up_writer.cpp", g_crs, writer_ct, w_rt, writer_cfg),
            kernel(
                "gate_up_compute.cpp",
                g_crs,
                [CB_X, CB_W, CB_OUT, K_TILES, GATE_UP_KB, item_blocks, share.pb],
                c_rt,
                compute_cfg,
            ),
        ]

        # finishers
        f_rt = ttnn.RuntimeArgs()
        for f, c in enumerate(self.fin):
            f_rt[c.x][c.y] = [f]
        fin_ct = [CB_META, CB_PART, CB_SC, CB_OUT, CB_ACT, S, P, NF] + mc + [hub_noc.x, hub_noc.y, status_addr]
        fin_cc = [CB_PART, CB_SC, CB_OUT, S, f32_bits(10.0), f32_bits(-10.0), f32_bits(-3.0e38), P, NF]
        fin_cc += [f32_bits(self.bias_correction[0]), f32_bits(self.bias_correction[1])]
        kernels += [
            kernel("finish_writer.cpp", f_crs, fin_ct, f_rt, writer_cfg),
            kernel("finish_compute.cpp", f_crs, fin_cc, f_rt, compute_cfg),
        ]

        # down
        d_list = [(c, b, d) for b in self.banks for d, c in enumerate(self.d_cores[b])]
        d_vcs = self._vcs(d_list)
        r_rt, w_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for (c, b, d), vc in zip(d_list, d_vcs):
            r_rt[c.x][c.y] = [w_down.buffer_address(), b, vc, 0, 1, d * self.down_block]
            w_rt[c.x][c.y] = [b * COLS_PER_BANK + d * self.cols, out_addr, status_addr]
        reader_ct = [CB_W, CB_CTL, CB_META, 1, self.down_block, 1, d_block_tiles, self.down_block]
        reader_ct += [packets(self.down_block), dr, dx, share.down_stride]
        writer_ct = [CB_META, CB_CTL, CB_ACT, CB_OUT, P, self.cols, hub_noc.x, hub_noc.y]
        kernels += [
            kernel("stream_reader.cpp", d_crs, reader_ct, r_rt, reader_cfg),
            kernel("down_writer.cpp", d_crs, writer_ct, w_rt, writer_cfg),
            kernel("down_compute.cpp", d_crs, [CB_ACT, CB_W, CB_OUT, P, self.cols], [], compute_cfg),
        ]
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
        return program, io + [self.out]

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
        hub_rt = ttnn.RuntimeArgs()
        hub_rt[self.hub.x][self.hub.y] = [
            self.x.buffer_address(),
            status_addr,
            self.scores.buffer_address(),
            self.bias.buffer_address(),
            self.picks.buffer_address(),
            self.cand.buffer_address(),
        ]
        hub_ct = [CB_X, CB_META, K_TILES * ROW_TILE] + mc
        hub_ct += [len(self.r_cores), self.num_experts, 8, round(2.827 * 65536)]
        hub_ct += [self.expert_base, self.expert_count, self.n_plain, self.plain_id0]
        r_rt, w_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for c, core in enumerate(self.r_cores):
            r_rt[core.x][core.y] = [self.w_router.buffer_address(), c % BANKS, c & 0x3, 0, 1, 0]
            w_rt[core.x][core.y] = [
                c,
                self.scores.buffer_address(),
                self.bias.buffer_address(),
                self.cand.buffer_address(),
            ]
        item = K_TILES * bf16_tile
        reader_ct = [CB_W, CB_CTL, CB_META, 1, item, K_TILES // kb, kb, kb * bf16_tile, 1, 4, 2, 0]
        hifi4 = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
        )
        return [
            kernel("hub_router.cpp", h_crs, hub_ct, hub_rt, writer_cfg, defines=rdef),
            kernel("stream_reader.cpp", r_crs, reader_ct, r_rt, reader_cfg, defines=[("STREAM_FIXED", "1")]),
            kernel(
                "router_writer.cpp",
                r_crs,
                [CB_X, CB_CTL, CB_OUT, K_TILES, hub_noc.x, hub_noc.y],
                w_rt,
                writer_cfg,
                rdef,
            ),
            kernel("router_compute.cpp", r_crs, [CB_X, CB_W, CB_OUT, K_TILES, kb], [], hifi4),
        ]

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
