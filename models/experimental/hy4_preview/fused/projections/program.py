# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""One chip's share of the attention projections of Hy4-preview decode for 8 token rows, as programs whose weights
are streamed from DRAM once (layout.py has the split over the 32 chips).

Every program is a set of UNITS on separate cores plus the hub. A unit is

    left operand   k_tiles [8, 32] BF16 row tiles resident in L1: of the input buffer the hub multicasts (stage 1),
                   or of the activation buffer that stage-1 units fill by multicast (stage 2)
    weights        its ct output tile columns, [k_tiles, ct] tiles K-major, streamed from its core's DRAM bank
                   (BFP8, BFP4, or two BFP4 terms accumulated into the same sums)
    finishing      in the SFPU on the FP32 sums (kernels/proj_sfpu.hpp): the deferred norm's row scalars, sigmoid,
                   RoPE, the gate multiply, the lane sums of squares, rounding
    destination    ct tiles of the output buffer in the hub's L1, or of the activation buffer on every core

and runs three kernels: kernels/proj_reader.cpp (the weight streamer, from the MoE piece's), proj_compute.cpp and
proj_writer.cpp. The hub (kernels/proj_hub.cpp, from the MoE piece's) multicasts the input buffer and the meta page
and releases the cores. The programs have no program semaphores (they are rewritten on every core at each launch):
the release is a word of a persistent L1 tensor, the hand-overs between stages are counters in the meta page.

    pre     x [8, 6144]                 -> q_a's 2 tiles (+ lane sums), the pool tile, the gate's 16 tiles (sigmoid),
                                           on full layers the index head's 4 query tiles
    query   q_a [8, 2048] | cos | sin   -> per head q_b (nope 6 tiles to the activation buffer, rope 2 tiles with
                                           RoPE to the output) and the absorption nope W_k (16 tiles)
    post    ctx of the 2 heads | gate   -> per head ctx W_v^T times the gate (8 tiles to the activation buffer), then
                                           o_proj's rows of the 2 heads: the chip's partial [8, 6144]

Fidelity (ProjProgram(fidelity=...), default "custom_mm" = the kernels as they were). With "recommended" every unit
runs hifi_mm at the cheapest schedule that takes its operands whole: units fed by the broadcast x or by ctx (BF16)
HiFi3 for BFP8 weights, units fed by tiles a unit before them rounded to 7 bits (q_b, the absorption, o_proj, the
Hadamard basis: Unit.left7) HiFi2; with BFP4 weights phases {0, 2} and LoFi. The split accumulators are merged in
FP32 in every case. The streams hold BFP(BF16(w)) (layout.stored_stream), which is what the tests' "exact arithmetic
on the stored weights" is taken on; pre_rounded=True makes them hold layout.stored(w) (it matters for two BFP4
terms).

Measured with "recommended" on checkpoint layers 0 to 5 (chip 13, BFP8, logs mk9_proj_prec_custom2 /
mk9_proj_prec_rec3; fused/fidelity/GUIDE.md section 8 has the table): against exact arithmetic on the stored weights
q_a 2.0e-2 (gain 0.985) -> 3.4e-3 (1.000), q_abs 2.6e-2 (0.976) -> 2.5e-3, the o_proj partial 3.2e-2 (0.973) ->
3.5e-3; against the reference q_a 2.2e-2 -> 0.85e-2 and the o_proj partial 3.3e-2 -> 1.1e-2 (ttnn.matmul HiFi2:
0.86e-2 and 1.25e-2). Traced, 8 calls per trace: pre 24.4 -> 24.8 us, query 24.2 -> 23.8, post 27.3 -> 27.3.

For a caller's program (fused/layer_attn; tests/test_interface.py runs each on one chip, bit-equal to the
stand-alone programs): ProjProgram(cores=, hub=, rect=, hub_buffers=, cb_base=, weights=), Unit.done /
pre_program(gate_done=) with done_address(group) and done_units(group), combine([programs]), mesh_stream(mesh_device,
shards) with host_shard(unit_stream(units, fmt), fmt) or prepare.load_host_stream(...) as shards, and the form
without the deferred norm: layout.layer_weights(attention, None, indexer) / prepare.layer_weights(layer,
gamma=False), set_params() (scale 1), the streams of `prepare.py --normalised` (<format>_xn).

What was generalised from the MoE program: its gate|up, finisher and down roles are one unit type here (any K, any
ct, the left operand from either buffer, the finishing as a mode); the streamer reads one fixed run of bytes per core
(a unit's block, both terms back to back) and therefore starts with the kernel instead of waiting for the routing
result; weights are BFP8 as well as BFP4; finishing happens on the core that did the matmul, on FP32 sums that never
leave DEST (the MoE's finisher takes FP32 partials through a circular buffer, which arrive at BF16 precision).

Interfaces are in the builders' docstrings (pre_units, query_units, post_units).

Measured on the board, one chip, 8 rows, 2026-11-19 (logs mk7_all_board_1, mk7_final_board_1; tests/):
  * traced, us per program as a trace of its own / with 8 calls per trace (kernel binaries resident), BFP8, and the
    same program with every kernel returning at once (its launch floor):
        pre 42.2 / 24.4 (floor 21.0 / 14.1)    pre of a full layer 49.4 / 27.0 (26.3 / 18.0)
        query 47.5 / 23.8 (28.1 / 19.0)        post 47.5 / 27.4 (30.7 / 21.9)
    Two BFP4 terms cost 0.1 to 1.2 us more. The programs are bound by their launch and their hand-overs, not by DRAM
    or the matmuls (pre: 5.8 us shorter without the DRAM reads, 1.8 us without the matmuls); fewer, larger units are
    faster (pre with 6 units 38.6, query with 8 units 35.3 against 81 with 30, post with 28 units 42.7).
  * custom_mm is LoFi (tests/test_probe.py): the left operand is rounded to 7 significant bits (1.99609375 comes
    back as 2.0), a BFP8 weight is seen cut to about 5 significant bits (32% of the entries differ from the stored
    value, gain 0.988), a BFP4 weight is seen as stored in 99.9% of the entries and one quantisation step off in the
    rest (3% relative rms on random weights, whatever the block size of the stream).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch

import ttnn
from models.experimental.hy4_preview.fused.moe.program import _crs, hub_tensor, packets, place
from models.experimental.hy4_preview.fused.projections import layout as L

KERNEL_DIR = Path(__file__).parent / "kernels"
MOCK = bool(os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH"))  # the mock Galaxy compiles and checks, runs nothing
CB_X, CB_META, CB_ACT, CB_CTL, CB_W, CB_SC, CB_OUT, CB_STAT = 0, 1, 2, 4, 5, 6, 16, 17
CB_IDS = dict(x=CB_X, meta=CB_META, act=CB_ACT, ctl=CB_CTL, w=CB_W, sc=CB_SC, out=CB_OUT, stat=CB_STAT)  # + cb_base
ROW_TILE = 2 * L.TILE * L.ROWS  # bytes of a [8, 32] BF16 tile
ROW_TILE_F32 = 2 * ROW_TILE
META_PAGE = 1536 + 4 * 16 + 64  # kernels/proj_common.hpp META_PAGE_BYTES
RND_NONE, RND_BF16, RND_7 = 0, 1, 2
NO_WAIT = 0xFFFF
STATUS_WORDS = 16 + 2 * 160
MAX_BLOCK_BYTES = 56 * 1024  # a streamed block: at most 4 NOC packets; the streamer's ring holds 3 blocks
DTYPES = {"bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}
# ProjProgram(fidelity=...). "custom_mm": every unit's matmul is the tree's LoFi custom_mm, as the kernels were
# written. A key of FIDELITY: every unit through hifi_mm (fused/fidelity/kernels/hifi_mm.h) at that (passes, phase
# step), the split accumulators added in FP32. "recommended": per unit, the cheapest schedule that takes its operands
# whole (recommended_fidelity): the second pass is free (it hides under the unpack of the weights), the third costs
# FPU time.
FIDELITY = {"lofi": (1, 1), "hifi2": (2, 1), "hifi3": (3, 1), "hifi4": (4, 1), "p02": (2, 2)}


def recommended_fidelity(fmt, left7):
    """The multiplier takes 5 significant bits of the weight and 7 of the row operand per pass; phase 1 adds the
    weight's low bits, phase 2 the operand's (GUIDE.md section 1). BFP8 has 7 bits, BFP4 3, a BF16 row operand 8, one
    rounded by the unit before it (RND_7) 7."""
    if fmt == "bfp8":
        return "hifi2" if left7 else "hifi3"
    return "lofi" if left7 else "p02"


CYCLE_NS = 1.0  # wall-clock cycles of HY4_TIMING are 1 ns on Blackhole (1 GHz reference)


@dataclass
class Unit:
    """One core's work (module docstring). wt: [32 k_tiles, 32 ct] right operand, fp32."""

    name: str
    wt: torch.Tensor
    ct: int
    fin: int = L.FIN_NONE
    rnd: int = RND_BF16
    in_first: int = 0
    src_act: bool = False
    dst_act: bool = False
    out_tile: int = 0
    wait: tuple = (NO_WAIT, 0)  # (counter, how many stage-1 units must have delivered)
    ctr: int = 0  # the counter a unit with dst_act increments
    aux: int = 0
    param: int = 0  # the finishing's parameter (kernels/proj_sfpu.hpp finish)
    left7: bool = False  # the left operand has 7 significant bits (it was rounded with RND_7 where it was made)
    done: int = (
        0  # done group of a unit that writes the output buffer: it counts in word 1 + done of the hub's go tensor
    )
    fidelity: str = None  # a key of FIDELITY for this unit, whatever the program's setting says

    @property
    def k_tiles(self):
        return self.wt.shape[0] // L.TILE

    def kind(self):
        return (self.k_tiles, self.ct, self.fin, self.rnd, self.src_act, self.dst_act, self.param)


def cached_stream_exists(cache):
    """Whether ttnn.as_tensor has a file for this cache name (it appends the dtype and layout)."""
    p = Path(cache)
    return any(p.parent.glob(p.name + "*"))


def choose_kb(k_tiles, ct, tile_bytes, max_block=MAX_BLOCK_BYTES):
    """K tiles per streamed block: the largest divisor of k_tiles whose block fits max_block bytes."""
    for kb in range(min(k_tiles, 256), 0, -1):
        if k_tiles % kb == 0 and kb * ct * tile_bytes <= max_block:
            return kb
    raise ValueError((k_tiles, ct, tile_bytes))


def stream_memory_config(rows):
    """A weight stream's memory config: one [rows, 32] shard of tiles per DRAM bank."""
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(L.BANKS - 1, 0))}),
            (rows, L.TILE),
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )


def unit_stream(units, fmt, terms=1, pre_rounded=False, banks=None):
    """The [rows, 8 * 32] stream of a list of units as a ProjProgram lays it out (unit i in bank banks[i % n], a
    bank's blocks in unit order, K-major tiles, one block per term, banks padded to the longest): no device
    needed, e.g. for the shards of mesh_stream."""
    banks = list(range(L.BANKS)) if banks is None else list(banks)
    blocks = [[] for _ in range(L.BANKS)]
    for i, u in enumerate(units):
        parts = L.term_list(u.wt.float(), fmt, terms, pre_rounded)
        blocks[banks[i % len(banks)]].append(torch.cat([L.block_tiles(m, u.ct) for m in parts], dim=0))
    stream, _ = L.bank_streams(blocks)
    return stream


def host_shard(stream, fmt):
    """A program's stream ([rows, 8 * 32] torch, ProjProgram.host_stream) packed on the host: one rounding, so the
    device holds layout.stored of it (a tensor created directly with the sharded config is BF16 first)."""
    return ttnn.from_torch(stream[None, None].float().contiguous(), dtype=DTYPES[fmt], layout=ttnn.TILE_LAYOUT)


def mesh_stream(mesh_device, shards):
    """One device tensor holding every chip's weight stream: shards = one host tensor per chip of the mesh, in
    row-major mesh order (host_shard(...), or ttnn.load_tensor of a prepared stream), all of one shape. Give it to
    the chips' ProjPrograms as ``weights=``: the buffer address is the same on every chip."""
    rows = shards[0].shape[-2]
    assert all(tuple(s.shape) == tuple(shards[0].shape) for s in shards)
    host = shards[0] if len(shards) == 1 else ttnn.from_host_shards(list(shards), mesh_device.shape)
    return ttnn.to_device(host, mesh_device, memory_config=stream_memory_config(rows))


HUB_BUFFERS = ("x", "out", "stat", "meta", "status")


class ProjProgram:
    """A set of units on one chip. x_tiles: tiles of the input buffer; out_tiles: of the output buffer; act_tiles:
    of the activation buffer (0: no second stage). fmt / terms: how the weights are stored."""

    def __init__(
        self,
        device,
        units,
        x_tiles,
        out_tiles,
        act_tiles=0,
        fmt="bfp8",
        terms=1,
        mode="side",
        ring=(3, 1),
        defines=(),
        banks=None,
        cache=None,
        max_block=MAX_BLOCK_BYTES,
        fidelity="custom_mm",
        pre_rounded=False,
        cores=None,
        hub=None,
        rect=None,
        hub_buffers=None,
        cb_base=0,
        weights=None,
        rows=L.ROWS,
    ):
        """For a caller that puts the units into a program of its own (fused/layer_attn); every default is the
        stand-alone program as it was:
        cores: the units' cores in unit order ((x, y) logical); default: next to the units' DRAM banks (place).
        hub: the hub core (x, y); default: the placement's first spare core.
        rect: (x0, y0, x1, y1) logical, inclusive: the multicast rectangle and the cores that get the "whole-grid"
              buffers (x, meta page, activation buffer, go words); the hub and every unit must be inside. Default:
              the whole worker grid.
        hub_buffers: {"x", "out", "stat", "meta", "status": L1 address on the hub core} of buffers the caller owns
              (hub_buffer_bytes() gives their sizes); the program then allocates none of them and set_x, set_params
              and read are the caller's. Default: tensors of its own.
        cb_base: added to every circular-buffer id (0, 1, 2, 4, 5, 6, 16, 17), so that two groups fit one program.
        weights: a device tensor that holds this program's stream (mesh_stream: one shard per chip); default: built
              from the units' wt, or loaded from `cache`.
        With these, descriptor() returns the kernels, buffers and io tensors a caller appends to its own program
        (combine() joins several ProjPrograms into one ProgramDescriptor).

        cache: a file name the weight stream is kept under (ttnn.as_tensor's cache_file_name): when it exists the
        stream is loaded from it and the units' wt are only used for their shapes.
        fidelity: "custom_mm" (default: the kernels as they were, bit for bit), "recommended" or a key of FIDELITY
        (see there): the units' matmuls through hifi_mm with the FP32 merge (kernels/proj_compute_hifi.cpp).
        pre_rounded: the stream is given every term already rounded (layout.term_list), so that it holds
        layout.stored(wt, fmt, terms) instead of the BF16-then-BFP values of layout.stored_stream; a cache written
        without it holds the other values, so give such a stream its own cache name."""
        assert fmt in DTYPES and terms in (1, 2) and (terms == 1 or fmt == "bfp4"), (fmt, terms)
        assert fidelity in ("custom_mm", "recommended") or fidelity in FIDELITY, fidelity
        self.fidelity, self.pre_rounded = fidelity, pre_rounded
        self.device, self.units = device, list(units)
        self.x_tiles, self.out_tiles, self.act_tiles = x_tiles, out_tiles, act_tiles
        self.fmt, self.terms, self.ring = fmt, terms, ring
        self.max_block = max_block  # bytes of a streamed block (it is read as equal NOC packets of at most 16 KB)
        self.tile_bytes = L.TILE_BYTES[fmt]
        self.banks = list(range(L.BANKS)) if banks is None else list(banks)
        nb = len(self.banks)
        per_bank = -(-len(self.units) // nb)
        self.placement = pl = place(device, per_bank, self.banks, mode)
        self.hub = pl.spare[0] if hub is None else ttnn.CoreCoord(*hub)
        # rows: token rows of a row tile, 8 or 16. 16 = one pass of the weights' stream for two row tiles: [16, 32]
        # tiles through hifi_mm16 (kernels/proj_compute_hifi.cpp), so only with a fidelity that goes through hifi_mm.
        assert rows in (L.ROWS, 2 * L.ROWS) and (rows == L.ROWS or fidelity != "custom_mm"), (rows, fidelity)
        self.rows = rows
        self.row_tile = 2 * L.TILE * rows
        self.defines = [("ROWS", str(rows))] + list(defines)
        self.rect = (0, 0, pl.grid[0] - 1, pl.grid[1] - 1) if rect is None else tuple(rect)
        x0, y0, x1, y1 = self.rect
        assert 0 <= x0 <= x1 < pl.grid[0] and 0 <= y0 <= y1 < pl.grid[1], self.rect
        self.grid_crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))])
        # unit i -> bank i % nb, that bank's (i // nb)-th nearest core
        self.bank_of = [self.banks[i % nb] for i in range(len(self.units))]
        if cores is None:
            self.core_of = [pl.readers[self.bank_of[i]][i // nb] for i in range(len(self.units))]
        else:
            assert len(cores) == len(self.units), (len(cores), len(self.units))
            self.core_of = [ttnn.CoreCoord(*c) for c in cores]
            assert len({(c.x, c.y) for c in self.core_of} | {(self.hub.x, self.hub.y)}) == len(self.units) + 1
        for c in self.core_of + [self.hub]:
            assert x0 <= c.x <= x1 and y0 <= c.y <= y1, f"core ({c.x}, {c.y}) is outside the rectangle {self.rect}"
        self.cb = {name: cb_base + i for name, i in CB_IDS.items()}
        self.done_words = 1 + max([u.done for u in self.units if not u.dst_act], default=0)
        assert self.done_words <= 15, "the go tensor has 16 words a core"

        # the weight stream: per bank the blocks of its units, in unit order
        self.offset_of, seen = [], [0] * L.BANKS
        for i, u in enumerate(self.units):
            assert u.wt.shape[1] == L.TILE * u.ct, (u.name, tuple(u.wt.shape), u.ct)
            b = self.bank_of[i]
            self.offset_of.append(seen[b] * self.tile_bytes)
            seen[b] += u.k_tiles * u.ct * terms
        self.stream_rows = max(1, max(seen)) * L.TILE
        self.weight_bytes = sum(u.k_tiles * u.ct for u in self.units) * terms * self.tile_bytes
        memory_config = stream_memory_config(self.stream_rows)
        kwargs = dict(dtype=DTYPES[fmt], layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)
        if weights is not None:
            assert cache is None and weights.shape[-2] >= self.stream_rows, (tuple(weights.shape), self.stream_rows)
            self.weights = weights
        elif cache is not None and cached_stream_exists(cache):
            self.weights = ttnn.as_tensor(None, cache_file_name=cache, **kwargs)
        else:
            stream = self.host_stream()
            assert stream.shape[0] == self.stream_rows
            if cache is None:
                self.weights = ttnn.from_torch(stream[None, None], **kwargs)
            else:
                self.weights = ttnn.as_tensor(stream[None, None], cache_file_name=cache, **kwargs)

        # the hub's buffers: tensors of this program, or addresses the caller gives
        if hub_buffers is None:
            z = lambda n, w: torch.zeros(n, w)
            self.x = hub_tensor(device, self.hub, z(x_tiles, 2 * 16 * rows), ttnn.bfloat16)
            self.out = hub_tensor(device, self.hub, z(out_tiles, 2 * 16 * rows), ttnn.bfloat16)
            self.stat = hub_tensor(device, self.hub, z(1, 2 * 16 * rows), ttnn.float32)
            self.meta = hub_tensor(device, self.hub, torch.zeros(1, L.META_IN_WORDS, dtype=torch.int32), ttnn.uint16)
            self.status = hub_tensor(device, self.hub, torch.zeros(1, STATUS_WORDS, dtype=torch.int32), ttnn.uint32)
            self.hub_addr = {name: getattr(self, name).buffer_address() for name in HUB_BUFFERS}
        else:
            assert set(hub_buffers) == set(HUB_BUFFERS), sorted(hub_buffers)
            assert all(a % 16 == 0 for a in hub_buffers.values()), "NOC writes: 16-byte aligned addresses"
            self.x = self.out = self.stat = self.meta = self.status = None
            self.hub_addr = dict(hub_buffers)
        # the go word: 16 words per core of the rectangle at one L1 address, zero; the hub increments word 0 everywhere
        n_cores = (x1 - x0 + 1) * (y1 - y0 + 1)
        self.go = ttnn.from_torch(
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
        if hub_buffers is None:
            self.set_params()
        self._program = None

    def hub_buffer_bytes(self):
        """Bytes of each hub buffer (what `hub_buffers` addresses must have room for)."""
        return {
            "x": self.x_tiles * self.row_tile,
            "out": self.out_tiles * self.row_tile,
            "stat": 2 * self.row_tile,
            "meta": 2 * L.META_IN_WORDS,
            "status": 4 * STATUS_WORDS,
        }

    def done_address(self, group=0):
        """L1 address, on the hub core, of the word that counts the units of done group `group` that have written
        their tiles into the output buffer; done_units(group) is its final value."""
        return self.go.buffer_address() + 4 * (1 + group)

    def done_units(self, group=0):
        return sum(1 for u in self.units if not u.dst_act and u.done == group)

    def host_stream(self):
        """[rows, 8 * 32]: per bank the blocks of its units (K-major tiles, one block per term), padded to the
        longest bank."""
        return unit_stream(self.units, self.fmt, self.terms, self.pre_rounded, self.banks)

    def stored(self, wt):
        """wt [32 a, 32 b] as this program's stream holds it (layout.stored_stream)."""
        return L.stored_stream(wt.float().contiguous(), self.fmt, self.terms, self.pre_rounded)

    def schedule(self, unit):
        """The unit's pass schedule: None = custom_mm, or a key of FIDELITY."""
        if unit.fidelity is not None:
            return unit.fidelity
        if self.fidelity == "custom_mm":
            return None
        return recommended_fidelity(self.fmt, unit.left7) if self.fidelity == "recommended" else self.fidelity

    # ------------------------------------------------------------------------------------------------------------
    def _noc(self, c):
        return self.placement.noc[(c.x, c.y)]

    def _cb(self, index, cores, dtype, page, pages, row_tile=False):
        kwargs = dict(buffer_index=index, data_format=dtype, page_size=page)
        if row_tile:
            kwargs["tile"] = ttnn.TileDescriptor(ttnn.Tile([self.rows, L.TILE]))
        return ttnn.CBDescriptor(
            total_size=page * pages, core_ranges=cores, format_descriptors=[ttnn.CBFormatDescriptor(**kwargs)]
        )

    def descriptor(self):
        """The ProgramDescriptor and its io tensors (the output buffer last)."""
        pl = self.placement
        hub_noc = self._noc(self.hub)
        x0, y0, x1, y1 = self.rect
        lo, hi = pl.noc[(x0, y0)], pl.noc[(x1, y1)]
        mc = [lo.x, lo.y, hi.x, hi.y, (x1 - x0 + 1) * (y1 - y0 + 1) - 1]
        h_crs = _crs([self.hub])
        gr, gx = self.ring
        CB_X, CB_META, CB_ACT, CB_CTL, CB_W, CB_SC, CB_OUT, CB_STAT = (self.cb[name] for name in CB_IDS)

        cbs = [  # the whole-grid buffers first: the same address on every core
            self._cb(CB_X, self.grid_crs, ttnn.bfloat16, self.row_tile, self.x_tiles, row_tile=True),
            self._cb(CB_META, self.grid_crs, ttnn.uint32, META_PAGE, 1),
        ]
        if self.act_tiles:
            cbs.append(self._cb(CB_ACT, self.grid_crs, ttnn.bfloat16, self.row_tile, self.act_tiles, row_tile=True))

        dm = ttnn.DataMovementConfigDescriptor
        reader_cfg = dm(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0)
        writer_cfg = dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1)
        compute_cfg = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, math_approx_mode=False, dst_full_sync_en=True
        )
        modes = [ttnn.UnpackToDestMode.Default] * 64
        modes[CB_SC] = ttnn.UnpackToDestMode.UnpackToDestFp32
        compute_cfg.unpack_to_dest_mode = modes

        def kernel(path, cores, ct, rt, config):
            return ttnn.KernelDescriptor(
                kernel_source=str(path),
                core_ranges=cores,
                compile_time_args=ct,
                runtime_args=rt,
                config=config,
                defines=self.defines,
            )

        status_addr, out_addr, stat_addr = (self.hub_addr[name] for name in ("status", "out", "stat"))
        hub_rt = ttnn.RuntimeArgs()
        go_addr = self.go.buffer_address()
        hub_rt[self.hub.x][self.hub.y] = [self.hub_addr["x"], self.hub_addr["meta"], status_addr, go_addr]
        hub_ct = [CB_X, CB_META, self.x_tiles * self.row_tile] + mc + [self.done_words]
        kernels = [kernel(KERNEL_DIR / "proj_hub.cpp", h_crs, hub_ct, hub_rt, writer_cfg)]

        kinds = {}
        for i, u in enumerate(self.units):
            kinds.setdefault(u.kind() + (self.schedule(u),), []).append(i)
        vcs = self._vcs()
        for (k_tiles, ct, fin, rnd, src_act, dst_act, param, schedule), members in kinds.items():
            cores = _crs([self.core_of[i] for i in members])
            kb = choose_kb(k_tiles, ct, self.tile_bytes, self.max_block)
            blocks_per_term = k_tiles // kb
            n_blocks = blocks_per_term * self.terms
            block_tiles = kb * ct
            block_bytes = block_tiles * self.tile_bytes
            cb_in = CB_ACT if src_act else CB_X
            in_push = self.act_tiles if src_act else self.x_tiles
            has_const = L.fin_has_const(fin)
            cbs += [
                self._cb(CB_W, cores, DTYPES[self.fmt], self.tile_bytes, gr * block_tiles),
                self._cb(CB_OUT, cores, ttnn.bfloat16, self.row_tile, ct, row_tile=True),
            ]
            if has_const:
                cbs.append(self._cb(CB_SC, cores, ttnn.float32, 4096, 1))
            if fin == L.FIN_STAT:
                cbs.append(self._cb(CB_STAT, cores, ttnn.float32, 2 * self.row_tile, 1, row_tile=True))
            r_rt, w_rt, c_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
            for i in members:
                u, c = self.units[i], self.core_of[i]
                r_rt[c.x][c.y] = [self.weights.buffer_address(), self.bank_of[i], vcs[i], self.offset_of[i]]
                w_rt[c.x][c.y] = [
                    u.out_tile,
                    out_addr,
                    status_addr,
                    u.wait[0],
                    u.wait[1],
                    u.ctr,
                    stat_addr,
                    go_addr,
                    1 + u.done,
                ]
                c_rt[c.x][c.y] = [u.in_first, u.aux]
            reader_ct = [CB_W, n_blocks, block_tiles, block_bytes, packets(block_bytes), gr, gx]
            writer_ct = [cb_in, CB_META, CB_CTL, CB_SC, CB_OUT, CB_STAT, CB_ACT, in_push, n_blocks, fin, ct]
            writer_ct += [int(dst_act), hub_noc.x, hub_noc.y] + mc
            compute_ct = [cb_in, CB_W, CB_OUT, CB_SC, CB_STAT, CB_X, in_push, kb, blocks_per_term, ct, fin, rnd, param]
            kernels += [
                kernel(KERNEL_DIR / "proj_reader.cpp", cores, reader_ct, r_rt, reader_cfg),
                kernel(KERNEL_DIR / "proj_writer.cpp", cores, writer_ct, w_rt, writer_cfg),
            ]
            if schedule is None:
                kernels.append(kernel(KERNEL_DIR / "proj_compute.cpp", cores, compute_ct, c_rt, compute_cfg))
            else:  # the same unit through hifi_mm: split accumulators, FP32 merge
                passes, step = FIDELITY[schedule]
                k = kernel(KERNEL_DIR / "proj_compute_hifi.cpp", cores, compute_ct, c_rt, compute_cfg)
                k.defines = self.defines + [
                    ("HY4_FID_PASSES", str(passes)),
                    ("HY4_FID_STEP", str(step)),
                    ("HY4_FID_SPLIT", "2"),
                ]
                kernels.append(k)
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        own = [t for t in (self.x, self.meta, self.status, self.stat) if t is not None]
        return program, [self.weights] + own + [self.go] + ([self.out] if self.out is not None else [])

    def reset(self):
        """Zero the go word on every core: after a run in which a unit gave up before the hub's release arrived."""
        n_cores = (self.rect[2] - self.rect[0] + 1) * (self.rect[3] - self.rect[1] + 1)
        host = ttnn.from_torch(
            torch.zeros(n_cores, 16, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        ttnn.copy_host_to_device_tensor(host, self.go)

    def _vcs(self):
        """A NOC0 virtual channel per unit, distinct from earlier units on the same row where possible."""
        vcs = []
        for i, core in enumerate(self.core_of):
            vc = (self.bank_of[i] + i // len(self.banks)) & 0x3
            for j in range(i):
                if self.core_of[j].y == core.y and vcs[j] == vc:
                    vc = (vc + 1) & 0x3
            vcs.append(vc)
        return vcs

    # ------------------------------------------------------------------------------------------------------------
    def set_x(self, x):
        """Host -> the hub's input buffer: [8, 32 * x_tiles]."""
        assert self.x is not None, "the hub buffers are the caller's (hub_buffers)"
        assert tuple(x.shape) == (self.rows, L.TILE * self.x_tiles), tuple(x.shape)
        host = ttnn.from_torch(L.to_row_tiles(x.float()), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.copy_host_to_device_tensor(host, self.x)

    def set_params(self, scale=None, lanes=None):
        """Host -> the hub's parameter page (layout.meta_words)."""
        assert self.meta is not None, "the hub buffers are the caller's (hub_buffers)"
        host = ttnn.from_torch(
            L.meta_words(scale, lanes, rows=self.rows), dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        ttnn.copy_host_to_device_tensor(host, self.meta)

    def __call__(self):
        """Run on what the hub's buffers hold. On the mock device the program is compiled and checked, not run."""
        if self._program is None:
            self._program = self.descriptor()
        program, io = self._program
        if MOCK:
            ttnn.experimental.prepare_generic_op(io, program)
        else:
            ttnn.generic_op(io, program)
        return self.out

    def read(self):
        """(output buffer [8, 32 * out_tiles], statistic tile [8, 32] fp32, status words)."""
        assert self.out is not None, "the hub buffers are the caller's (hub_buffers)"
        out = L.from_row_tiles(ttnn.to_torch(self.out).float())
        stat = L.from_row_tiles(ttnn.to_torch(self.stat).float())
        status = [int(v) & 0xFFFFFFFF for v in ttnn.to_torch(self.status).flatten().tolist()]
        return out, stat, status

    def out_units_done(self, group=None):
        """Word 1 + group of the hub's go tensor: how many units of that done group have written their tiles into
        the output buffer (what a cross-chip sender waits for), and how many units do. group=None: all groups."""
        go = ttnn.to_torch(self.go).to(torch.int64).reshape(-1, 16)
        hub = (self.hub.y - self.rect[1]) * (self.rect[2] - self.rect[0] + 1) + (self.hub.x - self.rect[0])
        groups = range(self.done_words) if group is None else [group]
        return (
            sum(int(go[hub, 1 + g]) & 0xFFFFFFFF for g in groups),
            sum(self.done_units(g) for g in groups),
        )

    def check_status(self, status):
        assert status[1:6] == [0, 0, 0, 0, 0], f"a core timed out or could not send: {[hex(v) for v in status[:6]]}"

    def unit_cycles(self, status):
        """With the HY4_TIMING define: per unit, cycles from the hub's release to its last hand-over."""
        out = []
        for c in self.core_of:
            n = self._noc(c)
            out.append(status[16 + 2 * ((n.y - 2) * 16 + n.x) + 1])
        return out

    def describe(self):
        return [
            (u.name, self.bank_of[i], (self.core_of[i].x, self.core_of[i].y), u.k_tiles * u.ct * self.terms)
            for i, u in enumerate(self.units)
        ]


# ----------------------------------------------------------------------------------------------------------------
# The three groups of a chip. `w` is layout.chip_weights(...).

PRE_QA, PRE_POOL, PRE_GATE = 0, 2, 3
PRE_GATE_TILES = L.HEADS_PER_CHIP * L.V // L.TILE  # 16
PRE_IDX = PRE_GATE + PRE_GATE_TILES  # 19
PRE_HIDDEN_TILES = L.HIDDEN // L.TILE
PRE_X_TILES = PRE_HIDDEN_TILES + 2  # x | cos | sin
IDX_TILES = L.IDX_DIM // L.TILE  # 4


def f32_bits(v):
    import struct

    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def pre_units(w, gate_ct=2, gate_done=0):
    """The pre group: every unit takes the broadcast x.

    in    input buffer: x [8, 6144] as 192 BF16 row tiles, NOT normalised (what the iHC site's pre-mix leaves: its
          ``y``, fused/ihc_site/op.py), then one cos and one sin tile of the rows' positions (layout.rope_tile; used on
          full layers); parameter page: scale[8] = 1 / RMS per row (its ``stat``). The input norm's weight is
          folded into the weights.
    out   output buffer, tiles: 0..1 q_a's columns [64 c, 64 c + 64), rounded to 7 significant bits (the left
          operand of q_b) | 2 the chip's pool tile (layout.pool: a column of the raw latent row, of the raw index
          key, or the head weights) | 3..18 the gate of heads 2 c, 2 c + 1 after the sigmoid | 19..22 on full layers
          index head c's query for the 8 rows, [8, 128] in the cache's frame (RoPE on the leading 64 dims, Hadamard
          basis), without q_a's norm scalar and without the head weight (both are positive-or-signed per-row
          factors the scan takes as multipliers: fused/indexer_scan/CONTRACT.md "fold = False").
          statistic tile (FP32): the lane sums of q_a's squares (layout.stat_tile_lanes).
    Tiles 0..2, 19..22 and the statistic tile are the chip's payload of Xa; the gate stays on the chip (post group).
    """
    units = [Unit("q_a", w["q_a"], 2, fin=L.FIN_STAT, rnd=RND_7, out_tile=PRE_QA)]
    if "pool" in w:
        units.append(Unit("pool", w["pool"], 1, fin=L.FIN_SCALE, out_tile=PRE_POOL))
    for n in range(0, PRE_GATE_TILES, gate_ct):
        wt = w["gate"][:, L.TILE * n : L.TILE * (n + gate_ct)]
        units.append(Unit(f"gate{n}", wt, gate_ct, fin=L.FIN_SIGMOID, out_tile=PRE_GATE + n, done=gate_done))
    if "idx_q" in w:
        assert w["idx_q"].shape[0] == L.HIDDEN, "the pre group takes the fused index-query weight"
        sign, magnitude = L.hadamard_sign()
        rope, rest = w["idx_q"][:, : L.ROPE], w["idx_q"][:, L.ROPE :]
        units += [
            Unit("idx.rope", rope, 2, fin=L.FIN_SCALE_ROPE, rnd=RND_7, dst_act=True, out_tile=0, aux=PRE_HIDDEN_TILES),
            Unit("idx.pass", rest, 2, fin=L.FIN_SCALE, rnd=RND_7, dst_act=True, out_tile=2),
            # the Hadamard basis: +-1 weights (exact in any format), the magnitude as the finishing's constant
            Unit(
                "idx.hadamard",
                sign,
                IDX_TILES,
                fin=L.FIN_CONST,
                src_act=True,
                out_tile=PRE_IDX,
                wait=(0, 2),
                param=f32_bits(magnitude),
                left7=True,
            ),
        ]
    return units


def pre_out_tiles(w):
    return PRE_IDX + (IDX_TILES if "idx_q" in w else 0)


Q_A_TILES = L.Q_LORA // L.TILE  # 64
QUERY_X_TILES = Q_A_TILES + 2  # q_a | cos | sin
NOPE_TILES, ROPE_TILES, ABS_TILES = L.NOPE // L.TILE, L.ROPE // L.TILE, L.KV_LORA // L.TILE  # 6, 2, 16
LAT_TILES = ABS_TILES + ROPE_TILES  # 18


def query_units(w, nope_ct=3, absorb_ct=4):
    """The query group of the chip's heads.

    in    input buffer: q_a [8, 2048] whole as 64 BF16 row tiles (the gather of every chip's pre tiles 0..1: Xa),
          then one cos and one sin tile of the rows' positions (layout.rope_tile of the interleaved tables);
          parameter page: lanes[8][8] = the lane sums of q_a's squares summed over the 32 chips (the statistic
          tiles of Xa; the units take rsqrt(sum / 2048 + 1e-6) of them) and scale[8] = 1.
    out   output buffer: head j of the chip at tiles 18 j .. 18 j + 17 = its absorbed query for the 8 rows,
          [8, 576] = (q_nope W_k | q_rope with RoPE), BF16, the cache's frame, not scaled by the softmax scale:
          row u of it is row (2 c + j) % 8 of page (u, c // 4) of the sparse attention's q
          (fused/sparse_mla/op.py; layout.sparse_mla_pages builds the pages from the 32 chips' buffers).
    """
    units, out_units = [], []
    for j, (wq, wk) in enumerate(zip(w["q_b"], w["w_k"])):
        for n in range(0, NOPE_TILES, nope_ct):
            units.append(
                Unit(
                    f"h{j}.nope{n}",
                    wq[:, L.TILE * n : L.TILE * (n + nope_ct)],
                    nope_ct,
                    fin=L.FIN_NORM,
                    rnd=RND_7,
                    dst_act=True,
                    out_tile=NOPE_TILES * j + n,
                    ctr=j,
                    param=L.Q_LORA,
                    left7=True,
                )
            )
        units.append(
            Unit(
                f"h{j}.rope",
                wq[:, L.NOPE :],
                ROPE_TILES,
                fin=L.FIN_NORM_ROPE,
                out_tile=LAT_TILES * j + ABS_TILES,
                aux=Q_A_TILES,
                param=L.Q_LORA,
                left7=True,
            )
        )
        for n in range(0, ABS_TILES, absorb_ct):
            out_units.append(
                Unit(
                    f"h{j}.absorb{n}",
                    wk[:, L.TILE * n : L.TILE * (n + absorb_ct)],
                    absorb_ct,
                    fin=L.FIN_ROUND,
                    src_act=True,
                    in_first=NOPE_TILES * j,
                    out_tile=LAT_TILES * j + n,
                    wait=(j, NOPE_TILES // nope_ct),
                    left7=True,
                )
            )
    return units + out_units


POST_CTX_TILES = L.KV_LORA // L.TILE  # 16 per head
POST_X_TILES = L.HEADS_PER_CHIP * POST_CTX_TILES + PRE_GATE_TILES  # ctx of the 2 heads | gate
V_TILES = L.V // L.TILE  # 8 per head


def post_units(w, value_ct=4, o_ct=4):
    """The post group of the chip's heads.

    in    input buffer: ctx of head j at tiles 16 j .. 16 j + 15 ([8, 512] BF16: the merged, normalised attention
          output of that head in latent space, reference/attention_decode.py absorbed_stages "ctx": what the
          log-sum-exp merge leaves on the chip that owns the head), then the 16 gate tiles the chip's pre group
          produced (its output tiles 3..18).
    out   output buffer: the chip's partial of the sublayer output, [8, 6144] as 192 BF16 row tiles, tile n = hidden
          columns [32 n, 32 n + 32): the layout of the MoE program's output buffer. The sum of the 32 chips'
          partials is the attention sublayer's output (X5 is a reduce, like the MoE's).
    """
    units, n_value = [], 0
    for j, wv in enumerate(w["w_v"]):
        for n in range(0, V_TILES, value_ct):
            units.append(
                Unit(
                    f"h{j}.value{n}",
                    wv[:, L.TILE * n : L.TILE * (n + value_ct)],
                    value_ct,
                    fin=L.FIN_GATE,
                    rnd=RND_7,
                    in_first=POST_CTX_TILES * j,
                    dst_act=True,
                    out_tile=V_TILES * j + n,
                    ctr=0,
                    aux=L.HEADS_PER_CHIP * POST_CTX_TILES + V_TILES * j + n,
                )
            )
            n_value += 1
    for n in range(0, L.HIDDEN // L.TILE, o_ct):
        units.append(
            Unit(
                f"o{n}",
                w["o"][:, L.TILE * n : L.TILE * (n + o_ct)],
                o_ct,
                fin=L.FIN_ROUND,
                src_act=True,
                out_tile=n,
                wait=(0, n_value),
                left7=True,
            )
        )
    return units


def pre_program(device, w, gate_ct=2, gate_done=0, **kw):
    """gate_done: the done group of the gate units (default 0: one count for all output units). With 1, done group 0
    is the units whose tiles are Xa's payload (q_a, the pool tile, the index query) and group 1 the gate units, which
    stay on the chip for the post group."""
    act_tiles = IDX_TILES if "idx_q" in w else 0
    return ProjProgram(device, pre_units(w, gate_ct, gate_done), PRE_X_TILES, pre_out_tiles(w), act_tiles, **kw)


def combine(programs):
    """Several ProjPrograms (disjoint rectangles and circular-buffer ids: rect, cb_base) as ONE program:
    (ProgramDescriptor, io tensors)."""
    kernels, cbs, io = [], [], []
    ids = [set(p.cb.values()) for p in programs]
    assert all(ids[i].isdisjoint(ids[j]) for i in range(len(ids)) for j in range(i)), "cb_base: the ids overlap"
    for p in programs:
        program, tensors = p.descriptor()
        kernels += list(program.kernels)
        cbs += list(program.cbs)
        io += [t for t in tensors if not any(t is u for u in io)]
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs), io


def query_program(device, w, nope_ct=3, absorb_ct=4, **kw):
    out_tiles = L.HEADS_PER_CHIP * LAT_TILES
    act_tiles = L.HEADS_PER_CHIP * NOPE_TILES
    return ProjProgram(device, query_units(w, nope_ct, absorb_ct), QUERY_X_TILES, out_tiles, act_tiles, **kw)


def post_program(device, w, value_ct=4, o_ct=4, **kw):
    act_tiles = L.HEADS_PER_CHIP * V_TILES
    return ProjProgram(device, post_units(w, value_ct, o_ct), POST_X_TILES, L.HIDDEN // L.TILE, act_tiles, **kw)
