# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Regions-sourced streaming decode (``GLM53_REGIONS_DECODE=stream``): the layout math and the planner, no ttnn.

The regions prefill path (``tt/moe.py`` ``_build_dispatch_path``) keeps every local expert as three DRAM-interleaved
TILE tensors: ``gate[e]`` / ``up[e]`` ``[K, I_local]`` and ``down[e]`` ``[I_local, H]``. A DRAM-interleaved tensor
puts tile page ``p`` in bank ``p % banks`` at byte offset ``(p // banks) * tile_bytes`` (the allocator / address
generator rule, ``interleaved_place``), and a ``[K, I_local]`` tile tensor numbers its tiles row-major
(``tile_page``: ``p = k * nt + n`` with ``nt = I_local / 32``). Whenever ``nt`` is a multiple of the bank count the
bank of tile ``(k, n)`` is ``n % banks`` for every ``k``: tile column ``n`` lives entirely in one bank, at the K
stride ``ppb = nt / banks`` tiles. On the 8x4 (``I_local = 256``, 8 banks, ``ppb = 1``) that is exactly the
bank-local, K-contiguous column the ``expert_stream`` reader streams from its stacked copy
(``op.gate_up_stream_layout``): the stacked copy duplicates bytes that already sit in the right banks, in the right
order, under a different base address per expert and projection. The down tensor ``[I_local, H]`` is bank-local
too when ``H / 32`` is a multiple of the bank count (bank ``b`` holds the output tile columns ``h = b + banks * m``),
with the ``cols`` columns of one core contiguous per K tile (k-major instead of the stacked layout's column-major).

So the kernels can read the regions lists directly (``REGIONS_SRC`` variants of the stream / dispatch kernels, the
per-expert buffer addresses as common runtime args) and the second copy is not needed: ``one_copy_bytes`` is the
DRAM it gives back (9.41 GiB per chip at the 8x4 shapes in bfp8), which is what decided between the 70 ms step with
the stacked copy and the 152 ms step on the regions kernel at >= 512 K x 4 users (``doc/mesh/regions_decode.md``).

Cost per expert and core (``packets_per_expert`` vs ``stacked_packets_per_expert``): at ``ppb = 1`` the gate|up
stream is byte-for-byte the stacked one (the same 16 x 8.7 KB packets per column); at ``ppb > 1`` (1024-wide slices
on a (2,4) block) a column's K tiles are ``ppb`` tiles apart, so the reader issues single-tile packets (``kt`` per
column against ``kt / 8``); the down block is always ``seg_tiles`` packets of ``cols`` tiles. ``bank_local`` is
false when either divisibility fails (12-bank Wormhole at these widths): the planner then refuses and the model keeps
today's paths. ``tests/test_regions_decode_host.py`` cross-checks the kernels' address walk against the page map.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass

from models.demos.glm53_flash.custom_ops.expert_stream.plan import TILE, TILE_BYTES, down_readers_for, fmt

MAX_PACKET_BYTES = 16384  # Blackhole NOC max burst (the stream op's `_packet_bytes` bound)
DRAM_ALIGN_MAX = (
    64  # the DRAM page alignment is 32 B (Wormhole) / 64 B (Blackhole): tiles of 64 B multiples stay unpadded
)
MAX_RUNTIME_ARGS = 341  # tt-metalium kernel_types.hpp: the per-kernel (unique or common) runtime-arg limit
REGIONS_DECODE_ENV = "GLM53_REGIONS_DECODE"
REGIONS_DECODE_MODES = ("off", "stream", "regions")


def regions_decode_mode(env=None) -> str:
    """``GLM53_REGIONS_DECODE``: ``stream`` (default since 2026-10-27: the stream / dispatch ops read the regions lists, one
    expert copy serves prefill and decode, bit-identical to the stacked copy), ``off`` (the earlier rule -- the stacked
    stream copy when kept, else the regions kernel),
    ``regions`` (every shape through the regions kernel even when the stacked copy is present: the A/B reference)."""
    env = os.environ if env is None else env
    mode = env.get(REGIONS_DECODE_ENV, "stream")
    if mode not in REGIONS_DECODE_MODES:
        raise ValueError(f"{REGIONS_DECODE_ENV}={mode!r}: expected one of {REGIONS_DECODE_MODES}")
    return mode


def resolve_keep_stacked(fits: bool, mode: str) -> bool:
    """The mesh model's auto rule: keep the stacked decode copy when the planner says both copies fit, except under
    ``GLM53_REGIONS_DECODE=stream`` where the regions lists serve decode and no second copy is ever built."""
    return bool(fits) and mode != "stream"


# ---------------------------------------------------------------------------------------------------- page map
def interleaved_place(page: int, banks: int, page_bytes: int) -> tuple[int, int]:
    """(bank, byte offset inside the bank relative to the buffer address) of page ``page`` of a DRAM-interleaved
    buffer: the allocator's round-robin (``Buffer::page_address``: ``page // banks`` pages into bank ``page % banks``)
    and the device address generator (``get_bank_offset_index`` / ``get_bank_index``)."""
    return page % banks, (page // banks) * page_bytes


def tile_page(k: int, n: int, nt: int) -> int:
    """Row-major tile numbering of a TILE tensor with ``nt`` tile columns: tile (k, n) is page ``k * nt + n``."""
    return k * nt + n


# ---------------------------------------------------------------------------------------------------- planner
@dataclass(frozen=True)
class RegionsStreamPlan:
    """What the REGIONS_SRC kernels need to walk the regions lists bank-locally (mirrors of the kernel arithmetic:
    ``gate_col_packets`` / ``down_block_packets`` enumerate the exact (offset, bytes) reads a core issues)."""

    banks: int
    kt: int  # K tiles (4096 / 32)
    nt: int  # intermediate tile columns of the chip's slice (I_local / 32)
    ppb: int  # column pairs per bank = nt / banks (0 when not bank-local)
    seg_tiles: int  # down K tiles = nt
    ht: int  # output tile columns (H / 32)
    C: int  # output columns per bank = ht / banks (0 when not bank-local)
    down_readers: int
    cols: int  # output columns per down core = C / readers
    w13_tile: int
    w2_tile: int
    gate_packet_tiles: int
    gate_packet_bytes: int
    gate_pages_per_col: int
    gate_src_step: int  # bytes between consecutive packets of one column
    down_packet_tiles: int  # = cols: one K tile of the core's columns
    down_packet_bytes: int
    down_pages_per_expert: int  # = seg_tiles
    down_src_step: int  # bytes between consecutive K tiles of the core's columns = C * w2_tile
    bank_local: bool
    reason: str

    # ---- kernel mirrors (byte offsets relative to the expert tensor's buffer address, inside the bank)
    def gate_tile_offset(self, k: int, q: int) -> int:
        """Offset of K tile ``k`` of this bank's column ``q`` (global column ``bank + banks * q``) in a gate / up
        tensor: page ``k * nt + bank + banks * q`` -> ``(k * ppb + q) * w13_tile``."""
        return (k * self.ppb + q) * self.w13_tile

    def gate_col_packets(self, q: int) -> list[tuple[int, int]]:
        """The reader's packets for column ``q``: ``gate_pages_per_col`` x (offset, bytes), as reader_gate_up.cpp
        REGIONS_SRC issues them (start ``q * w13_tile``, step ``gate_src_step``)."""
        return [
            (q * self.w13_tile + p * self.gate_src_step, self.gate_packet_bytes) for p in range(self.gate_pages_per_col)
        ]

    def down_tile_offset(self, i: int, m: int) -> int:
        """Offset of K tile ``i`` of this bank's output column ``m`` (global ``bank + banks * m``) in a down tensor:
        page ``i * ht + bank + banks * m`` -> ``(i * C + m) * w2_tile``."""
        return (i * self.C + m) * self.w2_tile

    def down_block_packets(self, c0: int) -> list[tuple[int, int]]:
        """The down reader's packets for one expert on a core owning columns ``[c0, c0 + cols)`` (reader_down.cpp
        REGIONS_SRC: start ``c0 * w2_tile``, step ``down_src_step``, ``seg_tiles`` packets of ``cols`` tiles)."""
        return [
            (c0 * self.w2_tile + i * self.down_src_step, self.down_packet_bytes)
            for i in range(self.down_pages_per_expert)
        ]

    def act_tile_column(self, bank: int, q: int) -> int:
        """The act tile column (global intermediate tile) of (bank, pair q): writer_gate_up.cpp REGIONS_SRC."""
        return bank + self.banks * q

    def out_tile_column(self, col0: int, c: int) -> int:
        """The output tile column of a down core's ``c``-th column from today's ``col0 = bank * C + r * cols``:
        writer_down.cpp REGIONS_SRC (``regions_src.hpp::regions_out_col``)."""
        bank, m0 = divmod(col0, self.C)
        return bank + self.banks * (m0 + c)

    @staticmethod
    def stacked_act_tile_column(bank: int, q: int, ppb: int) -> int:
        return bank * ppb + q

    @staticmethod
    def stacked_out_tile_column(col0: int, c: int) -> int:
        return col0 + c

    def packets_per_expert(self) -> dict[str, int]:
        """NOC packets one core issues per active expert under the regions-sourced layout."""
        return {"gate_up": 2 * self.ppb * self.gate_pages_per_col, "down": self.down_pages_per_expert}

    def common_args(self, E_local: int) -> dict[str, int]:
        """Common runtime-arg words per kernel (the per-expert buffer addresses): must stay under the limit."""
        return {"gate_up": 2 * E_local, "down": E_local, "limit": MAX_RUNTIME_ARGS}

    def as_dict(self) -> dict:
        return asdict(self)


def _largest_packet_tiles(tiles: int, tile_bytes: int) -> int:
    return max(d for d in range(1, tiles + 1) if tiles % d == 0 and d * tile_bytes <= MAX_PACKET_BYTES)


def stacked_packets_per_expert(
    K: int, I: int, H: int, banks: int = 8, w13="bfp8", w2="bfp8", down_readers=None
) -> dict:
    """The stacked layout's packets per active expert and core (op.py ``_packet_bytes``: the largest divisor of the
    column / segment within the 16 KB burst; the down block of ``cols * seg_tiles`` tiles in segment packets)."""
    kt, nt, ht = K // TILE, I // TILE, H // TILE
    t13, t2 = TILE_BYTES[fmt(w13)], TILE_BYTES[fmt(w2)]
    ppb = nt // banks
    gate_pages = kt // _largest_packet_tiles(kt, t13)
    seg = nt
    C = ht // banks
    readers = down_readers or down_readers_for(I, H, banks)
    cols = C // readers
    down_pages = (cols * seg) // _largest_packet_tiles(seg, t2)
    return {"gate_up": 2 * ppb * gate_pages, "down": down_pages, "cols": cols, "readers": readers}


def plan_regions_stream(
    K: int,
    I: int,
    H: int | None,
    banks: int = 8,
    w13="bfp8",
    w2="bfp8",
    down_readers: int | None = None,
    packet_tiles: int | None = None,
) -> RegionsStreamPlan:
    """The regions-sourced stream plan for one chip's expert slice (``K`` = hidden in, ``I`` = the chip's
    intermediate width, ``H`` = hidden out; ``None`` = the gate|up side only, the down fields stay 0) on ``banks``
    DRAM banks. ``down_readers``: the down op's readers per bank (None = ``down_readers_for``); ``packet_tiles``:
    the gate|up packet (None = the largest within the burst; forced to 1 tile at ``ppb > 1`` where a column's K
    tiles are not adjacent)."""
    kt, nt = K // TILE, I // TILE
    ht = H // TILE if H is not None else 0
    t13, t2 = TILE_BYTES[fmt(w13)], TILE_BYTES[fmt(w2)]
    reasons = []
    if nt % banks:
        reasons.append(
            f"I/32 = {nt} intermediate tile columns do not split over {banks} banks (the stacked layout refuses this shape too)"
        )
    if ht % banks:
        reasons.append(
            f"H/32 = {ht} output tile columns do not split over {banks} banks: a down column's K tiles rotate over the banks"
        )
    if t13 % DRAM_ALIGN_MAX or t2 % DRAM_ALIGN_MAX:
        reasons.append(
            f"tile bytes {t13} / {t2} are not multiples of {DRAM_ALIGN_MAX}: the interleaved pages would be padded"
        )
    ppb = nt // banks if nt % banks == 0 else 0
    C = ht // banks if ht % banks == 0 else 0
    readers = down_readers or (down_readers_for(I, H, banks) if C else 0)
    cols = C // readers if C and readers and C % readers == 0 else 0
    if C and cols == 0:
        reasons.append(f"{C} output columns per bank do not split over {readers} down readers")
    if cols > 8:
        reasons.append(f"{cols} output columns per down core exceed the 8 fp32 DST tiles")
    if ppb == 1:
        g_tiles = packet_tiles or _largest_packet_tiles(kt, t13)
        if kt % g_tiles or g_tiles * t13 > MAX_PACKET_BYTES:
            reasons.append(
                f"{g_tiles} tiles per gate|up packet do not divide K/32 = {kt} within the {MAX_PACKET_BYTES} B burst"
            )
            g_tiles = 1
        g_step = g_tiles * t13
    else:
        if packet_tiles not in (None, 1):
            reasons.append(f"at ppb = {ppb} a column's K tiles are {ppb} tiles apart: packets must be one tile")
        g_tiles = 1
        g_step = ppb * t13
    d_bytes = cols * t2
    if d_bytes > MAX_PACKET_BYTES:
        reasons.append(f"a down packet of {cols} tiles ({d_bytes} B) exceeds the burst")
    return RegionsStreamPlan(
        banks=banks,
        kt=kt,
        nt=nt,
        ppb=ppb,
        seg_tiles=nt,
        ht=ht,
        C=C,
        down_readers=readers,
        cols=cols,
        w13_tile=t13,
        w2_tile=t2,
        gate_packet_tiles=g_tiles,
        gate_packet_bytes=g_tiles * t13,
        gate_pages_per_col=kt // g_tiles,
        gate_src_step=g_step,
        down_packet_tiles=cols,
        down_packet_bytes=d_bytes,
        down_pages_per_expert=nt,
        down_src_step=C * t2,
        bank_local=not reasons,
        reason="; ".join(reasons),
    )


def regions_stream_feasible(K: int, I: int, H: int, banks: int = 8, w13="bfp8", w2="bfp8") -> tuple[bool, str]:
    """(ok, reason) of the regions-sourced stream at one chip's shapes (readers-independent: the bank locality)."""
    p = plan_regions_stream(K, I, H, banks=banks, w13=w13, w2=w2)
    return p.bank_local, p.reason


# ---------------------------------------------------------------------------------------------------- bytes
def expert_slice_bytes(K: int, I: int, H: int, w13="bfp8", w2="bfp8") -> int:
    """Bytes of one expert's slice: gate|up [K, 2 I] + down [I, H] in the given formats (tile-exact)."""
    t13, t2 = TILE_BYTES[fmt(w13)], TILE_BYTES[fmt(w2)]
    return (K // TILE) * (2 * I // TILE) * t13 + (I // TILE) * (H // TILE) * t2


def one_copy_bytes(E_local: int, K: int, I: int, H: int, n_moe: int, w13="bfp8", w2="bfp8") -> int:
    """The stacked copy's bytes per chip (= what ``GLM53_REGIONS_DECODE=stream`` gives back to the caches): the
    8x4 in bfp8 (72 x [4096, 512] + [256, 4096] per layer, 42 MoE layers) is 10.1 GB = 9.41 GiB."""
    return n_moe * E_local * expert_slice_bytes(K, I, H, w13, w2)


def active_read_bytes(n_active: int, K: int, I: int, H: int, w13="bfp8", w2="bfp8") -> int:
    """Weight bytes one chip streams per MoE layer and decode step: the active experts' slices once (both layouts
    read the same bytes; the regions-sourced layout only changes where the bytes start)."""
    return n_active * expert_slice_bytes(K, I, H, w13, w2)
