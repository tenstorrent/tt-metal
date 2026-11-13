# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only plan of the sequence-sharded DSA exchange (tt/tp_dsa_shard.py) on the TP pair (1x2 / 2x1) and the TP8 2x4
mesh (doc/ttnn/14-production-tp8-megakernel/dsa/shard_tp8_design.md, sections 2-4). No ttnn import: the geometry, the
stream blocks / destinations, the link holders per (chip, link, direction), the fabric packet-header budget per RISC,
the kernels' page / offset helpers (mirrors of kernels/mla/dsa_shard.hpp, whose constants are PARSED from the header and
asserted here) and a torch emulation of the 2-round all-gather + q assembly + 2-round reduce-scatter with the
partial-tree combine are all plain Python, so tests/s14/test_shard_plan.py runs without a device or a C++ build.

Geometry. Ranks are the mesh's row-major device order (ShardTensorToMesh / HeadShard): rank r = row C + col (2x4:
4 row + col; the pair 1x2: col). The PAIR axis is the axis of 2 chips (axis 0 on 2x4 / 2x1 = the on-board pair, axis 1
on 1x2), the LINE axis the other one (2x4: axis 1, the 4-chip cable line along a row; the pair: none, LINE = 1).
pair_pos(r) = r // LINE (2x4: the row), line_pos(r) = r % LINE (2x4: the column). Head tile g (32 heads [32 g, 32 g +
32)) = the chips with pair_pos g; chip (g, j) holds its rows 8 j .. 8 j + 7 (TP2: the whole tile)."""

from __future__ import annotations

import os
import re

KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")
_CONST = re.compile(r"constexpr\s+uint32_t\s+([A-Z][A-Z0-9_]*)\s*=\s*([0-9]+)\s*;")


def header_constants(rel: str) -> dict:
    """the plain integer `constexpr uint32_t NAME = <int>;` constants of a kernel header (path relative to
    tt/kernels): the host mirrors are asserted against these (program lesson: a stale Python SYNC_WORDS tripped the
    watcher on d0, 2026-10-22)"""
    with open(os.path.join(KDIR, rel)) as f:
        src = f.read()
    return {m.group(1): int(m.group(2)) for m in _CONST.finditer(src)}


def header_constants_define(rel: str, name: str) -> int:
    """a `#define NAME <int>` of a kernel header (e.g. kernels/mk/probe.hpp MK_PROBE_LROWS)"""
    with open(os.path.join(KDIR, rel)) as f:
        m = re.search(rf"#define\s+{name}\s+([0-9]+)", f.read())
    assert m, (rel, name)
    return int(m.group(1))


HDR = header_constants("mla/dsa_shard.hpp")
FS_HDR = header_constants("fabric_stream/fs_common.hpp")
UNIT, POOL_PAGE, TB, PPC = HDR["UNIT"], HDR["POOL_PAGE"], HDR["TB"], HDR["PPC"]
REC_BYTES, REC_HDR, RS_CHUNKS, T_MAX, ST_TILES = (HDR[k] for k in ("REC_BYTES", "REC_HDR", "RS_CHUNKS", "T_MAX",
                                                                   "ST_TILES"))  # fmt: skip
AG8_PPC, AG8_QPAGES, RS2_PPC, RS_PPC = HDR["AG8_PPC"], HDR["AG8_QPAGES"], HDR["RS2_PPC"], HDR["RS_PPC"]
AG32_PPC, AG32_QPAGES = HDR["AG32_PPC"], HDR["AG32_QPAGES"]  # 8x4 (shard_8x4_design.md section 2)
RS2B_PPC = HDR["RS2B_PPC"]  # 8x4 round 2: 2-row slices (16 x 256 B + 2 x 256 B = 4.5 KB -> 3 pages + pad)
TL_MAX = HDR["TL_MAX"]  # rev 60: tokens per token lane (the lane-local mailbox / kept blocks / CB_ASEL pages)
TL_SMALL = 4  # the lane-local token slots of the t <= 4 builds (dsa_common.hpp T_MAX without MLA_DSA_TMAX)
MSTG_PAGES = HDR["MSTG_PAGES"]  # rev 64: the KPH > 1 merge helpers' record staging pages (dsa_shard.hpp)
KPH_MAX = 16  # rev 64: a helper's per-list tie counts fill one 16-word histogram slot (dsa_shard.hpp KPH <= 16)
POS_BYTES = 128  # dsa_common.hpp POS_BYTES (the cur_pos page in CB_AKEY)
SEL_HDR = 3  # dsa_common.hpp SEL_HDR (CB_ASEL words before the pool ids)
MAX_CHUNKS = 64  # dsa_common.hpp MAX_CHUNKS
CAND_MAX = 512  # dsa_shard.hpp CAND_MAX = mladsa::TOPK (not a plain literal there)
CAND_PAGES = -(-(REC_HDR + CAND_MAX * REC_BYTES) // TB)
R8_FINE_BYTES, R8_SLICE_BYTES = HDR["R8_FINE_BYTES"], HDR["R8_SLICE_BYTES"]  # dsa_shard.hpp R8 (GLM53_DSA_SHARD_R8)
assert AG8_QPAGES + CAND_PAGES <= AG8_PPC and AG8_PPC % 2 == 0 and RS2_PPC % 2 == 0 and RS_PPC % 2 == 0
assert RS_CHUNKS * RS_PPC == 36  # rev 59: the RS round-1 message = 16 fp32 O tiles + 2 stats tiles = 36 pages exactly


def radix_rounds(r8: bool = False) -> int:
    """dsa_common.hpp ROUNDS = 32 / RADIX_BITS: 8 rounds of 4-bit digits, 4 of 8-bit digits under R8 (MLA_DSA_R8)"""
    return 4 if r8 else 8


def merge_msgs(nchips: int = 8, r8: bool = False) -> int:
    """dsa_shard.hpp MERGE_MSGS = 1 + ROUNDS + QUOTA_SLOTS (start, the radix rounds, the tie-quota slots: one 16-B slot
    per 4 merge helpers -- 2 at NCHIPS = 8, 8 at 32), checked on the text; r8: 4 rounds (MLA_DSA_R8)"""
    with open(os.path.join(KDIR, "mla", "dsa_shard.hpp")) as f:
        src = f.read()
    assert "constexpr uint32_t QUOTA_SLOTS = NCHIPS * 4 / mladsa::MSG_BYTES;" in src
    assert "constexpr uint32_t MERGE_MSGS = 1 + mladsa::ROUNDS + QUOTA_SLOTS;" in src
    return 1 + radix_rounds(r8) + nchips * 4 // 16


def merge_scr_bytes(kph: int = 1) -> int:
    """dsa_shard.hpp MERGE_SCR_BYTES (checked on the text): KPH = 1 a helper's chip candidate pages + keys + compacted
    candidates + per-core quotas; rev 64, KPH > 1: MSTG_PAGES staging pages + the KPH lists' keys + per-core quotas"""
    with open(os.path.join(KDIR, "mla", "dsa_shard.hpp")) as f:
        src = f.read()
    assert "MERGE_SCR_BYTES1 = CAND_PAGES * TB + 2 * CAND_MAX * 4 + mladsa::MAX_CHUNKS * 4;" in src
    assert "MERGE_SCR_BYTESK = MSTG_PAGES * TB + KPH * CAND_MAX * 4 + mladsa::MAX_CHUNKS * 4;" in src
    assert "MERGE_SCR_BYTES = KPH == 1 ? MERGE_SCR_BYTES1 : MERGE_SCR_BYTESK;" in src
    if kph == 1:
        return CAND_PAGES * TB + 2 * CAND_MAX * 4 + MAX_CHUNKS * 4
    return MSTG_PAGES * TB + kph * CAND_MAX * 4 + MAX_CHUNKS * 4


PIECE_ROWS = 8  # TP8: q / o rows of one chip in a 32-head tile (dsa_shard.hpp PIECE_ROWS at LINE = 4); a build's
# own value is Geometry.piece_rows (2 on the 8x4): the `*_rows` mirrors below take it
DT, ST = 16, 2  # O tiles (512 / 32) and stats tiles (m, l) of a partial (kernels/mla/att_common.hpp DT / STAT_TILES)
PKT_HDRS_PER_RISC = 12  # Blackhole: NUM_PACKET_HEADERS (6 x 2 x 2) / 2 data-movement RISCs (dev_mem_map.h)
FS_NHDR, FAR_NHDR = 4, 2 + 1  # fs_send.cpp NHDR per direction; lane_send.cpp NHDR + the flag header per direction


class Geometry:
    """the sharded exchange on a 1x2 / 2x1 pair or the 2x4 mesh (R x C mesh shape)"""

    def __init__(self, R: int, C: int):
        self.R, self.C = int(R), int(C)
        self.n = self.R * self.C
        if (self.R, self.C) in ((1, 2), (2, 1)):
            self.pair_axis, self.line_axis, self.L = (1 if self.C == 2 else 0), None, 1
        elif (self.R, self.C) == (2, 4):
            self.pair_axis, self.line_axis, self.L = 0, 1, 4
        elif (self.R, self.C) == (8, 4):  # the 32-chip Galaxy (shard_8x4_design.md): the column is the pair axis
            self.pair_axis, self.line_axis, self.L = 0, 1, 4
        else:
            raise ValueError(f"shard: the pair (1x2 / 2x1), the 2x4 or the 8x4 mesh, not {R}x{C}")
        self.P = self.n // self.L  # chips along the pair axis (2: the pair / the 2x4 column pair; 8: the 8x4 column)
        self.hpc = 64 // self.n  # heads per chip
        self.piece_rows = 32 // (self.n // 2) if self.n > 2 else 32  # a chip's q / o rows in a 32-head tile
        self.rs1_rows = 64 // self.P if self.P > 2 else 32  # round 1: the 8x4 column slice rows (the pair: whole tiles)
        self.rs2_rows = self.rs1_rows // self.L  # round 2: the row slice rows (2x4: 8; 8x4: 2)

    # ---- ranks
    def rank(self, r: int, c: int) -> int:
        return r * self.C + c

    def coord(self, rank: int) -> tuple:
        return divmod(rank, self.C)

    def pair_pos(self, rank: int) -> int:
        return rank // self.L

    def line_pos(self, rank: int) -> int:
        return rank % self.L

    def tile(self, rank: int) -> int:
        """the 32-head tile this chip's heads belong to (2x4: the mesh row; 8x4: rows 0-3 -> 0, rows 4-7 -> 1)"""
        return self.pair_pos(rank) // (self.P // 2)

    def piece(self, rank: int) -> int:
        """the chip's piece index in its tile: rows [piece_rows j, +piece_rows) (2x4: the column; 8x4: rank - 16 g)"""
        return rank - self.tile_ranks(self.tile(rank))[0]

    def ag_blk(self, rank: int) -> int:
        """y_ag block of chip `rank` (kernels/mla/dsa_shard.hpp ag_blk): the pair-axis chips of a line are adjacent"""
        return self.line_pos(rank) * self.P + self.pair_pos(rank)

    def pair_peer(self, rank: int) -> int:
        """the other chip of the rank's pair-axis pair (2x4: the column peer); P = 2 only"""
        assert self.P == 2, "pair_peer: the pair / the 2x4 mesh (the 8x4 column has 7 peers: column_peers)"
        g, j = self.pair_pos(rank), self.line_pos(rank)
        return (1 - g) * self.L + j

    def column_peers(self, rank: int) -> list:
        """the other chips of the rank's pair-axis line (its column), in position order"""
        j = self.line_pos(rank)
        return [i * self.L + j for i in range(self.P) if i != self.pair_pos(rank)]

    def line_peers(self, rank: int) -> list:
        g, j = self.pair_pos(rank), self.line_pos(rank)
        return [g * self.L + q for q in range(self.L) if q != j]

    def heads(self, rank: int, hr: int) -> range:
        return range(hr * rank, hr * (rank + 1))

    def tile_ranks(self, g: int) -> list:
        """the chips whose q pieces form head tile g (heads [32 g, 32 g + 32)), in rank order"""
        per = self.n // 2
        return list(range(g * per, (g + 1) * per))

    def pass_tile(self, rank: int, pass_: int) -> int:
        """the head tile pass 0 (the OTHER tile) / pass 1 (this chip's tile) attends with"""
        g = self.tile(rank)
        return 1 - g if pass_ == 0 else g

    def axis_pos(self, r: int, c: int, axis: int) -> tuple:
        """(position along `axis`, line index = the other coordinate) of mesh coordinate (r, c)"""
        return (r, c) if axis == 0 else (c, r)

    def axis_len(self, axis: int) -> int:
        return self.R if axis == 0 else self.C

    # ---- the streams of one launch of t rows (tt/tp_dsa_shard.py TPMlaShard._new_streams)
    def ag_chunks(self, t: int) -> int:
        """32-row chunks of one chip's AG block"""
        return t if self.L > 1 else 2 * t

    @property
    def ag_ppc(self) -> int:
        """pages per AG chunk: the pair's 16 (whole q rows), the 2x4's AG8 (8-row q piece 4 pages + 5 candidate pages +
        pad), the 8x4's AG32 (2-row q piece 1 page + 5 candidate pages: 6)"""
        if self.L == 1:
            return PPC
        return AG8_PPC if self.P == 2 else AG32_PPC

    def ag_width(self) -> int:
        return 32 * self.ag_ppc

    @property
    def rs1_ppc(self) -> int:
        """pages per x_rs1 / y_rs1 chunk: the pair exchange's RS_PPC (3 chunks = the 36-page fp32 message); the 8x4
        column scatter's 8-row slice = the 2x4 RS2 format (RS2_PPC)"""
        return RS2_PPC if self.P > 2 else RS_PPC

    @property
    def rs2_ppc(self) -> int:
        """pages per x_rs2 / y_rs2 chunk: 8-row slices (2x4) RS2_PPC; 2-row slices (8x4) RS2B_PPC"""
        return RS2_PPC if self.rs2_rows == 8 else RS2B_PPC

    def streams(self, t: int, agk: bool = False, agb: bool = False, rsb: bool = False) -> dict:
        """name -> dict(axis, kind, chunks (per block), width, link, blocks_in_y, src_blocks_in_x); the 8x4's AG under
        GLM53_DSA_SHARD_AGK / _AGB (agk / agb): the chunks / width of ag_stream_args (AGB: ONE y chunk per block, so
        AG1's dst = blk, AG2's bundle = P chunks -- the fabric streams' row_base accounting); the 8x4's scatters: the
        chunks / width of rs_stream_args (GLM53_DSA_SHARD_RSB, rsb: ONE chunk per destination block, so RS1's / RS2's
        dst = p and destination q's source base = q)"""
        assert not rsb or self.P > 2, ("RSB: the 8x4 scatters", self.P)
        ag1c, ag2c, agw = self.ag_chunks(t), self.P * self.ag_chunks(t), self.ag_width()
        rsa = rs_stream_args(t, rsb=rsb, P=self.P) if self.P > 2 else None
        if agk or agb:
            assert self.P > 2 and (agk or not agb), ("AGK / AGB: the 8x4; AGB on AGK", self.P, agk, agb)
            a = ag_stream_args(t, agk=agk, agb=agb, P=self.P)
            ag1c, ag2c, agw = a["ag1"]["rows"] // 32, a["ag2"]["rows"] // 32, a["ag1"]["width"]
        out = {"ag1": dict(axis=self.pair_axis, kind="ag", chunks=ag1c, width=agw, link=0,
                           y_blocks=self.n, x_blocks=1)}  # fmt: skip
        if self.P == 2:  # the pair exchange of the whole pass-0 partial (the pair / the 2x4 column pair)
            out["rs1"] = dict(axis=self.pair_axis, kind="rs2", chunks=t * RS_CHUNKS, width=32 * RS_PPC, link=1,
                              y_blocks=1, x_blocks=2)  # fmt: skip
        else:  # 8x4: the column reduce-scatter of 8-row slices (per-destination unicasts, as the 2x4 round 2)
            out["rs1"] = dict(axis=self.pair_axis, kind="scatter", chunks=rsa["rs1"]["rows"] // 32,
                              width=rsa["rs1"]["width"], link=1, y_blocks=self.P, x_blocks=self.P)  # fmt: skip
        if self.L > 1:
            out["ag2"] = dict(
                axis=self.line_axis,
                kind="ag",
                chunks=ag2c,
                width=agw,
                link=0,
                y_blocks=self.L,
                x_blocks=self.L,
            )  # (bundles: x = y = y_ag; the bundle = the column's P blocks)  # fmt: skip
            rs2c, rs2w = (rsa["rs2"]["rows"] // 32, rsa["rs2"]["width"]) if rsa else (t, 32 * self.rs2_ppc)
            out["rs2"] = dict(axis=self.line_axis, kind="scatter", chunks=rs2c, width=rs2w, link=1,
                              y_blocks=self.L, x_blocks=self.L)  # fmt: skip
        return out

    def block(self, name: str, t: int, r: int, c: int, rs1_early: bool = False, agk: bool = False,
              agb: bool = False, rsb: bool = False) -> dict:  # fmt: skip
        """chip (r, c)'s block of stream `name`: src chunk base in x, dst chunk base in y (on the receivers: the same
        address on every chip), local copy, up / down multicast ranges (the scatter: the chips above / below), and the
        scatter's destinations [(position q, src chunk base, direction 0 up / 1 down, hops)]; rev 65 rs1_early (the
        8x4 column scatter): the other half's destinations first (rs1_early_first), then the own half's; rsb (the 8x4
        scatters in block chunks): the chunk bases count destination blocks (1 chunk each)"""
        assert not rs1_early or self.P > 2, "RS1 early: the 8x4 column scatter"
        sp = self.streams(t, agk=agk, agb=agb, rsb=rsb)[name]
        p, line = self.axis_pos(r, c, sp["axis"])
        n = self.axis_len(sp["axis"])
        ch = sp["chunks"]
        up, dn = n - 1 - p, p
        if name == "ag1":  # this chip's block -> y_ag block ag_blk (line-major: blk = line n + p)
            return dict(p=p, src=0, dst=(line * n + p) * ch, local=True, up=up, dn=dn, dsts=[])
        if name == "ag2":  # the column's P-block bundle, already in y_ag at bundle p, multicast along the row
            return dict(p=p, src=p * ch, dst=p * ch, local=False, up=up, dn=dn, dsts=[])
        if name == "rs1" and sp["kind"] == "rs2":  # the pair exchange: the peer's block of x_rs1 -> its y_rs1
            return dict(p=p, src=(1 - p) * ch, dst=0, local=False, up=up, dn=dn, dsts=[])
        if name in ("rs1", "rs2"):  # per-destination unicast: x block q -> peer q's y block p
            dsts = [(q, q * ch, 0 if q > p else 1, abs(q - p)) for q in range(n) if q != p]
            if rs1_early and name == "rs1":
                first = set(rs1_early_first(p, self.P))
                dsts = [d for d in dsts if d[0] in first] + [d for d in dsts if d[0] not in first]
            return dict(p=p, src=None, dst=p * ch, local=False, up=up, dn=dn, dsts=dsts)
        raise KeyError(name)

    # ---- link holders (one worker connection per (chip, ethernet link, direction): fabric_erisc_router.cpp:130-134)
    def directions(self, r: int, c: int, axis: int) -> list:
        """the directions chip (r, c) sends in along `axis`: "a0+" / "a0-" / "a1+" / "a1-" (a multicast / scatter up
        reaches position + 1 ..; down position - 1 ..)"""
        p, _ = self.axis_pos(r, c, axis)
        n = self.axis_len(axis)
        return ([f"a{axis}+"] if n - 1 - p > 0 else []) + ([f"a{axis}-"] if p > 0 else [])

    def ar_directions(self, r: int, c: int) -> list:
        """tt/fabric_ar.py FabricPhase lane h: link h towards every neighbour (axis-0 up / down, axis-1 up / down)"""
        return [d for a in (0, 1) if self.axis_len(a) > 1 for d in self.directions(r, c, a)]

    def holders(self, r: int, c: int, t: int) -> list:
        """(name, link, direction, mode, released_by) per worker connection of chip (r, c)'s program, in the order of
        the design's link table: AG round 1, AG round 2, RS round 1, RS round 2 (gated: open after the AR lane of
        their link released it, closed before their consumers' signal), the o_proj AR lanes (late: hold the link only
        at launch start for the deferred credit and again for their own round)"""
        out = []
        for name, sp in self.streams(t).items():
            for d in self.directions(r, c, sp["axis"]):
                out.append((name, sp["link"], d, "gated", f"ar_lane{sp['link']}"))
        for h in (0, 1):
            for d in self.ar_directions(r, c):
                out.append((f"ar_lane{h}", h, d, "late", None))
        return out

    def release_lists(self, t: int) -> dict:
        """AR lane h -> the stream lanes it releases at launch start (FAR_LATE_OPEN: SEM_FSA += 1 on each)"""
        rel = {0: [], 1: []}
        for name, sp in self.streams(t).items():
            rel[sp["link"]].append(name)
        return rel

    def header_budget(self, r: int, c: int, t: int) -> dict:
        """fabric packet headers allocated per (lane core, RISC) of chip (r, c): fs_send (BRISC) 4 per direction,
        lane_send (BRISC) 2 + 1 flag per direction; every other RISC of the program allocates none"""
        out = {}
        for name, sp in self.streams(t).items():
            out[(name, "brisc")] = FS_NHDR * len(self.directions(r, c, sp["axis"]))
        for h in (0, 1):
            out[(f"ar_lane{h}", "brisc")] = FAR_NHDR * len(self.ar_directions(r, c))
        return out


def audit_links(holders: list) -> dict:
    """Host check (every build, the mock too) that no two worker -> EDM connections of one program hold the same
    (ethernet link, direction) at the same time (d0 card2 2026-10-22: three connections on link 0 hung the pair; one
    worker per EDM sender channel, fabric_erisc_router.cpp:130-134). holders = [(name, link, direction, mode,
    released_by)] (or the pair's old [(name, link, mode)]): "hold" = open for the whole launch (alone on its key); "late"
    = an AR lane under FAR_LATE_OPEN; "gated" = a stream under FS_LATE_OPEN / FS_AFTER_SEM, which must be released by
    the late holder of the same key. Allowed per (link, direction): one "hold" alone, or one "late" + at most one
    "gated" (released by that late holder)."""
    by = {}
    for h in holders:
        if len(h) == 3:
            name, link, mode = h
            d, rel = "*", None
        else:
            name, link, d, mode, rel = h
        by.setdefault((int(link), d), []).append((name, mode, rel))
    for key, hs in sorted(by.items()):
        modes = [m for _, m, _ in hs]
        if "hold" in modes and len(hs) > 1:
            raise ValueError(f"link {key}: a whole-launch holder shares the link: {hs}")
        if modes.count("late") > 1 or modes.count("gated") > 1:
            raise ValueError(f"link {key}: more than one late / gated connection: {hs}")
        if "gated" in modes and "late" not in modes:
            raise ValueError(f"link {key}: a gated stream without the releasing lane: {hs}")
        late = [n for n, m, _ in hs if m == "late"]
        for n, m, rel in hs:
            if m == "gated" and rel is not None and rel not in late:
                raise ValueError(f"link {key}: {n} is released by {rel}, but the late holder is {late}")
    return {f"{k[0]}{'' if k[1] == '*' else ':' + k[1]}": [f"{n} {m}" for n, m, _ in v] for k, v in sorted(by.items())}


def audit_headers(budget: dict) -> dict:
    """every (core, RISC) allocates <= the Blackhole pool (12 headers per RISC; 3 directions x 5 overflowed the 2x4
    middle chips, 2026-10-22)"""
    bad = {k: v for k, v in budget.items() if v > PKT_HDRS_PER_RISC}
    if bad:
        raise ValueError(f"fabric packet headers over {PKT_HDRS_PER_RISC} per RISC: {bad}")
    return {f"{k[0]}.{k[1]}": v for k, v in sorted(budget.items())}


def audit_targets(entries: list, rank: int, peer: int, valid: set, extra: list) -> dict:
    """Host check (every build, the mock too): after the per-chip patch every rank word reads `rank`, every peer word
    `peer` (the pair-axis peer's position), and every NoC target of the shard blocks / the streams' consumer and lane
    lists / the AR lanes' release lists is a worker core of the grid, never (0, 0) or a zero-padded pair (program lesson
    2026-10-21; d0 card0 2026-10-22: the leaders' composite patched at the members' slot). entries = (kernel, core, rank
    word, peer word, [(x, y)]); extra = [(what, (x, y))]."""
    bad, seen = [], {}
    for name, core, rk, pr, nocs in entries:
        seen[name] = seen.get(name, 0) + 1
        if rk != rank or (pr is not None and pr != peer):
            bad.append((name, core, "rank/peer", rk, pr))
        for xy in nocs:
            if tuple(xy) == (0, 0) or tuple(xy) not in valid:
                bad.append((name, core, "noc", tuple(xy)))
    for what, xy in extra:
        if tuple(xy) == (0, 0) or tuple(xy) not in valid:
            bad.append((what, tuple(xy)))
    if bad:
        raise ValueError(f"shard runtime args, rank {rank}: {bad[:8]}")
    return dict(rank=rank, patched=seen, extra=len(extra))


# ------------------------------------------------------------------ kernel offset mirrors (kernels/mla/dsa_shard.hpp)
# The compact q-piece / slice formats by the piece rows of the build (dsa_shard.hpp PIECE_RUN / PIECES_PER_PAGE /
# slice_off_rows / slice_run / agx_q_page / agx_q_off / agx_cand_page / rs1_page / rs2b_page): rows = Geometry.piece_rows
# (8 on the TP8 2x4, 2 on the 8x4). The TP8 names below (ag_q_page, slice_off, q_piece, ...) are the 8-row forms.
def piece_bytes(rows: int) -> int:
    """dsa_shard.hpp PIECE_BYTES: a tile's q piece = 2 runs of rows face rows (rows x 32 B each): 512 B / 128 B"""
    return 2 * rows * 32


def pieces_per_page(rows: int) -> int:
    """dsa_shard.hpp PIECES_PER_PAGE: 4 at TP8, 16 at TP32"""
    return TB // piece_bytes(rows)


def ag_pages(geo) -> tuple:
    """(AG_PPC, AG_QPAGES) of the build: the TP32 AG block (6 pages, 1 of q pieces), the TP8 one (10, 4)"""
    if geo.P > 2:
        return AG32_PPC, AG32_QPAGES
    return AG8_PPC, AG8_QPAGES


def agx_q_page(geo, T: int, blk: int, t: int, d: int) -> tuple:
    """dsa_shard.hpp agx_q_page / agx_q_off: (page, byte offset) of q tile d's piece of token t of block blk"""
    ppc, _ = ag_pages(geo)
    n = pieces_per_page(geo.piece_rows)
    return (blk * T + t) * ppc + d // n, (d % n) * piece_bytes(geo.piece_rows)


def agx_cand_page(geo, T: int, blk: int, t: int, byte: int) -> tuple:
    """dsa_shard.hpp agx_cand_page: (page, byte offset) of candidate byte `byte` of token t of block blk"""
    ppc, qpages = ag_pages(geo)
    return (blk * T + t) * ppc + qpages + byte // TB, byte % TB


def slice_off_rows(rows: int, j: int, tb: int) -> int:
    """dsa_shard.hpp slice_off_rows: run A of rows `rows` j .. in a tile of tb bytes (run B = + tb / 4); a face row is
    tb / 64 bytes and faces 0 / 2 hold rows 0..15 / 16..31"""
    r0 = rows * j
    return (r0 // 16) * (tb // 2) + (r0 % 16) * (tb // 64)


def slice_run(rows: int, tb: int) -> int:
    """dsa_shard.hpp slice_run: bytes of one run of `rows` face rows"""
    return rows * (tb // 64)


def rs1_page(T: int, blk: int, t: int) -> int:
    """dsa_shard.hpp rs1_page: the 8x4 round-1 slice chunk of (block = column position, token)"""
    return (blk * T + t) * RS2_PPC  # RS1_PPC = RS2_PPC


def rs2b_page(T: int, blk: int, t: int) -> int:
    """dsa_shard.hpp rs2b_page: the 8x4 round-2 slice chunk of (block = row position, token)"""
    return (blk * T + t) * RS2B_PPC


def rs_layout(rows: int, otb: int, ptb: int = 4096) -> list:
    """[(kind, index, byte offset in the slice chunk)] of one slice of `rows` rows: the DT O tiles then the ST stats
    tiles, each tb x rows / 32 bytes (the two runs); the chunk is RS2_PPC pages for 8 rows, RS2B_PPC for 2"""
    out, b = [], 0
    for d in range(DT):
        out.append(("o", d, b))
        b += otb * rows // 32
    for i in range(ST):
        out.append(("s", i, b))
        b += ptb * rows // 32
    assert b <= (RS2B_PPC if rows == 2 else RS2_PPC) * TB, (rows, b)
    return out


def ag_q_page(L: int, T: int, blk: int, t: int, d: int) -> tuple:
    """(page, byte offset) of q tile d of token t of block blk in x_ag / y_ag (TP8: the 8-row piece, 512 B)"""
    if L > 1:
        return (blk * T + t) * AG8_PPC + d // 4, (d % 4) * 512
    return (blk * 2 * T + t) * PPC + d, 0


def ag_cand_page(L: int, T: int, blk: int, t: int, byte: int) -> tuple:
    if L > 1:
        return (blk * T + t) * AG8_PPC + AG8_QPAGES + byte // TB, byte % TB
    return (blk * 2 * T + T + t) * PPC + byte // TB, byte % TB


def slice_off(j: int, tb: int) -> int:
    """run A of rows 8 j .. 8 j + 7 in a tile of tb bytes (run B = + tb / 4; tb / 8 bytes each)"""
    return slice_off_rows(PIECE_ROWS, j, tb)


def rs2_layout(otb: int, ptb: int = 4096) -> list:
    """the TP8 round-2 slice (8 rows): rs_layout(8, ...)"""
    return rs_layout(PIECE_ROWS, otb, ptb)


# ------------------------------------------------------------------ byte-level emulation of the q / slice formats
def tile_to_faces(x):
    """a [32, 32] tile (torch) -> its face-major element order (faces 0..3 of 16 x 16, row-major within a face):
    the byte order of a TILE page"""
    return x.reshape(2, 16, 2, 16).permute(0, 2, 1, 3).reshape(-1)


def faces_to_tile(v):
    return v.reshape(2, 2, 16, 16).permute(0, 2, 1, 3).reshape(32, 32)


def q_piece_rows(q_tile_faces, rows: int):
    """the latent core's q piece of one bf16 tile (face-major elements): rows 0 .. rows - 1 = face 0's first rows
    (rows x 16 elements) + face 1's (lat_dm1.cpp): 2 runs = 2 x rows x 32 B (512 B at TP8, 128 B on the 8x4)"""
    import torch

    w = rows * 16
    return torch.cat([q_tile_faces[0:w], q_tile_faces[256 : 256 + w]])


def assemble_q_tile_rows(pieces, rows: int):
    """the reader's q assembly (att_dsa_shard_dm0.cpp / ix_dm1.cpp): 32 / rows chips' pieces (run A, run B) -> one
    tile in face-major order; piece j -> rows rows j .. (run A at slice_off_rows(rows, j, 2048) / 2 elements, run B at
    + 512 / 2)"""
    import torch

    out = torch.zeros(1024, dtype=pieces[0].dtype)
    w = rows * 16
    for j, pc in enumerate(pieces):
        a = slice_off_rows(rows, j, 2048) // 2
        out[a : a + w] = pc[:w]
        out[a + 256 : a + 256 + w] = pc[w:]
    return out


def slice_runs_rows(tile_faces, rows: int, j: int):
    """rows `rows` j .. of a face-major tile (any element type): (run A, run B)"""
    n = tile_faces.numel()
    a = slice_off_rows(rows, j, n)  # (linear in the tile size: elements in, elements out)
    w = slice_run(rows, n)
    return tile_faces[a : a + w], tile_faces[a + n // 4 : a + n // 4 + w]


def land_slice_rows(dst_faces, rows: int, j: int, runs):
    n = dst_faces.numel()
    a = slice_off_rows(rows, j, n)
    w = slice_run(rows, n)
    dst_faces[a : a + w] = runs[0]
    dst_faces[a + n // 4 : a + n // 4 + w] = runs[1]
    return dst_faces


def q_piece(q_tile_faces):
    """the TP8 q piece (8 rows, 512 B)"""
    return q_piece_rows(q_tile_faces, PIECE_ROWS)


def assemble_q_tile(pieces):
    """the TP8 q assembly (4 pieces of 8 rows)"""
    return assemble_q_tile_rows(pieces, PIECE_ROWS)


def slice_runs(tile_faces, j: int):
    """rows 8 j .. 8 j + 7 of a face-major tile (any element type): (run A, run B)"""
    return slice_runs_rows(tile_faces, PIECE_ROWS, j)


def land_slice(dst_faces, j: int, runs):
    return land_slice_rows(dst_faces, PIECE_ROWS, j, runs)


# ------------------------------------------------------------------ the exchange, emulated in fp32 (torch)
def tf32(x):
    """an fp32 tensor as the FPU's matmul operand (tf32: 10 explicit mantissa bits, round to nearest even), as fp32 --
    the rescales of att_c.cpp run every O / coefficient tile through srcA / srcB this way (an approximation of the
    hardware rounding; finite inputs)"""
    import torch

    b = x.contiguous().float().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    b = (b + 0xFFF + ((b >> 13) & 1)) & 0xFFFFE000
    b = torch.where(b >= 1 << 31, b - (1 << 32), b)
    return b.to(torch.int32).view(torch.float32)


def combine(a, b, *, tf32_ops: bool = False):
    """kernels/mla/att_c.cpp combine (M = max, exp-weighted l / O), fp32; a, b = (m [rows], l [rows], O [rows, D]);
    tf32_ops: the O rescale's operands (coefficients, O tiles) rounded as the FPU's matmul operands"""
    import torch

    m1, l1, o1 = a
    m2, l2, o2 = b
    M = torch.maximum(m1, m2)
    ea, eb = torch.exp(m1 - M), torch.exp(m2 - M)
    r = tf32 if tf32_ops else (lambda v: v)
    return M, ea * l1 + eb * l2, r(ea)[:, None] * r(o1) + r(eb)[:, None] * r(o2)


def combine_chain(parts: list, *, tf32_ops: bool = False):
    """rev 64's pairwise chain of one round (take parts[0], combine the others in order) -- the reference of the fold"""
    cur = tuple(v.clone() for v in parts[0])
    for p in parts[1:]:
        cur = combine(cur, p, tf32_ops=tf32_ops)
    return cur


def fold_combine(parts: list, *, l_order=None, tf32_ops: bool = False):
    """rev 65, the fold combine of one round, per head: M = max_i m_i, c_i = exp(m_i - M), L = sum_i c_i l_i (in
    l_order: the kernel's sum order at the destination's rows, default the list order), O = sum_i c_i O_i (every operand
    rounded once with tf32_ops; the device accumulates in fp32 DST in its own order)"""
    import torch

    M = parts[0][0].clone()
    for m, _, _ in parts[1:]:
        M = torch.maximum(M, m)
    c = [torch.exp(m - M) for m, _, _ in parts]
    order = list(range(len(parts))) if l_order is None else list(l_order)
    assert sorted(order) == list(range(len(parts))), order
    L = c[order[0]] * parts[order[0]][1]
    for i in order[1:]:
        L = L + c[i] * parts[i][1]
    r = tf32 if tf32_ops else (lambda v: v)
    O = sum(r(ci)[:, None] * r(p[2]) for ci, p in zip(c, parts))
    return M, L, O


def fold1_l_order(j1: int, P: int = 8) -> list:
    """the source order of att_c.cpp fold1's l sum at the destination's row block j1: page q = 2 j + p of the l
    rotations holds source P/2 p + (j1 + j) % (P/2) there"""
    h = P // 2
    return [h * p + (j1 + j) % h for j in range(h) for p in range(2)]


def fold2_l_order(lpos: int, L: int = 4) -> list:
    """att_c.cpp fold2's l sum order at the destination's sub-block lpos: rotation j = source (lpos + j) % L"""
    return [(lpos + j) % L for j in range(L)]


def fold1_tile(j1: int):
    """FOLD1 [32, 32] (att_dsa_shard_dm1.cpp): row 8 j1 + r gathers rows 8 k + r, k < 4"""
    import torch

    f = torch.zeros(32, 32)
    for k in range(4):
        for r in range(8):
            f[8 * j1 + r, 8 * k + r] = 1.0
    return f


def fold2_tile(j1: int, lpos: int):
    """FOLD2 [32, 32]: row 2 j2 + r (j2 = 4 j1 + lpos) gathers rows 8 j1 + 2 k + r, k < 4"""
    import torch

    f = torch.zeros(32, 32)
    for k in range(4):
        for r in range(2):
            f[8 * j1 + 2 * lpos + r, 8 * j1 + 2 * k + r] = 1.0
    return f


def _land(dst_tile, rows: int, j: int, src_tile, js: int, both: bool = True):
    """rows `rows` js .. of src_tile [32, 32] -> rows `rows` j .. of dst_tile, through the face-major byte runs of
    dsa_shard.hpp slice_off_rows (run A only: columns 0..15, the stats' column 0 -- run B keeps dst's junk)"""
    df, sf = tile_to_faces(dst_tile), tile_to_faces(src_tile)
    a, b = slice_runs_rows(sf, rows, js)
    if not both:
        b = slice_runs_rows(df, rows, j)[1]
    return faces_to_tile(land_slice_rows(df.clone(), rows, j, (a, b)))


def _dev_diag(v):
    """att_c.cpp diag_of / diag_from: I x bcast_col(v) on the SFPU -- the off-diagonal entries are 0 * v[row] (NaN in a
    row whose v is inf; torch.diag would write exact zeros there)"""
    import torch

    return torch.eye(32) * v[:, None]


def fold1_tiles(src: list, j1: int, junk, *, tf32_ops: bool = False, stats_tf32: bool = False):
    """att_dsa_shard_dm1.cpp land_fold1 + att_c.cpp fold1 (rev 66) on whole tiles: src[s] = source s's shipped partial as
    tiles (m [32, 32], l [32, 32], O [16, 32, 32]; rows 8 j1 .. = its slice for this destination), junk() a fresh tile of
    a receive CB's stale content. The m rotations (4 x 2 pages) -> M = their elementwise max, c_p = exp(rotation 0 page p
    - M) (exact: CB_ARX unpacks to DST), X_p = c_p l_p, L = FOLD1 X_0 + FOLD1 X_1, F_p = FOLD1 diag(c_p), O = F_0 P_0 + F_1
    P_1 (tf32_ops: every matmul operand rounded as the FPU's). stats_tf32 models rev 65's stats path (the m / l / c tiles
    loaded from Tf32-dest CBs: cut to tf32 before every SFPU op). -> (M, L, O [16, 32, 32]) = the send CBs: rows 8 j1 ..
    carry the column-combined slice, L's / O's other rows are 0"""
    import torch

    r = tf32 if tf32_ops else (lambda v: v)
    s_ = tf32 if stats_tf32 else (lambda v: v)
    rot = {}  # (j, p) -> the m rotation page
    for j in range(4):
        for p in range(2):
            pg = junk()
            for k in range(4):
                pg = _land(pg, 8, k, src[4 * p + (k + j) % 4][0], j1)
            rot[j, p] = s_(pg)
    lp = []
    for p in range(2):
        pg = junk()
        for k in range(4):
            pg = _land(pg, 8, k, src[4 * p + k][1], j1)
        lp.append(s_(pg))
    M = rot[1, 0]
    for j, p in [(1, 1), (2, 0), (2, 1), (3, 0), (3, 1), (0, 0), (0, 1)]:  # the kernel's order (rotation 0 last)
        M = torch.maximum(M, rot[j, p])
    c = [s_(torch.exp(rot[0, p] - M)) for p in range(2)]
    X = [c[p] * lp[p] for p in range(2)]
    f1 = fold1_tile(j1)
    L = r(f1) @ r(X[0]) + r(f1) @ r(X[1])
    F = [f1 @ _dev_diag(r(c[p][:, 0])) for p in range(2)]
    O = torch.zeros(16, 32, 32)
    for d in range(16):
        P = []
        for p in range(2):
            pg = junk()
            for k in range(4):
                pg = _land(pg, 8, k, src[4 * p + k][2][d], j1)
            P.append(pg)
        O[d] = r(F[0]) @ r(P[0]) + r(F[1]) @ r(P[1])
    return M, L, O


def fold2_tiles(src: list, j1: int, lpos: int, junk, *, tf32_ops: bool = False, stats_tf32: bool = False):
    """att_dsa_shard_dm1.cpp land_fold2 + att_c.cpp fold2 (rev 66): src[s] = row source s's column-combined send tiles
    (M, L, O [16, 32, 32]; rows 8 j1 + 2 lpos .. = its slice for this destination); the rows outside 8 j1 .. of every
    landed page keep junk() (a CB's stale content). M = max of the 4 m rotations (rotation 0 included at every element),
    c = exp(rotation 0 - M), X = c l, L = FOLD2 X, G = FOLD2 diag(c), G' = diag(1 / L) G, o = G' P -> o [16, 32, 32]
    (rows 2 j2 .. = this chip's heads, j2 = 4 j1 + lpos; the other rows are never read)"""
    import torch

    r = tf32 if tf32_ops else (lambda v: v)
    s_ = tf32 if stats_tf32 else (lambda v: v)
    j2 = 4 * j1 + lpos
    rot = {}
    for j in range(4):
        pg = junk()
        for k in range(4):
            pg = _land(pg, 2, 4 * j1 + k, src[(k + j) % 4][0], j2)
        rot[j] = s_(pg)
    lp = junk()
    for k in range(4):
        lp = _land(lp, 2, 4 * j1 + k, src[k][1], j2)
    lp = s_(lp)
    M = torch.maximum(torch.maximum(torch.maximum(rot[1], rot[2]), rot[3]), rot[0])
    c = s_(torch.exp(rot[0] - M))
    f2 = fold2_tile(j1, lpos)
    L = r(f2) @ r(c * lp)
    G = f2 @ _dev_diag(r(c[:, 0]))
    G2 = _dev_diag(r(1.0 / L[:, 0])) @ r(G)
    o = torch.zeros(16, 32, 32)
    for d in range(16):
        pg = junk()
        for k in range(4):
            pg = _land(pg, 2, 4 * j1 + k, src[k][2][d], j2)
        o[d] = r(G2) @ r(pg)
    return o


# rev 66, THE FORMAT RULE of the owners' compute (kernels/mla/att_c.cpp; tt/mk_mla.py attention_compute_config takes
# this list): the attention CBs that unpack straight to DST as Float32 -- the exact fp32 stats / SFPU operands. Every
# other fp32 attention CB (the O partials, ADIAG) unpacks to srcA / srcB as Tf32: the matmul operands. A matmul operand
# from one of these CBs runs the FPU on a Float32-configured source (rev 65's fold combine: token 0 every decode step);
# a stats tile from a Tf32-dest CB is cut to tf32 before exp(m - M)
ATT_FP32_DEST = ("AMASK", "AS", "AM", "AE", "APA", "APB", "ARX", "ACOEF", "AID")
_MM_CALL = re.compile(
    r"\b(?:matmul_tiles|matmul_init|mm_tile_qk|mm_init_qk)\(\s*([A-Za-z_][\w.]*)\s*,\s*([A-Za-z_][\w.]*)"
)
_LOAD_CALL = re.compile(r"\bload\(\s*(CB_[A-Z0-9_]+)\s*,")


def att_c_format_audit(text: str | None = None) -> list:
    """THE FORMAT RULE on kernels/mla/att_c.cpp (or `text`): [(line, why)] for every matmul operand that is a
    Float32-dest CB (CB_<key> with key in ATT_FP32_DEST, or a Part's stats CB `x.s`) and, inside the fold combine
    (#if defined(MLA_DSA_SHARD_COMB) .. its #endif), every load() from a CB that is not Float32-dest (the stats)"""
    if text is None:
        with open(os.path.join(KDIR, "mla", "att_c.cpp")) as f:
            text = f.read()
    bad, comb, depth = [], False, 0
    for n, line in enumerate(text.splitlines(), 1):
        st = line.strip()
        if st.startswith("#if"):
            if comb:
                depth += 1
            elif st.startswith("#if defined(MLA_DSA_SHARD_COMB)"):
                comb, depth = True, 0
        elif st.startswith("#endif") and comb:
            if depth == 0:
                comb = False
            else:
                depth -= 1
        for m in _MM_CALL.finditer(line):
            for op in m.groups():
                key = op[3:] if op.startswith("CB_") else None
                if (key is not None and key in ATT_FP32_DEST) or op.endswith(".s"):
                    bad.append((n, f"matmul operand {op}: a Float32-dest CB"))
        if comb:
            for m in _LOAD_CALL.finditer(line):
                if m.group(1)[3:] not in ATT_FP32_DEST:
                    bad.append((n, f"fold-combine load({m.group(1)}): a Tf32-dest CB (stats cut to tf32)"))
    return bad


def emulate_exchange(geo: Geometry, q, parts_fn, *, junk=None, comb: str = "pair"):
    """The TP8 (or pair) exchange of one token, emulated per chip with the kernels' data movement:
      q: [64 heads, D] fp32 (head h on chip h // 8 at TP8, its q piece rows h % 8; the pair: 32 heads per chip);
      parts_fn(rank, q_tile [32, D]) -> the chip's partial (m [32], l [32], O [32, D]) of its OWN selected rows for the
      32 heads of q_tile (the in-chip tree's result).
    AG: each chip's q piece -> every chip (y_ag blocks by ag_blk); each chip assembles tile A / tile B from the 4 pieces.
    RS round 1: the pass-0 partial (the other pair position's tile) -> the pair peer, combined with its pass-1 partial
    (own tile) in the kernel's order combine(own, landed). TP8 round 2: each chip lands rows 8 j .. of its line peers'
    column-combined partials (+ its own slice from x_rs2) into the receive tile (junk rows = the previous message's
    rows: `junk` fills them, default the round-1 landing) and combines own, then the peers in column order.
    comb="fold" (rev 65, the 8x4 only): each round is the fold combine (fold_combine, in the kernel's l order).
    -> {rank: o [heads of the chip, D]} (o = O / l)."""
    import torch

    assert comb in ("pair", "fold"), comb
    assert comb == "pair" or geo.P > 2, "the fold combine: the 8x4"
    n, L = geo.n, geo.L
    hpc = 64 // n
    pieces = {r: q[hpc * r : hpc * (r + 1)] for r in range(n)}
    y_ag = {geo.ag_blk(r): pieces[r] for r in range(n)}
    tiles = {g: torch.cat([y_ag[geo.ag_blk(rr)] for rr in geo.tile_ranks(g)]) for g in (0, 1)}
    p0 = {r: parts_fn(r, tiles[geo.pass_tile(r, 0)]) for r in range(n)}
    p1 = {r: parts_fn(r, tiles[geo.pass_tile(r, 1)]) for r in range(n)}
    if geo.P > 2:  # 8x4 (shard_8x4_design.md): two reduce-scatters, own slice first then the peers in position order
        full = {}  # the 64-head partial in head order (tile 0 rows, then tile 1 rows)
        for r in range(n):
            a, b = (p1[r], p0[r]) if geo.tile(r) == 0 else (p0[r], p1[r])
            full[r] = tuple(torch.cat([x, y]) for x, y in zip(a, b))
        col = {}  # round 1 along the column: chip (i, c) combines the slice [rs1_rows i, +rs1_rows) of its column
        for r in range(n):
            i = geo.pair_pos(r)
            rows = slice(geo.rs1_rows * i, geo.rs1_rows * (i + 1))
            if comb == "fold":  # the sources in column position order (the fold's pages), the kernel's l order
                srcs = [tuple(v[rows] for v in full[geo.line_pos(r) + geo.L * s]) for s in range(geo.P)]
                col[r] = fold_combine(srcs, l_order=fold1_l_order(i % (geo.P // 2), geo.P))
                continue
            cur = None
            for src in [r] + geo.column_peers(r):
                sl = tuple(v[rows] for v in full[src])
                cur = tuple(v.clone() for v in sl) if cur is None else combine(cur, sl)
            col[r] = cur
        out = {}  # round 2 along the row: chip (i, j) combines the sub-slice [rs2_rows j, +rs2_rows) of its row
        for r in range(n):
            j = geo.line_pos(r)
            rows = slice(geo.rs2_rows * j, geo.rs2_rows * (j + 1))
            if comb == "fold":
                g0 = geo.pair_pos(r) * geo.L
                srcs = [tuple(v[rows] for v in col[g0 + s]) for s in range(geo.L)]
                m, l_, O = fold_combine(srcs, l_order=fold2_l_order(j, geo.L))
                out[r] = O / l_[:, None]
                continue
            cur = None
            for src in [r] + sorted(geo.line_peers(r), key=geo.line_pos):
                sl = tuple(v[rows] for v in col[src])
                cur = tuple(v.clone() for v in sl) if cur is None else combine(cur, sl)
            m, l_, O = cur
            out[r] = O / l_[:, None]
        return out
    # round 1: chip r lands p0 of its pair peer (that partial covers r's tile), combine(own pass 1, landed)
    col = {r: combine(p1[r], p0[geo.pair_peer(r)]) for r in range(n)}
    out = {}
    for r in range(n):
        if L == 1:
            m, l_, O = col[r]
            out[r] = O / l_[:, None]
            continue
        j = geo.line_pos(r)
        rows = slice(PIECE_ROWS * j, PIECE_ROWS * (j + 1))
        landed_prev = p0[geo.pair_peer(r)] if junk is None else junk  # the receive CBs' previous content
        order = [r] + sorted(geo.line_peers(r), key=geo.line_pos)  # own slice first, then by column
        cur = None
        for src in order:
            m, l_, O = (v.clone() for v in landed_prev)
            ms, ls, Os = col[src]
            m[rows], l_[rows], O[rows] = ms[rows], ls[rows], Os[rows]
            landed_prev = (m, l_, O)
            cur = (m.clone(), l_.clone(), O.clone()) if cur is None else combine(cur, (m, l_, O))
        m, l_, O = cur
        out[r] = (O / l_[:, None])[rows]
    return out


# ------------------------------------------------------------------ rev 60 (W): token lanes at t > 4
def lanes_reasons(
    t: int, lanes: int, chunks: int = 64, line: int = 1, groups: int = 1, nchips: int | None = None, kph: int = 1
) -> list:
    """why a t-row launch on `lanes` token lanes cannot run the sharded phase (tt/mk_mla.py shard_lanes_check's rules,
    host-only): [] = it can. nchips: the chips (default 2 line = the pair / the 2x4; the 8x4 passes 32); kph (rev 64):
    chips per merge helper -> nchips / kph helpers per lane"""
    why = []
    if nchips is None:
        nchips = 2 * line
    if t > T_MAX:
        why.append(f"t <= {T_MAX}")
    if groups != 1:
        why.append("one head group")
    if lanes < 1 or lanes & (lanes - 1) or chunks % lanes:
        why.append("lanes: a power of two dividing the chunk cores")
        return why
    if kph < 1 or kph > KPH_MAX or nchips % kph:
        why.append(f"kph: a divisor of the {nchips} chips <= {KPH_MAX}")
        return why
    if kph > 1 and nchips != 32:
        why.append("kph > 1: the 8x4 only")
    if -(-t // lanes) > TL_MAX:
        why.append(f"<= {TL_MAX} tokens per lane")
    cl = chunks // lanes
    if cl < nchips // kph:
        why.append(f"{nchips // kph} merge helpers among {cl} lane chunk cores")
    return why


# ------------------------------------------------------------------ rev 64: KPH chips per merge helper (8x4)
def shard_kph(nchips: int, chunks: int, lanes: int, pair: int, knob="1") -> int:
    """chips per merge helper of a sharded build (dsa_shard.hpp KPH; tt/mk_mla.py shard_kph passes
    GLM53_DSA_SHARD_KPH as `knob`): 1 on the pair / the 2x4 (pair <= 2: their builds unchanged) and for knob 1;
    "auto" -> the fewest chips per helper that seat NCHIPS / KPH helpers in each lane's chunks / lanes cores
    (ceil(nchips lanes / chunks), rounded up to a divisor of nchips: the 8x4's 64 chunk cores -> 1 / 1 / 2 / 4 at
    1 / 2 / 4 / 8 lanes); an int -> itself (a divisor of nchips, <= KPH_MAX)"""
    if pair <= 2:
        return 1
    if str(knob).strip().lower() == "auto":
        k = max(1, -(-nchips * lanes // chunks))
        while nchips % k:
            k += 1
    else:
        k = int(knob)
    if k < 1 or k > KPH_MAX or nchips % k:
        raise ValueError(f"GLM53_DSA_SHARD_KPH={knob}: chips per merge helper must divide {nchips} and be <= {KPH_MAX}")
    return k


def shard_kph_reserve(nchips: int, pair: int, knob="1") -> int:
    """the chips per helper the lane-core reservation plans for (tt/tp_dsa_shard.py reserve_lane_cores(rest, nchips /
    this)): the knob's int, or KPH_MAX for "auto" (the lanes are then capped by GLM53_MLA_DSA_LANES and the free
    cores; each t's shard_kph seats its helpers); 1 on the pair / the 2x4"""
    if pair <= 2:
        return 1
    if str(knob).strip().lower() == "auto":
        return KPH_MAX if nchips % KPH_MAX == 0 else 1
    return shard_kph(nchips, 1, 1, pair, knob)


def _flag(name: str) -> bool:
    v = os.environ.get(name, "0").strip()
    if v not in ("0", "1"):
        raise ValueError(f"{name}={v}: 0 or 1")
    return v == "1"


def shard_comb(pair: int) -> bool:
    """rev 65 GLM53_DSA_SHARD_COMB (default 0 = rev 64's pairwise chains): the fold combine of the 8x4 reduce-scatter
    rounds on the lane roots (dsa_shard.hpp COMB, MLA_DSA_SHARD_COMB; emulated by fold1_tiles / fold2_tiles); the pair /
    the 2x4 never take it"""
    return pair > 2 and _flag("GLM53_DSA_SHARD_COMB")


def shard_rs1_early(pair: int) -> bool:
    """rev 65 GLM53_DSA_SHARD_RS1_EARLY (default 0): the 8x4 roots signal the RS1 scatter lane after pass 0 too and the
    lane streams the other half's destinations (pass 0's slices) during pass 1 (dsa_shard.hpp RS1_EARLY, fs_read.cpp
    FS_WAIT2); the pair / the 2x4 never take it"""
    return pair > 2 and _flag("GLM53_DSA_SHARD_RS1_EARLY")


def shard_r8(pair: int) -> bool:
    """GLM53_DSA_SHARD_R8 (default 0 = 4-bit digits, 8 rounds + the tie): the 8x4 shard's local select and global merge
    in 8-bit radix digits, 4 rounds + the tie (dsa_common.hpp RADIX_BITS, dsa_shard.hpp R8_*: the kernels' MLA_DSA_R8
    define, emulated by local_select / helper_merge digit_bits=8: the same T, k_rem, per-core counts and quotas); the
    pair / the 2x4 never take it"""
    return pair > 2 and _flag("GLM53_DSA_SHARD_R8")


def shard_local_all(pair: int) -> bool:
    """rev 72 GLM53_DSA_SHARD_LOCAL_ALL (default 0; on in tt/galaxy_defaults.py): a sparse token whose chip holds <= 512 closed local pools skips the
    local select's radix rounds and tie round on the coordinator and every owner (dsa_shard.hpp LOCAL_ALL, the
    kernels' MLA_DSA_SHARD_LOCAL_ALL; emulated by local_select_all: the records, offsets and header count of
    local_select); the pair / the 2x4 never take it"""
    return pair > 2 and _flag("GLM53_DSA_SHARD_LOCAL_ALL")


def shard_round_marks(pair: int, r8: bool = True) -> bool:
    """rev 72 GLM53_DSA_SHARD_ROUND_MARKS (default 0; probe builds): the shard coordinator's 10 local-select round marks
    (dsa_shard.hpp ROUND_MARKS, decode_probe "rounds"); needs R8 (4 rounds + the tie fit the 16 probe marks)"""
    on = pair > 2 and _flag("GLM53_DSA_SHARD_ROUND_MARKS")
    if on and not r8:
        raise ValueError(
            "GLM53_DSA_SHARD_ROUND_MARKS=1 needs GLM53_DSA_SHARD_R8=1 (16 probe marks: 4 rounds + the tie)"
        )
    return on


def shard_sel2(pair: int, r8: bool = True) -> bool:
    """rev 73 GLM53_DSA_SHARD_SEL2 (default 0; on in tt/galaxy_defaults.py): the shard local select's rounds with the same integers at less
    coordinator work (dsa_shard.hpp SEL2: no per-round above counts -- the cores report theirs in the tie round --,
    packed report / pull sums, per-owner quota chunks; emulated by local_select_sel2); needs R8; the pair / the 2x4
    never take it"""
    on = pair > 2 and _flag("GLM53_DSA_SHARD_SEL2")
    if on and not r8:
        raise ValueError("GLM53_DSA_SHARD_SEL2=1 needs GLM53_DSA_SHARD_R8=1 (the two-level rounds)")
    return on


def local_all_pools(n_closed: int, c: int, C: int) -> int:
    """dsa_shard.hpp local_all_pools: the valid pools of chunk core c of C at n_closed closed local pools (pool tiles
    c, c + C, .. < ceil(n_closed / 32), each min(32, n_closed - its first pool) columns)"""
    nt = (n_closed + POOL_PAGE - 1) // POOL_PAGE
    npt = (nt - c + C - 1) // C if c < nt else 0
    if npt == 0:
        return 0
    tail = n_closed - (c + (npt - 1) * C) * POOL_PAGE
    return (npt - 1) * POOL_PAGE + min(tail, POOL_PAGE)


def local_all_offset(n_closed: int, c: int, C: int) -> int:
    """dsa_shard.hpp local_all_offset: core c's record offset = the valid pools of the cores before it"""
    return sum(local_all_pools(n_closed, k, C) for k in range(c))


def local_select_all(core_keys: list, k: int = 512) -> dict:
    """rev 72 LOCAL_ALL (ix_dm0.cpp + att_dsa_shard_dm1.cpp phase A, <= k keys on the chip's lane): no rounds, no tie
    message -- every core keeps all its keys in local order, its records at the analytic offset (the cores before it),
    the header count = every key. -> dict(n_rec, kept, records, offsets, messages) in local_select's terms"""
    total = sum(len(v) for v in core_keys)
    if total > k:
        raise ValueError(f"local_select_all: {total} keys > k = {k} (the rounds are needed)")
    kept, records, offsets, off = {}, [], [], 0
    for c, keys in enumerate(core_keys):
        offsets.append(off)
        kept[c] = [(kk, i) for i, kk in enumerate(keys)]
        records += [(kk, c) for kk in keys]
        off += len(keys)
    return dict(n_rec=off, kept=kept, records=records, offsets=offsets, messages=0)


def r8_header_check() -> None:
    """the R8 host mirrors vs the kernels' text (tt/mk_mla.py _shard_header_check): RADIX_BITS / ROUNDS / MSGS in
    dsa_common.hpp, the fine area / pull slice in dsa_shard.hpp"""
    with open(os.path.join(KDIR, "mla", "dsa_common.hpp")) as f:
        common = f.read()
    for line in (
        "constexpr uint32_t DIGIT_BITS = 4, BINS = 1u << DIGIT_BITS;",
        "constexpr uint32_t RADIX_BITS = 2 * DIGIT_BITS;",
        "constexpr uint32_t RADIX_MASK = (1u << RADIX_BITS) - 1, ROUNDS = 32 / RADIX_BITS;",
        "constexpr uint32_t MSGS = ROUNDS + 1;",
    ):
        assert line in common, ("dsa_common.hpp R8", line)
    with open(os.path.join(KDIR, "mla", "dsa_shard.hpp")) as f:
        src = f.read()
    assert "constexpr uint32_t R8_FINE_OFF = MBOX_SHARD_BYTES;" in src, "dsa_shard.hpp R8_FINE_OFF"
    assert (R8_FINE_BYTES, R8_SLICE_BYTES) == (256 * 2, 16 * 2), (R8_FINE_BYTES, R8_SLICE_BYTES)
    for tl in (TL_SMALL, TL_MAX):
        assert r8_fine_off(tl, 32) % 16 == 0 and mbox_bytes(tl, 32, r8=True) <= mbox_bytes(tl, 32), tl


RS1_PASS1_INC = HDR["RS1_PASS1_INC"]  # rev 67: dsa_shard.hpp -- a pass-1 signal's weight (fs_read.cpp FS_WAIT2 field)
assert RS1_PASS1_INC == 1 << 16


def rs1_lane_ships(pass_us: list) -> tuple:
    """the RS1 signals of every token lane root over one launch: pass_us[l] = (pass-0 durations, pass-1 durations) of
    lane l's tokens (the tree + the ship of each). The root runs pass 0 of all its lane tokens, then pass 1
    (att_dsa_shard_dm1.cpp: the pass loop outside the token loop) and signals after each token's ship. -> (done0,
    done1): per lane the signal times of its tokens' pass-0 / pass-1 slices"""
    done0, done1 = [], []
    for d0, d1 in pass_us:
        t_, a = 0.0, []
        for x in d0:
            t_ += x
            a.append(t_)
        b = []
        for x in d1:
            t_ += x
            b.append(t_)
        done0.append(a)
        done1.append(b)
    return done0, done1


def rs1_early_gates(done0: list, done1: list, t: int, *, fields: bool = True) -> tuple:
    """fs_read.cpp FS_WAIT2's two gates on the RS1 lane's wait semaphore as the roots' signals land in time order (a
    pass-0 signal before a pass-1 one at the same time): rev 67 (fields=True: pass 0 adds 1, pass 1 RS1_PASS1_INC; the
    first gate opens at t in bits [0, 16), the second at t in bits [16, 32)) or rev 65 / 66 (fields=False: every signal
    adds 1; the gates open at t and 2 t). -> (gate-1 time, gate-2 time); sound iff gate 1 >= every pass-0 signal and
    gate 2 >= every pass-1 signal (the lane must not stream a destination block before its slices are written)"""
    inc1 = RS1_PASS1_INC if fields else 1
    ev = sorted([(x, 0) for d in done0 for x in d] + [(x, 1) for d in done1 for x in d])
    cnt, g1, g2 = 0, None, None
    for time, pas in ev:
        cnt += inc1 if pas else 1
        lo, hi = (cnt & (RS1_PASS1_INC - 1), cnt // RS1_PASS1_INC) if fields else (cnt, cnt)
        if g1 is None and lo >= t:
            g1 = time
        if g2 is None and hi >= (t if fields else 2 * t):
            g2 = time
    return g1, g2


def rs1_early_first(p: int, P: int) -> list:
    """the column positions chip p's pass-0 slices go to (the other half of the column: pass 0 attends with the other
    head tile, Geometry.pass_tile): the RS1 scatter's first destinations under RS1 early, in position order"""
    return [q for q in range(P) if q // (P // 2) != p // (P // 2)]


def shard_tl(t: int, lanes: int = 1, kph: int = 1) -> int:
    """dsa_shard.hpp TL (the MLA_DSA_TMAX define of a shard build; tt/mk_mla.py shard_tl): the lane-local token slots
    of the owners' mailbox / kept blocks / the T_g probe words -- TL_SMALL up to t = 4, TL_MAX above (rev 60); rev 64:
    with KPH > 1 (more lanes, so more keys per chunk core and larger kept blocks) TL_SMALL when the lane's tokens
    ceil(t / lanes) <= 4 (dsa_shard.hpp static_assert TL in {4, TL_MAX})"""
    if t <= TL_SMALL:
        return TL_SMALL
    if kph > 1 and -(-t // max(1, lanes)) <= TL_SMALL:
        return TL_SMALL
    return TL_MAX


# the owners' L1 of the split-K phase (pure-Python mirrors; tt/mk_mla.py _akey_bytes / attention_cbs / dsa_mbox_bytes
# delegate their shard branches here, so the host suite checks the bytes the device program allocates)
def owner_keys(tp_tiles: int, chunks: int, lanes: int) -> int:
    """MLA_DSA_KEYS: keys one owner holds per sparse token = its pooled-key tiles over the lane's chunk cores x 32
    (tt/mk_mla.py MkMla._dsa_keys; tp_tiles = the slot's LOCAL pooled-key tiles)"""
    return -(-tp_tiles // (chunks // lanes)) * 32


def akey_bytes(keys: int, shard_t: int, tl: int, kph: int = 1) -> int:
    """CB_AKEY bytes per owner of a shard build (att_dsa_shard_dm1.cpp UB / KW / kept / stg): [0, 64) histogram, the
    cur_pos page, the region U = max(the keys + compacted candidates, MERGE_SCR_BYTES(kph) + 64), tl kept blocks of
    KW = min(512, keys) (key, pool) pairs, the 64-B aligned record staging (TB + 64) + 64 slack"""
    assert 1 <= shard_t <= tl <= TL_MAX, (shard_t, tl)
    kw = min(CAND_MAX, keys)
    return 64 + POS_BYTES + max(8 * keys, merge_scr_bytes(kph) + 64) + tl * 2 * kw * 4 + 64 + TB + 64 + 64


def asel_page_bytes(keys: int) -> int:
    """a CB_ASEL page of a shard build: SEL_HDR + min(512, keys) words + 52 B (one page per lane token)"""
    return (SEL_HDR + min(CAND_MAX, keys)) * 4 + 52


def mbox_bytes(tl: int, nchips: int, r8: bool = False) -> int:
    """the owners' shard mailbox (dsa_shard.hpp MBOX_SHARD_BYTES at mladsa::T_MAX = tl): the select rounds + tie
    (quota, offset) pairs, the global message + quotas, the merge messages (merge_msgs(nchips) slots) per lane token;
    r8 (GLM53_DSA_SHARD_R8): MSGS = 5 / MERGE_MSGS = 1 + 4 + .. slots, then the core's fine counts (R8_FINE_OFF =
    MBOX_SHARD_BYTES, R8_FINE_BYTES) -- never more than the 4-bit mailbox (the fine area takes the 4 rounds' slots)"""
    msgs = radix_rounds(r8) + 1  # dsa_common.hpp MSGS (+ the tie round)
    layout = tl * msgs * 16 + tl * MAX_CHUNKS * 4 * 2 + tl * 16 + tl * MAX_CHUNKS * 4 + tl * merge_msgs(nchips, r8) * 16
    return layout + (R8_FINE_BYTES if r8 else 0)


def r8_fine_off(tl: int, nchips: int) -> int:
    """dsa_shard.hpp R8_FINE_OFF = MBOX_SHARD_BYTES of an R8 build: the byte offset of a core's fine counts in its
    mailbox (the coordinator reads bucket b's R8_SLICE_BYTES at + b R8_SLICE_BYTES)"""
    return mbox_bytes(tl, nchips, r8=True) - R8_FINE_BYTES


def owner_l1(tp_tiles: int, t: int, lanes: int, kph: int = 1, chunks: int = 64, nchips: int = 32,
             r8: bool = False) -> dict:  # fmt: skip
    """the split-K phase's per-owner L1 of a t-row build on `lanes` lanes: CB_AKEY, CB_ASEL (ceil(t / lanes) pages),
    the mailbox (r8: GLM53_DSA_SHARD_R8's, with the fine counts); + the keys, TL and the lane tokens"""
    keys = owner_keys(tp_tiles, chunks, lanes)
    st = -(-t // lanes)
    tl = shard_tl(t, lanes, kph)
    return dict(
        keys=keys,
        shard_t=st,
        tl=tl,
        akey=akey_bytes(keys, st, tl, kph),
        asel=max(1, st) * asel_page_bytes(keys),
        mbox=mbox_bytes(tl, nchips, r8),
    )


def lane_of(t: int, lanes: int) -> tuple:
    """(lane, lane-local token j) of launch token t (dsa_shard.hpp lane_tok: t / LANES; the lane = t % LANES)"""
    return t % lanes, t // lanes


def lane_tokens(t_rows: int, t0: int, lanes: int) -> int:
    """dsa_shard.hpp lane_tokens: the tokens t0, t0 + lanes, .. < t_rows"""
    return -(-(t_rows - t0) // lanes) if t0 < t_rows else 0


def ag_producers(t: int, chunks: int = 64, lanes: int = 1) -> int:
    """the AG round-1 lane's producer increments per launch: per token its lane's chunk cores (every group-0 core
    increments once per token, even with 0 records) + its lane's index core (the record-count header), + the latent
    core's one q_lat copy (tt/tp_dsa_shard.py ag_wait)"""
    cl = chunks // lanes
    per = [0] * lanes
    for tt in range(t):
        lane, _ = lane_of(tt, lanes)
        per[lane] += cl + 1  # its chunk cores' records + its coordinator's header
    return sum(per) + 1


def mailbox_slots(t: int, lanes: int) -> dict:
    """{lane: [lane-local slot j of each of its tokens]} (the owners' mailbox / kept blocks / CB_ASEL pages)"""
    out = {}
    for tt in range(t):
        lane, j = lane_of(tt, lanes)
        out.setdefault(lane, []).append(j)
    return out


def lane_records(keys_local, lq, chunks: int, lanes: int, k: int = 512) -> list:
    """one chip's candidate list for a token on a lane of `chunks // lanes` chunk cores: the local exact top-k with ties in
    (lane-local core, local index) order, as records [(key, core)] in (core, local) order (att_dsa_shard_dm1.cpp phase
    A + ix_dm0's offsets). keys_local: int64 order keys of the chip's local pools lq (local pool ids)."""
    import torch

    cl = chunks // lanes
    core = (lq // POOL_PAGE) % cl
    local = (lq // POOL_PAGE // cl) * POOL_PAGE + lq % POOL_PAGE
    order = core * (1 << 24) + local
    n = int(keys_local.numel())
    if n <= k:
        sel = torch.arange(n)
    else:
        i1 = torch.sort(order, stable=True).indices
        i2 = torch.sort(keys_local[i1], descending=True, stable=True).indices
        sel = i1[i2[:k]]
    srt = torch.sort(order[sel]).indices
    sel = sel[srt]
    return [(int(keys_local[i]), int(core[i])) for i in sel.tolist()]


def lane_core_keys(keys_local, lq, chunks: int, lanes: int) -> list:
    """one chip's keys per lane chunk core in the kernel's local order (att_dsa_shard_dm1.cpp phase A: core c's local
    index i = pooled-key tile (c + (i / 32) C) column i % 32): [core c: [keys]] (the local_select input)"""
    import torch

    cl = chunks // lanes
    core = (lq // POOL_PAGE) % cl
    local = (lq // POOL_PAGE // cl) * POOL_PAGE + lq % POOL_PAGE
    out = []
    for c in range(cl):
        mine = torch.nonzero(core == c).reshape(-1)
        mine = mine[torch.sort(local[mine]).indices]
        assert torch.equal(local[mine], torch.arange(int(mine.numel()))), "a core's local indices are 0 .. n - 1"
        out.append([int(x) for x in keys_local[mine].tolist()])
    return out


def emulate_lane_merge(keys_global, C: int, chunks: int, lanes: int, k: int = 512, kph: int = 1,
                       digit_bits: int | None = None) -> dict:  # fmt: skip
    """one token of a t > 4 launch on `lanes` lanes: every chip's local top-k over its lane's chunk cores, the
    helpers' global merge (helper_merge, the replicas' T_g / helper quotas; rev 64: kph chips per helper) -> dict(T_g,
    per-chip core quotas, lists). digit_bits 4 / 8: the lists from the kernels' local select (local_select over
    lane_core_keys) and the merge in those digits (8: GLM53_DSA_SHARD_R8); None: the sorted reference (lane_records).
    keys_global: int64 order keys of the token's closed pools (global pool ids 0 .. n - 1; pool g on chip (g // 32) % C,
    local id (g // 32 // C) 32 + g % 32: tt/dsa_shard.py ShardMap)"""
    import torch

    g = torch.arange(int(keys_global.numel()))
    owner = (g // POOL_PAGE) % C
    lists = []
    for r in range(C):
        mine = torch.nonzero(owner == r).reshape(-1)
        lq = (mine // POOL_PAGE // C) * POOL_PAGE + mine % POOL_PAGE
        if digit_bits is None:
            lists.append(lane_records(keys_global[mine], lq, chunks, lanes, k))
        else:
            lists.append(local_select(lane_core_keys(keys_global[mine], lq, chunks, lanes), k, digit_bits)["records"])
    res = [helper_merge(lists, r, k, digit_bits=digit_bits or 4, kph=kph) for r in range(C)]
    assert len({(x["T_g"], tuple(x["helper_quota"])) for x in res}) == 1, "the replicas disagree"
    return dict(T_g=res[0]["T_g"], core_quota=[x["core_quota"] for x in res], lists=lists,
                helper_quota=res[0]["helper_quota"])  # fmt: skip


# ------------------------------------------------------------------ the shard phase probe (rev 59)
PROBE_LANE_ROW = {"ag1": 236, "ag2": 237, "rs1": 238, "rs2": 239}  # fs_send.cpp rows (mk_mla.py SHARD_PROBE_LANE0)
PROBE_ROOT_NC, PROBE_ROOT_BR, PROBE_IX0, PROBE_IX1, PROBE_LAT1 = 132, 0, 220, 221, 131  # the root = owner k = 0
ATT_NC_MARKS = ("att_start", "local_select", "merge", "pass0", "rs1_landed", "pass1", "rs2_landed", "finalize", "end")
ATT_BR_MARKS = ("att_start", "presweep", "pass0_pushed", "end")
IX0_MARKS = ("start", "i1_in", "y_in", "local_coord", "ag_landed", "merge_done")
ROUND_MARKS_N = (
    10  # rev 72 (GLM53_DSA_SHARD_ROUND_MARKS): 4 R8 rounds + the tie, 2 marks each, between y_in and local_coord
)
LANE_MARKS = ("start", "opened", "sent", "closed", "landed")


def _marks(row) -> list:
    return [int(v) & 0xFFFFFFFF for v in row if int(v) & 0xFFFFFFFF]


def decode_probe(p, *, line: int, mhz: float = 1350.0, ar=None) -> dict:
    """One chip's MkMla probe tensor [PROBE_ROWS, 16] (the last replay's marks; uint32 wall-clock cycles of this chip)
    -> the shard phase's marks in us after the root owner's attention start (the root = owner k = 0: chunk core 0 of
    group 0) and the derived spans:
      root NCRISC (row 132; its LAST 9 marks: the attention phase after mhc_pre's) ATT_NC_MARKS, root BRISC (row 0, last
      4) ATT_BR_MARKS, the index core ix_dm0 (row 220) IX0_MARKS, ix_dm1 (row 221; TP8: the q assembly = its last 2),
      the stream lanes (rows 236..239) LANE_MARKS; ar = FabricPhase.lane_marks() of this chip (optional: [[brisc x 8],
      [ncrisc x 5]] per lane: launch start, inputs ready, round 1 sent, round 2 sent, result done, launch end)."""
    rows = [[int(v) & 0xFFFFFFFF for v in r] for r in p.tolist()]
    nc, br = _marks(rows[PROBE_ROOT_NC]), _marks(rows[PROBE_ROOT_BR])
    if len(nc) < len(ATT_NC_MARKS):
        return dict(error=f"root NCRISC row {PROBE_ROOT_NC}: {len(nc)} marks")
    base = nc[-len(ATT_NC_MARKS)]

    def us(v):
        d = (int(v) - base) & 0xFFFFFFFF
        return round((d - (1 << 32) if d >= (1 << 31) else d) / mhz, 2)

    out = dict(root_nc={k: us(v) for k, v in zip(ATT_NC_MARKS, nc[-len(ATT_NC_MARKS) :])})
    if len(br) >= len(ATT_BR_MARKS):
        out["root_br"] = {k: us(v) for k, v in zip(ATT_BR_MARKS, br[-len(ATT_BR_MARKS) :])}
    ix0 = _marks(rows[PROBE_IX0])
    rounds = None
    if len(ix0) == len(IX0_MARKS) + ROUND_MARKS_N:  # rev 72 ROUND_MARKS: 10 marks after y_in (dsa_shard.hpp)
        rounds = ix0[3 : 3 + ROUND_MARKS_N]
        ix0 = ix0[:3] + ix0[3 + ROUND_MARKS_N :]
    if len(ix0) >= len(IX0_MARKS):
        out["ix0"] = {k: us(v) for k, v in zip(IX0_MARKS, ix0[-len(IX0_MARKS) :])}
    if rounds is not None:  # per round (4 R8 rounds, then the tie): histograms in, the broadcast done
        out["rounds"] = [dict(hist_in=us(rounds[2 * r]), out=us(rounds[2 * r + 1])) for r in range(ROUND_MARKS_N // 2)]
    ix1 = _marks(rows[PROBE_IX1])
    if line > 1 and len(ix1) >= 2:
        out["qasm"] = dict(ag_landed=us(ix1[-2]), assembled=us(ix1[-1]))
    lanes = {}
    for name, r in PROBE_LANE_ROW.items():
        m = _marks(rows[r])
        if len(m) >= len(LANE_MARKS):
            lanes[name] = {k: us(v) for k, v in zip(LANE_MARKS, m[-len(LANE_MARKS) :])}
    out["lanes"] = lanes
    if ar is not None:
        out["ar"] = [dict(start=us(b[0]), inputs=us(b[1]), result=us(b[4]), end=us(b[5])) for b, _ in ar]
    n = out["root_nc"]
    span = dict(attention=n["end"] - n["att_start"], local_select=n["local_select"] - n["att_start"],
                merge_wait=n["merge"] - n["local_select"], pass0=n["pass0"] - n["merge"],
                rs1_wait=n["rs1_landed"] - n["pass0"], pass1=n["pass1"] - n["rs1_landed"],
                rs2_wait=n["rs2_landed"] - n["pass1"], finalize=n["finalize"] - n["rs2_landed"])  # fmt: skip
    if "ix0" in out:
        i = out["ix0"]
        span.update(ag_total=i["ag_landed"] - i["local_coord"], merge=i["merge_done"] - i["ag_landed"])
        if "rounds" in out:  # rev 72 ROUND_MARKS: owners = histograms in after the previous broadcast (round 0: the
            # I4 scoring after y_in), coord = the sums + digits + broadcast; records = the records / header after the tie
            prev = i["y_in"]
            for r, x in enumerate(out["rounds"]):
                nm = f"r{r}" if r < ROUND_MARKS_N // 2 - 1 else "tie"
                span[f"{nm}_owners"] = x["hist_in"] - prev
                span[f"{nm}_coord"] = x["out"] - x["hist_in"]
                prev = x["out"]
            span["records"] = i["local_coord"] - prev
    for name, l_ in lanes.items():
        span[f"{name}_open_to_landed"] = l_["landed"] - l_["opened"]
        span[f"{name}_send"] = l_["sent"] - l_["opened"]
    if "qasm" in out:
        span["q_asm"] = out["qasm"]["assembled"] - out["qasm"]["ag_landed"]
    if ar is not None and out["ar"]:
        span["ar_inputs_to_result"] = out["ar"][0]["result"] - out["ar"][0]["inputs"]
    out["span"] = {k: round(v, 2) for k, v in span.items()}
    return out


# ------------------------------------------------------------------ the radix select's digit picks (emulation)
def pick_digit(h: list, k_rem: int) -> tuple:
    """ix_dm0.cpp's pick over one 16-bin level: the largest digit d with (keys of digit > d) < k_rem <= (keys of digit
    >= d), 0 when none -> (d, the keys of digit > d)"""
    d, ab = len(h) - 1, 0
    while d > 0 and ab + h[d] < k_rem:
        ab += h[d]
        d -= 1
    return d, ab


def r8_hist(keys: list, mask: int, prefix: int, shift: int) -> tuple:
    """dsa_shard.hpp r8_hist (GLM53_DSA_SHARD_R8) on one core: the 256 fine counts (uint16) of its keys that match
    (mask, prefix) by digit (k >> shift) & 255, and the 16 coarse counts as the kernel sums them (a bucket's 8 uint32
    words of 2 counts each, the halves added: no carry while the core holds <= 65535 keys) -> (fine, coarse)"""
    assert len(keys) <= 0xFFFF, len(keys)
    fine = [0] * 256
    for kk in keys:
        if (kk & mask) == prefix:
            fine[(kk >> shift) & 255] += 1
    words = [fine[2 * w] | (fine[2 * w + 1] << 16) for w in range(128)]  # little-endian uint16 pairs
    coarse = []
    for b in range(16):
        sw = sum(words[8 * b : 8 * b + 8]) & 0xFFFFFFFF
        coarse.append((sw & 0xFFFF) + (sw >> 16))
    assert coarse == [sum(fine[16 * b : 16 * b + 16]) for b in range(16)], "r8_hist: a carry between the halves"
    return fine, coarse


def radix_round(r: int, digit_bits: int, cores: list, mask: int, prefix: int, k_rem: int, sums: list,
                above: list | None = None, stop=None):  # fmt: skip
    """one radix round of the coordinator (ix_dm0.cpp) over the cores' compacted keys cores[c]: digit_bits 4 -- the
    16-bin reports summed in core order (sum_hist), the digit picked; 8 (R8) -- the coarse reports (r8_hist) summed and
    picked, then the pulled bucket's fine counts (r8_fine) summed and picked: the 4-bit rounds 2 r and 2 r + 1. Each
    level's sums are appended to `sums`; above[c] += core c's keys above the picked digit(s); stop(sum of the first
    level) -> True ends the round before the pick (the merge's done check). -> (prefix, mask, k_rem) or None (stopped)
    """
    shift = 32 - digit_bits * (r + 1)
    if digit_bits == 4:
        hist = [[0] * 16 for _ in cores]
        for c, keys in enumerate(cores):
            for kk in keys:
                if (kk & mask) == prefix:
                    hist[c][(kk >> shift) & 15] += 1
        fine = None
    else:
        fine, hist = zip(*(r8_hist(keys, mask, prefix, shift) for keys in cores)) if cores else ((), ())
    hs = [sum(h[b] for h in hist) for b in range(16)]
    sums.append(hs)
    if stop is not None and stop(sum(hs)):
        return None
    d, ab = pick_digit(hs, k_rem)
    k_rem -= ab
    if above is not None:
        for c, h in enumerate(hist):
            above[c] += sum(h[d + 1 :])
    if fine is not None:  # R8: the coordinator reads bucket d's 16 fine counts of every core (its slot), the same pick
        pulled = [f[16 * d : 16 * d + 16] for f in fine]
        hf = [sum(pl[b] for pl in pulled) for b in range(16)]
        sums.append(hf)
        e, ab = pick_digit(hf, k_rem)
        k_rem -= ab
        if above is not None:
            for c, pl in enumerate(pulled):
                above[c] += sum(pl[e + 1 :])
        d = (d << 4) | e
    return prefix | (d << shift), mask | (((1 << digit_bits) - 1) << shift), k_rem


def local_select(core_keys: list, k: int = 512, digit_bits: int = 4) -> dict:
    """the shard local select of one sparse token on one chip's token lane, step by step: ix_dm0.cpp (the coordinator)
    + att_dsa_shard_dm1.cpp phase A (chunk core c holds core_keys[c], its order keys in local order). Per round
    (radix_round: 16-bin reports, or R8's coarse report + the pulled bucket) the digit, the cores' keys above it
    (above[c]) and the compaction; the tie round's counts of keys == T -> (quota, record offset) per core in core order;
    each core keeps its keys > T and its first quota keys == T (local order) and writes them as records (key, c) at its
    offset. -> dict(T, k_rem, above, qpair, n_rec, kept {c: [(key, local index)]}, records, hists, messages)"""
    if digit_bits not in (4, 8):
        raise ValueError(f"digit_bits {digit_bits}: 4 or 8")
    C = len(core_keys)
    prefix, mask, k_rem, above, sums, msgs = 0, 0, k, [0] * C, [], 0
    src = [list(v) for v in core_keys]
    for r in range(32 // digit_bits):
        prefix, mask, k_rem = radix_round(r, digit_bits, src, mask, prefix, k_rem, sums, above)
        msgs += 1
        src = [[kk for kk in v if (kk & mask) == prefix] for v in src]  # the cores' compaction (cand)
    rem, off, qpair = k_rem, 0, []
    for v, ab in zip(src, above):  # the tie round: (quota, record offset) per core
        q = min(sum(1 for kk in v if kk == prefix), rem)
        qpair.append((q, off))
        off += ab + q
        rem -= q
    msgs += 1
    kept, records = {}, []
    for c, keys in enumerate(core_keys):
        quota, kc = qpair[c][0], []
        for i, kk in enumerate(keys):
            take = kk > prefix
            if not take and kk == prefix and quota > 0:
                quota -= 1
                take = True
            if take:
                kc.append((kk, i))
        assert len(kc) == above[c] + qpair[c][0] and len(records) == qpair[c][1], c  # its records at its offset
        kept[c] = kc
        records += [(kk, c) for kk, _ in kc]
    return dict(T=prefix, k_rem=k_rem, above=above, qpair=qpair, n_rec=off, kept=kept, records=records, hists=sums,
                messages=msgs)  # fmt: skip


def sel2_pack(counts: list) -> list:
    """dsa_shard.hpp r8_hist_packed / the pulled fine slice: 16 uint16 counts as 8 little-endian uint32 words"""
    assert len(counts) == 16 and all(0 <= v <= 0xFFFF for v in counts), counts
    return [counts[2 * w] | (counts[2 * w + 1] << 16) for w in range(8)]


def sel2_sum(slots: list) -> list:
    """dsa_shard.hpp sel2_sum: the slots' 8 packed words summed as uint32 (8 accumulators), each word's halves split ->
    16 counts. Exact while every count's total <= 65535 (asserted against the plain sums)"""
    acc = [sum(s[w] for s in slots) & 0xFFFFFFFF for w in range(8)]
    out = [v for a in acc for v in (a & 0xFFFF, a >> 16)]
    want = [sum((s[b // 2] >> (16 * (b % 2))) & 0xFFFF for s in slots) for b in range(16)]
    assert out == want, ("sel2_sum: a carry between the halves", want)
    return out


def sel2_quota_chunks(qpair: list) -> list:
    """ix_dm0.cpp send_own: owner c's mailbox view of the (quota, offset) table -- only the 16-B chunk holding its own
    pair lands (pairs 2 (c // 2) and 2 (c // 2) + 1) -> the pair owner c reads at c QUOTA_W"""
    words = [v for q, o in qpair for v in (q, o)]
    out = []
    for c in range(len(qpair)):
        w0 = (c * 8 & ~15) // 4
        chunk = words[w0 : w0 + 4]
        mine = (c * 8 - (c * 8 & ~15)) // 4
        out.append((chunk[mine], chunk[mine + 1]))
    return out


def local_select_sel2(core_keys: list, k: int = 512) -> dict:
    """rev 73 SEL2 (ix_dm0.cpp + att_dsa_shard_dm1.cpp phase A, GLM53_DSA_SHARD_SEL2 on R8) step by step: per round the
    cores' r8_hist with the coarse counts packed (sel2_pack), the coordinator's packed sum (sel2_sum) and coarse pick,
    the pulled bucket's packed fine sum and the fine pick -- no above counts; the cores compact and count the keys their
    compaction drops upward (gt); the tie round reports (eq, gt) per core -> (quota, record offset) in core order with
    off += gt + quota; each owner reads its pair from its 16-B chunk (sel2_quota_chunks). -> local_select's dict with
    gt in place of above"""
    total = sum(len(v) for v in core_keys)
    if total > 0xFFFF:
        raise ValueError(f"local_select_sel2: {total} keys on the lane > 65535 (sel2_sum's halves)")
    C = len(core_keys)
    prefix, mask, k_rem, msgs, sums, gt = 0, 0, k, 0, [], [0] * C
    src = [list(v) for v in core_keys]
    for r in range(4):
        shift = 32 - 8 * (r + 1)
        fine, coarse = zip(*(r8_hist(v, mask, prefix, shift) for v in src)) if src else ((), ())
        hs = sel2_sum([sel2_pack(list(cz)) for cz in coarse])
        sums.append(hs)
        d, ab = pick_digit(hs, k_rem)
        k_rem -= ab
        hf = sel2_sum([sel2_pack(list(f[16 * d : 16 * d + 16])) for f in fine])
        sums.append(hf)
        e, ab = pick_digit(hf, k_rem)
        k_rem -= ab
        prefix |= ((d << 4) | e) << shift
        mask |= 0xFF << shift
        msgs += 1
        for c, v in enumerate(src):  # the compaction (cand) + the upward drops
            keep = []
            for kk in v:
                if (kk & mask) == prefix:
                    keep.append(kk)
                elif (kk & mask) > prefix:
                    gt[c] += 1
            src[c] = keep
    rem, off, qpair = k_rem, 0, []
    for c, v in enumerate(src):  # the tie round: (eq, gt) per core -> (quota, record offset)
        q = min(sum(1 for kk in v if kk == prefix), rem)
        qpair.append((q, off))
        off += gt[c] + q
        rem -= q
    msgs += 1
    seen = sel2_quota_chunks(qpair)
    assert seen == qpair, "sel2: an owner's chunk does not hold its own pair"
    kept, records = {}, []
    for c, keys in enumerate(core_keys):
        quota, kc = seen[c][0], []
        for i, kk in enumerate(keys):
            take = kk > prefix
            if not take and kk == prefix and quota > 0:
                quota -= 1
                take = True
            if take:
                kc.append((kk, i))
        assert len(kc) == gt[c] + seen[c][0] and len(records) == seen[c][1], c
        kept[c] = kc
        records += [(kk, c) for kk, _ in kc]
    return dict(T=prefix, k_rem=k_rem, gt=gt, qpair=qpair, n_rec=off, kept=kept, records=records, hists=sums,
                messages=msgs)  # fmt: skip


# ------------------------------------------------------------------ the rev-59 global merge on helpers (emulation)
def helper_merge(lists: list, rank: int, k: int = 512, digit_bits: int = 4, kph: int = 1, own=None) -> dict:
    """ix_dm0.cpp (the coordinator) + att_dsa_shard_dm1.cpp merge_help (helper q holds chip q's list), step by step:
    lists[q] = chip q's candidate records [(key, core)] in its (core, local) order. Per round the helpers' 16-bin
    histograms of their compacted candidates are summed in helper order (sum_hist), the digit picked as the local
    select does, the helpers compact; round 0's total m <= k -> done (T_g = 0, every candidate kept); the tie round's
    counts -> per-helper quotas in helper (= chip) order; helper `rank` -> this chip's per-core quotas (its first quota
    keys == T_g in list order). -> dict(T_g, k_rem, done, helper_quota (per chip), core_quota {core: n}, messages,
    hists (the coordinator's per-round sums)).
    Rev 64, kph > 1 (the KPH merge): helper h holds the lists of chips h kph .. h kph + kph - 1 as the kernel does --
    their keys concatenated in one array (list i's after list i - 1's, len[i] each), ONE histogram over the union per
    round, the compaction in place list by list (the lists stay contiguous), the tie report = the kph per-list counts
    (words 0 .. kph - 1 of the helper's slot) -> the quotas in chip order h kph + i; helper rank // kph re-reads chip
    rank's records for the per-core quotas.
    Rev 66, own= (the key-packed y_ag, agk_merge): lists hold keys only ((key, None): the key pages) and the per-core
    quotas come from this chip's own x_ag records -- kph > 1 re-reads them (key and core), kph 1 compares its y_ag keys
    (mkeys) and takes the core ids from the x_ag records (rq) at the same index.
    digit_bits=8 (R8, GLM53_DSA_SHARD_R8): 4 rounds of radix_round's two levels -- each helper's coarse report
    (r8_hist), the coordinator's coarse pick, the pulled bucket's fine counts and the fine pick; `hists` holds both
    levels' sums per round (= the 4-bit run's rounds 2 r, 2 r + 1) and `messages` 4 rounds fewer."""
    if digit_bits not in (4, 8):
        raise ValueError(f"digit_bits {digit_bits}: 4 or 8")
    assert kph >= 1 and len(lists) % kph == 0, (kph, len(lists))
    nh = len(lists) // kph
    mk = [[kk for i in range(kph) for kk, _ in lists[h * kph + i]] for h in range(nh)]  # the helpers' mkeys
    ln = [[len(lists[h * kph + i]) for i in range(kph)] for h in range(nh)]  # their len[i]
    prefix, mask, k_rem, msgs, sums = 0, 0, k, 1, []  # (the start message)
    for r in range(32 // digit_bits):
        # one histogram over each helper's union (its whole compacted array), summed in helper order (sum_hist); round
        # 0's total m <= k -> done (before the pick: no pull)
        step = radix_round(r, digit_bits, mk, mask, prefix, k_rem, sums, stop=(lambda m: m <= k) if r == 0 else None)
        msgs += 1
        if step is None:
            return dict(T_g=0, k_rem=0, done=True, helper_quota=[0] * len(lists), core_quota={}, messages=msgs,
                        hists=sums)  # fmt: skip
        prefix, mask, k_rem = step
        for h in range(nh):  # in-place compaction, list by list: the write index never passes the read index
            v, m, b0 = mk[h], 0, 0
            for i in range(kph):
                mi = 0
                for e in range(ln[h][i]):
                    kk = v[b0 + e]
                    if (kk & mask) == prefix:
                        v[m] = kk
                        m += 1
                        mi += 1
                b0 += ln[h][i]
                ln[h][i] = mi
            del v[m:]
    rem, hq = k_rem, []
    for h in range(nh):  # the tie round: per-list counts of keys == T_g (word i of slot h) -> quotas in chip order
        b0 = 0
        for i in range(kph):
            n = sum(1 for kk in mk[h][b0 : b0 + ln[h][i]] if kk == prefix)
            b0 += ln[h][i]
            hq.append(min(n, rem))
            rem -= hq[-1]
    msgs += 1
    cq, q = {}, hq[rank]  # helper rank // kph: chip rank's records (re-read: the compaction dropped the core ids)
    scan = lists[rank]
    if own is not None:  # rev 68 AGK: the core ids from this chip's x_ag records
        scan = own if kph > 1 else [(kk, core) for (kk, _), (_, core) in zip(lists[rank], own)]
    for kk, core in scan:
        if q == 0:
            break
        if kk == prefix:
            cq[core] = cq.get(core, 0) + 1
            q -= 1
    return dict(T_g=prefix, k_rem=k_rem, done=False, helper_quota=hq, core_quota=cq, messages=msgs, hists=sums)


def serial_merge(lists: list, rank: int, k: int = 512) -> dict:
    """rev 58's serial merge on the coordinator (the reference): the k-th largest key of the union T_g, the first k_rem
    keys == T_g in (chip, core, local) order; this chip's per-core quotas"""
    allk = [(kk, q, core) for q, lst in enumerate(lists) for kk, core in lst]
    if len(allk) <= k:
        return dict(T_g=0, core_quota={})
    T = sorted((kk for kk, _, _ in allk), reverse=True)[k - 1]
    k_rem = k - sum(1 for kk, _, _ in allk if kk > T)
    cq = {}
    for kk, q, core in allk:
        if kk == T and k_rem > 0:
            k_rem -= 1
            if q == rank:
                cq[core] = cq.get(core, 0) + 1
    return dict(T_g=T, core_quota=cq)


# ------------------------------------------------------------------ rev 68: the key-packed all-gather (AGK, the 8x4)
AGK_PPC = HDR[
    "AGK_PPC"
]  # y_ag pages per token under GLM53_DSA_SHARD_AGK (dsa_shard.hpp AGK): the q page + the key page
FSR = header_constants("fabric_stream/fs_read.cpp")  # fs_read.cpp FS_KEYPACK's mirror of the record format
WORDS = TB // 4  # uint32 words per page


def shard_agk(pair: int) -> bool:
    """rev 68 GLM53_DSA_SHARD_AGK (default 0 = rev 65's AG32 y_ag): the key-packed all-gather of the 8x4 shard
    (dsa_shard.hpp AGK, fs_read.cpp / fs_send.cpp FS_KEYPACK, fabric_stream.py keypack=); the pair / the 2x4 never take
    it"""
    return pair > 2 and _flag("GLM53_DSA_SHARD_AGK")


def agk_header_check() -> None:
    """the AGK host mirrors vs the kernels' text (tt/mk_mla.py _shard_header_check): fs_read.cpp's record format =
    dsa_shard.hpp's, one key page holds a full list, the y_ag page helpers / the count as mirrored below"""
    assert (FSR["KP_REC_HDR"], FSR["KP_REC_BYTES"], FSR["KP_CAND_MAX"]) == (REC_HDR, REC_BYTES, CAND_MAX), FSR
    assert AGK_PPC == AG32_QPAGES + 1 and AGK_PPC % 2 == 0 and CAND_MAX * 4 == TB, AGK_PPC
    with open(os.path.join(KDIR, "mla", "dsa_shard.hpp")) as f:
        src = f.read()
    for line in (
        "return (blk * T + t) * AGK_PPC + d / PIECES_PER_PAGE;",
        "{ return (blk * T + t) * AGK_PPC + AG_QPAGES; }",
        "const uint32_t n = n_closed_local(p, q, C);",
        "return n < CAND_MAX ? n : CAND_MAX;",
    ):
        assert line in src, ("dsa_shard.hpp AGK", line)


def n_closed_local(p: int, r: int, C: int) -> int:
    """dsa_shard.hpp n_closed_local: the closed pools (< (p + 1) / 4) that live on chip r of C (pool page u on chip
    u % C)"""
    n = (p + 1) // 4
    fu, rem = divmod(n, POOL_PAGE)
    full = (fu - r + C - 1) // C if fu > r else 0
    return full * POOL_PAGE + (rem if fu % C == r else 0)


def agk_count(p: int, q: int, C: int = 32) -> int:
    """dsa_shard.hpp agk_count: chip q's candidate records of a sparse token at position p -- its local top-512 keeps
    every key below 512 local pools and exactly 512 above (the count the key page does not carry)"""
    return min(CAND_MAX, n_closed_local(p, q, C))


def agy_q_page(T: int, blk: int, t: int, d: int) -> tuple:
    """dsa_shard.hpp agy_q_page / agx_q_off: (page, byte offset) of q tile d's 2-row piece of token t of block blk in
    the key-packed y_ag"""
    n = pieces_per_page(2)
    return (blk * T + t) * AGK_PPC + d // n, (d % n) * piece_bytes(2)


def agk_key_page(T: int, blk: int, t: int) -> int:
    """dsa_shard.hpp agk_key_page: the key page of token t of block blk in the key-packed y_ag"""
    return (blk * T + t) * AGK_PPC + AG32_QPAGES


def x_ag_chunk(records: list, q_words=None, n_rec: int | None = None, stale=None) -> list:
    """one token's AG32 chunk of x_ag as AG32_PPC x 512 uint32 words, as the producers leave it: page 0 = the q-piece
    page (lat_dm1.cpp; q_words, default zeros), the candidate bytes from page AG32_QPAGES: word 0 = the coordinator's
    record count (ix_dm0.cpp; n_rec, default len(records)), record e = (key, chunk core, 0, 0) at byte REC_HDR + e
    REC_BYTES (the group-0 chunk cores, att_dsa_shard_dm1.cpp phase A); every other word = `stale` (an earlier launch's
    leftovers; default 0)"""
    w = [0] * (AG32_PPC * WORDS) if stale is None else list(stale)
    assert len(w) == AG32_PPC * WORDS
    if q_words is not None:
        w[:WORDS] = list(q_words)
    base = AG32_QPAGES * WORDS
    w[base : base + 4] = [len(records) if n_rec is None else n_rec, 0, 0, 0]
    for e, (kk, core) in enumerate(records):
        i = base + (REC_HDR + e * REC_BYTES) // 4
        w[i : i + 4] = [kk, core, 0, 0]
    return w


def keypack_chunk(words: list) -> list:
    """fs_read.cpp FS_KEYPACK on one staged x chunk (AG32_PPC pages of words), in place and in the kernel's order: page
    AG32_QPAGES = the n = min(header, 512) record keys as words 0 .. n - 1 (word e is written after record e's key word
    4 + 4 e was read and before any later record's), zeros after; -> the batch (fs_send.cpp ships its first AGK_PPC
    pages = the y_ag chunk)"""
    w = list(words)
    base = AG32_QPAGES * WORDS
    n = min(w[base], CAND_MAX)
    for e in range(n):
        w[base + e] = w[base + (REC_HDR + e * REC_BYTES) // 4]
    for e in range(n, CAND_MAX):
        w[base + e] = 0
    return w


def agk_keys(page: list, n: int) -> list:
    """a merge helper's view of one key page (att_dsa_shard_dm1.cpp merge_help under AGK): its first n words, n =
    agk_count (the kernel's watcher ASSERTs: word n - 1 != 0, word n == 0 below 512)"""
    assert len(page) == WORDS and 0 <= n <= CAND_MAX, (len(page), n)
    assert n == 0 or page[n - 1] != 0, "AGK: a zero key inside the list"
    assert n == CAND_MAX or page[n] == 0, "AGK: the count runs short of the keys"
    return list(page[:n])


def agk_records(chunk: list) -> list:
    """this chip's own records [(key, core)] from its x_ag chunk (the header's count): the per-core tie order"""
    base = AG32_QPAGES * WORDS
    n = min(chunk[base], CAND_MAX)
    return [
        tuple(chunk[base + (REC_HDR + e * REC_BYTES) // 4 : base + (REC_HDR + e * REC_BYTES) // 4 + 2])
        for e in range(n)
    ]


def agk_merge(key_pages: list, counts: list, own_chunk: list, rank: int, k: int = 512, kph: int = 1,
              digit_bits: int = 4) -> dict:  # fmt: skip
    """the global merge of one token on chip `rank` under AGK (att_dsa_shard_dm1.cpp merge_help + ix_dm0.cpp, rev 68):
    every chip's keys from its key page with the position's count, this chip's per-core tie order from its own x_ag
    chunk -> helper_merge's dict (to equal rev 65's helper_merge over the record lists, bitwise; digit_bits 8: R8)"""
    lists = [[(kk, None) for kk in agk_keys(pg, n)] for pg, n in zip(key_pages, counts)]
    return helper_merge(lists, rank, k, digit_bits=digit_bits, kph=kph, own=agk_records(own_chunk))


def emulate_agk_launch(keys_by_token: list, pos_by_token: list, lanes: int, kph: int, *, ranks=(0,), chunks: int = 64,
                       seed: int = 0, r8: bool = False) -> list:  # fmt: skip
    """one t-row launch on the 8x4 under AGK, host-side, against rev 65: per token t (lane t % lanes) its 32 chips'
    local top-512 records over the lane's chunk cores (lane_records), every chip's x_ag (AG32 chunks, stale words past
    the records), AG1 with the key pack + AG2 = y_ag (identical on every chip: block ag_blk(q), token t = the first
    AGK_PPC pages of keypack_chunk(chip q's chunk t); every page written once), then on each chip of `ranks` the q-piece
    placement and the AGK merge vs helper_merge over the records (and the serial merge) -> [per token dict(p, lane,
    counts, ref {rank: dict}, got {rank: dict}, serial {rank: dict} [, r8 {rank: dict}: the AGK merge in 8-bit digits
    (GLM53_DSA_SHARD_R8)])]. keys_by_token[t]: int64 order keys of token t's (p + 1) // 4 closed pools (global pool
    ids)"""
    import random

    import torch

    geo, C = Geometry(8, 4), 32
    T = len(keys_by_token)
    assert T == len(pos_by_token) and not lanes_reasons(T, lanes, chunks, geo.L, 1, nchips=C, kph=kph), (T, lanes, kph)
    rnd = random.Random(seed)
    lists, x_ag = [], {q: {} for q in range(C)}
    for t, (keys, p) in enumerate(zip(keys_by_token, pos_by_token)):
        assert int(keys.numel()) == (p + 1) // 4 and p + 1 > 2051, (t, p)
        g = torch.arange(int(keys.numel()))
        per = []
        for q in range(C):
            mine = torch.nonzero((g // POOL_PAGE) % C == q).reshape(-1)
            lq = (mine // POOL_PAGE // C) * POOL_PAGE + mine % POOL_PAGE
            rec = lane_records(keys[mine], lq, chunks, lanes, CAND_MAX)
            assert len(rec) == agk_count(p, q, C), ("the local top-512 count", t, q, len(rec), agk_count(p, q, C))
            per.append(rec)
            stale = [rnd.randrange(1, 1 << 32) for _ in range(AG32_PPC * WORDS)]
            q_words = [(q << 20) ^ (t << 12) ^ i for i in range(WORDS)]
            x_ag[q][t] = x_ag_chunk(rec, q_words=q_words, stale=stale)
        lists.append(per)
    y_ag = {}
    for q in range(C):  # AG1 (the column, local copy on) + AG2 (the row bundles): every chip's packed chunk lands
        blk = geo.ag_blk(q)
        for t in range(T):
            packed = keypack_chunk(x_ag[q][t])
            for i in range(AGK_PPC):
                pg = (blk * T + t) * AGK_PPC + i
                assert pg not in y_ag, ("y_ag page written twice", q, t, i)
                y_ag[pg] = packed[i * WORDS : (i + 1) * WORDS]
    assert sorted(y_ag) == list(range(C * T * AGK_PPC)), "y_ag: every page of 32 blocks x T tokens"
    out = []
    for t, p in enumerate(pos_by_token):
        counts = [agk_count(p, q, C) for q in range(C)]
        for q in range(C):  # ix_dm1's q assembly reads the piece where AG1 put x_ag's q page
            for d in (0, 7, 15):
                pg, off = agy_q_page(T, geo.ag_blk(q), t, d)
                xpg, xoff = agx_q_page(geo, T, 0, t, d)
                assert (
                    off == xoff
                    and y_ag[pg] == x_ag[q][t][(xpg - t * AG32_PPC) * WORDS : (xpg - t * AG32_PPC + 1) * WORDS]
                )
        pages = [y_ag[agk_key_page(T, geo.ag_blk(q), t)] for q in range(C)]
        res = dict(p=p, lane=t % lanes, counts=counts, ref={}, got={}, serial={})
        for rank in ranks:
            res["ref"][rank] = helper_merge(lists[t], rank, CAND_MAX, kph=kph)
            res["got"][rank] = agk_merge(pages, counts, x_ag[rank][t], rank, CAND_MAX, kph=kph)
            res["serial"][rank] = serial_merge(lists[t], rank, CAND_MAX)
            if r8:
                res.setdefault("r8", {})[rank] = agk_merge(pages, counts, x_ag[rank][t], rank, CAND_MAX, kph=kph,
                                                           digit_bits=8)  # fmt: skip
        out.append(res)
    return out


# ------------------------------------------------------------------ the AGK block chunks (GLM53_DSA_SHARD_AGB, on AGK)
def shard_agb(pair: int) -> bool:
    """GLM53_DSA_SHARD_AGB (default 0; needs GLM53_DSA_SHARD_AGK=1): y_ag as one 32-row chunk per block (2 t pages, the
    same page ids as AGK's t chunks of 2 pages) -- one fabric signal per block instead of per token, AG2 in 8-page
    batches (tt/tp_dsa_shard.py, fabric_stream.py keypack=(.., .., t), fs_send.cpp FS_KP_XPER); the 8x4 only"""
    on = pair > 2 and _flag("GLM53_DSA_SHARD_AGB")
    if on and not shard_agk(pair):
        raise ValueError("GLM53_DSA_SHARD_AGB=1 needs GLM53_DSA_SHARD_AGK=1 (the block chunks hold packed keys)")
    return on


def agb_ag1_sends(T: int, blk: int) -> list:
    """fs_send.cpp FS_KEYPACK + FS_KP_XPER = T on chip block `blk` (one y chunk = the block, row_base = blk): per x chunk
    (token) c and page pair col: (y page of the pair's first page, signal) -- g0 = (row_base XPER + c) YP + col, the
    signal on the block's last pair only"""
    yp, out = AGK_PPC, []
    for c in range(T):
        for col in range(0, yp, 2):
            out.append(((blk * T + c) * yp + col, col + 2 == yp and (c + 1) % T == 0))
    return out


def agb_ag2_sends(T: int, blk: int, nb: int) -> list:
    """fs_send.cpp (plain) on AG2's bundle chunk = block `blk` of y_ag [32 x 32 rows, 32 AGK_PPC T]: PPC = AGK_PPC T pages
    per chunk in nb-page batches -> (y page, signal) per page pair"""
    ppc = AGK_PPC * T
    assert ppc % nb == 0 and nb % 2 == 0, (ppc, nb)
    out = []
    for idx in range(0, ppc, 2):
        col = idx % ppc
        out.append((blk * ppc + col, col + 2 == ppc))
    return out


def ag_stream_counts(t: int, agk: bool = False, agb: bool = False, P: int = 8, nb2: int = 8) -> dict:
    """staging batches (one DRAM read barrier + one send flush each) and fabric signals per source of one launch's AG1 /
    AG2 lanes on the 8x4 (tt/tp_dsa_shard.py streams): rev 65 / AGK one chunk per batch and per signal (t, P t); AGB
    one signal per block (1, P) and AG2 batches of nb2 pages"""
    if not agb:
        return dict(ag1_batches=t, ag2_batches=P * t, ag1_signals=t, ag2_signals=P * t)
    ppc = AGK_PPC * t
    return dict(ag1_batches=t, ag2_batches=P * ppc // nb2, ag1_signals=1, ag2_signals=P)


def ag_stream_args(t: int, agk: bool = False, agb: bool = False, P: int = 8) -> dict:
    """the 8x4 shard's AG1 / AG2 FabricStreamAG arguments (tt/tp_dsa_shard.py builds its streams from these, the plan
    Geometry.streams / block from the same numbers): rows (rows_local: 32 per chunk), width (hidden) and the extras
    (keypack=, nb=, slots=). Rev 65: AG32 chunks of t tokens; AGK: the packed 2-page chunks, keypack (6, 1); AGB: one
    block chunk of 2 t pages (rows 32, keypack (6, 1, t), AG2 8-page batches in 3 slots)"""
    assert agk or not agb, "GLM53_DSA_SHARD_AGB on GLM53_DSA_SHARD_AGK"
    w32 = 32 * AG32_PPC
    if not agk:
        return dict(ag1=dict(rows=t * 32, width=w32, extra={}), ag2=dict(rows=P * t * 32, width=w32, extra={}))
    if not agb:
        return dict(ag1=dict(rows=t * 32, width=32 * AGK_PPC, extra=dict(keypack=(AG32_PPC, AG32_QPAGES))),
                    ag2=dict(rows=P * t * 32, width=32 * AGK_PPC, extra={}))  # fmt: skip
    yp = AGK_PPC * t
    nb = 8 if yp % 8 == 0 else (4 if yp % 4 == 0 else 2)
    return dict(ag1=dict(rows=32, width=32 * yp, extra=dict(keypack=(AG32_PPC, AG32_QPAGES, t))),
                ag2=dict(rows=P * 32, width=32 * yp, extra=dict(nb=nb, slots=3)))  # fmt: skip


def fs_ag_block(p: int, line: int, n: int, chunks: int, bundle: bool, local_copy: bool = True) -> tuple:
    """tt/fabric_stream.py FabricStreamAG._block (checked on its text): (src chunk base, dst chunk base = row_base,
    local write, up range, down range) of the chip at position p of line `line`, chunks = rows_local / 32"""
    with open(os.path.join(os.path.dirname(KDIR), "fabric_stream.py")) as f:
        src = f.read()
    for line_ in ("return p * self.chunks, p * self.chunks, False, self.n - 1 - p, p",
                  "return 0, (line * self.n + p) * self.chunks, self.local_copy, self.n - 1 - p, p",
                  "super().__init__(mesh, rows_local // 32, hidden, blocks * rows_local, **kw)"):  # fmt: skip
        assert line_ in src, ("fabric_stream.py FabricStreamAG", line_)
    if bundle:
        return p * chunks, p * chunks, False, n - 1 - p, p
    return 0, (line * n + p) * chunks, bool(local_copy), n - 1 - p, p


def audit_ag_blocks(geo, t: int, agk: bool = False, agb: bool = False, plan_knobs: dict | None = None) -> None:
    """tt/tp_dsa_shard.py TPMlaShard._audits' block check for the AG streams, host-only: every chip's AG1 / AG2 block
    as the fabric stream computes it from ag_stream_args (fs_ag_block) == the plan's Geometry.block; raises the
    audit's ValueError. plan_knobs: the (agk, agb) the plan is asked with (default: the streams'), to reproduce a
    stream / plan mismatch"""
    a = ag_stream_args(t, agk=agk, agb=agb, P=geo.P)
    pk = dict(agk=agk, agb=agb) if plan_knobs is None else plan_knobs
    for r in range(geo.R):
        for c in range(geo.C):
            for name, axis, bundle in (("ag1", geo.pair_axis, False), ("ag2", geo.line_axis, True)):
                p_, line = geo.axis_pos(r, c, axis)
                src, dst, local, up, dn = fs_ag_block(p_, line, geo.axis_len(axis), a[name]["rows"] // 32, bundle)
                got = dict(p=p_, src=src, dst=dst, local=bool(local), up=up, dn=dn, dsts=[])
                want = geo.block(name, t, r, c, **pk)
                if got != {k: want[k] for k in got}:
                    raise ValueError(f"chip {(r, c)} stream {name}: blocks {got} differ from the plan {want}")


# ------------------------------------------------------------------ the scatters in block chunks (GLM53_DSA_SHARD_RSB)
FS_NB, FS_SLOTS = 16, 4  # tt/fabric_stream.py _FabricStream's default nb= / slots= (checked on its text: fs_stage)


def shard_rsb(pair: int) -> bool:
    """GLM53_DSA_SHARD_RSB (default 0): the 8x4 reduce-scatters in BLOCK CHUNKS -- RS1 (the column scatter of 8-row
    slices) and RS2 (the row scatter of 2-row slices) view each destination block (t tokens x RS2_PPC / RS2B_PPC pages,
    the same page ids) as ONE 32-row chunk, so fs_send.cpp's chunk-end packet (a fused write + a FLUSHED atomic inc of
    the receiver's ARR counter: the receiving router drains its NoC writes before the inc) goes out once per
    destination instead of once per token, and the waiter waits for one per source (tt/tp_dsa_shard.py,
    rs_stream_args). No kernel change: the scatter's x and y chunks coincide, so fs_send.cpp's `col + 2 == PPC` is the
    block's last pair once PPC = t RS2_PPC. The 8x4 only (the pair / the 2x4 keep their streams)"""
    return pair > 2 and _flag("GLM53_DSA_SHARD_RSB")


def rs_stream_args(t: int, rsb: bool = False, P: int = 8, rsnp: bool = False) -> dict:
    """the 8x4 shard's RS1 / RS2 FabricStreamScatter arguments (tt/tp_dsa_shard.py builds both scatters and their x
    tensors from these, the plan Geometry.streams(t, rsb=) / block from the same numbers): rows (rows_local: 32 per
    chunk), width (hidden) and the extras (nb=, slots=). Knob off: t chunks of RS2_PPC / RS2B_PPC pages (one per
    token), the default batches (nb = min(16, PPC) = one chunk) in 4 slots -- exactly the pre-RSB constructor calls.
    RSB: one chunk of t PPC pages per destination block; RS1 keeps its 10-page batches in 4 slots (80 KB: the knob-off
    CB), RS2 ships 8-page batches in 2 slots (32 KB: the knob-off CB) when 4 t pages split into them, else the knob-off
    4-page batches in 4 slots. RSNP (GLM53_DSA_SHARD_RSNP, on RSB): the same rows / width (x / y / every page id
    unchanged) plus sub=(PPC, data pages) -- each chunk stages / sends rs_sent_pages(t) pages (t slices' data pages,
    paired across slices) in batches of the largest even divisor up to RSNP_NB_CAP (RS1 12: 3 slots of 12 = 72 KB,
    AG2's 3-slot ring, fewer batches than RSB's 10; RS2 RSB's batch), in RSB's slots or as many as RSB's CB holds"""
    assert P > 2, ("the 8x4 scatters", P)
    assert rsb or not rsnp, "RSNP: on RSB's block chunks"
    if not rsb:
        return dict(rs1=dict(rows=t * 32, width=32 * RS2_PPC, extra={}),
                    rs2=dict(rows=t * 32, width=32 * RS2B_PPC, extra={}))  # fmt: skip
    nb2 = 8 if (RS2B_PPC * t) % 8 == 0 else RS2B_PPC
    slots2 = FS_SLOTS * RS2B_PPC // nb2  # the knob-off bytes: 4 slots x 4 pages
    out = dict(rs1=dict(rows=32, width=32 * RS2_PPC * t, extra=dict(nb=RS2_PPC, slots=FS_SLOTS)),
               rs2=dict(rows=32, width=32 * RS2B_PPC * t, extra=dict(nb=nb2, slots=slots2)))  # fmt: skip
    if rsnp:
        for name, stride, data in (("rs1", RS2_PPC, RS1_DATA_PAGES), ("rs2", RS2B_PPC, RS2B_DATA_PAGES)):
            e, spc = out[name]["extra"], rs_sent_pages(t, data)
            nb = max(k for k in range(2, RSNP_NB_CAP.get(name, e["nb"]) + 1, 2) if spc % k == 0)
            slots = min(e["slots"], e["slots"] * e["nb"] // nb)  # the staging CB within RSB's bytes
            out[name]["extra"] = dict(nb=nb, slots=slots, sub=(stride, data))
    return out


# ------------------------------------------------------------------ the scatters without pad pages (GLM53_DSA_SHARD_RSNP)
RSNP_NB_CAP = {"rs1": 12}  # RSNP's RS1 batch bound (rs_stream_args; RS2: RSB's batch)


def rs_slice_bytes(rows: int, otb: int = 4096, ptb: int = 4096) -> int:
    """bytes of one 8x4 RS slice (rs_layout: the DT O tiles then the ST stats tiles, `rows` rows of each): the roots'
    att_dsa_shard_dm1.cpp writes / landing reads cover exactly [0, DT ob + ST pb) of the slice chunk (rsnp_header_check);
    fp32 O and stats (the largest tiles) 18 KB (8 rows) / 4.5 KB (2 rows); bf16 O 10 / 2.5 KB"""
    kind, _, off = rs_layout(rows, otb, ptb)[-1]
    assert kind == "s"
    return off + ptb * rows // 32


RS1_DATA_PAGES = -(-rs_slice_bytes(8) // TB)  # 9 of RS1_PPC = 10: the 8-row slice's pages, then one pad page
RS2B_DATA_PAGES = -(-rs_slice_bytes(2) // TB)  # 3 of RS2B_PPC = 4: the 2-row slice's pages, then one pad page
assert (RS1_DATA_PAGES, RS2B_DATA_PAGES) == (RS2_PPC - 1, RS2B_PPC - 1), (RS1_DATA_PAGES, RS2B_DATA_PAGES)


def shard_rsnp(pair: int) -> bool:
    """GLM53_DSA_SHARD_RSNP (default 0; needs GLM53_DSA_SHARD_RSB=1): the 8x4 scatters WITHOUT PAD PAGES. A slice
    chunk is RS2_PPC = 10 / RS2B_PPC = 4 pages for 9 / 3 data pages (rs_slice_bytes) because fs_send.cpp ships PAGE
    PAIRS (one 2 x 2 KiB fabric scatter packet) and a pair never straddled a chunk; the roots never write the pad page
    and no reader reads it. On RSB's block chunks the lane stages only the data pages and pairs them across slices
    (fs_read.cpp / fs_send.cpp FS_SUBPC / FS_SUBD, fabric_stream.py sub=): every data page lands at its knob-off page
    with the same bytes, the chunk-end signal stays on each destination's last pair; x / y / the roots unchanged. RS1 10
    t -> 9 t, RS2 4 t -> 3 t pages per destination (t even; odd t keeps the last slice's pad page). The 8x4 only"""
    on = pair > 2 and _flag("GLM53_DSA_SHARD_RSNP")
    if on and not shard_rsb(pair):
        raise ValueError("GLM53_DSA_SHARD_RSNP=1 needs GLM53_DSA_SHARD_RSB=1 (the pad-free pairs span a block chunk)")
    return on


def rs_sent_pages(t: int, data: int) -> int:
    """pages a destination's block chunk of t slices stages / sends under RSNP: t data pages per slice, rounded up to a
    pair (fs_send.cpp / fs_read.cpp DPC: an odd count keeps the last slice's first pad page)"""
    return -(-(t * data) // 2) * 2


def fs_sub_col(s: int, ppc: int, sub: tuple) -> int:
    """fs_send.cpp / fs_read.cpp sub_col (FS_SUBPC, FS_SUBD = sub; checked on their text: fs_stage): staged page s of a
    chunk of ppc pages -> its page in the chunk (sub-chunk s // data, data page s % data; s = the filler of an odd count
    -> the last sub-chunk's first pad page)"""
    stride, data = sub
    k = s // data
    return k * stride + (s - k * data) if k < ppc // stride else ppc - stride + data


def rsnp_header_check() -> None:
    """the RSNP host mirrors vs the kernels' text: the roots write / land a slice as DT O runs of ob = 2 slice_run(rows,
    otb) bytes then ST stats runs of pb = 2 slice_run(rows, ptb) bytes from byte 0 of its chunk (att_dsa_shard_dm1.cpp
    -- rs_slice_bytes, so the pad pages hold nothing anyone reads); fs_read.cpp and fs_send.cpp stage / send the same
    DPC pages per chunk through the same sub_col"""
    with open(os.path.join(KDIR, "mla", "att_dsa_shard_dm1.cpp")) as f:
        dm1 = f.read()
    for line_ in ("const uint32_t ob = 2 * mlashard::slice_run(mlashard::RS1_ROWS, otb);",
                  "const uint32_t pb = 2 * mlashard::slice_run(mlashard::RS1_ROWS, ptb);",
                  "ASSERT(DT * ob + ST * pb <= mlashard::RS1_PPC * mlashard::TB);",
                  "const uint32_t ob = 2 * mlashard::slice_run(mlashard::RS2_ROWS, otb);",
                  "ASSERT(DT * ob + ST * pb <= mlashard::RS2B_PPC * mlashard::TB);",
                  "const uint32_t b = DT * ob + i * pb;", "const uint32_t b = d * ob;",
                  "const uint32_t ob = 2 * mlashard::slice_run(rows, otb), pb = 2 * mlashard::slice_run(rows, ptb);",
                  "const uint32_t sl1 = 2 * mlashard::slice_run(mlashard::RS1_ROWS, otb);",
                  "const uint32_t sl2 = 2 * mlashard::slice_run(mlashard::RS2_ROWS, otb);"):  # fmt: skip
        assert line_ in dm1, ("att_dsa_shard_dm1.cpp RS slices", line_)
    with open(os.path.join(KDIR, "mla", "dsa_shard.hpp")) as f:
        assert "constexpr uint32_t RS1_PPC = RS2_PPC;" in f.read(), "dsa_shard.hpp RS1_PPC"
    assert slice_run(8, 4096) * 2 * (DT + ST) == rs_slice_bytes(8) == 18432
    assert slice_run(2, 4096) * 2 * (DT + ST) == rs_slice_bytes(2) == 4608
    for k in ("fs_send.cpp", "fs_read.cpp"):
        with open(os.path.join(KDIR, "fabric_stream", k)) as f:
            src = f.read()
        for line_ in ("constexpr uint32_t DPC = (SUB_N * FS_SUBD + 1) / 2 * 2;",
                      "return k < SUB_N ? k * FS_SUBPC + (s - k * FS_SUBD) : PPC - FS_SUBPC + FS_SUBD;",
                      "nbat = c_n * DPC / NB;"):  # fmt: skip
            assert line_ in src, (k, line_)


def fs_stage(rows: int, width: int, extra: dict) -> dict:
    """tt/fabric_stream.py _FabricStream (no keypack) from (rows_local, hidden, extras), checked on its text: chunks per
    block, pages per chunk, the effective batch (nb = min(nb, ppc)), slots, the staging CB bytes (slots x nb pages);
    spc = the pages each chunk stages / sends (ppc; sub= (GLM53_DSA_SHARD_RSNP): the data pages, rs_sent_pages), sub"""
    with open(os.path.join(os.path.dirname(KDIR), "fabric_stream.py")) as f:
        src = f.read()
    for line_ in (f"lanes=None, nb: int = {FS_NB}, slots: int = {FS_SLOTS},", "nb = min(nb, self.ppc)",
                  "total_size=self.slots * xnb * TB,", "spc = -(-(self.ppc // stride * data) // 2) * 2",
                  'sd = [("FS_SUBPC", str(self.sub[0])), ("FS_SUBD", str(self.sub[1]))]'):  # fmt: skip
        assert line_ in src, ("fabric_stream.py _FabricStream", line_)
    ppc = width // 32
    nb = min(int(extra.get("nb", FS_NB)), ppc)
    slots = int(extra.get("slots", FS_SLOTS))
    sub = tuple(int(v) for v in extra["sub"]) if extra.get("sub") is not None else None
    spc = ppc
    if sub is not None:
        assert ppc % sub[0] == 0 and 0 < sub[1] < sub[0], (ppc, sub)
        spc = rs_sent_pages(ppc // sub[0], sub[1])
    assert width % 32 == 0 and spc % nb == 0 and nb % 2 == 0, (width, nb, sub)
    return dict(chunks=rows // 32, ppc=ppc, nb=nb, slots=slots, cb_bytes=slots * nb * TB, spc=spc, sub=sub)


def fs_scatter_block(p: int, n: int, chunks: int, first: list | None = None) -> tuple:
    """tt/fabric_stream.py FabricStreamScatter._block / _dsts (checked on their text): (dst chunk base = row_base, local
    write, up range, down range, destinations [(q, source chunk base of block q, direction 0 up / 1 down, hops)]) of the
    chip at position p of a line of n, chunks = rows_local / 32; first= (RS1 early): those destinations first"""
    with open(os.path.join(os.path.dirname(KDIR), "fabric_stream.py")) as f:
        src = f.read()
    for line_ in ("return 0, p * self.chunks, False, self.n - 1 - p, p",
                  "out = [(q, q * self.chunks, 0 if q > p else 1, abs(q - p)) for q in range(self.n) if q != p]",
                  "out = [d for d in out if d[0] in early] + [d for d in out if d[0] not in early]",
                  "super().__init__(mesh, rows_local // 32, hidden, n * rows_local, **kw)"):  # fmt: skip
        assert line_ in src, ("fabric_stream.py FabricStreamScatter", line_)
    dsts = [(q, q * chunks, 0 if q > p else 1, abs(q - p)) for q in range(n) if q != p]
    if first is not None:
        early = set(first)
        dsts = [d for d in dsts if d[0] in early] + [d for d in dsts if d[0] not in early]
    return p * chunks, False, n - 1 - p, p, dsts


def fs_scatter_sends(rows: int, width: int, extra: dict, p: int, dsts: list) -> list:
    """fs_read.cpp + fs_send.cpp FS_SCATTER on the scatter's one lane (c_lo = 0, c_n = the block's chunks) of the chip
    at position p: per destination in `dsts` order ([(q, source chunk base, direction, hops)]) every page pair in send
    order -> (q, x page, y page, signal): x page = fs_read.cpp's (base + c_lo) PPC + b NB + i (the staged pair), y page
    = fs_send.cpp's g0 = (row_base + chunk) PPC + col with row_base = p chunks (the pair lands at g0, g0 + 1 of q's y),
    signal = the chunk-end packet (`col + 2 == PPC`: the fused write + flushed atomic inc of q's ARR counter)"""
    st = fs_stage(rows, width, extra)
    chunks, ppc, nb = st["chunks"], st["ppc"], st["nb"]
    row_base, c_lo, c_n = p * chunks, 0, chunks
    out = []
    for q, base, _, _ in dsts:
        for b in range(c_n * ppc // nb):
            for i in range(0, nb, 2):
                idx = b * nb + i
                chunk, col = c_lo + idx // ppc, idx % ppc
                out.append((q, (base + c_lo) * ppc + b * nb + i, (row_base + chunk) * ppc + col, col + 2 == ppc))
    return out


def fs_scatter_packets(rows: int, width: int, extra: dict, p: int, dsts: list) -> list:
    """fs_read.cpp then fs_send.cpp FS_SCATTER, with or without FS_SUBPC / FS_SUBD (extra sub=: GLM53_DSA_SHARD_RSNP),
    line by line on the scatter's one lane (c_lo = 0, c_n = the block's chunks) of the chip at position p: per
    destination in `dsts` order fs_read stages its batches (x pages in CB order), fs_send pops them as pairs -> per
    packet (q, (x page of each staged page), (y page each lands on), signal = the chunk-end packet: `col + 2 == PPC` /
    DPC). Knob off / RSB: the pairs of fs_scatter_sends (y pages g0, g0 + 1)"""
    st = fs_stage(rows, width, extra)
    chunks, ppc, nb, spc, sub = st["chunks"], st["ppc"], st["nb"], st["spc"], st["sub"]
    row_base, c_lo, c_n = p * chunks, 0, chunks
    nbat = c_n * spc // nb
    out = []
    for q, base, _, _ in dsts:
        staged = []  # fs_read.cpp: the lane's staged pages of destination q, in CB order
        for b in range(nbat):
            for i in range(nb):
                idx = b * nb + i
                if sub is None:
                    staged.append((base + c_lo) * ppc + b * nb + i)
                else:
                    staged.append((base + c_lo + idx // spc) * ppc + fs_sub_col(idx % spc, ppc, sub))
        for b in range(nbat):  # fs_send.cpp: the same batches, a pair per packet
            for i in range(0, nb, 2):
                idx = b * nb + i
                chunk, col = c_lo + idx // spc, idx % spc
                if sub is None:
                    g0 = (row_base + chunk) * ppc + col
                    g1 = g0 + 1
                else:
                    g0 = (row_base + chunk) * ppc + fs_sub_col(col, ppc, sub)
                    g1 = (row_base + chunk) * ppc + fs_sub_col(col + 1, ppc, sub)
                out.append((q, (staged[idx], staged[idx + 1]), (g0, g1), col + 2 == spc))
    return out


def rs_stream_counts(t: int, rsb: bool = False, P: int = 8, L: int = 4, rsnp: bool = False) -> dict:
    """per source chip and launch, the 8x4 scatters' staging batches (one DRAM read barrier + one send flush each),
    fabric packets (2 pages each), chunk-end signals (flushed atomic incs at the receivers) and the waiter's count per
    source lane (fs_send.cpp n_lane_ch: chunks per lane), plus each lane's staging CB bytes; bytes = the payload a
    source sends, mid_bytes = the payload one direction of the line's middle link carries (the n / 2 sources below it
    to the n / 2 destinations above: RS1's bound, design section 10)"""
    a = rs_stream_args(t, rsb=rsb, P=P, rsnp=rsnp)
    out = {}
    for name, n in (("rs1", P), ("rs2", L)):
        st = fs_stage(a[name]["rows"], a[name]["width"], a[name]["extra"])
        pages = (n - 1) * st["chunks"] * st["spc"]
        out[name] = dict(batches=pages // st["nb"], packets=pages // 2, signals=(n - 1) * st["chunks"],
                         wait_per_source=st["chunks"], cb_bytes=st["cb_bytes"], bytes=pages * TB,
                         mid_bytes=(n // 2) ** 2 * st["chunks"] * st["spc"] * TB)  # fmt: skip
    return out


def audit_rs_blocks(geo, t: int, rsb: bool = False, rs1_early: bool = False, plan_knobs: dict | None = None) -> None:
    """tt/tp_dsa_shard.py TPMlaShard._audits' block check for the 8x4 scatters, host-only: every chip's RS1 / RS2 block
    as FabricStreamScatter computes it from rs_stream_args (fs_scatter_block) == the plan's Geometry.block; raises the
    audit's ValueError. plan_knobs: the (rsb, rs1_early) the plan is asked with (default: the streams'), to reproduce a
    stream / plan mismatch"""
    assert geo.P > 2, ("the 8x4 scatters", geo.P)
    a = rs_stream_args(t, rsb=rsb, P=geo.P)
    pk = dict(rsb=rsb, rs1_early=rs1_early) if plan_knobs is None else plan_knobs
    for r in range(geo.R):
        for c in range(geo.C):
            for name, axis in (("rs1", geo.pair_axis), ("rs2", geo.line_axis)):
                p_, _ = geo.axis_pos(r, c, axis)
                first = rs1_early_first(p_, geo.P) if (rs1_early and name == "rs1") else None
                dst, local, up, dn, dsts = fs_scatter_block(p_, geo.axis_len(axis), a[name]["rows"] // 32, first)
                got = dict(p=p_, src=None, dst=dst, local=bool(local), up=up, dn=dn, dsts=dsts)
                want = geo.block(name, t, r, c, **pk)
                if got != {k: want[k] for k in got}:
                    raise ValueError(f"chip {(r, c)} stream {name}: blocks {got} differ from the plan {want}")
