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
TL_MAX = HDR["TL_MAX"]  # rev 60: tokens per token lane (the lane-local mailbox / kept blocks / CB_ASEL pages)
MAX_CHUNKS = 64  # dsa_common.hpp MAX_CHUNKS
CAND_MAX = 512  # dsa_shard.hpp CAND_MAX = mladsa::TOPK (not a plain literal there)
CAND_PAGES = -(-(REC_HDR + CAND_MAX * REC_BYTES) // TB)
assert AG8_QPAGES + CAND_PAGES <= AG8_PPC and AG8_PPC % 2 == 0 and RS2_PPC % 2 == 0 and RS_PPC % 2 == 0
assert RS_CHUNKS * RS_PPC == 36  # rev 59: the RS round-1 message = 16 fp32 O tiles + 2 stats tiles = 36 pages exactly


def merge_msgs() -> int:
    """dsa_shard.hpp MERGE_MSGS = 1 + ROUNDS + 2 (start, the radix rounds, 2 tie-quota slots), checked on the text"""
    with open(os.path.join(KDIR, "mla", "dsa_shard.hpp")) as f:
        assert "constexpr uint32_t MERGE_MSGS = 1 + mladsa::ROUNDS + 2;" in f.read()
    return 1 + 8 + 2


def merge_scr_bytes() -> int:
    """dsa_shard.hpp MERGE_SCR_BYTES: a helper's chip candidate pages + keys + compacted candidates + per-core quotas"""
    with open(os.path.join(KDIR, "mla", "dsa_shard.hpp")) as f:
        assert "MERGE_SCR_BYTES = CAND_PAGES * TB + 2 * CAND_MAX * 4 + mladsa::MAX_CHUNKS * 4;" in f.read()
    return CAND_PAGES * TB + 2 * CAND_MAX * 4 + 64 * 4


PIECE_ROWS = 8  # TP8: q / o rows of one chip in a 32-head tile (dsa_shard.hpp PIECE_ROWS at LINE = 4)
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
        else:
            raise ValueError(f"shard: the pair (1x2 / 2x1) or the 2x4 mesh, not {R}x{C}")

    # ---- ranks
    def rank(self, r: int, c: int) -> int:
        return r * self.C + c

    def coord(self, rank: int) -> tuple:
        return divmod(rank, self.C)

    def pair_pos(self, rank: int) -> int:
        return rank // self.L

    def line_pos(self, rank: int) -> int:
        return rank % self.L

    def ag_blk(self, rank: int) -> int:
        """y_ag block of chip `rank` (kernels/mla/dsa_shard.hpp ag_blk)"""
        return self.line_pos(rank) * 2 + self.pair_pos(rank)

    def pair_peer(self, rank: int) -> int:
        """the other chip of the rank's pair-axis pair (2x4: the column peer)"""
        g, j = self.pair_pos(rank), self.line_pos(rank)
        return (1 - g) * self.L + j

    def line_peers(self, rank: int) -> list:
        g, j = self.pair_pos(rank), self.line_pos(rank)
        return [g * self.L + q for q in range(self.L) if q != j]

    def heads(self, rank: int, hr: int) -> range:
        return range(hr * rank, hr * (rank + 1))

    def tile_ranks(self, g: int) -> list:
        """the chips whose q pieces form head tile g (heads [32 g, 32 g + 32)), in row order"""
        return [g * self.L + j for j in range(self.L)]

    def pass_tile(self, rank: int, pass_: int) -> int:
        """the head tile pass 0 (the OTHER pair position's) / pass 1 (this chip's) attends with"""
        g = self.pair_pos(rank)
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

    def ag_width(self) -> int:
        return 32 * (AG8_PPC if self.L > 1 else PPC)

    def streams(self, t: int) -> dict:
        """name -> dict(axis, kind, chunks (per block), width, link, blocks_in_y, src_blocks_in_x)"""
        out = {"ag1": dict(axis=self.pair_axis, kind="ag", chunks=self.ag_chunks(t), width=self.ag_width(), link=0,
                           y_blocks=self.n, x_blocks=1),
               "rs1": dict(axis=self.pair_axis, kind="rs2", chunks=t * RS_CHUNKS, width=32 * RS_PPC, link=1,
                           y_blocks=1, x_blocks=2)}  # fmt: skip
        if self.L > 1:
            out["ag2"] = dict(
                axis=self.line_axis,
                kind="ag",
                chunks=2 * self.ag_chunks(t),
                width=self.ag_width(),
                link=0,
                y_blocks=self.L,
                x_blocks=self.L,
            )  # (bundles: x = y = y_ag)  # fmt: skip
            out["rs2"] = dict(axis=self.line_axis, kind="scatter", chunks=t, width=32 * RS2_PPC, link=1,
                              y_blocks=self.L, x_blocks=self.L)  # fmt: skip
        return out

    def block(self, name: str, t: int, r: int, c: int) -> dict:
        """chip (r, c)'s block of stream `name`: src chunk base in x, dst chunk base in y (on the receivers: the same
        address on every chip), local copy, up / down multicast ranges (the scatter: the chips above / below), and the
        scatter's destinations [(position q, src chunk base, direction 0 up / 1 down, hops)]"""
        sp = self.streams(t)[name]
        p, line = self.axis_pos(r, c, sp["axis"])
        n = self.axis_len(sp["axis"])
        ch = sp["chunks"]
        up, dn = n - 1 - p, p
        if name == "ag1":  # this chip's block -> y_ag block ag_blk (line-major: blk = line n + p)
            return dict(p=p, src=0, dst=(line * n + p) * ch, local=True, up=up, dn=dn, dsts=[])
        if name == "ag2":  # the column's 2-block bundle, already in y_ag at bundle p, multicast along the row
            return dict(p=p, src=p * ch, dst=p * ch, local=False, up=up, dn=dn, dsts=[])
        if name == "rs1":  # the pair exchange: the peer's block of x_rs1 -> its y_rs1
            return dict(p=p, src=(1 - p) * ch, dst=0, local=False, up=up, dn=dn, dsts=[])
        if name == "rs2":  # per-destination unicast: x_rs2 block q -> peer q's y_rs2 block p
            dsts = [(q, q * ch, 0 if q > p else 1, abs(q - p)) for q in range(n) if q != p]
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
    return (j // 2) * (tb // 2) + (j % 2) * (tb // 8)


def rs2_layout(otb: int, ptb: int = 4096) -> list:
    """[(kind, index, byte offset in the slice chunk)] of one x_rs2 / y_rs2 slice: the DT O tiles then the ST stats
    tiles, tb / 4 bytes each (the two runs of the 8 rows)"""
    out, b = [], 0
    for d in range(DT):
        out.append(("o", d, b))
        b += otb // 4
    for i in range(ST):
        out.append(("s", i, b))
        b += ptb // 4
    assert b <= RS2_PPC * TB, b
    return out


# ------------------------------------------------------------------ byte-level emulation of the q / slice formats
def tile_to_faces(x):
    """a [32, 32] tile (torch) -> its face-major element order (faces 0..3 of 16 x 16, row-major within a face):
    the byte order of a TILE page"""
    return x.reshape(2, 16, 2, 16).permute(0, 2, 1, 3).reshape(-1)


def faces_to_tile(v):
    return v.reshape(2, 2, 16, 16).permute(0, 2, 1, 3).reshape(32, 32)


def q_piece(q_tile_faces):
    """the latent core's TP8 q piece of one bf16 tile (face-major elements): rows 0..7 = face 0 rows 0..7 (256 B) + face
    1 rows 0..7 (256 B) -> 256 elements = 512 B (lat_dm1.cpp)"""
    import torch

    return torch.cat([q_tile_faces[0:128], q_tile_faces[256:384]])


def assemble_q_tile(pieces):
    """the reader's q assembly (att_dsa_shard_dm0.cpp): 4 chips' pieces (each 256 elements: run A, run B) -> one tile
    in face-major order; piece j -> rows 8 j .. 8 j + 7 (run A at slice_off(j, 2048) / 2 elements, run B at + 512 / 2)
    """
    import torch

    out = torch.zeros(1024, dtype=pieces[0].dtype)
    for j, pc in enumerate(pieces):
        a = slice_off(j, 2048) // 2
        out[a : a + 128] = pc[:128]
        out[a + 256 : a + 256 + 128] = pc[128:]
    return out


def slice_runs(tile_faces, j: int):
    """rows 8 j .. 8 j + 7 of a face-major tile (any element type): (run A, run B)"""
    n = tile_faces.numel()
    a = slice_off(j, n)  # (slice_off is linear in the tile size: elements in, elements out)
    w = n // 8
    return tile_faces[a : a + w], tile_faces[a + n // 4 : a + n // 4 + w]


def land_slice(dst_faces, j: int, runs):
    n = dst_faces.numel()
    a = slice_off(j, n)
    w = n // 8
    dst_faces[a : a + w] = runs[0]
    dst_faces[a + n // 4 : a + n // 4 + w] = runs[1]
    return dst_faces


# ------------------------------------------------------------------ the exchange, emulated in fp32 (torch)
def combine(a, b):
    """kernels/mla/att_c.cpp combine (M = max, exp-weighted l / O), fp32; a, b = (m [rows], l [rows], O [rows, D])"""
    import torch

    m1, l1, o1 = a
    m2, l2, o2 = b
    M = torch.maximum(m1, m2)
    ea, eb = torch.exp(m1 - M), torch.exp(m2 - M)
    return M, ea * l1 + eb * l2, ea[:, None] * o1 + eb[:, None] * o2


def emulate_exchange(geo: Geometry, q, parts_fn, *, junk=None):
    """The TP8 (or pair) exchange of one token, emulated per chip with the kernels' data movement:
      q: [64 heads, D] fp32 (head h on chip h // 8 at TP8, its q piece rows h % 8; the pair: 32 heads per chip);
      parts_fn(rank, q_tile [32, D]) -> the chip's partial (m [32], l [32], O [32, D]) of its OWN selected rows for the
      32 heads of q_tile (the in-chip tree's result).
    AG: each chip's q piece -> every chip (y_ag blocks by ag_blk); each chip assembles tile A / tile B from the 4 pieces.
    RS round 1: the pass-0 partial (the other pair position's tile) -> the pair peer, combined with its pass-1 partial
    (own tile) in the kernel's order combine(own, landed). TP8 round 2: each chip lands rows 8 j .. of its line peers'
    column-combined partials (+ its own slice from x_rs2) into the receive tile (junk rows = the previous message's
    rows: `junk` fills them, default the round-1 landing) and combines own, then the peers in column order.
    -> {rank: o [heads of the chip, D]} (o = O / l)."""
    import torch

    n, L = geo.n, geo.L
    hpc = 64 // n
    pieces = {r: q[hpc * r : hpc * (r + 1)] for r in range(n)}
    y_ag = {geo.ag_blk(r): pieces[r] for r in range(n)}
    tiles = {g: torch.cat([y_ag[geo.ag_blk(rr)] for rr in geo.tile_ranks(g)]) for g in (0, 1)}
    p0 = {r: parts_fn(r, tiles[geo.pass_tile(r, 0)]) for r in range(n)}
    p1 = {r: parts_fn(r, tiles[geo.pass_tile(r, 1)]) for r in range(n)}
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
def lanes_reasons(t: int, lanes: int, chunks: int = 64, line: int = 1, groups: int = 1) -> list:
    """why a t-row launch on `lanes` token lanes cannot run the sharded phase (tt/mk_mla.py shard_lanes_check's rules,
    host-only): [] = it can"""
    why = []
    if t > T_MAX:
        why.append(f"t <= {T_MAX}")
    if groups != 1:
        why.append("one head group")
    if lanes < 1 or lanes & (lanes - 1) or chunks % lanes:
        why.append("lanes: a power of two dividing the chunk cores")
        return why
    if -(-t // lanes) > TL_MAX:
        why.append(f"<= {TL_MAX} tokens per lane")
    cl = chunks // lanes
    if cl < 2 * line:
        why.append(f"{2 * line} merge helpers among {cl} lane chunk cores")
    return why


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


def emulate_lane_merge(keys_global, C: int, chunks: int, lanes: int, k: int = 512) -> dict:
    """one token of a t > 4 launch on `lanes` lanes: every chip's local top-k over its lane's chunk cores, the
    helpers' global merge (helper_merge, the replicas' T_g / helper quotas) -> dict(T_g, per-chip core quotas, lists).
    keys_global: int64 order keys of the token's closed pools (global pool ids 0 .. n - 1; pool g on chip (g // 32) % C,
    local id (g // 32 // C) 32 + g % 32: tt/dsa_shard.py ShardMap)"""
    import torch

    g = torch.arange(int(keys_global.numel()))
    owner = (g // POOL_PAGE) % C
    lists = []
    for r in range(C):
        mine = torch.nonzero(owner == r).reshape(-1)
        lq = (mine // POOL_PAGE // C) * POOL_PAGE + mine % POOL_PAGE
        lists.append(lane_records(keys_global[mine], lq, chunks, lanes, k))
    res = [helper_merge(lists, r, k) for r in range(C)]
    assert len({(x["T_g"], tuple(x["helper_quota"])) for x in res}) == 1, "the replicas disagree"
    return dict(T_g=res[0]["T_g"], core_quota=[x["core_quota"] for x in res], lists=lists)


# ------------------------------------------------------------------ the shard phase probe (rev 59)
PROBE_LANE_ROW = {"ag1": 236, "ag2": 237, "rs1": 238, "rs2": 239}  # fs_send.cpp rows (mk_mla.py SHARD_PROBE_LANE0)
PROBE_ROOT_NC, PROBE_ROOT_BR, PROBE_IX0, PROBE_IX1, PROBE_LAT1 = 132, 0, 220, 221, 131  # the root = owner k = 0
ATT_NC_MARKS = ("att_start", "local_select", "merge", "pass0", "rs1_landed", "pass1", "rs2_landed", "finalize", "end")
ATT_BR_MARKS = ("att_start", "presweep", "pass0_pushed", "end")
IX0_MARKS = ("start", "i1_in", "y_in", "local_coord", "ag_landed", "merge_done")
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
    if len(ix0) >= len(IX0_MARKS):
        out["ix0"] = {k: us(v) for k, v in zip(IX0_MARKS, ix0[-len(IX0_MARKS) :])}
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
    for name, l_ in lanes.items():
        span[f"{name}_open_to_landed"] = l_["landed"] - l_["opened"]
        span[f"{name}_send"] = l_["sent"] - l_["opened"]
    if "qasm" in out:
        span["q_asm"] = out["qasm"]["assembled"] - out["qasm"]["ag_landed"]
    if ar is not None and out["ar"]:
        span["ar_inputs_to_result"] = out["ar"][0]["result"] - out["ar"][0]["inputs"]
    out["span"] = {k: round(v, 2) for k, v in span.items()}
    return out


# ------------------------------------------------------------------ the rev-59 global merge on helpers (emulation)
def helper_merge(lists: list, rank: int, k: int = 512, digit_bits: int = 4) -> dict:
    """ix_dm0.cpp (the coordinator) + att_dsa_shard_dm1.cpp merge_help (helper q holds chip q's list), step by step:
    lists[q] = chip q's candidate records [(key, core)] in its (core, local) order. Per round the helpers' 16-bin
    histograms of their compacted candidates are summed in helper order (sum_hist), the digit picked as the local
    select does, the helpers compact; round 0's total m <= k -> done (T_g = 0, every candidate kept); the tie round's
    counts -> per-helper quotas in helper (= chip) order; helper `rank` -> this chip's per-core quotas (its first quota
    keys == T_g in list order). -> dict(T_g, k_rem, done, helper_quota, core_quota {core: n}, messages)."""
    bins, rounds = 1 << digit_bits, 32 // digit_bits
    keys = [[kk for kk, _ in lst] for lst in lists]
    src = [list(v) for v in keys]
    prefix, mask, k_rem, msgs = 0, 0, k, 1  # (the start message)
    for r in range(rounds):
        shift = 32 - digit_bits * (r + 1)
        hist = [[0] * bins for _ in lists]
        for q, v in enumerate(src):
            for kk in v:
                if (kk & mask) == prefix:
                    hist[q][(kk >> shift) & (bins - 1)] += 1
        h = [sum(hq[b] for hq in hist) for b in range(bins)]
        msgs += 1
        if r == 0 and sum(h) <= k:
            return dict(T_g=0, k_rem=0, done=True, helper_quota=[0] * len(lists), core_quota={}, messages=msgs)
        d, ab = bins - 1, 0
        while d > 0 and ab + h[d] < k_rem:
            ab += h[d]
            d -= 1
        k_rem -= ab
        prefix |= d << shift
        mask |= (bins - 1) << shift
        src = [[kk for kk in v if (kk & mask) == prefix] for v in src]
    rem, hq = k_rem, []
    for v in src:  # the tie round: counts of keys == T_g per helper -> quotas in helper order
        n = sum(1 for kk in v if kk == prefix)
        hq.append(min(n, rem))
        rem -= hq[-1]
    msgs += 1
    cq, q = {}, hq[rank]
    for kk, core in lists[rank]:
        if q == 0:
            break
        if kk == prefix:
            cq[core] = cq.get(core, 0) + 1
            q -= 1
    return dict(T_g=prefix, k_rem=k_rem, done=False, helper_quota=hq, core_quota=cq, messages=msgs)


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
