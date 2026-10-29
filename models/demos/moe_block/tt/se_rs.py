# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""se_rs - the shared expert's down + reduce-scatter inside the fused head (pure planning; the device build in
se_engine3 / x_gather3 and the CPU protocol model share it).

matmul_reduce_scatter's schedule (plan_transport: per chip a block order by descending distance to the owner, start /
relay / fin roles, arrival thresholds) on the engine's own tiling: core (cx, cy) computes its down tiles block by block
(scatter -1, width: every column owns a slice of every block, so every core works on every entry and the entries go
in transport order; scatter -2, rows: the core row's MT row tiles fall into one or two blocks, all passes first). A
core's tiles of block b (the same on every chip) are its sub-block, dense [rows][cols] in slot e (b's entry index here)
of its partial buffer PART (P tiles a slot). start: handed to the port as packed; relay / fin: + the arrival(s) landed
in the core's zone (same core on the upstream chip, written by the upstream port with a fused increment of the core's
arrival counter), packed back in place (relay: handed to the port) or to the output (fin). Ports: the gather's movers,
(direction d, link l) forwarding the units (entry, cores of link l) whose role here is start(d) / relay(d)."""

from ttnn.operations.matmul_reduce_scatter.matmul_reduce_scatter_program_descriptor import plan_transport

DIRS = ("fwd", "bwd")


def rs_kind(G, ring):
    if not ring or G == 2:
        return "line"
    return "ring_balanced" if G % 2 == 0 else "ring_unbalanced"


def _split(n, parts):
    base, rem = divmod(n, parts)
    return [base + (1 if i < rem else 0) for i in range(parts)]


class RsPlan:
    """The down tiling and the transport schedule of the fused head's SE reduce-scatter (module docstring).
    G chips (positions 0 .. G - 1 along the TP line / ring), engine GX x GY cores of MT row tiles (one chunk of CR = GY MT
    row tiles), NTH output tile columns, scatter -1 (width) / -2 (rows), nv links, pwmax tiles a down pass row.
    compact (default: the rows scatter): a core's arrival zone holds only the tiles it gets - each slot at the sum of the
    core's earlier slots' tiles - not a P-tile slot per arrival of its (half, direction) (rows: a core's MT rows fall in
    one or two blocks, so a direction's arrivals are at most P tiles; G - 1 slots of P each took 3x the L1 - K2.7 at
    H 7168, 640 rows: 504 KB of zones forced 5 row chunks)."""

    def __init__(self, *, G, ring, GX, GY, MT, NTH, scatter, nv, pwmax, compact=None):
        self.G, self.GX, self.GY, self.MT, self.NTH, self.scatter, self.nv = G, GX, GY, MT, NTH, scatter, nv
        # (the ports' offsets come from a core's counts word: a byte a block, G <= 4)
        self.compact = (scatter == -2 and G <= 4) if compact is None else compact
        self.CR = GY * MT
        self.kind = rs_kind(G, ring)
        self.chips, self.slots = plan_transport(G, self.kind)
        n = GX * GY
        self.ncores = n
        if scatter == -1:
            assert NTH % G == 0
            self.NB = NTH // G
            self.w = _split(self.NB, GX)  # (column cx's tiles of every block)
            assert min(self.w) >= 1
            self.s = [sum(self.w[:c]) for c in range(GX)]
            self.npb = max(-(-w // pwmax) for w in self.w)  # (passes a block)
            self.P = MT * max(self.w)
        else:
            assert self.CR % G == 0
            self.RB = self.CR // G  # (row tiles a block)
            self.pn = _split(NTH, GX)
            self.n0 = [sum(self.pn[:c]) for c in range(GX)]
            self.npass = max(-(-p // pwmax) for p in self.pn)
            self.P = MT * max(self.pn)
        self.pwmax = pwmax
        # (the halves: a balanced ring sends each half's share of a block one way. By core index (the first half of the
        # cores) every core of the width scatter holds a slice of every block; in the rows scatter the first half of
        # the core rows holds whole blocks (0, 1 of 4), so every chip sent 1 block one way and 2 the other (GLM-5.3
        # ring of 4: 82 vs 165 packets a port, the RS 83 us vs 62.5). There the halves are column bands - every block
        # split by its output columns, the bands' tiles as even as the columns allow)
        self.hcol = None
        if scatter == -2 and self.kind == "ring_balanced" and GX > 1:
            pre = [sum(self.pn[:c]) for c in range(1, GX)]
            self.hcol = 1 + min(range(GX - 1), key=lambda i: abs(2 * pre[i] - NTH))

    # ---- geometry (chip-independent) ----
    def k(self, cx, cy):
        return cy * self.GX + cx

    def half(self, k):
        if getattr(self, "hcol", None):
            return 0 if k % self.GX < self.hcol else 1
        return 0 if k < -(-self.ncores // 2) else 1

    def link(self, k):
        return k % self.nv

    def rows_in(self, cy, b):
        """(first local row, rows) of core row cy's MT row tiles in block b (scatter -2)."""
        rs = [r for r in range(self.MT) if (cy * self.MT + r) // self.RB == b]
        return (rs[0], len(rs)) if rs else (0, 0)

    def count(self, cx, cy, b):
        """Tiles of core (cx, cy) in block b."""
        if self.scatter == -1:
            return self.MT * self.w[cx]
        return self.rows_in(cy, b)[1] * self.pn[cx]

    def counts_word(self, cx, cy):
        """Core (cx, cy)'s tile counts a block in one word for the ports: a byte a block (TP <= 4), or past TP 4 its (block,
        count) pairs - the rows scatter: a core's MT <= 2 rows fall in at most two blocks (x_gather3's cnt_of)."""
        assert self.P < 256, f"se_rs: a core's sub-block of {self.P} tiles > 255 (the counts word's bytes)"
        if self.G <= 4:
            return sum(self.count(cx, cy, b) << (8 * b) for b in range(self.G))
        nz = [(b, self.count(cx, cy, b)) for b in range(self.G) if self.count(cx, cy, b)]
        assert (
            self.scatter == -2 and len(nz) <= 2
        ), f"se_rs: TP {self.G} > 4 needs the rows scatter with a core's rows in at most two blocks (core {cx},{cy}: {nz})"
        nz += [(0xFF, 0)] * (2 - len(nz))
        return nz[0][0] | nz[0][1] << 8 | nz[1][0] << 16 | nz[1][1] << 24

    # ---- per chip ----
    def entries(self, p, h=0):
        return self.chips[p]["entries"][h]

    def role(self, p, b, h):
        """(direction | "fin", is_start)."""
        r = self.chips[p]["roles"][(b, h)]
        return r[0], r[1]

    def column_passes(self, p, cx):
        """Column cx's down passes on chip p, in order: [(first output tile column, width, entry or None)] (entry:
        scatter -1, the pass's block's entry index; scatter -2: None, the rows decide)."""
        out = []
        if self.scatter == -1:
            for e, b in enumerate(self.entries(p)):
                c0 = b * self.NB + self.s[cx]
                for pw in _split(self.w[cx], self.npb):
                    out.append((c0, pw, e))
                    c0 += pw
        else:
            c0 = self.n0[cx]
            for pw in _split(self.pn[cx], self.npass):
                out.append((c0, pw, None))
                c0 += pw
        return out

    def finb_off(self, p, cx, cy, e):
        """Tile offset of entry e's sub-block in the core's FINB (bf16, what is exchanged): compact, in entry order."""
        ents = self.entries(p, self.half(self.k(cx, cy)))
        return sum(self.count(cx, cy, b) for b in ents[:e])

    def finb_tiles(self, cx, cy):
        return sum(self.count(cx, cy, b) for b in range(self.G))

    def _needs_hold(self, p, cx, cy, e):
        h = self.half(self.k(cx, cy))
        d, start = self.role(p, self.entries(p, h)[e], h)
        return not (d != "fin" and start)

    def hold_offs(self, p, cx, cy):
        """{entry: HOLD tile offset} (fp32 own partials of the entries that add an arrival here), first fit over
        their live ranges (first pass .. its fin op); and the HOLD tiles used."""
        memo = self.__dict__.setdefault("_hold_memo", {})
        if (p, cx, cy) not in memo:
            memo[(p, cx, cy)] = self._hold_offs(p, cx, cy)
        return memo[(p, cx, cy)]

    def _hold_offs(self, p, cx, cy):
        ops = self.ops(p, cx, cy)
        h = self.half(self.k(cx, cy))
        ents = self.entries(p, h)
        passes = self.column_passes(p, cx)
        first = {}
        for oi, (kind, i) in enumerate(ops):
            if kind == "pass":
                c0, pw, e = passes[i]
                es = [e] if e is not None else [ents.index((cy * self.MT + r) // self.RB) for r in range(self.MT)]
                for e2 in es:
                    first.setdefault(e2, oi)
        end = {i: oi for oi, (kind, i) in enumerate(ops) if kind == "fin"}
        live, offs, top = [], {}, 0
        for oi in range(len(ops)):
            live = [(o, n, e) for o, n, e in live if end[e] >= oi]
            for e, f in first.items():
                if f == oi and self._needs_hold(p, cx, cy, e) and self.count(cx, cy, ents[e]) > 0:
                    n = self.count(cx, cy, ents[e])
                    o = 0
                    for lo, ln, _ in sorted(live):
                        if o + n <= lo:
                            break
                        o = max(o, lo + ln)
                    live.append((o, n, e))
                    offs[e] = o
                    top = max(top, o + n)
        return offs, top

    def pass_bases(self, p, cx, cy):
        """Per pass of core (cx, cy) on chip p: (width, [row r's first tile: HOLD offset | 1 << 31, or FINB offset])."""
        out = []
        ents = self.entries(p, self.half(self.k(cx, cy)))
        hold, _ = self.hold_offs(p, cx, cy)

        def at(e, within):
            if e in hold:
                return (hold[e] + within) | (1 << 31)
            return self.finb_off(p, cx, cy, e) + within

        for c0, pw, e in self.column_passes(p, cx):
            if self.scatter == -1:
                b = ents[e]
                off = c0 - b * self.NB - self.s[cx]
                out.append((pw, [at(e, r * self.w[cx] + off) for r in range(self.MT)]))
            else:
                bases = []
                for r in range(self.MT):
                    b = (cy * self.MT + r) // self.RB
                    r0, _ = self.rows_in(cy, b)
                    bases.append(at(ents.index(b), (r - r0) * self.pn[cx] + c0 - self.n0[cx]))
                out.append((pw, bases))
        return out

    def compute_args(self, p, cx, cy):
        """The RS compute tables: [nops] + ops (kind | idx << 8; kind 0 pass, 1 hand, 2 fin) + per pass [width, bases]
        + per entry [tiles, mode (1 fwd arrival, 2 bwd, 4 fin -> output), fwd / bwd zone offsets, HOLD off, FINB off].
        """
        kinds = {"pass": 0, "hand": 1, "fin": 2}
        ops = self.ops(p, cx, cy)
        out = [len(ops)] + [kinds[k] | (i << 8) for k, i in ops]
        for pw, bases in self.pass_bases(p, cx, cy):
            out += [pw] + bases
        h = self.half(self.k(cx, cy))
        ents = self.entries(p, h)
        hold, _ = self.hold_offs(p, cx, cy)
        arr = self.arrivals(p, cx, cy)
        zs = {dd: self.zone_slots(p, cx, cy, dd) for dd in DIRS}
        for e, b in enumerate(ents):
            d, _ = self.role(p, b, h)
            sl = {dd: [s_ for e_, s_, _ in arr[dd] if e_ == e] for dd in DIRS}
            mode = (1 if sl["fwd"] else 0) | (2 if sl["bwd"] else 0) | (4 if d == "fin" else 0)
            out += [
                self.count(cx, cy, b),
                mode,
                zs["fwd"][sl["fwd"][0]][0] if sl["fwd"] else 0,
                zs["bwd"][sl["bwd"][0]][0] if sl["bwd"] else 0,
                hold.get(e, 0),
                self.finb_off(p, cx, cy, e),
            ]
        return out

    def handoffs(self, p, cx, cy):
        """The BRISC's hand-offs in order: [(entry, direction)] (start sub-blocks, then relays after their add)."""
        h = self.half(self.k(cx, cy))
        ents = self.entries(p, h)
        out = []
        for kind, e in self.ops(p, cx, cy):
            if kind == "pass":
                continue
            d, start = self.role(p, ents[e], h)
            if kind == "hand" or (kind == "fin" and d != "fin"):
                out.append((e, d))
        return out

    def ops(self, p, cx, cy):
        """Core (cx, cy)'s compute program on chip p: [("pass", i) | ("hand", e) | ("fin", e)] - a pass's own partial
        packed; an entry handed to its port once its last pass is in (start) or after its add (relay; "fin" with the
        entry's role deciding); a finish runs behind the next entry's passes (matmul_reduce_scatter's lag)."""
        k = self.k(cx, cy)
        h = self.half(k)
        ents = self.entries(p, h)
        npass = len(self.column_passes(p, cx))
        last = {}  # entry -> its last pass
        for i, (c0, pw, e) in enumerate(self.column_passes(p, cx)):
            es = [e] if e is not None else [ents.index((cy * self.MT + r) // self.RB) for r in range(self.MT)]
            for e2 in es:
                last[e2] = i
        ops, pending = [], []
        for i in range(npass):
            ops.append(("pass", i))
            done = sorted(e for e, li in last.items() if li == i)
            if pending and done:
                ops += [("fin", e) for e in pending]
                pending = []
            for e in done:
                b = ents[e]
                if self.count(cx, cy, b) == 0:
                    continue
                d, start = self.role(p, b, h)
                if d != "fin" and start:
                    ops.append(("hand", e))
                else:
                    pending.append(e)
        ops += [("fin", e) for e in pending]
        return ops

    def arrivals(self, p, cx, cy):
        """Per direction: [(entry, zone slot, threshold)] of core (cx, cy) on chip p, in its entry order (the zone
        slot: the arrival's index in the chip's (half, direction) list; the threshold: the core's arrival counter once
        it landed - the upstream port's increments to this core, one per nonempty unit)."""
        k = self.k(cx, cy)
        h = self.half(k)
        sc = self.chips[p]
        ents = sc["entries"][h]
        out = {}
        for d in DIRS:
            lst = sc["arrivals"][(h, d)]
            if not lst:
                out[d] = []
                continue
            u = self._up(p, d)
            seq = [b for b in self.chips[u]["entries"][h] if self.chips[u]["roles"][(b, h)][0] == d]
            seq = [b for b in seq if self.count(cx, cy, b) > 0]
            out[d] = [
                (ents.index(b), slot, (seq.index(b) + 1) if self.count(cx, cy, b) > 0 else 0)
                for slot, b in enumerate(lst)
            ]
        return out

    def _up(self, p, d):
        q = p - 1 if d == "fwd" else p + 1
        if self.kind != "line":
            return q % self.G
        return q if 0 <= q < self.G else None

    def down(self, p, d):
        """The chip p's units flow to (direction d)."""
        q = p + 1 if d == "fwd" else p - 1
        if self.kind != "line":
            return q % self.G
        return q if 0 <= q < self.G else None

    def port_units(self, p, d):
        """Port (d, *) of chip p: [(entry, block, half mask, downstream slot half 0, slot half 1)] in entry order."""
        sc = self.chips[p]
        q = self.down(p, d)
        out = []
        ents = sc["entries"][0]
        assert ents == sc["entries"][1], "one entry order for both halves (no mirrored halves)"
        for e, b in enumerate(ents):
            mask = 0
            for h in (0, 1):
                dd, _ = self.role(p, b, h)
                if dd == d:
                    mask |= 1 << h
            if not mask:
                continue
            slots = [self.chips[q]["arrivals"][(h, d)].index(b) if (mask >> h) & 1 else 0 for h in (0, 1)]
            out.append((e, b, mask, slots[0], slots[1]))
        return out

    def finb_readers(self, p, cx, cy):
        """The ports (directions on core (cx, cy)'s link) that read its FINB: a unit of theirs takes tiles of it."""
        k = self.k(cx, cy)
        h = self.half(k)
        return [
            d
            for d in DIRS
            if self.down(p, d) is not None
            and any((m >> h) & 1 and self.count(cx, cy, b) > 0 for _, b, m, _, _ in self.port_units(p, d))
        ]

    def port_cores(self, l):
        """Cores (cx, cy) of link l, in k order."""
        return [(cx, cy) for cy in range(self.GY) for cx in range(self.GX) if self.link(self.k(cx, cy)) == l]

    def fin_map(self, p, cx, cy):
        """(y tile of the fin sub-block's tile 0, row stride, width): tile i -> y0 + (i // w) stride + i % w. y a chip:
        scatter -1 [CR, NB] tiles, -2 [RB, NTH]."""
        if self.scatter == -1:
            return cy * self.MT * self.NB + self.s[cx], self.NB, self.w[cx]
        r0, nr = self.rows_in(cy, p)
        if nr == 0:
            return 0, self.NTH, 1  # (no tile of the own block here)
        return (cy * self.MT + r0 - p * self.RB) * self.NTH + self.n0[cx], self.NTH, self.pn[cx]

    def zone_slots(self, p, cx, cy, d):
        """[(tile offset, tiles)] of core (cx, cy)'s direction-d zone slots on chip p, in slot order (the tiles its
        NCRISC credits for the slot; not compact: P-tile slots)."""
        lst = self.chips[p]["arrivals"].get((self.half(self.k(cx, cy)), d), [])
        if not self.compact:
            return [(s * self.P, self.P) for s in range(len(lst))]
        out, o = [], 0
        for b in lst:
            n = self.count(cx, cy, b)
            out.append((o, n))
            o += n
        return out

    def prefix_mask(self, q, h, d, slot):
        """The blocks of chip q's (half h, direction d) slots before `slot` (a bit a block): a port's compact offset of
        a core's slot is the sum of the core's tiles of these blocks (its counts word)."""
        m = 0
        for b in self.chips[q]["arrivals"].get((h, d), [])[:slot]:
            m |= 1 << b
        return m

    def zone_tiles(self, d):
        if not self.compact:
            return self.slots[d] * self.P
        cores = [(cx, cy) for cy in range(self.GY) for cx in range(self.GX)]
        return max(sum(n for _, n in self.zone_slots(p, cx, cy, d)) for p in range(self.G) for cx, cy in cores)


class LogitsPlan(RsPlan):
    """The router's logits reduce-scatter by tokens (fp32 on the wire: bf16 changes the picks): RB router cores, core
    j computing row tile j of every block (all E / 32 = ET column tiles, all of the chip's K slice), so at the fin it
    holds one row tile of the owner's tokens with every expert - its top-k is local. Same schedule, ports and protocol
    as RsPlan; a core's sub-block is its ET tiles of a block, all its matmuls run before the first hand-off."""

    def __init__(self, *, G, ring, RB, ET, nv):
        self.G, self.GX, self.GY, self.MT, self.nv, self.ET = G, RB, 1, 1, nv, ET
        self.kind = rs_kind(G, ring)
        self.chips, self.slots = plan_transport(G, self.kind)
        self.ncores, self.P, self.scatter = RB, ET, 0
        self.compact = False

    def count(self, cx, cy, b):
        return self.ET

    def counts_word(self, cx, cy):
        return 0  # (unused: every core has ET tiles of every block)

    def ops(self, p, cx, cy):
        h = self.half(self.k(cx, cy))
        out = [("pass", 0)]
        for e, b in enumerate(self.entries(p, h)):
            d, start = self.role(p, b, h)
            out.append(("hand", e) if d != "fin" and start else ("fin", e))
        return out
