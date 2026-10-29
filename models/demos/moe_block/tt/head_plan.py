# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""The fused head's layout planner: every candidate layout of the shape (engine GX x GY cores, MT row tiles a core row,
NCH row chunks, K step KB, the router's logits on the engine or on router cores), the ones whose L1 fits (the engine's
own arena sizing, se_engine3.SeEngine3 with no device; the router cores' arena), ranked by a cost model calibrated on
measured heads (MOE_BLOCK_WORKLOG.md, "the head planner"). Host only: tests/test_head_plan.py sweeps it over real and
synthetic shapes.

fit() is the head's whole host plan: every requirement its build checks (the engine's arena at each K step / W ring /
row chunks, the RS plan, the mover column's cells, the router cores' arena and runtime args, the routing exchange's
planner regions and port buffers, packet headers) on a HeadShape, no device. A shape fit() accepts builds (the build
plans again with the device's own free L1, at least what fit() assumed); a shape it refuses says which requirement
failed, with the numbers and what would fit (tests/test_fused_head_coverage.py sweeps it)."""

from dataclasses import dataclass, field, replace
from functools import lru_cache
from types import SimpleNamespace

from models.demos.moe_block.tt.se_engine3 import ARENA_MAX, arena_cap_of, dest_tiles

# the machine, host-side defaults (a device build reads its own): the compute grid, the fabric payload, the L1 a core
# has free when the head builds (bank 1461248 B less the allocator's base and what the block keeps before the head:
# measured 1383 KB on the LoudBox 2x4 / ring of 4 - arena caps 1334 KB at K3's payload, 1319 KB at GLM-Flash's)
GRID = (12, 10)
L1_FREE = 1383 * 1024
# (what fits() assumes below the device's measured free L1: the build plans again with the device's own figure - every
# layout fit() accepted then still fits)
L1_MARGIN = 8 * 1024
PORT_GROUP = 4  # (x_gather3: a port sender's packet group - group + 1 packet headers)
HEADERS = 12  # (fabric packet headers a RISC)
ROUTER_ARGS = 341  # (router_stream's per-core runtime-arg bound)
ROUTING_L = 2  # (the routing exchange's port tables: the combine's links)


@dataclass(frozen=True)
class HeadShape:
    """What the head computes, a chip: rows (a mesh row's tokens), H, I (the shared expert's intermediate a TP chip), G
    (TP), latent (K3's latent width, or None), rs_dim (-1 the width to the owners - with a latent; -2 the rows), the
    router (E candidates, top K, fold), links, ring (the TP axis wraps), grid (the compute grid)."""

    rows: int
    H: int
    I: int
    G: int
    E: int
    K: int
    latent: int | None = None
    fold: int = 1
    links: int = 2
    ring: bool = False
    grid: tuple = (12, 10)
    R: int = 1  # (mesh rows: N = R G chips - the routing exchange's)
    sp_ring: bool = False  # (the SP axis wraps)
    payload: int = 8256  # (the fabric's packet payload bytes: the gather's pages a packet)
    A: int = 16  # (the combine's accumulators: the routing planner's instances at most)
    w_dtype: str = "bf8"  # (the shared expert's weights: bf4 / bf8 / bf16)

    @property
    def rs_dim(self):
        return -1 if self.latent else -2

    @property
    def N(self):
        return self.R * self.G

    @property
    def S(self):  # (tokens a chip owns)
        return self.rows // self.G

    @property
    def EPC(self):
        return self.E // self.fold // self.N


@dataclass(frozen=True, order=True)
class HeadLayout:
    GY: int
    MT: int
    NCH: int
    KB: int
    on_engine: bool
    GX: int = 11
    compact: bool = False  # (the RS arrival zones compact - se_rs.RsPlan: only where P-tile slots do not fit)
    split: bool = False  # (the router's fin + top-K over two cores a row tile: router_stream's split)


@dataclass
class Cost:
    """The cost model's rates (fit_cost in tests/test_head_plan.py on CAL + CAL_COMPACT - 62 traced heads of K3 /
    K2.7 / GLM-Flash / GLM-5.3 at 128 .. 1280 rows on the 2x4 line and a ring of 4, P-tile and compact RS zones, K steps
    1 .. 16: median error 6%, worst 23%). us."""

    clk: float = 1.35e3  # (cycles a us: AICLK 1350 MHz)
    tile: float = 20  # (cycles a 32 x 32 x 32 matmul tile, HiFi2 bf8 W - effective, unpack / pack overlapped)
    round_us: float = 0.9774  # (a W round's fixed cost: its multicast, the semaphores, the pass setup)
    link_gbs: float = 20.89  # (a fabric link direction, payload GB/s, sustained)
    dram_gbs: float = 244.5  # (the W read a chip, GB/s)
    rtr_tile: float = 20  # (cycles a router logits tile, HiFi3 bf16)
    chain_us: float = 83.11  # (the routing chain's fixed part: the exchange, the planner)
    chunk_us: float = 48.04  # (a row chunk's fixed cost: its zones' credits, the RS's fill and drain)
    base_us: float = 26.21
    eng_overhead: float = 3.0  # (the logits phase on the engine vs its tiles)
    overlap: float = 0.0  # (the share of a chunk's gate / up under its own x gather: fit 0)
    noc_gbs: float = 11.69  # (a column's W stream down its cores, GB/s: every chunk streams all its W blocks again)
    chain_ek: float = 0.1852  # (the chain's top-K / joint rows / exchange: us an (expert tile x K))
    xmc_gbs: float = 43.98  # (an x family owner's row multicast, GB/s: fewer owners - a larger K step - more each)


COST = Cost()
TIE = 0.03


@dataclass
class Planned:
    lay: HeadLayout
    arena: int  # (bytes a core)
    sizes: dict
    t: float  # (predicted us)
    parts: dict = field(default_factory=dict)
    note: str = ""


def engine_width(shape: HeadShape, split=False):
    """The engine's columns: every one left of the mover column (split: but the one beside it - the router's second
    cores), at most a gate / up pair (and a latent tile) each."""
    GX = min(shape.grid[0] - 1 - (1 if split else 0), shape.I // 32)
    if shape.latent:
        GX = min(GX, shape.latent // 32 // shape.G)
    return GX


def router_cells(shape: HeadShape, GY, GX):
    """Where the router cores (a row tile each of the owners' rows) go: "column" (the mover column's cells under the
    ports and the watcher), "row" (a free core row under the engine), "rows" (the free rows under it - with the logits
    on the engine only: on the router cores their W_r goes out by one row's multicast), or None."""
    nrt, gy = shape.rows // 32 // shape.G, shape.grid[1]
    if nrt + 2 * shape.links + 1 <= gy:
        return "column"
    if GY < gy and nrt <= GX:
        return "row"
    if GY < gy and nrt <= GX * (gy - GY):
        return "rows"
    return None


def head_layouts(shape: HeadShape, *, mt_max=2):
    """(GY, MT, NCH): NCH row chunks of CR = GY x MT row tiles, chunk m = block m of every owner (whole owner blocks),
    the router cores' cells (a row tile each of the owners' rows) in the mover column or a free row under the engine."""
    rows, G, (gx, gy) = shape.rows, shape.G, shape.grid
    GX = engine_width(shape)
    RT = rows // 32
    if rows % 32 or RT % G:
        return []
    nrt = RT // G
    out = []
    for nch in range(1, nrt + 1):
        if nrt % nch:
            continue
        CR = RT // nch
        for GY in range(gy, 0, -1):
            if CR % GY or CR // GY > mt_max:
                continue
            if router_cells(shape, GY, GX) is None:
                continue
            out.append((GY, CR // GY, nch))
    return out


def k_steps(shape: HeadShape):
    klt = shape.H // 32 // shape.G
    return sorted((d for d in range(1, 17) if klt % d == 0 and (klt // d) % 2 == 0), reverse=True)


def _wdt(name):
    import ttnn

    return {"bf4": ttnn.bfloat4_b, "bf8": ttnn.bfloat8_b, "bf16": ttnn.bfloat16}[name]


def _al(v):
    return (v + 63) // 64 * 64


def size(shape: HeadShape, lay: HeadLayout, arena_max=ARENA_MAX, compact=None):
    """The head sized at this layout (host): (SeEngine3 or None, arena bytes - the engine's, grown to the router
    cores' least, as the build does; with why not: the bytes it would need, else 0 - region sizes, why not). Every
    check of the build past the engine too (cores_check)."""
    cap = arena_max(lay.KB) if callable(arena_max) else arena_max
    key = (shape, lay, cap, compact)
    if key not in _SIZED:
        if len(_SIZED) > 20000:
            _SIZED.clear()
        _SIZED[key] = _size(shape, lay, cap, compact)
    return _SIZED[key]


_SIZED = {}  # (size's answers: a pure function of its arguments)


def _size(shape, lay, cap, compact):
    from models.demos.moe_block.tt.router_stream import RouterStream
    from models.demos.moe_block.tt.se_engine3 import ArenaTooLarge, SeEngine3
    from models.demos.moe_block.tt.se_rs import RsPlan

    try:
        plan = RsPlan(
            G=shape.G,
            ring=shape.ring and shape.G > 2,
            GX=lay.GX,
            GY=lay.GY,
            MT=lay.MT,
            NTH=shape.H // 32,
            scatter=shape.rs_dim,
            nv=shape.links,
            pwmax=dest_tiles() // lay.MT,
            compact=lay.compact if compact is None else compact,
        )
        eng = SeEngine3(
            None,
            rows=shape.rows,
            H=shape.H,
            I=shape.I,
            glu="silu",
            GX=lay.GX,
            GY=lay.GY,
            MT=lay.MT,
            KB=lay.KB,
            shards=shape.G,
            rs=plan,
            latent=shape.latent // 32 // shape.G if shape.latent else None,
            prep=True,
            router=dict(ET=shape.E // 32, KTO=shape.H // shape.G // 32) if lay.on_engine else None,
            arena_max=cap,
            w_dtype=_wdt(shape.w_dtype),
        )
    except ArenaTooLarge as err:
        return None, err.need, {}, f"arena {err.need / 1024:.0f} KB > {cap / 1024:.0f} KB (W ring at the least)"
    except (ValueError, AssertionError, KeyError) as err:
        return None, 0, {}, " ".join(str(err).split())[:240] or type(err).__name__
    need = _al(RouterStream.min_bytes(shape.G, shape.E, shape.K, split=lay.split))
    if need > cap:
        return None, need, eng.sizes, f"router cores' arena {need / 1024:.0f} KB > {cap / 1024:.0f} KB"
    arena = max(eng.arena_bytes, need)
    why = cores_check(shape, lay, eng, plan, arena)
    if why:
        return None, arena, eng.sizes, why
    return eng, arena, eng.sizes, ""


def host_chips(R, C, ring, links):
    """The x gather's ring plan of every chip (fabric_all_gather_py._plan's, no device: positions along the row, next /
    prev, the schedules, which directions carry links, a port per (direction, link))."""
    from ttnn.bringup.combine_rf.fabric_all_gather_py import _needs_ready, _schedule

    chips = {}
    for r in range(R):
        for p in range(C):
            nxt = (r, (p + 1) % C) if (ring or p < C - 1) else None
            prv = (r, (p - 1) % C) if (ring or p > 0) else None
            fwd, bwd = _schedule(p, C, ring, ring)
            rg = dict(p=p, next=nxt, prev=prv, fwd=[(q, part) for q, part in fwd], bwd=[(q, part) for q, part in bwd])
            chips[(r, p)] = dict(out=p, G=C, rings=[rg])
    for coord, ch in chips.items():
        rg = ch["rings"][0]
        for d, peer in (("fwd", rg["next"]), ("bwd", rg["prev"])):
            if peer is not None and (rg[d] or _needs_ready(chips, coord, 0, d)):
                rg[f"{d}_links"] = list(range(links))
        ch["ports"] = {(0, d, l): None for d in ("fwd", "bwd") for l in range(links)}
    return chips


def routing_shell(R, C, ring, wrap_y, links):
    """A RoutingStream with its host-side state only (the paths, the senders, the forwarding slots): its pure methods
    on an R x C mesh (ring: the TP axis wraps, wrap_y: the SP axis; links: the gather's)."""
    from models.demos.moe_block.tt.routing_stream import RoutingStream

    rs = object.__new__(RoutingStream)
    rs.R, rs.C, rs.N, rs.L = R, C, R * C, ROUTING_L
    rs.xg = SimpleNamespace(chips=host_chips(R, C, ring, links), ring=ring)
    rs.wrap_x, rs.wrap_y = ring, wrap_y
    rs.mesh = None
    return rs


@lru_cache(maxsize=256)
def routing_paths(R, C, ring, wrap_y, links):
    """(the most packets an X sender sends, a Y sender, a port forwards; the most senders a chip) of the routing
    exchange on this mesh - or why its paths do not build."""
    rsh = routing_shell(R, C, ring, wrap_y, links)
    try:
        serves = {c: rsh.serve(c) for c in rsh.xg.chips}
        fw = rsh.fwd_plan()[0]
    except AssertionError as err:
        return str(err)
    nx = max(len(v) for sv in serves.values() for k_, v in sv.items() if k_[0] == "x")
    ny = max([len(v) for sv in serves.values() for k_, v in sv.items() if k_[0] == "y"] + [0])
    nf = max([len(v) for v in fw.values()] + [0])
    return nx, ny, nf, max(len(sv) for sv in serves.values())


def la_dests(G, ring):
    """The most latent destination blocks a port sends (one direction): a line's end chip all G - 1, a ring's half."""
    return G // 2 if ring and G > 2 else G - 1


def cores_check(shape: HeadShape, lay: HeadLayout, eng, rs_plan, arena):
    """Every per-core requirement past the engine's arena (HeadSE3 / RouterStream / RoutingStream / XGather3's checks):
    "" or why not, with the numbers."""
    from ttnn.bringup.combine_rf.routing_a2a import routing_a2a_sizes, slot_bytes, worst_rows

    from models.demos.moe_block.tt.router_stream import RouterStream

    G, L, (gx, gy) = shape.G, shape.links, shape.grid
    ring = shape.ring and G > 2
    # (the SE reduce-scatter's per-core counts word: a byte a block)
    if G > 4 or rs_plan.P >= 256:
        return f"reduce-scatter counts word: TP {G} <= 4 blocks of < 256 tiles (P {rs_plan.P})"
    if PORT_GROUP + 1 > HEADERS:
        return "packet headers"
    if shape.latent:
        nd = la_dests(G, ring)
        if nd > 3 or PORT_GROUP + 1 + nd > HEADERS:
            return f"latent all-to-all: {nd} destination blocks a port > 3 (packet headers: {HEADERS} a RISC)"
    # the mover column: ports (2 a link, from the top), the watcher, the router cores (a row tile each of the owners'
    # rows; with the split two) - or the router cores in free core rows under the engine
    nrt = shape.rows // 32 // G
    S2 = 2 if lay.split else 1
    if lay.GX * lay.GY // L > 128:
        return f"a port's reduce-scatter cores: {lay.GX * lay.GY // L} > 128 (x_gather3's rdc / seen tables)"
    if 3 * L > gy:
        return f"mover column: {2 * L} ports + {L} copy cores > {gy} cells"
    if nrt + 2 * L + 1 <= gy:
        if lay.split and lay.GX > gx - 2:
            return f"router split: column {gx - 2} must be free (engine {lay.GX} columns)"
    else:
        if lay.split:
            return f"router split: {nrt} router cores + {2 * L} ports + the watcher > the mover column's {gy} cells"
        free = (gy - lay.GY) * lay.GX
        if lay.GY >= gy or not (nrt <= lay.GX or (lay.on_engine and nrt <= free)):
            return (
                f"router cores: {nrt} (a row tile each of a chip's owned rows) fit neither the mover column ({gy} cells "
                f"less {2 * L} ports and the watcher) nor the {free} free cells under the engine "
                f"({lay.GX}x{lay.GY}{', one row: the logits on the router cores' if not lay.on_engine else ''})"
            )
    nc = S2 * nrt
    # the router cores' arena: the K block (<= 4 tiles dividing the chip's K) whose buffers fit, the ports' logits
    # sub-blocks
    ET = shape.E // 32
    ETL = ET // S2
    KT = shape.H // G // 32
    layout = lambda kbr: RouterStream.layout_of(G, ETL, shape.K, kbr, ETF=ET)  # noqa: E731
    fit_ = [d for d in range(1, 5) if KT % d == 0 and layout(d)[1] <= arena]
    roff, rbytes = layout(max(fit_) if fit_ else 1)
    if rbytes > arena:
        return f"router cores: {rbytes / 1024:.0f} KB > the arena's {arena / 1024:.0f} KB"
    ns = -(-nc // L)
    while not lay.on_engine and ns > 1 and (G - 1) * ns * ETL * 4096 > arena // 2:
        ns -= 1
    pbuf = (G - 1) * ns * ETL * 4096
    if pbuf > arena:
        return f"the ports' logits sub-blocks: {pbuf / 1024:.0f} KB > the arena's {arena / 1024:.0f} KB"
    # the routing exchange + planner (RoutingStream): instances whose regions fit the router cores, the ports' and the
    # watcher's packet buffers
    N, S, K, EPC, A = shape.N, shape.S, shape.K, shape.EPC, shape.A
    if N * EPC > 1024 or S > 2048 or EPC > 64 or S % 32:
        return f"routing planner: N EPC {N * EPC} <= 1024, S {S} <= 2048 tokens a chip, EPC {EPC} <= 64"
    rows_cap = worst_rows(N, S, K, EPC)
    for M in range(min(2 * nc, EPC, A, 14), 0, -1):
        sz = routing_a2a_sizes(N=N, S=S, K=K, EPC=EPC, A=A, rows_cap=rows_cap, M=M)
        step, per = _al(sz["plan_l1"]), -(-M // nc)
        pl0 = 0 if per * step <= roff["bias"] else _al(rbytes)
        if sz["fits"] and pl0 + per * step <= arena:
            break
    else:
        return f"routing planner: {sz['why'] or f'{per} region(s) of {step / 1024:.0f} KB past {pl0 / 1024:.0f} KB'} > the arena's {arena / 1024:.0f} KB"
    rt = routing_paths(shape.R, G, ring, bool(shape.sp_ring and shape.R > 2), L)
    if isinstance(rt, str):
        return f"routing exchange: {rt}"
    nx, ny, nf, nsend = rt
    if nf > 16:
        return f"routing exchange: a port forwards {nf} packets > 16"
    pcap = slot_bytes(S, K, EPC, M)
    if _al(pbuf) + (nx + nf) * pcap > arena:
        return (
            f"routing exchange: a port's buffers {(_al(pbuf) + (nx + nf) * pcap) / 1024:.0f} KB (logits "
            f"{pbuf / 1024:.0f} + {nx} + {nf} packets of {pcap / 1024:.1f} KB) > the arena's {arena / 1024:.0f} KB"
        )
    if 2 * ny * pcap > arena:
        return f"routing exchange: the watcher's Y buffers {2 * ny * pcap / 1024:.0f} KB > the arena's {arena / 1024:.0f} KB"
    # the router cores' runtime args (router_stream's bound): the first core's, the longest
    plen = 26 + 5 * M + 3 * N + 3 * nsend
    hand = 0 if lay.on_engine else G - 1
    nargs = dict(
        BRISC=1 + 6 * hand + 10 + 3 + 2 * nc + plen,
        NCRISC=2 * G + 10 + (2 + plen if nc < M else 2),
        compute=hand + G + 9,
    )
    big = {k: v for k, v in nargs.items() if v > ROUTER_ARGS}
    if big:
        return f"router cores' runtime args: {big} > {ROUTER_ARGS} ({nc} router cores, {M} planner instances, N {N})"
    return ""


def cost(shape: HeadShape, lay: HeadLayout, eng, c: Cost = None):
    """Predicted us and its parts. The SE path is a pipeline over the NCH row chunks: chunk m's x gather (link), its
    gate / up and down on the engine (the busiest column's tiles, its W rounds), its reduce-scatter (link), the W
    stream (DRAM; re-read every chunk with a W ring). One chunk: the gather, the compute (gate / up partly under the
    gather: its own shard first), the RS after the down - in series. The head ends at max(the SE path, the routing
    chain: the logits on the engine - delaying the SE path - or on the router cores, then top-K / exchange / planner).
    """
    c = c or COST
    G, MT, n = shape.G, lay.MT, lay.NCH
    KT, HT = shape.H // 32, shape.I // 32
    gu = max(2 * p for p in eng.po) + (eng.LWMAX if shape.latent else 0)  # (the busiest column's tiles a K tile)
    dn = max(sum(w) for w in eng.wd)
    t_gu = n * MT * KT * gu * c.tile / c.clk + n * eng.NPG * eng.NSG * c.round_us
    t_dn = n * MT * HT * dn * c.tile / c.clk + n * eng.NPD * eng.NSD * c.round_us
    # (links: a direction's busiest link - the line's end links carry G - 1 shards / blocks, a ring's (G - 1) / 2)
    hops = (G - 1) / 2 if shape.ring and G > 2 else G - 1
    rate = shape.links * c.link_gbs * 1e3
    t_x = hops * shape.rows * shape.H // G * 2 / rate
    t_rs = t_x
    t_lat = hops * shape.rows * shape.latent // G // G * 2 / rate if shape.latent else 0.0
    t_w = eng.dram_bytes()["w"] * (n if eng.WRING else 1) / (c.dram_gbs * 1e3)
    t_wn = n * eng.dram_bytes()["w"] / lay.GX / (c.noc_gbs * 1e3)  # (the busiest column's W stream, every chunk)
    t_math = t_gu + t_dn
    # (x: every K tile multicast along its core row once a pass by its family's owner - S0 / S1 columns of a family)
    t_xm = n * eng.NPG * (KT / 2) * MT * 2048 / min(eng.S0, eng.S1) / (c.xmc_gbs * 1e3)
    comp = max(t_math, t_wn, t_xm) / n + c.chunk_us * (n > 1)
    hidden = c.overlap * min(t_gu, t_x, comp * n) / n  # (gate / up under the chunk's own gather)
    steady = max(comp, (t_x + t_lat + t_rs) / n, t_w / n)
    t_se = (t_x + t_lat) / n + (n - 1) * steady + comp - hidden + t_rs / n
    if n == 1:
        t_se = max(t_se, t_w + t_dn + t_rs)
    # router: on the engine its logits phase (every chunk's rows, my expert tiles) delays the SE path and starts the
    # chain; on the router cores (a row tile each of the owners' rows: all E tiles x all of H) their matmul
    ET = shape.E // 32
    t_rc = ET * KT * c.rtr_tile / c.clk
    chain = c.chain_us + c.chain_ek * ET * shape.K
    etc = -(-ET // lay.GX)
    t_eng = c.eng_overhead * n * MT * etc * (KT // G) * c.rtr_tile / c.clk
    if lay.on_engine:
        t_se += t_eng
        t_chain = t_eng + chain
    else:
        t_chain = t_rc + chain
    t = c.base_us + max(t_se, t_chain)
    return t, dict(
        gu=round(t_gu),
        down=round(t_dn),
        x=round(t_x),
        rs=round(t_rs),
        w=round(t_w),
        wnoc=round(t_wn),
        xmc=round(t_xm),
        owners=min(eng.S0, eng.S1),
        se=round(t_se),
        chain=round(t_chain),
        wring=eng.WRING,
    )


def plan(shape: HeadShape, arena_max=ARENA_MAX, c: Cost = None, keep_unfit=False, placement="measured", split=False):
    """Every candidate layout that fits (arena_max: bytes, or the device's cap a K step - se_engine3.arena_cap), ranked
    by predicted time. The router's placement a layout: "measured" -
    fused_head.router_placement's (its rates from measured A/Bs: the cost model's engine / router-core gap is inside
    its error), the other one where that does not fit; "model": both, ranked by the cost model."""
    from models.demos.moe_block.tt.fused_head import router_placement

    out = []
    for GY, MT, NCH in head_layouts(shape):
        for kb in k_steps(shape):
            pref = None
            if placement == "measured":
                pref, _ = router_placement(
                    E=shape.E, H=shape.H, I=shape.I, G=shape.G, rows=shape.rows, latent=shape.latent,
                    GX=engine_width(shape), GY=GY, MT=MT, NCH=NCH,
                )  # fmt: skip
            got = []
            cells = router_cells(shape, GY, engine_width(shape))
            # (split: the router's fin + top-K over two cores a row tile - the logits on the engine, the router cores
            # in the mover column, an even expert tile count)
            sp = split and cells == "column" and (shape.E // 32) % 2 == 0
            if split and not sp:
                continue
            for on in (True, False) if pref in (None, True) else (False, True):
                if cells == "rows" and not on:  # (the router cores' W_r multicast: one row)
                    continue
                if sp and not on:
                    continue
                # (P-tile RS zones where they fit - the layouts measured before compact zones, byte for byte - else
                # compact: GLM-5.3 at 512 rows ran 331 us on slots, 349 compact)
                for compact in (False, True) if shape.rs_dim == -2 and shape.G <= 4 else (False,):
                    lay = HeadLayout(
                        GY=GY,
                        MT=MT,
                        NCH=NCH,
                        KB=kb,
                        on_engine=on,
                        GX=engine_width(shape, sp),
                        compact=compact,
                        split=sp,
                    )
                    eng, arena, sizes, why = size(shape, lay, arena_max)
                    if eng is not None:
                        break
                if eng is None:
                    if keep_unfit:
                        out.append(Planned(lay, arena, sizes, float("inf"), note=why))
                    continue
                t, parts = cost(shape, lay, eng, c)
                got.append(Planned(lay, arena, sizes, t, parts))
                if pref is not None:
                    break
            out += got
    out.sort(key=lambda p: (p.t, -p.lay.KB))
    # (inside TIE of the best the model does not separate layouts: there the fewest row chunks - every chunk streams
    # all of W again (K2.7 1024 rows: 2 chunks 680 us, 4 750; 768: 2 530, 3 554; GLM-5.3 1024: 2 607, 4 638) - then
    # the layout the fit-first rule took, every measured head's (GLM-5.3 at 512 rows: K step 6 on slots 331 us, 12
    # compact 349, 8 compact 357), then the prediction)
    if out and not split:
        old = legacy(shape, arena_max, c)
        pool = out + ([old] if old is not None else [])
        best = min(p.t for p in pool)
        near = [p for p in pool if p.t <= best * (1 + TIE)]
        pick = min(near, key=lambda p: (p.lay.NCH, p is not old, p.t))
        out = [pick] + [p for p in out if p.lay != pick.lay]
    return out


def legacy(shape: HeadShape, arena_max=ARENA_MAX, c: Cost = None):
    """The fit-first rule (before the planner): head_layouts' order (the fewest chunks, then the most core rows), the
    largest K step that fits, P-tile RS zones; the logits on the engine only at the first layout (where
    router_placement says so), else the router cores at the first layout that fits."""
    from models.demos.moe_block.tt.fused_head import router_placement

    lays = [
        (GY, MT, NCH) for GY, MT, NCH in head_layouts(shape) if router_cells(shape, GY, engine_width(shape)) != "rows"
    ]
    if not lays:
        return None
    GX = engine_width(shape)
    GY, MT, NCH = lays[0]
    on, _ = router_placement(
        E=shape.E, H=shape.H, I=shape.I, G=shape.G, rows=shape.rows, latent=shape.latent, GX=GX, GY=GY, MT=MT, NCH=NCH
    )
    tries = [(lays[0], True)] if on else []
    tries += [(l_, False) for l_ in lays]
    for (GY, MT, NCH), on in tries:
        for kb in k_steps(shape):
            lay = HeadLayout(GY=GY, MT=MT, NCH=NCH, KB=kb, on_engine=on, GX=GX)
            eng, arena, sizes, _ = size(shape, lay, arena_max)
            if eng is not None:
                t, parts = cost(shape, lay, eng, c)
                return Planned(lay, arena, sizes, t, parts, note="fit-first")
    return None


def describe(p: Planned):
    lay = p.lay
    return (
        f"engine {lay.GX}x{lay.GY} MT {lay.MT} NCH {lay.NCH} KB {lay.KB} router "
        f"{'engine' if lay.on_engine else 'cores'}: predicted {p.t:.0f} us {p.parts}, arena {p.arena // 1024} KB"
    )


@dataclass
class HeadFit:
    """fit()'s answer: ok, the planned layouts (every requirement met, best first), why not (with the numbers and what
    would fit), the unconstrained best (the cost model's pick with unlimited L1) and what L1 forced (note)."""

    ok: bool
    planned: list = field(default_factory=list)
    why: str = ""
    free: Planned | None = None
    note: str = ""

    def __bool__(self):
        return self.ok


def shape_checks(shape: HeadShape):
    """The shape-level requirements (no layout involved): why not, or ""."""
    G, H = shape.G, shape.H
    if G < 2:
        return f"TP {G}: the fused head gathers x over the TP axis (TP >= 2)"
    if G > 4:
        return (
            f"TP {G} > 4: the SE reduce-scatter's per-block tile counts are a byte each in one word (se_rs counts_word) "
            f"- a mesh with TP 2 or 4"
        )
    if shape.links not in (1, 2):
        return f"{shape.links} links: 1 or 2 (a port per direction and link)"
    if H % (32 * G):
        return f"H {H} is not whole tiles over TP {G} (a multiple of {32 * G})"
    klt = H // 32 // G
    if not k_steps(shape):
        return f"H / 32 / TP = {klt} tiles a shard: no K step <= 16 gives an even count a shard (an even shard)"
    if shape.I < 32 or shape.I % 32:
        return f"the shared expert's I / TP = {shape.I}: whole tiles, at least one (I_s a multiple of {32 * G})"
    if shape.latent and (shape.latent % (32 * G)):
        return f"latent {shape.latent} is not whole tiles over TP {G} (a multiple of {32 * G})"
    if shape.K % 2 or shape.K > 16:
        return (
            f"top-K {shape.K}: even and <= 16 (the joint routing rows - K ids + K weights in 64 B - router_topk, "
            f"dispatch_prep, routing_a2a; the ops head has the same bound)"
        )
    if shape.E % 32:
        return f"{shape.E} router candidates: whole tiles (a multiple of 32)"
    if shape.rows % (32 * G):
        nxt = -(-shape.rows // (32 * G)) * 32 * G
        return f"rows a mesh row {shape.rows} must be a multiple of 32 x TP = {32 * G}; nearest valid {nxt}"
    N, S, EPC = shape.N, shape.S, shape.EPC
    if N > 32 or shape.A > 16:
        return f"routing planner: N {N} <= 32 chips, accumulators {shape.A} <= 16"
    if S > 2048:
        return f"routing planner: {S} tokens a chip > 2048 (an entry's token field); chunk <= {2048 * N} tokens"
    if EPC > 64 or N * EPC > 1024:
        return f"routing planner: {EPC} experts a chip <= 64 and N x EPC {N * EPC} <= 1024 (fold the router)"
    if engine_width(shape) < 1:
        return f"no engine column: I / TP {shape.I} or latent tiles a chip"
    return ""


def _summary(unfit, shape, arena_max):
    """Why no candidate fits: the L1 failures by the least arena any layout needed vs its cap, the others by their
    reason at the layout with the fewest row chunks - both when both occur."""
    is_l1 = lambda p: p.note.startswith(("arena", "router cores' arena")) or "of L1" in p.note  # noqa: E731
    l1 = [p for p in unfit if is_l1(p)]
    out = []
    if l1:
        least = min(
            l1, key=lambda p: (p.arena or 1 << 40) - (arena_max(p.lay.KB) if callable(arena_max) else arena_max)
        )
        cap = arena_max(least.lay.KB) if callable(arena_max) else arena_max
        out.append(
            f"L1: the least arena any layout needs is {least.arena / 1024:.0f} KB ({describe_lay(least.lay)}), a core "
            f"has {cap / 1024:.0f} KB; tried K steps {sorted({p.lay.KB for p in l1})}, row chunks "
            f"{sorted({p.lay.NCH for p in l1})}, W rings off / 5 / 4 / 3"
        )
    other = [p for p in unfit if not is_l1(p)]
    if other:
        reasons = {}
        for p in sorted(other, key=lambda p: (p.lay.NCH, -p.lay.KB)):
            reasons.setdefault(p.note.split(":")[0], p)
        first = next(iter(reasons.values()))
        more = f" (and {len(reasons) - 1} other reason(s))" if len(reasons) > 1 else ""
        out.append(f"{first.note} at {describe_lay(first.lay)}{more}")
    return "; ".join(out)


def describe_lay(lay: HeadLayout):
    return (
        f"engine {lay.GX}x{lay.GY} MT {lay.MT}, {lay.NCH} row chunk{'s' if lay.NCH > 1 else ''}, K step {lay.KB}, router "
        f"{'on the engine' if lay.on_engine else 'cores'}{', split' if lay.split else ''}"
    )


def degradation(chosen: Planned, free: Planned):
    """What L1 forced: the planned layout vs the cost model's pick with unlimited L1, with the predicted cost."""
    if free is None or chosen.t <= free.t * 1.005:
        return ""
    a, b = chosen.lay, free.lay
    d = []
    if a.KB != b.KB:
        d.append(f"K step {b.KB} -> {a.KB}")
    if a.NCH != b.NCH:
        d.append(f"row chunks {b.NCH} -> {a.NCH}")
    if (a.GY, a.MT) != (b.GY, b.MT):
        d.append(f"engine rows x MT {b.GY}x{b.MT} -> {a.GY}x{a.MT}")
    if a.on_engine != b.on_engine:
        d.append(f"router logits {'engine' if b.on_engine else 'cores'} -> {'engine' if a.on_engine else 'cores'}")
    if a.compact != b.compact:
        d.append("compact RS zones")
    wr = chosen.parts.get("wring")
    if wr:
        d.append(f"W streamed from DRAM every chunk (a ring of {wr} slots)")
    return (
        f"L1-limited: {', '.join(d) or 'another layout'} - predicted {chosen.t:.0f} us vs {free.t:.0f} us with "
        f"unlimited L1 (+{chosen.t - free.t:.0f} us, +{100 * (chosen.t / free.t - 1):.0f}%)"
    )


UNLIMITED = 1 << 40


def fit(
    shape: HeadShape,
    arena_max=None,
    *,
    split=False,
    on_engine=None,
    KB=None,
    placement="measured",
    largest=True,
    explain=True,
):
    """The head's whole host plan (module docstring). arena_max: bytes, or a function of the K step (the build: the
    device's cap); None: fit()'s own from L1_FREE - L1_MARGIN and the shape's payload. on_engine / KB: given (A/B).
    largest: on an L1 refusal, also the largest rows a mesh row that fit (a few more plans); explain: the unconstrained
    pick and the L1 note (one more plan)."""
    if arena_max is None:
        arena_max = lambda kb: arena_cap_of(L1_FREE - L1_MARGIN, kb, payload=shape.payload)  # noqa: E731
    why = shape_checks(shape)
    if why:
        return HeadFit(False, why=why)
    if not head_layouts(shape):
        nrt = shape.rows // 32 // shape.G
        return HeadFit(
            False,
            why=f"no engine layout for {shape.rows} rows a mesh row ({nrt} row tiles an owner: row chunks of whole "
            f"owner blocks, MT <= 2, the router cores' cells)",
        )
    ps = plan(shape, arena_max, keep_unfit=True, placement=placement, split=split)
    want = lambda p: (on_engine is None or p.lay.on_engine == on_engine) and (not KB or p.lay.KB == KB)  # noqa: E731
    ok = [p for p in ps if p.t != float("inf") and want(p)]
    if not ok:
        unfit = [p for p in ps if p.t == float("inf") and want(p)] or [p for p in ps if p.t == float("inf")]
        if not unfit:
            return HeadFit(False, why=f"no layout with the router {on_engine=} and K step {KB}")
        why = _summary(unfit, shape, arena_max)
        if largest and "L1:" in why:
            r = shape.rows
            step = 32 * shape.G
            lo, hi = 0, r // step - 1
            while lo < hi:  # (the largest rows a mesh row that fit, a multiple of 32 x TP)
                mid = (lo + hi + 1) // 2
                f = fit(
                    replace(shape, rows=mid * step),
                    arena_max,
                    split=split,
                    on_engine=on_engine,
                    KB=KB,
                    largest=False,
                    explain=False,
                )
                lo, hi = (mid, hi) if f.ok else (lo, mid - 1)
            if lo:
                why += f"; the largest rows a mesh row that fit: {lo * step} (chunk {lo * step * shape.R})"
        return HeadFit(False, why=why)
    free = None
    if not explain:
        return HeadFit(True, planned=ok)
    try:
        fr = [p for p in plan(shape, UNLIMITED, placement=placement, split=split) if want(p)]
        free = fr[0] if fr else None
    except Exception:  # noqa: BLE001 - the unconstrained pick only explains the note
        free = None
    return HeadFit(True, planned=ok, free=free, note=degradation(ok[0], free))
