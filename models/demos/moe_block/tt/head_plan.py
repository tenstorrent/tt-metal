# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""The fused head's layout planner: every candidate layout of the shape (engine GX x GY cores, MT row tiles a core row,
NCH row chunks, K step KB, the router's logits on the engine or on router cores), the ones whose L1 fits (the engine's
own arena sizing, se_engine3.SeEngine3 with no device; the router cores' arena), ranked by a cost model calibrated on
measured heads (MOE_BLOCK_WORKLOG.md, "the head planner"). Host only: tests/test_head_plan.py sweeps it over real and
synthetic shapes."""

from dataclasses import dataclass, field

from models.demos.moe_block.tt.se_engine3 import ARENA_MAX, dest_tiles


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

    @property
    def rs_dim(self):
        return -1 if self.latent else -2


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


def size(shape: HeadShape, lay: HeadLayout, arena_max=ARENA_MAX, compact=None):
    """The engine sized at this layout (host): (SeEngine3 or None, arena bytes, region sizes, why not)."""
    from models.demos.moe_block.tt.router_stream import RouterStream
    from models.demos.moe_block.tt.se_engine3 import SeEngine3
    from models.demos.moe_block.tt.se_rs import RsPlan

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
    cap = arena_max(lay.KB) if callable(arena_max) else arena_max
    try:
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
        )
    except (ValueError, AssertionError) as err:
        return None, 0, {}, " ".join(str(err).split())[:200]
    need = (RouterStream.min_bytes(shape.G, shape.E, shape.K) + 63) // 64 * 64
    if need > cap:
        return None, eng.arena_bytes, eng.sizes, f"router cores' arena {need // 1024} KB"
    return eng, max(eng.arena_bytes, need if not lay.on_engine else 0), eng.sizes, ""


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
