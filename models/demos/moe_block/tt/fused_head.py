# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""The MoE block's fused head: one custom program per chip (MOE_BLOCK_WORKLOG.md, "the fused head"), HeadSE3: the x
all-gather over TP (tt/x_gather3.py's ports) straight into the shared expert engine's L1 (tt/se_engine3.py: gate / up
in shard arrival order, every weight resident), down + the SE reduce-scatter in the same program, the router (logits
on the engine, top-K, joint rows: tt/router_stream.py), the routing exchange + planner (tt/routing_stream.py),
dispatch prep (tt/prep_stream.py) and K3's latent all-to-all. Columns: the engine 0 .., the movers (ports, watcher,
router cores) in the last."""

from dataclasses import dataclass

import torch

import ttnn
from ttnn.bringup.combine_rf import persist_l1


@dataclass(frozen=True)
class RouterSpec:
    """The fused head's router chain: E experts, top K, the routed scale; logits_first: the ports send the logits
    ahead of the x gather, else after it; None: the head's choice (HeadSE3.logits_priority)."""

    E: int
    K: int
    route_scale: float = 1.0
    logits_first: bool | None = None
    on_engine: bool | None = None  # (the logits' matmul on the engine's cores at the start; None: the head's choice)
    # (on the engine: its W streams start once every router core has its partials - their DRAM reads would crowd the
    # partials' writes; None: the head's choice, HeadSE3.router_w_gate)
    w_gate: bool | None = None
    # (the fin sums the sources' partials in a fixed order - the predicted landing order: bit-identical runs; False in
    # landing order, A/B only - see RouterStream)
    ordered: bool = True
    fold: int = 1  # (E candidates, expert = candidate // fold: the joint rows carry experts - MoEModel's RouterConfig)
    # (a row tile's experts over two router cores - the fin and the top-K over half the tiles each, merged on the
    # first: router_stream's split; the logits on the engine, the second cores in the column left of the movers)
    split: bool = False
    score: str = "sigmoid"  # (the scores: "sigmoid", or "sqrtsoftplus" - DeepSeek-V4's sqrt(softplus), fp32)


@dataclass(frozen=True)
class RoutingSpec:
    """The fused head's routing exchange + combine planner: A combine accumulators, L links of the combine's ports
    (its port tables), rows_cap the plan's row capacity (default the worst case: N S min(K, EPC) + 32 EPC). prep: dispatch prep in the program
    (the owned tokens' route words, self.outputs["route"]); torus: (wrap X, wrap Y) of the dispatch's mesh."""

    A: int
    L: int = 2
    rows_cap: int = 0
    prep: bool = True
    torus: tuple = (False, False)


# the router placement's cost model (router_placement): per-core matmul rates and the routing chain past the logits
RTR_RATE = (
    1.3e12  # (FLOP/s a core, the router's HiFi3 bf16: K3 E 896 on the Galaxy, 316 us for 2 x 640 x 1792 x 896 / 5)
)
SE_RATE = 1.18e12  # (the SE path's, effective: K3 E 448 at 640 rows, Galaxy-projected 388 us for ~459 MFLOP a core)
ENG_OVERHEAD = 3.0  # (the logits phase on the engine vs its FLOPs: K3 E 448 ~35 us measured vs ~11)
CHAIN_US = 150.0  # (the routing chain past the logits: GLM's chain-bound head 166 us less its ~16 us logits phase)


def router_placement(*, E, H, I, G, rows, latent, GX, GY, MT, NCH):
    """(on the engine?, estimates us). The head ends at max(SE path, routing chain). On the router cores (a row
    tile each of every owner's rows: 2 x G 32 tokens x E x H / G FLOP a core, whatever the rows) the chain is their
    matmul + CHAIN_US; on the engine the logits phase (my MT x NCH row tiles x my expert tiles) delays the SE path and
    starts the chain. Pick the shorter max."""
    t_rc = 2 * 32 * E * H / RTR_RATE * 1e6
    etc = -(-(E // 32) // GX)
    t_eng = ENG_OVERHEAD * 2 * (MT * NCH * 32) * (etc * 32) * (H // G) / RTR_RATE * 1e6
    flops = 2 * rows * H * 3 * I + (2 * rows * H * latent // G if latent else 0)  # (gate / up / down, the latent)
    t_se = flops / (GX * GY) / SE_RATE * 1e6
    rc, eng = max(t_se, t_rc + CHAIN_US), max(t_se + t_eng, t_eng + CHAIN_US)
    return eng < rc, dict(
        router_cores=round(rc), engine=round(eng), t_rc=round(t_rc), t_eng=round(t_eng), se=round(t_se)
    )


def head_layouts(rows, G, *, grid_y=10, GX=11, num_links=2, router=True):
    """The fused head's engine layouts for `rows` (a mesh row's tokens) over TP = G, preferred first: (GY, MT, NCH) -
    NCH row chunks of CR = GY x MT row tiles, chunk m = block m of every owner (CR / G row tiles of each: whole), MT <=
    2 (MT 4 doubles a core's L1), the fewest chunks, then the most core rows. With the router chain its cores (a row
    tile each of the owners' rows: rows / 32 / G) must fit the mover column's free cells or a free row under the
    engine. The L1 fit is the engine's (SeEngine3 raises) - the caller tries these in order."""
    RT = rows // 32
    if rows % 32 or RT % G:
        return []
    nrt = RT // G
    out = []
    for nch in range(1, nrt + 1):
        if nrt % nch:
            continue
        CR = RT // nch
        for gy in range(grid_y, 0, -1):
            if CR % gy or CR // gy > 2:
                continue
            if router and not (nrt + 2 * num_links + 1 <= grid_y or (gy < grid_y and nrt <= GX)):
                continue
            out.append((gy, CR // gy, nch))
    return out


_ADDRS: dict = {}  # id(tensor) -> (tensor, its buffer address): long-lived io tensors' addresses, read once


def _addr(t):
    """t's buffer address, read once a tensor object (it holds t: an id is not reused while cached). Only for tensors
    that keep their buffer for their lifetime - a new allocation is a new object, so it misses."""
    hit = _ADDRS.get(id(t))
    if hit is not None and hit[0] is t:
        return hit[1]
    a = int(t.buffer_address())
    if len(_ADDRS) > 4096:
        _ADDRS.clear()
    _ADDRS[id(t)] = (t, a)
    return a


class HeadSE3:
    """The fused head's x chain, v3 (row-chunk-major, weights resident): x [rows, H / TP] a chip -> the x all-gather a
    row chunk at a time straight into the engine's x-own L1 (tt/x_gather3.py) -> per chunk gate / up -> h -> down ->
    y [rows, H] (this chip's partial), the next chunk arriving meanwhile. Engine on columns 0 .. GX - 1, movers next."""

    def __init__(
        self,
        mesh_device,
        *,
        rows,
        H,
        I,
        glu,
        GX=11,
        mover_col=None,
        GY=None,
        MT=None,
        KB=None,
        compact=None,
        num_links=2,
        fidelity=ttnn.MathFidelity.HiFi2,
        rs_dim=None,
        rs_mode="fused",
        router=None,
        latent=None,
        routing=None,
        l1_hole=None,
    ):
        """rs_dim (-2: rows to their owners, GLM; -1: the width, K3): the down + SE reduce-scatter - rs_mode "fused":
        in this program (se_rs: h from the L1 gate / up left it in, bf16 partials into the next chip's same core, fp32
        adds there, y the chip's reduced block - the only DRAM write); "mrs": (the bar) the engine stops at h in DRAM
        and the ported matmul_reduce_scatter op runs on it. None: the engine's own down, y the chip's partial.
        router (RouterSpec): the router chain on the mover column's free cores (tt/router_stream.py) - run() then
        leaves the owners' joint routing rows in self.outputs["rows"]. latent (K3, the latent width Hr): the latent
        down on the gathered x (column split, extra engine passes) and its all-to-all to the owners, the owners'
        [rows / TP, Hr] in self.outputs["latent"]. routing (RoutingSpec, with router): the routing exchange + combine
        planner (tt/routing_stream.py) on the router cores, the gather's ports and the watcher's RISCs - combine_plan's
        per-chip tables in self.routing.outs. l1_hole: two L1 tensors (above the arena), freed right after it is placed
        and after it is placed again over the mover column (the L1 this head keeps - the gather's semaphores, mailboxes,
        the router's and the routing's buffers - then sits above the arena, not stranded below it, where a caller that
        frees the arena between calls has other programs' circular buffers)."""
        from models.demos.moe_block.tt.se_engine3 import SeEngine3, dest_tiles
        from models.demos.moe_block.tt.x_gather3 import XGather3

        self.mesh = mesh_device
        grid = mesh_device.compute_with_storage_grid_size()
        assert GX < grid.x, "no column left for the movers"
        # the K step: the largest <= 16 tiles dividing a chip's shard (whole steps a shard) whose arena fits L1 (smaller
        # steps give the x families more owners: less x a core)
        G = tuple(mesh_device.shape)[1]
        klt = H // 32 // G
        self.mesh, self.rs_dim, self.num_links = mesh_device, rs_dim, num_links
        self.rs_mode = rs_mode if rs_dim is not None else None
        self.xg = None
        self.plan = None
        if self.rs_mode == "mrs":
            self._warm_rs(rows, H, I)
        grid_y = mesh_device.compute_with_storage_grid_size().y
        # (the engine's layout: given; with the fused RS and the router chain the planner's (tt/head_plan.py: every
        # layout / K step / router placement whose L1 fits, by its cost model), tried in its order - the device's L1
        # may still refuse one; else head_layouts' first whose arena fits L1 - at a K step, the largest first)
        self.planned = None
        planned = []
        if GY is None and self.rs_mode == "fused" and router is not None:
            from models.demos.moe_block.tt.head_plan import HeadShape, describe, plan

            shape = HeadShape(
                rows=rows, H=H, I=I, G=G, E=router.E, K=router.K, latent=latent, fold=router.fold,
                links=num_links, ring=self._ring(), grid=(GX + 1, grid_y),
            )  # fmt: skip
            from models.demos.moe_block.tt.se_engine3 import arena_cap

            planned = [
                p_
                for p_ in plan(shape, arena_max=lambda kb: arena_cap(mesh_device, kb), split=router.split)
                if (router.on_engine is None or p_.lay.on_engine == router.on_engine) and (not KB or p_.lay.KB == KB)
            ]
            assert planned, f"HeadSE3: the planner has no layout for {rows} rows ({shape})"
        layouts = (
            [(GY, MT, None)]
            if GY is not None
            else head_layouts(rows, G, grid_y=grid_y, GX=GX, num_links=num_links, router=router is not None)
        )
        assert layouts or planned, f"HeadSE3: no engine layout for {rows} rows over TP {G}"
        kbs = [KB] if KB else sorted((d for d in range(1, 17) if klt % d == 0 and (klt // d) % 2 == 0), reverse=True)
        last_err = None
        self._GX, self._dims = GX, dict(H=H, I=I, G=G, rows=rows)
        self._on = None  # (the planned router placement)
        self._compact = compact  # (the RS zones: the planner's; None - se_rs's default)

        def build(GY, MT, kb):  # (the RS plan and the engine at one layout and K step; False: its arena does not fit)
            nonlocal last_err
            self._lay = (GY, MT, rows // 32 // (GY * MT))
            if self.rs_mode == "fused":
                from models.demos.moe_block.tt.se_rs import RsPlan

                self.plan = RsPlan(
                    G=G,
                    ring=self._ring(),
                    GX=GX,
                    GY=GY,
                    MT=MT,
                    NTH=H // 32,
                    scatter=rs_dim,
                    nv=num_links,
                    pwmax=dest_tiles() // MT,
                    compact=self._compact,
                )
            try:
                self.eng = SeEngine3(
                    mesh_device,
                    rows=rows,
                    H=H,
                    I=I,
                    glu=glu,
                    GX=GX,
                    GY=GY,
                    MT=MT,
                    KB=kb,
                    fidelity=fidelity,
                    shards=G,
                    down=self.rs_mode != "mrs",
                    rs=self.plan,
                    latent=latent // 32 // G if latent else None,
                    prep=bool(routing is not None and routing.prep and self.plan is not None),
                    router=(
                        dict(ET=router.E // 32, KTO=H // G // 32)
                        if router is not None and self.router_on_engine(router, latent) and self.plan is not None
                        else None
                    ),
                )
                return True
            except (ValueError, RuntimeError) as err:  # (an arena too large: the engine's check, or the allocator's)
                if isinstance(err, RuntimeError) and "Out of Memory" not in str(err):
                    raise
                last_err = err
                return False

        self._rc_only = False
        # (the mover column: ports, watcher, router cores - the engine may be narrower, the planner's GX)
        MX = mover_col if mover_col is not None else GX
        if planned:
            for p_ in planned:
                self._on = p_.lay.on_engine
                self._compact = p_.lay.compact
                GX = p_.lay.GX
                self._GX = GX
                if build(p_.lay.GY, p_.lay.MT, p_.lay.KB):
                    self.planned = p_
                    from loguru import logger

                    logger.info(
                        f"HeadSE3: {rows} rows, planned {describe(p_)}; arena {self.eng.arena_bytes // 1024} KB of "
                        f"{self.eng.arena_max // 1024} KB (the device's cap)"
                    )
                    break
            else:
                raise ValueError(f"HeadSE3: no planned layout fits the device's L1 at {rows} rows ({last_err})")
        else:
            # (auto placement: the logits on the engine only at the preferred layout - else the router cores there,
            # not a costlier layout for the engine's sake - self._rc_only)
            auto = router is not None and router.on_engine is None
            for self._rc_only in (False, True)[: 2 if auto else 1]:
                lays = layouts[:1] if auto and not self._rc_only else layouts
                if any(build(GY, MT, kb) for (GY, MT, _) in lays for kb in kbs):
                    break
            else:
                raise ValueError(f"HeadSE3: no layout / K step fits L1 at {rows} rows ({last_err})")
        GY, MT = self._lay[:2]
        holes = list(l1_hole or [])
        if holes:  # (the arena placed: what this head keeps from here on lands in the hole, above it)
            ttnn.deallocate(holes.pop(0))
        movers = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(MX, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
        if self.rs_mode == "mrs":
            ttnn.deallocate(self.eng.arena)
        # (the mover column: ports at the top; with the router chain the watcher sits right below them and the router
        # cores take the rest as one rectangle - their W_r goes out by a column multicast. The router cores own a row
        # tile each of every owner's rows - all of the call's, not a chunk's (their zones and the ports' sub-blocks
        # are a call's: no reuse to credit, and the chain is as short as at one chunk): more than the column's free
        # cells - the row under the engine (GY < grid rows), their W_r by a row multicast)
        nrt = rows // 32 // G
        self.rcores = None
        if router is not None:
            if nrt + 2 * num_links + 1 <= grid.y:
                self.rcores = [ttnn.CoreCoord(MX, cy) for cy in range(grid.y - nrt, grid.y)]
                watcher = ttnn.CoreCoord(MX, grid.y - nrt - 1)
                if router.split:  # (each row tile's second half beside its first: the column left of the movers)
                    assert GX <= MX - 1, f"HeadSE3: the router's split needs column {MX - 1} free (engine GX {GX})"
                    self.rcores += [ttnn.CoreCoord(MX - 1, c.y) for c in self.rcores]
            else:
                # (a free core row under the engine; with the logits on the engine - no W_r multicast over them - the
                # free rows under it)
                assert not router.split, "HeadSE3: the router's split - the mover column's cells only"
                free = [ttnn.CoreCoord(c, y) for y in range(GY, grid.y) for c in range(GX)]
                if not (nrt <= GX or (self.router_on_engine(router, latent) and nrt <= len(free))):
                    raise ValueError(f"HeadSE3: no cells for {nrt} router cores (free core rows: GY {GY} < {grid.y})")
                self.rcores = free[:nrt]
                watcher = ttnn.CoreCoord(MX, grid.y - 1)
        else:
            watcher = None
        self.xg = XGather3(
            mesh_device,
            self.eng,
            num_links=num_links,
            sub_core_grid=movers,
            placement="simple" if grid.x - MX < 4 else "auto",
            watcher=watcher,
        )
        self.eng.ring_dir = self.xg.ring_dir  # (the latent's multi-hop sends: the gather's ring directions)
        if self.rs_mode == "mrs":  # (the gather's semaphores above the arena too: the arena is freed every call)
            self.eng.alloc_arena()
        import os as _os

        if "HW" in _os.environ.get("SE_PROBE", "").split(","):  # (debug: hang words - se_engine3._probes)
            import json as _json

            from models.demos.moe_block.tt import se_engine3 as _se3

            gcr = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
            self.hw = ttnn.from_torch(
                torch.zeros(grid.x * grid.y, 32, dtype=torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(gcr, (1, 32), ttnn.ShardOrientation.ROW_MAJOR),
                ),
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
            _se3.HW_ADDR = int(self.hw.buffer_address())
            _os.makedirs("generated/moe_block", exist_ok=True)  # (for a triage add-on that reads the words)
            _json.dump({"addr": _se3.HW_ADDR}, open("generated/moe_block/hw_addr.json", "w"))
        if self.rs_mode == "fused":  # the ports' mailboxes (a word per engine core, zero) and the arrival counters
            cores = sorted({(c.x, c.y) for ch in self.xg.chips.values() for c in ch["ports"].values()})
            crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*c), ttnn.CoreCoord(*c)) for c in cores])
            # (SE / latent hand-offs, the router cores', the routing's forwards and their landed count, then the logits
            # ports' sub-block words: 16 B a (unit, core) - a port's units (the owners it sends to) x its link's router
            # cores, at least 16)
            HS = persist_l1.handoff_bytes()  # (a hand-off word every HS bytes: 16, or 4 with PERSIST_L1=small)
            self.lw_off = (2 * GX * GY * HS // 4 + 4 * 8 + 4 * 16 + 4 + 3) // 4 * 4
            nrc = rows // 32 // G * (2 if router is not None and router.split else 1)
            self.port_words = max(16, (G - 1) * -(-nrc // num_links))
            # (then, with chunks, the RS zone credits: a port's zone-free word (its chip's cores) and zone-credit word
            # (its peer's port))
            self.zw_off = self.lw_off + 4 * self.port_words
            words = self.zw_off + (8 if self.eng.NCH > 1 else 0)
            self.mbox = ttnn.from_torch(
                torch.zeros(len(cores), words, dtype=torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    persist_l1.buffer_type(),
                    ttnn.ShardSpec(crs, (1, words), ttnn.ShardOrientation.ROW_MAJOR),
                ),
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
            # (arrivals fwd / bwd, the latent's; with chunks the ports' FINB-free credits)
            self.arr_sems = [
                persist_l1.semaphore(mesh_device, self.eng.crs, 0) for _ in range(4 if self.eng.NCH > 1 else 3)
            ]
            ttnn.synchronize_device(mesh_device)
        self.router = None
        if router is not None:
            from models.demos.moe_block.tt.router_stream import RouterStream

            used = {(c.x, c.y) for ch in self.xg.chips.values() for c in ch["ports"].values()}
            wset = {(self.xg.watcher_core(k).x, self.xg.watcher_core(k).y) for k in self.xg.chips}
            used |= wset
            free = self.rcores
            assert not {(c.x, c.y) for c in free} & used, "the router cores' rectangle overlaps the movers"
            assert self.rs_mode == "fused"
            ttnn.deallocate(self.eng.arena)  # (again, also over the router cores and the ports: lockstep L1)
            # (the router cores carve the same range: at least what they need - the engine's own may be smaller)
            need = (RouterStream.min_bytes(G, router.E, router.K, split=router.split) + 63) // 64 * 64
            if self.eng.arena_bytes < need <= self.eng.arena_max:
                self.eng.arena_bytes = need
            self.eng.alloc_arena(
                extra=free[: (2 if router.split else 1) * (rows // 32 // G)]
                + [ttnn.CoreCoord(*c) for c in sorted(used - wset)]
                + [watcher]
            )
            if holes:  # (placed again: the router's and the routing's buffers into the second hole, above it)
                ttnn.deallocate(holes.pop(0))

            self.router = RouterStream(
                mesh_device,
                self.xg,
                rows=rows,
                H=H,
                E=router.E,
                K=router.K,
                route_scale=router.route_scale,
                cores=free,
                prio_first=self.logits_priority(router, latent),
                on_engine=self.router_on_engine(router, latent),
                ordered=router.ordered,
                fold=router.fold,
                split=router.split,
                score=router.score,
                arena=(self.eng.arena, self.eng.base, self.eng.arena_bytes),
                words=int(self.mbox.buffer_address()) + 4 * self.lw_off,
                port_words=self.port_words,
            )
            if self.router.on_engine and self.router_w_gate(router, latent):
                self.router.gate = self.eng.gate_target()
        self.routing = None
        if routing is not None:
            from ttnn.bringup.combine_rf.combine_plan import port_luts
            from ttnn.bringup.combine_rf.routing_a2a import worst_rows

            from models.demos.moe_block.tt.routing_stream import RoutingStream

            R_, C_ = tuple(mesh_device.shape)
            luts, nps = port_luts(mesh_device, routing.L, tuple(routing.torus))  # (the combine's ports: its torus)
            S_ = rows // G
            EPC = router.E // router.fold // (R_ * C_)
            rows_cap = routing.rows_cap or worst_rows(R_ * C_, S_, router.K, EPC)  # (every row a chip can get)
            off0 = (max(self.router.bytes, self.router.pbuf_sub) + 63) // 64 * 64
            self.routing = RoutingStream(
                mesh_device,
                self.xg,
                self.router,
                S=S_,
                K=router.K,
                EPC=EPC,
                A=routing.A,
                L=routing.L,
                luts=luts,
                nps=nps,
                rows_cap=rows_cap,
                ycores=[watcher],
                arena=(self.eng.base, self.eng.arena_bytes, off0),
                flags=int(self.mbox.buffer_address()) + persist_l1.handoff_bytes() * 2 * GX * GY + 128,
            )
            self.router.ra = self.routing
            if routing.prep:
                self.router.prep = dict(EPC=EPC, torus=tuple(routing.torus))
            ttnn.synchronize_device(mesh_device)
        self.rows, self.H, self.I, self.rs_dim, self.num_links = rows, H, I, rs_dim, num_links
        self.latent = latent
        self.outputs = {}
        self.state = dict(key=None, arena=None)

    def upload(self, w_gu, w_down, w_latent=None):
        """w_gu / w_down per device (row-major mesh order). With rs_dim: (engine weights, W_down [I, H] bf8 a chip)."""
        if self.rs_mode != "mrs":
            pos = None
            if self.rs_mode == "fused":
                R, C = tuple(self.mesh.shape)
                pos = [self.xg.chips[(d // C, d % C)]["rings"][0]["p"] for d in range(R * C)]
            w_lat = None
            if self.latent:  # (the latent down [Hr, H]: chip p's columns of it, [H, Hr / TP])
                G = self.plan.G
                w_lat = [
                    w_latent[d][pos[d] * self.latent // G : (pos[d] + 1) * self.latent // G].T.contiguous()
                    for d in range(len(pos))
                ]
            return self.eng.upload(w_gu, w_down, pos=pos, w_lat=w_lat)
        R, C = tuple(self.mesh.shape)
        glob = torch.cat([torch.cat(list(w_down[r * C : (r + 1) * C]), dim=1) for r in range(R)], dim=0)
        wd = ttnn.from_torch(
            glob,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh, dims=(0, 1), mesh_shape=(R, C)),
        )
        return self.eng.upload(w_gu, w_down), wd

    def _ring(self):
        if self.xg is not None:
            return self.xg.ring
        cfg = ttnn.get_fabric_config()  # (as the gather decides it)
        return cfg in (ttnn.FabricConfig.FABRIC_2D_TORUS_X, ttnn.FabricConfig.FABRIC_2D_TORUS_XY) and (
            tuple(self.mesh.shape)[1] > 2
        )

    def _warm_rs(self, rows, H, I):
        """The op's persistent L1 (staging, mailbox) before the arena and the gather's semaphores (a call on zeros):
        the arena is freed for the op's CBs every call, and must find its place free again."""
        zeros = lambda shape, dt: ttnn.from_torch(  # noqa: E731
            torch.zeros(shape),
            dtype=dt,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        h, wd = zeros((1, 1, rows, I), ttnn.bfloat16), zeros((I, H), ttnn.bfloat8_b)
        for t in (self._rs(h, wd), h, wd):
            ttnn.deallocate(t)

    def _rs(self, h, wd):
        from ttnn.operations.ccl import Topology
        from ttnn.operations.matmul_reduce_scatter import matmul_reduce_scatter

        return matmul_reduce_scatter(
            h,
            wd,
            cluster_axis=1,
            scatter_dim=self.rs_dim,
            topology=Topology.Ring if self._ring() else Topology.Linear,
            num_links=self.num_links,
        )

    def router_on_engine(self, router, latent):
        """Where the logits' matmul runs: RouterSpec.on_engine, or router_placement's choice at the engine's layout. On
        the engine every chip's routing chain starts earlier but gate / up later by the logits phase (GLM ~16 us, K3 E
        448 ~35). Measured: GLM (the chain binds) 166 us on the engine vs 191 on the router cores, K3 E 448 (the SE path
        binds) 424 vs 388 Galaxy-projected; the Galaxy's K3 at E 896 on the router cores 650 us (their 316 us matmul
        bound it). An engine layout that does not fit L1: the router cores (self._rc_only)."""
        if router.on_engine is not None:
            return router.on_engine
        if self._on is not None:  # (the planner's)
            return self._on
        if self._rc_only:
            return False
        GY, MT, NCH = self._lay
        on, _ = router_placement(GX=self._GX, GY=GY, MT=MT, NCH=NCH, latent=latent, **self._dims, E=router.E)
        return on

    def router_w_gate(self, router, latent):
        """The engine's W streams gated on every engine core's partials written (RouterSpec.w_gate). Ungated, the
        cores done with their partials start their W streams (DRAM reads, NOC1) and slow the rest's writes (router
        cores only: GLM's in at ~47 us, ~22 gated - probe: the W streams held back). With the other owners' partials
        straight to the ports (5 router + the port cores take them) ungated is 18 -> 30 us, and the gate's later W
        costs the SE path ~10 us (GLM Galaxy-projected 179 gated, 166 not: both paths 166) - off by default."""
        if router.w_gate is not None:
            return router.w_gate
        return False

    def logits_priority(self, router, latent):
        """The ports' logits-vs-x-gather priority. From the Galaxy projections (tests/galaxy_projection.py) and the
        LoudBox runs: on a TP ring the gather is 1.5 shards a link direction and has slack under gate / up, and the
        routing chain is what binds - logits first (K3 Galaxy-projected 466 us vs 505 x first; GLM's chain binds on
        both). On a line the gather is 3 shards on the end links: with a latent all-to-all also on the ports during
        the gather (K3) the logits delay the SE path more than they shorten the chain (509 vs 527 us measured) - x
        first; without one (GLM) logits first. With the logits' matmul on the engine (router_on_engine) logits first
        on a line too (K3 Galaxy-projected 367 us vs 380 x first from the mesh's last launch, 397 vs 461 eager)."""
        if router.logits_first is not None:
            return router.logits_first
        # (measured, traced, 640 / 1024 rows, logits first vs x first: the 2x4 line - K3 638 vs 558 us, 1120 vs 1083,
        # GLM-5.3 392 vs 383, K2.7 443 vs 433; the ring of 4 - K3 536 vs 458, 877 vs 852, GLM-5.3 312 vs 296, K2.7 331
        # vs 326: x first on every mesh of up to 8 chips. Larger meshes (the Galaxy's N 32 routing chain binds) keep
        # the projections' rule below until measured)
        R_, C_ = tuple(self.mesh.shape)
        if R_ * C_ <= 8:
            return False
        if latent and self._ring():  # (measured on a ring of 4, K3 E 896: x first 453-455 us traced, logits first
            return False  # 479-483 - the chain still ends ~90 us before the SE path, ~+20 for the Galaxy's N 32)
        return self._ring() or not latent or self.router_on_engine(router, latent)

    def upload_router(self, w_router, bias):
        self.rw = self.router.upload(w_router, bias)

    def use_layer(self, key, rw):
        """Run with this layer's weights (several layers on one head): its router weights; the shared expert's are
        run()'s w. The kernels take every weight tensor at its base address (offsets are args of their own), so the
        programs stay and the prepared dispatch patches the weights' slots - no rebuild a layer."""
        self.layers = True
        self.rw = rw

    def _rebind(self):
        """The cached programs' arena-backed CBs (the engine's, the router cores') onto the arena tensor taken again at
        the same address (a CB binds its buffer: a freed one is no use) - not a rebuild of every chip's program."""
        tmpl = ttnn.cb_descriptor_from_sharded_tensor(0, self.eng.arena)
        dd = self.eng.arena_delta - self.state.get("delta", 0)  # (onto another tensor: the same L1, offsets shift)
        md = ttnn.MeshProgramDescriptor()
        for (r, c), prog, on in self.state["progs"]:
            cbs = list(prog.cbs)
            for i in on:
                cbs[i].set_buffer_from_cb(tmpl)
                cbs[i].address_offset = cbs[i].address_offset + dd
            prog.cbs = cbs
            md[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, c), ttnn.MeshCoordinate(r, c))] = prog
        self.state.update(desc=md, arena=self.eng.arena, delta=self.eng.arena_delta)

    def settle_arena(self):
        """Take the arena once more and give it back, filling any free L1 right above its range (see
        SeEngine3.realloc_arena): before the L1 it shares with another op is placed."""
        if not self.eng.arena.is_allocated():
            self.eng.realloc_arena(settle=True)
            if self.router is not None:
                self.router.arena, self.router.arena_delta = self.eng.arena, 0
        self.free_arena()

    def use_arena(self, arena, nbytes):
        """This call on the caller's L1 tensor (nbytes a core) holding the engine's arena range (the MoE block's one
        arena a forward, the exchange's): no allocation of its own; the caller frees it (not free_arena)."""
        d = self.eng.use_arena(arena, nbytes)
        if self.router is not None:
            self.router.arena, self.router.arena_delta = arena, d

    def free_arena(self):
        """Give the arena's L1 back between calls (a caller sharing the L1 left with another op): run() takes it
        again, at the same address."""
        if self.eng.arena.is_allocated():
            ttnn.deallocate(self.eng.arena)

    def _out(self, name, make):
        """A per-call output: with persist (the MoE block with its fused tail - nothing outlives a forward), made once
        and reused - its DRAM stays, the allocation (~10-20 us of host each, ahead of the head's launch) goes."""
        t = self._persist.get(name) if getattr(self, "persist", False) else None
        if t is None or not t.is_allocated():
            t = make()
            if getattr(self, "persist", False):
                self._persist[name] = t
        return t

    def run(self, x, w):
        if not hasattr(self, "_persist"):
            self._persist = {}
        if self.rs_mode == "fused" and not self.eng.arena.is_allocated():
            self.eng.realloc_arena()
            if self.router is not None:  # (its CBs are carved from the arena too)
                self.router.arena, self.router.arena_delta = self.eng.arena, 0
        if self.rs_mode == "fused":
            G = self.plan.G
            self.outputs = {}
            if self.router is not None:
                self.rout = self.outputs["rows"] = self._out("rows", self.router.out_tensor)
                if self.routing is not None:
                    self.routing.router_out = self.rout
                if self.router.prep is not None:
                    from ttnn.bringup.combine_rf.sparse_all_gather_2d import route_words

                    N = self.mesh.get_num_devices()
                    rw = route_words(N)
                    # the dispatch buffer [N S, W] row-major (W: the latent's width, else H): the program writes this
                    # chip's slots me S .. (its owned rows); the dispatch fills the rest
                    self.outputs["xbuf"] = self._out(
                        "xbuf",
                        lambda: ttnn.allocate_tensor_on_device(
                            ttnn.Shape([1, 1, N * self.rows // G, self.latent or self.H]),
                            ttnn.bfloat16,
                            ttnn.ROW_MAJOR_LAYOUT,
                            self.mesh,
                            ttnn.DRAM_MEMORY_CONFIG,
                        ),
                    )
                    self.outputs["route"] = self._out(
                        "route",
                        lambda: ttnn.allocate_tensor_on_device(
                            ttnn.Shape([1, 1, 1, rw * self.rows // G]),
                            ttnn.uint32,
                            ttnn.ROW_MAJOR_LAYOUT,
                            self.mesh,
                            ttnn.DRAM_MEMORY_CONFIG,
                        ),
                    )
            if self.latent:
                self.lout = self.outputs["latent"] = self._out(
                    "latent",
                    lambda: ttnn.allocate_tensor_on_device(
                        ttnn.Shape([1, 1, self.rows // G, self.latent]),
                        ttnn.bfloat16,
                        ttnn.TILE_LAYOUT,
                        self.mesh,
                        ttnn.DRAM_MEMORY_CONFIG,
                    ),
                )
            return self._run(
                x, w, self.H // G if self.rs_dim == -1 else self.H, self.rows // G if self.rs_dim == -2 else self.rows
            )
        if self.rs_mode == "mrs":
            w, wd = w
            if not self.eng.arena.is_allocated():  # (the arena is call scratch: freed for the op's CBs below)
                self.eng.alloc_arena()
            h = self._run(x, w, self.I)
            self.state.update(key=None, desc=None)  # (its CB descriptors hold the arena's buffer)
            ttnn.deallocate(self.eng.arena)  # (the queue keeps the order: the op's CBs reuse its L1 after it)
            out = self._rs(h, wd)
            ttnn.deallocate(h)
            return out
        return self._run(x, w, self.H)

    def _run(self, x, w, width, rows=None):
        from models.demos.moe_block.tt.se_engine3 import XREADY

        y = self._out(
            "y",
            lambda: ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, 1, rows or self.rows, width]),
                ttnn.bfloat16,
                ttnn.TILE_LAYOUT,
                self.mesh,
                ttnn.DRAM_MEMORY_CONFIG,
            ),
        )
        io = [x, w, y, self.eng.arena]
        io += list(self.outputs.values()) + (list(self.rw) if self.router is not None else [])
        if self.routing is not None:
            io += [self.routing.slots] + list(self.routing.outs.values())
            if self.routing.dbg is not None:
                io += [self.routing.dbg]
        # Prepared dispatch (ttnn.bringup.prepare_program): the descriptor is built and prepared once; each call
        # patches only the runtime args holding an io tensor's address (x: the block input, at a new address per
        # call - its args are its base address) and the CBs on the arena. Rebuilt when the arena's base or any other
        # io tensor's address moves (their args may point inside them, which exact-address patching cannot follow)
        # (several layers on the head - use_layer: the weights' slots are patched like x's, their addresses out of the
        # keys - the shared expert's w, the router's tensors)
        layers = getattr(self, "layers", False)
        nr = 4 + len(self.outputs)
        wpos = {1} | (set(range(nr, nr + len(self.rw))) if self.router is not None else set())
        keep = getattr(self, "persist", False)  # (the outputs reused across calls: their addresses can be cached)
        # (the key's addresses: the arena's read each call - a new tensor every forward - the rest, long-lived, from
        # their first read: buffer_address() on a mesh tensor costs ~20 us, ~17 of them a call were most of _run)
        skey = (self.eng.base, len(io)) + tuple(
            0 if layers and i in wpos else (_addr(t) if keep and i != 3 else int(t.buffer_address()))
            for i, t in enumerate(io)
            if i
        )
        if self.state.get("pp") is not None and self.state.get("skey") == skey:
            ttnn.bringup.run_prepared(self.state["pp"], io)
            return y
        key = (
            int(x.buffer_address()),
            int(y.buffer_address()),
            0 if layers else int(w.buffer_address()),
            self.eng.base,
        )
        key += tuple(int(t.buffer_address()) for t in self.outputs.values())
        if self.state["key"] == key and self.state["arena"] is not self.eng.arena:
            self._rebind()  # (the arena taken again at its address: only the CBs bound to it change)
        if self.state["key"] != key:
            progs = []
            md = ttnn.MeshProgramDescriptor()
            e = self.eng
            for coord in self.xg.chips:
                prog = ttnn.ProgramDescriptor()
                p = self.xg.chips[coord]["rings"][0]["p"]
                rsx, rse = None, {}
                if self.rs_mode == "fused":
                    virt = self.mesh.worker_core_from_logical_core
                    pc = {(d, l): virt(c) for (_, d, l), c in self.xg.chips[coord]["ports"].items()}
                    mbox = int(self.mbox.buffer_address())
                    arr = tuple(persist_l1.address(s_) for s_ in self.arr_sems)
                    zfree = mbox + 4 * self.zw_off
                    rsx = dict(
                        plan=self.plan,
                        p=p,
                        finb=e.base + e.off["finb"],
                        mbox=mbox,
                        zone=(e.base + e.off["arrf"], e.base + e.off["arrb"]),
                        arr=arr,
                        ffree=arr[3] if len(arr) > 3 else 0,
                        zfree=zfree,
                        zcred=zfree + 16,
                    )
                    rsx["coord"] = coord
                    la = None
                    if self.latent:
                        # (fence: the latent's end-to-end "started" word - the old router words' slot, zeroed, free)
                        la = dict(
                            word=mbox + persist_l1.handoff_bytes() * e.GX * e.GY + 128, ctr=arr[2], out=self.lout, fence=mbox + persist_l1.handoff_bytes() * e.GX * e.GY
                        )
                    rse = dict(
                        rs_pos=p,
                        rs_port=lambda d, l, pc=pc: (pc[(d, l)].x, pc[(d, l)].y),
                        rs_mbox=mbox,
                        rs_arr=arr[:2],
                        rs_chunk=dict(zfree=zfree, ffree=arr[3] if len(arr) > 3 else 0),
                        la=la,
                    )
                if self.router is not None:  # (first: the routing's serve plan is made by the router's planner args)
                    c2, k2 = self.router.pieces(coord, x, *self.rw, self.rout, route=self.outputs.get("route"))
                c0, k0 = self.xg.pieces(
                    coord, prog, x, XREADY, rs=rsx, router=self.router, la=rse.get("la"), routing=self.routing
                )
                if self.router is not None:
                    c0, k0 = c0 + c2, k0 + k2
                if self.routing is not None:
                    k0 = k0 + self.routing.y_kernels(coord, prog)
                c1, k1, sems = e.pieces(
                    x,
                    w,
                    y,
                    x_k0=p * self.xg.klt,
                    x_ct=self.xg.klt,
                    shard_order=self.xg.order(coord)[0],
                    ports=self.xg.port_cores(coord),
                    xloaded=self.xg.xloaded,
                    extra_sem_cores=[self.xg.watcher_core(coord)],
                    rtr=(
                        dict(x=x, w=self.rw[0], gate=self.router.gate is not None, **self.router.engine_args(coord))
                        if self.router is not None and self.router.on_engine
                        else None
                    ),
                    prep=(
                        dict(
                            xbuf=self.outputs["xbuf"],
                            me=coord[0] * tuple(self.mesh.shape)[1] + coord[1],
                            shard_order=self.xg.order(coord)[0],
                        )
                        if "xbuf" in self.outputs
                        else None
                    ),
                    **rse,
                )
                prog.cbs, prog.semaphores = c0 + c1, sems
                e.attach_mcast(prog)
                if self.router is not None:
                    self.router.attach_mcast(prog)
                prog.kernels = k0 + k1
                r, c = coord
                base = self.eng.base  # (which CBs the arena backs: _rebind's - a freed buffer reports no address)
                ab = int(self.eng.arena.buffer_address())  # (a caller's arena: at base - arena_delta)
                on = [
                    i
                    for i, cb in enumerate(prog.cbs)
                    if cb.has_buffer()
                    and (base in (cb.buffer_address(), cb.buffer_address() - cb.address_offset) or cb.buffer_address() == ab)
                ]
                # (its structural hash once: a cached descriptor is not re-hashed - kernel sources and all - each call)
                prog.custom_program_hash = ttnn.compute_program_descriptor_hash(prog)
                md[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, c), ttnn.MeshCoordinate(r, c))] = prog
                progs.append((coord, prog, on))
            self.state.update(key=key, desc=md, progs=progs, arena=self.eng.arena, delta=self.eng.arena_delta)
        self.state.update(pp=ttnn.bringup.prepare_program(self.state["desc"], io), skey=skey)
        ttnn.bringup.run_prepared(self.state["pp"], io)
        return y
