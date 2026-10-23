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


@dataclass(frozen=True)
class RoutingSpec:
    """The fused head's routing exchange + combine planner: A combine accumulators, L links of the combine's ports
    (its port tables), rows_cap the plan's row capacity (default 2 S K + 32 EPC). prep: dispatch prep in the program
    (the owned tokens' route words, self.outputs["route"]); torus: (wrap X, wrap Y) of the dispatch's mesh."""

    A: int
    L: int = 2
    rows_cap: int = 0
    prep: bool = True
    torus: tuple = (False, False)


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
        GY=10,
        MT=2,
        KB=None,
        num_links=2,
        fidelity=ttnn.MathFidelity.HiFi2,
        rs_dim=None,
        rs_mode="fused",
        router=None,
        latent=None,
        routing=None,
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
        per-chip tables in self.routing.outs."""
        from models.demos.moe_block.tt.se_engine3 import SeEngine3
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
        elif self.rs_mode == "fused":
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
                pwmax=8 // MT,
            )
        for kb in (
            [KB] if KB else sorted((d for d in range(1, 17) if klt % d == 0 and (klt // d) % 2 == 0), reverse=True)
        ):
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
                break
            except ValueError as err:
                last_err = err
                continue
        else:
            raise ValueError(f"HeadSE3: no K step fits L1 at {rows} rows ({last_err})")
        movers = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(GX, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
        if self.rs_mode == "mrs":
            ttnn.deallocate(self.eng.arena)
        # (the mover column: ports at the top; with the router chain the watcher sits right below them and the router
        # cores take the rest as one rectangle - their W_r goes out by a column multicast)
        nrt = rows // 32 // G
        watcher = ttnn.CoreCoord(GX, grid.y - nrt - 1) if router is not None else None
        self.xg = XGather3(
            mesh_device,
            self.eng,
            num_links=num_links,
            sub_core_grid=movers,
            placement="simple" if grid.x - GX < 4 else "auto",
            watcher=watcher,
        )
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
            # ports' sub-block words: 16 B a (unit, core), 16 a port)
            self.lw_off = (2 * GX * GY * 4 + 4 * 8 + 4 * 16 + 4 + 3) // 4 * 4
            words = self.lw_off + 4 * 16
            self.mbox = ttnn.from_torch(
                torch.zeros(len(cores), words, dtype=torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(crs, (1, words), ttnn.ShardOrientation.ROW_MAJOR),
                ),
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
            self.arr_sems = [ttnn.create_global_semaphore(mesh_device, self.eng.crs, 0) for _ in range(3)]
            ttnn.synchronize_device(mesh_device)
        self.router = None
        if router is not None:
            from models.demos.moe_block.tt.router_stream import RouterStream

            used = {(c.x, c.y) for ch in self.xg.chips.values() for c in ch["ports"].values()}
            wset = {(self.xg.watcher_core(k).x, self.xg.watcher_core(k).y) for k in self.xg.chips}
            used |= wset
            free = [ttnn.CoreCoord(GX, cy) for cy in range(grid.y - nrt, grid.y)]
            assert not {(c.x, c.y) for c in free} & used, "the router cores' rectangle overlaps the movers"
            assert self.rs_mode == "fused"
            ttnn.deallocate(self.eng.arena)  # (again, also over the router cores and the ports: lockstep L1)
            self.eng.alloc_arena(
                extra=free[: rows // 32 // G] + [ttnn.CoreCoord(*c) for c in sorted(used - wset)] + [watcher]
            )
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
                arena=(self.eng.arena, self.eng.base, self.eng.arena_bytes),
                words=int(self.mbox.buffer_address()) + 4 * self.lw_off,
            )
            if self.router.on_engine and self.router_w_gate(router, latent):
                self.router.gate = self.eng.gate_target()
        self.routing = None
        if routing is not None:
            from ttnn.bringup.combine_rf.combine_plan import port_luts

            from models.demos.moe_block.tt.routing_stream import RoutingStream

            R_, C_ = tuple(mesh_device.shape)
            luts, nps = port_luts(mesh_device, routing.L)
            S_ = rows // G
            EPC = router.E // (R_ * C_)
            rows_cap = routing.rows_cap or (2 * S_ * router.K + 32 * EPC + 31) // 32 * 32
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
                flags=int(self.mbox.buffer_address()) + 16 * 2 * GX * GY + 128,
            )
            self.router.ra = self.routing
            if routing.prep:
                self.router.prep = dict(EPC=EPC, torus=tuple(routing.torus))
            ttnn.synchronize_device(mesh_device)
        self.rows, self.H, self.I, self.rs_dim, self.num_links = rows, H, I, rs_dim, num_links
        self.latent = latent
        self.outputs = {}
        self.state = dict(key=None)

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
        """Where the logits' matmul runs. On the engine's cores at the start, every chip's routing chain starts earlier
        but gate / up starts later by the router phase (GLM ~16 us, K3 ~35: gate / up's end with no gather wait 89 vs 73
        / 307 vs 272 us, SE_PROBE=NOXWAIT). Galaxy-projected from the mesh's last launch: GLM (the chain binds) 166 us
        on the engine vs 191 on the router cores; K3 (the SE path binds) 424 vs 388 - so on where there is no latent."""
        if router.on_engine is not None:
            return router.on_engine
        return not latent

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
        return self._ring() or not latent or self.router_on_engine(router, latent)

    def upload_router(self, w_router, bias):
        self.rw = self.router.upload(w_router, bias)

    def run(self, x, w):
        if self.rs_mode == "fused":
            G = self.plan.G
            self.outputs = {}
            if self.router is not None:
                self.rout = self.outputs["rows"] = self.router.out_tensor()
                if self.routing is not None:
                    self.routing.router_out = self.rout
                if self.router.prep is not None:
                    from ttnn.bringup.combine_rf.sparse_all_gather_2d import route_words

                    N = self.mesh.get_num_devices()
                    rw = route_words(N)
                    # the dispatch buffer [N S, W] row-major (W: the latent's width, else H): the program writes this
                    # chip's slots me S .. (its owned rows); the dispatch fills the rest
                    self.outputs["xbuf"] = ttnn.allocate_tensor_on_device(
                        ttnn.Shape([1, 1, N * self.rows // G, self.latent or self.H]),
                        ttnn.bfloat16,
                        ttnn.ROW_MAJOR_LAYOUT,
                        self.mesh,
                        ttnn.DRAM_MEMORY_CONFIG,
                    )
                    self.outputs["route"] = ttnn.allocate_tensor_on_device(
                        ttnn.Shape([1, 1, 1, rw * self.rows // G]),
                        ttnn.uint32,
                        ttnn.ROW_MAJOR_LAYOUT,
                        self.mesh,
                        ttnn.DRAM_MEMORY_CONFIG,
                    )
            if self.latent:
                self.lout = self.outputs["latent"] = ttnn.allocate_tensor_on_device(
                    ttnn.Shape([1, 1, self.rows // G, self.latent]),
                    ttnn.bfloat16,
                    ttnn.TILE_LAYOUT,
                    self.mesh,
                    ttnn.DRAM_MEMORY_CONFIG,
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

        y = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, rows or self.rows, width]),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            self.mesh,
            ttnn.DRAM_MEMORY_CONFIG,
        )
        key = (int(x.buffer_address()), int(y.buffer_address()), int(w.buffer_address()), id(self.eng.arena))
        key += tuple(int(t.buffer_address()) for t in self.outputs.values())
        if self.state["key"] != key:
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
                    arr = tuple(int(ttnn.get_global_semaphore_address(s_)) for s_ in self.arr_sems)
                    rsx = dict(
                        plan=self.plan,
                        p=p,
                        finb=e.base + e.off["finb"],
                        mbox=mbox,
                        zone=(e.base + e.off["arrf"], e.base + e.off["arrb"]),
                        arr=arr,
                    )
                    rsx["coord"] = coord
                    la = None
                    if self.latent:
                        # (fence: the latent's end-to-end "started" word - the old router words' slot, zeroed, free)
                        la = dict(
                            word=mbox + 16 * e.GX * e.GY + 128, ctr=arr[2], out=self.lout, fence=mbox + 16 * e.GX * e.GY
                        )
                    rse = dict(
                        rs_pos=p,
                        rs_port=lambda d, l, pc=pc: (pc[(d, l)].x, pc[(d, l)].y),
                        rs_mbox=mbox,
                        rs_arr=arr[:2],
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
                md[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, c), ttnn.MeshCoordinate(r, c))] = prog
            self.state["key"], self.state["desc"] = key, md
        io = [x, w, y, self.eng.arena]
        io += list(self.outputs.values()) + (list(self.rw) if self.router is not None else [])
        if self.routing is not None:
            io += [self.routing.slots] + list(self.routing.outs.values())
            if self.routing.dbg is not None:
                io += [self.routing.dbg]
        ttnn.generic_op(io, self.state["desc"])
        return y
