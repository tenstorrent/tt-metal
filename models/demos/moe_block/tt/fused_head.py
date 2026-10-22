# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""The MoE block's fused head: one custom program per chip (MOE_BLOCK_WORKLOG.md, "the fused head"), built up a chain at
a time. So far the x chain: the x all-gather over TP (tt/tp_gather.py's movers, a chunk of 32 GY rows at a time) ->
the shared expert's gate / up + GLU (tt/lin2c.py engine, starts on chunk m once the watcher saw it land) -> h (DRAM)
-> down (a second engine, chunk by chunk) -> y. Columns (12 x 10 grid): gate / up 0 .., down after it, the movers in
the last column."""

from dataclasses import dataclass
from typing import Literal

import torch

import ttnn
from models.demos.moe_block.tt.lin2c import XREADY, Lin2c, semaphores
from models.demos.moe_block.tt.tp_gather import TpGather


@dataclass(frozen=True)
class HeadLayout:
    """Core columns of the fused head's engines (the movers take the columns after them)."""

    gu_cols: int = 8
    down_cols: int = 3
    rows: int = 10  # (engine core rows: a chunk is 32 x rows rows)
    num_links: int = 2
    group: int = 8  # (the movers' packets / runs in flight a core; <= 11: the packet header pool)
    w_depth: int = 4  # (the engines' W stream steps in flight a reader: lin2c nstw)
    movers: Literal["last", "first"] = "last"  # (the movers' column: east edge, or west edge by DRAM column 0)


class HeadXChain:
    """x [rows, H / TP] a chip -> y [rows, H] = SE(all-gathered x) (the shared expert's partial over this chip's
    I / TP slice; the reduce-scatter comes next). glu: "silu" / "situ"; I: the shared expert's I / TP a chip."""

    # probes (timing only, off by default): "gather_only": the movers alone; "chain_only": the engines alone (x_full as
    # it is); "nogate": both, the engines not waiting for the gather
    Probe = Literal["", "gather_only", "chain_only", "nogate"]

    def __init__(
        self, mesh_device, *, rows, H, I, glu, layout=HeadLayout(), fidelity=ttnn.MathFidelity.HiFi2, probe: Probe = ""
    ):
        self.mesh = mesh_device
        self.probe = probe
        G = tuple(mesh_device.shape)[1]
        L = self.layout = layout
        M = 32 * L.rows
        assert rows % M == 0, f"rows {rows} must be whole chunks of {M}"
        MC = rows // M
        grid = mesh_device.compute_with_storage_grid_size()
        mv = L.gu_cols + L.down_cols
        assert mv < grid.x, "no column left for the movers"
        nmv = grid.x - mv
        m0, e0 = (mv, 0) if L.movers == "last" else (0, nmv)  # (first mover column, first engine column)
        movers = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(m0, 0), ttnn.CoreCoord(m0 + nmv - 1, grid.y - 1))])
        self.tg = TpGather(
            mesh_device,
            rows,
            H // G,
            chunk_rows=M,
            num_links=L.num_links,
            group=L.group,
            sub_core_grid=movers,
            placement="simple" if nmv < 4 else "auto",
        )
        self.gu = Lin2c(
            mesh_device,
            cx0=e0,
            GX=L.gu_cols,
            GY=L.rows,
            MC=MC,
            K=H,
            N=2 * I,
            glu=glu,
            fidelity=fidelity,
            nstw=L.w_depth,
            xr_need=0 if probe in ("chain_only", "nogate") else 1,
        )
        self.dn = Lin2c(
            mesh_device,
            cx0=e0 + L.gu_cols,
            GX=L.down_cols,
            GY=L.rows,
            MC=MC,
            K=I,
            N=H,
            fidelity=fidelity,
            nstw=L.w_depth,
            xr_need=L.gu_cols,
        )
        self.rows, self.H, self.I = rows, H, I
        self.state = dict(key=None)

    def upload(self, w_gu, w_down):
        """Per device (row-major mesh order): w_gu [H, 2 I] as gate / up tile pairs (lin2c.glu_weight), w_down [I, H]."""
        return self.gu.upload(w_gu), self.dn.upload(w_down)

    def _alloc(self, cols):
        return ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, self.rows, cols]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh, ttnn.DRAM_MEMORY_CONFIG
        )

    def run(self, x, weights):
        wl, wld = weights
        bufs = self.state.get("bufs")
        if bufs is None:
            bufs = self.state["bufs"] = (self.tg.alloc_out(), self._alloc(self.I))
        x_full, h = bufs
        y = self._alloc(self.H)
        key = (int(x.buffer_address()), int(y.buffer_address()))
        if self.state["key"] != key:
            md = ttnn.MeshProgramDescriptor()
            senders = [c for cy in range(self.layout.rows) for c in self.gu.x_senders(cy)]
            for coord in self.tg.chips:
                prog = ttnn.ProgramDescriptor()
                gated = self.probe == ""
                c0, k0 = (
                    ([], [])
                    if self.probe == "chain_only"
                    else self.tg.pieces(coord, prog, x, x_full, notify=(senders, XREADY) if gated else None)
                )
                c1, k1 = self.gu.pieces(
                    x_full, wl, h, note=[self.dn.x_senders(cy) for cy in range(self.layout.rows)], note_sem=XREADY
                )
                c2, k2 = self.dn.pieces(h, wld, y)
                if self.probe == "gather_only":
                    c1, k1, c2, k2 = [], [], [], []
                prog.cbs, prog.kernels = c0 + c1 + c2, k0 + k1 + k2
                if self.probe != "gather_only":
                    prog.semaphores = semaphores(
                        self.gu, self.dn, extra_cores=[self.tg.watcher_core(coord)] if gated else []
                    )
                r, c = coord
                md[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, c), ttnn.MeshCoordinate(r, c))] = prog
            self.state["key"], self.state["desc"] = key, md
        ttnn.generic_op([x, x_full] + list(wl) + [h] + list(wld) + [y], self.state["desc"])
        return y


class HeadSE:
    """The fused head's x chain, v2: x [rows, H / TP] a chip -> the x all-gather straight into the shared expert
    engine's L1 (tt/x_gather.py) -> gate / up (each weight block read once, steps in shard arrival order: own shard
    first) -> h in L1 -> down -> y [rows, H] (this chip's partial). The engine on columns 0 .. GX - 1, the movers on
    the next column."""

    def __init__(self, mesh_device, *, rows, H, I, glu, GX=11, GY=10, num_links=2, fidelity=ttnn.MathFidelity.HiFi2):
        from models.demos.moe_block.tt.se_engine import SeEngine
        from models.demos.moe_block.tt.x_gather import XGather

        self.mesh = mesh_device
        grid = mesh_device.compute_with_storage_grid_size()
        assert GX < grid.x, "no column left for the movers"
        self.eng = SeEngine(mesh_device, rows=rows, H=H, I=I, glu=glu, GX=GX, GY=GY, fidelity=fidelity)
        movers = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(GX, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
        self.xg = XGather(
            mesh_device,
            self.eng,
            num_links=num_links,
            sub_core_grid=movers,
            placement="simple" if grid.x - GX < 4 else "auto",
        )
        self.rows, self.H = rows, H
        self.state = dict(key=None)

    def upload(self, w_gu, w_down):
        return self.eng.upload(w_gu, w_down)

    def run(self, x, w):
        from models.demos.moe_block.tt.se_engine import XREADY

        y = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, self.rows, self.H]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh, ttnn.DRAM_MEMORY_CONFIG
        )
        key = (int(x.buffer_address()), int(y.buffer_address()), int(w.buffer_address()))
        if self.state["key"] != key:
            md = ttnn.MeshProgramDescriptor()
            for coord in self.xg.chips:
                prog = ttnn.ProgramDescriptor()
                c0, k0 = self.xg.pieces(coord, prog, x, XREADY)
                order, need = self.xg.step_order(coord)
                p = self.xg.chips[coord]["rings"][0]["p"]
                c1, k1, sems = self.eng.pieces(
                    x,
                    w,
                    y,
                    order=order,
                    need=need,
                    x_k0=p * self.xg.klt,
                    x_ct=self.xg.klt,
                    extra_sem_cores=[self.xg.watcher_core(coord)],
                )
                prog.cbs, prog.kernels, prog.semaphores = c0 + c1, k0 + k1, sems
                r, c = coord
                md[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, c), ttnn.MeshCoordinate(r, c))] = prog
            self.state["key"], self.state["desc"] = key, md
        ttnn.generic_op([x, w, y, self.eng.arena], self.state["desc"])
        return y


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
    ):
        """rs_dim (-2: rows to their owners, GLM; -1: the width, K3): the down + SE reduce-scatter - rs_mode "fused":
        in this program (se_rs: h from the L1 gate / up left it in, bf16 partials into the next chip's same core, fp32
        adds there, y the chip's reduced block - the only DRAM write); "mrs": (the bar) the engine stops at h in DRAM
        and the ported matmul_reduce_scatter op runs on it. None: the engine's own down, y the chip's partial."""
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
                )
                break
            except ValueError:
                continue
        else:
            raise ValueError(f"HeadSE3: no K step fits L1 at {rows} rows")
        movers = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(GX, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
        if self.rs_mode == "mrs":
            ttnn.deallocate(self.eng.arena)
        self.xg = XGather3(
            mesh_device,
            self.eng,
            num_links=num_links,
            sub_core_grid=movers,
            placement="simple" if grid.x - GX < 4 else "auto",
        )
        if self.rs_mode == "mrs":  # (the gather's semaphores above the arena too: the arena is freed every call)
            self.eng.alloc_arena()
        if self.rs_mode == "fused":  # the ports' mailboxes (a word per engine core, zero) and the arrival counters
            cores = sorted({(c.x, c.y) for ch in self.xg.chips.values() for c in ch["ports"].values()})
            crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*c), ttnn.CoreCoord(*c)) for c in cores])
            words = GX * GY * 4
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
            self.arr_sems = [ttnn.create_global_semaphore(mesh_device, self.eng.crs, 0) for _ in range(2)]
            ttnn.synchronize_device(mesh_device)
        self.rows, self.H, self.I, self.rs_dim, self.num_links = rows, H, I, rs_dim, num_links
        self.state = dict(key=None)

    def upload(self, w_gu, w_down):
        """w_gu / w_down per device (row-major mesh order). With rs_dim: (engine weights, W_down [I, H] bf8 a chip)."""
        if self.rs_mode != "mrs":
            pos = None
            if self.rs_mode == "fused":
                R, C = tuple(self.mesh.shape)
                pos = [self.xg.chips[(d // C, d % C)]["rings"][0]["p"] for d in range(R * C)]
            return self.eng.upload(w_gu, w_down, pos=pos)
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

    def run(self, x, w):
        if self.rs_mode == "fused":
            G = self.plan.G
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
                    rse = dict(
                        rs_pos=p, rs_port=lambda d, l, pc=pc: (pc[(d, l)].x, pc[(d, l)].y), rs_mbox=mbox, rs_arr=arr
                    )
                c0, k0 = self.xg.pieces(coord, prog, x, XREADY, rs=rsx)
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
                    **rse,
                )
                prog.cbs, prog.semaphores = c0 + c1, sems
                e.attach_mcast(prog)
                prog.kernels = k0 + k1
                r, c = coord
                md[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, c), ttnn.MeshCoordinate(r, c))] = prog
            self.state["key"], self.state["desc"] = key, md
        ttnn.generic_op([x, w, y, self.eng.arena], self.state["desc"])
        return y
