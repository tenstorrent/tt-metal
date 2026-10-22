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
