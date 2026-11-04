# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``MoePersistent``: the hybrid MoE block's row gather, the routed experts and the row combine as ONE program per
chip (the first step of the persistent-layer program, doc/perf/persistent_layer_design.md).

Three programs today (`custom_ops/row_exchange` gather 56 us, `custom_ops/expert_stream` fused 215 us, row_exchange
reduce 38 us on the 8x4 at 4 users per row) become one: the exchange core and its workers (10 cores the expert
placement leaves idle) run the gather phase, multicast a release into the expert cores' semaphores once the gathered
tiles are written, the 112 expert cores run the fused gate|up + down kernels unchanged (their readers / writers wait
for that release), the down cores count their written output tiles on the combine workers, and the same 10 cores
run the reduce phase. Data still passes through the gathered / expert-output tensors (DRAM or L1 interleaved) as
today; what disappears is two launches and their pipeline fills, and the step is the one the resident-activation
version builds on (the gathered rows multicast straight into the expert cores' L1).

Semaphores (program-local, re-initialised by the dispatch per launch and reset by their waiters):
``SEM_STAGED`` / ``SEM_GO`` on the exchange cores (both phases), ``PSEM_DONE`` on the combine workers (the down
cores' completion count), ``PSEM0..PSEM0+3`` on every core of the grid (the gather release: reader / writer of the
gate|up cores, reader / writer of the down cores). Arrival semaphores and receive stagings: the two RowExchange
instances' persistent ones.
"""

from __future__ import annotations

from pathlib import Path

import ttnn
from models.demos.glm53_flash.custom_ops.row_exchange.op import CT_MAX, TILE, TILE_BYTES, RowExchange

KERNEL_DIR = Path(__file__).resolve().parent / "kernels"


def _with_defines(k: ttnn.KernelDescriptor, extra: list[tuple[str, str]]) -> ttnn.KernelDescriptor:
    """``k`` with ``extra`` appended to its defines (the descriptor's ``defines`` field is writable)."""
    k.defines = list(k.defines) + [tuple(d) for d in extra]
    return k


def grid_mcast_rects(device, grid: tuple[int, int]) -> list[tuple[int, int, int, int, int]]:
    """The logical ``grid`` (all of ``compute_with_storage_grid_size``) as physically contiguous NOC rects
    ``(x0, y0, x1, y1, cores)``: consecutive NOC columns with the same NOC row span (a NOC multicast may cover Tensix
    cores only, and the Galaxy Blackhole chips' worker columns are NOC x 2-7 and 10-15)."""
    cols: dict[int, list[int]] = {}
    for x in range(grid[0]):
        for y in range(grid[1]):
            n = device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
            cols.setdefault(int(n.x), []).append(int(n.y))
    rects = []
    run: list[int] = []
    for nx in sorted(cols) + [None]:
        if run and (nx is None or nx != run[-1] + 1 or sorted(cols[nx]) != sorted(cols[run[0]])):
            ys = sorted(cols[run[0]])
            assert ys == list(range(ys[0], ys[-1] + 1)), ("NOC rows not contiguous", ys)
            rects.append((run[0], ys[0], run[-1], ys[-1], len(run) * len(ys)))
            run = []
        if nx is not None:
            run.append(nx)
    assert sum(r[4] for r in rects) == grid[0] * grid[1], (rects, grid)
    return rects


def persistent_fused_groups(device, E_loc: int, I_loc: int, H: int, *, down_readers=None, cap: int = 12) -> int:
    """The fused expert op's gate|up group count that leaves the persistent block its cores: the largest choice
    (today's cap of 12) whose placement keeps ``1 + the gather's chunks`` cores free on this device's grid
    (12x10 at E_loc = 72, H = 4096: 11 groups = 88 + 16 cores, 16 free; 12 would leave 8 for the 10 needed)."""
    from models.demos.glm53_flash.custom_ops.expert_stream.plan import fused_core_plan, fused_group_choices

    grid = device.compute_with_storage_grid_size()
    banks = device.dram_grid_size().x
    need = 1 + -(-H // (TILE * CT_MAX)) + -(-E_loc // (TILE * CT_MAX))
    for g in sorted(fused_group_choices(E_loc), reverse=True):
        if g > cap:
            continue
        try:
            plan = fused_core_plan((grid.x, grid.y), E_loc, I_loc, H, banks=banks, groups=g, down_readers=down_readers)
        except AssertionError:
            continue
        if grid.x * grid.y - plan["gate_up_cores"] - plan["down_cores"] >= need:
            return g
    raise ValueError(f"no fused group count leaves {need} cores on {grid.x}x{grid.y} (E_loc {E_loc}, H {H})")


class MoePersistent:
    """``experts``: the model's ``ExpertStreamFused`` (its placement decides the free cores); ``T`` users per row;
    ``H`` the hidden width, ``E`` the routing width (the full routing matrix: ``routing_fold``)."""

    SEM_STAGED, SEM_GO, PSEM_DONE = 0, 1, 2
    PSEM0 = 12  # ..15: gate|up reader / writer, down reader / writer
    CB_SEND_A, CB_SEND_B, CB_TIN, CB_RM, CB_RM2, CB_T, CB_OUT, CB_SEL, CB_ZERO = range(9)

    def __init__(self, mesh, experts, T: int, H: int, E: int, *, link: int = 0):
        self.mesh = mesh
        self.device = mesh.device
        self.experts = experts
        # the fused single-program op (gate|up + down kernel sets, semaphores, groups): the ExpertStream wrapper
        # exposes gate_up / down and keeps it as ``fused_op``
        self.fused = getattr(experts, "fused_op", None) or experts
        assert hasattr(self.fused, "down_xy"), "the persistent block needs the fused expert op (GLM53_STREAM_FUSED=1)"
        self.T, self.H, self.E = int(T), int(H), int(E)
        grid = self.device.compute_with_storage_grid_size()
        self.grid = (grid.x, grid.y)
        busy = set(experts.gate_up.core_set) | set(experts.down.core_set)
        free = [(x, y) for y in range(grid.y) for x in range(grid.x) if (x, y) not in busy]
        n_workers = -(-self.H // (TILE * CT_MAX)) + -(-self.E // (TILE * CT_MAX))  # the gather's chunks
        assert len(free) >= 1 + n_workers, (len(free), n_workers, "the expert placement leaves too few cores")
        self.xchg_core = free[0]
        self.workers = free[1 : 1 + n_workers]
        # the two exchanges' geometry, receive stagings and arrival semaphores (their programs are not used)
        self.gather = RowExchange(
            mesh, "gather", [self.H, self.E], self.T, xchg_core=self.xchg_core, link=link, workers=self.workers
        )
        self.reduce = RowExchange(
            mesh, "reduce", [self.H], self.T, xchg_core=self.xchg_core, link=link, workers=self.workers
        )
        assert self.gather.ps == self.reduce.ps and self.gather.cpp == self.reduce.cpp, "one packet geometry"
        assert self.gather.peers == self.reduce.peers and self.gather.groups == self.reduce.groups
        self.nw_a, self.nw_b = self.gather.nchunks, self.reduce.nchunks
        assert self.nw_b <= self.nw_a
        self.all_xchg = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(*self.xchg_core), ttnn.CoreCoord(*self.xchg_core))]
            + [ttnn.CoreRange(w, w) for w in self.gather.workers]
        )
        self.worker_set = ttnn.CoreRangeSet([ttnn.CoreRange(w, w) for w in self.gather.workers])
        self.xchg_set = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(*self.xchg_core), ttnn.CoreCoord(*self.xchg_core))]
        )
        self.full_grid = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))]
        )
        # the grid's multicast rects: a NOC multicast may target Tensix cores only and the worker grid spans two
        # NOC column ranges on Blackhole (x 2-7 and 10-15 on the Galaxy chips; 8-9 are not Tensix), so one rect over
        # the whole grid stalls (the 2026-11-04 tracy_l4_b4_persist_g11 hang: the release reached four cores)
        self.mcast_rects = grid_mcast_rects(self.device, (grid.x, grid.y))
        self.down_xy = [self._noc(c) for c, *_ in experts.down.assignments]

    def mcast_args(self, core: tuple[int, int]) -> list[int]:
        """The multicast rects as runtime args for a kernel on logical ``core``: ``[n, (x0, y0, x1, y1, num_dests,
        loopback) x n]``; the sender's own rect counts it (the loopback variant), the others do not."""
        n = self.device.worker_core_from_logical_core(ttnn.CoreCoord(*core))
        assert len(self.mcast_rects) <= 4, self.mcast_rects  # the kernels hold four
        out = [len(self.mcast_rects)]
        for x0, y0, x1, y1, cores in self.mcast_rects:
            inside = x0 <= int(n.x) <= x1 and y0 <= int(n.y) <= y1
            out += [x0, y0, x1, y1, cores, int(inside)]
        return out

    def _noc(self, core) -> int:
        n = self.device.worker_core_from_logical_core(core)
        return (int(n.x) << 16) | int(n.y)

    def _cbs(self) -> list:
        def cb(index, cores, total, page):
            return ttnn.CBDescriptor(
                total_size=total,
                core_ranges=cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.bfloat16, page_size=page)
                ],
            )

        g, r = self.gather, self.reduce
        block = CT_MAX * TILE_BYTES
        return [
            cb(self.CB_SEND_A, self.all_xchg, g.send_bytes, g.ps),
            cb(self.CB_SEND_B, self.all_xchg, r.send_bytes, r.ps),
            cb(self.CB_TIN, self.worker_set, block, TILE_BYTES),
            cb(self.CB_RM, self.worker_set, block, TILE_BYTES),
            cb(self.CB_RM2, self.worker_set, block, TILE_BYTES),
            cb(self.CB_T, self.worker_set, block, TILE_BYTES),
            cb(self.CB_OUT, self.worker_set, block, TILE_BYTES),
            cb(self.CB_SEL, self.worker_set, TILE_BYTES, TILE_BYTES),
            cb(self.CB_ZERO, self.worker_set, g.ps, g.ps),
        ]

    def _xchg_kernels(self, coord, xin, xout, yin, yout) -> list:
        """The exchange cores' kernels for device ``coord``: ``xin`` = (x, routing), ``xout`` = (x_g, routing_g),
        ``yin`` = the expert output, ``yout`` = the combine output."""
        g, r = self.gather, self.reduce
        rr, c = coord
        me = rr
        T, nsrc = self.T, g.nsrc
        xchg_noc = self._noc(ttnn.CoreCoord(*self.xchg_core))
        in_a_ct = ttnn.TensorAccessorArgs(xin[0]).get_compile_time_args()
        out_a_ct = ttnn.TensorAccessorArgs(xout[0]).get_compile_time_args()
        for t in xin[1:]:
            assert (
                ttnn.TensorAccessorArgs(t).get_compile_time_args() == in_a_ct
            ), "the gather inputs share one placement"
        for t in xout[1:]:
            assert (
                ttnn.TensorAccessorArgs(t).get_compile_time_args() == out_a_ct
            ), "the gather outputs share one placement"
        in_b_ct = ttnn.TensorAccessorArgs(yin).get_compile_time_args()
        out_b_ct = ttnn.TensorAccessorArgs(yout).get_compile_time_args()
        recv_a_ct = ttnn.TensorAccessorArgs(g.recv).get_compile_time_args()
        recv_b_ct = ttnn.TensorAccessorArgs(r.recv).get_compile_time_args()
        kernels = []
        groups: dict[tuple[int, int, int], list[int]] = {}
        for k, (s, c0, ct) in enumerate(g.chunks):
            has_b = int(k < r.nchunks)
            ct_b = r.chunks[k][2] if has_b else 1
            groups.setdefault((ct, has_b, ct_b), []).append(k)
        for (ct_a, has_b, ct_b), ks in groups.items():
            cores = ttnn.CoreRangeSet([ttnn.CoreRange(g.workers[k], g.workers[k]) for k in ks])
            reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
            for k in ks:
                s, c0, _ = g.chunks[k]
                core = g.workers[k]
                sb, cb0 = (r.chunks[k][0], r.chunks[k][1]) if has_b else (0, 0)
                reader_rt[core.x][core.y] = [xin[s].buffer_address(), c0, yin.buffer_address(), cb0]
                writer_rt[core.x][core.y] = [
                    xout[s].buffer_address(),
                    c0,
                    k,
                    me,
                    xchg_noc,
                    g.recv.buffer_address(),
                    self._noc(core),
                    yout.buffer_address(),
                    cb0,
                    k,
                    r.recv.buffer_address(),
                ]
                compute_rt[core.x][core.y] = []
            kernels += [
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "reader_moe_xchg.cpp"),
                    core_ranges=cores,
                    compile_time_args=[
                        self.CB_TIN,
                        self.CB_SEL,
                        self.CB_ZERO,
                        ct_a,
                        ct_b,
                        T,
                        nsrc,
                        g.ps,
                        has_b,
                        self.PSEM_DONE,
                        len(self.down_xy),
                    ]
                    + in_a_ct
                    + in_b_ct,
                    runtime_args=reader_rt,
                    config=ttnn.ReaderConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "writer_moe_xchg.cpp"),
                    core_ranges=cores,
                    compile_time_args=[
                        self.CB_SEND_A,
                        self.CB_SEND_B,
                        self.CB_RM,
                        self.CB_RM2,
                        self.CB_T,
                        self.CB_OUT,
                        self.CB_ZERO,
                        ct_a,
                        ct_b,
                        T,
                        nsrc,
                        g.nchunks,
                        r.nchunks,
                        g.ps,
                        g.cpp,
                        g.np,
                        r.np,
                        self.SEM_GO,
                        self.SEM_STAGED,
                        has_b,
                    ]
                    + out_a_ct
                    + recv_a_ct
                    + out_b_ct
                    + recv_b_ct,
                    runtime_args=writer_rt,
                    config=ttnn.WriterConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "compute_moe_xchg.cpp"),
                    core_ranges=cores,
                    compile_time_args=[
                        self.CB_TIN,
                        self.CB_RM,
                        self.CB_RM2,
                        self.CB_T,
                        self.CB_SEL,
                        self.CB_OUT,
                        ct_a,
                        ct_b,
                        has_b,
                    ],
                    runtime_args=compute_rt,
                    config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True),
                ),
            ]
        # the exchange core
        peers = g.peers[coord]
        args = [me, xchg_noc, g.recv.buffer_address(), g.arrived_addr, r.recv.buffer_address(), r.arrived_addr]
        args += [self._noc(w) for w in g.workers]
        args += [len(peers)]
        for row2, dev, mesh_id, grp in peers:
            args += [row2, dev, mesh_id, grp]
        args += [len(g.groups[coord])] + self.mcast_args(self.xchg_core)
        xchg_rt = ttnn.RuntimeArgs()
        xchg_rt[self.xchg_core[0]][self.xchg_core[1]] = args
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "xchg_moe.cpp"),
                core_ranges=self.xchg_set,
                compile_time_args=[
                    self.CB_SEND_A,
                    self.CB_SEND_B,
                    self.SEM_STAGED,
                    self.SEM_GO,
                    self.nw_a,
                    self.nw_b,
                    g.nchunks,
                    r.nchunks,
                    g.ps,
                    g.cpp,
                    g.np,
                    r.np,
                    nsrc,
                    self.PSEM0,
                ]
                + recv_a_ct
                + recv_b_ct,
                runtime_args=xchg_rt,
                config=ttnn.WriterConfigDescriptor(),
            )
        )
        return kernels

    def _expert_kernels(self, xg, wg, w13, w2, act, out, base) -> tuple[list, list, list]:
        """(cbs, kernels, semaphores) of the fused expert program with the persistent hooks defined."""
        ex = self.fused
        gu_cbs, gu_kernels = ex.gate_up.descriptors(
            xg,
            wg,
            w13,
            act,
            base=base,
            cb=(0, 1, 2, 3, 4, 5, 16),
            fused_rt=[len(ex.down_xy), ex.SEM_BASE, ex.sems_per_group] + ex.down_xy,
        )
        dn_cbs, dn_kernels = ex.down.descriptors(
            act,
            wg,
            w2,
            out,
            base=base,
            cb=(8, 9, 10, 11, 17),
            fused_rt=[ex.groups, ex.SEM_BASE, ex.sems_per_group, ex.banks],
        )
        assert self.nw_b == 8, (self.nw_b, "the down writer signals eight combine workers")
        xys = [(f"PERSIST_XY{i}", str(self._noc(self.reduce.workers[i]))) for i in range(self.nw_b)]
        # the kernel lists are [reader, writer, compute] (``_BankStreamOp._kernels``)
        _with_defines(gu_kernels[0], [("PERSIST_WAIT_SEM", str(self.PSEM0))])
        _with_defines(gu_kernels[1], [("PERSIST_WAIT_SEM", str(self.PSEM0 + 1))])
        _with_defines(dn_kernels[0], [("PERSIST_WAIT_SEM", str(self.PSEM0 + 2))])
        _with_defines(
            dn_kernels[1],
            [("PERSIST_WAIT_SEM", str(self.PSEM0 + 3)), ("PERSIST_SIGNAL_SEM", str(self.PSEM_DONE))] + xys,
        )
        sems = [
            ttnn.SemaphoreDescriptor(ex.SEM_BASE + i, ttnn.CoreType.WORKER, ex.down.cores, 0) for i in range(ex.n_sems)
        ]
        return gu_cbs + dn_cbs, gu_kernels + dn_kernels, sems

    def __call__(self, x, routing, w13, w2, act, out, base, xg, wg, comb_out):
        """One program per chip: ``x`` [1,1,T,H] / ``routing`` [1,1,T,E] (this row's users), the expert weights
        (``w13`` / ``w2``, or the regions source), ``act`` / ``out`` the fused expert op's buffers (``alloc``),
        ``base`` the chip's routing-column page, ``xg`` / ``wg`` the gathered tiles (allocated by the caller in the
        expert op's input placement), ``comb_out`` [1,1,T,H] the combined output."""
        g = self.gather
        for t, w in ((x, self.H), (routing, self.E)):
            assert tuple(t.padded_shape)[-2:] == (TILE, w) and t.dtype == ttnn.bfloat16, (
                tuple(t.padded_shape),
                w,
                t.dtype,
            )
        assert tuple(out.padded_shape)[-2:] == (TILE, self.H) and tuple(comb_out.padded_shape)[-2:] == (TILE, self.H)
        mpd = ttnn.MeshProgramDescriptor()
        ex_cbs, ex_kernels, ex_sems = self._expert_kernels(xg, wg, w13, w2, act, out, base)
        for rr in range(g.rows):
            for c in range(g.cols):
                coord = (rr, c)
                kernels = self._xchg_kernels(coord, (x, routing), (xg, wg), out, comb_out) + ex_kernels
                sems = (
                    [
                        ttnn.SemaphoreDescriptor(self.SEM_STAGED, ttnn.CoreType.WORKER, self.all_xchg, 0),
                        ttnn.SemaphoreDescriptor(self.SEM_GO, ttnn.CoreType.WORKER, self.all_xchg, 0),
                        ttnn.SemaphoreDescriptor(self.PSEM_DONE, ttnn.CoreType.WORKER, self.all_xchg, 0),
                    ]
                    + [
                        ttnn.SemaphoreDescriptor(self.PSEM0 + i, ttnn.CoreType.WORKER, self.full_grid, 0)
                        for i in range(4)
                    ]
                    + ex_sems
                )
                program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=self._cbs() + ex_cbs)
                src = self.device.get_fabric_node_id(ttnn.MeshCoordinate(rr, c))
                conn = []
                for rep in g.groups[coord]:
                    conn += list(
                        ttnn.setup_fabric_connection(
                            src, self.device.get_fabric_node_id(rep), g.link, program, ttnn.CoreCoord(*self.xchg_core)
                        )
                    )
                xi = [i for i, k in enumerate(program.kernels) if str(k.kernel_source).endswith("xchg_moe.cpp")]
                assert len(xi) == 1
                xk = program.kernels[xi[0]]
                xk.runtime_args[self.xchg_core[0]][self.xchg_core[1]] = (
                    list(xk.runtime_args[self.xchg_core[0]][self.xchg_core[1]]) + conn
                )
                mpd[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(rr, c), ttnn.MeshCoordinate(rr, c))] = program
        weights = w13.tensors() if hasattr(w13, "tensors") else [w13, w2]
        io = (
            weights
            + [x, routing, xg, wg, act, out, comb_out, g.recv, self.reduce.recv]
            + ([base] if base is not None else [])
        )
        ttnn.generic_op(io, mpd)
        return comb_out
