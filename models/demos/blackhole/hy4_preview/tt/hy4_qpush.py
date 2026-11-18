# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""B29 (flag HY4_B29_QPUSH=1, default OFF): q all-gather replaced by a push-only program + a flag wait inside the split SDPA reader.

``QPush(md, S)(q_rm)``: q_rm [1,8,S,576] ROW_MAJOR bf16 (this column's 8 heads) -> every chip's persistent landing tensor (DRAM, pages of RP q rows (2: 2304 B), ceil(8S/RP) pages per source column) via fabric
packets with a fused increment of the per-source flag word (global semaphore) on the flag core. The program returns as soon as its own packets are out (no wait for the peers); the split reader
(hy4_split.sparse_sdpa_split(qland=...)) waits for flag[owner] after it has issued its first K chunk gather, so the 10 us K gather overlaps the wait for the slowest peer.
Ordering: the next layer's push of a chip starts after that chip finished the previous layer's exchange (program order), which needs the partials of every peer's split (read q first): the landing is never overwritten early.
Flags are never reset (monotonic): each split core keeps a launch counter in a second global semaphore (4S packets per source per launch).
"""
import os

import torch

import ttnn

KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hy4_qpush_kernels")
RING, NC = 8, 8
MC = (
    os.environ.get("HY4_B47B_PUSH_MC", "0") == "1"
)  # B47b: chip-multicast write + one multicast flag increment per lane instead of unicast packets per peer


def _acc(t):
    return list(ttnn.TensorAccessorArgs(t).get_compile_time_args())


def _cores(lst):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for (x, y) in lst])


class Push:
    """push-gather of ``nr`` pages of ``pb`` bytes per source column over the 8-chip ring axis (see module doc); ``rp`` pages per fabric packet (rp * pb <= 4352)"""

    def __init__(self, md, nr, pb, rp, links=2, lanes=None, mc=None):
        self.mc = MC if mc is None else bool(mc)
        assert tuple(md.shape)[1] == RING and rp * pb <= 4352
        self.md, self.nr, self.pb, self.links = md, nr, pb, links
        self.lanes = lanes or int(os.environ.get("HY4_B29_QLANES", str(links)))
        self.rp = rp
        self.np = -(-nr // self.rp)  # packets (landing pages) per source
        if os.environ.get("HY4_PUSH_LANES_CLAMP", "1") == "1":
            self.lanes = max(
                1, min(self.lanes, self.np)
            )  # B38: a lane without a packet must not open/close a fabric connection (see RESULT_B38)
        self.nh = min(8, int(os.environ.get("HY4_B29_NH", "8")))
        grid = md.compute_with_storage_grid_size()
        self.grid = grid
        self.comm = {(d, l): (self.lanes * d + l, grid.y - 1) for d in (0, 1) for l in range(self.lanes)}
        self.fcore = (grid.x - 1, grid.y - 1)
        assert self.lanes * 2 <= grid.x - 1
        all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
        self.flags = [ttnn.create_global_semaphore(md, all_cores, 0) for _ in range(NC)]
        self.cnt = ttnn.create_global_semaphore(md, all_cores, 0)  # per split core launch counter (local use only)
        self.flag_addrs = [ttnn.get_global_semaphore_address(g) for g in self.flags]
        self.cnt_addr = ttnn.get_global_semaphore_address(self.cnt)
        # landing tensor [NC * 4S pages of 2304 B]: page c*4S + p holds source column c's q rows 2p, 2p+1 (row r = h*S + s)
        self.land = ttnn.from_torch(
            torch.zeros(1, 1, NC * self.np, self.rp * self.pb // 2, dtype=torch.bfloat16),
            device=md,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(md),
        )
        ttnn.synchronize_device(md)

    def _phys(self, x, y):
        c = self.md.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
        return c.x, c.y

    def split_args(self):
        """runtime args of the split reader cores (after tok, g): flag core phys x y, 8 flag addresses, counter address"""
        fx, fy = self._phys(*self.fcore)
        return (
            [fx, fy] + list(self.flag_addrs) + [self.cnt_addr]
        )  # (packets per source / rows per packet go in as compile-time args: see sparse_sdpa_split)

    def comm_phys(self):
        """physical (x, y) of the comm cores in the order fwd lanes then bwd lanes (B47b fold: the writers of the q-prep program signal these)"""
        return [self._phys(*self.comm[(d, l)]) for d in (0, 1) for l in range(self.lanes)]

    def fold_mesh(self, kernels, sems, cbs, q_out, extra_tensors, tag):
        """B47b (HY4_B47B_QFOLD): MeshProgramDescriptor = the given single-chip program parts (the fused q-prep) + this push's comm cores. The comm cores wait for the
        q-prep writers (local semaphore 0, one increment per writer core) and then read ``q_out`` [1,8,S,576] RM from DRAM and push it exactly like ``__call__``.
        Returns (mpd, tensors) for ttnn.generic_op."""
        md = self.md
        PKT = self.rp * self.pb
        NP = self.np
        gper = max(len(range(ln, NP, self.lanes)) for ln in range(self.lanes))
        fx, fy = self._phys(*self.fcore)
        comm_set = _cores(list(self.comm.values()))
        CBI = 20
        cb = ttnn.CBDescriptor(
            total_size=PKT * gper,
            core_ranges=comm_set,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=CBI, data_format=ttnn.bfloat16, page_size=PKT)],
        )
        sem = ttnn.SemaphoreDescriptor(0, ttnn.CoreType.WORKER, comm_set, 0)
        rows, cols = tuple(md.shape)
        mpd = ttnn.MeshProgramDescriptor()
        for r in range(rows):
            for col in range(cols):
                coord = ttnn.MeshCoordinate(r, col)
                src = md.get_fabric_node_id(coord)
                fwd_n = md.get_fabric_node_id(ttnn.MeshCoordinate(r, (col + 1) % RING))
                bwd_n = md.get_fabric_node_id(ttnn.MeshCoordinate(r, (col - 1) % RING))
                ck, kidx = [], {}
                for name, ndist, dr in (("fwd", 4, 0), ("bwd", 3, 1)):
                    for lk in range(self.lanes):
                        cx, cy = self.comm[(dr, lk)]
                        rt = ttnn.RuntimeArgs()
                        rt[cx][cy] = [col, fx, fy] + list(self.flag_addrs)
                        kd = ttnn.KernelDescriptor(
                            kernel_source=f"{KDIR}/qp_comm.cpp",
                            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                            core_ranges=_cores([(cx, cy)]),
                            compile_time_args=[
                                dr,
                                ndist,
                                self.nr,
                                lk,
                                self.lanes,
                                CBI,
                                3 + NC,
                                self.nh,
                                self.rp,
                                self.pb,
                            ]
                            + _acc(q_out)
                            + _acc(self.land),
                            runtime_args=rt,
                            common_runtime_args=[q_out.buffer_address(), self.land.buffer_address()],
                            config=ttnn.WriterConfigDescriptor(),
                        )
                        kd.defines = [("QF_WAIT", str(self.fold_nsig))] + ([("QP_MC", "1")] if self.mc else [])
                        ck.append(kd)
                        kidx[(name, lk)] = len(kernels) + len(ck) - 1
                prog = ttnn.ProgramDescriptor(
                    kernels=list(kernels) + ck, semaphores=list(sems) + [sem], cbs=list(cbs) + [cb]
                )
                for (name, lk), ki in kidx.items():
                    dst = fwd_n if name == "fwd" else bwd_n
                    cx, cy = self.comm[(0 if name == "fwd" else 1, lk)]
                    fargs = ttnn.setup_routing_plane_connection(
                        src, [dst], [lk % self.links], prog, ki, ttnn.CoreCoord(cx, cy)
                    )
                    prog.kernels[ki].runtime_args[cx][cy].extend(fargs)
                prog.custom_program_hash = (0x7BA << 40) | (
                    hash(
                        (tag, self.nr, self.pb, self.links, self.lanes, self.nh, self.rp, col, self.fold_nsig, self.mc)
                    )
                    & ((1 << 40) - 1)
                )
                mpd[ttnn.MeshCoordinateRange(coord, coord)] = prog
        return mpd, extra_tensors + [self.land]

    fold_nsig = 0

    def __call__(self, q_rm):
        md = self.md
        PKT = self.rp * self.pb
        NP = self.np
        gper = max(len(range(ln, NP, self.lanes)) for ln in range(self.lanes))
        fx, fy = self._phys(*self.fcore)
        mpd = ttnn.MeshProgramDescriptor()
        rows, cols = tuple(md.shape)
        for r in range(rows):
            for col in range(cols):
                coord = ttnn.MeshCoordinate(r, col)
                src = md.get_fabric_node_id(coord)
                fwd_n = md.get_fabric_node_id(ttnn.MeshCoordinate(r, (col + 1) % RING))
                bwd_n = md.get_fabric_node_id(ttnn.MeshCoordinate(r, (col - 1) % RING))
                comm_set = _cores(list(self.comm.values()))
                cbs = [
                    ttnn.CBDescriptor(
                        total_size=PKT * gper,
                        core_ranges=comm_set,
                        format_descriptors=[
                            ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=PKT)
                        ],
                    )
                ]
                kernels, kidx = [], {}
                for name, ndist, dr in (("fwd", 4, 0), ("bwd", 3, 1)):
                    for lk in range(self.lanes):
                        cx, cy = self.comm[(dr, lk)]
                        rt = ttnn.RuntimeArgs()
                        rt[cx][cy] = [col, fx, fy] + list(self.flag_addrs)
                        kernels.append(
                            ttnn.KernelDescriptor(
                                kernel_source=f"{KDIR}/qp_comm.cpp",
                                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                                core_ranges=_cores([(cx, cy)]),
                                compile_time_args=[
                                    dr,
                                    ndist,
                                    self.nr,
                                    lk,
                                    self.lanes,
                                    0,
                                    3 + NC,
                                    self.nh,
                                    self.rp,
                                    self.pb,
                                ]
                                + _acc(q_rm)
                                + _acc(self.land),
                                runtime_args=rt,
                                common_runtime_args=[q_rm.buffer_address(), self.land.buffer_address()],
                                config=ttnn.WriterConfigDescriptor(),
                            )
                        )
                        if self.mc:
                            kernels[-1].defines = [("QP_MC", "1")]
                        kidx[(name, lk)] = len(kernels) - 1
                prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
                for (name, lk), ki in kidx.items():
                    dst = fwd_n if name == "fwd" else bwd_n
                    cx, cy = self.comm[(0 if name == "fwd" else 1, lk)]
                    fargs = ttnn.setup_routing_plane_connection(
                        src, [dst], [lk % self.links], prog, ki, ttnn.CoreCoord(cx, cy)
                    )
                    prog.kernels[ki].runtime_args[cx][cy].extend(fargs)
                prog.custom_program_hash = (0x7B9 << 40) | (
                    hash(
                        (
                            self.nr,
                            self.pb,
                            self.links,
                            self.lanes,
                            self.nh,
                            self.rp,
                            col,
                            self.mc,
                            tuple(_acc(q_rm)),
                            tuple(_acc(self.land)),
                        )
                    )
                    & ((1 << 40) - 1)
                )
                mpd[ttnn.MeshCoordinateRange(coord, coord)] = prog
        ttnn.generic_op([q_rm, self.land], mpd)
        return self.land


class QPush(Push):
    """q push of the split SDPA: q_rm [1,8,S,576] RM bf16 (page r = h*S + s, 1152 B), 3 rows per packet"""

    def __init__(self, md, S, links=2, lanes=None, mc=None):
        self.S = S
        super().__init__(md, 8 * S, 1152, int(os.environ.get("HY4_B29_QRP", "3")), links, lanes, mc)
