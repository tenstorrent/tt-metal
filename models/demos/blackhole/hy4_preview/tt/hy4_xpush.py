# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""B47b XPUSH (flag HY4_B47B_XPUSH=1, default OFF): the rows all-gather of the TP32 MoE inputs (x tokens + route words) replaced by a push program + a flag wait inside the tp24 kernels.

``XPush(md, T, mode)(x_tokens, route)``: x_tokens [T,1,1,6144] RM bf16 (this chip's T tokens), route [1,1,1,16 T] int32 RM (its route words) -> every chip of the 4-chip group line (mesh axis 0; mode 'row4': chips 4g..4g+3 of
a 1x8 row, for single-slot tests) receives them in persistent DRAM landing tensors:
  xland  [1,1,4 NPX,2048] bf16 (pages of 4096 B, NPX = 3 T pages per source position): source region = 8 bank blocks of BLK = 1536 T B, block b = [tile col j (24)][face f (2)][row r (T)][16 cols]: exactly the
         tiny-tile pieces (T rows x 32 B) the tp24 writer of bank core b reads for its K slice, no tilize op;
  rland  [1,1,1,16 Tt] int32 RM, Tt = 4 T: the route words of the 4 chips in group order (= the route all-gather output).
8 worker cores (one per bank) re-lay the rows into the packet buffers of the comm cores (NoC), 2 lanes x 2 directions of comm cores send them with ONE chip-multicast packet per 4 KB and one multicast
flag increment per lane (flag[source position] on a flag core). The program returns as soon as its packets are out. The consumer (tp24 reader + writer, define XPUSH) polls flag[i] >= launch counter x
(NPX + 1) of the 4 source positions before reading. Flags are never reset (monotonic); each consumer RISC keeps its launch counter in a global-semaphore word of its core.
Ordering: the next push of a chip starts after that chip finished the rows reduce of this layer, which needs every group peer's tp24 partial, which they produce after reading their landing: never overwritten early.
T in {2, 4} (4096 % (32 T) == 0, 64 B aligned pieces).
"""
import os

import torch

import ttnn

KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hy4_xpush_kernels")
HID = 6144


def _acc(t):
    return list(ttnn.TensorAccessorArgs(t).get_compile_time_args())


def _cores(lst):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for (x, y) in lst])


class XPush:
    def __init__(self, md, T, mode=None, lanes=2, links=2):
        assert T in (2, 4)
        self.md, self.T, self.lanes, self.links = md, T, lanes, links
        rows, cols = tuple(md.shape)
        self.mode = mode or ("axis0" if rows == 4 else "row4")
        self.Tt = 4 * T
        self.blk = 1536 * T
        self.srcb = 12288 * T
        self.npx = self.srcb // 4096  # 3 T
        self.npt = self.npx + 1  # packets (flag increments) per source per launch
        grid = md.compute_with_storage_grid_size()
        self.grid = grid
        self.nh = min(8, int(os.environ.get("HY4_B47B_XNH", "8")))
        y = grid.y - 1
        self.comm = {(d, l): (lanes * d + l, y) for d in (0, 1) for l in range(lanes)}
        self.workers = [
            (8 + b % 4, grid.y - 2 - b // 4) for b in range(8)
        ]  # unused rows of the top: any free core works (own program)
        self.fcore = (grid.x - 1, grid.y - 1)
        assert lanes * 2 <= grid.x - 1
        allc = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
        self.flags = [ttnn.create_global_semaphore(md, allc, 0) for _ in range(4)]
        self.cnt = [
            ttnn.create_global_semaphore(md, allc, 0) for _ in range(2)
        ]  # consumer launch counters (reader, writer RISC)
        self.flag_addrs = [ttnn.get_global_semaphore_address(g) for g in self.flags]
        self.cnt_addrs = [ttnn.get_global_semaphore_address(g) for g in self.cnt]
        rep = ttnn.ReplicateTensorToMesh(md)
        self.xland = ttnn.from_torch(
            torch.zeros(1, 1, 4 * self.npx, 2048, dtype=torch.bfloat16),
            device=md,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rep,
        )
        self.rland = ttnn.from_torch(
            torch.zeros(1, 1, 1, 16 * self.Tt, dtype=torch.int32),
            device=md,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.int32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rep,
        )
        ttnn.synchronize_device(md)

    def _phys(self, x, y):
        c = self.md.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
        return c.x, c.y

    def consumer_args(self, which):
        """runtime args appended to the tp24 reader (which = 0) / writer (which = 1) cores: flag core x y, 4 flag addresses, launch counter address"""
        fx, fy = self._phys(*self.fcore)
        return [fx, fy] + list(self.flag_addrs) + [self.cnt_addrs[which]]

    def consumer_defines(self):
        return [("XPUSH", "1"), ("XP_T", str(self.T)), ("XP_NPX", str(self.npx)), ("XP_NPT", str(self.npt))]

    def _pos(self, row, col):
        if self.mode == "axis0":
            r = row
            f = ttnn.MeshCoordinate(row + 1, col) if r < 3 else None
            b = ttnn.MeshCoordinate(row - 1, col) if r > 0 else None
        else:
            r = col % 4
            f = ttnn.MeshCoordinate(row, col + 1) if r < 3 else None
            b = ttnn.MeshCoordinate(row, col - 1) if r > 0 else None
        return r, f, b

    def __call__(self, x_tokens, route):
        md, T = self.md, self.T
        assert tuple(x_tokens.shape) == (T, 1, 1, HID) and x_tokens.layout == ttnn.ROW_MAJOR_LAYOUT
        fx, fy = self._phys(*self.fcore)
        comm_xy = [self._phys(*self.comm[(d, l)]) for d in (0, 1) for l in range(self.lanes)]
        comm_set = _cores(list(self.comm.values()))
        wk_set = _cores(self.workers)
        both = _cores(list(self.comm.values()) + self.workers)
        CB_SCR, CB_BUF = 0, 1
        cbs = [
            ttnn.CBDescriptor(
                total_size=64 + 24 * T * 64 + self.blk,
                core_ranges=wk_set,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_SCR, data_format=ttnn.bfloat16, page_size=64 + 24 * T * 64 + self.blk
                    )
                ],
            ),
            ttnn.CBDescriptor(
                total_size=self.srcb + 64 * T,
                core_ranges=both,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_BUF, data_format=ttnn.bfloat16, page_size=self.srcb + 64 * T
                    )
                ],
            ),
        ]
        sem = ttnn.SemaphoreDescriptor(0, ttnn.CoreType.WORKER, both, 0)
        rows, cols = tuple(md.shape)
        mpd = ttnn.MeshProgramDescriptor()
        for r in range(rows):
            for col in range(cols):
                coord = ttnn.MeshCoordinate(r, col)
                src = md.get_fabric_node_id(coord)
                pos, fn, bn = self._pos(r, col)
                nd = {0: 3 - pos, 1: pos}
                nbr = {0: fn, 1: bn}
                own_dir = 0 if nd[0] > 0 else 1
                kernels, kidx = [], {}
                # workers
                rt_w = ttnn.RuntimeArgs()
                for b, (wx, wy) in enumerate(self.workers):
                    rt_w[wx][wy] = [b] + [v for xy in comm_xy for v in xy]
                kernels.append(
                    ttnn.KernelDescriptor(
                        kernel_source=f"{KDIR}/xp_worker.cpp",
                        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                        core_ranges=wk_set,
                        compile_time_args=[CB_SCR, CB_BUF, T, len(comm_xy)] + _acc(x_tokens),
                        runtime_args=rt_w,
                        common_runtime_args=[x_tokens.buffer_address()],
                        config=ttnn.ReaderConfigDescriptor(),
                    )
                )
                for dr in (0, 1):
                    if nd[dr] == 0 and dr != own_dir:
                        continue
                    for lk in range(self.lanes):
                        cx, cy = self.comm[(dr, lk)]
                        rt = ttnn.RuntimeArgs()
                        rt[cx][cy] = [pos, fx, fy] + list(self.flag_addrs)
                        kernels.append(
                            ttnn.KernelDescriptor(
                                kernel_source=f"{KDIR}/xp_comm.cpp",
                                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                                core_ranges=_cores([(cx, cy)]),
                                compile_time_args=[
                                    dr,
                                    nd[dr],
                                    lk,
                                    self.lanes,
                                    CB_BUF,
                                    self.npx,
                                    3 + 4,
                                    self.nh,
                                    int(dr == own_dir),
                                    T,
                                ]
                                + _acc(self.xland)
                                + _acc(self.rland)
                                + _acc(route),
                                runtime_args=rt,
                                common_runtime_args=[
                                    self.xland.buffer_address(),
                                    self.rland.buffer_address(),
                                    route.buffer_address(),
                                ],
                                config=ttnn.WriterConfigDescriptor(),
                            )
                        )
                        if nd[dr] > 0:
                            kidx[(dr, lk)] = len(kernels) - 1
                prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[sem], cbs=cbs)
                for (dr, lk), ki in kidx.items():
                    cx, cy = self.comm[(dr, lk)]
                    fargs = ttnn.setup_routing_plane_connection(
                        src, [md.get_fabric_node_id(nbr[dr])], [lk % self.links], prog, ki, ttnn.CoreCoord(cx, cy)
                    )
                    prog.kernels[ki].runtime_args[cx][cy].extend(fargs)
                prog.custom_program_hash = (0x7BB << 40) | (
                    hash((T, self.lanes, self.links, self.nh, pos, own_dir, tuple(_acc(x_tokens)), tuple(_acc(route))))
                    & ((1 << 40) - 1)
                )
                mpd[ttnn.MeshCoordinateRange(coord, coord)] = prog
        ttnn.generic_op([x_tokens, route, self.xland, self.rland], mpd)
        return self.xland, self.rland
