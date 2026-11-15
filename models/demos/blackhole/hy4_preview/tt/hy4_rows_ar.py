# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""One-shot rows-phase reduce of the TP32 MoE tail (replaces reduce_scatter(cols) ... no: replaces RS rows + AG rows + select; the column phase is Hy4FuseAR).

A group of 4 chips on a line (mesh axis 0 of the 4x8 mesh; or, for single-slot tests on a 1x8 mesh, chips 4g..4g+3 of the row) holds 4 partials [Tt = 4T, 6144] of the same tokens
(chip position r holds the partial of its own experts shard).  Chip position r OWNS the token block r (rows r*T..r*T+T-1): every chip sends the T-row block of the owner for all 6144
columns to each of the 3 other chips (one fabric packet per (owner, reducer core), NRP reducer cores per chip, each reducer owns Q = 192 / NRP tile columns), the owner sums the 4
contributions on the Tensix (fp32 dest) and writes the T rows into rows 0..T-1 of an output [32, 6144] TILE DRAM tensor = the input of ``Hy4FuseAR`` (valid rows VR = T).
Wire format = tiny tile of H rows (face0 H rows | face1 H rows, 64 H bytes).  Cross-chip buffers are persistent L1 + a global semaphore; consecutive calls must be ordered by some
all-to-all dependency of the model (the all-gather of x of the next layer) or by a host synchronize in tests.  Flag-gated by callers, JIT only.
"""
import os

import torch

import ttnn

KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rows_ar_kernels")
CDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fuse_ar_kernels")
D = 6144
NTILE = D // 32  # 192 tile columns
NB = 4


def _cores(lst):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in lst])


class Hy4RowsReduce:
    def __init__(self, md, T=1, h=2, nrp=16, links=2, mode=None, stop=0, red=None, comm=None):
        """mode 'axis0': mesh (4, C), groups = columns, line along axis 0.  mode 'row4': mesh (1, 8), groups = chips 4g..4g+3 of the row (single-slot test).
        h = rows of the wire/compute tiny tile (>= T, power of 2: 2, 4, 8)."""
        self.md, self.T, self.h, self.nrp, self.links, self.stop = md, T, h, nrp, links, stop
        rows, cols = tuple(md.shape)
        self.mode = mode or ("axis0" if rows == 4 else "row4")
        assert self.mode in ("axis0", "row4")
        assert 4 * T <= 8 and h >= T and h in (2, 4, 8), (T, h)
        assert NTILE % nrp == 0
        self.Q = NTILE // nrp
        self.tileb = 64 * h
        self.chunk = self.Q * self.tileb
        assert self.chunk <= 4096, "one chunk = one fabric packet <= 4 KB"
        assert nrp % links == 0
        self.per = nrp // links
        self.red_all = red or [(4 + i % 8, i // 8) for i in range(nrp)]
        cy = (nrp + 7) // 8
        self.comm = comm or {(d, l): (4 + 2 * d + l, cy) for d in (0, 1) for l in range(links)}
        grid = md.compute_with_storage_grid_size()
        self.all_cores = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))]
        )
        self.gsems = [
            ttnn.create_global_semaphore(md, self.all_cores, 0) for _ in range(2)
        ]  # ping-pong: call n and n+2 share a set, n+1 uses the other (a chip that finished call n+1 implies every peer finished call n)
        self._par = 0
        rep = ttnn.ReplicateTensorToMesh(md)
        mc = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(_cores(self.red_all), [1, (NB - 1) * self.chunk // 2], ttnn.ShardOrientation.ROW_MAJOR),
        )
        self.recvs = [
            ttnn.from_torch(
                torch.zeros(nrp, (NB - 1) * self.chunk // 2, dtype=torch.bfloat16),
                device=md,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.bfloat16,
                memory_config=mc,
                mesh_mapper=rep,
            )
            for _ in range(2)
        ]
        self.out = ttnn.from_torch(
            torch.zeros(1, 1, 32, D, dtype=torch.bfloat16),
            device=md,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rep,
        )
        ttnn.synchronize_device(md)

    def _phys(self, x, y):
        c = self.md.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
        return c.x, c.y

    def _pos(self, row, col):
        """(position in group, fabric coordinate of the forward neighbour or None, backward neighbour or None)"""
        if self.mode == "axis0":
            r = row
            f = ttnn.MeshCoordinate(row + 1, col) if r < 3 else None
            b = ttnn.MeshCoordinate(row - 1, col) if r > 0 else None
        else:
            r = col % 4
            f = ttnn.MeshCoordinate(row, col + 1) if r < 3 else None
            b = ttnn.MeshCoordinate(row, col - 1) if r > 0 else None
        return r, f, b

    def _build(self, part, par=0):
        md = self.md
        self.recv, self.gsem = self.recvs[par], self.gsems[par]
        rows, cols = tuple(md.shape)
        T, h, Q, nrp, tileb, chunk = self.T, self.h, self.Q, self.nrp, self.tileb, self.chunk
        red = self.red_all
        red_set = _cores(red)
        comm_set = _cores(list(self.comm.values()))
        stg_set = _cores(red + list(self.comm.values()))
        red_xy = [self._phys(*c) for c in red]
        comm_xy = {k: self._phys(*v) for k, v in self.comm.items()}
        dev_recv = ttnn.get_device_tensors(self.recv)
        gsem_addr = ttnn.get_global_semaphore_address(self.gsem)
        recv_addr = self.recv.buffer_address()
        in_addr = part.buffer_address()
        out_addr = self.out.buffer_address()
        acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        CB_RECV, CB_OUT, CB_STG, CB_PAY, CB_OWN, CB_SCR = 8, 9, 10, 11, 12, 13
        link_p = [list(range(l * self.per, (l + 1) * self.per)) for l in range(self.links)]
        link_of = {p: l for l, ps in enumerate(link_p) for p in ps}

        def tdesc():
            return ttnn.TileDescriptor(h, 32)

        def tcb(i, pages):
            return ttnn.CBDescriptor(
                total_size=pages * tileb,
                core_ranges=red_set,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=tileb, tile=tdesc())
                ],
            )

        mpd = ttnn.MeshProgramDescriptor()
        for row in range(rows):
            for col in range(cols):
                coord = ttnn.MeshCoordinate(row, col)
                src = md.get_fabric_node_id(coord)
                r, fn, bn = self._pos(row, col)
                nd = {0: 3 - r, 1: r}
                idx = row * cols + col
                cb_recv = ttnn.cb_descriptor_from_sharded_tensor(CB_RECV, dev_recv[idx])
                cb_recv.core_ranges = red_set
                cb_recv.total_size = (NB - 1) * Q * tileb
                cb_recv.format_descriptors = [
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_RECV, data_format=ttnn.bfloat16, page_size=tileb, tile=tdesc()
                    )
                ]
                cb_stg = ttnn.CBDescriptor(
                    total_size=nrp * NB * chunk,
                    core_ranges=stg_set,
                    format_descriptors=[
                        ttnn.CBFormatDescriptor(buffer_index=CB_STG, data_format=ttnn.bfloat16, page_size=chunk)
                    ],
                )
                cb_scr = ttnn.CBDescriptor(
                    total_size=2 * Q * 256,
                    core_ranges=red_set,
                    format_descriptors=[
                        ttnn.CBFormatDescriptor(buffer_index=CB_SCR, data_format=ttnn.bfloat16, page_size=256)
                    ],
                )
                cbs = [cb_recv, tcb(CB_OUT, Q), cb_stg, tcb(CB_PAY, NB * Q), tcb(CB_OWN, Q), cb_scr]
                sems = [ttnn.SemaphoreDescriptor(s, ttnn.CoreType.WORKER, comm_set, 0) for s in range(1)]
                kernels = []
                rt_rd, rt_cp, rt_wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
                for p, (x, y) in enumerate(red):
                    fx, fy = comm_xy[(0, link_of[p])]
                    bx, by = comm_xy[(1, link_of[p])]
                    rt_rd[x][y] = [r, p, nd[0], nd[1], fx, fy, bx, by, p - link_p[link_of[p]][0]]
                    rt_cp[x][y] = []
                    rt_wr[x][y] = [p, (r * T) - ((r * T) & ~1)]
                fp32 = True
                kernels.append(
                    ttnn.KernelDescriptor(
                        kernel_source=f"{KDIR}/rp_reader.cpp",
                        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                        core_ranges=red_set,
                        compile_time_args=[CB_RECV, Q, CB_OWN, CB_PAY, tileb, T, h, CB_STG, CB_SCR] + acc(part),
                        runtime_args=rt_rd,
                        common_runtime_args=[in_addr, gsem_addr],
                        config=ttnn.ReaderConfigDescriptor(),
                    )
                )
                kernels.append(
                    ttnn.KernelDescriptor(
                        kernel_source=(f"{CDIR}/ar_red_compute.cpp" if self.Q <= 6 else f"{KDIR}/rp_compute.cpp"),
                        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                        core_ranges=red_set,
                        compile_time_args=(
                            [CB_RECV, CB_OUT, Q, NB, 1, CB_OWN]
                            if self.Q <= 6
                            else [
                                CB_RECV,
                                CB_OUT,
                                Q,
                                NB,
                                CB_OWN,
                                int(os.environ.get("HY4_RP_SUB", "6")),
                                int(os.environ.get("HY4_RP_MODE", "0")),
                            ]
                        ),
                        runtime_args=rt_cp,
                        config=ttnn.ComputeConfigDescriptor(
                            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=fp32
                        ),
                    )
                )
                kernels.append(
                    ttnn.KernelDescriptor(
                        kernel_source=f"{KDIR}/rp_writer.cpp",
                        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                        core_ranges=red_set,
                        compile_time_args=[CB_OUT, Q, T, h, tileb] + acc(self.out),
                        runtime_args=rt_wr,
                        common_runtime_args=[out_addr],
                        config=ttnn.WriterConfigDescriptor(),
                    )
                )
                comm_kidx = {}
                for dr, neigh in ((0, fn), (1, bn)):
                    if nd[dr] == 0:
                        continue
                    for lk in range(self.links):
                        x, y = self.comm[(dr, lk)]
                        rt = ttnn.RuntimeArgs()
                        rt[x][y] = [r] + [v for xy in red_xy for v in xy]
                        kernels.append(
                            ttnn.KernelDescriptor(
                                kernel_source=f"{KDIR}/rp_comm.cpp",
                                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                                core_ranges=_cores([(x, y)]),
                                compile_time_args=[
                                    dr,
                                    nd[dr],
                                    self.per,
                                    link_p[lk][0],
                                    CB_STG,
                                    chunk,
                                    1 + 2 * nrp,
                                    self.stop,
                                ],
                                runtime_args=rt,
                                common_runtime_args=[recv_addr, gsem_addr],
                                config=ttnn.WriterConfigDescriptor(),
                            )
                        )
                        comm_kidx[(dr, lk)] = (len(kernels) - 1, neigh)
                prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
                for (dr, lk), (ki, neigh) in comm_kidx.items():
                    x, y = self.comm[(dr, lk)]
                    dst = md.get_fabric_node_id(neigh)
                    fargs = ttnn.setup_routing_plane_connection(src, [dst], [lk], prog, ki, ttnn.CoreCoord(x, y))
                    prog.kernels[ki].runtime_args[x][y].extend(fargs)
                mpd[ttnn.MeshCoordinateRange(coord, coord)] = prog
        return mpd

    def __call__(self, part):
        """part: [1,1,32,6144] bf16 TILE DRAM (rows 0..4T-1 valid). Returns self.out: rows 0..T-1 = this chip's token block reduced over the group of 4 chips (other rows 0)."""
        par, self._par = self._par, 1 - self._par
        ttnn.generic_op([part, self.recvs[par], self.out], self._build(part, par))
        return self.out
