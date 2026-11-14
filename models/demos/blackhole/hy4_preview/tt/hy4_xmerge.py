# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fused partial exchange + merge of the Hy4 decode sparse attention (task fuse-sparse-merge, stage A2; flag HY4_FUSE=sparse,xchg): ONE mesh program replaces ``to_owner + all_to_all_async_generic + merge``.

Input x = the split kernel's owner-layout TILE output [8 owners][S users][32 head rows][576] of THIS chip (``emit_stats=3``): for owner j / user s 18 tiles: o (16, normalised by the local sum), ms (scaled max), l (local sum).
Per chip: 2 x 2 communication cores (ring direction x link) read the tiles of the owners at distance 1..4 / 1..3 from DRAM and send them as 4 KiB fabric packets (fused global-semaphore increment) into the landing
buffer (persistent L1) of the owner's merge cores; the merge cores (S x nsplit) read their own slot from DRAM, wait for the 7 remote slots, and run the merge compute (mrg2_compute) -> [1,S,32,512] TILE.
Persistent L1 landing buffer + global semaphore: consecutive calls are ordered by the q all-gather of the next call (no barrier here); a benchmark loop must contain it.
"""
import os

import torch

import ttnn

KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hy4_merge_kernels")
RING, NC = 8, 8


def _acc(t):
    return list(ttnn.TensorAccessorArgs(t).get_compile_time_args())


def _cores(lst):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for (x, y) in lst])


class XMerge:
    def __init__(self, md, S, nsplit=None, links=2, compact=None, lanes=None):
        nsplit = nsplit or int(os.environ.get("HY4_XM_NSPLIT", "4" if int(os.environ.get("HY4_XM_COMPACT", "0")) else "2"))
        compact = int(os.environ.get("HY4_XM_COMPACT", "0")) if compact is None else int(compact)  # 1 = compact 8x32 tiles: WORK IN PROGRESS (S=1 PCC 0.71, S=2 hang, see RESULT.md), default OFF
        assert tuple(md.shape)[1] == RING and 16 % nsplit == 0 and (16 // nsplit) % 2 == 0
        self.md, self.S, self.nsplit, self.links, self.compact = md, S, nsplit, links, compact
        self.tb = 512 if compact else 2048
        # communication lanes per ring direction (one comm core each); lane L -> fabric link L % links
        self.lanes = lanes or int(os.environ.get("HY4_XM_LANES", str(links)))
        self.tpc = 16 // nsplit
        self.lt = self.tpc + 2
        grid = md.compute_with_storage_grid_size()
        self.grid = grid
        n_m = S * nsplit
        self.mcores = [(i % grid.x, i // grid.x) for i in range(n_m)]
        self.comm = {(d, l): (self.lanes * d + l, grid.y - 1) for d in (0, 1) for l in range(self.lanes)}
        all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
        self.gsem = ttnn.create_global_semaphore(md, all_cores, 0)
        mset = _cores(self.mcores)
        self.mset = mset
        w = NC * self.lt * self.tb // 2  # bf16 elements per merge core: 8 slots x (tpc + 2) tiles
        mc = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(mset, [1, w], ttnn.ShardOrientation.ROW_MAJOR),
        )
        self.land = ttnn.from_torch(
            torch.zeros(n_m, w, dtype=torch.bfloat16),
            device=md,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=mc,
            mesh_mapper=ttnn.ReplicateTensorToMesh(md),
        )
        self.out = ttnn.from_torch(
            torch.zeros(1, S, 32, 512, dtype=torch.bfloat16),
            device=md,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(md),
        )  # zero-initialised: the compact writer only fills rows 0..7 of every tile
        ttnn.synchronize_device(md)

    def _phys(self, x, y):
        c = self.md.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
        return c.x, c.y

    def __call__(self, x, sink):
        md, S, ns, tpc = self.md, self.S, self.nsplit, self.tpc
        LT = self.lt
        bf = ttnn.bfloat16
        T = 2048
        ST, A, W, R, ONE, OUT, PKT = range(7)
        npk_g = 1 if self.compact else (tpc // 2 + 1)  # packets per (owner, user, merge core)
        gper = max(len(range(ln, ns, self.lanes)) for ln in range(self.lanes))
        pk_max = S * 4 * gper * npk_g
        gsem_addr = ttnn.get_global_semaphore_address(self.gsem)
        dev_land = ttnn.get_device_tensors(self.land)
        m_xy = [self._phys(*c) for c in self.mcores]
        mpd = ttnn.MeshProgramDescriptor()
        rows, cols = tuple(md.shape)
        for r in range(rows):
            for col in range(cols):
                coord = ttnn.MeshCoordinate(r, col)
                src = md.get_fabric_node_id(coord)
                fwd_n = md.get_fabric_node_id(ttnn.MeshCoordinate(r, (col + 1) % RING))
                bwd_n = md.get_fabric_node_id(ttnn.MeshCoordinate(r, (col - 1) % RING))
                idx = r * cols + col
                cb_land = ttnn.cb_descriptor_from_sharded_tensor(A, dev_land[idx])
                cb_land.core_ranges = self.mset
                tdesc = ttnn.TileDescriptor(8, 32) if self.compact else None
                cb_land.total_size = NC * LT * self.tb
                cb_land.format_descriptors = [
                    ttnn.CBFormatDescriptor(buffer_index=A, data_format=bf, page_size=self.tb, **({"tile": tdesc} if tdesc else {}))
                ]

                def cb(i, n_, cores, page=None, tiny=True):
                    page = page or self.tb
                    kw = {"tile": tdesc} if (tdesc and tiny) else {}
                    return ttnn.CBDescriptor(
                        total_size=page * n_,
                        core_ranges=cores,
                        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=bf, page_size=page, **kw)],
                    )

                comm_set = _cores(list(self.comm.values()))
                cbs = [
                    cb_land,
                    cb(ST, 1, self.mset),
                    cb(W, NC, self.mset),
                    cb(R, 1, self.mset),
                    cb(ONE, 1, self.mset),
                    cb(OUT, tpc, self.mset),
                    cb(PKT, pk_max, comm_set, page=2 * T, tiny=False),
                ]
                rt_r, rt_w, rt_c = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
                for m, (cx, cy) in enumerate(self.mcores):
                    s, g = m // ns, m % ns
                    rt_r[cx][cy] = [s, g, col]
                    rt_w[cx][cy] = [s, g * tpc]
                    rt_c[cx][cy] = []
                kernels = [
                    ttnn.KernelDescriptor(
                        kernel_source=f"{KDIR}/{'xm_reader' if self.compact else 'xm_reader_v1'}.cpp",
                        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                        core_ranges=self.mset,
                        compile_time_args=[S, tpc, ST, A, ONE, min(1, int(os.environ.get("HY4_XM_DBG", "0")))] + ([self.compact] if self.compact else []) + _acc(x) + _acc(sink),
                        runtime_args=rt_r,
                        common_runtime_args=[x.buffer_address(), sink.buffer_address(), gsem_addr],
                        config=ttnn.ReaderConfigDescriptor(),
                    ),
                    ttnn.KernelDescriptor(
                        kernel_source=f"{KDIR}/{'xm_writer' if self.compact else 'mrg_writer'}.cpp",
                        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                        core_ranges=self.mset,
                        compile_time_args=[tpc, OUT] + ([self.compact] if self.compact else []) + _acc(self.out),
                        runtime_args=rt_w,
                        common_runtime_args=[self.out.buffer_address()],
                        config=ttnn.WriterConfigDescriptor(),
                    ),
                    ttnn.KernelDescriptor(
                        kernel_source=f"{KDIR}/mrg2_compute.cpp",
                        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                        core_ranges=self.mset,
                        compile_time_args=[tpc, ST, A, W, R, ONE, OUT],
                        runtime_args=rt_c,
                        common_runtime_args=[],
                        config=ttnn.ComputeConfigDescriptor(
                            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True
                        ),
                    ),
                ]
                comm_kidx = {}
                for name, ndist, dr in (("fwd", 4, 0), ("bwd", 3, 1)):
                    for lk in range(self.lanes):
                        cx, cy = self.comm[(dr, lk)]
                        rt = ttnn.RuntimeArgs()
                        rt[cx][cy] = [col] + [v for xy in m_xy for v in xy]
                        fab_base = 1 + 2 * len(m_xy)
                        kernels.append(
                            ttnn.KernelDescriptor(
                                kernel_source=f"{KDIR}/{'xm_comm' if self.compact else 'xm_comm_v1'}.cpp",
                                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                                core_ranges=_cores([(cx, cy)]),
                                compile_time_args=[dr, ndist, S, tpc, ns, lk, self.lanes, PKT, fab_base, int(os.environ.get("HY4_XM_DBG", "0"))] + ([self.compact] if self.compact else []) + _acc(x),
                                runtime_args=rt,
                                common_runtime_args=[x.buffer_address(), self.land.buffer_address(), gsem_addr],
                                config=ttnn.WriterConfigDescriptor(),
                            )
                        )
                        comm_kidx[(name, lk)] = len(kernels) - 1
                prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
                for (name, lk), ki in comm_kidx.items():
                    dst = fwd_n if name == "fwd" else bwd_n
                    cx, cy = self.comm[(0 if name == "fwd" else 1, lk)]
                    fargs = ttnn.setup_routing_plane_connection(src, [dst], [lk % self.links], prog, ki, ttnn.CoreCoord(cx, cy))
                    prog.kernels[ki].runtime_args[cx][cy].extend(fargs)
                prog.custom_program_hash = (0x7A2 << 40) | (
                    hash((S, ns, self.links, self.lanes, self.compact, col, os.environ.get("HY4_XM_DBG", "0"), tuple(_acc(x)), tuple(_acc(sink)), tuple(_acc(self.out)))) & ((1 << 40) - 1)
                )
                mpd[ttnn.MeshCoordinateRange(coord, coord)] = prog
        ttnn.generic_op([x, sink, self.out, self.land], mpd)
        return self.out


class XMergeFused(XMerge):
    """A2b: split kernel + partial exchange + merge in ONE mesh program (flag HY4_FUSE=sparse,xchg2). The leader cores of the split kernel write their (o, ms, l) runs straight into the comm cores' persistent
    L1 staging (remote owners) or into the own landing slot (owner == this chip); the comm cores open their fabric connection at launch, wait for the runs (local semaphore) and send; no DRAM round trip, no
    separate program. Full 32x32 tiles, nsplit = lanes = links = 2 (the validated xchg config)."""

    SEMX = 15

    def __init__(self, md, S, nsplit=2, links=2):
        super().__init__(md, S, nsplit=nsplit, links=links, compact=0, lanes=links)
        grid = self.grid
        # merge cores below the split cores (8 S G <= 64 cores), comm cores on the last row
        y0 = int(os.environ.get("HY4_XM_Y0", "7"))
        self.mcores = [(i % grid.x, y0 + i // grid.x) for i in range(S * nsplit)]
        self.comm = {(d, l): (links * d + l + int(os.environ.get("HY4_XM_CX", "0")), int(os.environ.get("HY4_XM_CY", str(grid.y - 1)))) for d in (0, 1) for l in range(links)}
        # rebuild the landing tensor on the new merge cores
        self.mset = _cores(self.mcores)
        w = NC * self.lt * self.tb // 2
        mc = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(self.mset, [1, w], ttnn.ShardOrientation.ROW_MAJOR),
        )
        self.land = ttnn.from_torch(
            torch.zeros(len(self.mcores), w, dtype=torch.bfloat16),
            device=md, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, memory_config=mc,
            mesh_mapper=ttnn.ReplicateTensorToMesh(md),
        )
        self.gper = len(range(0, nsplit, links))
        self.pk_run = self.tpc // 2 + 1
        stage_pkts = S * 4 * self.gper * self.pk_run
        cset = _cores(list(self.comm.values()))
        sw = stage_pkts * 4096 // 2
        smc = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
            ttnn.ShardSpec(cset, [1, sw], ttnn.ShardOrientation.ROW_MAJOR),
        )
        self.stage = ttnn.from_torch(
            torch.zeros(len(self.comm), sw, dtype=torch.bfloat16),
            device=md, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, memory_config=smc,
            mesh_mapper=ttnn.ReplicateTensorToMesh(md),
        )
        ttnn.synchronize_device(md)

    def __call__(self, qg, kv, idx8, sink, mod, kc, G, ckc, scale):
        md, S, ns, tpc, LT = self.md, self.S, self.nsplit, self.tpc, self.lt
        T, PK = 2048, 4096
        ST, A, W, R, ONE, OUT = 40, 41, 42, 43, 44, 45
        gsem_addr = ttnn.get_global_semaphore_address(self.gsem)
        land_base, stage_base = self.land.buffer_address(), self.stage.buffer_address()
        dev_land = ttnn.get_device_tensors(self.land)
        m_xy = [self._phys(*c) for c in self.mcores]
        comm_xy = {k: self._phys(*v) for k, v in self.comm.items()}
        rows, cols = tuple(md.shape)
        mpd = ttnn.MeshProgramDescriptor()
        out_split = None
        for r in range(rows):
            for col in range(cols):
                coord = ttnn.MeshCoordinate(r, col)
                src = md.get_fabric_node_id(coord)
                fwd_n = md.get_fabric_node_id(ttnn.MeshCoordinate(r, (col + 1) % RING))
                bwd_n = md.get_fabric_node_id(ttnn.MeshCoordinate(r, (col - 1) % RING))
                idx = r * cols + col

                def xchg(t, g, col=col):
                    if g != 0:
                        return [0] * 10
                    j, s = t // S, t % S
                    ent = []
                    if j == col:
                        flag = 2
                        for gg in range(ns):
                            mx, my = m_xy[s * ns + gg]
                            ent += [mx, my, land_base + (col * LT) * T]
                    else:
                        d = (j - col) % RING
                        dr, dd = (0, d) if d <= 4 else (1, RING - d)
                        nd = 4 if dr == 0 else 3
                        di = nd - dd
                        flag = 1
                        for gg in range(ns):
                            lane, gidx = gg % self.links, gg // self.links
                            cx, cy = comm_xy[(dr, lane)]
                            k_run = (s * nd + di) * self.gper + gidx
                            ent += [cx, cy, stage_base + k_run * self.pk_run * PK]
                    return [flag, tpc, ns, PK, gsem_addr, self.SEMX] + ent

                parts = mod.sparse_sdpa_split(qg, kv, idx8, 512, scale, kc, G, ckc, None, emit_stats=3, owner_s=S, xchg=xchg, build_only=True, out=out_split)
                out_split = parts["out"]
                kernels = list(parts["kernels"])
                cbs = list(parts["cbs"])
                tdesc = None

                def cb(i, n_, cores, page=T):
                    return ttnn.CBDescriptor(
                        total_size=page * n_, core_ranges=cores,
                        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=page)],
                    )

                cb_land = ttnn.cb_descriptor_from_sharded_tensor(A, dev_land[idx])
                cb_land.core_ranges = self.mset
                cb_land.total_size = NC * LT * T
                cb_land.format_descriptors = [ttnn.CBFormatDescriptor(buffer_index=A, data_format=ttnn.bfloat16, page_size=T)]
                cbs += [cb_land, cb(ST, 1, self.mset), cb(W, NC, self.mset), cb(R, 1, self.mset), cb(ONE, 1, self.mset), cb(OUT, tpc, self.mset)]
                rt_r, rt_w, rt_c = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
                for m, (cx, cy) in enumerate(self.mcores):
                    s, g = m // ns, m % ns
                    rt_r[cx][cy] = [s, g, col]
                    rt_w[cx][cy] = [s, g * tpc]
                    rt_c[cx][cy] = []
                if os.environ.get("HY4_XM_NOMERGE") == "1":
                    cbs = [c_ for c_ in cbs if c_ is not cb_land]
                kernels += [] if os.environ.get("HY4_XM_NOMERGE") == "1" else [
                    ttnn.KernelDescriptor(
                        kernel_source=f"{KDIR}/xm_reader_v2.cpp", source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                        core_ranges=self.mset, compile_time_args=[S, tpc, ST, A, ONE, int(os.environ.get("HY4_XM_DBG", "0"))] + _acc(sink),
                        runtime_args=rt_r, common_runtime_args=[0, sink.buffer_address(), gsem_addr],
                        config=ttnn.ReaderConfigDescriptor()),
                    ttnn.KernelDescriptor(
                        kernel_source=f"{KDIR}/mrg_writer.cpp", source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                        core_ranges=self.mset, compile_time_args=[tpc, OUT] + _acc(self.out),
                        runtime_args=rt_w, common_runtime_args=[self.out.buffer_address()],
                        config=ttnn.WriterConfigDescriptor()),
                    ttnn.KernelDescriptor(
                        kernel_source=f"{KDIR}/mrg2_compute.cpp", source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                        core_ranges=self.mset, compile_time_args=[tpc, ST, A, W, R, ONE, OUT],
                        runtime_args=rt_c, common_runtime_args=[],
                        config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True)),
                ]
                comm_kidx = {}
                for name, ndist, dr in (("fwd", 4, 0), ("bwd", 3, 1)):
                    for lk in range(self.links):
                        if os.environ.get("HY4_XM_NOCOMM") == "1":
                            continue
                        cx, cy = self.comm[(dr, lk)]
                        rt = ttnn.RuntimeArgs()
                        rt[cx][cy] = [col] + [v for xy in m_xy for v in xy]
                        kernels.append(ttnn.KernelDescriptor(
                            kernel_source=f"{KDIR}/xm_comm_v2.cpp", source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                            core_ranges=_cores([(cx, cy)]),
                            compile_time_args=[dr, ndist, S, tpc, ns, lk, self.links, self.SEMX, 1 + 2 * len(m_xy), int(os.environ.get("HY4_XM_DBG", "0"))],
                            runtime_args=rt, common_runtime_args=[stage_base, land_base, gsem_addr],
                            config=ttnn.WriterConfigDescriptor()))
                        comm_kidx[(name, lk)] = len(kernels) - 1
                sem_cores = _cores(list(parts["cores"]) + list(self.comm.values()))
                sems = list(parts["semaphores"]) + [ttnn.SemaphoreDescriptor(self.SEMX, ttnn.CoreType.WORKER, sem_cores, 0)]
                prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
                for (name, lk), ki in comm_kidx.items():
                    dst = fwd_n if name == "fwd" else bwd_n
                    cx, cy = self.comm[(0 if name == "fwd" else 1, lk)]
                    fargs = ttnn.setup_routing_plane_connection(src, [dst], [lk], prog, ki, ttnn.CoreCoord(cx, cy))
                    prog.kernels[ki].runtime_args[cx][cy].extend(fargs)
                prog.custom_program_hash = (0x7B3 << 40) | (hash((parts["hash_parts"], os.environ.get("HY4_XM_DBG", "0"), os.environ.get("HY4_XM_NOCOMM", "0"), os.environ.get("HY4_XM_NOMERGE", "0"), ns, self.links, col, tuple(_acc(sink)), tuple(_acc(self.out)))) & ((1 << 40) - 1))
                mpd[ttnn.MeshCoordinateRange(coord, coord)] = prog
        ttnn.generic_op([qg, kv, idx8, out_split, self.out, self.land, self.stage, sink], mpd)
        return self.out
