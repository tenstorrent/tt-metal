# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""B47a (flag HY4_B47A_XPUSH=1, default OFF): the cross-column partial exchange of the compact S>1 verify path (S <= 4 users, TR = 1) runs INSIDE the split SDPA program, replacing ``all_to_all_async_generic`` (the A1 merge
program stays, its reader waits on per-source flags: hy4_xpush_kernels/mrg_reader_xp.cpp).

``XPush(md, S).split(qg, kv, idx8, mod, kc, G, ckc, scale)`` = the owner-layout compact split kernel (leaders write their bands to the DRAM output as before, then bump a global semaphore on the comm cores of
owner j) + 4 comm cores (ring direction x lane, row grid.y-1: kernel xp_comm.cpp) that send the finished owner tile rows (9 packets of 4 KiB per owner) to the landing tensor of chip j with a fused flag increment.
The program returns when its own sends are out: it does not wait for any peer. ``XPush.land`` is then the ``x`` input of ``hy4_merge.merge_fused(land, sink, TR=1, xp=self, hm_users=S)``.
Ordering: the next call's push starts after the next layer's q all-gather, which every chip issues only after its merge of this call: the landing is never overwritten early (as B29 QPush). Flags are never reset.
"""
import os

import torch

import ttnn

KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hy4_xpush_kernels")
RING, NC = 8, 8
NPK = 9


def _acc(t):
    return list(ttnn.TensorAccessorArgs(t).get_compile_time_args())


def _cores(lst):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for (x, y) in lst])


class XPush:
    reader_path = f"{KDIR}/mrg_reader_xp.cpp"

    def __init__(self, md, S, links=2, lanes=2):
        assert tuple(md.shape)[1] == RING and 1 <= S <= 4
        self.md, self.S, self.links, self.lanes = md, S, links, lanes
        self.nh = min(8, int(os.environ.get("HY4_B47A_NH", "8")))
        self.dbg = int(os.environ.get("HY4_B47A_DBG", "0"))
        grid = md.compute_with_storage_grid_size()
        self.grid = grid
        self.comm = {(d, l): (lanes * d + l, grid.y - 1) for d in (0, 1) for l in range(lanes)}
        self.fcore = (grid.x - 1, grid.y - 1)
        all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
        gsem = lambda: ttnn.create_global_semaphore(md, all_cores, 0)
        self.flags = [gsem() for _ in range(NC)]  # per source column (consumer side, monotonic)
        self.gs = [gsem() for _ in range(NC)]  # per owner (comm side, leaders count S, comm core resets)
        self.cnt = gsem()  # per consumer core launch counter (local use)
        self.flag_addrs = [ttnn.get_global_semaphore_address(g) for g in self.flags]
        self.gs_addrs = [ttnn.get_global_semaphore_address(g) for g in self.gs]
        self.cnt_addr = ttnn.get_global_semaphore_address(self.cnt)
        # landing tensor: page c*9 + p (4 KiB = 2 tiles) of source column c
        self.land = ttnn.from_torch(
            torch.zeros(1, 1, NC * NPK, 2048, dtype=torch.bfloat16),
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

    def consumer_args(self):
        fx, fy = self._phys(*self.fcore)
        return [fx, fy, self.cnt_addr] + list(self.flag_addrs)

    def split(self, qg, kv, idx8, mod, kc, G, ckc, scale, qland=None):
        md, S = self.md, self.S
        rows, cols = tuple(md.shape)
        fx, fy = self._phys(*self.fcore)
        comm_xy = {k: self._phys(*v) for k, v in self.comm.items()}
        mpd = ttnn.MeshProgramDescriptor()
        out_split = None
        PKTCB = 46
        for r in range(rows):
            for col in range(cols):
                coord = ttnn.MeshCoordinate(r, col)
                src = md.get_fabric_node_id(coord)
                fwd_n = md.get_fabric_node_id(ttnn.MeshCoordinate(r, (col + 1) % RING))
                bwd_n = md.get_fabric_node_id(ttnn.MeshCoordinate(r, (col - 1) % RING))

                def hook(t, g, col=col):
                    if g != 0:
                        return [0] * 10
                    j = t // S
                    d = (j - col) % RING
                    dr = 0 if d <= 4 else 1
                    ent = []
                    for ln in range(self.lanes):
                        ent += list(comm_xy[(dr, ln)])
                    return [3, self.lanes, self.gs_addrs[j]] + ent + [0] * (7 - len(ent))

                hook.xpush = True
                parts = mod.sparse_sdpa_split(
                    qg,
                    kv,
                    idx8,
                    512,
                    scale,
                    kc,
                    G,
                    ckc,
                    None,
                    emit_stats=3,
                    owner_s=S,
                    compact=True,
                    xchg=hook,
                    build_only=True,
                    out=out_split,
                    **({"qland": qland} if qland is not None else {}),
                )
                out_split = parts["out"]
                kernels = list(parts["kernels"])
                cbs = list(parts["cbs"])
                used = set(parts["cores"])
                assert not (used & set(self.comm.values())), "split cores overlap the comm cores"
                cbs.append(
                    ttnn.CBDescriptor(
                        total_size=self.nh * 4096,
                        core_ranges=_cores(list(self.comm.values())),
                        format_descriptors=[
                            ttnn.CBFormatDescriptor(buffer_index=PKTCB, data_format=ttnn.bfloat16, page_size=4096)
                        ],
                    )
                )
                kidx = {}
                argbase = 3 + NC + NC
                for name, ndist, dr in (("fwd", 4, 0), ("bwd", 3, 1)):
                    for ln in range(self.lanes):
                        cx, cy = self.comm[(dr, ln)]
                        rt = ttnn.RuntimeArgs()
                        rt[cx][cy] = [col, fx, fy] + list(self.flag_addrs) + list(self.gs_addrs)
                        kernels.append(
                            ttnn.KernelDescriptor(
                                kernel_source=f"{KDIR}/xp_comm.cpp",
                                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                                core_ranges=_cores([(cx, cy)]),
                                compile_time_args=[dr, ndist, ln, self.lanes, PKTCB, argbase, self.nh, S, self.dbg]
                                + _acc(out_split)
                                + _acc(self.land),
                                runtime_args=rt,
                                common_runtime_args=[out_split.buffer_address(), self.land.buffer_address()],
                                config=ttnn.WriterConfigDescriptor(),
                            )
                        )
                        kidx[(name, ln)] = len(kernels) - 1
                prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=list(parts["semaphores"]), cbs=cbs)
                for (name, ln), ki in kidx.items():
                    dst = fwd_n if name == "fwd" else bwd_n
                    cx, cy = self.comm[(0 if name == "fwd" else 1, ln)]
                    fargs = ttnn.setup_routing_plane_connection(
                        src, [dst], [ln % self.links], prog, ki, ttnn.CoreCoord(cx, cy)
                    )
                    prog.kernels[ki].runtime_args[cx][cy].extend(fargs)
                prog.custom_program_hash = (0x7C4 << 40) | (
                    hash(
                        (
                            parts["hash_parts"],
                            S,
                            qland[0].rp if qland is not None else 0,
                            self.lanes,
                            self.links,
                            self.nh,
                            self.dbg,
                            col,
                            tuple(_acc(out_split)),
                            tuple(_acc(self.land)),
                        )
                    )
                    & ((1 << 40) - 1)
                )
                mpd[ttnn.MeshCoordinateRange(coord, coord)] = prog
        ttnn.generic_op([qg, kv, idx8, out_split, self.land], mpd)
        return out_split
