# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hy4-Preview one-shot all-reduce over the 8 chips of a mesh row (B10 persist, boundary (b): attention o-path / MoE reduce -> next iHC).

Port of the DSV4.1 ``fuse_tail_ar.FuseTailAR`` (all-reduce half; the kernels are copies in ``fuse_ar_kernels/``) to the Hy4 hidden size D = 6144:
a chip's partial [32, 6144] bf16 TILE (rows 0..VR-1 valid) is cut into 8 slices of NT = 24 tiles (768 columns); every chip sends the valid rows of its 7 foreign
slices to the owner chip (compact 8x32 tiles, 512 B, one fabric packet per (owner, reducer)), the owner's NP = 4 reducer cores (Q = 6 tiles each) sum the 8 contributions on
the Tensix (fp32 dest), multicast the reduced chunks to the 7 other chips, and the chunks land on NM = 8 * NPL = 32 landing cores (6 compact tiles each = the 32-core
column-tile ownership of the Hy4 iHC kernel: core k owns column tiles [6k, 6k+6)).  Replaces reduce_scatter + all_gather (41-44 us measured, lat/RESULT.md).
All cross-chip buffers are persistent L1 tensors and all cross-chip semaphores are global semaphores (no barrier: the all-to-all dependency of consecutive calls orders
buffer reuse).  Flag-gated by the callers (default OFF), JIT only (no C++ build).
"""
import os

import torch

import ttnn

KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fuse_ar_kernels")

RING = 8
D = 6144
NT = D // 32 // RING  # 24 tiles per slice (768 columns)
NP = 4  # reducer cores per chip
Q = NT // NP  # 6 tiles per reducer chunk
NPL = 4  # landing cores per source chip
QL = NT // NPL  # 6 tiles per landing core
NM = RING * NPL  # 32 landing cores
NRECV = RING - 1
GRID_W = 13


def tile_bytes(T):
    assert 1 <= T <= 16, T
    return 512 if T <= 8 else 1024


def _cores(lst):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in lst])


def _rect(x0, y0, x1, y1):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))])


def _rowmajor_set(cores):
    n = len(cores)
    full = n // GRID_W
    ranges = []
    if full:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(GRID_W - 1, full - 1)))
    if n % GRID_W:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full), ttnn.CoreCoord(n % GRID_W - 1, full)))
    return ttnn.CoreRangeSet(ranges)


class Hy4FuseAR:
    def __init__(self, md, tileb=512, links=2, stop=0, red=None, comm=None, ihc_row0=None, nred=NP, recv_small=False):
        """ihc_row0 = R0 of the Hy4 iHC 4x8 rectangle: the landing cores are then the 32 iHC cores in k order (core k = cy * 4 + cx owns column tiles [6k, 6k + 6)); reducers / comm cores default to
        row 8 / row 9 (outside the rectangle rows 0..7). Without it: landing cores = the first 32 cores row-major, reducers (p, 7), comm cores (x, 8).
        """
        self.md = md
        rows, cols = tuple(md.shape)
        assert cols == RING
        self.rows, self.stop, self.links, self.tileb = rows, stop, links, tileb
        assert NT % nred == 0 and nred % NPL == 0, nred
        self.NP, self.Q = (
            nred,
            NT // nred,
        )  # reducer cores per chip and tiles per reducer chunk (NP = 4: Q 6, one reducer per landing core; NP = 8: Q 3, two reducers per landing core)
        global GRID_W
        GRID_W = int(md.compute_with_storage_grid_size().x)
        if ihc_row0 is not None:
            r0 = ihc_row0
            self.land = [(k % 4, r0 + k // 4) for k in range(NM)]
            self.land_set = _rect(0, r0, 3, r0 + 7)
            self.red_all = red or [(p, 8) for p in range(self.NP)]
            self.comm = comm or {
                (d, l): (2 * (0 if d == "fwd" else 1) + l, 9) for d in ("fwd", "bwd") for l in range(links)
            }
        else:
            self.land = [(m % GRID_W, m // GRID_W) for m in range(NM)]
            self.land_set = _rowmajor_set(self.land)
            self.red_all = red or [(p, 7) for p in range(self.NP)]
            self.comm = comm or {
                (d, l): (2 * (0 if d == "fwd" else 1) + l, 8) for d in ("fwd", "bwd") for l in range(links)
            }
        grid = md.compute_with_storage_grid_size()
        self.all_cores = _rect(0, 0, grid.x - 1, grid.y - 1)
        self.gsem = ttnn.create_global_semaphore(md, self.all_cores, 0)
        rep = ttnn.ReplicateTensorToMesh(md)

        def l1_sharded(core_set, n, w):
            mc = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(core_set, [1, w], ttnn.ShardOrientation.ROW_MAJOR),
            )
            return ttnn.from_torch(
                torch.zeros(n, w, dtype=torch.bfloat16),
                device=md,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.bfloat16,
                memory_config=mc,
                mesh_mapper=rep,
            )

        # landing: QL compact tiles per landing core (QL * tileb bytes); receive: NRECV slots x Q compact tiles per reducer core
        self.out = l1_sharded(self.land_set, NM, QL * tileb // 2)
        self.recv_small = False
        if (
            recv_small
        ):  # the receive buffer in the L1_SMALL bank (16 KB: fits NP >= 8 at 512 B tiles): keeps main L1 free for the static CBs of the big programs
            mc = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1_SMALL,
                ttnn.ShardSpec(_cores(self.red_all), [1, NRECV * self.Q * tileb // 2], ttnn.ShardOrientation.ROW_MAJOR),
            )
            self.recv = ttnn.from_torch(
                torch.zeros(self.NP, NRECV * self.Q * tileb // 2, dtype=torch.bfloat16),
                device=md,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.bfloat16,
                memory_config=mc,
                mesh_mapper=rep,
            )
            self.recv_small = True
        else:
            self.recv = l1_sharded(_cores(self.red_all), self.NP, NRECV * self.Q * tileb // 2)
        ttnn.synchronize_device(md)

    def l1_bytes_per_core(self):
        """persistent L1 of this object: (landing cores, reducer cores)"""
        return QL * self.tileb, NRECV * self.Q * self.tileb

    def _phys(self, x, y):
        c = self.md.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
        return c.x, c.y

    def _build(self, part, T):
        md = self.md
        TILE_B = tile_bytes(T)
        assert TILE_B <= self.tileb, (T, self.tileb)
        NP, Q = self.NP, self.Q
        CHUNK = Q * TILE_B
        red = self.red_all
        per = [NP // self.links + (1 if l < NP % self.links else 0) for l in range(self.links)]
        link_p = [list(range(sum(per[:l]), sum(per[: l + 1]))) for l in range(self.links)]
        link_of = {p: l for l, ps in enumerate(link_p) for p in ps}
        dev_recv = ttnn.get_device_tensors(self.recv)
        gsem_addr = ttnn.get_global_semaphore_address(self.gsem)
        recv_addr = self.recv.buffer_address()
        land_addr = self.out.buffer_address()
        in_addr = part.buffer_address()
        red_xy = [self._phys(*c) for c in red]
        land_xy = [self._phys(*c) for c in self.land]
        comm_xy = {k: self._phys(*v) for k, v in self.comm.items()}
        red_set = _cores(red)
        comm_set = _cores(list(self.comm.values()))
        stg_set = _cores(red + list(self.comm.values()))
        CB_RECV, CB_OUT, CB_STG, CB_PKT, CB_OWN = 8, 9, 10, 11, 12
        SEM0 = 0
        acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        fp32 = os.environ.get("HY4_FAR_FP32ACC", "1") == "1"

        def tcb(i, pages):
            return ttnn.CBDescriptor(
                total_size=pages * TILE_B,
                core_ranges=red_set,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=i,
                        data_format=ttnn.bfloat16,
                        page_size=TILE_B,
                        tile=ttnn.TileDescriptor(TILE_B // 64, 32),
                    )
                ],
            )

        mpd = ttnn.MeshProgramDescriptor()
        for r in range(self.rows):
            for col in range(RING):
                coord = ttnn.MeshCoordinate(r, col)
                src = md.get_fabric_node_id(coord)
                fwd_n = md.get_fabric_node_id(ttnn.MeshCoordinate(r, (col + 1) % RING))
                bwd_n = md.get_fabric_node_id(ttnn.MeshCoordinate(r, (col - 1) % RING))
                idx = r * RING + col
                cb_recv = ttnn.cb_descriptor_from_sharded_tensor(CB_RECV, dev_recv[idx])
                cb_recv.core_ranges = red_set
                cb_recv.total_size = NRECV * Q * TILE_B
                cb_recv.format_descriptors = [
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_RECV,
                        data_format=ttnn.bfloat16,
                        page_size=TILE_B,
                        tile=ttnn.TileDescriptor(TILE_B // 64, 32),
                    )
                ]
                cb_out, cb_own = tcb(CB_OUT, Q), tcb(CB_OWN, Q)
                cb_stg = ttnn.CBDescriptor(
                    total_size=NP * CHUNK,
                    core_ranges=stg_set,
                    format_descriptors=[
                        ttnn.CBFormatDescriptor(buffer_index=CB_STG, data_format=ttnn.bfloat16, page_size=CHUNK)
                    ],
                )
                maxpn = max(len(ps) for ps in link_p)
                cb_pkt = ttnn.CBDescriptor(
                    total_size=4 * maxpn * CHUNK,
                    core_ranges=comm_set,
                    format_descriptors=[
                        ttnn.CBFormatDescriptor(buffer_index=CB_PKT, data_format=ttnn.bfloat16, page_size=CHUNK)
                    ],
                )
                cbs = [cb_recv, cb_out, cb_stg, cb_pkt, cb_own]
                sems = [ttnn.SemaphoreDescriptor(SEM0 + p, ttnn.CoreType.WORKER, comm_set, 0) for p in range(NP)]
                kernels = []
                rt_rd, rt_cp, rt_wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
                for p, (x, y) in enumerate(red):
                    rt_rd[x][y] = [col, p]
                    rt_cp[x][y] = []
                    fx, fy = comm_xy[("fwd", link_of[p])]
                    bx, by = comm_xy[("bwd", link_of[p])]
                    rt_wr[x][y] = [col, p, fx, fy, bx, by] + [v for xy in land_xy for v in xy]
                kernels.append(
                    ttnn.KernelDescriptor(
                        kernel_source=f"{KDIR}/ar_red_reader.cpp",
                        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                        core_ranges=red_set,
                        compile_time_args=[CB_RECV, Q, NT, RING, 0, CB_OWN, TILE_B] + acc(part),
                        runtime_args=rt_rd,
                        common_runtime_args=[in_addr, gsem_addr],
                        config=ttnn.ReaderConfigDescriptor(),
                    )
                )
                kernels.append(
                    ttnn.KernelDescriptor(
                        kernel_source=f"{KDIR}/ar_red_compute.cpp",
                        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                        core_ranges=red_set,
                        compile_time_args=[CB_RECV, CB_OUT, Q, RING, 1 if fp32 else 0, CB_OWN],
                        runtime_args=rt_cp,
                        config=ttnn.ComputeConfigDescriptor(
                            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=fp32
                        ),
                    )
                )
                kernels.append(
                    ttnn.KernelDescriptor(
                        kernel_source=f"{KDIR}/ar_red_writer.cpp",
                        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                        core_ranges=red_set,
                        compile_time_args=[CB_OUT, CB_STG, Q, NP, SEM0, NPL, TILE_B],
                        runtime_args=rt_wr,
                        common_runtime_args=[gsem_addr, land_addr],
                        config=ttnn.WriterConfigDescriptor(),
                    )
                )
                kernels.append(
                    ttnn.KernelDescriptor(
                        kernel_source=f"{KDIR}/ar_land.cpp",
                        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                        core_ranges=self.land_set,
                        compile_time_args=[NP // NPL],
                        runtime_args=ttnn.RuntimeArgs(),
                        common_runtime_args=[gsem_addr],
                        config=ttnn.ReaderConfigDescriptor(),
                    )
                )
                comm_kidx = {}
                for name, ndist, dr in (("fwd", 4, 0), ("bwd", 3, 1)):
                    for lk in range(self.links):
                        x, y = self.comm[(name, lk)]
                        ps = link_p[lk]
                        rt = ttnn.RuntimeArgs()
                        rt[x][y] = [col] + [v for xy in red_xy for v in xy] + [v for xy in land_xy for v in xy]
                        fab_base = 1 + 2 * NP + 2 * NM
                        kernels.append(
                            ttnn.KernelDescriptor(
                                kernel_source=f"{KDIR}/ar_comm.cpp",
                                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                                core_ranges=_cores([(x, y)]),
                                compile_time_args=[
                                    dr,
                                    ndist,
                                    len(ps),
                                    ps[0],
                                    Q,
                                    NT,
                                    CB_PKT,
                                    CB_STG,
                                    SEM0,
                                    NP,
                                    fab_base,
                                    self.stop,
                                    0,
                                    TILE_B,
                                    NPL,
                                ]
                                + acc(part),
                                runtime_args=rt,
                                common_runtime_args=[in_addr, recv_addr, gsem_addr, land_addr],
                                config=ttnn.WriterConfigDescriptor(),
                            )
                        )
                        comm_kidx[(name, lk)] = len(kernels) - 1
                prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
                for (name, lk), ki in comm_kidx.items():
                    dst = fwd_n if name == "fwd" else bwd_n
                    x, y = self.comm[(name, lk)]
                    fargs = ttnn.setup_routing_plane_connection(src, [dst], [lk], prog, ki, ttnn.CoreCoord(x, y))
                    prog.kernels[ki].runtime_args[x][y].extend(fargs)
                mpd[ttnn.MeshCoordinateRange(coord, coord)] = prog
        return mpd

    def __call__(self, part, T=4):
        """All-reduce of the partial [1,1,32,6144] bf16 TILE DRAM tensor (rows 0..T-1 valid): returns the landing tensor (compact tiles, see decode)."""
        ttnn.generic_op([part, self.recv, self.out], self._build(part, T))
        return self.out

    @staticmethod
    def decode(land_host, T, tileb=512):
        """land_host [NM, QL * tileb / 2] bf16 (one chip) -> [T, 6144] fp32 (the reduced rows)."""
        rws = tileb // 64
        t = land_host.reshape(NM, -1)[:, : QL * tileb // 2].reshape(NM, QL, 2, rws, 16)[:, :, :, :T]
        t = t.permute(3, 0, 1, 2, 4).reshape(T, NM * QL * 32)
        return t.float()


# ---------------------------------------------------------------------------------------------- model glue (HY4_FUSE_AR=1, default OFF)
class FusedPartial:
    """The un-reduced [1,1,T,6144] bf16 TILE DRAM partial of a column-sharded matmul (attention o_proj, MoE routed+shared): returned instead of the all-reduced tensor when the layer fuses
    the all-reduce into the next iHC pre call (``Hy4FuseAR`` + ``Hy4IHCFused.pre(land=...)``)."""

    def __init__(self, part):
        self.part = part


_SHARED = {}


def enabled():
    return os.environ.get("HY4_FUSE_AR", "0") == "1"


def applies(T):
    return enabled() and 1 <= int(T) <= (16 if os.environ.get("HY4_FAR_T16", "0") == "1" else 8)


def shared(md):
    """ONE Hy4FuseAR per mesh (buffers + semaphores shared by all layers; consecutive calls are ordered by the all-to-all dependency).  Build it AFTER the CCLManager and every other persistent L1 / global
    semaphore allocation (probe finding: a persistent L1 buffer allocated before the CCLManager's global semaphores made the stock reduce_scatter/all_gather return wrong sums, PCC 0.93) and OUTSIDE a trace.
    """
    from models.demos.blackhole.hy4_preview.tt.hy4_ihc_fused import R0

    key = id(md)
    if key not in _SHARED:
        _SHARED[key] = Hy4FuseAR(
            md,
            tileb=1024 if os.environ.get("HY4_FAR_T16", "0") == "1" else 512,
            links=int(os.environ.get("HY4_FAR_LINKS", "2")),
            ihc_row0=R0,
            nred=int(os.environ.get("HY4_FAR_NRED", "4")),
        )
    return _SHARED[key]


def patch_dsv4_tail():
    """The DSV4 MoE block (``_TailDecode.forward``) returns ``fuse_tail_ar.FusedPartial(red)`` when ``ft_defer`` is set and ``fuse_tail_ar.applies(...)``: route that decision to this module."""
    from models.demos.blackhole.deepseek_v41_flash.tt import fuse_tail_ar as _FT

    if getattr(_FT, "_hy4_patched", False):
        return
    _FT.applies = lambda md, T, part=None: (
        applies(T)
        and (
            part is None
            or (
                part.dtype == ttnn.bfloat16
                and part.layout == ttnn.TILE_LAYOUT
                and not part.memory_config().is_sharded()
            )
        )
    )
    _FT._hy4_patched = True


def pre_with(ihc, far, x, w4, y, pp_prev, x_out):
    """Hy4IHCFused.pre for a y that is either a tensor or a FusedPartial / DSV4 FusedPartial (all-reduce fused: the one-shot AR lands in the iHC cores' L1)."""
    part = getattr(y, "part", None)
    if part is None:
        return ihc.pre(x, w4, y=y, pp_prev=pp_prev, x_out=x_out)
    T = int(x.shape[2]) // 4
    far(part, T=T)
    res = ihc.pre(x, w4, land=(far.out.buffer_address(), tile_bytes(T)), pp_prev=pp_prev, x_out=x_out)
    ttnn.deallocate(part)
    return res
