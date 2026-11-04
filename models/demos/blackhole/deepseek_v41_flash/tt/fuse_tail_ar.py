# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fused decode collective tail: ONE program for the all-reduce of the wo_b partial over the 8 chips of a mesh row (replaces lean reduce_scatter + all_gather), optionally with the
mHC post (mhc_expand3 math) consuming the gathered rows in the same program (``call_post``).

The partial is the [1,1,32,5120] bf16 TILE tensor of wo_b with only rows 0..VR-1 (VR = 8) valid. One-shot algorithm (see fuse_tail_kernels/ar_comm.cpp): every chip sends the valid pieces of its 7 foreign
640-column slices straight to the owner chips (fabric packets of NT/NP compact tiles, fused with a global-semaphore increment), the owner's NP reducer cores sum the 8 contributions on the Tensix and the
reduced chunks are multicast to the 7 other chips and land in the NM = 8 NP landing cores (Q = NT/NP tile columns each, 'compact' format: rows 0..7 of face 0 and face 1 = an 8x32 tiny tile, 512 B).
With the mHC post the landing cores are the first NM cores of the mHC expand grid (4 tile columns per core: NP = 5, Q = 4) and the expand reader waits for the landing semaphore instead of reading y from DRAM.

All cross-chip buffers are persistent (L1 tensors) and all cross-chip semaphores are global semaphores: there is no barrier, the all-to-all dependency of consecutive calls orders the buffer reuse.
Default off: DSV41_FUSE_TAIL=1 (model integration); this module is also driven by tests/test_fuse_tail_ar.py.
"""

import os

import torch

import ttnn

KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fuse_tail_kernels")
MHC_KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mhc_kernels")

RING = 8
NT = 20  # 32-column tiles of a 640-column slice
NP = 5  # reducer cores per chip
Q = NT // NP  # tiles per reducer chunk (= tile columns per mHC expand core)
NM = RING * NP  # landing cores (one per chunk of the whole row)
NRECV = (
    RING - 1
)  # receive slots per reducer: one per REMOTE chip (the own chunk lives in its own circular buffer): 14 KiB instead of 16
TILE_B = 512  # compact tile bytes (8 x 32 bf16, two faces)
CHUNK = Q * TILE_B
GRID_W = 13


def _cores(lst):
    from models.demos.blackhole.deepseek_v41_flash.tt.mhc_pre import _merged_set

    return _merged_set(list(lst))


def _rect(x0, y0, x1, y1):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))])


def _rowmajor_set(cores):
    """CoreRangeSet of a row-major list of cores (full rows of the grid + the partial last row)."""
    n = len(cores)
    full = n // GRID_W
    ranges = []
    if full:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(GRID_W - 1, full - 1)))
    if n % GRID_W:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full), ttnn.CoreCoord(n % GRID_W - 1, full)))
    return ttnn.CoreRangeSet(ranges)


class FuseTailAR:
    def __init__(self, md, stop=0, red_cores=None, comm_cores=None, in_l1=False, links=2):
        self.md = md
        rows, cols = tuple(md.shape)
        assert cols == RING
        self.rows = rows
        self.stop = stop
        self.red = red_cores or [(p, 7) for p in range(NP)]
        self.in_l1 = in_l1
        self.links = links
        # reducer chunks served by each link (contiguous): 5 reducers over 2 links -> [0,1,2], [3,4]
        per = [NP // links + (1 if l < NP % links else 0) for l in range(links)]
        self.link_p = [list(range(sum(per[:l]), sum(per[: l + 1]))) for l in range(links)]
        self.link_of = {p: l for l, ps in enumerate(self.link_p) for p in ps}
        self.comm = comm_cores or {
            (d, l): (2 * (0 if d == "fwd" else 1) + l, 8) for d in ("fwd", "bwd") for l in range(links)
        }
        # landing cores = the first NM cores of the mHC expand grid (row-major, 13 wide): slice m = 8-chip-row chunk (c * NP + p)
        global GRID_W
        GRID_W = int(md.compute_with_storage_grid_size().x)
        self.land = [(m % GRID_W, m // GRID_W) for m in range(NM)]
        grid = md.compute_with_storage_grid_size()
        self.all_cores = _rect(0, 0, grid.x - 1, grid.y - 1)
        self.gsem = ttnn.create_global_semaphore(md, self.all_cores, 0)
        rep = ttnn.ReplicateTensorToMesh(md)

        def l1_sharded(core_set, n, w, btype=None):
            mc = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                btype or ttnn.BufferType.L1,
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

        # L1 placement. Persistent L1 buffers are allocated TOP-DOWN (the first one sits at the top of L1) and a program's static circular buffers must end below the lowest
        # one ('Statically allocated circular buffers ... clash with L1 buffers ... static circular buffer region ends at 685968', rc5c server boot: the unified-MoE prefill
        # programs on the whole 12 x 10 compute grid; the 18 KiB of fuse-tail buffers were the lowest persistent buffers, at 682880). The 16 KiB receive buffer is exactly one
        # L1_SMALL bank (the device is opened with l1_small_size=16384, which nothing else uses in this model): it lives there when it fits
        # (DSV41_FUSE_TAIL_RECV_SMALL=0 or an allocation failure: the main L1 as before), leaving 2 KiB of main L1 (the landing buffer, which must be an ordinary L1 tensor).
        # DSV41_FUSE_TAIL_FILL=<bytes> (debug): a filler L1 buffer on every core allocated first, to emulate the persistent-L1 pressure of the server boot.
        _fill = int(os.environ.get("DSV41_FUSE_TAIL_FILL", "0"))
        self._fill = l1_sharded(self.all_cores, grid.x * grid.y, _fill // 2) if _fill > 0 else None
        self.out = l1_sharded(_rowmajor_set(self.land), NM, Q * TILE_B // 2)
        # receive buffer of the reducers: 8 slots x Q compact tiles; landing buffer: Q compact tiles per landing core
        _rcores = _rect(self.red[0][0], self.red[0][1], self.red[-1][0], self.red[-1][1])
        self.recv = None
        self.recv_small = False
        if os.environ.get("DSV41_FUSE_TAIL_RECV_SMALL", "1") == "1":
            try:
                self.recv = l1_sharded(_rcores, NP, NRECV * Q * TILE_B // 2, btype=ttnn.BufferType.L1_SMALL)
                self.recv_small = True
            except Exception as e:  # noqa: BLE001 (no L1_SMALL bank / not free / sharded L1_SMALL unsupported)
                print(
                    f"FUSETAIL recv buffer not placed in L1_SMALL ({type(e).__name__}); using the main L1", flush=True
                )
                self.recv = None
        if self.recv is None:
            self.recv = l1_sharded(_rcores, NP, NRECV * Q * TILE_B // 2)
        ttnn.synchronize_device(md)
        if os.environ.get("DSV41_FUSE_TAIL_DEBUG") == "1":
            print(
                f"FUSETAIL L1 fill {self._fill.buffer_address() if self._fill is not None else None} out {self.out.buffer_address()} recv {self.recv.buffer_address()} (L1_SMALL {self.recv_small})",
                flush=True,
            )

    # ------------------------------------------------------------------------------------------------
    def _phys(self, x, y):
        c = self.md.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
        return c.x, c.y

    def _build(self, part, post_args=None):
        """Mesh program descriptor of the all-reduce (+ the mHC post when post_args = dict(x, comb, post, out, T))."""
        md = self.md
        dev_recv = ttnn.get_device_tensors(self.recv)
        gsem_addr = ttnn.get_global_semaphore_address(self.gsem)
        recv_addr = self.recv.buffer_address()
        land_addr = self.out.buffer_address()
        in_addr = part.buffer_address()
        red_xy = [self._phys(*c) for c in self.red]
        land_xy = [self._phys(*c) for c in self.land]
        comm_xy = {k: self._phys(*v) for k, v in self.comm.items()}
        in_arr = []
        if self.in_l1:
            gcores = ttnn.corerange_to_cores(part.memory_config().shard_spec.grid, row_wise=True)
            assert len(gcores) == RING
            in_arr = [v for c_ in gcores for v in self._phys(c_.x, c_.y)]
        red_set = _cores(self.red)
        comm_set = _cores(list(self.comm.values()))
        stg_set = _cores(self.red + list(self.comm.values()))
        CB_RECV, CB_OUT, CB_STG, CB_PKT, CB_OWN = 8, 9, 10, 11, 12  # (the mHC expand uses 0..4 on its own cores)
        SEM0, SEM_PARTNER = 0, NP
        acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        fused = post_args is not None
        if fused:
            px = post_args
            T = px["T"]
            assert 4 < T <= 8, "fused post: T in 5..8 (two token groups of 4)"
            nt_total = RING * NT
            G = (T + 3) // 4
            csl = max(c for c in range(1, nt_total + 1) if nt_total % c == 0 and G * c <= 96)
            assert csl == NM and nt_total // csl == Q, (csl, NM, Q)
            mcores = [(k % GRID_W, k // GRID_W) for k in range(G * csl)]
            assert mcores[:NM] == self.land
            mset = _cores(mcores)
            gpc = Q
            assert max(c[1] for c in mcores) < min(c[1] for c in self.red)
        mpd = ttnn.MeshProgramDescriptor()
        rows = self.rows
        for r in range(rows):
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
                        tile=ttnn.TileDescriptor(8, 32),
                    )
                ]
                cb_out = ttnn.CBDescriptor(
                    total_size=Q * TILE_B,
                    core_ranges=red_set,
                    format_descriptors=[
                        ttnn.CBFormatDescriptor(
                            buffer_index=CB_OUT,
                            data_format=ttnn.bfloat16,
                            page_size=TILE_B,
                            tile=ttnn.TileDescriptor(8, 32),
                        )
                    ],
                )
                cb_own = ttnn.CBDescriptor(
                    total_size=Q * TILE_B,
                    core_ranges=red_set,
                    format_descriptors=[
                        ttnn.CBFormatDescriptor(
                            buffer_index=CB_OWN,
                            data_format=ttnn.bfloat16,
                            page_size=TILE_B,
                            tile=ttnn.TileDescriptor(8, 32),
                        )
                    ],
                )
                cb_stg = ttnn.CBDescriptor(
                    total_size=NP * CHUNK,
                    core_ranges=stg_set,
                    format_descriptors=[
                        ttnn.CBFormatDescriptor(buffer_index=CB_STG, data_format=ttnn.bfloat16, page_size=CHUNK)
                    ],
                )
                maxpn = max(len(ps) for ps in self.link_p)
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
                for p, (x, y) in enumerate(self.red):
                    rt_rd[x][y] = [col, p] + in_arr
                    rt_cp[x][y] = []
                    fx, fy = comm_xy[("fwd", self.link_of[p])]
                    bx, by = comm_xy[("bwd", self.link_of[p])]
                    rt_wr[x][y] = [col, p, fx, fy, bx, by] + [v for xy in land_xy for v in xy]
                kernels.append(
                    ttnn.KernelDescriptor(
                        kernel_source=f"{KDIR}/ar_red_reader.cpp",
                        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                        core_ranges=red_set,
                        compile_time_args=[CB_RECV, Q, NT, RING, int(self.in_l1), CB_OWN] + acc(part),
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
                        compile_time_args=[
                            CB_RECV,
                            CB_OUT,
                            Q,
                            RING,
                            1 if os.environ.get("DSV41_FUSE_TAIL_FP32ACC", "1") == "1" else 0,
                            CB_OWN,
                        ],
                        runtime_args=rt_cp,
                        config=ttnn.ComputeConfigDescriptor(
                            math_fidelity=ttnn.MathFidelity.HiFi4,
                            fp32_dest_acc_en=os.environ.get("DSV41_FUSE_TAIL_FP32ACC", "1") == "1",
                        ),
                    )
                )
                kernels.append(
                    ttnn.KernelDescriptor(
                        kernel_source=f"{KDIR}/ar_red_writer.cpp",
                        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                        core_ranges=red_set,
                        compile_time_args=[CB_OUT, CB_STG, Q, NP, SEM0],
                        runtime_args=rt_wr,
                        common_runtime_args=[gsem_addr, land_addr],
                        config=ttnn.WriterConfigDescriptor(),
                    )
                )
                if not fused:
                    kernels.append(
                        ttnn.KernelDescriptor(
                            kernel_source=f"{KDIR}/ar_land.cpp",
                            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                            core_ranges=_rowmajor_set(self.land),
                            compile_time_args=[],
                            runtime_args=ttnn.RuntimeArgs(),
                            common_runtime_args=[gsem_addr],
                            config=ttnn.ReaderConfigDescriptor(),
                        )
                    )
                comm_kidx = {}
                for name, ndist, dr in (("fwd", 4, 0), ("bwd", 3, 1)):
                    for lk in range(self.links):
                        x, y = self.comm[(name, lk)]
                        ps = self.link_p[lk]
                        rt = ttnn.RuntimeArgs()
                        rt[x][y] = [col] + [v for xy in red_xy for v in xy] + [v for xy in land_xy for v in xy] + in_arr
                        fab_base = 1 + 2 * NP + 2 * NM + len(in_arr)
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
                                    int(self.in_l1),
                                ]
                                + acc(part),
                                runtime_args=rt,
                                common_runtime_args=[in_addr, recv_addr, gsem_addr, land_addr],
                                config=ttnn.WriterConfigDescriptor(),
                            )
                        )
                        comm_kidx[(name, lk)] = len(kernels) - 1
                if fused:
                    sems.append(ttnn.SemaphoreDescriptor(SEM_PARTNER, ttnn.CoreType.WORKER, mset, 0))
                    kernels += self._mhc_kernels(px, mcores, mset, gsem_addr, land_addr, SEM_PARTNER, cbs, acc)
                prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
                for (name, lk), ki in comm_kidx.items():
                    dst = fwd_n if name == "fwd" else bwd_n
                    x, y = self.comm[(name, lk)]
                    fargs = ttnn.setup_routing_plane_connection(src, [dst], [lk], prog, ki, ttnn.CoreCoord(x, y))
                    prog.kernels[ki].runtime_args[x][y].extend(fargs)
                mpd[ttnn.MeshCoordinateRange(coord, coord)] = prog
        return mpd

    def _mhc_kernels(self, px, mcores, mset, gsem_addr, land_addr, sem_partner, cbs, acc):
        """The mHC expand (mhc_expand3 fast + token-group layout) reader / writer / compute on the 80 expand cores; the reader takes y from the landing buffers."""
        x, comb, post, out, T = px["x"], px["comb"], px["post"], px["out"], px["T"]
        nt = RING * NT
        G = (T + 3) // 4
        gpc = Q
        csl = NM
        CB_A, CB_B, CB_S, CB_O, CB_SA = 0, 1, 2, 3, 4

        def cb(i, pages, page, dt):
            return ttnn.CBDescriptor(
                total_size=pages * page,
                core_ranges=mset,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=dt, page_size=page)],
            )

        cbs += [
            cb(CB_A, 1, 4096, ttnn.float32),
            cb(CB_B, gpc, 4096, ttnn.float32),
            cb(CB_S, 1, 4096, ttnn.float32),
            cb(CB_O, min(gpc, 8), 4096, ttnn.float32),
            cb(CB_SA, max(1, (4 * 512 + 4095) // 4096), 4096, ttnn.float32),
        ]
        rt_rd, rt_wr, rt_cp = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for k, (cx, cy) in enumerate(mcores):
            g = k // csl
            if G > 1 and g == 1:
                px_, py_ = self._phys(*mcores[k - csl])  # partner = the landing core of the same column slice
                role = 1
            else:
                px_, py_ = self._phys(*mcores[k + csl]) if G > 1 else (0, 0)
                role = 0
            rt_rd[cx][cy] = [(k % csl) * gpc, g, role, px_, py_]
            rt_wr[cx][cy] = [(k % csl) * gpc, g]
            rt_cp[cx][cy] = []
        reader = ttnn.KernelDescriptor(
            kernel_source=f"{KDIR}/ft_expand_reader.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=mset,
            compile_time_args=[CB_B, CB_S, T, nt, G, gpc, sem_partner] + acc(x),
            runtime_args=rt_rd,
            common_runtime_args=[x.buffer_address(), gsem_addr, land_addr],
            config=ttnn.ReaderConfigDescriptor(),
        )
        writer = ttnn.KernelDescriptor(
            kernel_source=f"{MHC_KDIR}/mhc_expand5_writer.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=mset,
            compile_time_args=[CB_O, CB_A, CB_SA, T, nt, gpc, 0] + acc(out) + acc(comb) + acc(post),
            runtime_args=rt_wr,
            common_runtime_args=[out.buffer_address(), comb.buffer_address(), post.buffer_address()],
            config=ttnn.WriterConfigDescriptor(),
        )
        compute = ttnn.KernelDescriptor(
            kernel_source=f"{MHC_KDIR}/mhc_expand4_compute.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=mset,
            compile_time_args=[CB_A, CB_B, CB_O, T, 1, gpc],
            runtime_args=rt_cp,
            config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True),
        )
        return [reader, writer, compute]

    # ------------------------------------------------------------------------------------------------
    def __call__(self, part):
        """All-reduce only: returns the landing tensor (compact format, see decode)."""
        ttnn.generic_op([part, self.recv, self.out], self._build(part))
        return self.out

    def call_post(self, part, x, comb, post, out):
        """All-reduce + mHC post: out = post * allreduce(part) + comb^T x (the mhc_expand3 result, valid rows), x / comb / post / out as for mhc_expand3 (fp32 TILE, T in 5..8)."""
        T = int(x.shape[0])
        mpd = self._build(part, dict(x=x, comb=comb, post=post, out=out, T=T))
        ttnn.generic_op([part, x, comb, post, out, self.recv, self.out], mpd)
        return out

    # ------------------------------------------------------------------------------------------------
    @staticmethod
    def decode(land_host, vr=8):
        """land_host [NM, Q*256] bf16 (torch, one chip) -> [vr, 5120] fp32 (the reduced rows)."""
        t = land_host.reshape(NM, Q, 2, vr, 16)  # core, tile, face, row, col
        t = t.permute(3, 0, 1, 2, 4).reshape(vr, NM * Q * 32)
        return t.float()


def delay_op(md, dummy, us_per_chip):
    """Test helper: every chip (row-major index) spins for us_per_chip[idx] microseconds on core (10, 9)."""
    cs = _cores([(10, 9)])
    mpd = ttnn.MeshProgramDescriptor()
    rows, cols = tuple(md.shape)
    for r in range(rows):
        for c in range(cols):
            rt = ttnn.RuntimeArgs()
            rt[10][9] = [int(us_per_chip[r * cols + c])]
            k = ttnn.KernelDescriptor(
                kernel_source=f"{KDIR}/ft_delay.cpp",
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=cs,
                compile_time_args=[],
                runtime_args=rt,
                config=ttnn.WriterConfigDescriptor(),
            )
            mpd[
                ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, c), ttnn.MeshCoordinate(r, c))
            ] = ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[])
    ttnn.generic_op([dummy, dummy], mpd)
    return dummy


class FusedPartial:
    """Returned by the attention ``_finish`` instead of the all-reduced output when the layer asked for the fused tail (``attention._ft_defer``): the un-reduced wo_b partial."""

    def __init__(self, part):
        self.part = part


_SHARED = {}


def enabled():
    return os.environ.get("DSV41_FUSE_TAIL", "0") == "1"


def shared(md):
    """The per-mesh FuseTailAR (buffers and semaphores are shared by all layers: the all-to-all dependency of consecutive calls makes the reuse safe)."""
    key = id(md)
    if key not in _SHARED:
        _SHARED[key] = FuseTailAR(md, in_l1=False, links=int(os.environ.get("DSV41_FUSE_TAIL_LINKS", "2")))
    return _SHARED[key]


def applies(md, T, part=None):
    """True when the decode layer can use the fused all-reduce + mHC post: T = 8 token rows, 4x8 mesh, ring fabric, DRAM interleaved bf16 TILE partial."""
    if not enabled() or T != 8 or tuple(md.shape) != (4, 8):
        return False
    if part is not None:
        mc = part.memory_config()
        if part.dtype != ttnn.bfloat16 or part.layout != ttnn.TILE_LAYOUT or mc.buffer_type != ttnn.BufferType.DRAM:
            return False
        if mc.is_sharded():
            return False
    return True
