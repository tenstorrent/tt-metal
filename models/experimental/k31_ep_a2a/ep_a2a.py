# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""K3.1 expert-parallel all-to-all prototype (dispatch + combine) on a 1xN tt-fabric 1D line.

Built entirely with ttnn.generic_op + tt-fabric worker APIs (no C++ op / no rebuild): kernels JIT-compile.

Layout (all row-major, rows are opaque bytes, stored as uint32 words):
  x        [T, Rd/4]        this device's tokens (Rd = 7168 for fp8 / 14336 for bf16 hidden 7168)
  meta     [T, 16]          per token: 16 x u16 expert ids, 16 x bf16 router weights (64 B)
  recv_x   [N*T, Rd/4]      dispatch output: slot (s*T + t) = token t of source s (valid iff t hit this device)
  recv_meta[N, T*16]        dispatch output: page s = source s's whole meta block
  y        [N*T, Rc/4]      expert side: slot (s*T + t) = this device's router-weighted partial for (s, t)
  comb     [N*T, Rc/4]      combine output on the source: slot (d*T + t) = partial from device d for token t
Experts are placed contiguously: device d owns experts [d*EPD, (d+1)*EPD).
"""
import os

import ttnn

KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")
ROLE_EAST, ROLE_WEST, ROLE_SELF = 0, 1, 2
MODE_DISPATCH, MODE_COMBINE = 0, 1
META_BYTES = 64


def _crs(core):
    return ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])


def alloc_rows(mesh, rows, row_bytes, mem):
    assert row_bytes % 4 == 0
    return ttnn.allocate_tensor_on_device(
        ttnn.Shape([rows, row_bytes // 4]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, mesh, mem
    )


class EpA2A:
    """Owns the receive buffers + completion semaphores and builds the two MeshProgramDescriptors."""

    def __init__(
        self,
        mesh,
        N,
        T,
        K,
        EPD,
        disp_row_bytes,
        comb_row_bytes,
        links=2,
        payload=None,
        recv_mem=None,
        depth=4,
        batch=2,
        ring=False,
        flush_every=1,
    ):
        assert tuple(mesh.shape) == (1, N), f"prototype is 1xN line, got {tuple(mesh.shape)}"
        assert N <= 16 and K <= 16 and T * META_BYTES <= 4096
        self.mesh, self.N, self.T, self.K, self.EPD = mesh, N, T, K, EPD
        self.Rd, self.Rc, self.L = disp_row_bytes, comb_row_bytes, links
        self.payload = payload or ttnn.get_tt_fabric_max_payload_size_bytes()
        assert self.payload % 16 == 0
        self.depth, self.batch = depth, batch
        self.ring = ring
        assert not ring or N % 2 == 0
        self.defines = [(k, "1") for k in os.environ.get("EP_DEFINES", "").split(",") if k]
        self.defines.append(("EP_FLUSH_EVERY", str(flush_every)))
        assert depth % flush_every == 0 and depth % batch == 0
        assert depth % batch == 0
        mem = recv_mem or ttnn.L1_MEMORY_CONFIG
        self.recv_x = alloc_rows(mesh, N * T, self.Rd, mem)
        self.recv_meta = alloc_rows(mesh, N, T * META_BYTES, mem)
        self.comb = alloc_rows(mesh, N * T, self.Rc, mem)
        self.self_core = ttnn.CoreCoord(0, 0)
        grid = mesh.compute_with_storage_grid_size()
        allc = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
        self.sem_d = ttnn.create_global_semaphore(mesh, allc, 0)
        self.sem_c = ttnn.create_global_semaphore(mesh, allc, 0)
        self.sem_xy = mesh.worker_core_from_logical_core(self.self_core)
        self.hdr_bytes = ttnn.get_tt_fabric_packet_header_size_bytes()
        ttnn.synchronize_device(mesh)

    # role cores: SELF at (0,0); EAST link l at (1+l, 0); WEST link l at (1+l, 1)
    def _roles(self, c):
        roles = [(ROLE_SELF, 0, self.self_core)]
        for l in range(self.L):
            if c < self.N - 1 or self.ring:
                roles.append((ROLE_EAST, l, ttnn.CoreCoord(1 + l, 0)))
            if c > 0 or self.ring:
                roles.append((ROLE_WEST, l, ttnn.CoreCoord(1 + l, 1)))
        return roles

    def _program(self, mode, c, meta_src, row_src, out, outm, row_bytes, sem):
        N, T = self.N, self.T
        meta_total = T * META_BYTES if mode == MODE_DISPATCH else N * T * META_BYTES
        kernels, cbs, senders = [], [], []
        for role, link, core in self._roles(c):
            cr = _crs(core)
            cbs += [
                ttnn.CBDescriptor(
                    total_size=meta_total,
                    core_ranges=cr,
                    format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=meta_total)],
                ),
                ttnn.CBDescriptor(
                    total_size=self.depth * row_bytes,
                    core_ranges=cr,
                    format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=1, data_format=ttnn.uint32, page_size=row_bytes)],
                ),
                ttnn.CBDescriptor(
                    total_size=16 * self.hdr_bytes,
                    core_ranges=cr,
                    format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=2, data_format=ttnn.uint32, page_size=self.hdr_bytes)],
                ),
            ]
            common = [mode, N, c, T, self.K, self.EPD, row_bytes, self.L, link, role | (int(self.ring) << 8)]
            r_ct = common + [0, 1, self.batch]
            r_ct += ttnn.TensorAccessorArgs(meta_src).get_compile_time_args()
            r_ct += ttnn.TensorAccessorArgs(row_src).get_compile_time_args()
            r_rt = ttnn.RuntimeArgs()
            r_rt[core.x][core.y] = [meta_src.buffer_address(), row_src.buffer_address()]
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=os.path.join(KDIR, "ep_reader.cpp"),
                    core_ranges=cr,
                    compile_time_args=r_ct,
                    runtime_args=r_rt,
                    config=ttnn.ReaderConfigDescriptor(),
                )
            )
            w_ct = common + [0, 1, 2, self.payload, (N if self.ring else N - 1) * self.L, self.hdr_bytes]
            w_ct += ttnn.TensorAccessorArgs(out).get_compile_time_args()
            w_ct += ttnn.TensorAccessorArgs(outm).get_compile_time_args()
            w_rt = ttnn.RuntimeArgs()
            w_rt[core.x][core.y] = [
                out.buffer_address(),
                outm.buffer_address(),
                ttnn.get_global_semaphore_address(sem),
                self.sem_xy.x,
                self.sem_xy.y,
            ]
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=os.path.join(KDIR, "ep_writer.cpp"),
                    core_ranges=cr,
                    compile_time_args=w_ct,
                    defines=self.defines,
                    runtime_args=w_rt,
                    config=ttnn.WriterConfigDescriptor(),
                )
            )
            if role != ROLE_SELF:
                senders.append((len(kernels) - 1, role, link, core))
        prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        src = self.mesh.get_fabric_node_id(ttnn.MeshCoordinate(0, c))
        for kidx, role, link, core in senders:
            nbr = (c + 1) % N if role == ROLE_EAST else (c - 1) % N
            dst = self.mesh.get_fabric_node_id(ttnn.MeshCoordinate(0, nbr))
            ref = prog.kernels[kidx].runtime_args[core.x][core.y]
            ref.extend(ttnn.setup_fabric_connection(src, dst, link, prog, core))
        return prog

    def dispatch_program(self, x, meta):
        mpd = ttnn.MeshProgramDescriptor()
        for c in range(self.N):
            coord = ttnn.MeshCoordinate(0, c)
            mpd[ttnn.MeshCoordinateRange(coord, coord)] = self._program(
                MODE_DISPATCH, c, meta, x, self.recv_x, self.recv_meta, self.Rd, self.sem_d
            )
        return mpd

    def combine_program(self, y):
        mpd = ttnn.MeshProgramDescriptor()
        for c in range(self.N):
            coord = ttnn.MeshCoordinate(0, c)
            mpd[ttnn.MeshCoordinateRange(coord, coord)] = self._program(
                MODE_COMBINE, c, self.recv_meta, y, self.comb, self.recv_meta, self.Rc, self.sem_c
            )
        return mpd

    def dispatch(self, x, meta, mpd):
        return ttnn.generic_op([x, meta, self.recv_x, self.recv_meta], mpd)

    def combine(self, y, mpd):
        return ttnn.generic_op([y, self.recv_meta, self.comb], mpd)

    # ---------------- combine reduction (source side): out[t] = sum_d comb[d*T + t] over devices t hit
    def reduce_program(self, meta, out, max_cores=64):
        N, T = self.N, self.T
        CH = 2048
        nch = self.Rc // CH
        units = T * nch
        grid = self.mesh.compute_with_storage_grid_size()
        cores = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x)]
        # keep the a2a role cores (x<=2, y<=1) free so reduce can later be fused/overlapped with combine
        cores = [c for c in cores if not (c.x <= self.L and c.y <= 1)][: max_cores]
        C = min(len(cores), units)
        per = -(-units // C)
        cores = cores[:C]
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
        cbs = [
            ttnn.CBDescriptor(total_size=2 * N * CH, core_ranges=crs,
                              format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=CH)]),
            ttnn.CBDescriptor(total_size=T * META_BYTES + CH, core_ranges=crs,
                              format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=1, data_format=ttnn.uint32, page_size=T * META_BYTES + CH)]),
            ttnn.CBDescriptor(total_size=2 * CH, core_ranges=crs,
                              format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=16, data_format=ttnn.bfloat16, page_size=CH)]),
        ]
        mpd = ttnn.MeshProgramDescriptor()
        for c in range(N):
            r_rt, w_rt, c_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
            for i, core in enumerate(cores):
                u0, u1 = min(i * per, units), min((i + 1) * per, units)
                r_rt[core.x][core.y] = [meta.buffer_address(), self.comb.buffer_address(), u0, u1]
                w_rt[core.x][core.y] = [out.buffer_address(), u0, u1]
                c_rt[core.x][core.y] = [u1 - u0]
            r_ct = [N, c, T, self.K, self.EPD, self.Rc, 0, 1]
            r_ct += ttnn.TensorAccessorArgs(meta).get_compile_time_args()
            r_ct += ttnn.TensorAccessorArgs(self.comb).get_compile_time_args()
            w_ct = [self.Rc, 16] + ttnn.TensorAccessorArgs(out).get_compile_time_args()
            ks = [
                ttnn.KernelDescriptor(kernel_source=os.path.join(KDIR, "reduce_reader.cpp"), core_ranges=crs,
                                      compile_time_args=r_ct, runtime_args=r_rt, config=ttnn.ReaderConfigDescriptor()),
                ttnn.KernelDescriptor(kernel_source=os.path.join(KDIR, "reduce_writer.cpp"), core_ranges=crs,
                                      compile_time_args=w_ct, runtime_args=w_rt, config=ttnn.WriterConfigDescriptor()),
                ttnn.KernelDescriptor(kernel_source=os.path.join(KDIR, "reduce_compute.cpp"), core_ranges=crs,
                                      compile_time_args=[N, 0, 16], runtime_args=c_rt,
                                      config=ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=True)),
            ]
            coord = ttnn.MeshCoordinate(0, c)
            mpd[ttnn.MeshCoordinateRange(coord, coord)] = ttnn.ProgramDescriptor(kernels=ks, semaphores=[], cbs=cbs)
        return mpd

    def reduce(self, meta, out, mpd):
        return ttnn.generic_op([meta, self.comb, out], mpd)
