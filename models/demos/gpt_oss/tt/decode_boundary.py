# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Decode layer boundaries of the fused gpt-oss decode layer: the TP all-reduce, the residual add and the next RMSNorm
as one op on one core per device (kernels/boundary_*.cpp).

Inter-layer residual contract (decode, one token, TP over the mesh row):
  * the residual stream is replicated on every device as a *flat* BF16 vector: hidden value h at byte 2 h of a
    [1, 1, 32, 32 * flat_tiles] BF16 tile tensor on one core (BOUNDARY_CORE), zero padded past hidden (3 pages, 6 KB,
    for hidden 2880). Its tile structure is only a container: every op on it is elementwise or a whole-vector sum,
    and every reader / writer addresses it by byte offset;
  * each boundary also produces the next RMSNorm's output in the same flat form; the streamed projections read it in
    one contiguous read (its 32-value groups are the 1x32 activation tiles of experts/stream.py);
  * the row-parallel projections (o_proj, MoE down) write their per-device partial sums into a persistent flat
    partial buffer, which the boundary all-reduces.
So the only collective between two decoder layers is the boundary's own all-reduce of the 6 KB partial: there is no
gather, reshard or layout conversion of the residual. The stack's first boundary turns the embedding row into this
contract (`entry`); after the last layer `to_rows` returns the final-normed hidden in row 0 of the [1, 1, 32, hidden]
width-sharded layout the LM-head path reads.

All-reduce: every device NOC-copies its partial into slot `ring index` of the site's receive buffer and multicasts it
over the fabric into the same slot on every other device of the ring (fused write + semaphore increment); the
boundary then sums the slots in slot order (bit-identical residual on every device), adds the residual and applies
the RMSNorm (FP32 statistics, HiFi2). With fused_decode.DECODE_BOUNDARY_FUSED_SEND the sender runs inside the
producing o_proj / MoE down op (sending_program), so the transfer starts as soon as the partial is complete.
"""

import struct
from pathlib import Path

import torch

import ttnn

from .fused_decode import DECODE_BOUNDARY_CCL, DECODE_BOUNDARY_LINKS, residual_memory_config

KERNEL_DIR = Path(__file__).parent / "experts" / "kernels"
TILE = ttnn.TILE_SIZE
PAGE = TILE * TILE * 2


def _f32_bits(v):
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def flat_tiles(hidden):
    return -(-hidden // (TILE * TILE))


# The boundary core holds the flat residual, partial sum, receive buffers and normed output, and runs the boundary op
# and the producers' fabric sender. It must not be a streamed-matmul worker core (those sit next to the DRAM banks,
# columns 0-2 and 6-8), so the sender can join the producer programs; (0, 1), (1, 1), (0, 2), (3, 0) and (4, 0)
# measured alike.
BOUNDARY_CORE = (0, 1)


def flat_memory_config(mesh_device, hidden, pages=None):
    """[1, 1, 32, 32 * pages] BF16 tile tensor on the boundary core: `pages` 2 KB pages, contiguous."""
    pages = pages or flat_tiles(hidden)
    core = ttnn.CoreCoord(*BOUNDARY_CORE)
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet([ttnn.CoreRange(core, core)]), (TILE, pages * TILE), ttnn.ShardOrientation.ROW_MAJOR
        ),
    )


def flat_shape(hidden, pages=None):
    return [1, 1, TILE, (pages or flat_tiles(hidden)) * TILE]


def flat_to_tile_rows(vec, pages):
    """Host helper: a flat vector -> the [32, 32 * pages] matrix whose TILE-layout device bytes are the vector."""
    flat = torch.zeros(pages * TILE * TILE, dtype=vec.dtype)
    flat[: vec.numel()] = vec.reshape(-1)
    t = flat.reshape(pages, 2, 2, 16, 16)  # page, face row, face col, row, col
    return t.permute(1, 3, 0, 2, 4).reshape(TILE, pages * TILE)


def tile_rows_to_flat(mat, hidden):
    """Host helper: inverse of flat_to_tile_rows (device tensor read back with to_torch -> flat vector)."""
    pages = mat.shape[-1] // TILE
    t = mat.reshape(2, 16, pages, 2, 16).permute(2, 0, 3, 1, 4)
    return t.reshape(-1)[:hidden]


class PendingBoundary:
    """A layer boundary whose all-reduce partial the producer op already sent (or an entry boundary), executed by the
    op that consumes its normed output (DecodeBoundary.consumer_parts) or on its own (run). residual_out / x are set
    once it is scheduled."""

    def __init__(self, boundary, residual, gamma, partial=None, site=None, entry=False, sent=False):
        self.boundary = boundary
        self.residual, self.gamma, self.partial, self.site, self.entry = residual, gamma, partial, site, entry
        self.sent = sent  # the producer op already sent the partial (DecodeBoundary.sending_program)
        self.residual_out = self.x = None

    def run(self):
        self.residual_out, self.x = self.boundary(
            self.residual, self.gamma, partial=self.partial, site=self.site, entry=self.entry, sent=self.sent
        )
        return self.residual_out, self.x


class DecodeBoundary:
    """All-reduce of the flat partial sums + residual add + RMSNorm (see module doc).

    One persistent receive buffer (`slots` x FLAT_TILES pages) and receive semaphore per call site ("attn", "moe"),
    shared by every layer: every boundary waits for every device's partial, so a device cannot write layer i+1's
    partial into a slot before every device has consumed layer i's (the two sites alternate)."""

    def __init__(self, mesh_device, hidden, eps, cluster_axis, sites=("attn", "moe"), ccl=DECODE_BOUNDARY_CCL):
        assert ccl in ("fabric", "ttnn"), ccl
        self.ccl = ccl
        self.mesh_device = mesh_device
        self.hidden = hidden
        self.eps = eps
        self.cluster_axis = cluster_axis
        self.tiles = flat_tiles(hidden)
        self.ring = mesh_device.shape[cluster_axis]
        self.fwd_hops = self.ring // 2
        self.bwd_hops = self.ring - 1 - self.fwd_hops
        self.memory_config = flat_memory_config(mesh_device, hidden)
        self.core = self.memory_config.shard_spec.grid.bounding_box().start
        self.cores = ttnn.CoreRangeSet([ttnn.CoreRange(self.core, self.core)])
        grid = mesh_device.compute_with_storage_grid_size()
        all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
        recv_config = flat_memory_config(mesh_device, hidden, self.ring * self.tiles)
        self.sites = {
            name: (
                ttnn.empty(
                    flat_shape(hidden, self.ring * self.tiles),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh_device,
                    memory_config=recv_config,
                ),
                ttnn.create_global_semaphore(mesh_device, all_cores, 0),
            )
            for name in sites
        }
        if ccl == "ttnn":
            # ttnn.experimental.all_reduce_async on the flat partial (elementwise, so layout-agnostic): one persistent
            # scratch + semaphore per site, ring, 1 link (a 1-core payload; see the optimized-decoder stage).
            scratch_config = flat_memory_config(mesh_device, hidden, self.ring * self.tiles)
            self.ar_sites = {
                name: (
                    ttnn.empty(
                        flat_shape(hidden, self.ring * self.tiles),
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=mesh_device,
                        memory_config=scratch_config,
                    ),
                    ttnn.create_global_semaphore(mesh_device, all_cores, 0),
                )
                for name in sites
            }
        self.payload = self.tiles * PAGE
        max_payload = int(ttnn.get_tt_fabric_max_payload_size_bytes())
        self.chunk = min(self.payload, max_payload // PAGE * PAGE)
        self.chunks = -(-self.payload // self.chunk)
        phys = mesh_device.worker_core_from_logical_core(self.core)
        self.noc_xy = (phys.x, phys.y)
        self.coords = [
            ttnn.MeshCoordinate(r, c) for r in range(mesh_device.shape[0]) for c in range(mesh_device.shape[1])
        ]
        self.residual_rows_config = residual_memory_config(mesh_device, hidden)

    def flat_gamma(self, norm):
        """The RMSNorm weight as a flat BF16 [1, 1, 1, 32 * 32 * FLAT_TILES] row-major DRAM tensor (one page)."""
        w = ttnn.to_torch(ttnn.get_device_tensors(norm.tt_weight)[0]).reshape(-1).float()
        flat = torch.zeros(1, 1, 1, self.tiles * TILE * TILE)
        flat[..., : self.hidden] = w[: self.hidden]
        return ttnn.from_torch(
            flat,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def _empty_flat(self):
        return ttnn.empty(
            flat_shape(self.hidden),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=self.memory_config,
        )

    def _neighbor(self, coord, step):
        r, c = coord[0], coord[1]
        if self.cluster_axis == 1:
            c = (c + step) % self.ring
        else:
            r = (r + step) % self.ring
        return ttnn.MeshCoordinate(r, c)

    def _cb(self, index, size, dtype=ttnn.bfloat16, tensor=None):
        fmt = [
            ttnn.CBFormatDescriptor(
                buffer_index=index, data_format=dtype, page_size=PAGE if dtype == ttnn.bfloat16 else 4096
            )
        ]
        if tensor is not None:
            cb = ttnn.cb_descriptor_from_sharded_tensor(index, tensor)
            cb.total_size = size
            cb.format_descriptors = fmt
            return cb
        return ttnn.CBDescriptor(total_size=size, core_ranges=self.cores, format_descriptors=fmt)

    def _sender_kernel(self, coord, partial, site, producers=0, sem_id=0, link=0, links=1):
        """The fabric sender of `partial` into slot `ring index` of `site` on every device (kernels/boundary_sender.cpp);
        producers > 0: first wait until that many producer cores incremented program semaphore `sem_id`. With
        `links` > 1, sender `link` (RISCV_0 for link 0, RISCV_1 for link 1) sends every links-th packet."""
        recv, sem = self.sites[site]
        rt = ttnn.RuntimeArgs()
        rt[self.core.x][self.core.y] = [
            partial.buffer_address(),
            recv.buffer_address() + coord[self.cluster_axis] * self.payload,
            self.noc_xy[0],
            self.noc_xy[1],
            ttnn.get_global_semaphore_address(sem),
        ]
        return ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "boundary_sender.cpp"),
            core_ranges=self.cores,
            compile_time_args=[self.payload, self.chunk, self.fwd_hops, self.bwd_hops, producers, sem_id]
            + [link, links, 1 if link == 0 else 0],
            runtime_args=rt,
            config=ttnn.DataMovementConfigDescriptor(
                processor=(ttnn.DataMovementProcessor.RISCV_0, ttnn.DataMovementProcessor.RISCV_1)[link],
                noc=(ttnn.NOC.NOC_0, ttnn.NOC.NOC_1)[link],
            ),
        )

    def _add_sender(self, coord, program, sender, link=0):
        """Append the sender kernel to `program` with its forward / backward fabric connection arguments (`link`)."""
        program.kernels = list(program.kernels) + [sender]
        src = self.mesh_device.get_fabric_node_id(coord)
        fabric_args = []
        for step, hops in ((1, self.fwd_hops), (-1, self.bwd_hops)):
            fabric_args.append(1 if hops > 0 else 0)
            if hops > 0:
                fabric_args += ttnn.setup_fabric_connection(
                    src_fabric_node_id=src,
                    dst_fabric_node_id=self.mesh_device.get_fabric_node_id(self._neighbor(coord, step)),
                    link_idx=link,
                    program_descriptor=program,
                    worker_core=self.core,
                )
        program.kernels[len(program.kernels) - 1].runtime_args[self.core.x][self.core.y].extend(fabric_args)

    def sending_program(self, build, partial, site, producer_cores):
        """Fused matmul + all-reduce send: per device, the producer op's program (`build()`, whose `producer_cores`
        write the flat partial and then increment program semaphore 0 on the boundary core, see notify_args) plus the
        boundary sender, which opens its fabric connections while the producers run and multicasts the partial as
        soon as the last one is done. The layer's boundary op is then called with sent=True."""
        mesh_program = ttnn.MeshProgramDescriptor()
        n = producer_cores.num_cores()
        for coord in self.coords:
            program = build()
            assert not program.semaphores, "the producer program's semaphore 0 is the notification semaphore"
            program.semaphores = [
                ttnn.SemaphoreDescriptor(0, ttnn.CoreType.WORKER, producer_cores.merge(self.cores), 0)
            ]
            links = min(DECODE_BOUNDARY_LINKS, self.chunks)
            for link in range(links):
                sender = self._sender_kernel(coord, partial, site, producers=n, sem_id=0, link=link, links=links)
                self._add_sender(coord, program, sender, link)
            mesh_program[ttnn.MeshCoordinateRange(coord, coord)] = program
        return mesh_program

    def notify_args(self):
        """Producer writer compile-time args: [notify, boundary core NOC x, y, semaphore id]."""
        return [1, self.noc_xy[0], self.noc_xy[1], 0]

    def _parts(self, residual, gamma, partial, site, entry, norm, sent):
        """The boundary-core part of a boundary: (per-device kernel list builder, CBs, io tensors, residual', x)."""
        reduced = None
        if partial is not None and self.ccl == "ttnn":
            scratch, ar_sem = self.ar_sites[site]
            reduced = ttnn.experimental.all_reduce_async(
                partial,
                scratch,
                cluster_axis=self.cluster_axis,
                mesh_device=self.mesh_device,
                multi_device_global_semaphore=ar_sem,
                memory_config=self.memory_config,
                dtype=ttnn.bfloat16,
                topology=ttnn.Topology.Ring,
                num_links=1,
            )
            partial = None
        ccl = partial is not None
        out_res = self._empty_flat()
        out_x = self._empty_flat() if norm else None
        recv, sem = self.sites[site] if ccl else (reduced, None)
        sem_addr = ttnn.get_global_semaphore_address(sem) if ccl else 0
        slots = self.ring if ccl else (1 if reduced is not None else 0)
        expected = (self.ring - 1) * self.chunks + 1 if ccl else 0
        # CB indices clear of the streamed-matmul ops' (0-4, 16), which share programs with this part (consumer_parts).
        cb_res, cb_recv, cb_gamma, cb_scaler, cb_res_out, cb_sq, cb_rs, cb_x, cb_y = 8, 9, 10, 11, 24, 25, 26, 27, 28
        T = self.tiles
        cbs = [
            self._cb(cb_res, T * PAGE, tensor=None if entry else residual),
            self._cb(cb_res_out, T * PAGE, tensor=out_res),
        ]
        if slots:
            cbs.append(self._cb(cb_recv, slots * T * PAGE, tensor=recv))
        if norm:
            cbs += [
                self._cb(cb_gamma, T * PAGE),
                self._cb(cb_scaler, PAGE),
                self._cb(cb_sq, 4096, ttnn.float32),
                self._cb(cb_rs, 4096, ttnn.float32),
                self._cb(cb_x, T * PAGE, tensor=out_x),
                self._cb(cb_y, T * 4096, ttnn.float32),
            ]
        gamma_t = gamma if norm else out_res
        reader_ct = [cb_res, cb_recv, cb_gamma, cb_scaler, T, slots, expected]
        reader_ct += [1 if entry else 0, self.hidden * 2, 1 if norm else 0]
        reader_ct += ttnn.TensorAccessorArgs(gamma_t).get_compile_time_args()
        reader_ct += ttnn.TensorAccessorArgs(residual).get_compile_time_args()
        compute_ct = [cb_res, cb_recv, cb_gamma, cb_scaler, cb_res_out, cb_sq, cb_rs, cb_x, T, slots]
        compute_ct += [_f32_bits(1.0 / self.hidden), _f32_bits(self.eps), 1 if norm else 0, cb_y]
        reader_rt = ttnn.RuntimeArgs()
        reader_rt[self.core.x][self.core.y] = [gamma_t.buffer_address(), sem_addr, residual.buffer_address()]

        def kernels(coord):
            ks = [
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "boundary_reader.cpp"),
                    core_ranges=self.cores,
                    compile_time_args=reader_ct,
                    runtime_args=reader_rt,
                    config=ttnn.DataMovementConfigDescriptor(
                        processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_1
                    ),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "boundary_compute.cpp"),
                    core_ranges=self.cores,
                    compile_time_args=compute_ct,
                    runtime_args=[],
                    config=ttnn.ComputeConfigDescriptor(
                        math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, math_approx_mode=False
                    ),
                ),
            ]
            return ks

        io = [residual, out_res] + ([out_x, gamma] if norm else [])
        if ccl:
            io += [partial, recv]
        elif reduced is not None:
            io.append(reduced)
        return kernels, cbs, io, out_res, out_x, (ccl and not sent), reduced, cb_x

    def __call__(self, residual, gamma, partial=None, site=None, entry=False, norm=True, sent=False):
        """residual: the flat residual (or, entry=True, the [1, 1, 1, hidden] BF16 row-major embedding); partial: the
        flat partial sum to all-reduce (None: no all-reduce); sent: the producer op already sent it (sending_program);
        gamma: flat_gamma(norm) (None with norm=False). Returns (residual', x), x = None when norm=False."""
        kernels, cbs, io, out_res, out_x, send, reduced, _ = self._parts(
            residual, gamma, partial, site, entry, norm, sent
        )
        mesh_program = ttnn.MeshProgramDescriptor()
        for coord in self.coords:
            program = ttnn.ProgramDescriptor(kernels=kernels(coord), semaphores=[], cbs=cbs)
            if send:
                self._add_sender(coord, program, self._sender_kernel(coord, partial, site))
            mesh_program[ttnn.MeshCoordinateRange(coord, coord)] = program
        ttnn.generic_op(io, mesh_program)
        if reduced is not None:
            reduced.deallocate(True)
        return out_res, out_x

    def consumer_parts(self, pending, consumer_cores):
        """Boundary fused into the program of the op that consumes its normed output (`pending`, a PendingBoundary whose
        partial the producer op already sent, or an entry boundary): the boundary core's reader and compute plus a
        notifier (kernels/boundary_notify.cpp) that, once x is packed, increments program semaphore 0 on every
        `consumer_cores` core; the consumer's writers wait for it before reading x (see wait_args). The consumer's
        weight readers meanwhile stream ahead into their circular buffers. Returns (kernels, cbs, semaphores, io) to
        add to the consumer program and sets pending.residual / pending.x."""
        kernels, cbs, io, out_res, out_x, send, reduced, cb_x = self._parts(
            pending.residual, pending.gamma, pending.partial, pending.site, pending.entry, True, True
        )
        assert not send and reduced is None, "consumer fusion needs the producer-sent fabric all-reduce"
        rt = ttnn.RuntimeArgs()
        consumer_list = ttnn.corerange_to_cores(consumer_cores, row_wise=True)
        noc = [self.mesh_device.worker_core_from_logical_core(c) for c in consumer_list]
        rt[self.core.x][self.core.y] = [v for p in noc for v in (p.x, p.y)]
        notifier = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "boundary_notify.cpp"),
            core_ranges=self.cores,
            compile_time_args=[cb_x, self.tiles, len(noc), 0],
            runtime_args=rt,
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0),
        )
        semaphores = [ttnn.SemaphoreDescriptor(0, ttnn.CoreType.WORKER, consumer_cores.merge(self.cores), 0)]
        pending.residual_out, pending.x = out_res, out_x
        return kernels(self.coords[0]) + [notifier], cbs, semaphores, io

    @staticmethod
    def wait_args(pending):
        """Consumer writer compile-time args: [wait for the fused boundary's x (semaphore 0)]."""
        return [1 if pending is not None else 0]

    def to_rows(self, x):
        """Flat hidden -> BF16 [1, 1, 32, hidden] in the width-sharded residual layout (row 0 = the vector, rows 1..31
        zero), the input layout of the fused decode LM-head path (kernels/boundary_to_rows.cpp)."""
        out = ttnn.empty(
            [1, 1, TILE, self.hidden],
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=self.residual_rows_config,
        )
        cores = self.residual_rows_config.shard_spec.grid
        core_list = ttnn.corerange_to_cores(cores, row_wise=True)
        shard_tiles = self.residual_rows_config.shard_spec.shape[1] // TILE
        cb_out = 16
        cb = ttnn.cb_descriptor_from_sharded_tensor(cb_out, out)
        rt = ttnn.RuntimeArgs()
        for c, core in enumerate(core_list):
            rt[core.x][core.y] = [x.buffer_address(), c * shard_tiles]
        ct = [cb_out, shard_tiles, self.noc_xy[0], self.noc_xy[1]]
        kernel = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "boundary_to_rows.cpp"),
            core_ranges=cores,
            compile_time_args=ct,
            runtime_args=rt,
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0),
        )
        program = ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[cb])
        return ttnn.generic_op([x, out], program)
