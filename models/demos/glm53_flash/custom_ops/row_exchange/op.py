# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``RowExchange``: the hybrid MoE's row exchange as ONE generic op per direction with in-program fabric writes
(kernels in ``kernels/``; contract in README.md).

The hybrid placement spreads every MoE layer's experts over the mesh columns and their intermediate over the mesh
rows, so each row's users must reach every row before the experts (``gather``) and each chip's partial outputs must
return to the users' rows and be summed (``reduce``). ttnn's composites do that in 8 + 4 device ops per layer
(all_broadcast / concat / tilize / reduce_scatter / all_gather / selector matmul: 113 + 74 us at 4 users per row on
the 8x4, every op latency-bound). Here one program per direction does it:

* every chip's ``nchunks`` worker cores untilize their column chunk of the local tile row, cut the pieces (``T`` user
  rows each; gather: the chip's own users; reduce: the users of every other row), hand them to the exchange core;
* the exchange core sends every peer row its packets through the fabric (``np`` packets per peer of ``cpp`` pieces,
  a fused unicast write + atomic increment into the peer's receive staging and arrival semaphore), waits for the
  peers' packets, releases the workers;
* the workers assemble the ``[32, chunk]`` row-major block (row ``r * T + t`` = mesh row r's user t), tilize it and
  (reduce) multiply by the exact selector ``S[t, r * T + t] = 1`` in fp32 DST, so the row sum is one rounding.

The receive staging and the arrival semaphore are persistent per instance (one mesh allocation each, so the same
address on every chip): a peer may send the next layer's packets before this chip consumed the current ones of the
OTHER direction, so the two directions never share a staging, and within one direction the next use is ordered by
the exchange itself (a peer sends layer L+1's gather only after layer L's reduce completed, which needs this chip's
reduce packets, sent after this chip consumed layer L's gather). The fabric node ids and the routing direction of
every peer come from the mesh (``ttnn.get_eth_forwarding_direction``), one EDM connection per direction.
"""

from __future__ import annotations

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).resolve().parent / "kernels"
TILE = 32
TILE_BYTES = 2048
CT_MAX = 16  # tiles per chunk: the DST-sized untilize / tilize block
PKT_MAX = 12 * 1024  # fabric payload per packet (the router buffers carry 14 KB on Blackhole)
MODES = {"gather": 0, "reduce": 1}


def _ceil(a: int, b: int) -> int:
    return -(-a // b)


class RowExchange:
    """One direction of the row exchange over the mesh rows of ``mesh``: ``widths`` = the hidden widths of the
    streams exchanged together (gather: x and the routing matrix; reduce: the expert output), ``T`` users per row.
    ``__call__(inputs)`` returns the outputs ([1, 1, 32, W] bf16 TILE each, allocated on ``out_memory_config``)."""

    CB_SEND, CB_TIN, CB_RM, CB_RM2, CB_T, CB_OUT, CB_SEL, CB_ZERO = range(8)
    SEM_STAGED, SEM_GO = 0, 1

    def __init__(self, mesh, mode: str, widths: list[int], T: int, *, xchg_core=(0, 0), link: int = 0):
        self.mesh = mesh
        self.device = mesh.device
        self.mode = MODES[mode]
        self.rows, self.cols = mesh.rows, mesh.cols
        self.T = int(T)
        self.widths = [int(w) for w in widths]
        self.link = link
        assert self.rows > 1, "the row exchange needs several mesh rows"
        assert self.rows * self.T <= TILE, (self.rows, self.T, "the exchange assembles one tile row")
        assert all(w % TILE == 0 for w in self.widths), self.widths
        # chunks: (stream, first column tile, tiles)
        self.chunks: list[tuple[int, int, int]] = []
        for s, w in enumerate(self.widths):
            wt = w // TILE
            for c0 in range(0, wt, CT_MAX):
                self.chunks.append((s, c0, min(CT_MAX, wt - c0)))
        self.nchunks = len(self.chunks)
        self.ps = _ceil(max(self.T * ct * 64 for _, _, ct in self.chunks), 64) * 64  # piece slot bytes
        self.cpp = max(1, PKT_MAX // self.ps)  # pieces per packet
        self.np = _ceil(self.nchunks, self.cpp)  # packets per peer
        self.pkt = self.cpp * self.ps
        grid = self.device.compute_with_storage_grid_size()
        assert 1 + self.nchunks <= grid.x, (self.nchunks, grid.x, "one exchange core + the workers on grid row 0")
        self.xchg_core = ttnn.CoreCoord(*xchg_core)
        self.workers = [ttnn.CoreCoord(1 + k, 0) for k in range(self.nchunks)]
        self.all_cores = ttnn.CoreRangeSet(
            [ttnn.CoreRange(self.xchg_core, self.xchg_core)] + [ttnn.CoreRange(c, c) for c in self.workers]
        )
        self.xchg_set = ttnn.CoreRangeSet([ttnn.CoreRange(self.xchg_core, self.xchg_core)])
        self.worker_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in self.workers])
        # persistent per instance: the receive staging (one page per (source row, packet)) and the arrival semaphore
        self.recv = ttnn.allocate_tensor_on_device(
            ttnn.Shape([self.rows * self.np, self.pkt // 2]),
            ttnn.bfloat16,
            ttnn.ROW_MAJOR_LAYOUT,
            self.device,
            ttnn.DRAM_MEMORY_CONFIG,
        )
        self.arrived = ttnn.create_global_semaphore(self.device, self.xchg_set, 0)
        self.arrived_addr = int(ttnn.get_global_semaphore_address(self.arrived))
        self.send_bytes = (self.nchunks if self.mode == 0 else self.rows * self.nchunks) * self.ps
        # per device: the fabric peers (row, chip id, mesh id, direction group) and the groups' representative peers
        self.peers: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {}
        self.groups: dict[tuple[int, int], list[ttnn.MeshCoordinate]] = {}
        for r in range(self.rows):
            for c in range(self.cols):
                src = self.device.get_fabric_node_id(ttnn.MeshCoordinate(r, c))
                dirs: dict[int, int] = {}
                reps: list[ttnn.MeshCoordinate] = []
                peers = []
                for r2 in range(self.rows):
                    if r2 == r:
                        continue
                    coord = ttnn.MeshCoordinate(r2, c)
                    dst = self.device.get_fabric_node_id(coord)
                    d = ttnn.get_eth_forwarding_direction(src, dst)
                    assert d is not None, (r, c, r2, "no fabric route")
                    if d not in dirs:
                        dirs[d] = len(reps)
                        reps.append(coord)
                    peers.append((r2, int(dst.chip_id), int(dst.mesh_id), dirs[d]))
                assert len(reps) <= 4, reps
                self.peers[(r, c)] = peers
                self.groups[(r, c)] = reps

    @staticmethod
    def _noc(device, core) -> int:
        n = device.worker_core_from_logical_core(core)
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

        block = CT_MAX * TILE_BYTES
        return [
            cb(self.CB_SEND, self.all_cores, self.send_bytes, self.ps),  # first everywhere: one address
            cb(self.CB_TIN, self.worker_set, block, TILE_BYTES),
            cb(self.CB_RM, self.worker_set, block, TILE_BYTES),
            cb(self.CB_RM2, self.worker_set, block, TILE_BYTES),
            cb(self.CB_T, self.worker_set, block, TILE_BYTES),
            cb(self.CB_OUT, self.worker_set, block, TILE_BYTES),
            cb(self.CB_SEL, self.worker_set, TILE_BYTES, TILE_BYTES),
            cb(self.CB_ZERO, self.worker_set, self.ps, self.ps),
        ]

    def _program(self, coord: tuple[int, int], inputs, outputs) -> ttnn.ProgramDescriptor:
        r, c = coord
        T, rows = self.T, self.rows
        recv_ct = ttnn.TensorAccessorArgs(self.recv).get_compile_time_args()
        in_ct = ttnn.TensorAccessorArgs(inputs[0]).get_compile_time_args()
        out_ct = ttnn.TensorAccessorArgs(outputs[0]).get_compile_time_args()
        for t in inputs[1:]:
            assert ttnn.TensorAccessorArgs(t).get_compile_time_args() == in_ct, "the inputs share one placement"
        for t in outputs[1:]:
            assert ttnn.TensorAccessorArgs(t).get_compile_time_args() == out_ct, "the outputs share one placement"
        xchg_noc = self._noc(self.device, self.xchg_core)
        kernels = []
        by_ct: dict[int, list[int]] = {}
        for k, (_, _, ct) in enumerate(self.chunks):
            by_ct.setdefault(ct, []).append(k)
        for ct, ks in by_ct.items():
            cores = ttnn.CoreRangeSet([ttnn.CoreRange(self.workers[k], self.workers[k]) for k in ks])
            reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
            for k in ks:
                s, c0, _ = self.chunks[k]
                core = self.workers[k]
                reader_rt[core.x][core.y] = [inputs[s].buffer_address(), c0]
                writer_rt[core.x][core.y] = [
                    outputs[s].buffer_address(),
                    c0,
                    k,
                    r,
                    xchg_noc,
                    self.recv.buffer_address(),
                ]
                compute_rt[core.x][core.y] = []
            cb_src = self.CB_OUT if self.mode == 1 else self.CB_T
            kernels += [
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "reader_row_exchange.cpp"),
                    core_ranges=cores,
                    compile_time_args=[self.mode, self.CB_TIN, self.CB_SEL, self.CB_ZERO, ct, T, rows, self.ps] + in_ct,
                    runtime_args=reader_rt,
                    config=ttnn.ReaderConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "writer_row_exchange.cpp"),
                    core_ranges=cores,
                    compile_time_args=[
                        self.mode,
                        self.CB_SEND,
                        self.CB_RM,
                        self.CB_RM2,
                        cb_src,
                        self.CB_ZERO,
                        ct,
                        T,
                        rows,
                        self.nchunks,
                        self.ps,
                        self.cpp,
                        self.np,
                        self.SEM_GO,
                        self.SEM_STAGED,
                    ]
                    + out_ct
                    + recv_ct,
                    runtime_args=writer_rt,
                    config=ttnn.WriterConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "compute_row_exchange.cpp"),
                    core_ranges=cores,
                    compile_time_args=[
                        self.mode,
                        self.CB_TIN,
                        self.CB_RM,
                        self.CB_RM2,
                        self.CB_T,
                        self.CB_SEL,
                        self.CB_OUT,
                        ct,
                    ],
                    runtime_args=compute_rt,
                    config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True),
                ),
            ]
        # the exchange core: per-device runtime args (its peers' node ids, then the connection args appended below)
        peers = self.peers[coord]
        xchg_rt = ttnn.RuntimeArgs()
        args = [r, xchg_noc, self.recv.buffer_address(), self.arrived_addr]
        args += [self._noc(self.device, w) for w in self.workers]
        args += [len(peers)]
        for row2, dev, mesh_id, grp in peers:
            args += [row2, dev, mesh_id, grp]
        args += [len(self.groups[coord])]
        xchg_rt[self.xchg_core.x][self.xchg_core.y] = args
        xchg_kernel = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "xchg_row_exchange.cpp"),
            core_ranges=self.xchg_set,
            compile_time_args=[
                self.mode,
                self.CB_SEND,
                self.SEM_STAGED,
                self.SEM_GO,
                self.nchunks,
                self.nchunks,
                self.ps,
                self.cpp,
                self.np,
                rows,
            ]
            + recv_ct,
            runtime_args=xchg_rt,
            config=ttnn.WriterConfigDescriptor(),
        )
        kernels.append(xchg_kernel)
        sems = [
            ttnn.SemaphoreDescriptor(self.SEM_STAGED, ttnn.CoreType.WORKER, self.all_cores, 0),
            ttnn.SemaphoreDescriptor(self.SEM_GO, ttnn.CoreType.WORKER, self.all_cores, 0),
        ]
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=self._cbs())
        # the fabric connections (one per routing direction) append their flow-control semaphores to the program
        src = self.device.get_fabric_node_id(ttnn.MeshCoordinate(r, c))
        conn = []
        for rep in self.groups[coord]:
            dst = self.device.get_fabric_node_id(rep)
            conn += list(ttnn.setup_fabric_connection(src, dst, self.link, program, self.xchg_core))
        xk = program.kernels[len(kernels) - 1]
        full = list(xk.runtime_args[self.xchg_core.x][self.xchg_core.y]) + conn
        xk.runtime_args[self.xchg_core.x][self.xchg_core.y] = full
        assert list(program.kernels[len(kernels) - 1].runtime_args[self.xchg_core.x][self.xchg_core.y]) == full
        return program

    def allocate_outputs(self, memory_config=ttnn.DRAM_MEMORY_CONFIG) -> list[ttnn.Tensor]:
        return [
            ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, 1, TILE, w]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.device, memory_config
            )
            for w in self.widths
        ]

    def __call__(self, inputs: list[ttnn.Tensor], outputs: list[ttnn.Tensor] | None = None, memory_config=None):
        assert len(inputs) == len(self.widths), (len(inputs), self.widths)
        for t, w in zip(inputs, self.widths):
            assert tuple(t.padded_shape)[-2:] == (TILE, w), (tuple(t.padded_shape), w)
        if outputs is None:
            outputs = self.allocate_outputs(memory_config or ttnn.DRAM_MEMORY_CONFIG)
        mpd = ttnn.MeshProgramDescriptor()
        for r in range(self.rows):
            for c in range(self.cols):
                coord = ttnn.MeshCoordinate(r, c)
                mpd[ttnn.MeshCoordinateRange(coord, coord)] = self._program((r, c), inputs, outputs)
        ttnn.generic_op(list(inputs) + list(outputs) + [self.recv], mpd)
        return outputs
