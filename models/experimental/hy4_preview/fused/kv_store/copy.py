# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""``kv_copy``: move frames between two cache tensors, or inside one, without leaving the chip or the DRAM bank
(KV_STORE.md 4, 3b: the conversation's prefix into the prefill stack's contiguous workbench slot before a prefill,
the new rows back into the pool afterwards; also a fork inside the pool).

One program, the same on every chip of a mesh, no fabric: ``lanes`` copier cores per DRAM bank; on each the two
data-movement processors run ``kernels/kv_copy_reader.cpp`` (DRAM -> L1, NOC0) and ``kernels/kv_copy_writer.cpp``
(L1 -> DRAM, NOC1) through a circular buffer. What to copy is the PLAN, a device tensor of (source frame,
destination frame) pairs the host rewrites before a launch: a traced program follows it. One launch copies every
pair for every kind given (latent rows: 78 slots of 22,528 B per frame and bank; index keys: 21 of 4,352 B).

    copy = KvCopy(mesh_device, [(pool_latent, Geometry.pool(LATENT), bench_latent, Geometry.contiguous(LATENT, 100)),
                                (pool_index, Geometry.pool(INDEX), bench_index, Geometry.contiguous(INDEX, 100))])
    copy.set_pairs(store.copy_in(key, restart))     # [(pool frame, workbench frame), ...]
    copy()

A frame must not be both a source and a destination of one plan.
"""

from __future__ import annotations

import torch

import ttnn
from models.experimental.hy4_preview.fused.kv_store import frames as fr
from models.experimental.hy4_preview.fused.kv_store.device import INCLUDE, KERNEL_DIR, check_layout, core_set

PLAN_WORDS = 512  # kernels/kv_copy.hpp
PLAN_HEADER = 2
MAX_PAIRS = (PLAN_WORDS - PLAN_HEADER) // 2
MAX_KINDS = 2
NOC_PACKET = 16384
MODE_NO_WRITE, MODE_NO_READ, MODE_RETURN = 1, 2, 4
CB_PLAN_READER, CB_PLAN_WRITER, CB_DATA0 = 0, 1, 2


def grid_cores(lanes: int):
    """Lane x of bank y on logical core (x, y): one rectangle, the cheapest launch."""
    return {bank: [(lane, bank) for lane in range(lanes)] for bank in range(fr.BANKS)}


def row_cores(mesh_device, lanes: int):
    """A bank's lanes on the worker row of the bank's NOC0 endpoint, nearest first
    (models/demos/gpt_oss/tt/fused_decode/stream.py row_placement, the placement of the fastest DRAM streams)."""
    from models.demos.gpt_oss.tt.fused_decode import stream

    placed = stream.row_placement(mesh_device, lanes)
    return {bank: [(c.x, c.y) for c in placed[bank]] for bank in range(fr.BANKS)}


def bank_workers(mesh_device, coord):
    """[(x, y)] per DRAM bank: the worker core nearest the bank's NOC0 endpoint on chip ``coord``. The chips of a
    Galaxy differ in it (the right-hand banks' column is 6 on some and 7 on others)."""
    a = mesh_device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0, ttnn.MeshCoordinate(*coord))
    workers = [a[b] for b in sorted(a)] if isinstance(a, dict) else list(a)
    return [(w.x, w.y) for w in workers]


def own_row_cores(mesh_device, coord, lanes: int):
    """Placement "own": as "rows", but from THIS chip's bank workers: a bank's lanes on its worker's row, nearest
    its worker first, inside its segment of the row (from its worker to the next bank's worker on that row; the
    first bank of a row starts at column 0), then what the row has left."""
    grid = mesh_device.compute_with_storage_grid_size()
    workers = bank_workers(mesh_device, coord)
    used, out = set(), {b: [] for b in range(fr.BANKS)}
    segments = {}
    for b, (wx, wy) in enumerate(workers):
        mates = sorted(x for x, y in workers if y == wy)
        k = mates.index(wx)
        segments[b] = (wx if k else 0, mates[k + 1] if k + 1 < len(mates) else grid.x)
    for phase in range(2):
        for b, (wx, wy) in enumerate(workers):
            lo, hi = segments[b] if phase == 0 else (0, grid.x)
            for x in sorted(range(lo, hi), key=lambda x: (abs(x - wx), x)):
                if len(out[b]) < lanes and (x, wy) not in used:
                    used.add((x, wy))
                    out[b].append((x, wy))
    assert all(
        len(v) == lanes for v in out.values()
    ), f"chip {coord}: no room for {lanes} lanes per bank on the banks' rows"
    return out


class KvCopy:
    def __init__(
        self,
        mesh_device,
        kinds,
        lanes: int = 4,
        depth: int = 2,
        write_batch: int = 1,
        placement: str = "rows",
        mode: int = 0,
        coords=None,
    ):
        """``kinds``: up to two (source tensor, source geometry, destination tensor, destination geometry).
        The defaults are the fastest measured on one chip (tests/test_kv_copy.py test_copy_rate: 4 lanes per bank
        on the bank's own worker row, 193 to 198 GB/s per chip; 8 lanes on the same rows 127, a 4 x 8 grid 112).
        ``placement``: "grid" | "rows" (one placement for every chip, from the mesh's reference chip) | "own"
        (each chip's from its own bank workers: one program per chip). ``coords``: the chips that run it (default
        all)."""
        assert 1 <= len(kinds) <= MAX_KINDS and 1 <= depth <= 8 and write_batch >= 1
        self.mesh_device, self.kinds, self.lanes, self.depth, self.write_batch = (
            mesh_device,
            list(kinds),
            lanes,
            depth,
            write_batch,
        )
        self.mode = mode
        grid = mesh_device.compute_with_storage_grid_size()
        rows, cols = tuple(mesh_device.shape)
        self.coords = (
            [(r, c) for r in range(rows) for c in range(cols)] if coords is None else [tuple(c) for c in coords]
        )
        self.placement = placement
        if placement == "own":
            self.cores_of = {coord: own_row_cores(mesh_device, coord, lanes) for coord in self.coords}
        else:
            shared = {"grid": lambda: grid_cores(lanes), "rows": lambda: row_cores(mesh_device, lanes)}[placement]()
            self.cores_of = {coord: shared for coord in self.coords}
        for cores in self.cores_of.values():
            flat = [c for bank in range(fr.BANKS) for c in cores[bank]]
            assert len(set(flat)) == lanes * fr.BANKS and all(x < grid.x and y < grid.y for x, y in flat), flat
        for src, sg, dst, dg in self.kinds:
            assert sg.kind == dg.kind
            assert sg.kind.shard_bytes % self.packets(sg.kind) == 0
        self.plan = ttnn.from_torch(
            torch.zeros(1, 1, 1, PLAN_WORDS, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        assert self.plan.buffer_aligned_page_size() == PLAN_WORDS * 4
        self.pairs = []
        self._descriptor = None

    @staticmethod
    def packets(kind: fr.Kind) -> int:
        """NoC packets a shard is cut into: the fewest of equal size, each a multiple of 64 bytes."""
        n = -(-kind.shard_bytes // NOC_PACKET)
        while kind.shard_bytes % n or (kind.shard_bytes // n) % 64:
            n += 1
        return n

    def check(self, src_frames: int, dst_frames: int) -> None:
        """Layouts of the tensors, given how many frames (pool) or users (cache) each holds."""
        for src, sg, dst, dg in self.kinds:
            check_layout(src, sg, src_frames)
            check_layout(dst, dg, dst_frames)

    def set_pairs(self, pairs) -> None:
        """(source frame, destination frame) pairs of the next launches; frame numbers as each tensor's geometry
        counts them (``Geometry.frame_base``)."""
        pairs = [(int(s), int(d)) for s, d in pairs]
        assert len(pairs) <= MAX_PAIRS, f"{len(pairs)} pairs: at most {MAX_PAIRS} per launch"
        same = any(src.buffer_address() == dst.buffer_address() for src, _, dst, _ in self.kinds)
        if same:
            assert not {s for s, _ in pairs} & {d for _, d in pairs}, "a frame is both read and written"
        assert len({d for _, d in pairs}) == len(pairs), "a destination frame appears twice"
        host = torch.zeros(PLAN_WORDS, dtype=torch.int32)
        host[0] = len(pairs)
        if pairs:
            host[PLAN_HEADER : PLAN_HEADER + 2 * len(pairs)] = torch.tensor(pairs, dtype=torch.int32).reshape(-1)
        t = ttnn.from_torch(host.reshape(1, 1, 1, PLAN_WORDS), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.copy_host_to_device_tensor(t, self.plan)
        self.pairs = pairs

    def bytes_per_chip(self, pairs: int | None = None) -> int:
        n = len(self.pairs) if pairs is None else pairs
        return n * sum(sg.kind.frame_bytes_per_chip for _, sg, _, _ in self.kinds)

    def program(self, coord=None):
        placed = self.cores_of[self.coords[0] if coord is None else coord]
        flat = [c for bank in range(fr.BANKS) for c in placed[bank]]
        capacity = self.depth + self.write_batch
        named = [
            ("kinds", len(self.kinds)),
            ("depth", self.depth),
            ("write_batch", self.write_batch),
            ("capacity", capacity),
            ("mode", self.mode),
            ("cb_plan_reader", CB_PLAN_READER),
            ("cb_plan_writer", CB_PLAN_WRITER),
            ("cb_data0", CB_DATA0),
            ("cb_data1", CB_DATA0 + 1),
        ]
        src_addr, dst_addr = [0] * MAX_KINDS, [0] * MAX_KINDS
        cores = core_set(flat)

        def cb(index, page, pages):
            return ttnn.CBDescriptor(
                total_size=page * pages,
                core_ranges=cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.uint8, page_size=page)
                ],
            )

        cbs = [cb(CB_PLAN_READER, PLAN_WORDS * 4, 1), cb(CB_PLAN_WRITER, PLAN_WORDS * 4, 1)]
        for k in range(MAX_KINDS):
            if k < len(self.kinds):
                src, sg, dst, dg = self.kinds[k]
                kind = sg.kind
                packets = self.packets(kind)
                values = [
                    kind.shard_bytes,
                    packets,
                    kind.layers,
                    sg.frame_base(1),
                    sg.slot_bytes(1),
                    dg.frame_base(1),
                    dg.slot_bytes(1),
                ]
                src_addr[k], dst_addr[k] = src.buffer_address(), dst.buffer_address()
                cbs.append(cb(CB_DATA0 + k, kind.shard_bytes // packets, capacity))
            else:
                values = [64, 1, 0, 0, 0, 0, 0]
            names = ["shard_bytes", "packets", "slots", "src_frame", "src_slot", "dst_frame", "dst_slot"]
            named += [(f"k{k}_{n}", v) for n, v in zip(names, values)]
        rt = ttnn.RuntimeArgs()
        for bank in range(fr.BANKS):
            for lane, (x, y) in enumerate(placed[bank]):
                rt[x][y] = [bank, lane, self.lanes, (bank + lane) & 3]
        common = [self.plan.buffer_address()] + src_addr + dst_addr
        plan_ct = list(ttnn.TensorAccessorArgs(self.plan).get_compile_time_args())
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=KERNEL_DIR + source,
                core_ranges=cores,
                compile_time_args=plan_ct,
                named_compile_time_args=named,
                runtime_args=rt,
                common_runtime_args=common,
                config=ttnn.DataMovementConfigDescriptor(processor=processor, noc=noc),
                compiler_include_paths=INCLUDE,
            )
            for source, processor, noc in (
                ("kv_copy_reader.cpp", ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_0),
                ("kv_copy_writer.cpp", ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_1),
            )
        ]
        return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)

    def descriptor(self):
        """One program for the whole mesh when every chip runs the same one, else one per chip."""
        rows, cols = tuple(self.mesh_device.shape)
        mesh_desc = ttnn.MeshProgramDescriptor()
        if self.placement != "own" and len(self.coords) == rows * cols:
            whole = ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(rows - 1, cols - 1))
            mesh_desc[whole] = self.program()
            return mesh_desc
        for coord in self.coords:
            mesh_desc[
                ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(*coord), ttnn.MeshCoordinate(*coord))
            ] = self.program(coord)
        return mesh_desc

    def io(self):
        tensors = [self.plan]
        for src, _, dst, _ in self.kinds:
            tensors += [src, dst]
        return tensors

    def compile(self):
        ttnn.experimental.prepare_generic_op(self.io(), self.descriptor())

    def __call__(self):
        if self._descriptor is None:
            self._descriptor = self.descriptor()
        return ttnn.generic_op(self.io(), self._descriptor)


class KvCompare(KvCopy):
    """Compare frames of two tensors on the device (``kernels/kv_compare.cpp``): what ``KvCopy`` would move for a
    plan is read from both and compared in L1, on every chip, and only counts come back to the host.

        cmp = KvCompare(mesh, [(pool, pool_geometry, slot_tensor, slot_geometry), ...])
        cmp.set_pairs(pairs, positions)     # the frames of ONE conversation in order; positions: a multiple of 32
        cmp();  stats = cmp.read()          # {"rows": compared, "bad": rows that differ, "first": ...}

    Latent rows are compared on their 656 bytes (the page padding is not the tensor's); a 32-row index-key block as
    a whole."""

    def __init__(self, mesh_device, kinds, lanes: int = 4, chip_index=None):
        super().__init__(mesh_device, kinds, lanes=lanes, placement="rows")
        cols = tuple(mesh_device.shape)[1]
        self.chip_index = chip_index or (lambda coord: coord[0] * cols + coord[1])
        self.result = ttnn.from_torch(
            torch.zeros(1, 1, fr.BANKS * lanes, 16, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.positions = 0

    def set_pairs(self, pairs, positions: int) -> None:
        assert positions % 32 == 0 and positions <= len(pairs) * fr.FRAME_TOKENS
        pairs = [(int(s), int(d)) for s, d in pairs]
        assert len(pairs) <= MAX_PAIRS
        host = torch.zeros(PLAN_WORDS, dtype=torch.int32)
        host[0], host[1] = len(pairs), positions
        if pairs:
            host[PLAN_HEADER : PLAN_HEADER + 2 * len(pairs)] = torch.tensor(pairs, dtype=torch.int32).reshape(-1)
        t = ttnn.from_torch(host.reshape(1, 1, 1, PLAN_WORDS), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.copy_host_to_device_tensor(t, self.plan)
        self.pairs, self.positions = pairs, positions

    def program(self, coord=None):
        coord = self.coords[0] if coord is None else coord
        placed = self.cores_of[coord]
        flat = [c for bank in range(fr.BANKS) for c in placed[bank]]
        cores = core_set(flat)
        named = [
            ("kinds", len(self.kinds)),
            ("cb_plan_reader", CB_PLAN_READER),
            ("cb_data0", CB_DATA0),
            ("cb_data1", CB_DATA0 + 1),
        ]
        a_addr, b_addr, page = [0] * MAX_KINDS, [0] * MAX_KINDS, 64
        names = [
            "shard_bytes",
            "packets",
            "slots",
            "src_frame",
            "src_slot",
            "dst_frame",
            "dst_slot",
            "row_stride",
            "row_bytes",
        ]
        for k in range(MAX_KINDS):
            if k < len(self.kinds):
                src, sg, dst, dg = self.kinds[k]
                kind = sg.kind
                packets = self.packets(kind)
                pkt = kind.shard_bytes // packets
                stride = kind.row_stride or pkt
                row_bytes = kind.width if kind.row_stride else pkt
                assert pkt % stride == 0 and row_bytes % 4 == 0
                values = [
                    kind.shard_bytes,
                    packets,
                    kind.layers,
                    sg.frame_base(1),
                    sg.slot_bytes(1),
                    dg.frame_base(1),
                    dg.slot_bytes(1),
                    stride,
                    row_bytes,
                ]
                a_addr[k], b_addr[k] = src.buffer_address(), dst.buffer_address()
                page = max(page, pkt)
            else:
                values = [64, 1, 0, 0, 0, 0, 0, 64, 64]
            named += [(f"k{k}_{n}", v) for n, v in zip(names, values)]

        def cb(index, size):
            return ttnn.CBDescriptor(
                total_size=size,
                core_ranges=cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.uint8, page_size=size)
                ],
            )

        rt = ttnn.RuntimeArgs()
        for bank in range(fr.BANKS):
            for lane, (x, y) in enumerate(placed[bank]):
                rt[x][y] = [bank, lane, self.lanes, self.chip_index(coord)]
        kernel = ttnn.KernelDescriptor(
            kernel_source=KERNEL_DIR + "kv_compare.cpp",
            core_ranges=cores,
            compile_time_args=list(ttnn.TensorAccessorArgs(self.plan).get_compile_time_args())
            + list(ttnn.TensorAccessorArgs(self.result).get_compile_time_args()),
            named_compile_time_args=named,
            runtime_args=rt,
            common_runtime_args=[self.plan.buffer_address()] + a_addr + b_addr + [self.result.buffer_address()],
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0),
            compiler_include_paths=INCLUDE,
        )
        return ttnn.ProgramDescriptor(
            kernels=[kernel],
            semaphores=[],
            cbs=[cb(CB_PLAN_READER, PLAN_WORDS * 4), cb(CB_DATA0, page), cb(CB_DATA0 + 1, page)],
        )

    def descriptor(self):
        mesh_desc = ttnn.MeshProgramDescriptor()
        for coord in self.coords:  # the chip index is a per-chip argument: one program per chip
            mesh_desc[
                ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(*coord), ttnn.MeshCoordinate(*coord))
            ] = self.program(coord)
        return mesh_desc

    def io(self):
        return super().io() + [self.result]

    def read(self) -> dict:
        """Counts over all chips and cores. ``first``: (device, bank, lane, pair, kind, source offset) of a row
        that differs, or None."""
        devices = self.mesh_device.get_num_devices()
        t = (
            ttnn.to_torch(self.result, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)).to(torch.int64)
            & 0xFFFFFFFF
        )
        t = t.reshape(devices, fr.BANKS * self.lanes, 16)
        assert bool((t[:, :, 0] == 0x4B564350).all()), "a compare core did not report"
        bad = t[:, :, 3]
        first = None
        if int(bad.sum()):
            dev, core = (bad > 0).nonzero()[0].tolist()
            first = (
                dev,
                core // self.lanes,
                core % self.lanes,
                int(t[dev, core, 4]),
                int(t[dev, core, 5]),
                int(t[dev, core, 6]),
            )
        return {
            "packets": int(t[:, :, 1].sum()),
            "rows": int(t[:, :, 2].sum()),
            "bad": int(bad.sum()),
            "first": first,
            "visible": t[:, 0, 7].tolist(),
        }
