# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The two cross-chip steps every fused layer program has (fused/README.md section 2), as parts of a per-chip program:

    broadcast   the layer input x [8, hidden] BF16 from the root chip into the hub core's x buffer of every chip
                (X1 of the attention program, X1' of the MLP program)
    reduce      the sum of the chips' partial outputs [8, hidden] BF16 to the root chip (X5, X6), in two stages:
                the chips of a mesh row to the row's chip in the root's column (the row's "head"), then the heads to
                the root. Each head sums on `parts` reduce cores, core k owning part k of the payload.

Roles on a chip (logical cores; the same on every chip unless said):
    hub          one core, the caller's kernel. It waits for word W_X, runs or releases the chip's compute, and hands
                 the chip's partial output on with hy4l::feed_reduce (kernels/layer_common.hpp): to the sender on a
                 chip that is not a head, into its own reduce cores' stage-1 landing on a head.
    sender       one core, the collective's sender kernel with an explicit op table (sender.py):
                 the root: one op, x from its staging into every chip's hub x buffer, counting W_X there (this chip
                 by a NoC copy); a chip that is not a head: `parts` ops into its head's stage-1 landing, counting
                 W_S1; a head that is not the root: `parts` ops into the root's stage-2 landing, counting W_S2.
    reduce       `parts` cores on the heads: kernels/reduce_dm.cpp and reduce_sum.cpp. The root's write the sum, the
                 layer output F, to the cores the caller names (the iHC site cores).

Measured on the board (layer_mlp/tests/test_transport.py, 98,304 B, 4 reduce cores): exact on 1, 2, 4, 8 and 32
chips; on 32 chips 62.1 us per call traced for both steps with nothing between them: the broadcast's send side
11.9 us, stage 1 landed on the root at 20.6 us, stage 2 at 43.5 us, F written at 57.0 us (the stage-2 sum of 8 slots
x 12 tiles a core is 13.5 us of it).

Persistent state (allocated once; alive across programs and trace replays): the words (16 uint32 on every core),
the sender's staging (one payload), the two landings (cols and rows slots of one part per reduce core).

Why one landing per stage is enough (the collective alternates two): a chip sends its partial of step n + 1 only
after x of step n + 1 has reached it, the root sends that only when its program of step n has ended, and that ends
only after the root has summed stage 2 of step n, which needs every head's stage-1 sum of step n. So nothing of step
n + 1 can land while a landing of step n is still being read. The same argument covers the hub's x buffer: its
readers (the chip's compute) have finished before the chip's partial exists.

Waits and who ends them (the wait graph of the two steps; a caller adds its compute between W_X and feed_reduce):
    root x source  -> nothing cross-chip (its payload is local)
    sender         -> its own routers (fabric open / close); the ready bits of its ops, set by this chip's x
                      source (root), hub (not a head) or reduce cores (a head that is not the root)
    hub            -> W_X: the root's sender (which waits only for the root's x source)
    reduce stage 1 -> W_S1: this chip's hub and the senders of the other chips of the row (each waits for its hub)
    reduce stage 2 -> W_S2: the root's own stage 1 and the other heads' senders (each waits for its stage 1)
Order: x source < root sender < every hub < senders of chips that are not heads < stage 1 of every head < senders
of the heads < stage 2. Every wait depends only on earlier entries: no cycle. A fabric packet is a plain L1 write
on arrival, so no chip has to be running its program to receive, and a landing never pushes back.
Every wait on a word is bounded (HY4L_WAIT_LIMIT polls, the sender's HY4_COLLECTIVE_WAIT_LIMIT): a wait that runs
out leaves a code in the core's status word (Transport.status) and the kernel carries on, so the program ends. The
fabric's own open / close handshakes are not bounded. After any status: Transport.reset() with the mesh idle.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

import ttnn
from models.experimental.hy4_preview.fused.collective import collective as cl
from models.experimental.hy4_preview.fused.layer_common.sender import Sender, mesh_branches, unicast

KERNEL_DIR = Path(__file__).parent / "kernels"
TILE_BYTES = 2048  # a 32 x 32 BF16 tile: the unit the reduce cores sum
W_X, W_S1, W_S2, W_GATHER, W_READY, W_STATUS, W_RUNS, W_FED, W_STAMP = 0, 1, 2, 3, 4, 5, 6, 7, 8
W_SENDER_STATUS = 12  # .. 14: an op of counter W_X / W_S1 / W_S2 that was never handed over (see layer_common.hpp)
WORDS = 16
CB_S1, CB_SUM, CB_S2, CB_F = 24, 25, 26, 27  # on the reduce cores


def is_mock() -> bool:
    return bool(os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH"))


def core_set(cores):
    out = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*cores[0]), ttnn.CoreCoord(*cores[0]))])
    for c in cores[1:]:
        out = out.merge(ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*c), ttnn.CoreCoord(*c))]))
    return out


def grid_set(mesh_device):
    grid = mesh_device.compute_with_storage_grid_size()
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])


def core_buffer(mesh_device, cores, nbytes, dtype=ttnn.bfloat16, width=256):
    """A zeroed L1 buffer of ``nbytes`` on each of ``cores`` (one shard per core; a CoreRangeSet is taken as it
    is), on every chip: a row-major [nbytes / item / width, width] shard. Its address is the same on every core of
    every chip, whichever cores it is sharded on."""
    item = {ttnn.bfloat16: 2, ttnn.uint16: 2, ttnn.uint32: 4, ttnn.float32: 4}[dtype]
    assert nbytes % (width * item) == 0, (nbytes, width, item)
    shape = [nbytes // (width * item), width]
    crs = cores if isinstance(cores, ttnn.CoreRangeSet) else core_set(sorted(set(cores)))
    count = crs.num_cores()
    shard = ttnn.ShardSpec(crs, shape, ttnn.ShardOrientation.ROW_MAJOR)
    floating = dtype in (ttnn.bfloat16, ttnn.float32)
    zeros = torch.zeros(count * shape[0], width, dtype=torch.float32 if floating else torch.int32)
    return ttnn.from_torch(
        zeros,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard),
    )


def address(tensor) -> int:
    return ttnn.get_device_tensors(tensor)[0].buffer_address()


class Topology:
    """Which chips take part, the root, and the two reduce stages. ``chips``: (row, col) coordinates (default the
    whole mesh); every mesh row that has a chip must have its chip of the root's column (its head)."""

    def __init__(self, mesh_shape, root=(0, 0), chips=None):
        self.mesh_shape = tuple(mesh_shape)
        rows, cols = self.mesh_shape
        self.root = tuple(root)
        self.chips = (
            sorted(tuple(c) for c in chips) if chips is not None else [(r, c) for r in range(rows) for c in range(cols)]
        )
        assert self.root in self.chips
        self.rows = sorted({r for r, _ in self.chips})
        self.members = {r: [c for c in self.chips if c[0] == r] for r in self.rows}
        for r in self.rows:
            assert (r, self.root[1]) in self.members[r], f"mesh row {r} has no chip in the root's column"
        self.whole_mesh = len(self.chips) == rows * cols

    def chip(self, coord) -> int:
        return coord[0] * self.mesh_shape[1] + coord[1]

    def head(self, coord):
        return (coord[0], self.root[1])

    def is_head(self, coord) -> bool:
        return coord[1] == self.root[1]

    def n1(self, coord) -> int:
        return len(self.members[coord[0]])

    def idx1(self, coord) -> int:
        return self.members[coord[0]].index(tuple(coord))

    @property
    def n2(self) -> int:
        return len(self.rows)

    def idx2(self, coord) -> int:
        return self.rows.index(coord[0])

    def x_routes(self):
        """The broadcast's routes from the root: the four mesh branches when every chip takes part, else one
        unicast per other chip."""
        rows, cols = self.mesh_shape
        if self.whole_mesh and rows > 1 and cols > 1:
            return mesh_branches(self.mesh_shape, self.root)
        others = [c for c in self.chips if c != self.root]
        assert len(others) <= cl.PACKET_HEADERS_PER_RISC, "part of the mesh: at most 12 unicast routes"
        return [unicast(c) for c in others]


class Transport:
    """Persistent state and per-chip program parts of the broadcast and the reduce (module docstring).

    ``hub_core``, ``sender_core``, ``reduce_cores`` (``parts`` of them): logical cores, the same on every chip;
    ``root_sender_core`` / ``root_reduce_cores``: the root's, when its core layout differs. ``x``: the hub's x buffer
    (a tensor of ``payload_bytes`` on the hub core; made here when None). ``payload_bytes``: [8, hidden] BF16."""

    def __init__(
        self,
        mesh_device,
        topology: Topology,
        hub_core,
        sender_core,
        reduce_cores,
        payload_bytes=98304,
        x=None,
        root_sender_core=None,
        root_reduce_cores=None,
        wait_limit=cl.DEFAULT_WAIT_LIMIT,
        timing=False,
        arena=None,
        x_address=None,
        served_word=None,
        parse_page=None,
    ):
        """arena: (tensor, {"staging": offset, "landing1": offset, "landing2": offset}): the staging and the two
        landings are ranges of that L1 tensor (one over the whole worker grid, so that each core uses its copy for
        its own role and the address space taken is the largest role instead of the sum: layer_attn/PLAN.md section
        4) instead of tensors made here. x_address: the hub's x buffer as an L1 address on the hub core (a range of
        the caller's own tensor) instead of a tensor. served_word: the index of a transport word the chip's sender
        reports its served ops in (Sender(served=...); bit k = op k), or None: the sender as it always was."""
        self.served_word = served_word
        # parse_page: an L1 address on the sender core: the senders are built parse-only (Sender(parse_page=...)) and
        # send nothing; parsed[coord] is what each chip's page must hold after a run
        self.parse_page, self.parsed = parse_page, {}
        self.mesh_device, self.topo = mesh_device, topology
        self.hub_core = tuple(hub_core)
        self._sender = {False: tuple(sender_core), True: tuple(root_sender_core or sender_core)}
        self._reduce = {
            False: [tuple(c) for c in reduce_cores],
            True: [tuple(c) for c in (root_reduce_cores or reduce_cores)],
        }
        self.parts = len(reduce_cores)
        assert len(self._reduce[True]) == self.parts
        self.payload_bytes = int(payload_bytes)
        self.part_bytes = self.payload_bytes // self.parts
        assert self.part_bytes * self.parts == self.payload_bytes and self.part_bytes % TILE_BYTES == 0
        self.tiles = self.part_bytes // TILE_BYTES
        self.wait_limit, self.timing = int(wait_limit), timing
        max_payload = int(ttnn.get_tt_fabric_max_payload_size_bytes())
        self.chunk = max_payload - max_payload % 64
        rows, cols = topology.mesh_shape
        grid = mesh_device.compute_with_storage_grid_size()
        self.grid = (grid.x, grid.y)
        self.words = core_buffer(mesh_device, grid_set(mesh_device), 4 * WORDS, ttnn.uint32, width=WORDS)
        self._arena = arena is not None
        self._offset = {"staging": 0, "landing1": 0, "landing2": 0}
        if self._arena:
            tensor, offsets = arena
            self.staging = self.landing1 = self.landing2 = tensor
            self._offset = {k: int(offsets[k]) for k in self._offset}
            assert all(v % 64 == 0 for v in self._offset.values())
        else:
            self.staging = core_buffer(mesh_device, list(self._sender.values()), self.payload_bytes)
            all_reduce = self._reduce[False] + self._reduce[True]
            self.landing1 = core_buffer(mesh_device, all_reduce, cols * self.part_bytes)
            self.landing2 = core_buffer(mesh_device, all_reduce, rows * self.part_bytes)
        if x_address is not None:
            self.x = None
        else:
            self.x = x if x is not None else core_buffer(mesh_device, [self.hub_core], self.payload_bytes)
        self._words_zero = ttnn.from_torch(
            torch.zeros(grid.x * grid.y, WORDS, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        ttnn.synchronize_device(mesh_device)
        self.words_address = address(self.words)
        self.staging_address = address(self.staging) + self._offset["staging"]
        self.landing1_address = address(self.landing1) + self._offset["landing1"]
        self.landing2_address = address(self.landing2) + self._offset["landing2"]
        self.x_address = int(x_address) if x_address is not None else address(self.x)
        self.defines = [
            ("HY4L_WORDS", f"{self.words_address}u"),
            ("HY4L_WAIT_LIMIT", f"{self.wait_limit}u"),
            ("HY4L_TIMING", "1" if timing else "0"),
        ]
        self.include_paths = [str(KERNEL_DIR), str(cl.KERNEL_DIR)]

    # ------------------------------------------------------------------------------------------ cores
    def sender_core(self, coord):
        return self._sender[tuple(coord) == self.topo.root]

    def reduce_cores(self, coord):
        return self._reduce[tuple(coord) == self.topo.root]

    def noc(self, core):
        return self.mesh_device.worker_core_from_logical_core(ttnn.CoreCoord(*core))

    def noc_word(self, core) -> int:
        v = self.noc(core)
        return v.x | (v.y << 16)

    def word_address(self, index) -> int:
        return self.words_address + 4 * index

    def io(self):
        if self._arena:
            return [self.words, self.staging] + ([self.x] if self.x is not None else [])
        return [self.words, self.staging, self.landing1, self.landing2, self.x]

    def l1_bytes(self) -> int:
        """L1 address space this transport takes on every core (the hub's x buffer aside)."""
        rows, cols = self.topo.mesh_shape
        return 4 * WORDS + self.payload_bytes + (rows + cols) * self.part_bytes

    # ------------------------------------------------------------------------------------------ program parts
    X_OP = 0  # the root's broadcast op

    def x_source_args(self):
        """What the root's x source needs to hand the layer input to the sender: (sender NoC x | y << 16, staging
        address, ready word address, the broadcast op's ready bit). The source writes ``payload_bytes`` (in any
        number of pieces, from any cores) to the staging and then adds the bit once."""
        core = self.sender_core(self.topo.root)
        return [self.noc_word(core), self.staging_address, self.word_address(W_READY), Sender.bit(self.X_OP)]

    def feed_args(self, coord):
        """The hub's arguments for hy4l::feed_reduce on the chip at ``coord``."""
        coord = tuple(coord)
        topo = self.topo
        if not topo.is_head(coord):
            bits = sum(Sender.bit(k) for k in range(self.parts))
            return [
                0,
                self.parts,
                self.part_bytes,
                self.noc_word(self.sender_core(coord)),
                self.staging_address,
                self.word_address(W_READY),
                bits,
            ]
        slot = self.landing1_address + topo.idx1(coord) * self.part_bytes
        return [1, self.parts, self.part_bytes, slot, self.word_address(W_S1)] + [
            self.noc_word(c) for c in self.reduce_cores(coord)
        ]

    def add_sender(self, desc, coord):
        """Append the chip's sender kernel with its ops."""
        coord = tuple(coord)
        served = None if self.served_word is None else self.word_address(self.served_word)
        sender = Sender(
            self.mesh_device,
            coord,
            self.sender_core(coord),
            self.word_address(W_READY),
            self.wait_limit,
            served=served,
            parse_page=self.parse_page,
        )
        if coord == self.topo.root:
            self.broadcast_op(sender)
        else:
            self.reduce_ops(sender, coord)
        index = sender.finish(desc)
        self.parsed[coord] = sender.parsed
        return index

    def broadcast_op(self, sender):
        """The root's broadcast as the FIRST op of a caller's Sender (a program whose sender has further ops)."""
        topo = self.topo
        op = sender.op(
            self.staging_address,
            self.payload_bytes,
            min(self.chunk, self.payload_bytes),
            self.hub_core,
            self.x_address,
            self.word_address(W_X),
            topo.x_routes(),
            own=True,
        )
        assert op == self.X_OP
        return op

    def reduce_ops(self, sender, coord):
        """The reduce's ops of a chip that is not the root as the FIRST ``parts`` ops of a caller's Sender: a chip
        that is not a head sends its partial to its head, a head its row sum to the root."""
        coord = tuple(coord)
        topo = self.topo
        assert coord != topo.root
        head = topo.is_head(coord)
        target = topo.root if head else topo.head(coord)
        base = (
            self.landing2_address + topo.idx2(coord) * self.part_bytes
            if head
            else (self.landing1_address + topo.idx1(coord) * self.part_bytes)
        )
        counter = self.word_address(W_S2 if head else W_S1)
        for k, core in enumerate(self.reduce_cores(target)):
            op = sender.op(
                self.staging_address + k * self.part_bytes,
                self.part_bytes,
                min(self.chunk, self.part_bytes),
                core,
                base,
                counter,
                [unicast(target)],
                own=False,
            )
            assert op == k

    def add_reduce(self, desc, coord, f_targets=None, f_address=0):
        """Append the reduce cores' kernels and buffers of a head. The root's write F to ``f_targets``: per reduce
        core the logical cores that take its part in equal pieces, at ``f_address`` on each."""
        coord = tuple(coord)
        topo = self.topo
        assert topo.is_head(coord)
        root = coord == topo.root
        cores = self.reduce_cores(coord)
        crs = core_set(cores)
        chip = topo.chip(coord)
        n1, n2 = topo.n1(coord), topo.n2
        landing1 = ttnn.get_device_tensors(self.landing1)[chip]
        landing2 = ttnn.get_device_tensors(self.landing2)[chip]

        def over(cb, tensor, offset, size):
            d = ttnn.cb_descriptor_from_sharded_tensor(cb, tensor, address_offset=offset, total_size=size)
            d.core_ranges = crs
            d.format_descriptors = [
                ttnn.CBFormatDescriptor(buffer_index=cb, data_format=ttnn.bfloat16, page_size=TILE_BYTES)
            ]
            return d

        def scratch(cb):
            return ttnn.CBDescriptor(
                total_size=self.part_bytes,
                core_ranges=crs,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=cb, data_format=ttnn.bfloat16, page_size=TILE_BYTES)
                ],
            )

        o1, o2 = self._offset["landing1"], self._offset["landing2"]
        cbs = [over(CB_S1, landing1, o1, n1 * self.part_bytes)]
        if root:
            cbs += [
                over(CB_SUM, landing2, o2 + topo.idx2(coord) * self.part_bytes, self.part_bytes),
                over(CB_S2, landing2, o2, n2 * self.part_bytes),
                scratch(CB_F),
            ]
        else:
            cbs.append(scratch(CB_SUM))
        ct = [CB_S1, CB_SUM, CB_S2, CB_F, self.tiles, n1, n2, int(root)]
        rt = ttnn.RuntimeArgs()
        for k, (x, y) in enumerate(cores):
            if root:
                targets = list(f_targets[k]) if f_targets else []
                piece = self.part_bytes // len(targets) if targets else 0
                assert not targets or piece * len(targets) == self.part_bytes
                rt[x][y] = [len(targets), piece, f_address] + [self.noc_word(t) for t in targets]
            else:
                rt[x][y] = [
                    self.noc_word(self.sender_core(coord)),
                    self.staging_address + k * self.part_bytes,
                    self.word_address(W_READY),
                    Sender.bit(k),
                    self.part_bytes,
                ]
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "reduce_dm.cpp"),
                core_ranges=crs,
                compile_time_args=ct,
                runtime_args=rt,
                defines=self.defines,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_1
                ),
                compiler_include_paths=self.include_paths,
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "reduce_sum.cpp"),
                core_ranges=crs,
                compile_time_args=ct,
                defines=self.defines,
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True
                ),
                compiler_include_paths=self.include_paths,
            ),
        ]
        desc.cbs = list(desc.cbs) + cbs
        desc.kernels = list(desc.kernels) + kernels

    # ------------------------------------------------------------------------------------------ the transport alone
    def xfer_program(self, source, partial, f_targets=None, f_address=0):
        """The MeshProgramDescriptor of the two steps with no compute between them (kernels/xfer_hub.cpp):
        ``source`` (on the root's hub core) is broadcast into every chip's x buffer, and every chip's ``partial`` (on
        its hub core, filled by the host) is summed to the root. Returns (io tensors, descriptor)."""
        mesh_desc = ttnn.MeshProgramDescriptor()
        topo = self.topo
        hx, hy = self.hub_core
        for coord in topo.chips:
            desc = ttnn.ProgramDescriptor(kernels=[], semaphores=[], cbs=[])
            rt = ttnn.RuntimeArgs()
            rt[hx][hy] = (
                [int(coord == topo.root), address(source), self.payload_bytes]
                + self.x_source_args()
                + [address(partial)]
                + self.feed_args(coord)
            )
            desc.kernels = [
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "xfer_hub.cpp"),
                    core_ranges=core_set([self.hub_core]),
                    compile_time_args=[],
                    runtime_args=rt,
                    defines=self.defines,
                    config=ttnn.DataMovementConfigDescriptor(
                        processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1
                    ),
                    compiler_include_paths=self.include_paths,
                )
            ]
            if topo.is_head(coord):
                self.add_reduce(desc, coord, f_targets, f_address)
            self.add_sender(desc, coord)
            mesh_desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(*coord), ttnn.MeshCoordinate(*coord))] = desc
        return self.io() + [source, partial], mesh_desc

    # ------------------------------------------------------------------------------------------ host side
    def read_words(self) -> dict:
        """{(row, col): [cores, 16] int64} of the chips that take part; core (x, y) is row y * grid_x + x."""
        ttnn.synchronize_device(self.mesh_device)
        tensors = ttnn.get_device_tensors(self.words)
        return {
            c: ttnn.to_torch(tensors[self.topo.chip(c)]).reshape(-1, WORDS).to(torch.int64) & 0xFFFFFFFF
            for c in self.topo.chips
        }

    def core_words(self, words, core):
        return words[core[1] * self.grid[0] + core[0]]

    def status(self) -> dict:
        """Everything that is not at rest: {(coord, (x, y)): {word index: value}} over the counters, the ready word,
        the status word and the sender's status words of every core. Empty when the last run completed."""
        out = {}
        for coord, words in self.read_words().items():
            for i in range(words.shape[0]):
                row = words[i]
                bad = {
                    w: hex(int(row[w]))
                    for w in (W_X, W_S1, W_S2, W_GATHER, W_READY, W_STATUS, 12, 13, 14)
                    if int(row[w])
                }
                if bad:
                    out[(coord, (i % self.grid[0], i // self.grid[0]))] = bad
        return out

    def stamps(self, core_of) -> dict:
        """{coord: [4 stamps]} of the core ``core_of(coord)`` gives (None: skip the chip)."""
        out = {}
        for coord, words in self.read_words().items():
            core = core_of(coord)
            if core is not None:
                out[coord] = [int(v) for v in self.core_words(words, core)[W_STAMP : W_STAMP + 4]]
        return out

    def reset(self):
        """Zero every core's words on every chip. Call it with the mesh idle, after a run that left a status."""
        ttnn.synchronize_device(self.mesh_device)
        ttnn.copy_host_to_device_tensor(self._words_zero, self.words)
        ttnn.synchronize_device(self.mesh_device)

    def release(self):
        for t in (self.words,) if self._arena else (self.words, self.staging, self.landing1, self.landing2):
            ttnn.deallocate(t)
