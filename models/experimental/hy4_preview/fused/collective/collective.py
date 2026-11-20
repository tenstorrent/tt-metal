# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A small cross-chip collective inside a program: all-gather / all-reduce-sum of a few hundred bytes to about
50 KB over a ring of chips, as a fragment of a larger ``ttnn.generic_op`` program.

What ttnn.all_reduce does with 2 to 4 programs per mesh axis, each opening and closing its fabric connections,
this does with one sender kernel per chip that another piece's program carries along:

    landing core(s)                    sender core (one per chip)               ring members
    payload in L1 --hand_over()-->     staging --fabric, one or more routes-->  landing slot `ring index`
                                               --NoC copy-------------------->  this chip's own slot
    wait_landed() <------------------------------------ arrival counter (persistent) of the landing core

Every member ends with every member's payload in its landing, slot i from ring member i: that is the all-gather.
The all-reduce is the gather plus a sum of the slots in slot order by the consumer's compute kernel
(kernels/collective_compute.hpp sum_slots), so every member computes the same bits.

A payload may be striped over several landing cores (``parts``): core k hands over, receives and sums part k on
its own, so the sum runs on ``parts`` cores at once and the landing takes 1 / parts of the L1 (see below).

Persistent state (class Site), allocated once from Python and alive across programs and trace replays:
  landing  L1 on the landing cores: 2 parities x members slots x the tiles of one part, per core
  ctrl     L1 on the landing cores: arrival counter per parity, the round, a status word (kernels/collective.hpp)
  staging  L1 on the sender core: one payload
The allocator gives an L1 buffer one address range on every core of every chip, which is what lets a chip address
its peers' landing; it also means a buffer's bytes are taken from every core's L1, whichever cores it is sharded
on. A landing sharded over ``parts`` cores is therefore ``parts`` times smaller than the same payload on one core.

The fragment contract (what another piece's program has to do) is in ``Fragment`` below; the kernel side is in
kernels/collective.hpp (data movement) and kernels/collective_compute.hpp (compute).

Waits, and who ends them (every chip runs the same roles; a chip's program starts whenever its queue gets to it):
  sender   fabric open / close handshakes          the chip's own routers (always running)
           ready word, one bit pair per op         a landing core's hand_over() on the same chip
           a free slot in the router's channel     the router draining towards chips that take packets as plain
                                                   L1 writes: a landing never pushes back
  landing  arrival counter >= members              one count from the own sender's NoC copy, one from each other
  core                                             member's sender. A member's send waits for nothing another
                                                   chip does (a later stage's send waits for the stage before's
                                                   arrivals, which wait for no later send): no cycle
  compute  pages in a landing buffer               the landing core's data-movement kernel, after the counter
The counter and ready waits are bounded by HY4_COLLECTIVE_WAIT_LIMIT polls (kernels/collective.hpp); the fabric's
own handshakes and channel waits are the fabric's and are not. What cannot end a wait is a member that never
runs the program, or members that disagree on the round's parity (stale state: Site.check_idle / Site.reset).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
KERNELS = str(KERNEL_DIR)  # absolute: a snapshot of this directory compiles its own kernels
TILE = 32
TILE_BYTES = {ttnn.float32: 4096, ttnn.bfloat16: 2048, ttnn.uint32: 4096}
PACKET_HEADERS_PER_RISC = 12  # MEM_PACKET_HEADER_POOL_SIZE / 2 RISCs / PACKET_HEADER_MAX_SIZE on Blackhole
MAX_OPS = 16  # two bits per op in the sender's ready word
USE_ARGS = 10  # hy4_collective::Use
# ctrl words, as elements of row 0 of a core's ctrl tile (16 bytes apart)
CTRL_COUNTER0, CTRL_COUNTER1, CTRL_ROUND, CTRL_STATUS = 0, 4, 8, 12
# Timing stamps (kernels built with timing=True), in the order of hy4_collective::Stamp: stamp n is at byte
# 64 + 16 n of the ctrl tile, which is row 1 + n // 4, element 4 * (n % 4).
STAMPS = (
    "start",
    "hand_over",
    "handed",
    "landed",
    "finish",
    "sender_start",
    "sender_open",
    "sender_ready",
    "sender_sent",
    "sender_closed",
)
# Polls (10 ns each on the board) before a kernel's wait gives up and writes the status word; 0 waits for ever.
# The default, a third of a second, suits a program that is little more than the collective (chips start it within
# milliseconds of each other); a program in which a member may be busy for long before it sends needs more, or 0.
DEFAULT_WAIT_LIMIT = 30_000_000

# Default cores: the sender next to the first landing core. Landing cores are a row of ``parts`` cores from there.
SENDER_CORE = (4, 0)
LANDING_CORE = (5, 0)


def is_mock() -> bool:
    return bool(os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH"))


def axis_rings(mesh_shape, axis: int, lines=None) -> list:
    """The rings along mesh ``axis``: one per line of the other axis (``lines``: which of them, default all), each a
    list of (row, col) in increasing coordinate along ``axis``."""
    rows, cols = mesh_shape
    if axis == 1:
        return [[(r, c) for c in range(cols)] for r in (range(rows) if lines is None else lines)]
    return [[(r, c) for r in range(rows)] for c in (range(cols) if lines is None else lines)]


def row_cores(core, count: int = 1) -> list:
    """``count`` logical cores in a row, from ``core`` towards increasing x."""
    return [(core[0] + k, core[1]) for k in range(count)]


def core_set(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*c), ttnn.CoreCoord(*c)) for c in cores])


def row_range(core, count: int = 1):
    """The CoreRangeSet of row_cores(core, count): one range, so shard k of a tensor on it is core k."""
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*core), ttnn.CoreCoord(core[0] + count - 1, core[1]))])


def tile_column(mesh_device, core, tiles, dtype, data=None, cores: int = 1):
    """A [cores * tiles * 32, 32] tile tensor in L1 on every chip, ``tiles`` tiles on each of ``cores`` cores in a
    row from ``core`` (core k holds rows k * tiles * 32 onwards; tile t of a core at byte t * tile_bytes): zeros, or
    ``data`` [chips, cores * tiles * 32, 32] with one slice per chip (row-major chip order)."""
    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(row_range(core, cores), (tiles * TILE, TILE), ttnn.ShardOrientation.ROW_MAJOR),
    )
    tensor = ttnn.from_torch(
        torch.zeros(1, 1, cores * tiles * TILE, TILE),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=memory_config,
    )
    if data is not None:
        write_tile_column(mesh_device, tensor, data)
    return tensor


def write_tile_column(mesh_device, tensor, data):
    """Overwrite a tile_column tensor with ``data`` [chips, rows, 32], one slice per chip. A host write: the mesh
    must be idle, and it cannot be part of a trace. (Writing through from_torch(device=...) with a mesh mapper
    takes a second device buffer of the tensor's size while it converts; this does not.)"""
    chips = mesh_device.get_num_devices()
    assert data.shape[0] == chips and data.shape[-1] == TILE, tuple(data.shape)
    host = ttnn.from_torch(
        data.reshape(chips, 1, -1, TILE),
        dtype=tensor.dtype,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    ttnn.synchronize_device(mesh_device)
    ttnn.copy_host_to_device_tensor(host, tensor)


class Site:
    """The persistent state of one collective site (module docstring).

    ``rings``: lists of (row, col) mesh coordinates, all of one length; a chip is in at most one of them. Chips in
    no ring take no part. ``payload_bytes`` is what travels, in ``parts`` equal parts, part k from and to landing
    core k (a row of ``parts`` cores from ``landing_core``); a slot is one part rounded up to whole tiles of
    ``dtype`` (the payload is tile memory: an elementwise sum or a gather does not care how rows are laid into it).
    ``routes``: "unicast" (one packet route per other member, any ring) or "multicast" (two walks, one each way;
    the ring must be a whole mesh row or column in coordinate order).
    """

    def __init__(
        self,
        mesh_device,
        rings,
        payload_bytes: int,
        dtype=ttnn.float32,
        landing_core=LANDING_CORE,
        sender_core=SENDER_CORE,
        routes: str = "unicast",
        parts: int = 1,
        chunk_bytes: int | None = None,
        name: str = "site",
    ):
        assert routes in ("unicast", "multicast"), routes
        self.mesh_device, self.name, self.dtype, self.routes = mesh_device, name, dtype, routes
        self.mesh_shape = tuple(mesh_device.shape)
        self.rings = [[tuple(c) for c in ring] for ring in rings]
        self.members = len(self.rings[0])
        assert self.members >= 2 and all(len(ring) == self.members for ring in self.rings), "rings of one size >= 2"
        self.where = {}  # (row, col) -> (ring, index in it)
        for ring in self.rings:
            for index, coord in enumerate(ring):
                assert coord not in self.where, f"{coord} is in two rings of {name}"
                self.where[coord] = (ring, index)

        self.parts = int(parts)
        self.landing_core, self.sender_core = tuple(landing_core), tuple(sender_core)
        self.landing_cores = row_cores(self.landing_core, self.parts)
        grid = mesh_device.compute_with_storage_grid_size()
        assert (
            1 <= self.parts <= MAX_OPS and self.landing_cores[-1][0] < grid.x
        ), f"{self.parts} landing cores do not fit"
        assert self.sender_core not in self.landing_cores, "the sender core must not be a landing core"

        self.tile_bytes = TILE_BYTES[dtype]
        self.payload_bytes = int(payload_bytes)
        self.part_bytes = self.payload_bytes // self.parts
        assert self.part_bytes * self.parts == self.payload_bytes and self.part_bytes > 0, "equal parts"
        assert self.part_bytes % 16 == 0, "a part is a multiple of 16 bytes"
        self.tiles = -(-self.part_bytes // self.tile_bytes)  # of one part
        self.slot_stride = self.tiles * self.tile_bytes
        self.parity_stride = self.members * self.slot_stride
        max_payload = int(ttnn.get_tt_fabric_max_payload_size_bytes())
        self.chunk_bytes = min(self.part_bytes, chunk_bytes or max_payload)
        assert self.chunk_bytes % 16 == 0 and self.chunk_bytes <= max_payload, (self.chunk_bytes, max_payload)
        if routes == "multicast":
            self._check_whole_lines()

        self.landing = tile_column(
            mesh_device, self.landing_core, 2 * self.members * self.tiles, dtype, cores=self.parts
        )
        self.ctrl = tile_column(mesh_device, self.landing_core, 1, ttnn.uint32, cores=self.parts)
        self.staging = tile_column(mesh_device, self.sender_core, self.parts * self.tiles, dtype)
        replicate = ttnn.ReplicateTensorToMesh(mesh_device)
        self._ctrl_zero = ttnn.from_torch(
            torch.zeros(1, 1, self.parts * TILE, TILE),
            dtype=ttnn.uint32,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=replicate,
        )
        self._landing_zero = ttnn.from_torch(
            torch.zeros(1, 1, self.parts * 2 * self.members * self.tiles * TILE, TILE),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=replicate,
        )
        # The zeros must be on every chip before any chip's program can write a peer's landing or counter.
        ttnn.synchronize_device(mesh_device)
        self.landing_address = ttnn.get_device_tensors(self.landing)[0].buffer_address()
        self.ctrl_address = ttnn.get_device_tensors(self.ctrl)[0].buffer_address()
        self.staging_address = ttnn.get_device_tensors(self.staging)[0].buffer_address()

    # ------------------------------------------------------------------------------------------
    def _check_whole_lines(self):
        rows, cols = self.mesh_shape
        for ring in self.rings:
            along_cols = ring == [(ring[0][0], c) for c in range(cols)]
            along_rows = ring == [(r, ring[0][1]) for r in range(rows)]
            assert along_cols or along_rows, f"{self.name}: multicast needs whole mesh rows or columns, got {ring}"

    def chip(self, coord) -> int:
        return coord[0] * self.mesh_shape[1] + coord[1]

    def l1_bytes(self) -> int:
        """L1 this site takes on every core of every chip: the landing and ctrl of one core, and the staging."""
        return 2 * self.parity_stride + self.tile_bytes + self.parts * self.slot_stride

    def routes_from(self, coord) -> list:
        """The packet routes of the member at ``coord``: (representative destination coord, kind, destination
        coord or None, hops e | w << 8 | n << 16 | s << 24). The representative names the first hop's direction."""
        ring, index = self.where[coord]
        n = self.members
        if self.routes == "unicast":
            return [(dst, 0, dst, 0) for dst in ring if dst != coord]
        # One walk each way: `ahead` chips towards increasing coordinate (east on axis 1, south on axis 0), the
        # rest the other way; both wrap around the torus.
        ahead, behind = n // 2, n - 1 - n // 2
        along_cols = ring[0][0] == ring[-1][0]
        out = []
        for hops, step, shift in ((ahead, 1, 0 if along_cols else 24), (behind, -1, 8 if along_cols else 16)):
            if hops:
                out.append((ring[(index + step) % n], 1, None, hops << shift))
        return out

    # ------------------------------------------------------------------------------------------
    # host side: what the device holds, and the reset
    # ------------------------------------------------------------------------------------------
    def read_ctrl(self) -> dict:
        """{(row, col): [{"counter": [parity 0, parity 1], "round": r, "status": s} per landing core]} for every
        member chip."""
        ttnn.synchronize_device(self.mesh_device)
        tensors = ttnn.get_device_tensors(self.ctrl)
        out = {}
        for coord in self.where:
            words = ttnn.to_torch(tensors[self.chip(coord)]).reshape(self.parts, TILE, TILE).to(torch.int64)
            words = words[:, 0] & 0xFFFFFFFF
            out[coord] = [
                {
                    "counter": [int(row[CTRL_COUNTER0]), int(row[CTRL_COUNTER1])],
                    "round": int(row[CTRL_ROUND]),
                    "status": int(row[CTRL_STATUS]),
                }
                for row in words
            ]
        return out

    def read_stamps(self) -> dict:
        """{(row, col): [{stamp name: wall clock, low 32 bits} per landing core]}: what kernels built with
        timing=True left on their last run. The sender's stamps are on landing core 0 only."""
        ttnn.synchronize_device(self.mesh_device)
        tensors = ttnn.get_device_tensors(self.ctrl)
        out = {}
        for coord in self.where:
            words = ttnn.to_torch(tensors[self.chip(coord)]).reshape(self.parts, TILE, TILE).to(torch.int64)
            words = words & 0xFFFFFFFF
            out[coord] = [{name: int(core[1 + n // 4, 4 * (n % 4)]) for n, name in enumerate(STAMPS)} for core in words]
        return out

    def summary(self) -> str:
        """The control words in one line: the rounds seen, and every core that is not idle."""
        state = self.read_ctrl()
        rounds = sorted({w["round"] for cores in state.values() for w in cores})
        odd = {
            (coord, k): (w["counter"], w["round"], hex(w["status"]))
            for coord, cores in state.items()
            for k, w in enumerate(cores)
            if w["counter"] != [0, 0] or w["status"]
        }
        return f"{self.name}: rounds {rounds}, not idle {odd}"

    def check_idle(self) -> int:
        """Assert that the site is at rest: counters zero, no status, one round on every core of every member.
        Returns that round."""
        state = self.read_ctrl()
        bad = {c: cores for c, cores in state.items() if any(w["counter"] != [0, 0] or w["status"] for w in cores)}
        assert not bad, f"{self.name}: counters / status left behind: {bad}"
        rounds = {w["round"] for cores in state.values() for w in cores}
        assert len(rounds) == 1, f"{self.name}: members disagree on the round: {sorted(rounds)}"
        return rounds.pop()

    def poke_ctrl(self, counter0=0, counter1=0, round_=0, status=0, rounds=None):
        """Write the control words of every core of every chip: a test's way to leave the state a run that stopped
        inside a collective would. The same values everywhere, except ``rounds`` {(row, col): round} for the chips
        it names."""
        chips = self.mesh_device.get_num_devices()
        words = torch.zeros(chips, self.parts, TILE, TILE, dtype=torch.int64)
        for element, value in (
            (CTRL_COUNTER0, counter0),
            (CTRL_COUNTER1, counter1),
            (CTRL_ROUND, round_),
            (CTRL_STATUS, status),
        ):
            words[:, :, 0, element] = value
        for coord, value in (rounds or {}).items():
            words[self.chip(coord), :, 0, CTRL_ROUND] = value
        write_tile_column(self.mesh_device, self.ctrl, words.reshape(chips, self.parts * TILE, TILE))
        ttnn.synchronize_device(self.mesh_device)

    def reset(self, clear_landing: bool = True):
        """Zero the counters, the round and the status on every chip (and the landing): the state a fresh site has.
        Call it with the mesh idle, after a run that stopped inside a collective or reported a status."""
        ttnn.synchronize_device(self.mesh_device)
        ttnn.copy_host_to_device_tensor(self._ctrl_zero, self.ctrl)
        if clear_landing:
            ttnn.copy_host_to_device_tensor(self._landing_zero, self.landing)
        ttnn.synchronize_device(self.mesh_device)

    def release(self):
        for tensor in (self.landing, self.ctrl, self.staging):
            ttnn.deallocate(tensor)


@dataclass
class Use:
    """One use of a site in one chip's program: ``core_args[(x, y)]`` are the ten runtime args
    (hy4_collective::read_use) of the data-movement kernel on that landing core, ``landing_cbs`` the circular
    buffers of the two parities (on every landing core), ``ops`` the sender ops it takes, one per landing core."""

    site: Site
    ops: list
    core_args: dict
    landing_cbs: tuple


class Fragment:
    """The collective's part of ONE chip's ProgramDescriptor. The piece that owns the program does:

        frag = Fragment(mesh_device, coord, desc, sender_core, landing_cores)  # reserves the ready semaphore
        use = frag.add(site, cb_parity0, cb_parity1)                # per collective; appends the landing CBs
        ... its own kernels: a landing core's data-movement kernel gets use.core_args[core] (10 runtime args) ...
        frag.finish()                                               # appends the sender kernel and its connections

    Contract:
      cores       ``sender_core`` runs kernels/collective_sender.cpp on RISCV_0 (BRISC) and nothing else on that
                  RISC; the sites' landing cores (``landing_cores``: every landing core of every site added) are
                  the caller's. All are logical worker cores, the same on every chip. One sender core serves up
                  to 16 ops: a use takes one op per landing core of its site.
      semaphores  one program semaphore (``frag.ready_id``, the lowest id free in ``desc`` when the Fragment is
                  made) on the sender core and the landing cores, plus two per fabric direction on the sender core
                  (at most 8), allocated by ttnn.setup_routing_plane_connection in finish().
      CB ids      none on the sender core. On the landing cores the two ids the caller passes with a site's first
                  use, laid over the site's landing (members * tiles pages each); pass None to add no CBs and read
                  the landing by address.
      kernel      a landing core's data-movement kernel includes collective.hpp and calls hand_over(),
                  wait_landed() and finish() per use (all three on the landing core: they read the site's round);
                  a compute kernel that reads the landing includes collective_compute.hpp (wait_landing, sum_slots).
                  Kernels that include them need ``compiler_include_paths=[str(KERNEL_DIR)]`` and
                  ``defines=frag.defines`` (the wait bound and the timing switch, as the sender has them).
      packet headers  one per distinct route (uses over the same ring share them), at most 12 on the sender's RISC:
                  a 4-ring and an 8-ring by unicast take 3 + 7, by multicast 2 + 2.
      fabric      the sender core holds link ``link`` of every direction its routes leave by for the whole program;
                  no other kernel of the program may connect to those routers.
    """

    def __init__(
        self,
        mesh_device,
        coord,
        desc,
        sender_core=SENDER_CORE,
        landing_cores=(LANDING_CORE,),
        wait_limit=DEFAULT_WAIT_LIMIT,
        link=0,
        timing=False,
    ):
        self.mesh_device, self.coord, self.desc = mesh_device, tuple(coord), desc
        self.sender_core, self.wait_limit, self.link = tuple(sender_core), int(wait_limit), link
        self.defines = kernel_defines(wait_limit, timing)
        self.landing_cores = {tuple(c) for c in landing_cores}
        self.uses = []
        self.num_ops = 0
        self._finished = False
        # The ready word: one program semaphore id, free on the sender core and on every landing core.
        taken = {int(s.id) for s in desc.semaphores}
        self.ready_id = next(i for i in range(16) if i not in taken)
        desc.semaphores = list(desc.semaphores) + [
            ttnn.SemaphoreDescriptor(
                id=self.ready_id, core_ranges=core_set([self.sender_core] + sorted(self.landing_cores)), initial_value=0
            )
        ]

    def add(self, site: Site, cb_parity0=None, cb_parity1=None) -> Use:
        assert not self._finished
        assert site.sender_core == self.sender_core, (site.sender_core, self.sender_core)
        assert self.coord in site.where, f"{self.coord} is in no ring of {site.name}"
        assert (
            set(site.landing_cores) <= self.landing_cores
        ), f"{site.landing_cores} are not all in {self.landing_cores}"
        assert self.num_ops + site.parts <= MAX_OPS, f"more than {MAX_OPS} sender ops"
        sender = self.mesh_device.worker_core_from_logical_core(ttnn.CoreCoord(*self.sender_core))
        ops = list(range(self.num_ops, self.num_ops + site.parts))
        self.num_ops += site.parts
        core_args = {
            core: [
                site.ctrl_address,
                site.landing_address,
                site.parity_stride,
                site.slot_stride,
                site.members,
                site.part_bytes,
                site.staging_address + k * site.slot_stride,
                sender.x | (sender.y << 16),
                self.ready_id,
                ops[k],
            ]
            for k, core in enumerate(site.landing_cores)
        }
        cbs = ()
        if cb_parity0 is not None:
            landing = ttnn.get_device_tensors(site.landing)[site.chip(self.coord)]
            cbs = tuple(
                ttnn.cb_descriptor_from_sharded_tensor(
                    cb, landing, address_offset=parity * site.parity_stride, total_size=site.parity_stride
                )
                for parity, cb in enumerate((cb_parity0, cb_parity1))
            )
            self.desc.cbs = list(self.desc.cbs) + list(cbs)
        use = Use(site, ops, core_args, cbs)
        self.uses.append(use)
        return use

    def finish(self):
        """Append the sender kernel with its op table and fabric connections. Call once, after every add()."""
        assert not self._finished and self.uses
        self._finished = True
        mesh = self.mesh_device
        node = lambda coord: mesh.get_fabric_node_id(ttnn.MeshCoordinate(*coord))
        src = node(self.coord)
        directions, connection_nodes = [], []  # one connection per first-hop direction
        route_keys, route_table, op_table = [], [], []  # routes are shared by the ops that have them in common
        for use in self.uses:
            site = use.site
            _, index = site.where[self.coord]
            route_ids = []
            for towards, kind, dst, hops in site.routes_from(self.coord):
                direction = ttnn.get_eth_forwarding_direction(src, node(towards))
                assert direction is not None, f"no route from {self.coord} to {towards}"
                if direction not in directions:
                    directions.append(direction)
                    connection_nodes.append(node(towards))
                key = (directions.index(direction), kind, node(dst).chip_id if dst is not None else 0, hops)
                if key not in route_keys:
                    route_keys.append(key)
                    route_table += list(key)
                route_ids.append(route_keys.index(key))
            for k, core in enumerate(site.landing_cores):
                landing = mesh.worker_core_from_logical_core(ttnn.CoreCoord(*core))
                op_table += [
                    site.staging_address + k * site.slot_stride,
                    site.part_bytes,
                    site.chunk_bytes,
                    landing.x | (landing.y << 16),
                    site.landing_address + index * site.slot_stride,
                    site.parity_stride,
                    site.ctrl_address,
                    len(route_ids),
                ] + route_ids
        num_routes = len(route_keys)
        assert num_routes <= PACKET_HEADERS_PER_RISC, f"{num_routes} routes, {PACKET_HEADERS_PER_RISC} packet headers"
        head = [self.ready_id, self.num_ops, len(connection_nodes), num_routes]
        kernel = ttnn.KernelDescriptor(
            kernel_source=f"{KERNELS}/collective_sender.cpp",
            core_ranges=core_set([self.sender_core]),
            compile_time_args=[num_routes],
            defines=self.defines,
            # NoC 0, as the tree's other fabric senders: the landing cores' hand-over uses NoC 1.
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0),
            compiler_include_paths=[str(KERNEL_DIR)],
        )
        self.desc.kernels = list(self.desc.kernels) + [kernel]
        kernel_index = len(self.desc.kernels) - 1
        fabric_args = ttnn.setup_routing_plane_connection(
            mesh, src, connection_nodes, [self.link], self.desc, kernel_index, ttnn.CoreCoord(*self.sender_core)
        )
        x, y = self.sender_core
        self.desc.kernels[kernel_index].runtime_args[x][y] = head + route_table + op_table + list(fabric_args)
        return kernel_index


def kernel_defines(wait_limit=DEFAULT_WAIT_LIMIT, timing=False) -> list:
    """The defines of a kernel that includes collective.hpp: the wait bound, and whether it leaves timing stamps.
    A program's sender and landing kernels should agree on them (``Fragment.defines``)."""
    return [("HY4_COLLECTIVE_WAIT_LIMIT", str(int(wait_limit))), ("HY4_COLLECTIVE_TIMING", "1" if timing else "0")]


def append_all_gather(frag: Fragment, site: Site, cb_parity0=None, cb_parity1=None) -> Use:
    """Add an all-gather over ``site`` to the chip program ``frag`` builds: after wait_landed() the landing of the
    round's parity holds every member's payload, slot i from ring member i."""
    return frag.add(site, cb_parity0, cb_parity1)


def append_all_reduce(frag: Fragment, site: Site, cb_parity0=None, cb_parity1=None) -> Use:
    """Add an all-reduce-sum over ``site``: the all-gather, whose landing the caller's compute kernel sums in slot
    order with hy4_collective::sum_slots(wait_landing(cb_parity0, cb_parity1, ...), members, tiles, cb_out). The CB
    ids go with a site's first use in the program; later uses of the same site pass none and read the same two."""
    return frag.add(site, cb_parity0, cb_parity1)


# ---------------------------------------------------------------------------------------------
# The standalone collectives: the fragment in the smallest program that uses it
# ---------------------------------------------------------------------------------------------
CB_IN, CB_OUT, CB_LANDING = 0, 1, 2  # landing CBs: 2 + 2 * (distinct site) + parity
MAX_STAGES = 4
MODES = {"reduce": 0, "gather": 1, "land": 2}


class Collective:
    """all_reduce_sum (one to four stages) or all_gather as a program of its own, on tensors in L1 on the landing
    cores: the input [parts * tiles * 32, 32] per chip (``new_input``), the output the same for a reduce and members
    times that for a gather. ``sites``: one Site, or a list for a staged reduce: each stage sums the stage before's
    sums over its own rings. Two sites, one per mesh axis, is the reduce over all 32 chips; a site may also be
    listed more than once (several uses of one site in one program).

    mode "land" is the gather as a fused program sees it: the transport alone, the result left in the site's
    landing (``read`` takes it from there); "gather" also copies it into an output tensor of its own.

    The sum order of a reduce: within a ring ((slot 0 + slot 1) + slot 2) + ..., slots in ring order; a later
    stage does the same over the earlier stage's sums. The adds are the SFPU's in Float32 (Float32 payloads are
    unpacked to dest, so no mantissa bit is lost); a BFloat16 stage packs its Float32 sum once.
    """

    def __init__(self, mesh_device, sites, mode="reduce", wait_limit=DEFAULT_WAIT_LIMIT, timing=False):
        sites = list(sites) if isinstance(sites, (list, tuple)) else [sites]
        assert mode in MODES and 1 <= len(sites) <= (MAX_STAGES if mode == "reduce" else 1)
        first = sites[0]
        for site in sites:
            assert (site.dtype, site.tiles, site.part_bytes, site.parts) == (
                first.dtype,
                first.tiles,
                first.part_bytes,
                first.parts,
            )
            assert (site.landing_core, site.sender_core) == (first.landing_core, first.sender_core)
            assert set(site.where) == set(first.where), "every stage must cover the same chips"
        self.mesh_device, self.sites, self.mode, self.wait_limit = mesh_device, sites, mode, wait_limit
        self.timing = timing
        self.distinct = list(dict.fromkeys(sites))
        self.dtype, self.tiles, self.parts, self.core = first.dtype, first.tiles, first.parts, first.landing_core
        self.cores = first.landing_cores
        self.out_tiles = self.tiles * (first.members if mode == "gather" else 1)  # per core
        self.output = tile_column(mesh_device, self.core, self.out_tiles, self.dtype, cores=self.parts)
        self._descriptor = None
        self._input = None

    def new_input(self, data=None):
        """An input tensor: [chips, parts * tiles * 32, 32] values, one slice per chip (zeros when None); landing
        core k holds rows k * tiles * 32 onwards."""
        return tile_column(self.mesh_device, self.core, self.tiles, self.dtype, data, cores=self.parts)

    def write_input(self, input_tensor, data):
        """Overwrite an input tensor made by new_input with new per-chip values (the mesh must be idle)."""
        write_tile_column(self.mesh_device, input_tensor, data)

    def descriptor(self, input_tensor):
        """The MeshProgramDescriptor: one program per member chip."""
        mesh, sites, stages = self.mesh_device, self.sites, len(self.sites)
        inputs = ttnn.get_device_tensors(input_tensor)
        outputs = ttnn.get_device_tensors(self.output)
        site_cbs = {site: (CB_LANDING + 2 * i, CB_LANDING + 2 * i + 1) for i, site in enumerate(self.distinct)}
        stage_cbs = [cb for site in sites for cb in site_cbs[site]] + [0] * (2 * (MAX_STAGES - stages))
        cores = row_range(self.core, self.parts)
        mesh_desc = ttnn.MeshProgramDescriptor()
        for coord in sorted(sites[0].where):
            chip = sites[0].chip(coord)
            desc = ttnn.ProgramDescriptor(kernels=[], semaphores=[], cbs=[])
            desc.cbs = [
                ttnn.cb_descriptor_from_sharded_tensor(CB_IN, inputs[chip]),
                ttnn.cb_descriptor_from_sharded_tensor(CB_OUT, outputs[chip]),
            ]
            frag = Fragment(mesh, coord, desc, sites[0].sender_core, self.cores, self.wait_limit, timing=self.timing)
            uses, seen = [], set()
            for site in sites:
                # A site's landing buffers are added with its first use; later uses read the same two.
                cbs = site_cbs[site] if self.mode == "reduce" and site not in seen else (None, None)
                seen.add(site)
                append = append_all_reduce if self.mode == "reduce" else append_all_gather
                uses.append(append(frag, site, *cbs))
            worker_rt = ttnn.RuntimeArgs()
            for x, y in self.cores:
                worker_rt[x][y] = [a for use in uses for a in use.core_args[(x, y)]] + (
                    [outputs[chip].buffer_address()] if self.mode != "reduce" else []
                )
            kernels = [
                ttnn.KernelDescriptor(
                    kernel_source=f"{KERNELS}/collective_worker.cpp",
                    core_ranges=cores,
                    compile_time_args=[MODES[self.mode], stages, self.tiles, CB_IN, CB_OUT] + stage_cbs,
                    defines=frag.defines,
                    runtime_args=worker_rt,
                    config=ttnn.DataMovementConfigDescriptor(
                        processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_1
                    ),
                    compiler_include_paths=[str(KERNEL_DIR)],
                )
            ]
            if self.mode == "reduce":
                compute = ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True
                )
                if self.dtype == ttnn.float32:
                    modes = [ttnn.UnpackToDestMode.Default] * 64
                    for cb in stage_cbs[: 2 * stages]:
                        modes[cb] = ttnn.UnpackToDestMode.UnpackToDestFp32
                    compute.unpack_to_dest_mode = modes
                stage_args = [a for site in sites for a in (site.members, *site_cbs[site])]
                kernels.append(
                    ttnn.KernelDescriptor(
                        kernel_source=f"{KERNELS}/collective_sum.cpp",
                        core_ranges=cores,
                        compile_time_args=[stages, self.tiles, CB_OUT] + stage_args + [0] * (3 * (MAX_STAGES - stages)),
                        config=compute,
                        compiler_include_paths=[str(KERNEL_DIR)],
                    )
                )
            desc.kernels = kernels
            frag.finish()
            mesh_desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(*coord), ttnn.MeshCoordinate(*coord))] = desc
        return mesh_desc

    def _io(self, input_tensor):
        persistent = [t for site in self.distinct for t in (site.landing, site.ctrl, site.staging)]
        return [input_tensor] + persistent + [self.output]

    def compile(self, input_tensor):
        """Compile and size-check the programs without running them (what the mock Galaxy can do)."""
        ttnn.experimental.prepare_generic_op(self._io(input_tensor), self.descriptor(input_tensor))

    def __call__(self, input_tensor):
        """Run on ``input_tensor`` (from new_input); returns the output tensor, which the next call overwrites."""
        if self._input is not input_tensor:
            self._descriptor, self._input = self.descriptor(input_tensor), input_tensor
        return ttnn.generic_op(self._io(input_tensor), self._descriptor)

    # host reference -----------------------------------------------------------------------------
    def reference(self, data: torch.Tensor, rounding: str = "away") -> dict:
        """What every member must hold for per-chip inputs ``data`` [chips, parts * tiles * 32, 32]:
        {(row, col): tensor}. A gather is the members' inputs one after another in ring order. A reduce is summed
        in the device's order with Float32 adds; a BFloat16 stage brings its Float32 sum to BFloat16 once
        (``rounding``, see to_bf16)."""
        out = {}
        value = {coord: data[self.sites[0].chip(coord)].to(torch.float32) for coord in self.sites[0].where}
        if self.mode != "reduce":
            for coord, (ring, _) in self.sites[0].where.items():
                out[coord] = torch.cat([value[m] for m in ring], dim=0)
            return out
        for site in self.sites:
            summed = {}
            for ring in site.rings:
                total = value[ring[0]].clone()
                for member in ring[1:]:
                    total = total + value[member]
                if self.dtype == ttnn.bfloat16:
                    total = to_bf16(total, rounding)  # what a stage leaves in its BF16 output
                for member in ring:
                    summed[member] = total
            value = summed
        return value

    def read(self) -> dict:
        """The result on every member chip, laid out as ``reference`` lays it: {(row, col): [parts * tiles * 32, 32]}
        of a reduce, members times that of a gather (from the output tensor) or a land (from the landing of the
        last round's parity)."""
        site = self.sites[0]
        parts, members, rows = self.parts, site.members, self.tiles * TILE
        if self.mode == "reduce":
            tensors = ttnn.get_device_tensors(self.output)
            return {c: ttnn.to_torch(tensors[site.chip(c)]).reshape(parts * rows, TILE) for c in site.where}
        if self.mode == "gather":
            tensors = ttnn.get_device_tensors(self.output)
            landed = {c: ttnn.to_torch(tensors[site.chip(c)]).reshape(parts, members, rows, TILE) for c in site.where}
        else:
            rounds = site.read_ctrl()
            tensors = ttnn.get_device_tensors(site.landing)
            landed = {}
            for coord in site.where:
                parity = (rounds[coord][0]["round"] - 1) & 1
                landing = ttnn.to_torch(tensors[site.chip(coord)]).reshape(parts, 2, members, rows, TILE)
                landed[coord] = landing[:, parity]
        # device order: landing core, then slot; reference order: slot (member), then landing core
        return {c: t.permute(1, 0, 2, 3).reshape(members * parts * rows, TILE) for c, t in landed.items()}


def to_bf16(x: torch.Tensor, rounding: str = "away") -> torch.Tensor:
    """Float32 values brought to BFloat16 precision (returned as Float32): "away" rounds a tie away from zero,
    which is what the device's packer does with an FP32 accumulator (the BFloat16 sums of the tests match it bit
    for bit); "even" is torch's own conversion."""
    if rounding == "even":
        return x.to(torch.bfloat16).to(torch.float32)
    bits = x.to(torch.float32).contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    bits = (bits + 0x8000) & 0xFFFF0000  # sign and magnitude: adding to the pattern grows the magnitude
    bits = torch.where(bits >= 1 << 31, bits - (1 << 32), bits).to(torch.int32)
    return bits.view(torch.float32)


def payload_mask(part_bytes: int, dtype, tiles: int, parts: int = 1) -> torch.Tensor:
    """[parts * tiles * 32, 32] bool: the elements of a tile column that lie in the first ``part_bytes`` bytes of
    each part of ``tiles`` tiles. A tile is stored as four 16 x 16 faces (rows 0-15 x cols 0-15, rows 0-15 x cols
    16-31, then rows 16-31 likewise)."""
    item = TILE_BYTES[dtype] // (TILE * TILE)
    stored = torch.arange(tiles * TILE * TILE) < part_bytes // item
    faces = stored.reshape(tiles, 2, 2, 16, 16)  # tile, face row, face col, row, col
    return faces.permute(0, 1, 3, 2, 4).reshape(tiles * TILE, TILE).repeat(parts, 1)
